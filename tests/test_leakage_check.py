import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import _paths
import leakage_check as lc

ROOT = _paths.ROOT


def quiet_run(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return lc.run(argv)


def by_check(findings, name):
    return [f for f in findings if f["check"] == name]


def run_frames(train, test, target, extra=()):
    with tempfile.TemporaryDirectory() as d:
        train.to_csv(f"{d}/tr.csv", index=False)
        test.to_csv(f"{d}/te.csv", index=False)
        return quiet_run(["--train", f"{d}/tr.csv", "--test", f"{d}/te.csv", "--target", target, *extra])


class DemoDatasetTests(unittest.TestCase):
    """The shipped demo has four planted leaks; all four must be found."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        script = Path(cls.tmp.name) / "make_leaky_data.py"
        script.write_text((ROOT / "examples" / "make_leaky_data.py").read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.run([sys.executable, str(script)], check=True, capture_output=True)
        cls.train = str(Path(cls.tmp.name) / "train.csv")
        cls.test = str(Path(cls.tmp.name) / "test.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_four_leaks_found(self):
        code, f = quiet_run(["--train", self.train, "--test", self.test, "--target", "churned",
                             "--time", "signup_date", "--group", "customer_id"])
        self.assertEqual(code, 1)
        dup = by_check(f, "duplicates")[0]
        self.assertEqual(dup["level"], "leak")
        self.assertEqual(dup["details"]["rows"], 60)
        sf = by_check(f, "single_feature")[0]
        self.assertIn("cancellation_reason", sf["details"]["flagged"])
        self.assertAlmostEqual(sf["details"]["scores"]["cancellation_reason"], 1.0, places=2)
        self.assertEqual(by_check(f, "time_order")[0]["level"], "leak")
        grp = by_check(f, "group_overlap")[0]
        self.assertEqual(grp["level"], "leak")
        self.assertEqual(grp["details"]["groups"], 583)

    def test_json_output_is_valid(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lc.run(["--train", self.train, "--test", self.test, "--target", "churned", "--json"])
        data = json.loads(buf.getvalue())
        self.assertIn("SUSPECTED LEAKAGE", data["verdict"])

    def test_script_runs_from_command_line(self):
        script = ROOT / "skills" / "leakage-hunter" / "scripts" / "leakage_check.py"
        r = subprocess.run([sys.executable, str(script), "--train", self.train, "--test", self.test,
                            "--target", "churned"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("Verdict", r.stdout)


class CleanDataTests(unittest.TestCase):
    def make(self, n=1500, seed=0):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "user": np.arange(n),
            "when": pd.date_range("2024-01-01", periods=n, freq="h"),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "cat": rng.choice(["x", "y", "z"], n),
        })
        df["y"] = (df["a"] * 0.5 + rng.normal(size=n) > 0).astype(int)
        return df

    def test_clean_time_and_group_split_has_no_leaks(self):
        df = self.make()
        code, f = run_frames(df.iloc[:1200], df.iloc[1200:], "y", ["--time", "when", "--group", "user"])
        self.assertEqual(code, 0, [x for x in f if x["level"] == "leak"])
        self.assertEqual(by_check(f, "time_order")[0]["level"], "ok")

    def test_one_boundary_tie_is_a_warning_not_a_leak(self):
        df = self.make()
        train, test = df.iloc[:1200].copy(), df.iloc[1200:]
        train.loc[train.index[-1], "when"] = test["when"].min() + pd.Timedelta(hours=1)
        code, f = run_frames(train, test, "y", ["--time", "when"])
        self.assertEqual(by_check(f, "time_order")[0]["level"], "warn")
        self.assertEqual(code, 0)

    def test_duplicates_are_normalised(self):
        df = self.make(400)
        train, test = df.iloc[:300].copy(), df.iloc[300:].copy()
        row = train.iloc[0].copy()
        row["cat"] = " " + str(row["cat"]).upper() + " "  # case + whitespace variant
        test = pd.concat([test, pd.DataFrame([row])], ignore_index=True)
        code, f = run_frames(train, test, "y")
        self.assertGreaterEqual(by_check(f, "duplicates")[0]["details"]["rows"], 1)

    def test_identifier_column_is_warned(self):
        df = self.make()
        code, f = run_frames(df.iloc[:1200], df.iloc[1200:], "y")
        self.assertEqual(by_check(f, "identifiers")[0]["level"], "warn")
        self.assertIn("user", by_check(f, "identifiers")[0]["details"]["columns"])


class TargetTypeTests(unittest.TestCase):
    def test_regression_leak_detected(self):
        rng = np.random.default_rng(1)
        n = 1200
        y = rng.normal(size=n) * 10
        df = pd.DataFrame({"noise": rng.normal(size=n), "leak": y + rng.normal(size=n) * 0.1, "y": y})
        code, f = run_frames(df.iloc[:900], df.iloc[900:], "y")
        self.assertEqual(code, 1)
        self.assertIn("leak", by_check(f, "single_feature")[0]["details"]["flagged"])

    def test_string_multiclass_target(self):
        rng = np.random.default_rng(2)
        n = 900
        label = rng.choice(["cat", "dog", "bird"], n)
        df = pd.DataFrame({"x": rng.normal(size=n), "echo": label, "y": label})
        code, f = run_frames(df.iloc[:700], df.iloc[700:], "y")
        self.assertIn("echo", by_check(f, "single_feature")[0]["details"]["flagged"])

    def test_missing_target_rows_are_dropped_not_crashing(self):
        rng = np.random.default_rng(3)
        n = 600
        df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.integers(0, 2, n).astype(float)})
        df.loc[::50, "y"] = np.nan
        code, f = run_frames(df.iloc[:450], df.iloc[450:], "y")
        self.assertIn(code, (0, 1))
        self.assertTrue(by_check(f, "target_missing"))

    def test_single_class_test_is_skipped_not_crashing(self):
        df = pd.DataFrame({"x": range(100), "y": [0, 1] * 30 + [0] * 40})
        code, f = run_frames(df.iloc[:60], df.iloc[60:], "y")
        self.assertEqual(by_check(f, "single_feature")[0]["level"], "skip")

    def test_data_mode_random_split(self):
        rng = np.random.default_rng(4)
        n = 500
        df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.integers(0, 2, n)})
        with tempfile.TemporaryDirectory() as d:
            df.to_csv(f"{d}/all.csv", index=False)
            code, f = quiet_run(["--data", f"{d}/all.csv", "--target", "y"])
        self.assertEqual(code, 0)

    def test_inf_and_nan_features_do_not_crash(self):
        rng = np.random.default_rng(5)
        n = 400
        df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.integers(0, 2, n)})
        df.loc[::7, "x"] = np.inf
        df["z"] = np.where(rng.random(n) < 0.3, np.nan, 1.0)
        code, f = run_frames(df.iloc[:300], df.iloc[300:], "y")
        self.assertIn(code, (0, 1))


class UsageTests(unittest.TestCase):
    def assert_exit_2(self, argv):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            quiet_run(argv)
        self.assertEqual(cm.exception.code, 2)

    def test_missing_file_exits_2(self):
        self.assert_exit_2(["--data", "does_not_exist.csv", "--target", "y"])

    def test_missing_target_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]}).to_csv(f"{d}/x.csv", index=False)
            self.assert_exit_2(["--data", f"{d}/x.csv", "--target", "nope"])

    def test_needs_data_or_train_and_test(self):
        self.assert_exit_2(["--target", "y"])

    def test_data_and_train_are_exclusive(self):
        self.assert_exit_2(["--data", "a.csv", "--train", "b.csv", "--test", "c.csv", "--target", "y"])


if __name__ == "__main__":
    unittest.main()
