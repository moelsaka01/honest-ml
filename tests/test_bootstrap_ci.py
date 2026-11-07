import contextlib
import io
import tempfile
import unittest

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import bootstrap_ci as bc


def quiet(argv):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        res = bc.run(argv)
    return res, out.getvalue()


class BootstrapTests(unittest.TestCase):
    def frame(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.2).astype(int)
        good = np.where(rng.random(n) < 0.9, y, 1 - y)       # ~90% correct
        same = np.where(rng.random(n) < 0.9, y, 1 - y)       # equally good, different noise
        score = np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1)
        return pd.DataFrame({"y": y, "good": good, "same": same, "score": score,
                             "user": rng.integers(0, 100, n)})

    def run_df(self, df, argv):
        with tempfile.TemporaryDirectory() as d:
            df.to_csv(f"{d}/p.csv", index=False)
            return quiet(["--data", f"{d}/p.csv", *argv])

    def test_accuracy_ci_brackets_point_and_baseline_row(self):
        res, text = self.run_df(self.frame(), ["--y-true", "y", "--pred", "good", "--metric", "accuracy"])
        lo, hi = res["ci"]["good"]
        self.assertLessEqual(lo, res["point"]["good"])
        self.assertGreaterEqual(hi, res["point"]["good"])
        self.assertIn("baseline", res["point"])
        self.assertAlmostEqual(res["point"]["baseline"], 1 - self.frame()["y"].mean(), places=3)

    def test_accuracy_does_not_beat_baseline_by_much_message(self):
        df = self.frame()
        df["lazy"] = 0
        res, text = self.run_df(df, ["--y-true", "y", "--pred", "lazy", "--metric", "accuracy"])
        self.assertIn("overlaps the baseline", text)

    def test_f1_baseline_is_zero_for_majority(self):
        res, _ = self.run_df(self.frame(), ["--y-true", "y", "--pred", "good", "--metric", "f1"])
        self.assertEqual(res["point"]["baseline"], 0.0)

    def test_paired_comparison_no_clear_difference(self):
        res, text = self.run_df(self.frame(1500), ["--y-true", "y", "--pred", "good", "same",
                                                   "--metric", "accuracy"])
        d, lo, hi = res["diff"]
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)
        self.assertIn("no clear difference", text)

    def test_paired_comparison_clear_difference(self):
        df = self.frame(1500)
        df["bad"] = np.random.default_rng(9).integers(0, 2, len(df))
        res, text = self.run_df(df, ["--y-true", "y", "--pred", "bad", "good", "--metric", "accuracy"])
        self.assertIn("clear difference; good is better", text)

    def test_lower_is_better_direction_for_mae(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=800)
        df = pd.DataFrame({"y": y, "a": y + rng.normal(size=800) * 0.1, "b": y + rng.normal(size=800) * 2})
        res, text = self.run_df(df, ["--y-true", "y", "--pred", "a", "b", "--metric", "mae",
                                     "--baseline-value", "0"])
        self.assertIn("a is better", text)

    def test_auc_and_group_bootstrap(self):
        res, _ = self.run_df(self.frame(), ["--y-true", "y", "--pred", "score", "--metric", "roc_auc",
                                            "--group", "user"])
        self.assertGreater(res["point"]["score"], 0.7)
        self.assertAlmostEqual(res["point"]["baseline"], 0.5)

    def test_group_bootstrap_interval_is_wider_than_row_bootstrap(self):
        rng = np.random.default_rng(5)
        users = np.repeat(np.arange(40), 25)
        user_effect = rng.random(40)[users]
        y = (rng.random(len(users)) < 0.5).astype(int)
        pred = np.where(rng.random(len(users)) < user_effect, y, 1 - y)  # accuracy varies by user
        df = pd.DataFrame({"y": y, "p": pred, "u": users})
        row, _ = self.run_df(df, ["--y-true", "y", "--pred", "p", "--metric", "accuracy", "--no-baseline"])
        grp, _ = self.run_df(df, ["--y-true", "y", "--pred", "p", "--metric", "accuracy", "--no-baseline",
                                  "--group", "u"])
        width = lambda r: r["ci"]["p"][1] - r["ci"]["p"][0]
        self.assertGreater(width(grp), width(row))

    def test_string_labels_and_pos_label(self):
        rng = np.random.default_rng(2)
        y = np.where(rng.random(400) < 0.3, "spam", "ham")
        p = np.where(rng.random(400) < 0.9, y, np.where(y == "spam", "ham", "spam"))
        res, _ = self.run_df(pd.DataFrame({"y": y, "p": p}),
                             ["--y-true", "y", "--pred", "p", "--metric", "f1", "--pos-label", "spam"])
        self.assertGreater(res["point"]["p"], 0.6)

    def test_multiclass_f1_is_macro(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 3, 300)
        res, _ = self.run_df(pd.DataFrame({"y": y, "p": y}), ["--y-true", "y", "--pred", "p", "--metric", "f1"])
        self.assertAlmostEqual(res["point"]["p"], 1.0)

    def test_rows_with_missing_values_are_dropped(self):
        df = self.frame(300)
        df.loc[:9, "good"] = np.nan
        _, text = self.run_df(df, ["--y-true", "y", "--pred", "good", "--metric", "accuracy"])
        self.assertIn("dropped 10 rows", text)

    def test_bad_usage_exits_2(self):
        for argv in (["--data", "nope.csv", "--y-true", "y", "--pred", "p", "--metric", "accuracy"],):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                quiet(argv)
            self.assertEqual(cm.exception.code, 2)
        with tempfile.TemporaryDirectory() as d:
            self.frame().to_csv(f"{d}/p.csv", index=False)
            for argv in (["--y-true", "zzz", "--pred", "good", "--metric", "accuracy"],
                         ["--y-true", "y", "--pred", "a", "b", "c", "--metric", "accuracy"],
                         ["--y-true", "y", "--pred", "good", "--metric", "accuracy", "--n-boot", "5"]):
                with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                    quiet(["--data", f"{d}/p.csv", *argv])
                self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
