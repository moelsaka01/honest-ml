"""Real-world input problems found by stress testing: non-ASCII names on a legacy console, bad metric inputs,
private paths in run folders, and a dirty git tree."""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import _paths
import bootstrap_ci as bc
import repro

ROOT = _paths.ROOT
SCRIPTS = ROOT / "skills"


def script(skill, name):
    return str(SCRIPTS / skill / "scripts" / name)


def run_script(args, env_extra=None, cwd=None):
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, cwd=cwd)


class LegacyConsoleTests(unittest.TestCase):
    """A Windows cp1252 console cannot print most non-English column names: the scripts must not crash."""

    ASCII = {"PYTHONIOENCODING": "ascii:strict"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        n = 300
        self.df = pd.DataFrame({"größe_€": rng.normal(size=n), "名前": rng.choice(["x", "y"], n),
                                "y": rng.integers(0, 2, n)})
        self.df.to_csv(self.d / "data.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, r):
        self.assertNotIn("Traceback", r.stderr, r.stderr[-600:])
        self.assertIn(r.returncode, (0, 1), r.stderr[-600:])

    def test_leakage_check(self):
        self.check(run_script([script("leakage-hunter", "leakage_check.py"), "--data", str(self.d / "data.csv"),
                               "--target", "y"], self.ASCII))

    def test_audit(self):
        self.check(run_script([script("dataset-audit", "audit.py"), "--data", str(self.d / "data.csv"),
                               "--target", "y"], self.ASCII))

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(1)
        pd.DataFrame({"y": rng.integers(0, 2, 200), "prédiction": rng.random(200)}).to_csv(
            self.d / "p.csv", index=False)
        self.check(run_script([script("honest-eval", "bootstrap_ci.py"), "--data", str(self.d / "p.csv"),
                               "--y-true", "y", "--pred", "prédiction", "--metric", "roc_auc"], self.ASCII))

    def test_run_eval(self):
        (self.d / "c.jsonl").write_text(json.dumps({"id": "café", "input": "q", "expected": "x", "tags": ["日本"]},
                                                   ensure_ascii=False) + "\n", encoding="utf-8")
        (self.d / "o.jsonl").write_text(json.dumps({"id": "café", "output": "no"}) + "\n", encoding="utf-8")
        self.check(run_script([script("llm-eval", "run_eval.py"), "--cases", str(self.d / "c.jsonl"),
                               "--outputs", str(self.d / "o.jsonl")], self.ASCII))

    def test_roast_scan(self):
        f = self.d / "größe_€.py"
        f.write_text("import sklearn\nfrom sklearn.preprocessing import StandardScaler\n"
                     "from sklearn.model_selection import train_test_split\n"
                     "X = StandardScaler().fit_transform(X)\nXa, Xb = train_test_split(X)\n", encoding="utf-8")
        self.check(run_script([script("ml-roast", "roast_scan.py"), str(f)], self.ASCII))

    def test_repro_snapshot_prints_a_non_ascii_path(self):
        r = run_script([script("reproducible-runs", "repro.py"), "snapshot", "--runs-dir", str(self.d / "runs"),
                        "--name", "größe", "--seed", "0"], self.ASCII)
        self.check(r)


class MetricInputTests(unittest.TestCase):
    def frame(self, **cols):
        rng = np.random.default_rng(0)
        n = 300
        base = {"y": rng.integers(0, 2, n)}
        base.update({k: v(n, rng) if callable(v) else v for k, v in cols.items()})
        return pd.DataFrame(base)

    def run_df(self, df, argv):
        with tempfile.TemporaryDirectory() as d:
            df.to_csv(f"{d}/p.csv", index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = bc.run(["--data", f"{d}/p.csv", *argv])
            return res, buf.getvalue()

    def assert_stops(self, df, argv, fragment):
        err = io.StringIO()
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(err):
            self.run_df(df, argv)
        self.assertEqual(cm.exception.code, 2)
        self.assertIn(fragment, err.getvalue())

    def test_probabilities_given_to_accuracy_stop_with_a_clear_message(self):
        df = self.frame(p=lambda n, rng: rng.random(n))
        for metric in ("accuracy", "f1", "precision", "recall"):
            self.assert_stops(df, ["--y-true", "y", "--pred", "p", "--metric", metric], "hard class predictions")

    def test_real_labels_are_still_accepted(self):
        df = self.frame(p=lambda n, rng: rng.integers(0, 2, n))
        res, _ = self.run_df(df, ["--y-true", "y", "--pred", "p", "--metric", "accuracy"])
        self.assertIn("p", res["point"])

    def test_string_labels_are_still_accepted(self):
        rng = np.random.default_rng(2)
        y = rng.choice(["spam", "ham"], 300)
        df = pd.DataFrame({"y": y, "p": np.where(rng.random(300) < 0.8, y, "spam")})
        res, _ = self.run_df(df, ["--y-true", "y", "--pred", "p", "--metric", "f1"])
        self.assertFalse(np.isnan(res["point"]["p"]))

    def test_single_class_roc_auc_stops(self):
        df = self.frame(p=lambda n, rng: rng.random(n)).assign(y=1)
        self.assert_stops(df, ["--y-true", "y", "--pred", "p", "--metric", "roc_auc"], "both classes")

    def test_multiclass_roc_auc_needs_a_pos_label_and_then_works(self):
        df = self.frame(p=lambda n, rng: rng.random(n)).assign(y=lambda d: np.arange(len(d)) % 3)
        self.assert_stops(df, ["--y-true", "y", "--pred", "p", "--metric", "roc_auc"], "--pos-label")
        res, _ = self.run_df(df, ["--y-true", "y", "--pred", "p", "--metric", "roc_auc", "--pos-label", "1"])
        self.assertFalse(np.isnan(res["point"]["p"]))

    def test_regression_metric_on_text_stops(self):
        df = pd.DataFrame({"y": ["a", "b"] * 50, "p": [1.0, 2.0] * 50})
        self.assert_stops(df, ["--y-true", "y", "--pred", "p", "--metric", "rmse"], "needs numeric")

    def test_non_numeric_scores_for_roc_auc_stop(self):
        df = self.frame(p=lambda n, rng: rng.choice(["hi", "lo"], n))
        self.assert_stops(df, ["--y-true", "y", "--pred", "p", "--metric", "roc_auc"], "scores or probabilities")

    def test_infinite_predictions_are_dropped_and_counted_not_propagated(self):
        df = self.frame(p=lambda n, rng: rng.normal(size=n)).assign(y=lambda d: d["y"].astype(float))
        df.loc[::50, "p"] = np.inf
        res, text = self.run_df(df, ["--y-true", "y", "--pred", "p", "--metric", "rmse"])
        self.assertTrue(np.isfinite(res["point"]["p"]))
        self.assertIn("non-finite", text)
        self.assertIn("dropped 6 rows", text)

    def test_single_class_resamples_do_not_spam_warnings(self):
        rng = np.random.default_rng(3)
        y = np.zeros(40, int)
        y[:2] = 1                                           # so rare that many resamples have one class
        df = pd.DataFrame({"y": y, "p": rng.random(40)})
        with tempfile.TemporaryDirectory() as d:
            df.to_csv(f"{d}/p.csv", index=False)
            r = run_script([script("honest-eval", "bootstrap_ci.py"), "--data", f"{d}/p.csv", "--y-true", "y",
                            "--pred", "p", "--metric", "roc_auc"])
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertNotIn("UndefinedMetricWarning", r.stderr)


class PrivacyTests(unittest.TestCase):
    def test_mask_home_hides_the_home_folder_in_every_spelling(self):
        home = str(Path.home())
        for text in (f"{home}/venv/bin/python", home.replace("\\", "/") + "/x", f"{home}\\Scripts\\python.exe"):
            masked = repro.mask_home(text)
            self.assertNotIn(home, masked)
            self.assertTrue(masked.startswith("~"), masked)

    def test_a_snapshot_contains_no_home_folder(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as d:   # data under the home folder, like real projects
            data = Path(d) / "train.csv"
            data.write_text("a,b\n1,2\n", encoding="utf-8")
            run_dir = repro.snapshot_run(Path(d) / "runs", "t", seed=0, data=[str(data)])
            home = str(Path.home())
            for f in run_dir.iterdir():
                text = f.read_text(encoding="utf-8")
                self.assertNotIn(home, text, f.name)
                self.assertNotIn(home.replace("\\", "/"), text, f.name)
            self.assertIn("~", (run_dir / "data_hash.txt").read_text(encoding="utf-8"))

    def test_the_data_hash_is_unchanged_by_masking(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("a\n1\n", encoding="utf-8")
            self.assertEqual(repro.data_fingerprint(p)["sha256"], hashlib.sha256(p.read_bytes()).hexdigest())


@unittest.skipUnless(subprocess.run(["git", "--version"], capture_output=True).returncode == 0, "git not installed")
class DirtyGitTests(unittest.TestCase):
    def git(self, d, *args):
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        subprocess.run(["git", *args], cwd=d, check=True, capture_output=True, env=env)

    def make_repo(self, d):
        self.git(d, "init", "-q")
        (Path(d) / "train.py").write_text("print(1)\n", encoding="utf-8")
        self.git(d, "add", "-A")
        self.git(d, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")

    def test_clean_tree_records_no_diff(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_repo(d)
            info = repro.git_info(d)
            self.assertFalse(info["dirty"])
            self.assertNotIn("diff_sha256", info)

    def test_dirty_tree_lists_files_and_fingerprints_the_diff(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_repo(d)
            (Path(d) / "train.py").write_text("print(2)\n", encoding="utf-8")
            (Path(d) / "new.py").write_text("x = 1\n", encoding="utf-8")        # untracked
            info = repro.git_info(d)
            self.assertTrue(info["dirty"])
            self.assertEqual(sorted(info["changed_files"]), ["new.py", "train.py"])
            self.assertEqual(len(info["diff_sha256"]), 64)
            first = info["diff_sha256"]
            (Path(d) / "train.py").write_text("print(3)\n", encoding="utf-8")
            self.assertNotEqual(repro.git_info(d)["diff_sha256"], first)       # different change, different hash

    def test_snapshot_writes_the_dirty_details(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_repo(d)
            (Path(d) / "train.py").write_text("print(2)\n", encoding="utf-8")
            run_dir = repro.snapshot_run(Path(d) / "runs", "x", seed=0, repo=d)
            git = json.loads((run_dir / "git.txt").read_text(encoding="utf-8"))
            self.assertTrue(git["dirty"])
            self.assertIn("train.py", git["changed_files"])


if __name__ == "__main__":
    unittest.main()
