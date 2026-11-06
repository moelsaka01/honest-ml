import contextlib
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
import repro


class SeedTests(unittest.TestCase):
    def test_set_seed_makes_python_and_numpy_repeatable(self):
        repro.set_seed(123)
        a = (random.random(), np.random.rand())
        repro.set_seed(123)
        b = (random.random(), np.random.rand())
        self.assertEqual(a, b)
        self.assertEqual(os.environ["PYTHONHASHSEED"], "123")


class FingerprintTests(unittest.TestCase):
    def test_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            h1 = repro.data_fingerprint(p)
            self.assertEqual(h1["lines"], 3)
            p.write_text("a,b\n1,2\n3,5\n", encoding="utf-8")
            self.assertNotEqual(h1["sha256"], repro.data_fingerprint(p)["sha256"])

    def test_directory_fingerprint_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("1", encoding="utf-8")
            (Path(d) / "b.txt").write_text("2", encoding="utf-8")
            info = repro.data_fingerprint(d)
            self.assertEqual(info["files"], 2)
            with self.assertRaises(FileNotFoundError):
                repro.data_fingerprint(Path(d) / "missing.csv")


class SnapshotTests(unittest.TestCase):
    def test_snapshot_writes_everything_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "d.csv"
            data.write_text("a\n1\n", encoding="utf-8")
            r1 = repro.snapshot_run(Path(d) / "runs", "my run!", seed=7, data=[str(data)], config={"lr": 0.1})
            r2 = repro.snapshot_run(Path(d) / "runs", "my run!", seed=7, data=[str(data)], config={"lr": 0.1})
            self.assertNotEqual(r1, r2)
            for name in ("seed.txt", "env.txt", "git.txt", "data_hash.txt", "config.json"):
                self.assertTrue((r1 / name).exists(), name)
            self.assertEqual((r1 / "seed.txt").read_text(encoding="utf-8").strip(), "7")
            self.assertEqual(json.loads((r1 / "config.json").read_text(encoding="utf-8"))["lr"], 0.1)
            self.assertEqual(len(json.loads((r1 / "data_hash.txt").read_text(encoding="utf-8"))), 1)
            self.assertIn("python:", (r1 / "env.txt").read_text(encoding="utf-8"))
            git = json.loads((r1 / "git.txt").read_text(encoding="utf-8"))
            self.assertIn("sha", git)
            repro.write_metrics(r1, {"auc": 0.7})
            self.assertEqual(json.loads((r1 / "metrics.json").read_text(encoding="utf-8"))["auc"], 0.7)

    def test_config_file_is_copied(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "cfg.yaml"
            cfg.write_text("lr: 0.1\n", encoding="utf-8")
            run = repro.snapshot_run(Path(d) / "runs", "x", seed=1, config=cfg)
            self.assertEqual((run / "config.yaml").read_text(encoding="utf-8"), "lr: 0.1\n")

    def test_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                repro.snapshot_run(Path(d) / "runs", "x", seed=1, config=Path(d) / "nope.yaml")


class SummaryTests(unittest.TestCase):
    def test_summarize(self):
        r = repro.summarize([0.7, 0.72, 0.68])
        self.assertAlmostEqual(r["mean"], 0.7)
        self.assertAlmostEqual(r["std"], 0.02)
        self.assertIsNone(repro.summarize([0.7])["std"])
        with self.assertRaises(ValueError):
            repro.summarize([])

    def test_cli(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(repro.main(["summarize", "0.7", "0.72", "0.68"]), 0)
        self.assertIn("+/-", buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            repro.main(["summarize", "0.7"])
        self.assertIn("no variance estimate", buf.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(repro.main(["snapshot", "--name", "x", "--seed", "1", "--data", "nope.csv",
                                         "--runs-dir", tempfile.mkdtemp()]), 2)


if __name__ == "__main__":
    unittest.main()
