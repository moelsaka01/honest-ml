"""The README promises specific behaviour from the demo: keep it honest and reproducible."""
import re
import subprocess
import sys
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

import _paths

ROOT = _paths.ROOT
EXAMPLES = ROOT / "examples"


def run_demo():
    if not (EXAMPLES / "train.csv").exists():
        subprocess.run([sys.executable, str(EXAMPLES / "make_leaky_data.py")], check=True, capture_output=True)
    return subprocess.run([sys.executable, str(EXAMPLES / "demo_train.py")], capture_output=True, text=True,
                          timeout=300)


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = run_demo()
        cls.out = cls.proc.stdout

    def test_runs_cleanly(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-800:])
        self.out.encode("ascii")                           # safe on any Windows console

    def rows(self):
        found = re.findall(r"^\s+(\d)\. .*?\s(\d\.\d{3})\s+\[(\d\.\d{3}), (\d\.\d{3})\]", self.out, re.MULTILINE)
        return {int(n): (float(a), float(lo), float(hi)) for n, a, lo, hi in found}

    def test_naive_is_perfect_and_honest_is_not(self):
        rows = self.rows()
        self.assertEqual(set(rows), {1, 2, 3, 4})
        self.assertGreaterEqual(rows[1][0], 0.99)           # the lie
        for n in (2, 3, 4):
            self.assertLess(rows[n][0], 0.8)                # every fix removes the inflation
            self.assertGreater(rows[n][0], 0.55)            # the model still has real signal
        auc, lo, hi = rows[4]
        self.assertLessEqual(lo, auc)
        self.assertGreaterEqual(hi, auc)
        self.assertLess(hi - lo, 0.08, "the honest interval should be tight enough to be meaningful")
        self.assertGreater(lo, 0.5, "the honest model should beat the 0.5 baseline")

    def test_real_checker_finds_all_four_leaks(self):
        leaks = [l for l in self.out.splitlines() if l.strip().startswith("[LEAK]")]
        self.assertEqual(len(leaks), 4, leaks)
        joined = " ".join(leaks)
        for needle in ("also appear in train", "cancellation_reason", "dated after", "Groups in both"):
            self.assertIn(needle, joined)

    def test_model_comparison_is_reported_with_an_interval(self):
        m = re.search(r"AUC ([+-]\d\.\d{3}), 95% CI \[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\] -> (.+)", self.out)
        self.assertIsNotNone(m, self.out[-400:])
        d, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        self.assertLessEqual(lo, d)
        self.assertGreaterEqual(hi, d)
        excludes_zero = lo > 0 or hi < 0
        self.assertEqual(excludes_zero, "no clear difference" not in m.group(4))

    def test_readme_banner_matches_the_demo(self):
        # Library versions shift the honest numbers slightly (sklearn 1.5 vs 1.9 differ by ~0.01), so the
        # README must agree within a tolerance, not digit for digit.
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"`(\d\.\d{2}) AUC, 95% CI \[(\d\.\d{2}), (\d\.\d{2})\]", readme)
        self.assertIsNotNone(m, "README banner is missing the honest result")
        banner = [float(x) for x in m.groups()]
        auc, lo, hi = self.rows()[4]
        for claimed, actual in zip(banner, (auc, lo, hi)):
            self.assertAlmostEqual(claimed, actual, delta=0.025)
        self.assertIn("1.000 AUC", readme)

    def test_demo_is_deterministic(self):
        again = run_demo().stdout
        self.assertEqual(again, self.out)


class FastWeightedAucTests(unittest.TestCase):
    def test_matches_sklearn_exactly_including_ties(self):
        src = (EXAMPLES / "demo_train.py").read_text(encoding="utf-8")
        ns = {"np": np}
        exec(src[src.index("def weighted_auc"):src.index("def auc_ci")], ns)
        rng = np.random.default_rng(1)
        for trial in range(8):
            n = 250
            y = (rng.random(n) < 0.3).astype(int)
            p = np.round(rng.random(n), 1) if trial % 2 else rng.random(n)      # odd trials: heavy ties
            weights = rng.multinomial(n, np.full(n, 1 / n), size=15)
            fast = ns["weighted_auc"](y, p, weights)
            ref = [roc_auc_score(y, p, sample_weight=w) for w in weights
                   if w[y == 1].sum() > 0 and w[y == 0].sum() > 0]
            self.assertTrue(np.allclose(fast, ref), trial)


if __name__ == "__main__":
    unittest.main()
