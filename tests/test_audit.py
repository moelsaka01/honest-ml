import contextlib
import io
import tempfile
import unittest

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import audit


def issues(df, **kw):
    return audit.audit(df, **kw)["issues"]


def has(found, column, text):
    return any(i["column"] == column and text in i["issue"] for i in found)


class AuditTests(unittest.TestCase):
    def base(self, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"x": rng.normal(size=n), "y": rng.integers(0, 2, n)})

    def test_clean_data_has_no_high_severity(self):
        found = issues(self.base(), target="y")
        self.assertFalse([i for i in found if i["severity"] == "high"], found)

    def test_missing_labels_and_imbalance(self):
        df = self.base()
        df["y"] = (np.random.default_rng(1).random(len(df)) < 0.005).astype(float)
        df.loc[:9, "y"] = np.nan
        found = issues(df, target="y")
        self.assertTrue(has(found, "y", "missing labels"))
        self.assertTrue(has(found, "y", "severe imbalance"))

    def test_missingness_that_predicts_target(self):
        df = self.base()
        df["after"] = np.where(df["y"] == 1, "reason", None)
        found = issues(df, target="y")
        self.assertTrue(has(found, "after", "missingness predicts target"))

    def test_sentinels_negatives_and_age(self):
        df = self.base()
        df["amount"] = np.where(np.arange(len(df)) % 10 == 0, -999, np.abs(df["x"]) * 10)
        df["age"] = 30 + np.abs(df["x"]) * 10
        df.loc[5, "age"] = 250
        found = issues(df, target="y")
        self.assertTrue(has(found, "amount", "sentinel"))
        self.assertTrue(has(found, "age", "above 120"))

    def test_duplicates_ids_and_constant(self):
        df = self.base(200)
        df = pd.concat([df, df.iloc[:20]], ignore_index=True)
        df["id"] = list(range(180)) + list(range(20)) + list(range(20))
        df["const"] = 1
        found = issues(df, target="y", id_col="id")
        self.assertTrue(has(found, "*", "exact duplicate rows"))
        self.assertTrue(has(found, "id", "repeated ids"))
        self.assertTrue(has(found, "const", "constant"))

    def test_mixed_types_and_spelling_variants(self):
        df = self.base(300)
        df["mixed"] = ["12", "7", "n/a"] * 100
        df["city"] = ["Paris", "paris ", "Rome"] * 100
        found = issues(df, target="y")
        self.assertTrue(has(found, "mixed", "mixed types"))
        self.assertTrue(has(found, "city", "differ only by case"))

    def test_future_dates_and_unparseable(self):
        df = self.base(120)
        df["d"] = ["2024-01-05"] * 118 + ["2999-01-01", "not a date"]
        found = issues(df, target="y")
        self.assertTrue(has(found, "d", "future"))
        self.assertTrue(has(found, "d", "unparseable"))

    def test_dates_are_not_called_high_cardinality(self):
        df = self.base(300)
        df["d"] = pd.date_range("2024-01-01", periods=300).strftime("%Y-%m-%d")
        found = issues(df, target="y")
        self.assertFalse(has(found, "d", "cardinality"))
        self.assertFalse(has(found, "d", "one per row"))

    def test_label_conflicts_detected(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3, 3] * 20, "y": [0, 1, 0, 1, 0, 1] * 20})
        found = issues(df, target="y")
        self.assertTrue(has(found, "y", "conflicting labels"))

    def test_target_drift_over_time(self):
        n = 600
        df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=n, freq="D"), "x": range(n)})
        df["y"] = (df.index > n // 2).astype(int)
        found = issues(df, target="y", time="t")
        self.assertTrue(has(found, "y", "varies by"))

    def test_empty_and_all_missing_do_not_crash(self):
        self.assertTrue(has(issues(pd.DataFrame({"a": []})), "*", "zero rows"))
        df = self.base(50)
        df["empty"] = np.nan
        self.assertTrue(has(issues(df, target="y"), "empty", "entirely missing"))

    def test_regression_target_and_bool_columns(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"flag": rng.random(300) < 0.5, "y": rng.normal(size=300) * 100})
        rep = audit.audit(df, target="y")
        self.assertEqual(rep["target"]["kind"], "regression")

    def test_cli_runs_and_validates_columns(self):
        with tempfile.TemporaryDirectory() as d:
            self.base().to_csv(f"{d}/d.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                audit.run(["--data", f"{d}/d.csv", "--target", "y"])
            self.assertIn("Rows: 1000", out.getvalue())
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                audit.run(["--data", f"{d}/d.csv", "--target", "nope"])
            self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            audit.run(["--data", "missing.csv"])
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
