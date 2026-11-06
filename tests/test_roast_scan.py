import contextlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import _paths
import roast_scan as rs


def scan(code: str, suffix=".py"):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"t{suffix}"
        if suffix == ".ipynb":
            p.write_text(json.dumps({"cells": [{"cell_type": "code", "source": code.splitlines(True)}]}),
                         encoding="utf-8")
        else:
            p.write_text(textwrap.dedent(code), encoding="utf-8")
        return rs.scan_lines(rs.load_lines(p))


def rules(findings):
    return {f["rule"] for f in findings}


GOOD = """
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify=y)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
"""


class RuleTests(unittest.TestCase):
    def test_clean_script_has_no_fatal_or_serious(self):
        found = scan(GOOD)
        self.assertFalse([f for f in found if f["level"] in ("FATAL", "SERIOUS")], found)

    def test_scaler_fit_before_split_is_fatal(self):
        found = scan("""
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            Xs = StandardScaler().fit_transform(X)
            X_train, X_test = train_test_split(Xs, random_state=0)
        """)
        self.assertIn("R1", rules(found))

    def test_scaler_fit_after_split_on_train_is_fine(self):
        found = scan("""
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            X_train, X_test = train_test_split(X, random_state=0)
            sc = StandardScaler()
            a = sc.fit_transform(X_train)
            b = sc.transform(X_test)
        """)
        self.assertNotIn("R1", rules(found))
        self.assertNotIn("R2", rules(found))

    def test_fit_on_test_is_fatal_once(self):
        found = scan("""
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            sc.fit(X_test)
        """)
        self.assertEqual([f["rule"] for f in found if f["level"] == "FATAL"], ["R2"])
        found = scan("""
            from sklearn.ensemble import RandomForestClassifier
            RandomForestClassifier(random_state=0).fit(X_test, y_test)
        """)
        self.assertIn("R2", rules(found))

    def test_smote_before_split_is_fatal(self):
        found = scan("""
            from imblearn.over_sampling import SMOTE
            from sklearn.model_selection import train_test_split
            X_res, y_res = SMOTE().fit_resample(X, y)
            X_train, X_test = train_test_split(X_res, random_state=0)
        """)
        self.assertIn("R1", rules(found))

    def test_time_group_seed_rules(self):
        found = scan("""
            import pandas as pd
            from sklearn.model_selection import train_test_split
            df["date"] = pd.to_datetime(df["date"])
            df["customer_id"]
            a, b = train_test_split(df)
        """)
        self.assertTrue({"R3", "R4", "R5"} <= rules(found))

    def test_time_series_split_silences_r3(self):
        found = scan("""
            import pandas as pd
            from sklearn.model_selection import TimeSeriesSplit, train_test_split
            df["date"] = pd.to_datetime(df["date"])
            a, b = train_test_split(df, shuffle=False, random_state=0)
        """)
        self.assertNotIn("R3", rules(found))

    def test_early_stopping_on_test_and_rolling(self):
        found = scan("""
            import xgboost as xgb
            m = xgb.XGBClassifier(random_state=0)
            m.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            df["f"] = df["v"].rolling(7).mean()
        """)
        self.assertTrue({"R6", "R12"} <= rules(found))

    def test_torch_rules(self):
        found = scan("""
            import torch
            import torch.nn as nn
            loss = nn.CrossEntropyLoss()
            probs = torch.softmax(logits, 1)
            torch.manual_seed(0)
        """)
        self.assertIn("R9", rules(found))
        self.assertIn("R10", rules(found))
        ok = scan("""
            import torch
            model.eval()
            with torch.no_grad():
                pass
            torch.manual_seed(0)
        """)
        self.assertNotIn("R9", rules(ok))

    def test_non_ml_file_is_ignored(self):
        self.assertEqual(scan("import json\nprint(json.dumps({}))\nx = df.dropna()"), [])

    def test_comments_do_not_trigger(self):
        found = scan("""
            from sklearn.preprocessing import StandardScaler
            # StandardScaler().fit_transform(X)  <- do not do this
            from sklearn.model_selection import train_test_split
            X_train, X_test = train_test_split(X, random_state=0)
        """)
        self.assertNotIn("R1", rules(found))

    def test_notebook_support_and_locations(self):
        found = scan("""from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X)
""", suffix=".ipynb")
        self.assertIn("R1", rules(found))
        self.assertIn("code cell 1", found[0]["where"])


class CliTests(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code, findings = rs.run(argv)
        return code, findings, buf.getvalue()

    def test_bad_example_is_fatal_and_scored_low(self):
        code, findings, text = self.run_cli([str(_paths.ROOT / "examples" / "bad_example.py")])
        self.assertEqual(code, 1)
        self.assertIn("R1", rules(findings))
        self.assertLess(rs.trust_score(findings), 3)
        self.assertIn("FATAL", text)

    def test_directory_scan_skips_venvs_and_json_mode(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "venv").mkdir()
            (Path(d) / "venv" / "bad.py").write_text("from sklearn.preprocessing import StandardScaler\n"
                                                     "StandardScaler().fit_transform(X)\n", encoding="utf-8")
            (Path(d) / "ok.py").write_text(GOOD, encoding="utf-8")
            code, findings, text = self.run_cli([d, "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(text)["findings"], [f for f in findings])

    def test_usage_errors_exit_2(self):
        for argv in (["does_not_exist.py"],):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                self.run_cli(argv)
            self.assertEqual(cm.exception.code, 2)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                self.run_cli([d])
            self.assertEqual(cm.exception.code, 2)

    def test_trust_score_bounds(self):
        self.assertEqual(rs.trust_score([]), 10.0)
        many = [{"level": "FATAL"}] * 5
        self.assertEqual(rs.trust_score(many), 0.0)


if __name__ == "__main__":
    unittest.main()
