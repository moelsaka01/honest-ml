"""Scanner tests: every rule must fire on a flawed snippet AND stay silent on the correct idiom."""
import contextlib
import io
import json
import sysconfig
import tempfile
import textwrap
import unittest
from pathlib import Path

import _paths
import roast_scan as rs

PRELUDE_SK = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
"""


def scan(code: str, prelude=PRELUDE_SK, suffix=".py"):
    body = textwrap.dedent(code)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"t{suffix}"
        if suffix == ".ipynb":
            p.write_text(json.dumps({"cells": [{"cell_type": "code", "source": body.splitlines(True)}]}),
                         encoding="utf-8")
        else:
            p.write_text(textwrap.dedent(prelude) + body, encoding="utf-8")
        return rs.scan_lines(rs.load_lines(p))


def rules(findings, levels=None):
    return {f["rule"] for f in findings if levels is None or f["level"] in levels}


# ----------------------------------------------------------------------------- flawed code
BAD = {
    "R1": {
        "scaler_before_split": """
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(Xs, y, random_state=0)
        """,
        "inline_scaler_before_split": """
            Xs = StandardScaler().fit_transform(X)
            X_train, X_test = train_test_split(Xs, random_state=0)
        """,
        "fit_on_source_after_split": """
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            scaler = StandardScaler().fit(X)
            X_train_s = scaler.transform(X_train)
        """,
        "pca_before_cross_val": """
            Xp = PCA(n_components=5).fit_transform(X)
            print(cross_val_score(RandomForestClassifier(random_state=0), Xp, y, cv=5))
        """,
        "smote_before_split": """
            from imblearn.over_sampling import SMOTE
            X_res, y_res = SMOTE().fit_resample(X, y)
            X_train, X_test = train_test_split(X_res, random_state=0)
        """,
        "concat_train_and_test": """
            X_train, X_test = train_test_split(X, random_state=0)
            scaler = StandardScaler().fit(pd.concat([X_train, X_test]))
        """,
        "select_k_best_before_split": """
            from sklearn.feature_selection import SelectKBest
            Xk = SelectKBest(k=10).fit_transform(X, y)
            X_train, X_test = train_test_split(Xk, random_state=0)
        """,
        "tfidf_before_split": """
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer()
            M = vec.fit_transform(texts)
            X_train, X_test = train_test_split(M, random_state=0)
        """,
    },
    "R2": {
        "scaler_fit_transform_test": """
            X_train, X_test = train_test_split(X, random_state=0)
            scaler = StandardScaler()
            a = scaler.fit_transform(X_train)
            b = scaler.fit_transform(X_test)
        """,
        "model_fit_test": """
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            RandomForestClassifier(random_state=0).fit(X_test, y_test)
        """,
        "grid_search_on_test": """
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            GridSearchCV(RandomForestClassifier(random_state=0), {"max_depth": [3, 5]}).fit(X_test, y_test)
        """,
        "derived_test_variable": """
            X_train, X_test = train_test_split(X, random_state=0)
            scaled = X_test * 2
            StandardScaler().fit(scaled)
        """,
    },
    "R3": {
        "random_split_dates": """
            df = pd.read_csv("d.csv", parse_dates=["order_date"])
            X_train, X_test = train_test_split(df, random_state=0)
        """,
        "shuffled_kfold_dates": """
            df["ts"] = pd.to_datetime(df["ts"])
            for tr, te in KFold(5, shuffle=True, random_state=0).split(df):
                pass
        """,
    },
    "R4": {
        "customer_ids_random_split": """
            df = pd.read_csv("d.csv")
            X = df.drop(columns=["customer_id", "y"])
            X_train, X_test = train_test_split(df, random_state=0)
        """,
        "patient_id_cross_val": """
            groups = df["patient_id"]
            print(cross_val_score(RandomForestClassifier(random_state=0), X, y, cv=5))
        """,
    },
    "R5": {
        "no_seed_anywhere": """
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            m = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
        """,
        "torch_training_no_seed": """
            import torch
            loss = model(x).sum()
            loss.backward()
            model.eval()
            with torch.no_grad():
                pass
        """,
    },
    "R6": {
        "xgboost_eval_set_test": """
            import xgboost as xgb
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            xgb.XGBClassifier().fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)
        """,
        "keras_validation_data_test": """
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
        """,
    },
    "R7": {
        "accuracy_without_baseline": """
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            m = RandomForestClassifier(random_state=0).fit(X_train, y_train)
            print(accuracy_score(y_test, m.predict(X_test)))
        """,
        "model_score_without_baseline": """
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            m = RandomForestClassifier(random_state=0).fit(X_train, y_train)
            print(m.score(X_test, y_test))
        """,
    },
    "R8": {
        "bare_dropna": """
            df = df.dropna()
        """,
    },
    "R9": {
        "torch_without_eval": """
            import torch
            torch.manual_seed(0)
            loss = criterion(model(x), y)
            loss.backward()
        """,
    },
    "R10": {
        "softmax_in_forward_with_ce": """
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            torch.manual_seed(0)
            class Net(nn.Module):
                def forward(self, x):
                    return F.softmax(self.fc(x), dim=1)
            loss_fn = nn.CrossEntropyLoss()
            loss_fn(Net()(x), y).backward()
            Net().eval()
            with torch.no_grad():
                pass
        """,
    },
    "R11": {
        "no_pipeline": """
            X_train, X_test = train_test_split(X, random_state=0)
            sc = StandardScaler()
            a = sc.fit_transform(X_train)
            b = sc.transform(X_test)
        """,
    },
    "R12": {
        "rolling_without_shift": """
            df["f"] = df["sales"].rolling(7).mean()
        """,
        "centered_rolling": """
            df["f"] = df["sales"].shift(1).rolling(7, center=True).mean()
        """,
    },
    "R13": {
        "best_model_by_test_score": """
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            best_score = 0
            for depth in [2, 4, 8]:
                m = RandomForestClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
                s = accuracy_score(y_test, m.predict(X_test))
                if s > best_score:
                    best_score = s
        """,
    },
    "R14": {
        "get_dummies_before_split": """
            X = pd.get_dummies(df)
            X_train, X_test = train_test_split(X, random_state=0)
        """,
    },
    "R15": {
        "fillna_mean_before_split": """
            df = df.fillna(df.mean())
            X_train, X_test = train_test_split(df, random_state=0)
        """,
        "manual_standardise_before_split": """
            df["x"] = (df["x"] - df["x"].mean()) / df["x"].std()
            X_train, X_test = train_test_split(df, random_state=0)
        """,
    },
    "R16": {
        "group_target_mean_transform_before_split": """
            df["user_mean"] = df.groupby("user")["y"].transform("mean")
            X_train, X_test = train_test_split(df, random_state=0)
        """,
        "target_encoding_via_map_before_split": """
            enc = df.groupby("city")["churned"].mean()
            df["city_enc"] = df["city"].map(enc)
            X_train, X_test = train_test_split(df, random_state=0)
        """,
        "agg_dict_with_target_column": """
            stats = df.groupby("store").agg({"target": "mean"})
            df["store_rate"] = df["store"].map(stats["target"])
            X_train, X_test = train_test_split(df, random_state=0)
        """,
        "target_column_named_by_the_label_assignment": """
            y = df["clicks"]
            df["ad_clicks"] = df.groupby("ad")["clicks"].transform("mean")
            X_train, X_test = train_test_split(df, random_state=0)
        """,
    },
    "R17": {
        "drop_duplicates_on_train_only_after_split": """
            X_train, X_test = train_test_split(df, random_state=0)
            X_train = X_train.drop_duplicates()
        """,
    },
    "R18": {
        "print_best_score_as_test": """
            g = GridSearchCV(RandomForestClassifier(random_state=0), {"max_depth": [2, 4]}, cv=5).fit(X, y)
            print("test accuracy", g.best_score_)
        """,
        "best_score_stored_as_test_score": """
            search = GridSearchCV(RandomForestClassifier(random_state=0), {"max_depth": [2, 4]}, cv=5).fit(X, y)
            test_score = search.best_score_
        """,
        "fstring_report": """
            g = GridSearchCV(RandomForestClassifier(random_state=0), {"max_depth": [2, 4]}, cv=5).fit(X, y)
            print(f"Test AUC: {g.best_score_:.3f}")
        """,
    },
}


class FlawedCodeIsCaught(unittest.TestCase):
    def test_every_rule_fires_on_every_bad_snippet(self):
        for rule, cases in BAD.items():
            for name, code in cases.items():
                with self.subTest(rule=rule, case=name):
                    self.assertIn(rule, rules(scan(code)), f"{rule}/{name} was not detected")

    def test_every_rule_in_the_docstring_has_a_bad_snippet(self):
        documented = {line.split()[0] for line in rs.__doc__.splitlines() if line.strip().startswith("R") and
                      line.strip()[1:2].isdigit()}
        self.assertEqual(documented, set(BAD), "a documented rule has no flawed-code test (or vice versa)")
        self.assertEqual(set(rs.FIXES), documented)


# ----------------------------------------------------------------------------- correct code
GOOD = {
    "idiomatic_pipeline_with_baseline": """
        SEED = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify=y)
        base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=SEED)).fit(X_train, y_train)
        print(accuracy_score(y_test, base.predict(X_test)), accuracy_score(y_test, model.predict(X_test)))
    """,
    "scaler_fit_train_transform_test_in_pipeline_file": """
        X_train, X_test = train_test_split(X, random_state=0)
        scaler = StandardScaler().fit(X_train)
        a, b = scaler.transform(X_train), scaler.transform(X_test)
        joblib.dump(scaler, "s.pkl")
    """,
    "fit_transform_train_then_transform_test": """
        X_train, X_test = train_test_split(X, random_state=0)
        scaler = StandardScaler()
        a = scaler.fit_transform(X_train)
        b = scaler.transform(X_test)
        import pickle; pickle.dump(scaler, open("s", "wb"))
    """,
    "pipeline_in_cross_val_on_full_data": """
        pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
        print(cross_val_score(pipe, X, y, cv=5))
        dummy = DummyClassifier().fit(X, y)
    """,
    "group_kfold_with_ids": """
        df = pd.read_csv("d.csv")
        for tr, te in GroupKFold(5).split(df, groups=df["customer_id"]):
            pass
    """,
    "time_series_split_with_dates": """
        df["d"] = pd.to_datetime(df["d"])
        for tr, te in TimeSeriesSplit(5).split(df):
            pass
    """,
    "manual_time_cutoff": """
        df["order_date"] = pd.to_datetime(df["order_date"])
        train = df[df["order_date"] < "2024-06-01"]
        test = df[df["order_date"] >= "2024-06-01"]
        X_train, X_val = train_test_split(train, random_state=0)
    """,
    "no_shuffle_split_on_dates": """
        df["d"] = pd.to_datetime(df["d"])
        a, b = train_test_split(df, shuffle=False)
    """,
    "xgboost_early_stopping_on_validation": """
        import xgboost as xgb
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
        xgb.XGBClassifier().fit(X_train, y_train, eval_set=[(X_val, y_val)])
    """,
    "refit_on_train_plus_validation": """
        X_train, X_val = train_test_split(X, random_state=0)
        final = StandardScaler().fit(np.concatenate([X_train, X_val]))
    """,
    "torch_done_right": """
        import torch
        torch.manual_seed(0)
        loss = criterion(model(x), y)
        loss.backward()
        model.eval()
        with torch.no_grad():
            pass
    """,
    "rolling_with_shift": """
        df["f"] = df["sales"].shift(1).rolling(7).mean()
        df["g"] = df.groupby("store")["sales"].transform(lambda s: s.shift(1).rolling(3).mean())
    """,
    "rolling_then_shifted_later": """
        df["r"] = df["v"].rolling(3).mean()
        df["r"] = df["r"].shift(1)
    """,
    "eda_group_target_mean_is_not_a_feature": """
        print(df.groupby("region")["churned"].mean())
        rate = df.groupby("region")["churned"].mean()
        rate.plot()
        X_train, X_test = train_test_split(df, random_state=0)
    """,
    "target_encoding_from_train_only": """
        X_train, X_test = train_test_split(df, random_state=0)
        enc = X_train.groupby("city")["churned"].mean()
        X_train["city_enc"] = X_train["city"].map(enc)
        X_test["city_enc"] = X_test["city"].map(enc)
    """,
    "group_mean_of_a_feature_not_the_target": """
        df["user_avg_spend"] = df.groupby("user")["spend"].transform("mean")
        X_train, X_test = train_test_split(df, random_state=0)
    """,
    "dedupe_before_the_split": """
        df = df.drop_duplicates()
        X_train, X_test = train_test_split(df, random_state=0)
    """,
    "best_score_reported_as_cv_and_test_scored_properly": """
        base = DummyClassifier().fit(X_train, y_train)
        g = GridSearchCV(RandomForestClassifier(random_state=0), {"max_depth": [2, 4]}, cv=5).fit(X_train, y_train)
        print("cross-validation score", g.best_score_)
        print("test accuracy", g.score(X_test, y_test))
    """,
    "test_word_without_best_score": """
        base = DummyClassifier().fit(X_train, y_train)
        print("test accuracy", accuracy_score(y_test, g.predict(X_test)))
    """,
    "preprocess_function_without_split": """
        def preprocess(frame, scaler):
            return scaler.fit_transform(frame)
        def fit_scaler(X):
            return StandardScaler().fit(X)
    """,
    "kfold_loop_fits_on_train_fold": """
        for train_idx, test_idx in KFold(5, shuffle=True, random_state=0).split(X):
            sc = StandardScaler()
            a = sc.fit_transform(X[train_idx])
            b = sc.transform(X[test_idx])
        joblib.dump(sc, "x")
    """,
    "names_that_merely_contain_test": """
        latest_values = X * 2
        contest_scores = y
        sc = StandardScaler().fit(latest_values)
        m = RandomForestClassifier(random_state=0).fit(latest_values, contest_scores)
        joblib.dump(sc, "x")
    """,
    "global_numpy_seed_covers_sklearn": """
        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        m = RandomForestClassifier().fit(X_train, y_train)
    """,
    "xgboost_lightgbm_default_seed_is_deterministic": """
        import xgboost, lightgbm
        a = xgboost.XGBClassifier().fit(X, y)
        b = lightgbm.LGBMClassifier().fit(X, y)
    """,
    "selection_loop_on_validation": """
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
        best_score = 0
        for depth in [2, 4, 8]:
            m = RandomForestClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
            s = accuracy_score(y_val, m.predict(X_val))
            if s > best_score:
                best_score = s
        base = DummyClassifier().fit(X_train, y_train)
    """,
    "cross_validation_loop_with_best_fold": """
        best_fold = 0
        for train_idx, test_idx in KFold(5, shuffle=True, random_state=0).split(X):
            m = RandomForestClassifier(random_state=0).fit(X[train_idx], y[train_idx])
            s = accuracy_score(y[test_idx], m.predict(X[test_idx]))
            best_fold = max(best_fold, s)
        base = DummyClassifier()
    """,
    "dropna_with_subset": """
        df = df.dropna(subset=["a", "b"])
    """,
    "smote_on_train_only": """
        from imblearn.over_sampling import SMOTE
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        Xr, yr = SMOTE(random_state=0).fit_resample(X_train, y_train)
        joblib.dump(Xr, "x")
    """,
    "function_names_containing_test_are_not_data": """
        xp, dev = _array_api_for_tests("numpy", "cpu")
        dtype_name = _dtype_for_tests(xp, dev)
        X = X.astype(dtype_name)
        est = RandomForestClassifier(random_state=0).fit(X, y)
        dummy = DummyClassifier().fit(X, y)
    """,
    "bad_code_only_in_strings_comments_docstrings": '''
        """Do NOT do this: StandardScaler().fit_transform(X) before train_test_split(X)."""
        # scaler.fit_transform(X_test)
        msg = "model.fit(X_test, y_test)  # RandomForestClassifier()"
    ''',
}


class CorrectCodeIsLeftAlone(unittest.TestCase):
    def test_no_fatal_or_serious_on_any_good_snippet(self):
        for name, code in GOOD.items():
            with self.subTest(case=name):
                prelude = PRELUDE_SK + "import joblib\n"
                found = scan(code, prelude=prelude)
                bad = [f for f in found if f["level"] in ("FATAL", "SERIOUS")]
                self.assertEqual(bad, [], f"false positive on '{name}': {[(f['rule'], f['problem']) for f in bad]}")

    def test_non_ml_files_are_ignored(self):
        self.assertEqual(scan("import json\nx = df.dropna()\ndf['a'].rolling(3).mean()", prelude=""), [])


# ----------------------------------------------------------------------------- notebooks, parsing, CLI
class NotebookAndParsingTests(unittest.TestCase):
    def test_notebook_with_magics_and_locations(self):
        code = ("%matplotlib inline\n!pip install scikit-learn\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.model_selection import train_test_split\n"
                "Xs = StandardScaler().fit_transform(X)\n"
                "X_train, X_test = train_test_split(Xs, random_state=0)\n")
        found = scan(code, prelude="", suffix=".ipynb")
        self.assertIn("R1", rules(found))
        self.assertIn("code cell 1", [f for f in found if f["rule"] == "R1"][0]["where"])
        self.assertIn("line 5", [f for f in found if f["rule"] == "R1"][0]["where"])

    def test_syntax_errors_are_reported_not_swallowed(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "broken.py").write_text("import sklearn\ndef f(:\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code, findings = rs.run([d])
        self.assertIn("NOT SCANNED", buf.getvalue())
        self.assertIn("broken.py:2", buf.getvalue())
        self.assertEqual(code, 0)

    def test_scanner_does_not_crash_on_real_world_code(self):
        """Crash-test on thousands of real files: scikit-learn and pandas sources."""
        lib = Path(sysconfig.get_paths()["purelib"])
        files = [p for pkg in ("sklearn", "pandas") for p in (lib / pkg).rglob("*.py")
                 if "tests" not in p.parts][:1500]
        self.assertGreater(len(files), 200)
        unparsed = []
        for f in files:
            try:
                rs.scan_lines(rs.load_lines(f))
            except rs.ParseError as exc:
                unparsed.append(str(exc))
        self.assertEqual(unparsed, [], "valid Python must always parse (magics are notebook-only)")

    def test_library_internals_produce_few_fatal_findings(self):
        """Library code is not user ML code; FATAL hits there would signal false positives."""
        lib = Path(sysconfig.get_paths()["purelib"]) / "sklearn"
        fatal = []
        for f in [p for p in lib.rglob("*.py") if "tests" not in p.parts][:1200]:
            try:
                fatal += [(f.name, x["where"], x["problem"]) for x in rs.scan_lines(rs.load_lines(f))
                          if x["level"] == "FATAL"]
            except rs.ParseError:
                pass
        self.assertEqual(fatal, [])


class MagicTests(unittest.TestCase):
    def test_percent_continuation_lines_in_py_files_are_not_magics(self):
        code = ("import sklearn\nmsg = ('a %s b %s'\n       % (1, 2))\n"
                "text = \"\"\"\n%not a magic\n!nor this\n\"\"\"\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.py"
            p.write_text(code, encoding="utf-8")
            self.assertEqual(rs.scan_lines(rs.load_lines(p)), [])

    def test_cell_magics_that_are_not_python_are_dropped(self):
        nb = {"cells": [{"cell_type": "code", "source": ["%%bash\n", "pip install foo bar\n", "ls -la\n"]},
                        {"cell_type": "code", "source": ["%%time\n", "x = 1\n"]}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.ipynb"
            p.write_text(json.dumps(nb), encoding="utf-8")
            self.assertEqual(rs.scan_lines(rs.load_lines(p)), [])

    def test_label_encoder_on_target_is_not_flagged(self):
        found = scan("""
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            X_train, X_test = train_test_split(X, random_state=0)
        """, prelude=PRELUDE_SK + "from sklearn.preprocessing import LabelEncoder\n")
        self.assertNotIn("R1", rules(found))


class CliTests(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code, findings = rs.run(argv)
        return code, findings, buf.getvalue()

    def test_bad_example_is_fatal_and_scored_low(self):
        code, findings, text = self.run_cli([str(_paths.ROOT / "examples" / "bad_example.py")])
        self.assertEqual(code, 1)
        self.assertTrue({"R1", "R3", "R5", "R7"} <= rules(findings))
        self.assertLess(rs.trust_score(findings), 3)
        self.assertIn("FATAL", text)

    def test_the_honest_demo_scores_clean(self):
        code, findings, _ = self.run_cli([str(_paths.ROOT / "examples" / "demo_train.py")])
        self.assertEqual(code, 0)
        self.assertEqual(rules(findings, {"FATAL", "SERIOUS"}), set())

    def test_the_skills_own_scripts_are_clean_of_fatal(self):
        code, findings, _ = self.run_cli([str(_paths.ROOT / "skills")])
        self.assertEqual(code, 0, [f for f in findings if f["level"] == "FATAL"])

    def test_directory_scan_skips_venvs_and_json_mode(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "venv").mkdir()
            (Path(d) / "venv" / "bad.py").write_text(
                "from sklearn.preprocessing import StandardScaler\nStandardScaler().fit_transform(X)\n",
                encoding="utf-8")
            (Path(d) / "ok.py").write_text(textwrap.dedent(PRELUDE_SK) + textwrap.dedent(
                GOOD["idiomatic_pipeline_with_baseline"]), encoding="utf-8")
            code, findings, text = self.run_cli([d, "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(text)["findings"], findings)

    def test_usage_errors_exit_2(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            self.run_cli(["does_not_exist.py"])
        self.assertEqual(cm.exception.code, 2)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                self.run_cli([d])
            self.assertEqual(cm.exception.code, 2)

    def test_trust_score_bounds(self):
        self.assertEqual(rs.trust_score([]), 10.0)
        self.assertEqual(rs.trust_score([{"level": "FATAL"}] * 5), 0.0)


if __name__ == "__main__":
    unittest.main()
