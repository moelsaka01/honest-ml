"""honest-ml demo: the SAME churn data under four evaluation protocols.

    python examples/make_leaky_data.py
    python examples/demo_train.py

Shows how a "1.000 AUC" model falls to its honest value as each leak is removed, runs the real
leakage-hunter checker on the naive split, and ends with an honest model comparison.
Needs only pandas, numpy and scikit-learn. Seeded (seed 0) and repeatable on one install, about 10 seconds;
the honest AUC moves by roughly 0.02 between scikit-learn versions and platforms.
"""
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")   # keep the recording clean on mismatched library installs
HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "skills" / "leakage-hunter" / "scripts" / "leakage_check.py"
SEED = 0
N_BOOT = 1000
TARGET = "churned"

if not (HERE / "train.csv").exists():
    raise SystemExit("Run `python examples/make_leaky_data.py` first.")

raw = pd.concat([pd.read_csv(HERE / "train.csv"), pd.read_csv(HERE / "test.csv")], ignore_index=True)
raw["cancellation_reason"] = raw["cancellation_reason"].fillna("none")


def estimator(features, df, kind="hgb"):
    cat = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in features if c not in cat]
    if kind == "logreg":
        prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                                  ("num", StandardScaler(), num)])
        return make_pipeline(prep, LogisticRegression(max_iter=1000, random_state=SEED))
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    return make_pipeline(prep, HistGradientBoostingClassifier(random_state=SEED))


def out_of_fold(df, features, splitter, groups=None, kinds=("hgb",)):
    """5-fold out-of-fold probabilities: every row is predicted by a model that never saw it."""
    y = df[TARGET].to_numpy()
    preds = {k: np.zeros(len(df)) for k in kinds}
    for train_idx, test_idx in splitter.split(df, y, groups):
        for k in kinds:
            m = estimator(features, df, k).fit(df.iloc[train_idx][features], y[train_idx])
            preds[k][test_idx] = m.predict_proba(df.iloc[test_idx][features])[:, 1]
    return y, preds


def bootstrap_weights(groups, n_boot=N_BOOT):
    """Resample whole groups (customers) with replacement, expressed as per-row weights (n_boot x n_rows)."""
    rng = np.random.default_rng(SEED)
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = inverse.max() + 1
    counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
    return counts[:, inverse]


def weighted_auc(y, p, weights):
    """Exact AUC (ties count half) for many weight vectors at once: one value per row of `weights`."""
    order = np.argsort(p, kind="stable")
    y_s, p_s, w_s = y[order], p[order], weights[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(p_s)) + 1]                  # blocks of tied scores
    pos = np.add.reduceat(w_s * y_s, starts, axis=1)
    neg = np.add.reduceat(w_s * (1 - y_s), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    denom = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (pos * (below + 0.5 * neg)).sum(axis=1) / denom
    return auc[denom > 0]                                                   # drop degenerate resamples


def auc_ci(y, p, groups=None):
    """95% bootstrap CI for AUC. Rows of one customer are not independent, so resample customers."""
    groups = np.arange(len(y)) if groups is None else groups
    return np.quantile(weighted_auc(y, p, bootstrap_weights(groups)), [0.025, 0.975])


def row(label, y, p, groups=None):
    lo, hi = auc_ci(y, p, groups)
    print(f"  {label:<58} {roc_auc_score(y, p):>6.3f}   [{lo:.3f}, {hi:.3f}]")


kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)

print("honest-ml demo: one churn dataset, four ways to evaluate it (5-fold, out-of-fold AUC)\n")
print(f"  {'protocol':<58} {'AUC':>6}   95% CI")
print("  " + "-" * 82)

# 1 ---- naive: random folds, every column ----------------------------------------------------
all_cols = [c for c in raw.columns if c not in (TARGET, "signup_date")]
y1, p1 = out_of_fold(raw, all_cols, kfold)
row("1. naive: random folds, every column", y1, p1["hgb"])

# 2 ---- drop the column that gives away the label ---------------------------------------------
feats2 = [c for c in all_cols if c != "cancellation_reason"]
y2, p2 = out_of_fold(raw, feats2, kfold)
row("2. drop cancellation_reason (only filled in after churn)", y2, p2["hgb"])

# 3 ---- dedupe rows, drop the id column as a feature -------------------------------------------------
dedup = raw.drop_duplicates(subset=[c for c in raw.columns if c != TARGET]).reset_index(drop=True)
feats3 = [c for c in feats2 if c != "customer_id"]
y3, p3 = out_of_fold(dedup, feats3, kfold)
row("3. + remove duplicate rows, id is not a feature", y3, p3["hgb"])

# 4 ---- honest: every customer lives in exactly one fold ------------------------------------------------
groups = dedup["customer_id"].to_numpy()
y4, p4 = out_of_fold(dedup, feats3, GroupKFold(n_splits=5), groups, kinds=("hgb", "logreg"))
row("4. + split by customer (GroupKFold)  <-- HONEST", y4, p4["hgb"], groups)
print(f"  {'   baseline: constant score':<58} {0.5:>6.3f}")
print("  " + "-" * 82)
honest = roc_auc_score(y4, p4["hgb"])
print(f"  Same data, same model. The 1.000 was never real: the honest AUC is {honest:.2f}.\n")

# What the real checker says about the naive split ---------------------------------------------------
tr, te = train_test_split(raw, test_size=0.2, random_state=SEED)
with tempfile.TemporaryDirectory() as d:
    tr.to_csv(Path(d) / "train.csv", index=False)
    te.to_csv(Path(d) / "test.csv", index=False)
    out = subprocess.run([sys.executable, str(CHECKER), "--train", str(Path(d) / "train.csv"),
                          "--test", str(Path(d) / "test.csv"), "--target", TARGET,
                          "--time", "signup_date", "--group", "customer_id"],
                         capture_output=True, text=True).stdout
print("leakage-hunter on a naive 80/20 split of the same data:")
for line in out.splitlines():
    if line.startswith("[LEAK]"):
        print("  " + textwrap.shorten(line, width=120, placeholder=" ..."))
print()

# Honest model comparison: paired, grouped bootstrap on the same out-of-fold rows ----------------------
W = bootstrap_weights(groups)
both = (W * (y4 == 1)).sum(axis=1) > 0
both &= (W * (y4 == 0)).sum(axis=1) > 0
diffs = weighted_auc(y4, p4["logreg"], W[both]) - weighted_auc(y4, p4["hgb"], W[both])
lo, hi = np.quantile(diffs, [0.025, 0.975])
d = roc_auc_score(y4, p4["logreg"]) - honest
verdict = "no clear difference" if lo <= 0 <= hi else ("logistic regression is better" if d > 0 else
                                                       "gradient boosting is better")
print("Is the fancier model actually better? (paired bootstrap over customers)")
print(f"  logistic regression minus gradient boosting: AUC {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
