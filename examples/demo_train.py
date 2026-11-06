"""The honest-ml story in 40 lines: a leaky model that reports ~1.00 AUC, then the honest number.

    python examples/make_leaky_data.py
    python examples/demo_train.py

Uses only pandas, numpy and scikit-learn.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
if not (HERE / "train.csv").exists():
    raise SystemExit("Run `python examples/make_leaky_data.py` first.")

df = pd.concat([pd.read_csv(HERE / "train.csv"), pd.read_csv(HERE / "test.csv")], ignore_index=True)
df["cancellation_reason"] = df["cancellation_reason"].fillna("none")
SEED = 0


def model(features):
    cat = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    return make_pipeline(prep, HistGradientBoostingClassifier(random_state=SEED))


def bootstrap_ci(y, p, n=500):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), (n, len(y)))
    vals = [roc_auc_score(y[i], p[i]) for i in idx if len(set(y[i])) == 2]
    return np.quantile(vals, [0.025, 0.975])


# ---- the lie: random split, every column, no questions asked --------------------------
feats = [c for c in df.columns if c not in ("churned", "signup_date")]
tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
m = model(feats).fit(tr[feats], tr["churned"])
auc = roc_auc_score(te["churned"], m.predict_proba(te[feats])[:, 1])
print(f"[naive]  random split, all columns          test AUC = {auc:.3f}   <- 'ship it!'")

# ---- the truth: dedupe, drop leaky/id columns, split by time AND customer --------------
clean = df.drop_duplicates(subset=[c for c in df.columns if c != "churned"]).copy()
clean["signup_date"] = pd.to_datetime(clean["signup_date"])
clean = clean.sort_values("signup_date")
cut = clean["signup_date"].quantile(0.8)
tr, te = clean[clean["signup_date"] <= cut], clean[clean["signup_date"] > cut]
te = te[~te["customer_id"].isin(set(tr["customer_id"]))]          # no customer on both sides
feats = ["tenure_months", "monthly_charges", "support_calls"]       # no id, no cancellation_reason
m = model(feats).fit(tr[feats], tr["churned"])
p = m.predict_proba(te[feats])[:, 1]
y = te["churned"].to_numpy()
lo, hi = bootstrap_ci(y, p)
print(f"[honest] time + customer split, clean cols  test AUC = {roc_auc_score(y, p):.3f}   "
      f"95% CI [{lo:.3f}, {hi:.3f}]   baseline = 0.500   (n_test={len(te)})")
