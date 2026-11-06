"""Generate a churn dataset with four planted leaks for the leakage-hunter demo.

Leaks:
  1. `cancellation_reason` is only filled for churned customers (target leak).
  2. Some rows are duplicated across train and test.
  3. Train contains rows dated after test starts (shuffled time split).
  4. The same customer_id appears in both train and test (group leak).
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 4000
customers = rng.integers(0, 1500, n)
signup = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D")
tenure = rng.integers(1, 60, n)
monthly = rng.normal(50, 15, n).round(2)
support_calls = rng.poisson(2, n)
logit = -1.5 + 0.35 * support_calls - 0.03 * tenure + 0.01 * (monthly - 50)
churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
reason = np.where(churned == 1, rng.choice(["price", "service", "moved"], n), "")

df = pd.DataFrame({
    "customer_id": customers,
    "signup_date": signup.strftime("%Y-%m-%d"),
    "tenure_months": tenure,
    "monthly_charges": monthly,
    "support_calls": support_calls,
    "cancellation_reason": reason,
    "churned": churned,
})

shuffled = df.sample(frac=1, random_state=1)
train, test = shuffled.iloc[:3200], shuffled.iloc[3200:]
test = pd.concat([test, train.sample(60, random_state=2)])

out = Path(__file__).parent
train.to_csv(out / "train.csv", index=False)
test.to_csv(out / "test.csv", index=False)
print(f"Wrote {out / 'train.csv'} ({len(train)} rows) and {out / 'test.csv'} ({len(test)} rows)")
