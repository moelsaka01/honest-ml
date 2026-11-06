#!/usr/bin/env python3
"""leakage_check.py - fast sanity checks for ML data leakage.

Usage:
  python leakage_check.py --train train.csv --test test.csv --target y
  python leakage_check.py --data all.csv --target y --test-size 0.2
  Optional: --time <col> --group <col> --top 10

Requires: pandas, scikit-learn, numpy
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

AUC_FLAG = 0.90
R2_FLAG = 0.80


def load(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.data:
        df = pd.read_csv(args.data)
        return train_test_split(df, test_size=args.test_size, random_state=0)
    if not (args.train and args.test):
        sys.exit("Provide --data, or both --train and --test.")
    return pd.read_csv(args.train), pd.read_csv(args.test)


def is_classification(y: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20


def check_duplicates(train, test, target):
    cols = [c for c in train.columns if c != target and c in test.columns]
    a = pd.util.hash_pandas_object(train[cols].astype(str), index=False)
    b = pd.util.hash_pandas_object(test[cols].astype(str), index=False)
    overlap = int(b.isin(set(a)).sum())
    pct = 100 * overlap / max(len(test), 1)
    flag = "LEAK" if overlap else "ok"
    print(f"[{flag}] Duplicate feature rows in test also in train: {overlap} ({pct:.1f}%)")
    return overlap > 0


def single_feature_scores(train, test, target, top):
    y_tr, y_te = train[target], test[target]
    clf = is_classification(y_tr)
    if clf:
        classes = y_tr.astype(str).unique()
        if len(classes) != 2:
            print("[info] Multiclass target: single-feature check uses one-vs-rest on the most common class.")
        pos = y_tr.astype(str).value_counts().index[0]
        y_tr = (y_tr.astype(str) == pos).astype(int)
        y_te = (y_te.astype(str) == pos).astype(int)
        if y_te.nunique() < 2:
            print("[skip] Test target has one class; cannot compute AUC.")
            return False
    results = []
    for col in train.columns:
        if col == target or col not in test.columns:
            continue
        x_tr, x_te = train[[col]].copy(), test[[col]].copy()
        if not pd.api.types.is_numeric_dtype(x_tr[col]):
            codes = {v: i for i, v in enumerate(x_tr[col].astype(str).unique())}
            x_tr[col] = x_tr[col].astype(str).map(codes).astype(float)
            x_te[col] = x_te[col].astype(str).map(codes).fillna(-1).astype(float)
        x_tr, x_te = x_tr.fillna(-999), x_te.fillna(-999)
        model = (DecisionTreeClassifier if clf else DecisionTreeRegressor)(max_depth=4, random_state=0)
        try:
            model.fit(x_tr, y_tr)
            if clf:
                score = roc_auc_score(y_te, model.predict_proba(x_te)[:, 1])
                score = max(score, 1 - score)
            else:
                score = r2_score(y_te, model.predict(x_te))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {col}: {exc}")
            continue
        results.append((col, score))
    results.sort(key=lambda t: t[1], reverse=True)
    limit = AUC_FLAG if clf else R2_FLAG
    metric = "AUC" if clf else "R2"
    print(f"\nTop single-feature {metric} (flag >= {limit}):")
    flagged = False
    for col, score in results[:top]:
        mark = "LEAK?" if score >= limit else "     "
        flagged |= score >= limit
        print(f"  {mark} {col:<40} {score:.3f}")
    return flagged


def check_time(train, test, col):
    t_tr = pd.to_datetime(train[col], errors="coerce")
    t_te = pd.to_datetime(test[col], errors="coerce")
    later = int((t_tr > t_te.min()).sum())
    flag = "LEAK" if later else "ok"
    print(f"\n[{flag}] Train rows dated after the earliest test row: {later} "
          f"(train max {t_tr.max()}, test min {t_te.min()})")
    return later > 0


def check_group(train, test, col):
    shared = set(train[col].dropna()) & set(test[col].dropna())
    flag = "LEAK" if shared else "ok"
    print(f"\n[{flag}] Groups in both train and test ('{col}'): {len(shared)}")
    return bool(shared)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train"); p.add_argument("--test"); p.add_argument("--data")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--target", required=True)
    p.add_argument("--time"); p.add_argument("--group")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    train, test = load(args)
    if args.target not in train.columns:
        sys.exit(f"Target '{args.target}' not in columns.")
    print(f"Train {train.shape}, test {test.shape}, target '{args.target}'\n")

    leaks = [check_duplicates(train, test, args.target)]
    leaks.append(single_feature_scores(train, test, args.target, args.top))
    if args.time:
        leaks.append(check_time(train, test, args.time))
    if args.group:
        leaks.append(check_group(train, test, args.group))

    print("\nVerdict:", "SUSPECTED LEAKAGE - investigate flagged items" if any(leaks) else "no leakage detected by these checks")
    sys.exit(1 if any(leaks) else 0)


if __name__ == "__main__":
    main()
