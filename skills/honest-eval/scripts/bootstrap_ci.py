#!/usr/bin/env python3
"""bootstrap_ci.py - confidence intervals, baselines and paired comparisons for model predictions.

Input: one CSV with the true label and one or two prediction columns (from the TEST set).

Usage:
  # one model, with a baseline row
  python bootstrap_ci.py --data preds.csv --y-true y --pred model_a --metric f1
  # two models, paired bootstrap on the difference (B - A)
  python bootstrap_ci.py --data preds.csv --y-true y --pred model_a model_b --metric roc_auc
  # repeated users/patients/documents: resample whole groups, not rows
  python bootstrap_ci.py --data preds.csv --y-true y --pred p --metric accuracy --group user_id

Metrics: accuracy f1 precision recall roc_auc pr_auc mae rmse r2
  roc_auc / pr_auc expect scores or probabilities; the others expect hard predictions.
  f1/precision/recall are binary (positive class = --pos-label, default: rarest class) or macro for >2 classes.

Baseline: majority class (classification) or a constant (regression). Pass --baseline-value with the
TRAIN majority class / TRAIN mean; otherwise it is derived from the test labels, which is slightly optimistic.

Exit codes: 0 ok, 2 bad usage. Requires: pandas, numpy, scikit-learn
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)

HIGHER_BETTER = {"accuracy": True, "f1": True, "precision": True, "recall": True, "roc_auc": True,
                 "pr_auc": True, "mae": False, "rmse": False, "r2": True}
NEEDS_SCORE = {"roc_auc", "pr_auc"}
REGRESSION = {"mae", "rmse", "r2"}


def die(message: str):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def make_metric(name: str, y_true: np.ndarray, pos_label=None):
    """Return f(y, p) -> float or nan. pos_label only matters for binary/ranking metrics."""
    classes = np.unique(y_true)
    binary = len(classes) == 2
    if pos_label is None and binary:
        counts = {c: int((y_true == c).sum()) for c in classes}
        pos_label = min(counts, key=counts.get)

    def f(y, p):
        try:
            if name == "accuracy":
                return accuracy_score(y, p)
            if name in ("f1", "precision", "recall"):
                fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[name]
                if binary:
                    return fn(y, p, pos_label=pos_label, zero_division=0)
                return fn(y, p, average="macro", zero_division=0)
            if name == "roc_auc":
                return roc_auc_score((y == pos_label).astype(int), p)
            if name == "pr_auc":
                return average_precision_score((y == pos_label).astype(int), p)
            if name == "mae":
                return mean_absolute_error(y, p)
            if name == "rmse":
                return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))
            if name == "r2":
                return r2_score(y, p)
        except ValueError:  # e.g. a resample with a single class
            return float("nan")
        raise ValueError(name)

    return f


def resample_indices(n: int, rng: np.random.Generator, groups: np.ndarray | None,
                     group_index: dict | None):
    if groups is None:
        return rng.integers(0, n, n)
    keys = list(group_index)
    picked = rng.integers(0, len(keys), len(keys))
    return np.concatenate([group_index[keys[i]] for i in picked])


def bootstrap(y, preds: dict, metric_fn, n_boot=1000, seed=0, groups=None):
    """Return (point estimates, {name: array of resampled metrics}) using SHARED resamples."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    group_index = None
    if groups is not None:
        groups = np.asarray(groups)
        group_index = {g: np.flatnonzero(groups == g) for g in pd.unique(groups)}
    point = {k: metric_fn(y, np.asarray(v)) for k, v in preds.items()}
    draws = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        idx = resample_indices(len(y), rng, groups, group_index)
        for k, v in preds.items():
            draws[k][b] = metric_fn(y[idx], np.asarray(v)[idx])
    return point, draws


def interval(x: np.ndarray, level=0.95):
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def baseline_predictions(name, y, baseline_value=None):
    y = np.asarray(y)
    if name in REGRESSION:
        value = float(baseline_value) if baseline_value is not None else float(np.mean(y.astype(float)))
        return np.full(len(y), value), value
    if name in NEEDS_SCORE:
        return np.full(len(y), 0.5), "constant score"
    if baseline_value is not None:
        value = type(y[0])(baseline_value) if not isinstance(y[0], str) else str(baseline_value)
    else:
        vals, counts = np.unique(y, return_counts=True)
        value = vals[np.argmax(counts)]
    return np.full(len(y), value, dtype=y.dtype), value


def run(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--y-true", required=True)
    p.add_argument("--pred", nargs="+", required=True, help="one or two prediction columns")
    p.add_argument("--metric", required=True, choices=sorted(HIGHER_BETTER))
    p.add_argument("--pos-label")
    p.add_argument("--group")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline-value")
    p.add_argument("--no-baseline", action="store_true")
    args = p.parse_args(argv)

    if len(args.pred) > 2:
        p.error("--pred takes one or two columns.")
    if args.n_boot < 100:
        p.error("--n-boot must be >= 100 for a meaningful interval.")
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        die(f"file not found: {args.data}")
    for c in [args.y_true, *args.pred] + ([args.group] if args.group else []):
        if c not in df.columns:
            p.error(f"column '{c}' not in {args.data}. Columns: {list(df.columns)[:15]}")
    cols = [args.y_true, *args.pred] + ([args.group] if args.group else [])
    before = len(df)
    df = df.dropna(subset=cols).reset_index(drop=True)
    if len(df) == 0:
        die("no rows left after dropping missing values.")
    dropped = before - len(df)

    y = df[args.y_true].to_numpy()
    pos = args.pos_label
    if pos is not None and not isinstance(y[0], str):
        try:
            pos = type(y[0])(pos)
        except ValueError:
            p.error(f"--pos-label '{pos}' cannot be converted to the label type.")
    metric_fn = make_metric(args.metric, y, pos)

    preds = {c: df[c].to_numpy() for c in args.pred}
    base_value = None
    if not args.no_baseline:
        bp, base_value = baseline_predictions(args.metric, y, args.baseline_value)
        preds = {"baseline": bp, **preds}
    groups = df[args.group].to_numpy() if args.group else None
    point, draws = bootstrap(y, preds, metric_fn, args.n_boot, args.seed, groups)

    higher = HIGHER_BETTER[args.metric]
    print(f"n={len(df)}{' (dropped %d rows with missing values)' % dropped if dropped else ''}, "
          f"metric={args.metric} ({'higher' if higher else 'lower'} is better), "
          f"{args.n_boot} bootstrap resamples{' of groups in ' + args.group if args.group else ''}")
    if len(df) < 30:
        print("WARNING: fewer than 30 rows; intervals are very wide and unreliable.")
    print(f"\n{'model':<20} {args.metric:>10}   95% CI")
    result = {"point": point, "ci": {}}
    for k in preds:
        lo, hi = interval(draws[k])
        result["ci"][k] = (lo, hi)
        label = k if k != "baseline" else f"baseline ({base_value})"
        print(f"{label:<20} {point[k]:>10.4f}   [{lo:.4f}, {hi:.4f}]")

    if len(args.pred) == 2:
        a, b = args.pred
        diff = draws[b] - draws[a]
        lo, hi = interval(diff)
        d = point[b] - point[a]
        result["diff"] = (d, lo, hi)
        print(f"\nPaired difference ({b} - {a}): {d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
        if np.isnan(lo):
            print("Verdict: too few valid resamples to say.")
        elif lo > 0 or hi < 0:
            better = b if (d > 0) == higher else a
            print(f"Verdict: clear difference; {better} is better on this test set.")
        else:
            print("Verdict: no clear difference (the interval includes 0). Do not call either model a winner.")
    if not args.no_baseline:
        best = max(args.pred, key=lambda c: point[c] if higher else -point[c])
        bl_lo, bl_hi = result["ci"]["baseline"]
        m_lo, m_hi = result["ci"][best]
        overlap = (m_lo <= bl_hi) if higher else (m_hi >= bl_lo)
        if overlap:
            print(f"\nNote: {best}'s interval overlaps the baseline. The model may not beat the dumbest guess.")
    return result


if __name__ == "__main__":
    run()
