#!/usr/bin/env python3
"""leakage_check.py - fast sanity checks for ML data leakage.

Checks (tabular data, CSV or Parquet):
  1. Duplicate feature rows shared by train and test (normalised: case, whitespace, float noise)
  2. Single-feature predictive power on the test set (target-leak suspects)
  3. Identifier-like columns used as features
  4. Time order: train rows dated after the earliest test row   (--time)
  5. Group overlap: the same user/customer/patient in both sets (--group)
  6. Point in time: feature values recorded AFTER the moment of prediction (--decision-time)

Usage:
  python leakage_check.py --train train.csv --test test.csv --target y
  python leakage_check.py --data all.csv --target y --test-size 0.2
  Optional: --time <col> --group <col> --top 10 --json

Point-in-time check (opt-in). A chronological split does not catch a feature that was backfilled
after the decision. That needs a timestamp for when each value was written. Give the column that
holds the moment of prediction and name each feature's timestamp column:
  --decision-time scored_at --asof balance:balance_updated_at --asof risk:risk_loaded_at
Columns named <feature>_asof / _as_of / _updated_at / _recorded_at / _loaded_at / _ingested_at /
_observed_at / _snapshot_at are paired with <feature> automatically. Without such timestamps the
script cannot tell a backfilled value from an honest one, and it says so.

Exit codes: 0 = no hard leak found, 1 = suspected leakage, 2 = bad usage / unreadable input.

NOT covered (needs a code review, see SKILL.md): preprocessing fitted before the split,
tuning on the test set, and feature availability when no as-of timestamps are provided.

Requires: pandas, scikit-learn, numpy
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

AUC_FLAG, AUC_WARN = 0.90, 0.80
R2_FLAG, R2_WARN = 0.80, 0.60
TIME_LEAK_PCT = 0.01  # >1% of train rows after the earliest test row = leak, below = warn
MISSING = "__missing__"
ID_NAME = re.compile(r"(^|_)(id|uuid|guid|key|index)$", re.IGNORECASE)
LEAKY_NAME = re.compile(
    r"(status|outcome|result|refund|closed|resolution|cancel|churn_|days_to|final_|_after|post_)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- io
def die(message: str):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def read_table(path: str) -> pd.DataFrame:
    try:
        if str(path).lower().endswith((".parquet", ".pq")):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except FileNotFoundError:
        die(f"file not found: {path}")
    except Exception as exc:  # noqa: BLE001
        die(f"could not read {path}: {exc}")


def load(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.data:
        df = read_table(args.data)
        train, test = train_test_split(df, test_size=args.test_size, random_state=0)
    else:
        train, test = read_table(args.train), read_table(args.test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------- helpers
def finding(check: str, level: str, message: str, **details) -> dict:
    """level: ok | warn | leak | skip | info"""
    return {"check": check, "level": level, "message": message, "details": details}


def is_classification(y: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(y)) or pd.api.types.is_bool_dtype(y) or y.nunique() <= 20


def _normalise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(float).round(6).astype(str)
        else:
            out[c] = s.astype(object).where(s.notna(), "").astype(str).str.strip().str.lower()
    return pd.DataFrame(out, index=df.index)


def _categories(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), MISSING).astype(str)


# ---------------------------------------------------------------------- checks
def check_duplicates(train, test, target):
    cols = [c for c in train.columns if c != target and c in test.columns]
    if not cols:
        return finding("duplicates", "skip", "No shared feature columns to compare.")
    a = pd.util.hash_pandas_object(_normalise(train, cols), index=False)
    b = pd.util.hash_pandas_object(_normalise(test, cols), index=False)
    overlap = int(b.isin(set(a)).sum())
    pct = 100 * overlap / max(len(test), 1)
    level = "leak" if overlap else "ok"
    return finding(
        "duplicates", level,
        f"Test rows whose features also appear in train: {overlap} ({pct:.1f}%)",
        rows=overlap, percent=round(pct, 2),
    )


def _score_column(x_tr, x_te, y_tr, y_te, clf):
    """Predictive power of one column alone, fitted on train, scored on test."""
    if pd.api.types.is_numeric_dtype(x_tr) or pd.api.types.is_bool_dtype(x_tr):
        a = x_tr.astype(float).replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_frame()
        b = x_te.astype(float).replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_frame()
        leaf = max(5, int(0.01 * len(a)))
        model = (DecisionTreeClassifier if clf else DecisionTreeRegressor)(
            max_depth=4, min_samples_leaf=leaf, random_state=0)
        model.fit(a, y_tr)
        if clf:
            pred = model.predict_proba(b)[:, list(model.classes_).index(1)]
        else:
            pred = model.predict(b)
    else:  # categorical / text: smoothed target encoding fitted on train
        k_tr, k_te = _categories(x_tr), _categories(x_te)
        prior = float(y_tr.mean())
        stats = y_tr.groupby(k_tr).agg(["mean", "count"])
        enc = (stats["count"] * stats["mean"] + 10 * prior) / (stats["count"] + 10)
        pred = k_te.map(enc).fillna(prior).astype(float).to_numpy()
    if clf:
        auc = roc_auc_score(y_te, pred)
        return max(auc, 1 - auc)
    return r2_score(y_te, pred)


def single_feature_scores(train, test, target, top):
    y_tr, y_te = train[target], test[target]
    clf = is_classification(y_tr)
    if clf:
        s_tr, s_te = y_tr.astype(str), y_te.astype(str)
        pos = s_tr.value_counts().index[0]
        y_tr, y_te = (s_tr == pos).astype(int), (s_te == pos).astype(int)
        if y_tr.nunique() < 2:
            return [finding("single_feature", "skip", "Train target has a single class.")]
        if y_te.nunique() < 2:
            return [finding("single_feature", "skip", "Test target has a single class; AUC undefined.")]
        metric, flag, warn = "AUC", AUC_FLAG, AUC_WARN
        note = f"one-vs-rest on class '{pos}'" if s_tr.nunique() > 2 else f"positive class '{pos}'"
    else:
        y_tr, y_te = y_tr.astype(float), y_te.astype(float)
        if y_tr.nunique() < 2:
            return [finding("single_feature", "skip", "Train target is constant.")]
        metric, flag, warn = "R2", R2_FLAG, R2_WARN
        note = "regression"

    results, skipped = [], []
    for col in train.columns:
        if col == target or col not in test.columns:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = _score_column(train[col], test[col], y_tr, y_te, clf)
            if score == score:  # not NaN
                results.append((col, float(score)))
        except Exception as exc:  # noqa: BLE001
            skipped.append((col, str(exc)[:80]))
    results.sort(key=lambda t: t[1], reverse=True)

    out = []
    shown = {c for c, _ in results[:top]} | {c for c, s in results if s >= warn}
    rows = [(c, s) for c, s in results if c in shown]
    flagged = [c for c, s in results if s >= flag]
    warned = [c for c, s in results if warn <= s < flag]
    level = "leak" if flagged else "warn" if warned else "ok"
    msg = (f"Single-feature {metric} on test ({note}); flag >= {flag}, warn >= {warn}. "
           + (f"Suspects: {', '.join(flagged)}." if flagged else "No column predicts the target alone."))
    out.append(finding("single_feature", level, msg, metric=metric,
                       scores={c: round(s, 4) for c, s in rows}, flagged=flagged, warned=warned))
    if skipped:
        out.append(finding("single_feature_skipped", "info",
                           "Columns that could not be scored: " + "; ".join(f"{c} ({e})" for c, e in skipped)))
    return out


def check_identifiers(train, target):
    hits = []
    n = len(train)
    for col in train.columns:
        if col == target:
            continue
        s = train[col]
        name_hit = bool(ID_NAME.search(col))
        unique_hit = (n >= 50 and s.nunique(dropna=True) / n > 0.95
                      and not pd.api.types.is_float_dtype(s))
        if name_hit or unique_hit:
            hits.append(col)
    if hits:
        return finding("identifiers", "warn",
                       f"Identifier-like columns (drop as features, or use only as --group): {', '.join(hits)}",
                       columns=hits)
    return finding("identifiers", "ok", "No identifier-like feature columns.")


def check_names(train, target):
    hits = [c for c in train.columns if c != target and LEAKY_NAME.search(c)]
    if hits:
        return finding("name_hints", "info",
                       "Column names that often hide future information (confirm each exists at "
                       f"prediction time): {', '.join(hits)}", columns=hits)
    return None


def _to_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


def check_time(train, test, col):
    for name, df in (("train", train), ("test", test)):
        if col not in df.columns:
            return finding("time_order", "skip", f"Time column '{col}' missing from {name}.")
    t_tr, t_te = _to_time(train[col]), _to_time(test[col])
    n_tr, n_te = int(t_tr.notna().sum()), int(t_te.notna().sum())
    if n_tr == 0 or n_te == 0:
        return finding("time_order", "skip", f"Could not parse '{col}' as dates or numbers.")
    cutoff = t_te.min()
    later = int((t_tr > cutoff).sum())
    pct = later / n_tr
    level = "leak" if pct > TIME_LEAK_PCT else "warn" if later else "ok"
    msg = (f"Train rows dated after the earliest test row: {later} ({100 * pct:.1f}% of train). "
           f"train max={t_tr.max()}, test min={cutoff}")
    if level == "leak":
        msg += ". Looks like a random split on time-ordered data; split by time instead."
    return finding("time_order", level, msg, rows=later, percent=round(100 * pct, 2))


def check_group(train, test, col):
    for name, df in (("train", train), ("test", test)):
        if col not in df.columns:
            return finding("group_overlap", "skip", f"Group column '{col}' missing from {name}.")
    shared = set(train[col].dropna()) & set(test[col].dropna())
    rows = int(test[col].isin(shared).sum())
    pct = 100 * rows / max(len(test), 1)
    level = "leak" if shared else "ok"
    return finding("group_overlap", level,
                   f"Groups in both train and test ('{col}'): {len(shared)} "
                   f"({rows} test rows, {pct:.1f}%)", groups=len(shared), test_rows=rows)


ASOF_SUFFIX = re.compile(
    r"(_asof|_as_of|_updated_at|_recorded_at|_loaded_at|_ingested_at|_observed_at|_snapshot_at)$",
    re.IGNORECASE)


def find_asof_pairs(columns, explicit, decision):
    """{feature: timestamp column}. Explicit --asof pairs win; the rest are matched by name suffix."""
    pairs = {}
    for spec in explicit:
        if ":" not in spec or not all(part.strip() for part in spec.split(":", 1)):
            die(f"--asof expects FEATURE:TIMESTAMP_COLUMN, got '{spec}'")
        feature, stamp = (part.strip() for part in spec.split(":", 1))
        pairs[feature] = stamp
    present = set(columns)
    for col in columns:
        match = ASOF_SUFFIX.search(col)
        if match and col != decision:
            feature = col[:match.start()]
            if feature in present and feature not in pairs:
                pairs[feature] = col
    return pairs


def check_asof(train, test, decision, pairs):
    """Flag features whose value was written after the moment the prediction would have been made."""
    for name, df in (("train", train), ("test", test)):
        if decision not in df.columns:
            return finding("point_in_time", "skip", f"Decision-time column '{decision}' missing from {name}.")
    if not pairs:
        return finding(
            "point_in_time", "info",
            f"Decision time '{decision}' given, but no per-feature as-of timestamps found, so a value backfilled "
            "after the decision cannot be told from an honest one. Add FEATURE_asof columns or pass "
            "--asof FEATURE:TIMESTAMP_COLUMN, or ask the data owner for load / snapshot history.")
    rows, unusable = {}, []
    for feature, stamp in pairs.items():
        for name, df in (("train", train), ("test", test)):
            if feature not in df.columns or stamp not in df.columns:
                unusable.append(f"{feature} ({'feature' if feature not in df.columns else stamp} missing from {name})")
                break
            try:
                recorded, decided = _to_time(df[stamp]), _to_time(df[decision])
                valid = recorded.notna() & decided.notna() & df[feature].notna()
                late = int((valid & (recorded > decided)).sum())
            except TypeError:
                unusable.append(f"{feature} ({stamp} and {decision} are different kinds of time)")
                break
            n = int(valid.sum())
            rows.setdefault(feature, {})[name] = {"late": late, "checked": n,
                                                  "percent": round(100 * late / n, 2) if n else 0.0}
    worst = {f: max(s["percent"] for s in sp.values()) for f, sp in rows.items()}
    leaks = [f for f, w in worst.items() if w > 100 * TIME_LEAK_PCT]
    warns = [f for f, w in worst.items() if 0 < w <= 100 * TIME_LEAK_PCT]
    level = "leak" if leaks else "warn" if warns else "ok"
    if not rows:
        return finding("point_in_time", "skip", "No usable as-of pairs: " + "; ".join(unusable))

    def describe(features):
        return ", ".join(
            f"{f} ({sum(s['late'] for s in rows[f].values())} rows, up to {worst[f]:.1f}% of a split)"
            for f in features)

    if leaks:
        msg = ("Features whose value was recorded AFTER the decision time (not known when predicting): "
               + describe(leaks) + ". Rebuild them as of the decision time, or drop them.")
    elif warns:
        msg = "A few rows have a feature value recorded after the decision time: " + describe(warns)
    else:
        msg = (f"No checked feature was recorded after decision time '{decision}' "
               f"({len(rows)} feature(s) with as-of timestamps).")
    if unusable:
        msg += " Not checked: " + "; ".join(unusable) + "."
    return finding("point_in_time", level, msg, decision_time=decision, features=rows,
                   flagged=leaks, warned=warns, pairs={f: pairs[f] for f in rows})


# ------------------------------------------------------------------------ main
TAGS = {"ok": "[ok]   ", "warn": "[WARN] ", "leak": "[LEAK] ", "skip": "[skip] ", "info": "[info] "}


def run(argv=None) -> tuple[int, list[dict]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train"); p.add_argument("--test"); p.add_argument("--data")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--target", required=True)
    p.add_argument("--time"); p.add_argument("--group")
    p.add_argument("--decision-time", help="column holding the moment each prediction would be made "
                   "(enables the point-in-time check)")
    p.add_argument("--asof", action="append", default=[], metavar="FEATURE:TIMESTAMP_COLUMN",
                   help="when this feature's value was recorded; repeatable. Needs --decision-time "
                   "(or --time, which is then used as the decision time)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true", help="print findings as JSON")
    args = p.parse_args(argv)

    if not args.data and not (args.train and args.test):
        p.error("provide --data, or both --train and --test.")
    if args.data and (args.train or args.test):
        p.error("use either --data or --train/--test, not both.")
    if not 0 < args.test_size < 1:
        p.error("--test-size must be between 0 and 1.")
    decision = args.decision_time or (args.time if args.asof else None)
    if args.asof and not decision:
        p.error("--asof needs --decision-time (the column with the moment of prediction).")

    train, test = load(args)
    for name, df in (("train", train), ("test", test)):
        if args.target not in df.columns:
            p.error(f"target '{args.target}' not in {name} columns: {list(df.columns)[:15]}")
        if len(df) == 0:
            p.error(f"{name} set is empty.")

    findings: list[dict] = []
    dropped = int(train[args.target].isna().sum() + test[args.target].isna().sum())
    train = train.dropna(subset=[args.target]).reset_index(drop=True)
    test = test.dropna(subset=[args.target]).reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        p.error("no rows with a target value left after dropping missing targets.")
    if dropped:
        findings.append(finding("target_missing", "info", f"Dropped {dropped} rows with a missing target."))

    findings.append(check_duplicates(train, test, args.target))
    findings.extend(single_feature_scores(train, test, args.target, args.top))
    findings.append(check_identifiers(train, args.target))
    hint = check_names(train, args.target)
    if hint:
        findings.append(hint)
    if args.time:
        findings.append(check_time(train, test, args.time))
    if args.group:
        findings.append(check_group(train, test, args.group))
    if decision:
        pairs = find_asof_pairs(list(train.columns), args.asof, decision)
        findings.append(check_asof(train, test, decision, pairs))

    leaks = [f for f in findings if f["level"] == "leak"]
    warns = [f for f in findings if f["level"] == "warn"]
    if leaks:
        verdict = "SUSPECTED LEAKAGE - investigate the [LEAK] items before reporting any metric"
    elif warns:
        verdict = "No hard leak found, but review the [WARN] items"
    else:
        verdict = "No leakage detected by these checks"

    if args.json:
        print(json.dumps({"verdict": verdict, "findings": findings}, indent=2, default=str))
    else:
        print(f"Train {train.shape}, test {test.shape}, target '{args.target}'"
              + (" (random split of --data; time/group/duplicate checks describe that split)" if args.data else "")
              + "\n")
        for f in findings:
            print(TAGS[f["level"]] + f["message"])
            if f["check"] == "single_feature":
                for c, s in f["details"].get("scores", {}).items():
                    mark = "<-- suspect" if c in f["details"]["flagged"] else ""
                    print(f"          {c:<36} {s:.3f} {mark}".rstrip())
        print(f"\nVerdict: {verdict}")
        print("Not covered by this script: preprocessing fitted before the split, tuning on test"
              + ("." if decision else ", and whether values were backfilled after the decision "
                 "(needs as-of timestamps: see --decision-time and --asof)."))
        print("Review the code for those, and confirm with the data owner that each feature exists at "
              "prediction time.")
    return (1 if leaks else 0), findings


def _safe_console():
    """Column names can be non-ASCII; never crash on a legacy console encoding."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main():
    _safe_console()
    code, _ = run()
    sys.exit(code)


if __name__ == "__main__":
    main()
