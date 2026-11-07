#!/usr/bin/env python3
"""audit.py - profile a dataset for problems that silently break models.

Usage:
  python audit.py --data data.csv [--target y] [--time date_col] [--id id_col] [--json]

Reports an issue table (column | issue | severity | action) covering: target balance and
label conflicts, missingness (and whether missingness predicts the target), exact duplicates,
sentinel values, impossible values, outliers / unit mix-ups, constant and high-cardinality
columns, mixed types, inconsistent category spelling, future dates, drift of the target over time,
and column names that often hide future information.

Exit codes: 0 = audit completed, 2 = bad usage / unreadable input. Findings never change the
exit code; read the table. Requires: pandas, numpy, scikit-learn
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SENTINELS = {-1, -9, -99, -999, -9999, -99999, 99, 999, 9999, 99999, 999999}
NONNEG_NAME = re.compile(
    r"(age|price|amount|count|qty|quantity|salary|income|height|weight|tenure|duration|"
    r"distance|charge|cost|revenue|calls|visits|clicks)", re.IGNORECASE)
DATE_VALUE = re.compile(r"^\s*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})")
LEAKY_NAME = re.compile(
    r"(status|outcome|result|refund|closed|resolution|cancel|churn_|days_to|final_|_after|post_)",
    re.IGNORECASE)
SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


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


def is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def audit(df: pd.DataFrame, target: str | None = None, time: str | None = None,
          id_col: str | None = None) -> dict:
    issues: list[dict] = []

    def add(column, issue, severity, action):
        issues.append({"column": column, "issue": issue, "severity": severity, "action": action})

    n = len(df)
    report: dict = {"rows": n, "columns": df.shape[1], "target": None, "time_profile": None}
    if n == 0:
        add("*", "dataset has zero rows", "high", "check the export / query that produced it")
        report["issues"] = issues
        return report

    feature_cols = [c for c in df.columns if c not in (target, id_col)]

    # ---- target -----------------------------------------------------------
    y_bin = None
    clf = False
    if target:
        y = df[target]
        miss = int(y.isna().sum())
        if miss:
            add(target, f"{miss} missing labels ({100 * miss / n:.1f}%)", "high",
                "drop or label them; never impute the target")
        yv = y.dropna()
        clf = (not is_num(y)) or pd.api.types.is_bool_dtype(y) or yv.nunique() <= 20
        if yv.nunique() < 2:
            add(target, "target has fewer than 2 distinct values", "high", "no model can be trained on this")
        elif clf:
            vc = yv.astype(str).value_counts()
            share = vc / vc.sum()
            report["target"] = {"kind": "classification", "classes": {k: int(v) for k, v in vc.items()}}
            minority = float(share.min())
            if minority < 0.01:
                add(target, f"severe imbalance: rarest class is {100 * minority:.2f}%", "high",
                    "use PR-AUC / recall@precision; never headline accuracy")
            elif minority < 0.10:
                add(target, f"imbalanced: rarest class is {100 * minority:.1f}%", "medium",
                    "report baseline, PR-AUC and the confusion matrix")
            if (vc < 5).any():
                add(target, "a class has fewer than 5 examples", "medium",
                    "merge or drop it; stratified splits will fail")
            lowered = {k.strip().lower(): k for k in vc.index}
            pos = next((lowered[k] for k in ("1", "1.0", "true", "yes") if k in lowered), vc.index[-1])
            report["target"]["positive_class"] = pos
            y_bin = (y.astype(str) == pos).astype(int).where(y.notna())
        else:
            report["target"] = {"kind": "regression", "mean": float(yv.mean()), "std": float(yv.std()),
                                "min": float(yv.min()), "max": float(yv.max())}
            y_bin = None

    # ---- exact duplicates ---------------------------------------------------
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        add("*", f"{dup_rows} exact duplicate rows ({100 * dup_rows / n:.1f}%)", "medium",
            "decide if repeats are real events or ingestion bugs; dedupe BEFORE splitting")
    if id_col and id_col in df.columns:
        dup_ids = int(df[id_col].dropna().duplicated().sum())
        if dup_ids:
            add(id_col, f"{dup_ids} repeated ids", "medium",
                "one entity in many rows: split with GroupKFold on this column")

    # ---- label conflicts (noise proxy) ---------------------------------------
    if target and feature_cols and clf and report["target"] and len(feature_cols) <= 60:
        try:
            keys = pd.util.hash_pandas_object(
                df[feature_cols].astype(object).where(df[feature_cols].notna(), "").astype(str), index=False)
            g = df.assign(_k=keys).dropna(subset=[target]).groupby("_k")[target].agg(
                lambda s: s.astype(str).nunique())
            sizes = df.assign(_k=keys).groupby("_k").size()
            repeated = sizes[sizes > 1].index
            if len(repeated):
                conflict = int((g.reindex(repeated) > 1).sum())
                share = conflict / len(repeated)
                report["target"]["label_conflict_share"] = round(share, 4)
                if share > 0.05:
                    add(target, f"{100 * share:.1f}% of repeated feature rows carry conflicting labels",
                        "high" if share > 0.2 else "medium",
                        "label noise or missing features; this caps achievable accuracy")
        except Exception:  # noqa: BLE001
            pass

    # ---- per column ------------------------------------------------------------
    now = pd.Timestamp.now()
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        n_miss = int(s.isna().sum())
        pct = 100 * n_miss / n

        if n_miss == n:
            add(col, "entirely missing", "high", "drop the column")
            continue
        if pct > 50:
            add(col, f"{pct:.0f}% missing", "high", "drop, or model missingness explicitly")
        elif pct > 5:
            add(col, f"{pct:.1f}% missing", "medium", "choose imputation inside the CV pipeline")
        elif n_miss:
            add(col, f"{pct:.1f}% missing", "low", "impute inside the CV pipeline")

        # does missingness itself predict the target?
        if n_miss and col != id_col and target:
            ind = s.isna().astype(int)
            try:
                if y_bin is not None and clf:
                    ok = y_bin.notna()
                    if ind[ok].nunique() == 2 and y_bin[ok].nunique() == 2:
                        auc = roc_auc_score(y_bin[ok], ind[ok])
                        auc = max(auc, 1 - auc)
                        if auc >= 0.75:
                            add(col, f"missingness predicts target (AUC {auc:.2f})",
                                "high" if auc >= 0.9 else "medium",
                                "missing-not-at-random; likely only filled after the outcome. "
                                "Confirm it exists at prediction time")
                elif not clf and is_num(df[target]) and ind.nunique() == 2:
                    c = abs(np.corrcoef(ind[df[target].notna()], df[target].dropna())[0, 1])
                    if c >= 0.5:
                        add(col, f"missingness correlates with target (|r|={c:.2f})", "medium",
                            "missing-not-at-random; check availability at prediction time")
            except Exception:  # noqa: BLE001
                pass

        nun = int(s.nunique(dropna=True))
        if nun <= 1:
            add(col, "constant column", "low", "drop it")
            continue
        top_share = s.value_counts(normalize=True, dropna=True).iloc[0]
        if top_share > 0.995:
            add(col, f"near-constant ({100 * top_share:.2f}% one value)", "low", "probably drop")

        # numeric columns
        if is_num(s):
            v = s.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            if len(v) != len(s.dropna()):
                add(col, "contains infinite values", "high", "replace with NaN or fix upstream")
            if len(v) == 0:
                continue
            lo, hi = v.min(), v.max()
            sent = [x for x in SENTINELS if x in (lo, hi) and (v == x).mean() >= 0.01
                    and v.nunique() > 3]
            if sent:
                add(col, f"possible sentinel value(s) {sent} ({100 * (v.isin(sent)).mean():.1f}% of rows)",
                    "medium", "convert to NaN and add a missing indicator")
            if NONNEG_NAME.search(col) and (v < 0).any() and not sent:
                add(col, f"{int((v < 0).sum())} negative value(s) in a quantity-like column", "medium",
                    "check units / refunds / data entry")
            if re.search(r"age", col, re.IGNORECASE) and (v > 120).any():
                add(col, f"{int((v > 120).sum())} value(s) above 120 in an age-like column", "high",
                    "impossible values; fix or drop")
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                extreme = float(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).mean())
                if extreme > 0.01:
                    add(col, f"{100 * extreme:.1f}% extreme outliers (>3 IQR)", "medium",
                        "inspect; consider log / winsorise inside the pipeline")
            med = v.median()
            if med > 0 and hi / med > 1000:
                add(col, f"max is {hi / med:.0f}x the median", "medium",
                    "possible unit mix-up (cm vs m, cents vs dollars) or a data-entry error")
            continue

        # non-numeric columns
        so = s.dropna().astype(str)
        # dates stored as text (checked first so they are not mistaken for categories)
        sample = so.head(200)
        if sample.map(lambda x: bool(DATE_VALUE.match(x))).mean() > 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(s, errors="coerce")
            # years beyond the datetime64[ns] range (after 2262) become NaT on older pandas: still "future"
            years = pd.to_numeric(s.astype(str).str.extract(r"^\s*(\d{4})-")[0], errors="coerce")
            out_of_range = int((dt.isna() & s.notna() & (years > now.year)).sum())
            bad = int(dt.isna().sum() - n_miss) - out_of_range
            if bad > 0:
                add(col, f"{bad} unparseable date string(s)", "medium", "fix formats / timezones")
            future = int((dt > now).sum()) + out_of_range
            if future:
                add(col, f"{future} dates in the future", "high" if future / n > 0.01 else "medium",
                    "check timezone / typo; future info cannot exist at prediction time")
            continue

        if nun / max(n - n_miss, 1) > 0.9 and nun > 20 and col != id_col:
            add(col, f"{nun} distinct values, almost one per row (id or free text)", "medium",
                "exclude as a feature (memorisation); use as group key if it identifies entities")
        elif nun > 50:
            add(col, f"high cardinality ({nun} categories)", "low",
                "target-encode inside the pipeline or hash; rare categories will appear at inference")

        # numbers stored as text / mixed types
        parsed = pd.to_numeric(so, errors="coerce")
        frac = float(parsed.notna().mean())
        if frac == 1.0:
            add(col, "numbers stored as text", "low", "cast to numeric")
        elif 0.5 <= frac < 1.0:
            add(col, f"mixed types: {100 * frac:.0f}% numeric strings, rest text", "medium",
                f"inspect non-numeric values, e.g. {so[parsed.isna()].unique()[:3].tolist()}")

        # inconsistent spelling of the same category
        norm = so.str.strip().str.lower()
        if so.nunique() > norm.nunique():
            add(col, f"{so.nunique() - norm.nunique()} category variants differ only by case/whitespace",
                "medium", "normalise strings before encoding")


    # datetime-typed columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            future = int((df[col] > now).sum())
            if future:
                add(col, f"{future} dates in the future", "medium", "check timezone / typo")

    # ---- names that hide future information -----------------------------------
    for col in feature_cols:
        if LEAKY_NAME.search(str(col)):
            add(col, "name suggests information from after the outcome", "medium",
                "confirm it exists, with the same definition, at prediction time; run leakage-hunter")

    # ---- target drift over time --------------------------------------------------
    if time and time in df.columns:
        t = df[time]
        if is_num(t):
            tt = pd.to_datetime(t, errors="coerce") if t.between(1e9, 4e9).all() else None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tt = pd.to_datetime(t, errors="coerce")
        if tt is None or tt.notna().sum() == 0:
            add(time, "could not parse the time column", "medium", "fix the format before time-based splits")
        else:
            report["time_profile"] = {"min": str(tt.min()), "max": str(tt.max())}
            if target and y_bin is not None:
                monthly = y_bin.groupby(tt.dt.to_period("M")).agg(["mean", "count"])
                monthly = monthly[monthly["count"] >= 10]
                if len(monthly) >= 3:
                    report["time_profile"]["monthly_target_rate"] = {
                        str(k): round(float(v), 4) for k, v in monthly["mean"].items()}
                    spread = float(monthly["mean"].max() - monthly["mean"].min())
                    if spread > 0.10:
                        add(target, f"target rate varies by {100 * spread:.0f} points across months",
                            "medium", "distribution shifts over time: use a time-based split")

    issues.sort(key=lambda i: (SEV_ORDER[i["severity"]], str(i["column"])))
    report["issues"] = issues
    return report


def render(report: dict, target: str | None) -> str:
    out = [f"Rows: {report['rows']}   Columns: {report['columns']}",
           "Confirm with the data owner: what does ONE row represent (a user? an event? a day?)", ""]
    t = report["target"]
    if t and t["kind"] == "classification":
        out.append("Target classes: " + ", ".join(f"{k}={v}" for k, v in t["classes"].items())
                   + f"   (positive class for rates: {t.get('positive_class')})")
        if "label_conflict_share" in t:
            out.append(f"Repeated feature rows with conflicting labels: {100 * t['label_conflict_share']:.1f}%")
    elif t:
        out.append(f"Target: mean={t['mean']:.4g} std={t['std']:.4g} min={t['min']:.4g} max={t['max']:.4g}")
    tp = report["time_profile"]
    if tp:
        out.append(f"Time coverage: {tp['min']} to {tp['max']}")
        rates = tp.get("monthly_target_rate", {})
        if rates:
            lo = min(rates, key=rates.get); hi = max(rates, key=rates.get)
            cls = (t or {}).get("positive_class", "positive class")
            out.append(f"Monthly rate of target '{cls}': lowest {rates[lo]:.3f} ({lo}), "
                       f"highest {rates[hi]:.3f} ({hi}), {len(rates)} months")
    out.append("")
    if not report["issues"]:
        out.append("No issues found by these checks. That does not prove the data is clean.")
    else:
        w = max(8, min(30, max(len(str(i["column"])) for i in report["issues"])))
        out.append(f"{'column':<{w}} | {'severity':<8} | issue -> action")
        out.append("-" * (w + 40))
        for i in report["issues"]:
            out.append(f"{str(i['column']):<{w}} | {i['severity']:<8} | {i['issue']} -> {i['action']}")
    out += ["", "Still needs a human: label quality (read 20 random rows per class), train/serve skew "
            "(is each column available, same definition, at prediction time?), images/text shortcuts."]
    return "\n".join(out)


def run(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--target")
    p.add_argument("--time")
    p.add_argument("--id", dest="id_col")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    df = read_table(args.data)
    for flag, col in (("--target", args.target), ("--time", args.time), ("--id", args.id_col)):
        if col and col not in df.columns:
            p.error(f"{flag} column '{col}' not found. Columns: {list(df.columns)[:20]}")
    report = audit(df, args.target, args.time, args.id_col)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report, args.target))
    return report


if __name__ == "__main__":
    run()
