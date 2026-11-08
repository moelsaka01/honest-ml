---
name: leakage-hunter
description: Use before reporting any metric from a machine learning model (tabular, time-series, text, or image), and whenever a score looks too good (very high AUC/accuracy/R2, train and test almost equal, one feature doing all the work). Detects duplicate rows across splits, target leakage, time-order mistakes, shared users/groups, and preprocessing fitted before the split. Includes a runnable checker for CSV/Parquet.
---

# Leakage Hunter

A model that scores 0.99 on the first try is almost always leaking. Prove it is not before anyone sees the number.

`<skill-dir>` below means the folder that contains this SKILL.md. The checker needs `pandas`, `scikit-learn`, `numpy` (`pip install pandas scikit-learn`).

## Stop and investigate if any of these hold

- The score is far above the baseline or published results for the task.
- Train and test scores are nearly identical *and* very high.
- One feature gives most of the performance.
- The score drops sharply when you switch from a random split to a time-based or group-based split.

## The checklist (run every item, in order)

1. **Run the checker** whenever you have train and test files (or one file):
   ```bash
   python <skill-dir>/scripts/leakage_check.py --train train.csv --test test.csv --target <target> [--time <date_col>] [--group <id_col>]
   python <skill-dir>/scripts/leakage_check.py --data all.csv --target <target>      # random 80/20 split of one file
   ```
   It reports: duplicate feature rows shared by train and test (case, whitespace and float noise are normalised), the predictive power of each single feature on the test set, identifier-like columns, train rows dated after the earliest test row (`--time`), and groups present in both sets (`--group`). Exit code `1` means a suspected leak, `0` means none found by these checks, `2` means bad input. Add `--json` for machine-readable output.
   `[LEAK]` lines must be resolved or explained. `[WARN]` lines need a decision. A single feature scoring AUC >= 0.9 (or R2 >= 0.8) is *suspect*, not automatically wrong: confirm whether it exists at prediction time before dropping or keeping it.
   With `--data` the split is random, so time/group/duplicate results describe that random split. That is the point: they show what a naive split would leak.

   **Point-in-time check (opt-in).** A chronological split does not catch a feature that was *backfilled* after the decision (a balance, a score or a label-derived field written later). The script can only see that if the data records when each value was written. Pass the column holding the moment of prediction and each feature's timestamp:
   ```bash
   python <skill-dir>/scripts/leakage_check.py --train train.csv --test test.csv --target <target> \
     --decision-time <scored_at> --asof <feature>:<feature_updated_at>
   ```
   Columns named `<feature>_asof`, `_as_of`, `_updated_at`, `_recorded_at`, `_loaded_at`, `_ingested_at`, `_observed_at` or `_snapshot_at` are paired automatically. A feature with values stamped after the decision time is a `[LEAK]` (over 1% of rows) or `[WARN]` (a few rows). **If no as-of timestamps exist, the script says it cannot tell** and you must ask the data owner for load or snapshot history; do not report availability as verified.

2. **Target leakage (check by reading the columns).** For every feature ask: *would this value be known at the moment of prediction, with this definition, as it was then (not as it was corrected or backfilled later)?* Red flags: names like `*_status`, `*_outcome`, `*_result`, `days_to_*`, `closed_at`, `refund_*`, `cancellation_*`, columns filled in only after the event, IDs that encode the label, and aggregates computed over the whole dataset (including the future).

3. **Preprocessing leakage (the checker cannot see this; read the code).** Scalers, imputers, encoders, PCA, feature selection, SMOTE/oversampling, target encoding and text vectorizers must be **fitted on train only**. Wrap them with the model in a `sklearn.pipeline.Pipeline` and cross-validate the pipeline. Never `fit`/`fit_transform` on the full dataset before splitting.

4. **Temporal leakage.** If rows have timestamps and the model will predict the future, split by time (train on the past, test on the future) and use `TimeSeriesSplit`, not shuffled `KFold`. Rolling/lag features must use only the past: `df[col].shift(1).rolling(w).mean()`.

5. **Group leakage.** If the same user, patient, customer, device, document source or image source appears in many rows, keep all of its rows in one split: `GroupKFold` / `GroupShuffleSplit` / `StratifiedGroupKFold` with `groups=<id column>`.

6. **Tuning on the test set.** Hyperparameters, thresholds, early stopping and feature choices are decided on validation data or inner CV. The test set is used once, at the end. If it was looked at earlier, say so in the report.

7. **Text, image and LLM data.** Check for near-duplicates across splits. Text: normalise (lowercase, collapse whitespace) and hash:
   ```python
   import hashlib, re
   norm = lambda s: hashlib.md5(re.sub(r"\s+", " ", s.lower()).strip().encode()).hexdigest()
   overlap = set(map(norm, train_texts)) & set(map(norm, test_texts))
   ```
   Images: compare perceptual hashes. For LLM benchmarks, check that test items do not appear in few-shot prompts, fine-tuning data or retrieval corpora.

## Output contract

Before reporting any metric, write this report (see `honest-eval` for the metric table that follows it):

```
Leakage report
- Split strategy: <random | time | group> because <reason>
- Checker: <ran, exit code N | not run because ...>
- Duplicate overlap train/test: <n rows / %>
- Top single-feature scores: <feature: score, ...> -> <cleared | removed | investigating>
- Preprocessing fitted on: <train only via Pipeline | ...>
- Test set used: <once, at the end | also earlier for ...>
- Verdict: <clean | leak found and fixed (metric before -> after) | unresolved risk: ...>
```

If anything is unresolved, say so next to the metric. Never present a number you have not cleared. When the verdict is "leak found and fixed", show the metric before and after; the drop is the finding.

## Works with

`dataset-audit` runs before this (it finds columns that will not exist at prediction time). `honest-eval` runs after it (baseline, interval, single test use). `ml-roast` re-checks these items when reviewing code.
