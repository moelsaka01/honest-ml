---
name: leakage-hunter
description: Use whenever you build, train, or report results for a machine learning model on tabular, time-series, text, or image data, and before claiming any metric. Detects train/test contamination, target leakage, temporal leakage, group leakage, and preprocessing fitted on the full dataset. Trigger on "train a model", "accuracy is", "AUC", "suspiciously good", "why is my model so good", or any sklearn/xgboost/lightgbm/pytorch training code.
---

# Leakage Hunter

A model that scores 0.99 on day one is almost always leaking. Your job is to prove it isn't before anyone sees the number.

## When a metric looks too good, assume leakage first

Stop and investigate if any of these hold:
- Test score is within ~1% of train score on a hard problem.
- A single feature gives most of the performance.
- Performance is far above published baselines for the task.
- Performance collapses when you change the random split to a time-based or group-based split.

## The checklist (run all of it, in order)

1. **Duplicate rows across splits.** Exact and near-duplicate rows in both train and test inflate scores. Run the bundled script:
   ```bash
   python <skill-dir>/scripts/leakage_check.py --train train.csv --test test.csv --target <target_col> [--time <time_col>] [--group <group_col>]
   ```
   It reports duplicate overlap, single-feature AUC/R² (target-leak suspects), time-order violations, and group overlap.

2. **Target leakage.** For every feature ask: *would this value be known at prediction time?* Red flags: columns named `*_status`, `*_outcome`, `*_result`, `days_to_*`, `closed_at`, `refund_*`, IDs that encode the label, or aggregates computed over the whole dataset. Any single feature with AUC > 0.9 (classification) or R² > 0.8 (regression) is guilty until proven innocent.

3. **Preprocessing leakage.** Scalers, imputers, encoders, PCA, feature selection, SMOTE/oversampling, and target encoding must be **fit on train only**. Enforce it with a `sklearn.pipeline.Pipeline` inside cross-validation. Never call `fit_transform` on the full dataset before splitting.

4. **Temporal leakage.** If rows have a timestamp and the model will predict the future, split by time (train on past, test on future). Use `TimeSeriesSplit`, never shuffled `KFold`. Rolling features must only use past windows (`shift(1)` before `rolling`).

5. **Group leakage.** If the same patient, user, customer, device, image source, or document appears multiple times, all of its rows must land in one split. Use `GroupKFold` / `GroupShuffleSplit`.

6. **Tuning on the test set.** Hyperparameters, thresholds, early stopping, and feature choices must be decided on a validation set or inner CV. The test set is touched **once**, at the end.

7. **Text/LLM data.** Check for benchmark contamination (test items appearing in pretraining or few-shot prompts) and near-duplicate documents across splits (normalize whitespace/case and hash).

## Output contract

Before reporting any metric, write a short **Leakage report**:

```
Leakage report
- Split strategy: <random | time | group> because <reason>
- Duplicate overlap train/test: <n rows / %>
- Top single-feature scores: <feature: score, ...> -> <cleared | removed | investigating>
- Preprocessing fit on: <train only via Pipeline | ...>
- Test set used: <once, at the end>
- Verdict: <clean | leak found and fixed | unresolved risk: ...>
```

If any item is unresolved, say so next to the metric. Never present a number you have not cleared.
