---
name: dataset-audit
description: Use at the start of any ML or data-science task, before modeling, when given a new dataset (CSV, Parquet, database table, image folder, text corpus). Profiles the data and finds problems that silently break models - label noise, class imbalance, missingness patterns, duplicates, outliers, unit errors, and train/serve skew. Trigger on "here's the data", "load the dataset", "EDA", "explore", or the first time a dataset appears in a task.
---

# Dataset Audit

Most "model problems" are data problems found too late. Audit before you model.

## Steps

1. **Shape and meaning.** Rows, columns, dtypes, and what one row represents (a user? a transaction? a day?). If you can't say what a row is, ask.

2. **Target.** Distribution, class balance, missing labels, and how the label was produced (human annotation? a rule? a future event?). Labels derived from a rule mean the model may just learn the rule.

3. **Missingness.** Percent missing per column, and whether missingness correlates with the target (missing-not-at-random is a signal *and* a leakage risk).

4. **Duplicates.** Exact duplicate rows and duplicate IDs. Decide whether they're real repeats or ingestion bugs.

5. **Ranges and units.** Min/max/percentiles for numeric columns. Look for impossible values (negative ages, future dates, 999 sentinels, mixed units like cm and m, timezone mixing).

6. **Cardinality.** High-cardinality categoricals (IDs, free text) and near-constant columns. IDs as features usually mean memorization.

7. **Time coverage.** Date range, gaps, and whether the distribution shifts over time (plot target rate by month).

8. **Train/serve skew.** Which columns will actually exist, with the same definition, at prediction time? Mark any that won't.

9. **Images/text.** Sample and look at 20 random examples per class. Check resolution/length distribution, corrupted files, near-duplicates, and watermark or filename shortcuts that reveal the label.

## Output contract

Produce a **Data audit** with: one-line description of a row, target summary, a table of issues (`column | issue | severity | action`), and a list of columns excluded and why. Flag anything that should be confirmed with the data owner instead of guessing.
