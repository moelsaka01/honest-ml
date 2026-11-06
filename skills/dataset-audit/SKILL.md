---
name: dataset-audit
description: Use the first time a new dataset (CSV, Parquet, table, image folder, text corpus) appears in a machine learning task, before any modelling. Finds label noise and conflicts, class imbalance, missingness that predicts the target, duplicates, sentinel and impossible values, unit mix-ups, mixed types, future dates, drift over time, and columns that will not exist at prediction time. Includes a runnable profiler for CSV/Parquet.
---

# Dataset Audit

Most "model problems" are data problems found too late. Audit before you model.

`<skill-dir>` means the folder that contains this SKILL.md. The profiler needs `pandas`, `numpy`, `scikit-learn`.

## Steps

1. **Run the profiler** on tabular data:
   ```bash
   python <skill-dir>/scripts/audit.py --data data.csv --target <target> [--time <date_col>] [--id <id_col>]
   ```
   It prints an issue table (`column | severity | issue -> action`) covering: missing labels and class balance, repeated feature rows with conflicting labels (a label-noise proxy), missingness per column and whether missingness predicts the target, exact duplicates and repeated ids, sentinel values (-999, 9999...), negative or impossible values in quantity-like columns, extreme outliers and max/median ratios (unit mix-ups), constant and near-constant columns, id-like and high-cardinality columns, numbers stored as text, mixed types, category spellings that differ only by case/whitespace, unparseable and future dates, target drift across months, and column names that hint at post-outcome information. Add `--json` for structured output. Exit code is `0` even with findings; usage errors exit `2`.

2. **Shape and meaning.** State what one row represents (a user? a transaction? a day?). If you cannot, ask. The script cannot know this.

3. **Target provenance.** How was the label produced: human annotation, a rule, a future event? Labels derived from a rule mean the model may just relearn the rule.

4. **Read the data.** For images and text, view about 20 random examples per class. Check resolution/length distributions, corrupted files, near-duplicates, and shortcuts that reveal the label (watermarks, filenames, folder names, boilerplate).

5. **Train/serve skew.** For every column, decide whether it exists, with the same definition and freshness, at prediction time. Mark any that will not. The script flags suspicious *names* and missingness patterns; only the data owner can confirm.

6. **Act.** Fix what is fixable, exclude what is not, and list anything to confirm with the data owner instead of guessing. Do any dedupe and cleaning *before* splitting, and do any fitted imputation inside the cross-validated pipeline.

## Output contract

Produce a **Data audit**:

```
Data audit: <file>
- One row = <description>          - Rows x columns: <n x m>
- Target: <summary: balance, missing, how produced>
- Issues:
  column | issue | severity | action
- Columns excluded and why: <...>
- To confirm with the data owner: <...>
```

## Works with

Hand the cleaned data and the excluded-column list to `leakage-hunter` (which checks the splits) and `reproducible-runs` (which records the data hash).
