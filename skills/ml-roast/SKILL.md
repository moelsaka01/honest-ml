---
name: ml-roast
description: Use when asked to review, audit, critique, or "roast" machine learning code, a notebook, a Kaggle solution, an ML pull request, or the experiment section of a paper, or to judge whether reported ML results can be trusted. Produces a blunt, prioritised review with exact fixes and a trust score out of 10. Includes a static scanner for .py and .ipynb files.
---

# ML Roast

Be the reviewer who rejects the paper: kindly, specifically, and with fixes.

`<skill-dir>` means the folder that contains this SKILL.md. The scanner uses only the standard library.

## Process

1. **Run the scanner first** for a fast, objective pass:
   ```bash
   python <skill-dir>/scripts/roast_scan.py path/to/train.py        # file, notebook or directory
   ```
   It parses the code (Python `ast`, notebooks cell by cell) and tracks which variables hold train, test, validation or full data, following `train_test_split` outputs, names like `X_train` / `y_test` / `train_idx`, and assignments. Strings, comments and docstrings never trigger anything. It reports, with file and line and a fix:
   - **FATAL**: R1 preprocessing / SMOTE / feature selection fitted on the full data, before the split or before cross-validation; R2 anything fitted on test data.
   - **SERIOUS**: R3 random split though the code handles dates; R4 id-like columns with no group split; R5 no seed; R6 early stopping on the test set; R7 metric with no baseline; R9 torch training that never calls `.eval()`; R10 softmax inside a model trained with `CrossEntropyLoss`; R12 rolling windows without `shift`; R13 a loop that picks the best candidate using the test set; R15 imputation / standardisation with full-data statistics.
   - **MINOR**: R8 `dropna()` without `subset`; R11 no Pipeline or saved artifact; R14 `get_dummies` before the split.
   Exit code `1` means at least one FATAL hit. `--json` gives structured output. A file it cannot parse is listed as `NOT SCANNED`, never skipped silently.
   It is a **data-flow check, not a proof**. It cannot see leakage that depends on what the data means (a feature that encodes the label, a time-dependent task with no date column). Confirm every hit by opening the line, and still read the code: no findings does not mean clean.

2. **Then read the code yourself**, in this order (most damaging first):
   1. **Leakage** (`leakage-hunter`): shared groups, target-derived features, tuning on test, features unavailable at prediction time.
   2. **Evaluation** (`honest-eval`): no baseline, wrong metric for imbalance, no error bars, single seed, cherry-picked results, test set reused.
   3. **Data** (`dataset-audit`): unexamined duplicates, label noise, train/serve skew, silent row drops, unseen categories at inference.
   4. **Correctness bugs**: shape/broadcast mistakes, wrong axis, labels misaligned after a shuffle or merge, dropout/batchnorm in the wrong mode, softmax applied twice, loss on logits vs probabilities mixed up, class-index order mismatch.
   5. **Reproducibility** (`reproducible-runs`): seeds, config, pinned environment, data and code identity.
   6. **Engineering**: copy-pasted cells, magic numbers, no tests for feature functions, no saved pipeline (preprocessor and model together).

3. If a results claim is made, check that the numbers exist (a run log, a metrics file) and that the claim fits them. Say so when you cannot verify a number.

## Format

Group findings by severity:

- **Fatal**: results are invalid until fixed.
- **Serious**: results are probably inflated or fragile.
- **Minor**: hygiene and maintainability.

For each finding give: file and line (or cell), what is wrong in one sentence, why it matters, and the exact fix as code. End with a **Trust score from 0 to 10** and the single most important fix. One line on what is done well is enough; no padding.

## Trust score rubric

| Score | Meaning |
|---|---|
| 9-10 | Every check above passes with evidence (logs, intervals, seeds). |
| 7-8 | Minor issues only. |
| 5-6 | One or more serious issues unresolved; direction of the result probably holds, size does not. |
| 3-4 | One fatal issue or several serious ones; do not quote the number. |
| 0-2 | Multiple fatal issues, or the number cannot be reproduced or traced. |

The scanner prints a "scan score". Use it as a floor-check, not as your answer: your score also weighs evaluation design and data issues a scanner cannot see.

## Example

Input: `python <skill-dir>/scripts/roast_scan.py examples/bad_example.py` on the repo's deliberately flawed churn script gives:

```
FATAL (1)
  bad_example.py:20  [R1]  `scaler.fit_transform(...)` fits a preprocessing/sampling step before the train/test split
SERIOUS (3)
  bad_example.py:22  [R5]  2 call(s) without random_state/seed (RandomForestClassifier, train_test_split) and no global seed
  bad_example.py:22  [R3]  random/shuffled split although the code handles date/time columns
  bad_example.py:27  [R7]  metric is computed but no baseline model (Dummy*, majority/mean prediction) is compared
MINOR (3)
  bad_example.py:12  [R8]  `dropna()` without `subset=` silently drops rows that have any NaN
  bad_example.py:16  [R14] pd.get_dummies before the split
  bad_example.py:19  [R11] preprocessing is applied step by step with no Pipeline or saved artifact
```

and the roast is:

> **Trust 3/10.** *Fatal:* `bad_example.py:20` fits `StandardScaler` on every row before `train_test_split`, so test statistics leak into training. Fix: `make_pipeline(StandardScaler(), RandomForestClassifier(random_state=SEED))` fitted on `X_train` only. *Serious:* line 22 splits randomly although the data has signup dates and repeated `customer_id`s: use a time cutoff and `GroupShuffleSplit(groups=df.customer_id)`; there is no seed, so the accuracy changes every run; and line 27 prints accuracy with no `DummyClassifier` baseline on an imbalanced target (about 17% churn, so predicting "no" scores about 83%). *Minor:* `dropna()` silently drops rows, and `get_dummies` before the split can give train and production different columns. **Most important fix:** split by time and group first, then fit everything on train. Done well: it keeps the target out of the feature matrix.

## Works with

`ml-roast` is the reviewer for all other skills. It calls on `leakage-hunter`, `honest-eval`, `dataset-audit` and `reproducible-runs` for their checks and on `llm-eval` for LLM changes.
