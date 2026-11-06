---
name: honest-eval
description: Use whenever you evaluate, compare, or report the performance of a machine learning model or an LLM pipeline. Forces a baseline, the right metric for the problem, confidence intervals, and a single untouched test set. Trigger on "evaluate", "compare models", "which model is better", "report results", "accuracy", "F1", "beat the baseline", or writing results tables.
---

# Honest Eval

A metric without a baseline and an error bar is a rumor. Report numbers a skeptical reviewer would accept.

## Rules

1. **Baseline first.** Before any real model, compute the dumbest reasonable baseline and report it in the same table:
   - Classification: majority class (`DummyClassifier(strategy="most_frequent")`) and stratified random.
   - Regression: predict the train mean/median.
   - Time series: last value (naive) and seasonal naive.
   - Ranking/retrieval: BM25 or random.
   - LLM tasks: the simplest prompt with the cheapest model.
   If the model barely beats the baseline, say that plainly.

2. **Pick the metric for the problem, not the one that looks best.**
   - Imbalanced classes: never headline accuracy. Use PR-AUC, recall at a fixed precision, or F1 for the minority class, and show the confusion matrix.
   - Costs differ per error type: report the cost-weighted metric or the operating point.
   - Regression: MAE alongside RMSE; MAPE only if no target is near zero.
   - Probabilities used downstream: report calibration (Brier score, reliability curve).

3. **Uncertainty is mandatory.** Report mean ± std over CV folds, or a 95% bootstrap CI on the test set (1,000 resamples). When comparing two models on the same data, use a paired test (paired bootstrap or McNemar). A difference inside the CI is "no clear difference", not a win.

4. **One test set, touched once.** Model selection, tuning, thresholds, and early stopping use validation or inner CV. Evaluate on test exactly once at the end. If you looked at test earlier, say so.

5. **Slice the results.** Report performance on meaningful subgroups (class, region, device, document length, new vs returning users). An average hides where the model fails.

6. **Match evaluation to deployment.** If production predicts the future, evaluate on a later time window. If production sees new users, hold out users, not rows.

## Output contract

Every results summary must include a table like:

```
| Model     | Metric (primary) | 95% CI        | Secondary metric | Notes             |
|-----------|------------------|---------------|------------------|-------------------|
| Baseline  | 0.50             | [0.48, 0.52]  | ...              | majority class    |
| Candidate | 0.71             | [0.68, 0.74]  | ...              | tuned on val only |
```

Then one sentence: is the improvement real, how big is it in practical terms, and what is the biggest remaining risk. Pair with `leakage-hunter` before trusting any number.
