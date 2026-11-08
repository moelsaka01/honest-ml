---
name: honest-eval
description: Use when reporting, comparing, or choosing between machine learning models, i.e. writing a results table or saying one model is better than another. Forces a baseline, a metric that fits the problem, confidence intervals, a paired comparison, and a test set used once. For LLM apps, prompts, RAG or agents use llm-eval instead. Includes a bootstrap CI / paired-comparison script.
---

# Honest Eval

A metric without a baseline and an error bar is a rumor. Report numbers a skeptical reviewer would accept.

`<skill-dir>` means the folder that contains this SKILL.md. The script needs `pandas`, `scikit-learn`, `numpy`.

## Rules

1. **Baseline first.** Before any real model, compute the dumbest reasonable baseline and put it in the same table:
   - Classification: majority class (`DummyClassifier(strategy="most_frequent")`) and stratified random.
   - Regression: predict the train mean or median.
   - Time series: last value (naive) and seasonal naive.
   - Ranking/retrieval: BM25 or random.
   If the model barely beats the baseline, say that plainly.

2. **Pick the metric for the problem, not the one that looks best.**
   - Imbalanced classes: never headline accuracy. Use PR-AUC, recall at a fixed precision, or F1 of the minority class, and show the confusion matrix.
   - Unequal error costs: report the cost-weighted metric or the chosen operating point.
   - Regression: MAE next to RMSE; MAPE only if no target is near zero.
   - Probabilities used downstream: report calibration (Brier score, reliability curve).

3. **Uncertainty is mandatory.** Report mean +/- std over CV folds, or a 95% bootstrap interval on the test set. When two models are scored on the same rows, compare them with a *paired* bootstrap on the difference. A difference whose interval includes 0 is "no clear difference", not a win. If rows are not independent (many rows per user/patient/document), resample whole groups.

4. **One test set, touched once.** Model selection, tuning, thresholds and early stopping use validation data or inner CV. Evaluate on test once, at the end. If you looked earlier, say so.

5. **Slice the results.** Report subgroups that matter (class, region, device, document length, new vs returning users). An average hides where the model fails.

6. **Match evaluation to deployment.** If production predicts the future, evaluate on a later time window. If production sees new users, hold out users, not rows. Run `leakage-hunter` before trusting any number.

7. **Seeds.** For anything stochastic (neural nets, tree ensembles, CV splits), report mean +/- std over at least 3 seeds (see `reproducible-runs`).

## The script

Save the final model's predictions on the test set to a CSV (`y_true`, and one prediction column per model), then:

```bash
# one model + automatic baseline row
python <skill-dir>/scripts/bootstrap_ci.py --data preds.csv --y-true y --pred model_a --metric f1
# two models: paired bootstrap on (B - A) with a verdict
python <skill-dir>/scripts/bootstrap_ci.py --data preds.csv --y-true y --pred model_a model_b --metric roc_auc
# repeated users/patients: resample groups
python <skill-dir>/scripts/bootstrap_ci.py --data preds.csv --y-true y --pred p --metric accuracy --group user_id
```

Metrics: `accuracy f1 precision recall roc_auc pr_auc mae rmse r2`. `roc_auc` and `pr_auc` need scores or probabilities; the others need hard predictions. For binary `f1/precision/recall`, the positive class defaults to the rarest class (override with `--pos-label`); with more than two classes they are macro-averaged. Pass `--baseline-value` with the TRAIN majority class or TRAIN mean so the baseline does not peek at test labels. The script prints a baseline row, 95% intervals, the paired difference, and warns when the best model's interval overlaps the baseline. Rows with missing or infinite values are dropped and counted. It stops with a clear error, instead of printing `nan`, when the inputs do not fit the metric: probabilities given to `accuracy`/`f1`, one class given to `roc_auc`, a multiclass label with `roc_auc` and no `--pos-label`, or non-numeric values for a regression metric.

## Output contract

Every results summary contains a table like this, generated from real numbers:

```
| Model     | Metric (primary) | 95% CI        | Secondary metric | Notes             |
|-----------|------------------|---------------|------------------|-------------------|
| Baseline  | <value>          | [<lo>, <hi>]  | ...              | majority class    |
| Candidate | <value>          | [<lo>, <hi>]  | ...              | tuned on val only |
```

Then one sentence each on: whether the improvement is real (does the paired interval exclude 0?), how big it is in practical terms, and the biggest remaining risk. Never write a number you did not compute.

## Works with

`leakage-hunter` runs first. `reproducible-runs` supplies the multi-seed results and the run record. `llm-eval` takes over for LLM applications. `ml-roast` checks that a report follows these rules.
