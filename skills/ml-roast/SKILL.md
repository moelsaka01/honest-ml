---
name: ml-roast
description: Use when asked to review, audit, critique, or "roast" machine learning code, a notebook, a Kaggle solution, a paper's experiment section, or an ML pull request. Produces a blunt, prioritized review of everything that makes the results untrustworthy. Trigger on "review my model", "roast my notebook", "is this right", "check my ML code", or "why doesn't this generalize".
---

# ML Roast

Be the reviewer who rejects the paper - kindly, specifically, and with fixes.

## Review in this order (most damaging first)

1. **Leakage** - anything from `leakage-hunter`: preprocessing before split, shuffled time series, shared groups, target-derived features, tuning on test.
2. **Evaluation** - anything from `honest-eval`: no baseline, wrong metric for imbalance, no error bars, single seed, cherry-picked results.
3. **Data** - unexamined duplicates, label noise, train/serve skew, silent row drops (`dropna()` on the whole frame), unhandled categories at inference.
4. **Correctness bugs** - shape/broadcast mistakes, wrong axis, labels misaligned after shuffle or merge, `model.eval()` / `torch.no_grad()` missing at inference, dropout/batchnorm in the wrong mode, softmax applied twice, loss on logits vs probabilities mixed up, class index order mismatch.
5. **Reproducibility** - anything from `reproducible-runs`.
6. **Engineering** - copy-pasted cells, magic numbers, no tests for feature functions, no saved pipeline (preprocessor + model together).

## Format

Group findings by severity:

- **Fatal** - results are invalid until fixed.
- **Serious** - results are probably inflated or fragile.
- **Minor** - hygiene and maintainability.

For each finding give: the file/cell and line, what's wrong in one sentence, why it matters, and the exact fix (code). End with a **Trust score from 0-10** for the reported results and the single most important fix. Do not pad with praise; one line on what's done well is enough.
