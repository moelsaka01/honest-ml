<div align="center">

# honest-ml

**Your AI agent's model is lying to you.**

Agent skills that stop Claude Code, Codex, and Cursor from shipping ML results that look great and mean nothing.

<!-- Replace with a real GIF: agent reports 0.99 AUC -> leakage-hunter catches it -> honest 0.71 -->
<!-- ![demo](assets/demo.gif) -->

`0.99 AUC` → **leakage found** → `0.71 AUC, 95% CI [0.68, 0.74], beats baseline by 0.21`

</div>

---

## The problem

Ask any coding agent to "train a model on this CSV" and you'll often get:

- `fit_transform` on the **whole dataset** before splitting
- shuffled K-fold on **time series**
- the same user in **train and test**
- a feature that **is** the label in disguise
- accuracy on a 95/5 imbalanced dataset, no baseline, one seed, no error bars

…and then a confident *"The model achieves 99% accuracy!"*

These are the same mistakes that sink Kaggle notebooks, papers, and production models. **honest-ml** teaches your agent to catch them automatically.

## The skills

| Skill | What it makes your agent do |
|---|---|
| 🔍 **[leakage-hunter](skills/leakage-hunter/SKILL.md)** | Hunts train/test contamination, target, temporal, group, and preprocessing leakage before any metric is reported. Ships with a runnable checker. |
| 📏 **[honest-eval](skills/honest-eval/SKILL.md)** | Baseline first, the right metric, confidence intervals, paired comparisons, test set touched once. |
| 🧪 **[dataset-audit](skills/dataset-audit/SKILL.md)** | Profiles new data for label noise, missingness, duplicates, unit errors, and train/serve skew before modeling. |
| 🔁 **[reproducible-runs](skills/reproducible-runs/SKILL.md)** | Seeds, pinned envs, config files, data hashes, git SHA, one directory per run, multi-seed results. |
| 🤖 **[llm-eval](skills/llm-eval/SKILL.md)** | Real eval sets for prompts, RAG, and agents: programmatic graders, calibrated LLM-as-judge, flip reports, cost and latency. |
| 🔥 **[ml-roast](skills/ml-roast/SKILL.md)** | A blunt, prioritized review of ML code with fixes and a 0-10 trust score. |

Skills trigger on their own when the agent is doing ML work. You can also call them by name: *"roast my notebook"*, *"run leakage-hunter on this"*.

## Install

**Claude Code (plugin)**
```bash
/plugin marketplace add moelsaka01/honest-ml
/plugin install honest-ml@honest-ml
```

**Any agent (skills CLI)**
```bash
npx skills add moelsaka01/honest-ml
```

**Manual**
```bash
git clone https://github.com/moelsaka01/honest-ml
cp -r honest-ml/skills/* ~/.claude/skills/      # Claude Code
cp -r honest-ml/skills/* ~/.codex/skills/       # Codex
```

## Try it in 30 seconds

```bash
pip install pandas scikit-learn
python examples/make_leaky_data.py
python skills/leakage-hunter/scripts/leakage_check.py \
  --train examples/train.csv --test examples/test.csv \
  --target churned --time signup_date --group customer_id
```

The example dataset contains four planted leaks. The checker finds them. Then ask your agent to *"train a churn model on examples/"* with and without honest-ml installed and compare what it reports.

## Philosophy

1. **Assume leakage until proven otherwise.** Great numbers are a symptom.
2. **No metric without a baseline and an error bar.**
3. **The test set is sacred.** Touch it once.
4. **Say the uncomfortable thing.** "The model barely beats predicting the mean" is a valid result.

## Contributing

PRs welcome, especially new failure modes with a real-world example of how they burned someone. Each skill is a single `SKILL.md`. Keep them short, specific, and checkable.

## License

MIT
