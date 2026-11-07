---
name: llm-eval
description: Use when building, changing, or comparing an LLM application - a prompt, RAG pipeline, agent, model swap, or fine-tune - and before claiming it works or is better. Replaces "it looks good on three examples" with a real eval set, programmatic and judged grading, retrieval checks, flip reports and cost/latency. Includes a provider-agnostic grader and templates. For classic ML models use honest-eval.
---

# LLM Eval

Three hand-picked examples are a demo, not an eval.

`<skill-dir>` means the folder that contains this SKILL.md. `scripts/run_eval.py` needs only `numpy` and does not call any model API: you produce the outputs with your own client, it grades them.

## 1. Build the eval set first

- Start with **30-50 cases** from real or realistic inputs: typical cases, edge cases, adversarial/off-topic inputs, and every bug you have already fixed (regression cases). Fewer than 30 gives intervals too wide to trust.
- Store them as data, one JSON object per line (`evals/cases.jsonl`, see `<skill-dir>/templates/cases.example.jsonl`): `id`, `input`, and either `expected` or `checks`, optionally `tags`.
- Keep a held-out slice (tag it `heldout`) that you never look at while iterating on prompts.

## 2. Grade with the cheapest reliable method

In order of preference:
1. **Programmatic checks** in `checks`: `contains`, `not_contains`, `equals`, `regex`, `json_valid`, `json_keys`, `max_chars`, `numeric` (with tolerance). Use these for JSON validity, schema, required facts, numbers, forbidden strings, refusals.
2. **Reference comparison**: `expected` alone means "output contains it" (case-insensitive).
3. **LLM-as-judge** only for fuzzy qualities. Use a specific rubric with pass/fail or 1-5, ask for the reason *before* the score, use a different (ideally stronger) model than the one under test, and **hand-check 10-20 judgments** against your own to confirm agreement. Store the verdict on the output row (`"judge_pass": true`) and grade it with `{"type": "field_true", "field": "judge_pass"}`.

## 3. RAG: evaluate retrieval and generation separately

- Retrieval: store the retrieved ids on each output row (`"retrieved_ids": ["d3", "d7"]`) and add `{"type": "field_contains", "field": "retrieved_ids", "value": "<relevant id>"}` to get hit@k. Compute MRR from the same data if needed.
- Generation: faithfulness (claims supported by the retrieved context) and answer correctness.
- A good answer from bad retrieval is luck.

## 4. Run and grade

Run the system over every case, write one line per run to `outputs.jsonl` (`{"id", "output", "run", "cost", "latency_s"}`, see `<skill-dir>/templates/outputs.example.jsonl`), then:

```bash
python <skill-dir>/scripts/run_eval.py --cases evals/cases.jsonl --outputs out_new.jsonl
python <skill-dir>/scripts/run_eval.py --cases evals/cases.jsonl --outputs out_new.jsonl --baseline out_old.jsonl
```

It reports pass rate with a 95% bootstrap interval, average cost, p50 latency, failing cases with the reason, a per-tag breakdown, and with `--baseline` the paired difference, **cases that flipped to fail / to pass**, and a verdict. It warns about ungraded cases, missing outputs and unknown ids, and exits `2` on malformed input.

## 5. Compare changes properly

- Run old and new on the **same** cases. Read the flipped cases, not only the average.
- Outputs vary: run each case 3+ times at the temperature production uses (several lines with the same `id`; a case's score is the share of runs that pass).
- Weigh cost and latency. A 2% gain at 4x cost is a trade-off, not a win.
- Keep the eval set fixed while comparing; add new cases only between comparisons, and re-run both versions when you do.
- For agents, also check the trajectory: right tool, right arguments, no repeated loops, and a step/cost budget.

## Output contract

```
Eval: <name>, <n> graded cases, <runs> run(s) each
| Version | Pass rate | 95% CI | Avg cost | p50 latency |
Flipped to fail: <ids>    Flipped to pass: <ids>
Verdict: <ship | don't ship | needs more cases>, because <reason>
```

## Works with

`leakage-hunter` item 7 covers benchmark contamination and test items leaking into prompts. `honest-eval` covers classic ML metrics. `ml-roast` flags LLM changes shipped on "looks good" evidence.
