---
name: llm-eval
description: Use when building, changing, or comparing LLM applications - prompts, RAG pipelines, agents, model swaps, or fine-tunes. Replaces "it looks good on three examples" with a real eval set, scored graders, and regression checks. Trigger on "prompt", "RAG", "agent", "switch models", "is this better", "eval", "LLM-as-judge", or "vibes".
---

# LLM Eval

Three hand-picked examples are a demo, not an eval.

## Build the eval set first

1. **At least 30-50 cases** to start, drawn from real or realistic inputs. Include: typical cases, edge cases, adversarial/off-topic inputs, and every bug you've already fixed (regression cases).
2. Store them as data (`evals/cases.jsonl`), each with `id`, `input`, and either an `expected` answer or a `rubric`.
3. Keep a held-out slice you never look at while iterating on prompts.

## Grade with the cheapest reliable method

In order of preference:
1. **Exact / programmatic checks**: JSON validity, schema match, required fields, regex, numeric tolerance, citation present, tool called with the right args.
2. **Reference comparison**: does the answer contain the key facts from `expected`?
3. **LLM-as-judge** only for fuzzy qualities. Use a specific rubric with a 1-5 or pass/fail scale, ask for the reason before the score, use a different (ideally stronger) model than the one being tested, and spot-check 10-20 judgments by hand to confirm the judge agrees with a human.

## RAG specifics

Evaluate retrieval and generation separately:
- Retrieval: recall@k / hit@k and MRR against labeled relevant chunks.
- Generation: faithfulness (claims supported by retrieved context) and answer correctness.
A good answer from bad retrieval is luck.

## Compare changes properly

- Run old and new versions on the **same** cases; report pass rate for each and list cases that flipped (pass->fail and fail->pass).
- LLM outputs vary: run each case 3+ times at the production temperature, or at temperature 0 if that is what production uses, and report the spread.
- Report cost (tokens/$) and latency alongside quality. A 2% gain at 4x cost is a trade-off, not a win.

## Output contract

```
Eval: <name>, <n> cases, <runs> runs each
| Version | Pass rate | 95% CI | Avg cost | p50 latency |
Flipped to fail: <ids>    Flipped to pass: <ids>
Verdict: <ship | don't ship | needs more cases>, because <reason>
```
