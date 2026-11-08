#!/usr/bin/env python3
"""run_eval.py - grade LLM outputs against an eval set and compare two versions.

Provider-agnostic: YOU produce the outputs (run your prompt / RAG pipeline / agent over the cases and
save one JSON line per run); this script grades them, adds bootstrap intervals, and lists flipped cases.

  cases.jsonl     {"id": "c1", "input": "...", "expected": "Paris", "checks": [...], "tags": ["geo"]}
  outputs.jsonl   {"id": "c1", "output": "...", "run": 0, "cost": 0.0012, "latency_s": 1.4}

Checks (all must pass; default when only "expected" is given: contains it, case-insensitive):
  {"type": "contains", "value": "x", "case": false}      {"type": "not_contains", "value": "x"}
  {"type": "equals", "value": "x"}                        {"type": "regex", "pattern": "^\\d+$"}
  {"type": "json_valid"}                                  {"type": "json_keys", "keys": ["a", "b"]}
  {"type": "max_chars", "n": 500}                         {"type": "numeric", "value": 3.14, "tol": 0.01}
  {"type": "field_true", "field": "judge_pass"}           (reads an extra field of the outputs row,
  {"type": "field_contains", "field": "retrieved_ids", "value": "doc7"}   e.g. a stored judge verdict / RAG hits)

Usage:
  python run_eval.py --cases evals/cases.jsonl --outputs out_new.jsonl
  python run_eval.py --cases evals/cases.jsonl --outputs out_new.jsonl --baseline out_old.jsonl

Negative controls (an eval that cannot fail cannot meaningfully pass):
  --known-bad bad.jsonl   outputs you know are WRONG, same format as --outputs. Every case they cover must
                          FAIL; a case that passes a deliberately wrong output has a check that cannot fail.
  --junk-check            also grade built-in junk (empty, "N/A", "I don't know.", {}, null, lorem ipsum).
                          Cases that pass junk are listed as a WARNING (some cases may rightly accept it).

Exit codes: 0 ok, 1 a known-bad output passed, 2 bad input.
A case with several runs counts as the fraction of runs that pass.
Requires: numpy
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

import numpy as np


def die(message: str):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    die(f"{path} line {n}: invalid JSON ({exc.msg})")
    except FileNotFoundError:
        die(f"file not found: {path}")
    return rows


# ---------------------------------------------------------------------- graders
def _text(output) -> str:
    return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)


def run_check(check: dict, output, row: dict | None = None) -> tuple[bool, str]:
    kind = check.get("type")
    row = row or {}
    text = _text(output)
    if kind in ("contains", "not_contains"):
        value = str(check["value"])
        case = check.get("case", False)
        hit = (value in text) if case else (value.lower() in text.lower())
        ok = hit if kind == "contains" else not hit
        return ok, f"{kind} {value!r}"
    if kind == "equals":
        return text.strip() == str(check["value"]).strip(), f"equals {check['value']!r}"
    if kind == "regex":
        return re.search(check["pattern"], text, re.MULTILINE) is not None, f"regex {check['pattern']!r}"
    if kind == "json_valid":
        try:
            json.loads(output) if isinstance(output, str) else None
            return True, "json_valid"
        except json.JSONDecodeError:
            return False, "json_valid"
    if kind == "json_keys":
        try:
            obj = json.loads(output) if isinstance(output, str) else output
            return isinstance(obj, dict) and all(k in obj for k in check["keys"]), f"json_keys {check['keys']}"
        except json.JSONDecodeError:
            return False, f"json_keys {check['keys']}"
    if kind == "field_true":  # e.g. a judge verdict you stored on the output row: {"judge_pass": true}
        return bool(row.get(check["field"])), f"field_true {check['field']}"
    if kind == "field_contains":  # e.g. RAG retrieval: {"retrieved_ids": ["d3", "d7"]}
        value = row.get(check["field"])
        return isinstance(value, (list, tuple, set)) and check["value"] in value,             f"field_contains {check['field']} {check['value']!r}"
    if kind == "max_chars":
        return len(text) <= int(check["n"]), f"max_chars {check['n']}"
    if kind == "numeric":
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        tol = float(check.get("tol", 0.0))
        return any(abs(float(x) - float(check["value"])) <= tol for x in nums), f"numeric {check['value']}+/-{tol}"
    return False, f"unknown check type {kind!r}"


def checks_for(case: dict) -> list[dict]:
    if case.get("checks"):
        return case["checks"]
    if case.get("expected") not in (None, ""):
        return [{"type": "contains", "value": case["expected"]}]
    return []


# --------------------------------------------------------------------- scoring
def score(cases: list[dict], outputs: list[dict]) -> dict:
    """Per-case pass rate (fraction of runs passing), plus cost / latency."""
    by_id = defaultdict(list)
    for o in outputs:
        if "id" not in o or "output" not in o:
            die("every outputs line needs 'id' and 'output'.")
        by_id[str(o["id"])].append(o)
    case_ids = [str(c.get("id")) for c in cases]
    if len(set(case_ids)) != len(case_ids):
        die("duplicate case ids in the cases file.")
    unknown = sorted(set(by_id) - set(case_ids))

    rates, failures, ungraded, missing = {}, {}, [], []
    costs, lats = [], []
    for c in cases:
        cid = str(c["id"])
        chk = checks_for(c)
        if not chk:
            ungraded.append(cid)
            continue
        runs = by_id.get(cid, [])
        if not runs:
            missing.append(cid)
            rates[cid] = 0.0
            failures[cid] = ["no output"]
            continue
        passed = 0
        why = []
        for r in runs:
            results = [run_check(k, r["output"], r) for k in chk]
            if all(ok for ok, _ in results):
                passed += 1
            else:
                why += [d for ok, d in results if not ok]
            if r.get("cost") is not None:
                costs.append(float(r["cost"]))
            if r.get("latency_s") is not None:
                lats.append(float(r["latency_s"]))
        rates[cid] = passed / len(runs)
        if passed < len(runs):
            failures[cid] = sorted(set(why))
    return {"rates": rates, "failures": failures, "ungraded": ungraded, "missing": missing,
            "unknown_outputs": unknown, "costs": costs, "latencies": lats,
            "runs_per_case": {cid: len(by_id.get(cid, [])) for cid in rates}}


JUNK_OUTPUTS = ["", "N/A", "I don't know.", "{}", "null", "lorem ipsum dolor sit amet"]


def negative_control(cases: list[dict], graded_ids: list[str], bad_rows: list[dict]) -> dict:
    """Grade outputs that are known to be wrong. Any graded case they pass has a check that cannot fail."""
    s = score(cases, bad_rows)
    covered = sorted(i for i in s["rates"] if i not in s["missing"] and i in set(graded_ids))
    return {"covered": covered,
            "cannot_fail": sorted(i for i in covered if s["rates"][i] > 0),
            "uncovered": sorted(set(graded_ids) - set(covered)),
            "unknown": s["unknown_outputs"]}


def junk_rows(graded_ids: list[str]) -> list[dict]:
    return [{"id": cid, "output": junk, "run": k} for cid in graded_ids for k, junk in enumerate(JUNK_OUTPUTS)]


def boot_ci(values: np.ndarray, n_boot=2000, seed=0) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot=2000, seed=0):
    d = b - a
    lo, hi = boot_ci(d, n_boot, seed)
    return float(d.mean()), lo, hi


def summarize(name: str, s: dict) -> dict:
    ids = sorted(s["rates"])
    vals = np.array([s["rates"][i] for i in ids], dtype=float)
    lo, hi = boot_ci(vals)
    return {"name": name, "ids": ids, "vals": vals, "mean": float(vals.mean()) if len(vals) else float("nan"),
            "ci": (lo, hi), "cost": float(np.mean(s["costs"])) if s["costs"] else None,
            "p50": float(np.median(s["latencies"])) if s["latencies"] else None}


def fmt(x, spec=".3f", none="-"):
    return none if x is None else format(x, spec)


def run(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", required=True)
    p.add_argument("--outputs", required=True)
    p.add_argument("--baseline", help="outputs of the previous version, graded on the same cases")
    p.add_argument("--name", default="candidate")
    p.add_argument("--baseline-name", default="baseline")
    p.add_argument("--known-bad", help="outputs known to be WRONG (same format as --outputs); each covered case must fail")
    p.add_argument("--junk-check", action="store_true",
                   help="also grade built-in junk outputs and warn about cases that pass them")
    args = p.parse_args(argv)

    cases = read_jsonl(args.cases)
    if not cases:
        die("cases file is empty.")
    for c in cases:
        if "id" not in c:
            die("every case needs an 'id'.")
    new = score(cases, read_jsonl(args.outputs))
    if not new["rates"]:
        die("no gradable cases: each case needs 'checks' or 'expected'.")
    old = score(cases, read_jsonl(args.baseline)) if args.baseline else None

    rows = [summarize(args.name, new)] + ([summarize(args.baseline_name, old)] if old else [])
    n = len(new["rates"])
    runs = max(new["runs_per_case"].values())
    print(f"Eval: {args.cases}, {n} graded cases, up to {runs} run(s) each")
    if n < 30:
        print("WARNING: fewer than 30 graded cases; the interval is wide and the result fragile. Add cases.")
    if new["ungraded"]:
        print(f"WARNING: {len(new['ungraded'])} cases have no checks/expected and were NOT graded: "
              f"{', '.join(new['ungraded'][:10])}")
    if new["missing"]:
        print(f"WARNING: {len(new['missing'])} cases have no output (counted as fail): {', '.join(new['missing'][:10])}")
    if new["unknown_outputs"]:
        print(f"WARNING: outputs for unknown case ids ignored: {', '.join(new['unknown_outputs'][:10])}")
    print(f"\n{'version':<14} {'pass rate':>9}  {'95% CI':<16} {'avg cost':>10} {'p50 latency':>12}")
    for r in rows:
        ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        print(f"{r['name']:<14} {r['mean']:>9.3f}  {ci:<16} {fmt(r['cost'], '.5f'):>10} {fmt(r['p50'], '.2f'):>12}")

    out: dict = {"new": rows[0]["mean"], "verdict": None}
    if old:
        # compare on cases graded in BOTH versions
        common = sorted(set(new["rates"]) & set(old["rates"]))
        a = np.array([old["rates"][i] for i in common])
        b = np.array([new["rates"][i] for i in common])
        to_fail = [i for i, x, y in zip(common, a, b) if x >= 0.5 > y]
        to_pass = [i for i, x, y in zip(common, a, b) if x < 0.5 <= y]
        d, lo, hi = paired_diff_ci(a, b)
        print(f"\nPaired difference ({args.name} - {args.baseline_name}): {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"Flipped to fail: {', '.join(to_fail) or 'none'}")
        print(f"Flipped to pass: {', '.join(to_pass) or 'none'}")
        if n < 30:
            verdict = "needs more cases"
        elif lo > 0:
            verdict = "candidate is better"
        elif hi < 0:
            verdict = "candidate regressed; don't ship"
        else:
            verdict = "no clear difference"
        print(f"Verdict: {verdict}  (also weigh cost and latency)")
        out.update(diff=(d, lo, hi), flipped_to_fail=to_fail, flipped_to_pass=to_pass, verdict=verdict)

    if new["failures"]:
        print("\nFailing cases (first 10):")
        for cid, why in list(sorted(new["failures"].items()))[:10]:
            print(f"  {cid}: {'; '.join(why)}")

    tags = defaultdict(list)
    for c in cases:
        for t in c.get("tags", []) or []:
            if str(c["id"]) in new["rates"]:
                tags[t].append(new["rates"][str(c["id"])])
    if tags:
        print("\nBy tag:")
        for t, v in sorted(tags.items()):
            print(f"  {t:<20} {np.mean(v):.3f}  (n={len(v)})")

    graded = sorted(new["rates"])
    out["cannot_fail"], out["junk_passes"] = [], []
    if args.known_bad:
        nc = negative_control(cases, graded, read_jsonl(args.known_bad))
        out["cannot_fail"] = nc["cannot_fail"]
        print("\nNegative control (known-bad outputs, every one must fail):")
        if nc["unknown"]:
            print(f"  WARNING: known-bad outputs for unknown case ids ignored: {', '.join(nc['unknown'][:10])}")
        if not nc["covered"]:
            print("  WARNING: the known-bad file covers none of the graded cases, so nothing was controlled.")
        elif nc["cannot_fail"]:
            print(f"  FAIL: {len(nc['cannot_fail'])} of {len(nc['covered'])} covered cases PASSED a deliberately "
                  f"wrong output, so their checks cannot fail: {', '.join(nc['cannot_fail'][:10])}")
            print("  Strengthen those checks (add a required fact, a number, a regex) before trusting their pass rate.")
        else:
            print(f"  OK: all {len(nc['covered'])} covered cases failed on their wrong output.")
        if nc["covered"] and nc["uncovered"]:
            print(f"  {len(nc['uncovered'])} graded cases have no known-bad output (not controlled): "
                  f"{', '.join(nc['uncovered'][:10])}")
    if args.junk_check:
        s = score(cases, junk_rows(graded))
        out["junk_passes"] = sorted(i for i in graded if s["rates"].get(i, 0) > 0)
        print("\nJunk check (empty, 'N/A', \"I don't know.\", {}, null, lorem ipsum):")
        if out["junk_passes"]:
            print(f"  WARNING: {len(out['junk_passes'])} of {len(graded)} cases pass at least one junk output: "
                  f"{', '.join(out['junk_passes'][:10])}")
            print("  Intended for cases that should accept a refusal or an empty answer; otherwise the check is "
                  "too weak.")
        else:
            print(f"  OK: no junk output passes any of the {len(graded)} graded cases.")
    return out


def main():
    out = run()
    sys.exit(1 if out["cannot_fail"] else 0)


if __name__ == "__main__":
    main()
