#!/usr/bin/env python3
"""roast_scan.py - heuristic static scan for the ML mistakes that make results untrustworthy.

Reads .py files and .ipynb notebooks (a single file or a directory) and prints findings grouped by
severity with the exact line and a fix. It is a fast first pass: line/regex heuristics, no execution.
Treat every hit as "verify this", and still read the code; absence of findings is not a clean bill.

Usage:
  python roast_scan.py path/to/train.py
  python roast_scan.py notebooks/ --json

Severity: FATAL (results invalid), SERIOUS (probably inflated or fragile), MINOR (hygiene).
Exit codes: 0 = no FATAL finding, 1 = at least one FATAL finding, 2 = bad usage. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".ipynb_checkpoints", "site-packages"}
PREPROC = re.compile(r"(scal|imput|encod|pca|norm|select|smote|vectoriz|tfidf|poly|binner|sampler|ros\b|rus\b|"
                     r"transformer|kbest|variancethreshold)", re.IGNORECASE)
SPLIT_CALL = re.compile(r"(train_test_split\(|\bKFold\(|StratifiedKFold\(|ShuffleSplit\(|\.split\()")
SEEDISH = re.compile(r"(random_state|random_seed|\bseed\s*[=(]|manual_seed|set_seed|np\.random\.seed|random\.seed)")
TIMEISH = re.compile(r"(pd\.to_datetime|['\"]\w*(date|time|timestamp)\w*['\"]|\w+_at['\"]|\.dt\.)", re.IGNORECASE)
GROUPISH = re.compile(r"['\"](user|customer|patient|subject|session|device|account|member)_?id['\"]", re.IGNORECASE)
SCORE_CALL = re.compile(r"(accuracy_score|\.score\(|f1_score|roc_auc_score|r2_score|mean_squared_error)")
IMPORT_LINE = re.compile(r"\s*(import|from)\s")
BASELINE = re.compile(r"(Dummy(Classifier|Regressor)|baseline|most_frequent|majority|naive)", re.IGNORECASE)

FIXES = {
    "R1": "Split first, then fit on train only. Put the step inside sklearn.pipeline.Pipeline and pass the "
          "Pipeline to cross_val_score / GridSearchCV.",
    "R2": "Never fit on test data. Fit on X_train only; call .transform / .predict on X_test.",
    "R3": "Split by time: train on the past, test on the future (TimeSeriesSplit or a date cutoff). "
          "Do not shuffle time-ordered rows.",
    "R4": "Keep every row of one user/patient/device in the same split: GroupKFold / GroupShuffleSplit "
          "with groups=<id column>.",
    "R5": "Seed everything from one config value (random_state=seed for sklearn, np/torch/random seeds) "
          "and report mean +/- std over 3+ seeds.",
    "R6": "Early-stop and tune on a validation split or inner CV. Keep the test set untouched until the end.",
    "R7": "Report a baseline in the same table: DummyClassifier(strategy='most_frequent') / DummyRegressor, "
          "plus a metric suited to the class balance, with a bootstrap CI.",
    "R8": "Use dropna(subset=[...]) on the columns you need and print how many rows were dropped.",
    "R9": "Call model.eval() before inference/validation and wrap it in torch.no_grad() (or inference_mode()).",
    "R10": "CrossEntropyLoss expects raw logits. Only apply softmax to outputs you display or threshold.",
    "R11": "Save preprocessing and model together as one Pipeline so inference applies identical steps.",
    "R12": "Use shift(1) before rolling so a row never sees its own or future values: "
           "df[col].shift(1).rolling(w).mean().",
}
LEVEL_ORDER = {"FATAL": 0, "SERIOUS": 1, "MINOR": 2}


def load_lines(path: Path) -> list[tuple[str, str]]:
    """Return [(location, code_line)], notebooks flattened cell by cell."""
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"cannot read notebook {path}: {exc}") from exc
        out, cell_no = [], 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            cell_no += 1
            src = cell.get("source", [])
            text = "".join(src) if isinstance(src, list) else str(src)
            for i, line in enumerate(text.splitlines(), 1):
                out.append((f"{path.name} [code cell {cell_no}] line {i}", line))
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return [(f"{path.name}:{i}", line) for i, line in enumerate(text.splitlines(), 1)]


def strip_comment(line: str) -> str:
    # crude but safe enough: drop everything after an unquoted-looking '#'
    in_s = None
    for i, ch in enumerate(line):
        if ch in "\"'":
            in_s = None if in_s == ch else (in_s or ch)
        elif ch == "#" and in_s is None:
            return line[:i]
    return line


def scan_lines(lines: list[tuple[str, str]]) -> list[dict]:
    code = [(loc, strip_comment(l)) for loc, l in lines]
    full = "\n".join(l for _, l in code)
    findings: list[dict] = []

    def add(rule, level, loc, msg):
        findings.append({"rule": rule, "level": level, "where": loc, "problem": msg, "fix": FIXES[rule]})

    split_idx = next((i for i, (_, l) in enumerate(code) if SPLIT_CALL.search(l) and "train_test_split" in l), None)
    if split_idx is None:
        split_idx = next((i for i, (_, l) in enumerate(code) if SPLIT_CALL.search(l)), None)

    first_split_loc = code[split_idx][0] if split_idx is not None else None
    uses_ml = bool(re.search(r"(sklearn|xgboost|lightgbm|catboost|torch|tensorflow|keras)", full))
    if not uses_ml:
        return findings

    flagged_r1 = False
    r2_locs: set[str] = set()
    for i, (loc, line) in enumerate(code):
        # R1: preprocessing fitted on data before the split / on a non-train variable
        for m in re.finditer(r"(\w+)(?:\([^)]*\))?\.(fit_transform|fit_resample|fit)\(\s*([^)]*)\)", line):
            obj, method, arg = m.group(1), m.group(2), m.group(3)
            first_arg = arg.split(",")[0].strip()
            if not PREPROC.search(obj) and method != "fit_resample":
                continue
            if re.search(r"(train|\btr\b|_tr\b|^tr_)", first_arg, re.IGNORECASE):
                continue
            if re.search(r"test|_te\b|\bte_|valid|_val\b", first_arg, re.IGNORECASE):
                r2_locs.add(loc)
                add("R2", "FATAL", loc, f"`{obj}.{method}({first_arg})` fits a preprocessing step on test/validation data")
            elif (split_idx is None or i < split_idx) and not flagged_r1:
                flagged_r1 = True
                add("R1", "FATAL", loc, f"`{obj}.{method}({first_arg})` fits a preprocessing/sampling step on the "
                                        "full dataset" + (" before the split" if split_idx is not None else
                                                            " (no train/test split found in this file)"))
        # R2: models or searches fitted on test
        if loc not in r2_locs and re.search(r"\.fit\(\s*(X_?te(st)?|x_?te(st)?|test\w*)\b", line):
            add("R2", "FATAL", loc, "`.fit(...)` is called with test data")
        # R6: early stopping / eval_set on test
        if re.search(r"eval_set\s*=\s*\[\(\s*X_?te(st)?\b", line, re.IGNORECASE) or \
                re.search(r"validation_data\s*=\s*\(\s*X_?te(st)?\b", line, re.IGNORECASE):
            add("R6", "SERIOUS", loc, "early stopping / validation uses the test set")
        # R8: dropna on the whole frame
        if re.search(r"\.dropna\(\s*\)", line) or re.search(r"\.dropna\(\s*(axis|how|inplace)\b", line):
            add("R8", "MINOR", loc, "`dropna()` without `subset=` silently drops rows that have any NaN")

    # R3: random split on time-ordered data
    if split_idx is not None and TIMEISH.search(full) and "TimeSeriesSplit" not in full \
            and "shuffle=False" not in full:
        add("R3", "SERIOUS", first_split_loc, "random/shuffled split, but the file references date/time columns")
    # R4: group-like ids but no group-aware splitter
    if split_idx is not None and GROUPISH.search(full) and not re.search(r"Group(KFold|ShuffleSplit)|StratifiedGroupKFold|groups\s*=", full):
        add("R4", "SERIOUS", first_split_loc, "id-like column (user/customer/patient/...) present but no group-aware split")
    # R5: no seed anywhere
    if (split_idx is not None or re.search(r"(Classifier|Regressor)\(", full)) and not SEEDISH.search(full):
        add("R5", "SERIOUS", first_split_loc or code[0][0], "no seed / random_state anywhere; results change on every run")
    # R7: scores but no baseline
    score_loc = None
    for cloc, ctext in code:
        if IMPORT_LINE.match(ctext):
            continue
        if SCORE_CALL.search(ctext):
            score_loc = cloc
            break
    if score_loc and not BASELINE.search(full):
        add("R7", "SERIOUS", score_loc, "metric is computed but no baseline model is compared")
    # R9: torch without eval()
    if re.search(r"\bimport torch\b|from torch\b", full):
        if not re.search(r"\.eval\(\)", full):
            add("R9", "SERIOUS", code[0][0], "torch model never switched to `.eval()`; dropout/batchnorm stay in train mode")
        elif not re.search(r"no_grad|inference_mode", full):
            loc = next((loc for loc, l in code if re.search(r"\.eval\(\)", l)), code[0][0])
            add("R9", "MINOR", loc, "`.eval()` used but no `torch.no_grad()` / `inference_mode()`: wasteful and leaks graph memory")
    # R10: softmax + cross entropy
    if re.search(r"softmax", full, re.IGNORECASE) and re.search(r"CrossEntropyLoss|F\.cross_entropy|cross_entropy\(", full):
        loc = next(loc for loc, l in code if re.search(r"softmax", l, re.IGNORECASE))
        add("R10", "MINOR", loc, "softmax and a cross-entropy loss both appear: verify softmax is not applied to the logits fed to the loss")
    # R11: preprocessing objects but no Pipeline
    if re.search(r"(StandardScaler|MinMaxScaler|SimpleImputer|OneHotEncoder|OrdinalEncoder|PCA)\(", full) \
            and not re.search(r"(Pipeline|make_pipeline|ColumnTransformer|joblib\.dump|pickle\.dump)", full):
        loc = next(loc for loc, l in code if re.search(r"(StandardScaler|MinMaxScaler|SimpleImputer|OneHotEncoder|OrdinalEncoder|PCA)\(", l))
        add("R11", "MINOR", loc, "preprocessing is applied step by step with no Pipeline or saved artifact")
    # R12: rolling without shift
    if re.search(r"\.rolling\(", full) and not re.search(r"\.shift\(", full):
        loc = next(loc for loc, l in code if ".rolling(" in l)
        add("R12", "SERIOUS", loc, "rolling window without `shift(1)` includes the current row (target leakage if it feeds the label)")

    # de-duplicate identical findings
    seen, unique = set(), []
    for f in findings:
        key = (f["rule"], f["where"], f["problem"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def trust_score(findings: list[dict]) -> float:
    s = 10.0 - 4.0 * sum(f["level"] == "FATAL" for f in findings) \
        - 1.5 * sum(f["level"] == "SERIOUS" for f in findings) \
        - 0.3 * sum(f["level"] == "MINOR" for f in findings)
    return max(0.0, round(s, 1))


def collect(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix in (".py", ".ipynb")
                  and not any(part in SKIP_DIRS for part in p.parts))


def run(argv=None) -> tuple[int, list[dict]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    root = Path(args.path)
    if not root.exists():
        print(f"error: path not found: {args.path}", file=sys.stderr)
        raise SystemExit(2)
    files = collect(root)
    if not files:
        print(f"error: no .py or .ipynb files under {args.path}", file=sys.stderr)
        raise SystemExit(2)

    findings: list[dict] = []
    for f in files:
        try:
            findings += scan_lines(load_lines(f))
        except ValueError as exc:
            print(f"warning: {exc}", file=sys.stderr)
    findings.sort(key=lambda f: (LEVEL_ORDER[f["level"]], f["where"]))
    score = trust_score(findings)

    if args.json:
        print(json.dumps({"scan_score": score, "findings": findings}, indent=2))
    else:
        print(f"Scanned {len(files)} file(s).")
        for level in ("FATAL", "SERIOUS", "MINOR"):
            group = [f for f in findings if f["level"] == level]
            if not group:
                continue
            print(f"\n{level} ({len(group)})")
            for f in group:
                print(f"  {f['where']}  [{f['rule']}]\n    problem: {f['problem']}\n    fix:     {f['fix']}")
        if not findings:
            print("\nNo findings from the heuristic scan. Still read the code: this scanner cannot see everything.")
        print(f"\nHeuristic scan score: {score}/10 (a starting point; your trust score must also weigh "
              "evaluation design and data issues the scanner cannot see)")
    return (1 if any(f["level"] == "FATAL" for f in findings) else 0), findings


if __name__ == "__main__":
    sys.exit(run()[0])
