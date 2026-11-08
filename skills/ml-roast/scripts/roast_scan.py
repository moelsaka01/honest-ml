#!/usr/bin/env python3
"""roast_scan.py - static scan for the ML mistakes that make results untrustworthy.

Reads .py files and .ipynb notebooks (one file or a directory), parses them with Python's `ast`
module and tracks which variables hold TRAIN, TEST or FULL data (following train_test_split outputs,
name conventions such as X_train / y_test / train_idx, and assignments), then reports findings with
the exact line and a fix. Strings, comments and docstrings never trigger a rule.

Usage:
  python roast_scan.py path/to/train.py
  python roast_scan.py notebooks/ --json

Severity: FATAL (results invalid), SERIOUS (probably inflated or fragile), MINOR (hygiene).
Exit codes: 0 = no FATAL finding, 1 = at least one FATAL finding, 2 = bad usage.
Files that cannot be parsed are reported (never silently skipped). Standard library only.

Rules
  R1  FATAL    preprocessing / resampling fitted on the full data (before or outside the split)
  R2  FATAL    anything fitted on test or validation data
  R3  SERIOUS  random split although the code handles dates / times
  R4  SERIOUS  id-like columns (user, patient, ...) but no group-aware split
  R5  SERIOUS  stochastic code with no seed
  R6  SERIOUS  early stopping / validation_data / eval_set uses the test set
  R7  SERIOUS  metric computed but no baseline model in the code
  R8  MINOR    dropna() without subset
  R9  SERIOUS  torch training code that never calls .eval()   (MINOR: no no_grad / inference_mode)
  R10 SERIOUS  softmax inside a model that is trained with CrossEntropyLoss (double softmax)
  R11 MINOR    preprocessing with no Pipeline / ColumnTransformer / saved artifact
  R12 SERIOUS  rolling / expanding window without shift (includes the current or future rows)
  R13 SERIOUS  loop that picks the best candidate using the test set
  R14 MINOR    pd.get_dummies before the split (train/serve columns can differ)
  R15 SERIOUS  imputation / standardisation with statistics of the full data before the split
  R16 SERIOUS  group target mean / target encoding built from the full data before the split
  R17 MINOR    drop_duplicates() on one split after the split (train/test overlap is not removed)
  R18 SERIOUS  a search's best_score_ (a validation score) reported as a test score
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".ipynb_checkpoints",
             "site-packages", "build", "dist"}
ML_MODULES = {"sklearn", "xgboost", "lightgbm", "catboost", "torch", "tensorflow", "keras", "imblearn",
              "category_encoders", "statsmodels", "transformers", "pytorch_lightning", "lightning"}

PREPROC = re.compile(
    r"(Scaler|Imputer|Encoder|PCA|SelectKBest|SelectPercentile|SelectFromModel|VarianceThreshold|RFE|RFECV|"
    r"SMOTE|ADASYN|RandomOverSampler|RandomUnderSampler|TomekLinks|Vectorizer|Tfidf|PolynomialFeatures|"
    r"KBinsDiscretizer|Normalizer|QuantileTransformer|PowerTransformer|Binarizer|TruncatedSVD|NMF|"
    r"FactorAnalysis|ICA|KNNImputer|IterativeImputer|FunctionTransformer)")
NOT_PREPROC = re.compile(r"(^Label(Encoder|Binarizer)$|MultiLabelBinarizer|label)", re.IGNORECASE)
PREPROC_NAME = re.compile(r"(scal|imput|encod|pca|norm|selector|smote|sampler|vectoriz|tfidf|poly|binner|"
                          r"transformer|svd)", re.IGNORECASE)
ESTIMATOR_STOCHASTIC = re.compile(
    r"^(RandomForest|ExtraTrees|GradientBoosting|HistGradientBoosting|MLP|SGD|KMeans|"
    r"MiniBatchKMeans|DecisionTree|ExtraTree|Bagging|AdaBoost|IsolationForest|RandomizedSearchCV|Perceptron|"
    r"PassiveAggressive|TSNE|GaussianMixture|RandomForestClassifier|RandomTreesEmbedding)")
SPLITTERS_SEEDED = {"train_test_split", "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit"}
SPLITTERS_SHUFFLE_OPT = {"KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold"}
RANDOM_SPLIT_FUNCS = SPLITTERS_SEEDED | SPLITTERS_SHUFFLE_OPT | {"cross_val_score", "cross_validate",
                                                                  "cross_val_predict", "GridSearchCV",
                                                                  "RandomizedSearchCV"}
METRIC_NAMES = {"accuracy_score", "f1_score", "roc_auc_score", "r2_score", "mean_squared_error",
                "mean_absolute_error", "precision_score", "recall_score", "average_precision_score",
                "log_loss", "balanced_accuracy_score", "matthews_corrcoef", "root_mean_squared_error"}
SEED_FUNCS = {"seed", "manual_seed", "set_seed", "seed_everything", "set_random_seed", "random_seed"}
SEED_KWARGS = {"random_state", "seed", "random_seed", "seed_value", "rng", "random_state_"}
TEST_TOKENS = {"test", "tests", "testing", "te", "holdout", "heldout"}
VAL_TOKENS = {"val", "valid", "validation", "vali", "dev"}
TRAIN_TOKENS = {"train", "training", "trn", "tr", "trainset"}
GLUED = re.compile(r"^[xy]s?(test|train|val|valid)s?$", re.IGNORECASE)   # Xtest, ytrain, ...


def tokens(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def name_kind(name: str) -> str | None:
    toks = set(tokens(name))
    glued = GLUED.match(name)
    if glued:
        toks.add(glued.group(1).lower())
    if toks & TEST_TOKENS:
        return "test"
    if toks & VAL_TOKENS:
        return "val"
    if toks & TRAIN_TOKENS:
        return "train"
    return None


TIME_STR = re.compile(r"(date|time|timestamp|datetime|_at$|^dt$)", re.IGNORECASE)
GROUP_STR = re.compile(r"^(user|customer|patient|subject|session|device|account|member|client|person|"
                       r"visitor|player|driver|store|hospital|household|family|speaker|author)_?(id|key|uuid)?$",
                       re.IGNORECASE)
GROUP_STR_STRICT = re.compile(r"^(user|customer|patient|subject|session|device|account|member|client|"
                              r"person|visitor|player|driver|store|hospital|household|family|speaker|"
                              r"author)[_ ]?(id|key|uuid)$", re.IGNORECASE)
LEVEL_ORDER = {"FATAL": 0, "SERIOUS": 1, "MINOR": 2}

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
    "R10": "CrossEntropyLoss expects raw logits. Remove softmax from forward() and apply it only to outputs "
           "you display or threshold.",
    "R11": "Save preprocessing and model together as one Pipeline so inference applies identical steps.",
    "R12": "Use shift(1) before rolling so a row never sees its own or future values: "
           "df[col].shift(1).rolling(w).mean().",
    "R13": "Pick the best candidate on a validation split or inner CV (GridSearchCV), then score the test set "
           "exactly once for the winner.",
    "R14": "Use OneHotEncoder(handle_unknown='ignore') inside a ColumnTransformer/Pipeline so the same columns "
           "exist at inference.",
    "R15": "Compute fill values / means / stds on the train split only (SimpleImputer / StandardScaler inside "
           "a Pipeline), then apply them to validation and test.",
    "R16": "Split first, then compute the per-group target statistic on the train rows only (out-of-fold, or "
           "sklearn TargetEncoder inside a Pipeline) and apply it to validation and test.",
    "R17": "Deduplicate the full data before splitting, and keep all rows of one entity in the same split, so a "
           "row cannot appear in both train and test.",
    "R18": "best_score_ is the cross-validation score that picked the model. Report the score of the refit "
           "model on the untouched test set (search.score(X_test, y_test)), once.",
}

TARGETISH = re.compile(r"^(y|target|label|labels|churn|churned|outcome|survived|default|fraud|converted|clicked)$",
                       re.IGNORECASE)
TARGET_VARS = re.compile(r"^(y|target|label|labels|y_all|y_full)$", re.IGNORECASE)
AGG_NAMES = {"mean", "median", "sum", "std", "var", "max", "min", "transform", "agg", "aggregate", "apply"}


def const_strings(node) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


class ParseError(Exception):
    pass


# ------------------------------------------------------------------ loading
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
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith("%%") and \
                    not re.match(r"%%(time|timeit|capture|prun|debug)\b", lines[0].lstrip()):
                lines = ["pass"] * len(lines)           # %%bash, %%sql, ... : not Python
            for i, line in enumerate(lines, 1):
                out.append((f"{path.name} [code cell {cell_no}] line {i}", _neutralise_magics(line)))
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return [(f"{path.name}:{i}", line) for i, line in enumerate(text.splitlines(), 1)]


def _neutralise_magics(line: str) -> str:
    """Notebook magics (%timeit, !pip, ?obj) are not Python; keep the line count by replacing them."""
    s = line.lstrip()
    if s.startswith(("%", "!", "?")) or (s.endswith("?") and " " not in s):
        return line[: len(line) - len(s)] + "pass"
    return line


# ---------------------------------------------------------------- AST helpers
def name_of(node) -> str | None:
    """Last identifier of a Name / Attribute chain (np.random.seed -> 'seed'); else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def call_name(call: ast.Call) -> str | None:
    return name_of(call.func)


def names_in(node) -> list[str]:
    return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]


def kw(call: ast.Call, key: str):
    for k in call.keywords:
        if k.arg == key:
            return k.value
    return None


def has_star_kwargs(call: ast.Call) -> bool:
    return any(k.arg is None for k in call.keywords)


def const_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def const_false(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def scope_nodes(scope):
    """All nodes of this scope, not descending into nested functions, lambdas or classes."""
    stack = list(reversed(list(ast.iter_child_nodes(scope))))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def data_names(node) -> list[str]:
    """Variable names that carry data in an expression. A plain function name (`_for_tests(...)`,
    `len(...)`) is not data, so it is skipped; method receivers (`X_test.copy()`) are kept."""
    out, stack = [], [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                stack.append(n.func)
            stack.extend(n.args)
            stack.extend(k.value for k in n.keywords)
        elif isinstance(n, ast.Name):
            out.append(n.id)
        else:
            stack.extend(ast.iter_child_nodes(n))
    return out


def store_names(target) -> list[str]:
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]


# ------------------------------------------------------------------- analysis
class Analyzer:
    def __init__(self, tree: ast.AST, locs: list[str]):
        self.tree, self.locs = tree, locs
        self.findings: list[dict] = []
        self._seen: set[tuple] = set()
        self.target_cols = self._target_columns()

    def _target_columns(self) -> set:
        """Column names that end up as the label (y = df['churned'], target = df.pop('y'), ...)."""
        cols: set = set()
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Assign):
                names = [x for t in n.targets for x in store_names(t)]
                if any(TARGET_VARS.match(nm) for nm in names):
                    cols.update(const_strings(n.value))
        return cols

    def is_target_col(self, col: str) -> bool:
        return col in self.target_cols or bool(TARGETISH.match(col))

    def groupby_target_agg(self, call: ast.Call) -> bool:
        """df.groupby(k)[target].mean() / .transform(...) / .agg({target: ...}): a per-group target statistic."""
        f = call.func
        if not isinstance(f, ast.Attribute) or f.attr not in AGG_NAMES:
            return False
        v = f.value
        if isinstance(v, ast.Subscript) and any(isinstance(c, ast.Call) and call_name(c) == "groupby"
                                                for c in ast.walk(v.value)):
            return any(self.is_target_col(c) for c in const_strings(v.slice))
        if f.attr in ("agg", "aggregate") and isinstance(v, ast.Call) and call_name(v) == "groupby":
            return any(self.is_target_col(c) for a in call.args if isinstance(a, ast.Dict)
                       for k in a.keys if k is not None for c in const_strings(k))
        return False

    def assign_rules(self, node: ast.Assign, tainted: set, split_line):
        before_split = split_line is not None and node.lineno < split_line
        has_agg = any(isinstance(c, ast.Call) and self.groupby_target_agg(c) for c in ast.walk(node.value))
        uses_enc = any(isinstance(c, ast.Name) and c.id in tainted for c in ast.walk(node.value))
        makes_column = any(isinstance(t, (ast.Subscript, ast.Attribute)) for t in node.targets)
        if has_agg and not makes_column:
            tainted.update(n for t in node.targets for n in store_names(t))   # enc = df.groupby(k)[y].mean()
        elif (has_agg or uses_enc) and makes_column and before_split:         # df['k_enc'] = ...: a feature
            self.add("R16", "SERIOUS", node, "a per-group statistic of the target becomes a feature using the "
                                             "data before the split (target leakage)")
        # R18: best_score_ is a validation score
        if any(isinstance(c, ast.Attribute) and c.attr == "best_score_" for c in ast.walk(node.value)):
            names = [n for t in node.targets for n in store_names(t)]
            if any(set(tokens(n)) & TEST_TOKENS for n in names):
                self.add("R18", "SERIOUS", node, "a validation score (`best_score_`) is stored as a test score")

    def loc(self, node) -> str:
        i = max(0, min(getattr(node, "lineno", 1) - 1, len(self.locs) - 1))
        return self.locs[i] if self.locs else "?"

    def add(self, rule: str, level: str, node, problem: str):
        key = (rule, self.loc(node), problem)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append({"rule": rule, "level": level, "where": self.loc(node),
                              "problem": problem, "fix": FIXES[rule]})

    # ---- gating --------------------------------------------------------------
    def imports_ml(self) -> bool:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] in ML_MODULES for a in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in ML_MODULES:
                    return True
        return False

    def imports(self, module: str) -> bool:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == module for a in node.names):
                return True
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == module:
                return True
        return False

    # ---- main ----------------------------------------------------------------
    def run(self) -> list[dict]:
        if not self.imports_ml():
            return []
        scopes = [self.tree] + [n for n in ast.walk(self.tree)
                                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for scope in scopes:
            self.analyze_scope(scope)
        self.file_rules()
        return self.findings

    # ---- data-flow: which variables hold train / test / full data -----------------
    def analyze_scope(self, scope):
        nodes = sorted(scope_nodes(scope), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
        env: dict[str, str] = {}
        sources: set[str] = set()
        ctors: dict[str, str] = {}       # variable -> constructor class name
        split_line: int | None = None

        def name_class(n: str) -> str | None:
            if n in env:
                return env[n]
            return name_kind(n)

        def classify(expr) -> str | None:
            classes = set()
            for n in data_names(expr):
                if n in sources:
                    classes.add("full")
                c = name_class(n)
                if c:
                    classes.add(c)
            if "test" in classes and classes & {"train", "val", "full"}:
                return "full"
            if "full" in classes:
                return "full"
            if "train" in classes:        # train, or train + validation (legitimate refit data)
                return "train"
            for c in ("test", "val"):
                if c in classes:
                    return c
            return None

        # pass 1: record splits and propagate classes through assignments, in source order
        for node in nodes:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                cn = call_name(node.value)
                if cn == "train_test_split":
                    outs = [n for t in node.targets for n in
                            ([e for e in t.elts] if isinstance(t, (ast.Tuple, ast.List)) else [t])]
                    out_names = [o.id if isinstance(o, ast.Name) else None for o in outs]
                    for a in node.value.args:
                        sources.update(names_in(a))
                    for i, nm in enumerate(out_names):
                        if nm:
                            env[nm] = name_kind(nm) or ("train" if i % 2 == 0 else "test")
                    split_line = node.lineno if split_line is None else min(split_line, node.lineno)
                    continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and getattr(node, "value", None) is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [n for t in targets for n in store_names(t)]
                value = node.value
                # remember what object each variable is (for preprocessor detection)
                if isinstance(value, ast.Call):
                    root = value
                    while isinstance(root, ast.Call) and isinstance(root.func, ast.Attribute) \
                            and isinstance(root.func.value, ast.Call) and root.func.attr in ("fit", "set_params"):
                        root = root.func.value
                    cls = call_name(root)
                    for nm in names:
                        if cls:
                            ctors[nm] = cls
                # do not propagate "data class" from estimators or fitted estimators
                is_estimator_value = isinstance(value, ast.Call) and (
                    call_name(value) in ("fit", "partial_fit", "set_params")
                    or (call_name(value) and re.match(r"^[A-Z]", call_name(value))
                        and not isinstance(value.func, ast.Attribute)))
                if is_estimator_value:
                    continue
                cls = classify(value)
                for nm in names:
                    byname = name_kind(nm)
                    if byname:
                        env[nm] = byname
                    elif cls:
                        env[nm] = cls
            elif isinstance(node, ast.For):
                pass

        for node in nodes:
            if isinstance(node, ast.Call) and call_name(node) in ("cross_val_score", "cross_validate",
                                                                  "cross_val_predict", "split") \
                    and (node.args or node.keywords):
                split_line = node.lineno if split_line is None else min(split_line, node.lineno)

        self.scope_rules(scope, nodes, classify, ctors, sources, split_line)

    def is_preprocessor(self, recv, ctors: dict[str, str]) -> bool:
        if isinstance(recv, ast.Call):
            cn = call_name(recv)
            if cn == "fit" or cn == "set_params":      # chained .fit(...).transform: look through
                return False
            return bool(cn and PREPROC.search(cn) and not NOT_PREPROC.search(cn))
        if isinstance(recv, ast.Name):
            cls = ctors.get(recv.id)
            if cls and cls[:1].isupper():
                return bool(PREPROC.search(cls) and not NOT_PREPROC.search(cls))
            return bool(PREPROC_NAME.search(recv.id) and not NOT_PREPROC.search(recv.id))
        return False

    def scope_rules(self, scope, nodes, classify, ctors, sources, split_line):
        tainted: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Assign):
                self.assign_rules(node, tainted, split_line)
            if isinstance(node, ast.Call):
                self.call_rules(node, classify, ctors, split_line)
            elif isinstance(node, ast.BinOp) and split_line is not None and node.lineno < split_line:
                if isinstance(node.op, ast.Div) and isinstance(node.left, ast.BinOp) \
                        and isinstance(node.left.op, ast.Sub) \
                        and any(isinstance(c, ast.Call) and call_name(c) == "mean" for c in ast.walk(node.left.right)) \
                        and any(isinstance(c, ast.Call) and call_name(c) == "std" for c in ast.walk(node.right)):
                    self.add("R15", "SERIOUS", node, "manual standardisation (x - mean) / std with statistics "
                                                     "of the data before the split")
            elif isinstance(node, (ast.For, ast.While)):
                self.selection_loop(node, classify)

    def call_rules(self, call, classify, ctors, split_line):
        cn = call_name(call)
        if isinstance(call.func, ast.Attribute) and call.func.attr in ("fit", "fit_transform", "fit_resample",
                                                                        "partial_fit"):
            attr = call.func.attr
            data_args = list(call.args) + [k.value for k in call.keywords if k.arg in ("X", "y")]
            classes = [classify(a) for a in data_args]
            prep = self.is_preprocessor(call.func.value, ctors) or attr == "fit_resample"
            if "test" in classes:
                what = "a preprocessing step" if prep else "a model / search"
                self.add("R2", "FATAL", call, f"`{self._src(call.func)}(...)` fits {what} on test data")
            elif prep and attr in ("fit", "fit_transform", "fit_resample"):
                first = classes[0] if classes else None
                if first == "full":
                    self.add("R1", "FATAL", call, f"`{self._src(call.func)}(...)` fits a preprocessing/sampling "
                                                  "step on the full data the split is made from")
                elif first is None and split_line is not None and call.lineno < split_line:
                    self.add("R1", "FATAL", call, f"`{self._src(call.func)}(...)` fits a preprocessing/sampling "
                                                  "step before the train/test split")
        for k in call.keywords:                                    # R6
            if k.arg in ("eval_set", "validation_data", "evals", "eval_dataset") and classify(k.value) == "test":
                self.add("R6", "SERIOUS", call, f"`{k.arg}` uses the test set for early stopping/validation")
        if cn in ("print", "info", "log", "write") and any(            # R18: "test ... best_score_" in a report
                isinstance(c, ast.Attribute) and c.attr == "best_score_" for a in call.args for c in ast.walk(a)) \
                and any(re.search(r"\btest\b", s, re.IGNORECASE) for a in call.args for s in const_strings(a)):
            self.add("R18", "SERIOUS", call, "a validation score (`best_score_`) is printed as a test score")
        if cn == "drop_duplicates" and isinstance(call.func, ast.Attribute) and split_line is not None \
                and call.lineno > split_line and classify(call.func.value) in ("train", "test", "val"):
            self.add("R17", "MINOR", call, "drop_duplicates() on one split after the split: rows shared between "
                                           "train and test are not removed")
        if split_line is not None and call.lineno < split_line:    # R15 / R14
            if cn == "fillna" and any(isinstance(c, ast.Call) and call_name(c) in ("mean", "median", "mode")
                                      for a in call.args for c in ast.walk(a)):
                self.add("R15", "SERIOUS", call, "fillna() with a mean/median/mode computed on the data before "
                                                 "the split")
            if cn == "get_dummies":
                self.add("R14", "MINOR", call, "pd.get_dummies before the split: the columns can differ between "
                                               "train, test and production")

    def selection_loop(self, loop, classify):
        it = loop.iter if isinstance(loop, ast.For) else None
        if it is not None and any(isinstance(c, ast.Call) and call_name(c) == "split" for c in ast.walk(it)):
            return                                     # cross-validation loop: held-out fold is fine
        body = list(ast.walk(loop))
        fits = [n for n in body if isinstance(n, ast.Call) and call_name(n) in ("fit", "fit_transform")]
        scored_test = False
        for n in body:
            if isinstance(n, ast.Call):
                cn = call_name(n)
                if cn and (cn in METRIC_NAMES or cn in ("score", "predict", "predict_proba", "evaluate")):
                    if any(classify(a) == "test" for a in list(n.args) + [k.value for k in n.keywords]):
                        scored_test = True
        picks_best = False
        for n in body:
            if isinstance(n, ast.Assign) and any(re.search("best", t, re.IGNORECASE)
                                                 for tg in n.targets for t in store_names(tg)):
                picks_best = True
            if isinstance(n, ast.Compare) and any(re.search("best", x, re.IGNORECASE) for x in names_in(n)):
                picks_best = True
        if fits and scored_test and picks_best:
            self.add("R13", "SERIOUS", loop, "loop tracks a 'best' candidate using scores on the test set")

    @staticmethod
    def _src(node) -> str:
        try:
            return ast.unparse(node)
        except Exception:  # noqa: BLE001
            return "<call>"

    # ---- file-level rules ----------------------------------------------------------------
    def file_rules(self):
        calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)]
        first_line = lambda items: min(items, key=lambda n: n.lineno)  # noqa: E731

        # seeds (R5)
        global_seed = any(call_name(c) in SEED_FUNCS for c in calls)
        missing = []
        for c in calls:
            cn = call_name(c)
            if not cn or has_star_kwargs(c):
                continue
            seeded = any(k.arg in SEED_KWARGS for k in c.keywords)
            if cn in SPLITTERS_SEEDED and not seeded and not const_false(kw(c, "shuffle")):
                missing.append((c, cn))
            elif cn in SPLITTERS_SHUFFLE_OPT and const_true(kw(c, "shuffle")) and not seeded:
                missing.append((c, cn))
            elif ESTIMATOR_STOCHASTIC.match(cn) and not seeded and (isinstance(c.func, (ast.Name, ast.Attribute))):
                missing.append((c, cn))
        if missing and not global_seed:
            names = sorted({m[1] for m in missing})
            self.add("R5", "SERIOUS", first_line([m[0] for m in missing]),
                     f"{len(missing)} call(s) without random_state/seed ({', '.join(names[:4])}) and no global "
                     "seed: results change on every run")
        elif self.uses_torch_training() and not any(call_name(c) in ("manual_seed", "seed_everything", "set_seed")
                                                    for c in calls):
            self.add("R5", "SERIOUS", self.tree.body[0] if getattr(self.tree, "body", None) else self.tree,
                     "torch training code without torch.manual_seed / set_seed")

        # baseline (R7)
        metric_calls = [c for c in calls if call_name(c) in METRIC_NAMES
                        or (isinstance(c.func, ast.Attribute) and c.func.attr == "score" and c.args)]
        if metric_calls and not self.has_baseline():
            self.add("R7", "SERIOUS", first_line(metric_calls), "metric is computed but no baseline model "
                                                                "(Dummy*, majority/mean prediction) is compared")

        # R8 dropna
        for c in calls:
            if call_name(c) == "dropna" and isinstance(c.func, ast.Attribute) and kw(c, "subset") is None:
                self.add("R8", "MINOR", c, "`dropna()` without `subset=` silently drops rows that have any NaN")

        # R12 rolling
        for stmt in ast.walk(self.tree):
            if not isinstance(stmt, ast.stmt) or isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call) and call_name(sub) in ("rolling", "expanding") \
                        and isinstance(sub.func, ast.Attribute):
                    chain_shift = any(isinstance(c, ast.Call) and call_name(c) in ("shift",)
                                      for c in ast.walk(sub.func.value))
                    stmt_shift = any(isinstance(c, ast.Call) and call_name(c) == "shift" for c in ast.walk(stmt))
                    centered = const_true(kw(sub, "center"))
                    keys = self.stmt_target_keys(stmt)
                    later_shift = bool(keys) and any(
                        isinstance(c, ast.Call) and call_name(c) == "shift" and isinstance(c.func, ast.Attribute)
                        and keys & self.name_keys(c.func.value) for c in ast.walk(self.tree))
                    if centered or not (chain_shift or stmt_shift or later_shift):
                        why = "centered window uses future rows" if centered else \
                            "window without `shift(1)` includes the current row"
                        self.add("R12", "SERIOUS", sub, f"{call_name(sub)}(): {why}")

        # R3 / R4 random split
        split_calls = [c for c in calls if call_name(c) in RANDOM_SPLIT_FUNCS]
        random_split = [c for c in split_calls if not const_false(kw(c, "shuffle"))
                        and call_name(c) not in ("GridSearchCV", "RandomizedSearchCV")]
        if random_split:
            if self.has_time_hint() and not self.has_time_split():
                self.add("R3", "SERIOUS", first_line(random_split),
                         "random/shuffled split although the code handles date/time columns "
                         "(if the model predicts the future, split by time)")
            if self.has_group_hint() and not self.has_group_split():
                self.add("R4", "SERIOUS", first_line(random_split),
                         "id-like column (user/customer/patient/...) present but no group-aware split")

        # R9 torch eval
        if self.uses_torch_training():
            if not any(isinstance(c.func, ast.Attribute) and c.func.attr == "eval" and not c.args for c in calls):
                self.add("R9", "SERIOUS", first_line([c for c in calls if call_name(c) == "backward"] or calls),
                         "torch model is trained but `.eval()` is never called; dropout/batchnorm stay in "
                         "train mode at evaluation")
            elif not any(call_name(c) in ("no_grad", "inference_mode") for c in calls):
                ev = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "eval" and not c.args]
                self.add("R9", "MINOR", ev[0], "`.eval()` used but no `torch.no_grad()` / `inference_mode()`: "
                                              "slower and keeps the autograd graph in memory")

        # R10 double softmax
        ce = any(call_name(c) in ("CrossEntropyLoss", "cross_entropy") for c in calls)
        if ce:
            for fn in ast.walk(self.tree):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "forward":
                    for c in ast.walk(fn):
                        if isinstance(c, ast.Call) and call_name(c) in ("softmax", "Softmax", "log_softmax"):
                            self.add("R10", "SERIOUS", c, "softmax inside forward() while the loss is "
                                                          "CrossEntropyLoss (which applies softmax itself)")
            for c in calls:
                if call_name(c) == "Softmax" and self.inside_sequential(c):
                    self.add("R10", "SERIOUS", c, "nn.Softmax in the model while the loss is CrossEntropyLoss")

        # R11 no pipeline
        preproc_calls = [c for c in calls if call_name(c) and PREPROC.search(call_name(c) or "")
                         and re.match(r"^[A-Z]", call_name(c) or "")]
        has_pipe = any(call_name(c) in ("Pipeline", "make_pipeline", "ColumnTransformer", "make_column_transformer",
                                        "dump", "save", "save_model", "to_pickle") for c in calls)
        if preproc_calls and not has_pipe and not any(
                isinstance(n, ast.Attribute) and n.attr in ("dump", "save") for n in ast.walk(self.tree)):
            self.add("R11", "MINOR", first_line(preproc_calls),
                     "preprocessing is applied step by step with no Pipeline or saved artifact")

    @staticmethod
    def name_keys(node) -> set:
        keys = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                keys.add(n.id)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                keys.add(n.value)
        return keys

    def stmt_target_keys(self, stmt) -> set:
        if not isinstance(stmt, ast.Assign):
            return set()
        keys = set()
        for t in stmt.targets:
            if isinstance(t, ast.Subscript):
                keys |= {n.value for n in ast.walk(t.slice) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            elif isinstance(t, ast.Name):
                keys.add(t.id)
        return keys

    # ---- helpers for file rules ----------------------------------------------------------------
    def uses_torch_training(self) -> bool:
        if not self.imports("torch"):
            return False
        return any(isinstance(c, ast.Call) and call_name(c) == "backward" for c in ast.walk(self.tree))

    def has_baseline(self) -> bool:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and call_name(n) and re.match(r"^Dummy(Classifier|Regressor)$", call_name(n)):
                return True
            ident = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else \
                n.arg if isinstance(n, ast.arg) else n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else None
            if ident and re.search(r"(baseline|naive|majority|dummy|trivial)", ident, re.IGNORECASE):
                return True
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and \
                    re.search(r"(baseline|most_frequent|majority|naive|dummy)", n.value, re.IGNORECASE):
                return True
        return False

    def has_time_hint(self) -> bool:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) <= 40 \
                    and " " not in n.value.strip() and TIME_STR.search(n.value):
                return True
            if isinstance(n, ast.Call) and call_name(n) in ("to_datetime", "parse_dates", "strptime"):
                return True
            if isinstance(n, ast.Attribute) and n.attr == "dt":
                return True
            if isinstance(n, ast.keyword) and n.arg == "parse_dates":
                return True
        return False

    def has_time_split(self) -> bool:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and call_name(n) == "TimeSeriesSplit":
                return True
            if isinstance(n, ast.Call) and const_false(kw(n, "shuffle")):
                return True
            if isinstance(n, ast.Compare):
                text = " ".join(names_in(n) + [a.attr for a in ast.walk(n) if isinstance(a, ast.Attribute)] +
                                [c.value for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)])
                if TIME_STR.search(text) or re.search(r"(cutoff|cut_off|split_date|threshold_date)", text, re.IGNORECASE):
                    return True
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice) and isinstance(n.value, ast.Attribute) \
                    and n.value.attr == "iloc" and (n.slice.lower is None) != (n.slice.upper is None):
                return True
        return False

    def has_group_hint(self) -> bool:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and GROUP_STR_STRICT.match(n.value.strip()):
                return True
            if isinstance(n, ast.Attribute) and GROUP_STR_STRICT.match(n.attr):
                return True
        return False

    def has_group_split(self) -> bool:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and re.search(r"Group", call_name(n) or ""):
                return True
            if isinstance(n, ast.Call) and any(k.arg == "groups" for k in n.keywords):
                return True
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Not, ast.Invert)) and any(
                    isinstance(c, ast.Call) and call_name(c) == "isin" for c in ast.walk(n)):
                return True
        return False

    def inside_sequential(self, target) -> bool:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and call_name(n) == "Sequential" and any(t is target for t in ast.walk(n)):
                return True
        return False


def scan_lines(lines: list[tuple[str, str]]) -> list[dict]:
    """Scan [(location, code_line)] and return findings. Raises ParseError on unparseable code."""
    if not lines:
        return []
    src = "\n".join(t for _, t in lines)
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError) as exc:
        where = lines[min(max((getattr(exc, "lineno", 1) or 1) - 1, 0), len(lines) - 1)][0]
        raise ParseError(f"{where}: {getattr(exc, 'msg', exc)}") from exc
    return Analyzer(tree, [loc for loc, _ in lines]).run()


# --------------------------------------------------------------------- reporting
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
    unparsed: list[str] = []
    for f in files:
        try:
            findings += scan_lines(load_lines(f))
        except ParseError as exc:
            unparsed.append(str(exc))
        except ValueError as exc:
            unparsed.append(str(exc))
    findings.sort(key=lambda f: (LEVEL_ORDER[f["level"]], f["where"]))
    score = trust_score(findings)

    if args.json:
        print(json.dumps({"scan_score": score, "findings": findings, "unparsed": unparsed}, indent=2))
    else:
        print(f"Scanned {len(files) - len(unparsed)} of {len(files)} file(s).")
        for u in unparsed:
            print(f"NOT SCANNED (could not parse): {u}")
        for level in ("FATAL", "SERIOUS", "MINOR"):
            group = [f for f in findings if f["level"] == level]
            if not group:
                continue
            print(f"\n{level} ({len(group)})")
            for f in group:
                print(f"  {f['where']}  [{f['rule']}]\n    problem: {f['problem']}\n    fix:     {f['fix']}")
        if not findings:
            print("\nNo findings. The scanner checks data-flow patterns, not everything: still read the code.")
        print(f"\nScan score: {score}/10 (a starting point; your trust score must also weigh evaluation "
              "design and data issues the scanner cannot see)")
    return (1 if any(f["level"] == "FATAL" for f in findings) else 0), findings


def _safe_console():
    """Source files can contain non-ASCII names and text; never crash on a legacy console encoding."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


if __name__ == "__main__":
    _safe_console()
    sys.exit(run()[0])
