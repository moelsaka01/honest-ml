---
name: reproducible-runs
description: Use when writing, changing, or running a machine learning training, fine-tuning, or experiment script or notebook, and whenever a result cannot be reproduced or changes between runs. Sets seeds, records config, environment, git commit and data hashes in one directory per run, and reports results across several seeds. Includes a small helper module (seeds, run snapshot, seed summary).
---

# Reproducible Runs

If you cannot rerun it and get the same number, you do not have a result. You have an anecdote.

`<skill-dir>` means the folder that contains this SKILL.md. `scripts/repro.py` uses only the standard library (numpy/torch are used if installed). Copy it next to the training script, or import it by path.

## Every training script must

1. **Seed everything once, from config.**
   ```python
   from repro import set_seed
   set_seed(cfg["seed"])                       # python, numpy, torch (if installed), PYTHONHASHSEED
   ```
   Also pass `random_state=cfg["seed"]` to every sklearn splitter and model, and seed DataLoader workers:
   ```python
   def seed_worker(worker_id):
       s = torch.initial_seed() % 2**32
       np.random.seed(s); random.seed(s)
   loader = DataLoader(ds, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(cfg["seed"]))
   ```
   `set_seed(seed, deterministic_torch=True)` also forces deterministic cuDNN/torch algorithms. That makes runs repeatable where supported, but it can slow training, so use it when you need bit-for-bit repeatability, not by default. GPU runs are not guaranteed identical across hardware or library versions; say so when it matters.

2. **Read hyperparameters, paths and the seed from one config** (YAML/JSON/dataclass) that is saved with the outputs. No learning rates buried in code.

3. **Snapshot the run** before training:
   ```python
   from repro import snapshot_run, write_metrics
   run_dir = snapshot_run("runs", "lgbm-baseline", seed=cfg["seed"], data=["data/train.csv"], config="config.yaml")
   # ... train and evaluate ...
   write_metrics(run_dir, {"roc_auc": 0.71})
   ```
   or from the shell: `python <skill-dir>/scripts/repro.py snapshot --name lgbm-baseline --seed 0 --data data/train.csv --config config.yaml`.
   This creates `runs/<timestamp>_<name>/` containing `seed.txt`, `env.txt` (Python, OS, GPU/torch if present, full `pip freeze`), `git.txt` (commit SHA, branch, whether the tree was dirty; if dirty, the changed files and a SHA-256 of `git diff HEAD`), `data_hash.txt` (SHA-256 and line count per data file, or per directory), and a copy of the config. Your home folder is written as `~`, so a run folder does not publish your username. Two runs in the same second never overwrite each other.

4. **Pin the environment** for the project: `requirements.txt` with exact versions (`pip freeze`) or a lockfile (`uv.lock`, `poetry.lock`).

5. **Commit before the run.** If `git.txt` says `"dirty": true`, the SHA does not describe the code that ran (`changed_files` and `diff_sha256` say what differed). Commit, or save the diff yourself (`git diff HEAD > run.diff`), and say so.

6. **Report variance across seeds.** For any headline result, run at least 3 seeds:
   ```bash
   python <skill-dir>/scripts/repro.py summarize 0.71 0.69 0.73 0.70 0.72
   ```
   It prints mean +/- std (and warns for fewer than 3 values). A single-seed win is not a win. Vary the seed for model initialisation *and* for data splitting when the split is random.

## Notebooks

A notebook must run top to bottom on a fresh kernel (Restart & Run All) without errors. Move reusable logic into `.py` modules. Never rely on cell order or hidden state.

## Before saying "done"

Rerun from scratch with the saved config and confirm the metric matches within seed noise. In your reply state: seed(s), commit SHA (and whether dirty), data hash, and the run directory.

## Works with

`honest-eval` consumes the multi-seed numbers. `dataset-audit` decides which data file gets hashed. `ml-roast` flags missing seeds, configs and pipelines.
