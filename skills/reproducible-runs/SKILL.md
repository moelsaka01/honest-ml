---
name: reproducible-runs
description: Use when writing or modifying any ML training, fine-tuning, or experiment script, notebook, or pipeline. Makes runs reproducible - seeds, pinned environments, config files, data versioning, and saved artifacts - so results can be rerun and trusted. Trigger on "train.py", "notebook", "fine-tune", "experiment", "rerun", "results changed", or "can't reproduce".
---

# Reproducible Runs

If you can't rerun it and get the same number, you don't have a result. You have an anecdote.

## Every training script must have

1. **Seeds everywhere**, set once at the top from config:
   ```python
   import os, random, numpy as np
   def set_seed(seed: int) -> None:
       os.environ["PYTHONHASHSEED"] = str(seed)
       random.seed(seed); np.random.seed(seed)
       try:
           import torch
           torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
           torch.backends.cudnn.deterministic = True
           torch.backends.cudnn.benchmark = False
       except ImportError:
           pass
   ```
   Also pass `random_state=seed` to every sklearn splitter/model and seed DataLoader workers.

2. **Config, not magic numbers.** Hyperparameters, paths, and seeds live in one config (YAML/JSON/dataclass) that is saved next to the outputs. No hardcoded learning rates buried in code.

3. **Pinned environment.** Write `requirements.txt` with exact versions (`pip freeze`) or a lockfile (`uv.lock`, `poetry.lock`). Record Python version, CUDA version, and GPU model in the run metadata.

4. **Data identity.** Record the dataset path plus a content hash (or DVC / dataset version tag) and row count. If the data changes, the hash changes.

5. **Code identity.** Record the git commit SHA and whether the tree was dirty.

6. **One output directory per run**: `runs/<timestamp>_<short-name>/` containing `config.yaml`, `metrics.json`, `env.txt`, `data_hash.txt`, `git_sha.txt`, the model artifact, and the log.

7. **Report variance across seeds.** For any headline result, run at least 3 seeds and report mean ± std. A single-seed win is not a win.

## Notebooks

Notebooks must run top-to-bottom on a fresh kernel ("Restart & Run All") without errors. Move reusable logic into `.py` modules. Never rely on cell execution order or hidden state.

## Before saying "done"

Rerun the script from scratch with the saved config and confirm the metric matches within noise. State in your reply: seed(s), commit SHA, data hash, and where the run directory is.
