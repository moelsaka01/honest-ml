#!/usr/bin/env python3
"""repro.py - seeds and run snapshots so an ML result can be rerun and trusted.

Use as a module (copy this file next to your training script):

    from repro import set_seed, snapshot_run
    set_seed(cfg["seed"])
    run_dir = snapshot_run("runs", "baseline-lgbm", seed=cfg["seed"], data=["data/train.csv"],
                           config=cfg)           # writes config/env/git/data-hash files
    ... train ...
    write_metrics(run_dir, {"auc": 0.71})

or from the shell:

    python repro.py snapshot --runs-dir runs --name baseline --seed 0 --data data/train.csv --config cfg.yaml
    python repro.py summarize 0.71 0.69 0.73 0.70 0.72      # mean +/- std over seeds

Run directory contents: config.(yaml|json) or config.json, env.txt, git.txt, data_hash.txt,
seed.txt, metrics.json (after write_metrics). Standard library only (numpy / torch used if present).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


# ------------------------------------------------------------------------ seeds
def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed python, numpy and torch (if installed). Pass random_state=seed to sklearn too."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:  # slower, but bit-for-bit repeatable where supported
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# ------------------------------------------------------------------ identities
def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def mask_home(text: str) -> str:
    """Replace your home folder (and so your username) with ~ so run folders are safe to commit."""
    home = str(Path.home())
    for variant in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
        text = re.sub(re.escape(variant), "~", text, flags=re.IGNORECASE)
    return text


def data_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    if p.is_dir():  # hash names + contents of every file, in sorted order
        h = hashlib.sha256()
        files = sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            h.update(str(f.relative_to(p)).replace("\\", "/").encode())
            h.update(file_sha256(f).encode())
        return {"path": mask_home(str(path)), "sha256": h.hexdigest(), "files": len(files)}
    info = {"path": mask_home(str(path)), "sha256": file_sha256(p), "bytes": p.stat().st_size}
    if p.suffix.lower() in (".csv", ".tsv", ".txt", ".jsonl"):
        with open(p, "rb") as f:
            lines = sum(1 for _ in f)
        info["lines"] = lines  # includes the header for CSV/TSV
    return info


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 60) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(repo: str | None = None) -> dict:
    sha = _run(["git", "rev-parse", "HEAD"], repo)
    if sha is None:
        return {"sha": None, "dirty": None, "note": "not a git repository (or git not installed)"}
    status = _run(["git", "status", "--porcelain"], repo)
    info = {"sha": sha, "dirty": bool(status), "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)}
    if status:  # a dirty tree is not reproducible from the SHA alone: say what differs, and fingerprint it
        info["changed_files"] = [re.sub(r"^\S+\s+", "", line) for line in status.splitlines()][:50]
        diff = _run(["git", "diff", "HEAD"], repo)
        info["diff_sha256"] = hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
        info["note"] = ("uncommitted changes: commit them, or keep this diff to reproduce the run "
                        "(diff_sha256 identifies it; untracked files are in changed_files but not in the diff)")
    return info


def env_info() -> str:
    lines = [f"python: {sys.version.split()[0]}", f"platform: {platform.platform()}",
             f"executable: {sys.executable}"]
    try:
        import torch
        lines.append(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    frozen = _run([sys.executable, "-m", "pip", "freeze"], timeout=120)
    lines.append("\n# pip freeze\n" + (frozen if frozen is not None else "(pip freeze failed)"))
    return mask_home("\n".join(lines) + "\n")


# --------------------------------------------------------------------- run dir
def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return slug or "run"


def snapshot_run(runs_dir: str | Path, name: str, seed: int, data: list[str] | None = None,
                 config: dict | str | Path | None = None, repo: str | None = None) -> Path:
    """Create runs/<timestamp>_<name>/ and record everything needed to rerun. Returns the path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / f"{stamp}_{_slug(name)}"
    n = 1
    while run_dir.exists():  # two runs in the same second must not overwrite each other
        n += 1
        run_dir = Path(runs_dir) / f"{stamp}_{_slug(name)}-{n}"
    run_dir.mkdir(parents=True)

    (run_dir / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
    (run_dir / "env.txt").write_text(env_info(), encoding="utf-8")
    (run_dir / "git.txt").write_text(json.dumps(git_info(repo), indent=2) + "\n", encoding="utf-8")

    if config is not None:
        if isinstance(config, dict):
            (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        else:
            src = Path(config)
            if not src.exists():
                raise FileNotFoundError(f"config file not found: {config}")
            shutil.copyfile(src, run_dir / f"config{src.suffix or '.txt'}")

    fingerprints = [data_fingerprint(d) for d in (data or [])]
    (run_dir / "data_hash.txt").write_text(json.dumps(fingerprints, indent=2) + "\n", encoding="utf-8")
    return run_dir


def write_metrics(run_dir: str | Path, metrics: dict) -> None:
    Path(run_dir, "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


def summarize(values: list[float]) -> dict:
    """Mean and sample standard deviation over seeds. A single seed gives std=None on purpose."""
    if not values:
        raise ValueError("no values")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return {"n_seeds": len(values), "mean": mean, "std": std, "min": min(values), "max": max(values)}


# ------------------------------------------------------------------------- cli
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="create a run directory with env/git/data hashes")
    s.add_argument("--runs-dir", default="runs")
    s.add_argument("--name", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--data", nargs="*", default=[])
    s.add_argument("--config")
    m = sub.add_parser("summarize", help="mean +/- std over seeds")
    m.add_argument("values", nargs="+", type=float)
    args = p.parse_args(argv)

    if args.cmd == "snapshot":
        try:
            print(snapshot_run(args.runs_dir, args.name, args.seed, args.data, args.config))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0
    r = summarize(args.values)
    if r["std"] is None:
        print(f"1 seed: {r['mean']:.4f}  (no variance estimate; run at least 3 seeds before claiming a win)")
    else:
        print(f"{r['n_seeds']} seeds: {r['mean']:.4f} +/- {r['std']:.4f}  (min {r['min']:.4f}, max {r['max']:.4f})")
        if r["n_seeds"] < 3:
            print("note: fewer than 3 seeds; treat the std as rough.")
    return 0


def _safe_console():
    """Names and paths can be non-ASCII; never crash on a legacy console encoding."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


if __name__ == "__main__":
    _safe_console()
    sys.exit(main())
