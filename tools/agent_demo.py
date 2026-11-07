"""Run the REAL Claude Code agent on the demo data and print its work live, for screen recording.

    python tools/agent_demo.py --mode baseline --dir <folder with train.csv and test.csv>
    python tools/agent_demo.py --mode pack     --dir <same kind of folder>

`baseline` runs with all skills disabled; `pack` loads this repository as a plugin. Both get the
identical prompt. Nothing is scripted: the agent's tool calls and replies stream as they happen
(claude -p --output-format stream-json), shown in a compact, readable form. Your home folder and
username are masked on screen and in the saved transcript (see redact()); nothing else is edited. Nothing here changes
what the agent does; it only formats its output.
"""
import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ('I have customer churn data in train.csv and test.csv in this folder. '
          'Train a model to predict the "churned" column and tell me the test AUC.')
WIDTH = 108
C = {"dim": "\x1b[90m", "red": "\x1b[91m", "green": "\x1b[92m", "yellow": "\x1b[93m", "blue": "\x1b[94m",
     "bold": "\x1b[1m", "off": "\x1b[0m"}


def _redaction_rules():
    """Display-level privacy: the agent's scratch folder, your home folder and your username are masked.
    Only text shown on screen and written to --transcript is changed; the agent itself is untouched."""
    home = str(Path.home())
    rules = [(re.compile(r"""[^\s"'`]*[\\/]claude[\\/][^\\/\s"'`]+[\\/][^\\/\s"'`]+[\\/]scratchpad""",
                         re.IGNORECASE), "<scratch>")]
    for variant in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
        rules.append((re.compile(re.escape(variant), re.IGNORECASE), "~"))
    user = getpass.getuser()
    if len(user) >= 3:
        rules.append((re.compile(r"\b" + re.escape(user) + r"\b", re.IGNORECASE), "user"))
    return rules


_RULES = _redaction_rules()


def redact(text):
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def find_claude():
    return shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude.exe")


def say(text="", colour=None, end="\n"):
    text = redact(text)
    sys.stdout.write((C[colour] if colour else "") + text + (C["off"] if colour else "") + end)
    sys.stdout.flush()


def wrap(text, indent="  ", colour=None):
    for para in text.strip().splitlines():
        for line in textwrap.wrap(para, WIDTH - len(indent)) or [""]:
            say(indent + line, colour)


def short(value, n=96):
    value = " ".join(str(value).split())
    return value if len(value) <= n else value[: n - 3] + "..."


def show_tool_use(block):
    name, inp = block.get("name", "?"), block.get("input", {})
    if name == "Skill":
        say(f"  > Skill   {inp.get('skill') or inp.get('name') or inp}", "green")
    elif name == "Bash":
        say(f"  > Bash    {short(inp.get('command', ''))}", "blue")
    elif name in ("Write", "Edit", "Read"):
        say(f"  > {name:<7} {short(inp.get('file_path', ''))}", "blue")
    else:
        say(f"  > {name:<7} {short(json.dumps(inp))}", "blue")


def show_tool_result(block):
    content = block.get("content")
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    lines = [l for l in str(content or "").splitlines() if l.strip()]
    for line in lines[:4]:
        say("      " + short(line, WIDTH - 8), "dim")
    if len(lines) > 4:
        say(f"      ... ({len(lines) - 4} more lines)", "dim")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["baseline", "pack"], required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--transcript", help="also save the raw stream-json here")
    ap.add_argument("--title")
    ap.add_argument("--wait-for", help="do not start until this file exists (lets a recorder start first)")
    ap.add_argument("--done-file", help="create this file when the run has finished")
    ap.add_argument("--hold", type=float, default=0, help="seconds to keep the final screen visible")
    args = ap.parse_args()

    os.system("")                                                    # enable ANSI colours on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # agent text may contain non-ASCII
    if args.title:
        os.system(f"title {args.title}")
    cmd = [find_claude(), "-p", PROMPT, "--output-format", "stream-json", "--verbose", "--no-session-persistence",
           "--setting-sources", "project", "--permission-mode", "acceptEdits",
           "--allowedTools", "Bash", "Write", "Read", "Edit", "Glob", "Grep"]
    label = "WITHOUT honest-ml" if args.mode == "baseline" else "WITH honest-ml installed"
    if args.mode == "baseline":
        cmd.append("--disable-slash-commands")
    else:
        cmd += ["--plugin-dir", str(ROOT)]

    if args.wait_for:
        while not Path(args.wait_for).exists():
            time.sleep(0.2)
    say(f"  Claude Code, {label}", "yellow")
    say("")
    say('  $ claude -p "' + PROMPT + '"', "bold")
    say("")
    raw = open(args.transcript, "w", encoding="utf-8") if args.transcript else None
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=args.dir, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if raw:
            raw.write(redact(line))
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "assistant":
            for block in event["message"].get("content", []):
                if block.get("type") == "text" and block["text"].strip():
                    wrap(block["text"])
                elif block.get("type") == "tool_use":
                    show_tool_use(block)
        elif kind == "user":
            content = event["message"].get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        show_tool_result(block)
        elif kind == "result":
            say("")
            say(f"  done in {time.time() - started:.0f}s", "dim")
    proc.wait()
    if raw:
        raw.close()
    say("")
    if args.done_file:                      # signal first so the recorder stops while the window is still open
        Path(args.done_file).write_text("done", encoding="utf-8")
    time.sleep(args.hold)                   # keep the final screen up; the window must not close before ffmpeg stops


if __name__ == "__main__":
    main()
