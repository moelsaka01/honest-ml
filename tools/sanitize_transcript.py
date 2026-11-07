"""Turn a raw `claude -p --output-format stream-json` capture into a publishable transcript.

    python tools/sanitize_transcript.py raw.jsonl out.jsonl --mode pack --seconds 103

Keeps only the conversation: the agent's text, each tool call with its input, each tool result and the
final answer. Drops everything about the machine and account (init event with tool/plugin/MCP lists, ids,
sockets, usage, rate limits, thinking) and masks your home folder and username (agent_demo.redact).
Writes `out.meta.json` next to the output with the model, versions and prompt.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_demo import PROMPT, redact  # noqa: E402


def text_of(content):
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["baseline", "pack"], required=True)
    ap.add_argument("--seconds", type=float)
    args = ap.parse_args()

    events, meta = [], {"mode": args.mode, "prompt": PROMPT}
    for line in Path(args.raw).read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        kind = e.get("type")
        if kind == "system" and e.get("subtype") == "init":
            meta["model"] = e.get("model")
            meta["claude_code_version"] = e.get("claude_code_version")
            meta["honest_ml_version"] = next((p.get("version") for p in e.get("plugins", [])
                                              if p.get("name") == "honest-ml"), None)
            meta["skills_available"] = sorted(s for s in e.get("skills", []) if s.startswith("honest-ml:"))
        elif kind == "assistant":
            for b in e["message"].get("content", []):
                if b.get("type") == "text" and b["text"].strip():
                    events.append({"type": "assistant_text", "text": b["text"]})
                elif b.get("type") == "tool_use":
                    events.append({"type": "tool_use", "name": b.get("name"), "input": b.get("input")})
        elif kind == "user" and isinstance(e["message"].get("content"), list):
            for b in e["message"]["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    events.append({"type": "tool_result", "content": text_of(b.get("content"))})
        elif kind == "result":
            events.append({"type": "final", "text": e.get("result", "")})
            meta["turns"] = e.get("num_turns")
    if args.seconds:
        meta["wall_clock_seconds"] = args.seconds
    meta["skills_invoked"] = [ev["input"].get("skill") for ev in events
                              if ev["type"] == "tool_use" and ev["name"] == "Skill"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(redact(json.dumps(ev, ensure_ascii=False)) for ev in events) + "\n", encoding="utf-8")
    out.with_suffix(".meta.json").write_text(redact(json.dumps(meta, indent=2, ensure_ascii=False)) + "\n",
                                             encoding="utf-8")
    print(f"wrote {out} ({len(events)} events) and {out.with_suffix('.meta.json').name}")


if __name__ == "__main__":
    main()
