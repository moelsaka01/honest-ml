"""Render assets/demo.gif from the REAL output of examples/demo_train.py.

    pip install pillow
    python examples/make_leaky_data.py
    python tools/make_demo_gif.py

The script runs the demo, captures what it prints, and replays it as a terminal animation, so the GIF
can never show numbers the code does not produce. Regenerate it whenever the demo changes.
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "demo.gif"
COMMAND = "python examples/demo_train.py"

BG, FG, DIM = (13, 17, 23), (201, 209, 217), (110, 118, 129)
RED, GREEN, YELLOW, BLUE = (255, 123, 114), (126, 231, 135), (255, 212, 121), (121, 192, 255)
FONT_CANDIDATES = ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                   "/System/Library/Fonts/Menlo.ttc"]
SIZE = 16


def load_font():
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, SIZE)
    return ImageFont.load_default()


def colour_of(line):
    s = line.strip()
    if s.startswith("1. naive") or s.startswith("[LEAK]"):
        return RED
    if "HONEST" in line or "honest AUC" in line or "-> logistic" in line or "-> gradient" in line \
            or "-> no clear" in line:
        return GREEN
    if set(s) <= {"-"} and s:
        return DIM
    if s.startswith(("honest-ml demo", "leakage-hunter on", "Is the fancier")):
        return YELLOW
    if s.startswith("protocol"):
        return BLUE
    return FG


def main():
    demo = subprocess.run([sys.executable, str(ROOT / "examples" / "demo_train.py")], capture_output=True,
                          text=True, cwd=ROOT)
    if demo.returncode != 0:
        raise SystemExit(f"demo failed:\n{demo.stderr}")
    lines = demo.stdout.rstrip("\n").split("\n")

    font = load_font()
    char_w = font.getlength("M")
    line_h = SIZE + 7
    pad = 22
    cols = max(len(l) for l in lines) + 2
    width = int(cols * char_w + 2 * pad)
    height = int((len(lines) + 4) * line_h + 2 * pad + 20)

    def frame(shown, typed=None, cursor=True):
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        d.ellipse((pad, 10, pad + 10, 20), fill=(255, 95, 86))
        d.ellipse((pad + 18, 10, pad + 28, 20), fill=(255, 189, 46))
        d.ellipse((pad + 36, 10, pad + 46, 20), fill=(39, 201, 63))
        y = pad + 16
        prompt = "$ " + (COMMAND if typed is None else COMMAND[:typed])
        d.text((pad, y), prompt + ("_" if cursor and shown == 0 else ""), font=font, fill=FG)
        y += line_h + 4
        for line in lines[:shown]:
            d.text((pad, y), line, font=font, fill=colour_of(line))
            y += line_h
        return img

    frames, durations = [], []

    def add(img, ms):
        frames.append(img)
        durations.append(ms)

    for n in range(0, len(COMMAND) + 1, 6):                   # typing the command
        add(frame(0, typed=n), 70)
    add(frame(0, typed=len(COMMAND)), 500)

    pause_after = lambda l: 900 if (l.strip().startswith(("1. naive", "4. +")) or set(l.strip()) <= {"-"} and l.strip()
                                    ) else 140 if l.strip() else 260                                   # noqa: E731
    for i, line in enumerate(lines, 1):
        add(frame(i), pause_after(line))
    add(frame(len(lines)), 5000)                                # hold the final screen

    # quantise to keep the file small
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    OUT.parent.mkdir(exist_ok=True)
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True,
                disposal=1)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(frames)} frames, {width}x{height})")


if __name__ == "__main__":
    main()
