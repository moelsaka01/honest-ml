"""Make the skill scripts importable from the tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for rel in ("leakage-hunter", "dataset-audit", "honest-eval", "reproducible-runs", "llm-eval", "ml-roast"):
    sys.path.insert(0, str(ROOT / "skills" / rel / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
