import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from birth_detection.analysis import analyze

if __name__ == "__main__":
    analyze(ROOT)
