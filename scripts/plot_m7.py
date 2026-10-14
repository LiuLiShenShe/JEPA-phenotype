import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from birth_detection.figures import generate_figures

if __name__ == "__main__":
    generate_figures(ROOT)
