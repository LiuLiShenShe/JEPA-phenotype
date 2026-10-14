import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from birth_detection.training import train_birth, train_forecast, train_matcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["history_RF", "single_RF", "PointNet_GRU", "single_PointNet", "C1", "C2", "forecast_RF"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:1")
    arguments = parser.parse_args()
    if arguments.method in {"C1", "C2"}:
        train_matcher(ROOT, arguments.method, arguments.seed, arguments.device)
    elif arguments.method == "forecast_RF":
        train_forecast(ROOT, arguments.seed)
    else:
        train_birth(ROOT, arguments.method, arguments.seed, arguments.device)
