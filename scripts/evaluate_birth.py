import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from birth_detection.common import protocol, write_json
from birth_detection.evaluation import candidate_evaluation_cache, evaluate_candidates, evaluate_forecast, select_birth
from organ_tracking.engine import run_tracking, select_tracking
from birth_detection.training import predict_test
from birth_detection.inference import predict_test_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    arguments = parser.parse_args()
    settings = protocol(ROOT)
    if arguments.split == "test" and not (ROOT / "reports/m7/system_selection.json").exists():
        raise ValueError("Complete and freeze Validation system selection before Test")
    if arguments.split == "test":
        predict_test(ROOT)
        predict_test_pairs(ROOT)
    births = select_birth(ROOT)
    tracks = select_tracking(ROOT)
    cache = candidate_evaluation_cache(ROOT, arguments.split)
    for method in ["difference", "candidate_oracle", "single_RF", "history_RF", "single_PointNet", "PointNet_GRU"]:
        seeds = [17] if method in {"difference", "candidate_oracle"} else settings["seeds"]
        for seed in seeds:
            threshold = births.get(method, {}).get("threshold")
            evaluate_candidates(ROOT, method, seed, arguments.split, threshold, save=True, cache=cache)
            print("BIRTH EVALUATED", method, seed, arguments.split, flush=True)
    del cache
    evaluate_forecast(ROOT, arguments.split).to_csv(ROOT / f"reports/m7/forecast_metrics_{arguments.split}.csv", index=False)
    system_rows = []
    for method in ["C1", "C2"]:
        for seed in settings["seeds"]:
            result = run_tracking(ROOT, method, seed, arguments.split, tracks[method]["threshold"])
            system_rows.extend(result.to_dict("records"))
            result = run_tracking(ROOT, method, seed, arguments.split, tracks[method]["threshold"], births["chosen_method"], births[births["chosen_method"]]["threshold"])
            system_rows.extend(result.to_dict("records"))
    import pandas as pd

    table = pd.DataFrame(system_rows)
    table.to_csv(ROOT / f"reports/m7/tracking_metrics_{arguments.split}.csv", index=False)
    if arguments.split == "val":
        current = pd.read_csv(ROOT / "reports/m6_b/tracking_per_plant.csv").query("split == 'val' and pipeline == 'B3'").idf1.mean()
        summary = table.groupby("method")[["idf1", "new_event_f1"]].mean().reset_index()
        eligible = summary[summary.idf1 >= current].sort_values(["new_event_f1", "idf1", "method"], ascending=[False, False, True])
        chosen = eligible.iloc[0] if len(eligible) else summary.sort_values(["idf1", "method"], ascending=[False, True]).iloc[0]
        write_json(ROOT / "reports/m7/system_selection.json", {"selection_split": "val", "chosen_method": chosen.method,
                   "mean_validation_IDF1": float(chosen.idf1), "mean_validation_event_F1": float(chosen.new_event_f1),
                   "baseline_imported_IDF1": float(current), "birth_model": births["chosen_method"], "no_test_selection": True})
