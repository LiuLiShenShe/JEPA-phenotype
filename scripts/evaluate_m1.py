#!/usr/bin/env python3
"""Evaluate locked runs on untouched plants and aggregate seeds at plant level."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

from datasets.m1_feature_dataset import FeatureStore, M1FeatureDataset, M1Index
from training.common import TARGET_NAMES, TrainStandardizer, clip_predictions, file_hash, write_json
from training.metrics import cluster_interval, evaluate, plant_scores, prediction_frame
from training.trainer import load_head, model_inference, predict_checkpoint, tensor_inputs


def evaluate_neural(device):
    selection_path = ROOT / "reports/m1/model_selection.json"
    training = json.loads((ROOT / "reports/m1/training_complete.json").read_text())
    if training["selection_registry_sha256"] != file_hash(selection_path) or training["test_used_for_selection"]:
        raise ValueError("Selections changed since training or used test data")
    locked = json.loads(selection_path.read_text())
    index = M1Index(ROOT)
    cache = json.loads((ROOT / "reports/m1/feature_cache.json").read_text())
    store = FeatureStore(ROOT / cache["directory"],root=ROOT)
    repeatability = []
    ablations = []
    for model_name in ("B2a","B2b"):
        for horizon in locked["horizons"]:
            for seed in locked["seeds"]:
                run_name = f"{model_name}_h{horizon}_s{seed}"
                checkpoint_path = ROOT / "outputs/m1/checkpoints" / run_name / "checkpoint.pt"
                model, checkpoint = load_head(checkpoint_path,device)
                if checkpoint["config"]["selection_registry_sha256"] != file_hash(selection_path):
                    raise ValueError("Checkpoint not bound to locked selection")
                if checkpoint["feature_cache_signature"] != store.metadata["features_sha256"]:
                    raise ValueError("Checkpoint feature version mismatch")
                for split in ("val","test"):
                    dataset = M1FeatureDataset(index,store,split,horizon)
                    if split == "test" and set(dataset.rows.plant_uid) & (set(checkpoint["train_plants"])|set(checkpoint["val_plants"])):
                        raise ValueError("Unseen-plant evaluation violated")
                    predictions = predict_checkpoint(checkpoint_path,dataset,device)
                    output_dir = ROOT / "outputs/m1/predictions" / run_name
                    output_dir.mkdir(parents=True,exist_ok=True)
                    prediction_frame(dataset.rows,dataset.targets,predictions,model=model_name,seed=seed).to_csv(output_dir / f"{split}.csv",index=False)
                    metrics, plants = evaluate(dataset.rows,dataset.targets,predictions)
                    write_json(output_dir / f"{split}_metrics.json",metrics)
                    plants.to_csv(output_dir / f"{split}_plants.csv",index=False)
                    if split == "test":
                        repeated = predict_checkpoint(checkpoint_path,dataset,device)
                        max_difference = float(np.abs(repeated-predictions).max())
                        if not np.array_equal(repeated,predictions):
                            raise RuntimeError("Repeated same-checkpoint inference differs")
                        repeatability.append({"run":run_name,"device":device,"max_difference":max_difference,
                                              "checkpoint_sha256":file_hash(checkpoint_path)})
                        if model_name == "B2b":
                            inputs = tensor_inputs(dataset,TrainStandardizer.from_state(checkpoint["time_scaler"]),next(model.parameters()).device)
                            original = inputs["features"]
                            for ablation in ("repeat_latest","reverse_history"):
                                inputs["features"] = original[:, -1:, :].repeat(1,3,1) if ablation == "repeat_latest" else original.flip(1)
                                values = clip_predictions(TrainStandardizer.from_state(checkpoint["target_scaler"]).inverse(model_inference(model,inputs)))
                                plant_errors = plant_scores(dataset.rows,dataset.targets,values)
                                plant_errors["horizon"] = horizon
                                plant_errors["seed"] = seed
                                plant_errors["ablation"] = ablation
                                ablations.append(plant_errors)
                                prediction_frame(dataset.rows,dataset.targets,values,model=f"B2b_{ablation}",seed=seed).to_csv(output_dir / f"test_{ablation}.csv",index=False)
                print(f"Evaluated {run_name}",flush=True)
    pd.concat(ablations,ignore_index=True).to_csv(ROOT / "reports/m1/temporal_ablation_plants.csv",index=False)
    write_json(ROOT / "reports/m1/inference_repeatability.json",{"status":"PASS","runs":repeatability,"test_used_for_selection":False,
               "selection_registry_sha256":file_hash(selection_path)})


def aggregate():
    run_summaries = []
    plant_frames = []
    for directory in sorted((ROOT / "outputs/m1/predictions").iterdir()):
        if not (directory / "test_metrics.json").exists():
            continue
        name_parts = directory.name.split("_")
        model = name_parts[0]
        horizon = int(name_parts[1][1:])
        seed = int(name_parts[2][1:]) if len(name_parts)>2 else 42
        metrics = json.loads((directory / "test_metrics.json").read_text())
        plants = pd.read_csv(directory / "test_plants.csv")
        plants["model"] = model
        plants["horizon"] = horizon
        plants["seed"] = seed
        plant_frames.append(plants)
        for target in TARGET_NAMES:
            run_summaries.append({"model":model,"horizon":horizon,"seed":seed,"target":target,
                                  "plants":metrics["plants"],"windows":metrics["windows"],**{key:value for key,value in metrics[target].items() if not key.endswith("ci95")}})
    all_plants = pd.concat(plant_frames,ignore_index=True)
    all_plants.to_csv(ROOT / "reports/m1/plant_level_errors.csv",index=False)
    runs = pd.DataFrame(run_summaries)
    runs.to_csv(ROOT / "reports/m1/per_seed_metrics.csv",index=False)
    expected = {(model,horizon) for model in ("B0","B1","B1b","B2a","B2b") for horizon in (2,4,8)}
    if set(zip(runs.model,runs.horizon)) != expected:
        raise ValueError("Missing model/horizon outputs")
    summary = []
    for (model,horizon,target),group in runs.groupby(["model","horizon","target"]):
        seed_count = len(group)
        if seed_count != (3 if model in {"B2a","B2b"} else 1) or set(group.plants) != {46}:
            raise ValueError("Incomplete seeds or unseen test plant coverage")
        record = {"model":model,"horizon":horizon,"target":target,"seeds":seed_count,
                  "condition":"Oracle-phenotype" if model in {"B0","B1"} else ("Time-only" if model=="B1b" else "RGB+time"),
                  "test_plants":46,"test_windows":int(group.windows.iloc[0])}
        for metric in ("window_mae","window_rmse","plant_macro_mae","plant_macro_rmse"):
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1)) if seed_count>1 else 0.0
        independent = all_plants.loc[all_plants.model.eq(model)&all_plants.horizon.eq(horizon)&all_plants.target.eq(target)].groupby("plant_uid")[["mae","rmse"]].mean()
        for metric in ("mae","rmse"):
            lower,upper = cluster_interval(independent[metric])
            record[f"plant_macro_{metric}_ci95_low"] = lower
            record[f"plant_macro_{metric}_ci95_high"] = upper
        summary.append(record)
    comparison = pd.DataFrame(summary)
    comparison.to_csv(ROOT / "reports/m1/baseline_comparison.csv",index=False)
    pairs = []
    for horizon in (2,4,8):
        for target in TARGET_NAMES:
            plants = all_plants.loc[all_plants.horizon.eq(horizon)&all_plants.target.eq(target)&all_plants.model.isin(["B2a","B2b"])].groupby(["model","plant_uid"]).mae.mean().unstack("model")
            differences = plants.B2b-plants.B2a
            lower,upper = cluster_interval(differences)
            family_lower,family_upper = cluster_interval(differences,confidence=1-.05/6,replicates=10000)
            pairs.append({"horizon":horizon,"target":target,"plants":len(plants),"gru_minus_mlp_mae":float(differences.mean()),
                          "ci95_low":lower,"ci95_high":upper,"gru_lower_exploratory_ci":upper<0,
                          "bonferroni_family_ci_low":family_lower,"bonferroni_family_ci_high":family_upper,
                          "gru_lower_family_adjusted_ci":family_upper<0,
                          "note":"95% exploratory CI plus 99.1667% per-comparison Bonferroni-family interval; paired plant scores averaged across seeds"})
    pd.DataFrame(pairs).to_csv(ROOT / "reports/m1/gru_vs_mlp_paired.csv",index=False)
    ablation = pd.read_csv(ROOT / "reports/m1/temporal_ablation_plants.csv")
    results = []
    for (name,horizon,target),group in ablation.groupby(["ablation","horizon","target"]):
        original = all_plants.loc[all_plants.model.eq("B2b")&all_plants.horizon.eq(horizon)&all_plants.target.eq(target)].groupby("plant_uid").mae.mean()
        changed = group.groupby("plant_uid").mae.mean()
        difference = changed-original
        lower,upper = cluster_interval(difference)
        results.append({"ablation":name,"horizon":horizon,"target":target,"changed_minus_original_mae":float(difference.mean()),
                        "ci95_low":lower,"ci95_high":upper,"note":"post hoc input intervention; not a retrained ablation"})
    pd.DataFrame(results).to_csv(ROOT / "reports/m1/temporal_ablation.csv",index=False)
    write_json(ROOT / "reports/m1/evaluation_complete.json",{"status":"complete","main_neural_runs":18,"simple_model_runs":9,
               "model_horizon_pairs":15,"test_plants":46,"bootstrap_unit":"plant_uid","bootstrap_replicates":2000,
               "seed_aggregation":"mean per-seed errors; CI resamples seed-averaged plant errors, not ensemble predictions",
               "paired_comparisons":pairs,"test_used_for_selection":False})
    print(comparison[["model","horizon","target","plant_macro_mae_mean"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--aggregate-only",action="store_true")
    args = parser.parse_args()
    if not args.aggregate_only:
        evaluate_neural(args.device)
    aggregate()
