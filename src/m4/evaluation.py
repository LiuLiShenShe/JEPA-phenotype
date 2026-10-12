from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m4.common import artifact_protocol_matches, load_protocol, write_json
from m4.metrics import count_metrics, paired_interval


def summarize(root):
    root = Path(root)
    protocol, lock = load_protocol(root)
    records = []
    ssl = []
    for path in sorted((root / "outputs/m4/checkpoints").glob("*/*/training.json")):
        record = json.loads(path.read_text())
        if not artifact_protocol_matches(root, record["protocol_sha256"], lock):
            raise ValueError("Checkpoint protocol mismatch in aggregation")
        (ssl if "ssl_seed" in str(path) else records).append(record)
    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("No real downstream results to report")
    report = root / "reports/m4"
    report.mkdir(parents=True, exist_ok=True)
    serializable = [{name: value for name, value in row.items() if name not in ("labelled_plants", "validation_plants", "class_weights")} for row in records]
    pd.DataFrame(serializable).to_csv(report / "training_resources.csv", index=False)
    pd.DataFrame(ssl).to_csv(report / "pretraining_resources.csv", index=False)
    head_shapes = pd.DataFrame(serializable).groupby(["task", "mode", "condition"]).agg(
        trainable_parameters=("trainable_parameters", "first"), total_parameters=("total_parameters", "first"),
        runs=("seed", "count")).reset_index()
    head_shapes.to_csv(report / "parameter_counts.csv", index=False)
    count_rows = []
    segment_rows = []
    plant_rows = []
    all_predictions = []
    stage_rows = []
    for record in records:
        names = ("partition", "seed", "condition", "mode", "label_fraction")
        metadata = {name: record[name] for name in names}
        if record["task"] == "count":
            predictions = pd.read_csv(root / "outputs/m4/predictions" / f"{record['id']}.csv")
            metrics, plants = count_metrics(predictions)
            count_rows.append(metadata | metrics | {"status": "COMPLETED", "labelled_plants": len(record["labelled_plants"]),
                                                     "epochs": record["epochs"], "best_epoch": record["best_epoch"]})
            plant_rows.append(plants.assign(**metadata))
            all_predictions.append(predictions)
            for stage, bounds in protocol["stages"].items():
                selected = predictions.loc[predictions.DAS.between(*bounds)]
                if len(selected):
                    stage_rows.append(metadata | {"stage": stage} | count_metrics(selected)[0])
            registered = np.zeros(len(predictions), dtype=bool)
            for bounds in protocol["stages"].values():
                registered |= predictions.DAS.between(*bounds).to_numpy()
            if not registered.all():
                stage_rows.append(metadata | {"stage": "outside_registered_stages_DAS93"} | count_metrics(predictions.loc[~registered])[0])
        else:
            segment_rows.append(metadata | {name: value for name, value in record.items() if name.startswith(("plant_macro_", "pooled_"))}
                                | {"status": "COMPLETED", "sampled_points": protocol["points"]})
    count = pd.DataFrame(count_rows)
    segmentation = pd.DataFrame(segment_rows)
    count.to_csv(report / "label_efficiency.csv", index=False)
    count.loc[count.label_fraction.eq(1.0)].to_csv(report / "pretrained_comparison.csv", index=False)
    segmentation.to_csv(report / "segmentation_metrics.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(report / "stage_metrics.csv", index=False)
    plants = pd.concat(plant_rows, ignore_index=True)
    plants.to_csv(report / "count_plant_metrics.csv", index=False)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions["absolute_error"] = np.abs(predictions.prediction - predictions.truth)
    predictions.to_csv(report / "count_validation_predictions.csv", index=False)
    primary_predictions = predictions.loc[predictions.condition.eq("P2") & predictions.label_fraction.eq(1.0) & predictions["mode"].eq("finetune")]
    failures = primary_predictions.groupby(["partition", "plant_uid", "DAS"], as_index=False).agg(
        truth=("truth", "first"), prediction=("prediction", "mean"), mae=("absolute_error", "mean"))
    failures.sort_values("mae", ascending=False).head(40).to_csv(report / "failure_cases.csv", index=False)
    diagnostics = {}
    for row in ssl:
        folder = root / "outputs/m4/checkpoints" / row["partition"] / f"ssl_seed{row['seed']}"
        diagnostics[f"{row['partition']}_seed{row['seed']}"] = {stage: json.loads((folder / f"diagnostics_{stage}.json").read_text())
                                                                for stage in ("initial", "final")}
    write_json(report / "representation_diagnostics.json", diagnostics)
    checks = fairness_checks(table)
    write_json(report / "fairness.json", checks)
    comparisons = []
    for (partition, mode, fraction), values in count.groupby(["partition", "mode", "label_fraction"], sort=True):
        pivot = values.pivot(index="seed", columns="condition", values="plant_macro_mae")
        if not {"P0", "P2"}.issubset(pivot.columns):
            continue
        paired = pivot[["P0", "P2"]].dropna()
        subset = plants.loc[plants.partition.eq(partition) & plants["mode"].eq(mode) & plants.label_fraction.eq(fraction)]
        averaged = subset.groupby(["plant_uid", "condition"]).mae.mean().unstack()
        if not {"P0", "P2"}.issubset(averaged.columns):
            continue
        matched = averaged[["P0", "P2"]].dropna()
        interval = paired_interval((matched.P2 - matched.P0).to_numpy(), protocol["primary"]["bootstrap_replicates"],
                                   protocol["primary"]["bootstrap_seed"])
        comparisons.append({"partition": partition, "mode": mode, "label_fraction": fraction,
                            "p0_mae": float(paired.P0.mean()), "p2_mae": float(paired.P2.mean()),
                            "absolute_improvement": float((paired.P0 - paired.P2).mean()),
                            "relative_improvement": float(1 - paired.P2.mean() / paired.P0.mean()),
                            "paired_seeds": len(paired), "improving_seeds": int((paired.P2 < paired.P0).sum()), **interval})
    pd.DataFrame(comparisons).to_csv(report / "paired_comparisons.csv", index=False)
    cross_fold = plants.loc[plants.partition.str.startswith("fold")]
    cross_fold_records = []
    for (condition, mode, fraction, seed), subset in cross_fold.groupby(["condition", "mode", "label_fraction", "seed"]):
        if subset.plant_uid.duplicated().any():
            raise ValueError("Validation folds duplicate independent plants")
        cross_fold_records.append({"condition": condition, "mode": mode, "label_fraction": fraction, "seed": int(seed),
                                   "independent_plants": len(subset), "validation_folds": subset.partition.nunique(),
                                   "plant_macro_mae": float(subset.mae.mean()), "plant_macro_rmse": float(subset.rmse.mean())})
    pd.DataFrame(cross_fold_records).to_csv(report / "cross_fold_validation.csv", index=False)
    baselines = historical_baselines(root, protocol)
    baselines.to_csv(report / "strong_baseline_validation.csv", index=False)
    required_count = {(partition, seed, condition, fraction, mode) for partition, seed, condition, fraction, mode
                      in itertools.product(protocol["partitions"], protocol["seeds"], ("P0", "P2"), protocol["label_fractions"], ("probe", "finetune"))}
    observed_count = {(row["partition"], row["seed"], row["condition"], row["label_fraction"], row["mode"])
                      for row in count.to_dict("records")}
    required_seg = {(partition, seed, condition, mode) for partition, seed, condition, mode
                    in itertools.product(protocol["partitions"], protocol["seeds"], ("P0", "P2"), ("probe", "finetune"))}
    required_seg |= {(partition, seed, "PointNet", "finetune") for partition, seed in itertools.product(protocol["partitions"], protocol["seeds"])}
    observed_seg = {(row["partition"], row["seed"], row["condition"], row["mode"]) for row in segment_rows}
    required_ssl = set(itertools.product(protocol["partitions"], protocol["seeds"]))
    observed_ssl = {(row["partition"], row["seed"]) for row in ssl}
    coverage = {"count_expected": len(required_count), "count_completed": len(required_count & observed_count),
                "count_missing": [list(value) for value in sorted(required_count - observed_count)],
                "segmentation_expected": len(required_seg), "segmentation_completed": len(required_seg & observed_seg),
                "segmentation_missing": [list(value) for value in sorted(required_seg - observed_seg)],
                "ssl_expected": len(required_ssl), "ssl_completed": len(required_ssl & observed_ssl),
                "ssl_missing": [list(value) for value in sorted(required_ssl - observed_ssl)],
                "p1_expected_conditions": "same 90 count and 30 segmentation runs, unavailable external checkpoint",
                "p1_status": "UNAVAILABLE", "p0_p2_matrix_complete": required_count <= observed_count and required_seg <= observed_seg and required_ssl <= observed_ssl,
                "engineering_pass": False, "engineering_blocker": "mandatory P1 official checkpoint condition unavailable"}
    write_json(report / "coverage.json", coverage)
    decision = decide(protocol, comparisons, count, segmentation, baselines, diagnostics, coverage, checks)
    write_json(report / "decision.json", decision)
    write_json(report / "evaluation_complete.json", {"status": "COMPLETED" if coverage["p0_p2_matrix_complete"] else "PARTIAL",
                                                     "no_test_inference": True, "decision": decision["decision"]})
    return count, segmentation, baselines, diagnostics, coverage, decision


def fairness_checks(table):
    records = []
    for keys, group in table.loc[table.condition.isin(("P0", "P2"))].groupby(["partition", "seed", "task", "mode", "label_fraction"]):
        if set(group.condition) != {"P0", "P2"}:
            continue
        records.append({"partition": keys[0], "seed": int(keys[1]), "task": keys[2], "mode": keys[3], "label_fraction": keys[4],
                        "same_head_initialization": group.initial_head_hash.nunique() == 1,
                        "same_trainable_parameters": group.trainable_parameters.nunique() == 1,
                        "same_optimizer_steps": group.optimizer_steps.nunique() == 1,
                        "same_labelled_plants": len({tuple(value) for value in group.labelled_plants}) == 1})
    passed = all(all(record[name] for name in ("same_head_initialization", "same_trainable_parameters", "same_optimizer_steps", "same_labelled_plants")) for record in records)
    return {"status": "PASS" if passed else "FAIL", "paired_runs": len(records), "records": records}


def historical_baselines(root, protocol):
    records = []
    for partition in protocol["partitions"]:
        for model in ("A1-PointNet", "A3-PointNet-Geometry"):
            for seed in protocol["seeds"]:
                path = root / "outputs/m3_5" / partition / "current" / f"{model}_seed{seed}_train_val.csv"
                rows = pd.read_csv(path)
                rows = rows.loc[rows.split.eq("val")].rename(columns={"pred_visible_organ_count": "prediction", "true_visible_organ_count": "truth"})
                metrics, _ = count_metrics(rows)
                records.append({"partition": partition, "seed": seed, "condition": model, **metrics,
                                "source": str(path.relative_to(root)), "comparison_note": "historical 50-epoch multi-target supervision, not matched-capacity ablation"})
        path = root / "outputs/m3_5" / partition / "current" / "simple_train_val.csv"
        rows = pd.read_csv(path)
        for model, subset in rows.loc[rows.split.eq("val")].groupby("model"):
            selected = subset.rename(columns={"pred_visible_organ_count": "prediction", "true_visible_organ_count": "truth"})
            records.append({"partition": partition, "seed": -1, "condition": model, **count_metrics(selected)[0],
                            "source": str(path.relative_to(root)), "comparison_note": "unchanged historical train/validation fit"})
    return pd.DataFrame(records)


def decide(protocol, comparisons, count, segmentation, baselines, diagnostics, coverage, fairness):
    primary = next((row for row in comparisons if row["partition"] == "official" and row["mode"] == "finetune" and row["label_fraction"] == 1.0), None)
    rules = protocol["primary"]
    folds = [row for row in comparisons if row["partition"].startswith("fold") and row["mode"] == "finetune" and row["label_fraction"] == 1.0]
    strong = baselines.loc[baselines.partition.eq("official") & baselines.condition.eq("A3-PointNet-Geometry")].plant_macro_mae.mean()
    no_collapse = bool(diagnostics) and not any(record["final"]["collapse_warning"] for record in diagnostics.values())
    checks = {"primary_complete_three_seeds": bool(primary and primary["paired_seeds"] == 3),
              "absolute_threshold": bool(primary and primary["absolute_improvement"] >= rules["minimum_absolute_improvement_leaves"]),
              "relative_threshold": bool(primary and primary["relative_improvement"] >= rules["minimum_relative_improvement"]),
              "all_official_seeds_improve": bool(primary and primary["improving_seeds"] == 3),
              "plant_paired_ci": bool(primary and primary["ci95_high"] < 0),
              "at_least_three_validation_folds_improve": sum(row["absolute_improvement"] > 0 for row in folds) >= rules["minimum_improving_validation_folds"],
              "strong_baseline_noninferiority": bool(primary and primary["p2_mae"] <= strong + rules["strong_baseline_noninferiority_margin_leaves"]),
              "no_collapse_warning": no_collapse, "paired_fairness": fairness["status"] == "PASS",
              "complete_p0_p2_matrix": coverage["p0_p2_matrix_complete"]}
    decision = "REPRESENTATION-SUCCESS" if all(checks.values()) else "LIMITED-BENEFIT"
    if coverage["p0_p2_matrix_complete"] and decision != "REPRESENTATION-SUCCESS" and not segmentation.empty:
        fine = count.loc[count["mode"].eq("finetune") & count.label_fraction.eq(1.0)].groupby("condition").plant_macro_mae.mean()
        segment = segmentation.loc[segmentation["mode"].eq("finetune")].groupby("condition").plant_macro_miou.mean()
        baseline = baselines.loc[baselines.condition.eq("A3-PointNet-Geometry")].plant_macro_mae.mean()
        if {"P0", "P2"}.issubset(fine.index) and {"P0", "P2", "PointNet"}.issubset(segment.index):
            if baseline < fine.P2 and fine.P0 <= fine.P2 and segment.PointNet > segment.P2 and segment.P0 >= segment.P2:
                decision = "REPRESENTATION-PIVOT"
    return {"decision": decision, "checks": checks, "primary_comparison": primary,
            "official_strong_baseline_mae": float(strong), "improving_validation_folds": sum(row["absolute_improvement"] > 0 for row in folds),
            "scope": "locked validation feasibility evidence; no test-driven task selection, no untouched confirmatory test",
            "external_pretraining_unavailable": True, "engineering_pass": False,
            "temporal_jepa_started": False, "warning": "20 epochs and 24 independent SSL plants per partition do not settle large-scale/pretrained Point-JEPA capability"}
