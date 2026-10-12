from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m4.metrics import paired_interval
from m5.common import load_protocol
from m4.common import write_json


def paired_comparison(plants, left, right, partition, fraction, metric="mae", task="count"):
    selected = plants.loc[plants.partition.eq(partition) & plants.label_fraction.eq(fraction) & plants.task.eq(task)]
    left_rows, right_rows = selected.loc[selected.condition.eq(left)], selected.loc[selected.condition.eq(right)]
    merged = left_rows.merge(right_rows, on=["plant_uid", "seed"], suffixes=("_left", "_right"), validate="one_to_one")
    if not len(merged):
        raise ValueError("Missing plant-paired comparison")
    merged["delta"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    seed_deltas = merged.groupby("seed").delta.mean()
    interval = paired_interval(merged.groupby("plant_uid").delta.mean().to_numpy())
    left_metric, right_metric = float(merged[f"{metric}_left"].mean()), float(merged[f"{metric}_right"].mean())
    return {"left": left, "right": right, "partition": partition, "label_fraction": fraction, "task": task,
            "metric": metric, "left_metric": left_metric, "right_metric": right_metric, **interval,
            "absolute_improvement": right_metric - left_metric if task == "count" else left_metric - right_metric,
            "relative_improvement": (right_metric - left_metric) / max(right_metric, 1e-12) if task == "count" else (left_metric - right_metric) / max(right_metric, 1e-12),
            "seeds": len(seed_deltas), "improving_seeds": int((seed_deltas < 0).sum()) if task == "count" else int((seed_deltas > 0).sum()),
            "seed_deltas": json.dumps(seed_deltas.to_dict(), sort_keys=True),
            "inference": "plant bootstrap of seed-averaged paired differences; not scan or seed pseudo-replicates"}


def decide(protocol, comparisons, complete, risk_records):
    rules = protocol["primary"]
    candidates = [name for name in ("P3", "P4") if name in protocol["conditions"]]
    decisions = []
    for condition in candidates:
        checks = []
        for right in ("P0", "P2", "S500"):
            official = comparisons.loc[comparisons.left.eq(condition) & comparisons.right.eq(right) & comparisons.partition.eq("official") &
                                       comparisons.task.eq("count") & comparisons.label_fraction.eq(.5)]
            if len(official) != 1:
                checks.append({"right": right, "passed": False, "reason": "incomplete"})
                continue
            row = official.iloc[0]
            folds = comparisons.loc[comparisons.left.eq(condition) & comparisons.right.eq(right) & comparisons.partition.str.startswith("fold") &
                                    comparisons.task.eq("count") & comparisons.label_fraction.eq(.5)]
            improving_folds = int((folds.absolute_improvement > 0).sum())
            passed = (row.absolute_improvement >= rules["minimum_absolute_improvement_leaves"] and
                      row.relative_improvement >= rules["minimum_relative_improvement"] and row.ci95_high < 0 and
                      row.seeds == 3 and row.improving_seeds == 3 and improving_folds >= 3)
            checks.append({"right": right, "passed": bool(passed), "absolute_improvement": float(row.absolute_improvement),
                           "relative_improvement": float(row.relative_improvement), "ci95": [float(row.ci95_low), float(row.ci95_high)],
                           "improving_seeds": int(row.improving_seeds), "improving_folds": improving_folds})
        segment = comparisons.loc[comparisons.left.eq(condition) & comparisons.right.eq("P0") & comparisons.task.eq("segmentation") &
                                  comparisons.partition.eq("official")]
        negative_transfer = bool(len(segment) == 1 and segment.iloc[0].mean_delta <= -protocol["semantic_risk"]["minimum_drop_miou"] and segment.iloc[0].ci95_high < 0)
        direction = comparisons.loc[comparisons.left.eq(condition) & comparisons.right.eq("P0") & comparisons.task.eq("count") &
                                    comparisons.partition.eq("official") & comparisons.label_fraction.eq(.5)]
        local_benefit = bool(len(direction) == 1 and direction.iloc[0].absolute_improvement > 0 and direction.iloc[0].improving_seeds >= 2)
        representation = [record for record in risk_records if record["partition"] == "official" and record["condition"] == condition]
        representation_ok = (len(representation) == 3 and all(not record["hard_collapse"] and record["pooled_rank"] >= 2. and record["token_rank"] >= 2.
                                                             for record in representation))
        decisions.append({"condition": condition, "controls": checks, "semantic_negative_transfer": negative_transfer,
                          "representation_gate": representation_ok,
                          "primary_pass": complete and all(check["passed"] for check in checks) and not negative_transfer and representation_ok,
                          "local_primary_benefit": local_benefit})
    if any(record["primary_pass"] for record in decisions):
        decision = "JEPA-CONTINUE"
    elif complete and any(record["local_primary_benefit"] and not record["semantic_negative_transfer"] for record in decisions):
        decision = "JEPA-LIMITED"
    else:
        decision = "JEPA-STOP"
    return {"decision": decision, "candidate_checks": decisions, "matrix_complete": complete,
            "representation_risks": risk_records, "evidence_scope": "locked exploratory train/validation comparison; no untouched confirmatory test",
            "next_stage_started": False, "stop_scope": "current generic Point-JEPA branch only; not plant phenotyping research",
            "success_from_auxiliary_task_allowed": False}


def summarize(root):
    root = Path(root)
    protocol, lock = load_protocol(root)
    report = root / "reports/m5"
    records, plants = [], []
    old_records = pd.read_csv(root / "reports/m4/training_resources.csv")
    for record in old_records.loc[old_records.condition.isin(("P0", "P2")) & old_records["mode"].eq("finetune")].to_dict("records"):
        record["historical"] = True
        if record["task"] == "count":
            frame = pd.read_csv(root / "outputs/m4/predictions" / f"{record['id']}.csv")
            frame["error"] = np.abs(frame.prediction - frame.truth)
            per_plant = frame.groupby("plant_uid").error.mean().reset_index(name="mae")
        else:
            per_plant = pd.read_csv(root / "outputs/m4/predictions" / f"{record['id']}_plants.csv")
        records.append(record)
        for field in ("partition", "seed", "condition", "label_fraction", "task"):
            per_plant[field] = record[field]
        plants.append(per_plant)
    for path in (root / "outputs/m5/downstream").glob("*/training.json"):
        record = json.loads(path.read_text())
        records.append(record)
        frame = pd.read_csv(path.parent / "plants.csv")
        for field in ("partition", "seed", "condition", "label_fraction", "task"):
            frame[field] = record[field]
        plants.append(frame)
    table, plant_table = pd.DataFrame(records), pd.concat(plants, ignore_index=True)
    table.to_csv(report / "downstream_metrics.csv", index=False)
    plant_table.to_csv(report / "downstream_plant_metrics.csv", index=False)
    keys = ["partition", "seed", "condition", "label_fraction", "task"]
    if table.duplicated(keys).any():
        raise ValueError("Duplicate downstream runs")
    expected = len(protocol["partitions"]) * len(protocol["seeds"]) * len(protocol["conditions"]) * (len(protocol["label_fractions"]) + 1)
    expected_keys = {(partition, seed, condition, fraction, task) for partition in protocol["partitions"]
                     for seed in protocol["seeds"] for condition in protocol["conditions"]
                     for task, fractions in (("count", protocol["label_fractions"]), ("segmentation", [1.])) for fraction in fractions}
    observed_keys = {tuple(row) for row in table[keys].itertuples(index=False, name=None)}
    complete = observed_keys == expected_keys and table.status.eq("COMPLETED").all()
    comparisons = []
    for partition in protocol["partitions"]:
        for condition in protocol["conditions"]:
            if condition == "P0":
                continue
            for right in [name for name in ("P0", "P2", "S500", "P3") if name != condition and name in protocol["conditions"]]:
                for fraction in protocol["label_fractions"]:
                    comparisons.append(paired_comparison(plant_table, condition, right, partition, fraction))
                comparisons.append(paired_comparison(plant_table, condition, right, partition, 1., "miou", "segmentation"))
    paired = pd.DataFrame(comparisons)
    paired.to_csv(report / "paired_comparisons.csv", index=False)
    risk_records, diagnostics, ssl_records = [], {}, []
    sampling_rows = []
    for path in (root / "outputs/m5/checkpoints").glob("*/*/training.json"):
        record = json.loads(path.read_text())
        ssl_records.append(record)
        if "sampling_exposures" in record:
            total_exposures = sum(sum(species.values()) for species in record["sampling_exposures"].values())
            for source, species_counts in record["sampling_exposures"].items():
                for species, count in species_counts.items():
                    sampling_rows.append({"partition": record["partition"], "condition": record["condition"], "seed": record["seed"],
                                          "source": source, "species": species, "sample_exposures": count, "actual_probability": count / total_exposures})
        entries = {}
        for diagnostic_path in path.parent.glob("diagnostics_step*.json"):
            data = json.loads(diagnostic_path.read_text())
            entries[diagnostic_path.stem] = data
        diagnostics[f"{record['partition']}_{record['condition']}_seed{record['seed']}"] = entries
        final = entries.get(f"diagnostics_step{record['optimizer_steps']}")
        if final:
            risk_records.append({"partition": record["partition"], "condition": record["condition"], "seed": record["seed"],
                                 "pooled_rank": final["pooled"]["effective_rank"], "token_rank": final["token"]["effective_rank"],
                                 "hard_collapse": final["hard_collapse"]})
    pd.DataFrame(ssl_records).to_csv(report / "pretraining_dataset_comparison.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(report / "sampling_exposures.csv", index=False)
    for path in (root / "outputs/m5/probes").glob("*/representation.json"):
        record = json.loads(path.read_text())
        key = f"{record['partition']}_{record['condition']}_seed{record['seed']}"
        diagnostics.setdefault(key, {})["posttraining_probe"] = record
    write_json(report / "representation_diagnostics.json", diagnostics)
    budget = []
    for step in (0, 120, 500, 2000):
        path = root / "outputs/m5/checkpoints/official/M5A_seed17" / f"diagnostics_step{step}.json"
        if path.exists():
            data = json.loads(path.read_text())
            budget.append({"step": step, "ssl_loss": data.get("ssl_loss_last_100"),
                           **{f"{kind}_{field}": data[kind][field] for kind in ("pooled", "token") for field in ("std_mean", "effective_rank", "variance_mean", "largest_eigenvalue_fraction")},
                           "different_plants_cosine_uncentred": data["similarities"]["different_plants"]["uncentred"]["quantiles_0_05_50_95_100"][2],
                           "different_plants_cosine_centred": data["similarities"]["different_plants"]["centred"]["quantiles_0_05_50_95_100"][2]})
    pd.DataFrame(budget).to_csv(report / "ssl_budget_comparison.csv", index=False)
    probe_records = [json.loads(path.read_text()) for path in (root / "outputs/m5/probes").glob("*/summary.json")]
    pd.DataFrame(probe_records).to_csv(report / "leaf_structure_probe.csv", index=False)
    probe_plant_rows = []
    for path in (root / "outputs/m5/probes").glob("*/summary.json"):
        record = json.loads(path.read_text())
        frame = pd.read_csv(path.parent / "plants.csv")
        for field in ("partition", "seed", "condition"):
            frame[field] = record[field]
        probe_plant_rows.append(frame)
    if probe_plant_rows:
        probe_plants = pd.concat(probe_plant_rows, ignore_index=True)
        probe_plants.to_csv(report / "leaf_probe_plant_metrics.csv", index=False)
        probe_pairs = []
        for partition in protocol["partitions"]:
            for left in ("S500", "P3", "P4"):
                if left not in protocol["conditions"]:
                    continue
                for right in ("P0", "P2"):
                    merged = probe_plants.loc[probe_plants.partition.eq(partition) & probe_plants.condition.eq(left)].merge(
                        probe_plants.loc[probe_plants.partition.eq(partition) & probe_plants.condition.eq(right)],
                        on=["plant_uid", "seed"], suffixes=("_left", "_right"), validate="one_to_one")
                    for metric in ("centred_pair_auc", "centred_distance_matched_margin"):
                        merged["delta"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
                        probe_pairs.append({"left": left, "right": right, "partition": partition, "metric": metric,
                                            **paired_interval(merged.groupby("plant_uid").delta.mean().to_numpy()),
                                            "scope": "train-plant diagnostic only; positive indicates larger probe score"})
        pd.DataFrame(probe_pairs).to_csv(report / "leaf_probe_paired.csv", index=False)
    decision = decide(protocol, paired, complete, risk_records)
    write_json(report / "decision.json", decision)
    fairness = []
    for pair, group in table.groupby(["partition", "seed", "label_fraction", "task"]):
        checks = {field: group[field].nunique() == 1 for field in ("initial_head_hash", "epochs", "optimizer_steps", "trainable_parameters")}
        fairness.append({"pair": list(pair), "checks": checks, "passed": all(checks.values())})
    write_json(report / "fairness.json", {"status": "PASS" if all(value["passed"] for value in fairness) else "FAIL", "records": fairness})
    write_json(report / "evaluation_complete.json", {"expected_downstream_runs": expected, "completed_downstream_runs": len(table),
               "complete": complete, "expected_probe_runs": len(protocol["partitions"]) * len(protocol["seeds"]) * len(protocol["conditions"]),
               "completed_probe_runs": len(probe_records), "protocol_sha256": lock["protocol_sha256"]})
    return decision
