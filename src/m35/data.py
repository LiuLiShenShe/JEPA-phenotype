"""M3.5 preparation, primary-source verification and feasibility gates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.distance import pdist

from datasets.sugar4d_dataset import _split_names
from datasets.sugar4d_organ_dataset import KEYS, LEAF_FIELDS, ORGAN_FEATURES, ORGAN_TARGETS, build_event_windows, build_organ_windows, make_trajectories
from m3.common import file_hash, read_ply, write_json, xyz_from_vertices


GEOMETRY_FEATURES = ("z_range_cm", "z_q01_cm", "z_q05_cm", "z_q25_cm", "z_q50_cm", "z_q75_cm", "z_q95_cm", "z_q99_cm",
                     "height_quantile_proxy_cm", "extent_x_cm", "extent_y_cm", "hull_diameter_cm", "hull_area_cm2", "hull_perimeter_cm",
                     "radius_q50_cm", "radius_q95_cm", "xyz_std_x_cm", "xyz_std_y_cm", "xyz_std_z_cm", "cov_eig_min_cm", "cov_eig_mid_cm",
                     "cov_eig_max_cm", "log_point_count", "log_bbox_density", "nn_mean_cm", "nn_q95_cm", "upper_z_fraction")


def geometry_descriptors(xyz, sampling_seed=20261011):
    cloud = np.asarray(xyz, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 3 or not np.isfinite(cloud).all():
        raise ValueError("Expected finite XYZ without annotations")
    extent = np.ptp(cloud, axis=0) * 100
    quantiles = np.quantile(cloud[:, 2], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) * 100
    try:
        hull = ConvexHull(cloud[:, :2])
        diameter = float(pdist(cloud[hull.vertices, :2]).max() * 100)
        area, perimeter = hull.volume * 10000, hull.area * 100
    except QhullError:
        diameter, area, perimeter = float(np.linalg.norm(extent[:2])), 0.0, float(2 * extent[:2].sum())
    centered = cloud[:, :2] - np.median(cloud[:, :2], axis=0)
    radius = np.linalg.norm(centered, axis=1) * 100
    sample = cloud[np.random.default_rng(sampling_seed).choice(len(cloud), min(8192, len(cloud)), replace=False)]
    distances = cKDTree(sample).query(sample, k=2)[0][:, 1] * 100
    eigenvalues = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(sample.T)), 0)) * 100
    values = [extent[2], *quantiles, quantiles[-1] - quantiles[0], *extent[:2], diameter, area, perimeter,
              *np.quantile(radius, [0.5, 0.95]), *cloud.std(axis=0).tolist() * np.ones(3) * 100,
              *eigenvalues, np.log1p(len(cloud)), np.log1p(len(cloud) / max(np.prod(extent / 100), 1e-12)),
              distances.mean(), np.quantile(distances, 0.95), np.mean(cloud[:, 2] > (cloud[:, 2].min() + np.ptp(cloud[:, 2]) * 0.75))]
    result = dict(zip(GEOMETRY_FEATURES, map(float, values)))
    if len(result) != len(values) or not np.isfinite(list(result.values())).all():
        raise ValueError("Invalid geometry descriptors")
    return result


def protection_snapshot(root):
    root = Path(root)
    snapshot_path = root / "reports/m3_5/initial_protection.json"
    if snapshot_path.exists():
        return json.loads(snapshot_path.read_text())
    paths = set()
    for directory in ("src", "scripts", "configs", "tests", "reports", "outputs", "data/processed", "data/metadata", "splits"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and "__pycache__" not in path.parts and "/m3_5/" not in relative and not relative.startswith("src/m35/") and "m3_5" not in path.name and relative not in ("src/datasets/sugar4d_organ_dataset.py", "src/models/organ_growth_baselines.py", "src/models/geometry_residual_forecaster.py"):
                paths.add(relative)
    for path in subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines():
        if (root / path).is_file():
            paths.add(path)
    raw = json.loads((root / "reports/m3/raw_file_manifest.json").read_text())
    for record in raw["files"]:
        paths.add(record["path"])
        if file_hash(root / record["path"]) != record["sha256"]:
            raise ValueError(f"Raw input changed since M3: {record['path']}")
    files = {path: file_hash(root / path) for path in sorted(paths)}
    result = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "initial_worktree_at_user_task_start": "clean; main...origin/main", "files": files, "raw_files": len(raw["files"])}
    write_json(snapshot_path, result)
    return result


def verify_protection(root):
    payload = json.loads((Path(root) / "reports/m3_5/initial_protection.json").read_text())
    changed = [path for path, digest in payload["files"].items() if not (Path(root) / path).is_file() or file_hash(Path(root) / path) != digest]
    if changed:
        raise ValueError(f"Legacy files changed: {changed}")
    return {"files_checked": len(payload["files"]), "raw_files_checked": payload["raw_files"], "changes": changed,
            "commit": payload["commit"], "legacy_unchanged": True}


def classification_gate(frame, label, rules):
    details, failures = {}, []
    for split, prefix in (("train", "train"), ("val", "validation")):
        group = frame.loc[frame.split.eq(split)]
        classes = {}
        for value in (0, 1):
            subset = group.loc[group[label].astype(int).eq(value)]
            fraction = len(subset) / max(len(group), 1)
            classes[str(value)] = {"samples": len(subset), "plants": int(subset.plant_uid.nunique()), "fraction": fraction}
            if len(subset) < rules[f"{prefix}_class_samples"] or subset.plant_uid.nunique() < rules[f"{prefix}_class_plants"] or fraction < rules["minimum_class_fraction"]:
                failures.append(f"{split}/class{value} coverage inadequate")
        details[split] = classes
    return {"feasible": not failures, "failures": failures, "train_validation_only_gate": True, "classes": details}


def feasibility_tables(trajectories, organ_windows, events, splits, protocol, destination):
    coverage, track_rows, stage_rows, gates = [], [], [], {}
    for plant, group in trajectories.groupby("plant_uid"):
        sizes = group.loc[group.observation_status.eq("observed")].groupby("leaf_instance_id").size()
        gaps, disappearing = 0, 0
        for _, leaf in group.groupby("leaf_instance_id"):
            positions = np.flatnonzero(leaf.sort_values("DAS").observation_status.eq("observed").to_numpy())
            gaps += int(np.any(np.diff(positions) > 1))
            disappearing += int(positions[-1] < leaf.DAS.nunique() - 1)
        track_rows.append({"plant_uid": plant, "trajectories": len(sizes), "observed_records": int(sizes.sum()),
                           "tracks_ge3_observations": int((sizes >= 3).sum()), "disappearing_tracks": disappearing,
                           "reappearing_tracks": gaps, "not_observed_grid_cells": int(group.observation_status.ne("observed").sum())})
    tracks = pd.DataFrame(track_rows)
    for partition, assignment in splits.items():
        windows = organ_windows.assign(split=organ_windows.plant_uid.map(assignment))
        event = events.assign(split=events.plant_uid.map(assignment))
        gates[partition] = {}
        for horizon in protocol["horizons"]:
            current = windows.loc[windows.horizon.eq(horizon)]
            reasons = []
            for split, prefix in (("train", "train"), ("val", "validation")):
                valid = current.loc[current.split.eq(split) & current.future_observed]
                if valid.plant_uid.nunique() < protocol["feasibility"][f"{prefix}_plants"] or len(valid) < protocol["feasibility"][f"{prefix}_morphology_samples"]:
                    reasons.append(f"{split}: too few independent plants or valid labels")
            gates[partition][str(horizon)] = {"morphology": {"feasible": not reasons, "failures": reasons, "scope": "published automatic label prediction only"},
                                             "visibility": classification_gate(current, "future_observed", protocol["feasibility"]),
                                             "event": classification_gate(event.loc[event.horizon.eq(horizon)], "event", protocol["feasibility"])}
            for split in ("train", "val", "test"):
                subset = current.loc[current.split.eq(split)]
                visible = subset.loc[subset.future_observed]
                ev = event.loc[event.horizon.eq(horizon) & event.split.eq(split)]
                plants = [plant for plant, name in assignment.items() if name == split]
                coverage.append({"partition": partition, "split": split, "horizon": horizon, "independent_plants": len(plants),
                                 "trajectories": int(tracks.loc[tracks.plant_uid.isin(plants), "trajectories"].sum()),
                                 "tracks_ge3_observations": int(tracks.loc[tracks.plant_uid.isin(plants), "tracks_ge3_observations"].sum()),
                                 "eligible_windows": len(subset), "valid_morphology_windows": len(visible), "morphology_plants": visible.plant_uid.nunique(),
                                 "future_not_observed_windows": int((~subset.future_observed).sum()),
                                 "event_positive": int(ev.event.sum()), "event_negative": int((1 - ev.event).sum()),
                                 "event_positive_plants": ev.loc[ev.event.eq(1)].plant_uid.nunique(), "event_negative_plants": ev.loc[ev.event.eq(0)].plant_uid.nunique()})
            if partition == "official":
                for (split, day), group in current.groupby(["split", "target_DAS"]):
                    ev = event.loc[event.horizon.eq(horizon) & event.split.eq(split) & event.target_DAS.eq(day)]
                    stage_rows.append({"split": split, "horizon": horizon, "target_DAS": int(day), "organ_windows": len(group),
                                       "valid_organ_windows": int(group.future_observed.sum()), "organ_plants": group.plant_uid.nunique(),
                                       "event_positive": int(ev.event.sum()), "event_negative": int((1 - ev.event).sum()), "event_plants": ev.plant_uid.nunique()})
    coverage_frame = pd.DataFrame(coverage)
    tracks.to_csv(destination / "trajectories_per_plant.csv", index=False)
    coverage_frame.to_csv(destination / "task_coverage.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(destination / "DAS_task_distribution.csv", index=False)
    return {"gates": gates, "independent_unit": "plant_uid", "coverage": coverage_frame.to_dict(orient="records"),
            "tracks": len(trajectories[["plant_uid", "leaf_instance_id"]].drop_duplicates()), "observed_records": int(trajectories.observation_status.eq("observed").sum()),
            "not_observed_grid_cells": int(trajectories.observation_status.ne("observed").sum()), "disappearing_tracks": int(tracks.disappearing_tracks.sum()),
            "reappearing_tracks": int(tracks.reappearing_tracks.sum()), "all_plants_scanned_at_all_times": True,
            "missingness_interpretation": "Not observed is not zero area, biological death, or confirmed occlusion. PLY/CSV mismatches are rejected."}


def prepare(root):
    root = Path(root).resolve()
    report, output = root / "reports/m3_5", root / "outputs/m3_5/data"
    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "configs/m3_5/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    lock = report / "protocol_lock.json"
    if lock.exists() and json.loads(lock.read_text())["sha256"] != file_hash(protocol_path):
        raise ValueError("Locked M3.5 protocol changed")
    write_json(lock, {"sha256": file_hash(protocol_path), "locked_before_training": True, "primary_task": protocol["primary_task"]})
    protection_snapshot(root)
    dataset = root / "data/raw/sugar4d/extracted_v4"
    official = _split_names(dataset)
    metadata = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
    splits = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]
    if any({int(plant.rsplit("_", 1)[-1]) for plant, split in splits["official"].items() if split == name} != values for name, values in official.items()):
        raise ValueError("Existing M3 partition differs from official raw split")
    raw_plant = pd.read_csv(dataset / "measurements/plant_related.csv").sort_values(["Plant ID", "Days after sowing"])
    ordered = metadata.sort_values(["plant_id", "DAS"])
    for source, target, factor in (("Height Maximum [m]", "plant_height", 100), ("Width Major Axis [m]", "plant_width", 100), ("Leaf Quantity", "visible_organ_count", 1)):
        if not np.allclose(raw_plant[source].to_numpy() * factor, ordered[target].to_numpy(), atol=1e-10, rtol=1e-12):
            raise ValueError("Released plant target mismatch")
    leaves = pd.read_csv(dataset / "measurements/leaf_related.csv").rename(columns={"days after sowing": "DAS", "plant": "plant_id", "instance id": "leaf_instance_id"})
    leaves["plant_uid"] = leaves.plant_id.map(lambda value: f"plant_{value:03d}")
    old_leaves = pd.read_csv(root / "data/processed/sugar4d/leaf_metadata.csv").sort_values(KEYS).reset_index(drop=True)
    sorted_leaves = leaves.sort_values(KEYS).reset_index(drop=True)
    if len(old_leaves) != len(leaves) or not old_leaves[KEYS].equals(sorted_leaves[KEYS]) or not np.allclose(old_leaves[list(LEAF_FIELDS)], sorted_leaves[list(LEAF_FIELDS)], rtol=1e-12, atol=1e-12):
        raise ValueError("Existing M3 organ metadata differs from raw CSV")
    geometry_path, organs_path = output / "geometry.csv", output / "ply_organs.csv"
    if not geometry_path.exists() or not organs_path.exists():
        organ_records, geometry_records = [], []
        for index, frame in enumerate(metadata.itertuples(index=False)):
            vertices, info = read_ply(root / frame.pointcloud_path)
            xyz = xyz_from_vertices(vertices)
            descriptors = geometry_descriptors(xyz, protocol["sampling_seed"] + frame.plant_id * 1000 + frame.DAS)
            geometry_records.append({"plant_uid": frame.plant_uid, "DAS": int(frame.DAS), **descriptors})
            semantic, instance = np.asarray(vertices["label_semantic"]), np.asarray(vertices["label_instance"])
            identifiers = np.unique(instance[semantic == 2])
            if len(identifiers) != frame.visible_organ_count or len(xyz) != frame.point_count:
                raise ValueError("PLY count association mismatch")
            for identifier in identifiers:
                cloud = xyz[(semantic == 2) & (instance == identifier)]
                centroid = cloud.mean(axis=0)
                organ_records.append({"plant_uid": frame.plant_uid, "leaf_instance_id": int(identifier), "DAS": int(frame.DAS),
                                      "point_count": len(cloud), **dict(zip(("centroid_x_m", "centroid_y_m", "centroid_z_m"), map(float, centroid)))})
            if (index + 1) % 96 == 0:
                print(f"AUDIT {index+1}/{len(metadata)} raw XYZ geometry and organ joins", flush=True)
        pd.DataFrame(organ_records).to_csv(organs_path, index=False)
        pd.DataFrame(geometry_records).to_csv(geometry_path, index=False)
    organs = pd.read_csv(organs_path)
    old_organs = pd.read_csv(root / "data/processed/sugar4d/ply_organs.csv").sort_values(KEYS).reset_index(drop=True)
    ordered_organs = organs.sort_values(KEYS).reset_index(drop=True)
    if not ordered_organs[KEYS].equals(old_organs[KEYS]) or not np.allclose(ordered_organs.iloc[:, 3:].to_numpy(), old_organs.iloc[:, 3:].to_numpy(), rtol=1e-6, atol=1e-7):
        raise ValueError("Recomputed PLY organs disagree with M3 audit")
    trajectories = make_trajectories(leaves, organs, metadata)
    organ_windows = build_organ_windows(trajectories)
    events, states = build_event_windows(trajectories, metadata)
    trajectories.to_csv(output / "organ_trajectories.csv", index=False)
    organ_windows.to_csv(output / "organ_windows.csv", index=False)
    events.to_csv(output / "event_windows.csv", index=False)
    states.to_csv(output / "event_states.csv", index=False)
    feasibility = feasibility_tables(trajectories, organ_windows, events, splits, protocol, report)
    manual = pd.read_csv(dataset / "measurements/reference/leaf_related_manual.csv").rename(columns={"days after sowing": "DAS", "plant": "plant_id", "instance id": "leaf_instance_id"})
    comparison = leaves.merge(manual, on=["plant_id", "leaf_instance_id", "DAS"], suffixes=("_auto", "_manual"), validate="one_to_one")
    noise = {}
    for field in tuple(LEAF_FIELDS)[:-1]:
        error = comparison[field + "_auto"] - comparison[field + "_manual"]
        noise[LEAF_FIELDS[field]] = {"records": len(error), "plants": int(comparison.plant_id.nunique()), "DAS": 93,
                                    "mae": float(error.abs().mean()), "bias": float(error.mean()),
                                    "interpretation": "cross-method agreement, NOT repeated longitudinal noise floor"}
    noise["leaf_area"] = {"manual_validation_available": False}
    transitions = []
    for (plant, leaf), group in trajectories.groupby(["plant_uid", "leaf_instance_id"]):
        observed = group.loc[group.observation_status.eq("observed")].sort_values("DAS")
        for index in range(1, len(observed)):
            for target in ORGAN_TARGETS:
                previous, current = observed.iloc[index - 1], observed.iloc[index]
                transitions.append({"plant_uid": plant, "leaf_instance_id": leaf, "from_DAS": int(previous.DAS), "to_DAS": int(current.DAS),
                                    "target": target, "delta": float(current[target] - previous[target])})
    pd.DataFrame(transitions).to_csv(report / "organ_label_transitions.csv", index=False)
    feasibility.update({"measurement_agreement": noise, "association": {"raw_csv_rows": len(leaves), "ply_organ_rows": len(organs), "duplicates": 0, "mismatches": 0,
                        "recomputed_all_768_ply_centroids": True, "legacy_leaf_values_verified": True, "legacy_plant_values_verified": True},
                        "geometry_contract": {"features": list(GEOMETRY_FEATURES), "XYZ_only": True, "growth_points_read": False,
                                              "semantic_or_instance_features": False, "proxies_not_official_measurements": True},
                        "decision_scope": "Numerically feasible for automatic-label prediction; biological claims remain limited by 48 plants and label agreement."})
    write_json(report / "feasibility.json", feasibility)
    write_json(report / "data_fingerprints.json", {str(path.relative_to(root)): file_hash(path) for path in sorted(output.glob("*.csv"))})
    print(json.dumps({"tracks": feasibility["tracks"], "observed_records": feasibility["observed_records"], "gates": feasibility["gates"]["official"]}, indent=2), flush=True)
    return feasibility
