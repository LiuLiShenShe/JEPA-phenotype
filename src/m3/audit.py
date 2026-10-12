from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3.common import file_hash, read_ply, write_json, xyz_from_vertices


LEAF_KEYS = ["plant_uid", "leaf_instance_id", "DAS"]


def detailed_audit(root, metadata, leaves, protocol):
    root = Path(root)
    dataset = root / "data/raw/sugar4d/extracted_v4"
    processed = root / "data/processed/sugar4d"
    reports = root / "reports/m3"
    cache = processed / "point_cache"
    cache.mkdir(parents=True, exist_ok=True)
    points = np.load(dataset / "supplements/growth_points.npy", allow_pickle=True).item()
    measurements = pd.read_csv(dataset / "measurements/plant_related.csv").set_index(["Plant ID", "Days after sowing"])
    anchors = {}
    organ_rows = []
    anomalies = []
    geometry_rows = []
    cache_rows = []
    for frame in metadata.sort_values(["plant_uid", "DAS"]).itertuples(index=False):
        vertices, info = read_ply(root / frame.pointcloud_path)
        xyz = xyz_from_vertices(vertices)
        semantic = np.asarray(vertices["label_semantic"])
        instances = np.asarray(vertices["label_instance"])
        norms = np.linalg.norm(np.column_stack([vertices[name] for name in ("nx", "ny", "nz")]), axis=1)
        if not np.isfinite(norms).all():
            raise ValueError("Nonfinite normals")
        valid_pairs = ((semantic == 0) & (instances == -2)) | ((semantic == 1) & (instances == -1)) | ((semantic == 2) & (instances >= 0))
        if not valid_pairs.all():
            raise ValueError("Semantic/instance mapping inconsistency")
        labels = np.unique(instances[semantic == 2])
        if frame.plant_uid not in anchors:
            if frame.DAS != 37:
                raise ValueError("Initial anchor must only use first actual scan")
            anchors[frame.plant_uid] = np.asarray([np.median(xyz[:, 0]), np.median(xyz[:, 1]), 0.0], dtype=np.float32)
        for leaf_id in labels:
            positions = xyz[instances == leaf_id]
            center = positions.mean(axis=0)
            organ_rows.append({"plant_uid": frame.plant_uid, "leaf_instance_id": int(leaf_id), "DAS": frame.DAS, "point_count": len(positions),
                               "centroid_x_m": float(center[0]), "centroid_y_m": float(center[1]), "centroid_z_m": float(center[2])})
        extent = np.ptp(xyz, axis=0)
        voxel_quantity = measurements.loc[(frame.plant_id, frame.DAS), "Points Quantity"]
        growth_point = np.asarray(points[frame.plant_uid])
        height_proxy = float((xyz[:, 2].max() - growth_point[2]) * 100)
        if len(labels) != frame.visible_organ_count:
            anomalies.append({"plant_uid": frame.plant_uid, "DAS": frame.DAS, "type": "leaf_quantity_mismatch", "details": f"PLY={len(labels)}, CSV={frame.visible_organ_count}"})
        metadata.loc[(metadata.plant_uid == frame.plant_uid) & (metadata.DAS == frame.DAS), "visible_organ_count"] = len(labels)
        geometry_rows.append({"plant_uid": frame.plant_uid, "DAS": frame.DAS, "point_count": len(xyz), "csv_points_quantity": int(voxel_quantity),
                              "semantic_values": json.dumps(np.unique(semantic).tolist()), "instance_values": json.dumps(np.unique(instances).tolist()),
                              "leaf_instance_count": len(labels), "normal_nonunit_fraction": float((np.abs(norms - 1) > 0.01).mean()),
                              "extent_x_cm": float(extent[0] * 100), "extent_y_cm": float(extent[1] * 100), "extent_z_cm": float(extent[2] * 100),
                              "growthpoint_height_proxy_cm": height_proxy, "csv_height_cm": frame.plant_height,
                              "height_proxy_minus_csv_cm": height_proxy - frame.plant_height,
                              "density_points_per_bbox_m3": float(len(xyz) / max(float(np.prod(extent)), 1e-12)),
                              "min_z_m": float(xyz[:, 2].min()), "max_z_m": float(xyz[:, 2].max())})
        generator = np.random.default_rng(protocol["sampling_seed"] + frame.plant_id * 1000 + frame.DAS)
        indices = generator.choice(len(xyz), protocol["cache_points"], replace=False)
        cached = xyz[indices] - anchors[frame.plant_uid]
        destination = cache / (Path(frame.pointcloud_path).stem + ".npy")
        np.save(destination, cached.astype(np.float32), allow_pickle=False)
        cache_rows.append({"pointcloud": frame.pointcloud_path, "cache": str(destination.relative_to(root)), "sha256": file_hash(destination), "points": len(cached)})
    metadata["visible_organ_count_source"] = "number of unique nonnegative instance IDs on semantic class leaf in released PLY; GT used only as supervision/oracle"
    organs = pd.DataFrame(organ_rows)
    if organs.duplicated(LEAF_KEYS).any() or leaves.duplicated(LEAF_KEYS).any():
        raise ValueError("Duplicate leaf association")
    associations = organs.merge(leaves, on=LEAF_KEYS, how="outer", indicator=True, validate="one_to_one", suffixes=("", "_csv"))
    mismatch = associations.loc[associations._merge.ne("both")]
    for row in mismatch.itertuples(index=False):
        anomalies.append({"plant_uid": row.plant_uid, "DAS": row.DAS, "type": "organ_csv_association", "details": str(row)})
    days = sorted(metadata.DAS.unique())
    day_positions = {int(day): position for position, day in enumerate(days)}
    full_rows = []
    reappearance = []
    jumps = []
    for (plant_uid, leaf_id), group in associations.groupby(["plant_uid", "leaf_instance_id"]):
        observed = group.loc[group._merge.ne("right_only")].sort_values("DAS")
        lookup = group.set_index("DAS")
        positions = [day_positions[int(value)] for value in observed.DAS]
        if any(right - left > 1 for left, right in zip(positions, positions[1:])):
            reappearance.append({"plant_uid": plant_uid, "leaf_instance_id": int(leaf_id), "observed_DAS": observed.DAS.astype(int).tolist()})
        for day in days:
            record = {"plant_uid": plant_uid, "leaf_instance_id": int(leaf_id), "DAS": int(day), "observation_status": "not_observed"}
            if day in lookup.index:
                record.update(lookup.loc[day].drop(labels="_merge").to_dict())
                record["observation_status"] = "observed" if lookup.loc[day, "_merge"] == "both" else "csv_only" if lookup.loc[day, "_merge"] == "right_only" else "ply_only"
            full_rows.append(record)
        centers = observed[["centroid_x_m", "centroid_y_m", "centroid_z_m"]].to_numpy()
        for position in range(1, len(observed)):
            displacement = float(np.linalg.norm(centers[position] - centers[position - 1]) * 100)
            jumps.append({"plant_uid": plant_uid, "leaf_instance_id": int(leaf_id), "from_DAS": int(observed.DAS.iloc[position - 1]), "to_DAS": int(observed.DAS.iloc[position]), "displacement_cm": displacement})
    full = pd.DataFrame(full_rows)
    full["leaf_instance_id"] = full.leaf_instance_id.astype(int)
    full.to_csv(processed / "organ_temporal_grid.csv", index=False)
    organs.to_csv(processed / "ply_organs.csv", index=False)
    mismatch.to_csv(reports / "organ_csv_mismatches.csv", index=False)
    geometry = pd.DataFrame(geometry_rows)
    geometry.to_csv(reports / "geometry_audit.csv", index=False)
    pd.DataFrame(jumps).to_csv(reports / "leaf_centroid_displacements.csv", index=False)
    write_json(reports / "leaf_reappearance.json", {"instances": reappearance, "criterion": "missing actual scan between two observed scans; not inferred biological birth/death"})
    first = organs.groupby(["plant_uid", "leaf_instance_id"]).DAS.min().reset_index(name="first_observed_DAS")
    first.to_csv(reports / "organ_first_observations.csv", index=False)
    manual = pd.read_csv(dataset / "measurements/reference/leaf_related_manual.csv").rename(columns={"days after sowing": "DAS", "plant": "plant_id", "instance id": "leaf_instance_id"})
    manual["plant_uid"] = manual.plant_id.map(lambda value: f"plant_{int(value):03d}")
    manual_join = manual.merge(leaves, on=LEAF_KEYS, suffixes=("_manual", "_auto"), how="left", validate="one_to_one", indicator=True)
    validation = []
    for column in ("total length [cm]", "leaf blade length [cm]", "petiole length [cm]", "leaf blade width [cm]"):
        subset = manual_join.dropna(subset=[column + "_manual", column + "_auto"])
        error = subset[column + "_auto"] - subset[column + "_manual"]
        validation.append({"parameter": column, "joined_leaves": len(subset), "MAE_cm": float(error.abs().mean()), "RMSE_cm": float(np.sqrt(np.square(error).mean())), "mean_error_cm": float(error.mean()), "scope": "DAS93 only, automatic template estimates vs destructive manual reference"})
    pd.DataFrame(validation).to_csv(reports / "manual_leaf_validation.csv", index=False)
    fields = []
    raw_csv = pd.read_csv(dataset / "measurements/plant_related.csv")
    for column in raw_csv.columns:
        converted = pd.to_numeric(raw_csv[column], errors="coerce")
        fields.append({"field": column, "dtype": str(raw_csv[column].dtype), "null_count": int(raw_csv[column].isna().sum()), "nonnumeric_count": int((converted.isna() & raw_csv[column].notna()).sum()),
                       "numeric_min": float(converted.min()) if converted.notna().any() else None, "numeric_max": float(converted.max()) if converted.notna().any() else None})
        if (converted.isna() & raw_csv[column].notna()).any():
            for position in np.flatnonzero((converted.isna() & raw_csv[column].notna()).to_numpy()):
                row = raw_csv.iloc[position]
                anomalies.append({"plant_uid": f"plant_{int(row['Plant ID']):03d}", "DAS": int(row["Days after sowing"]), "type": "csv_nonnumeric_value", "details": f"{column}: {row[column]} (original unchanged; not selected as target)"})
    write_json(reports / "actual_field_schema.json", {"plant": fields, "leaf": list(leaves.columns), "reference": list(manual.columns)})
    pd.DataFrame(anomalies, columns=["plant_uid", "DAS", "type", "details"]).to_csv(reports / "data_anomalies.csv", index=False)
    write_json(reports / "sampling_cache_manifest.json", {"sampling_seed": protocol["sampling_seed"], "pool_points": protocol["cache_points"], "input": "only XYZ", "unit": "metre", "anchors": {name: values.tolist() for name, values in anchors.items()},
               "anchor_rule": "subtract initial DAS37 XYZ median x/y only; z remains original floor-referenced coordinate; same anchor all dates; no per-frame rescaling", "files": cache_rows})
    raw_records = []
    for path in sorted(dataset.rglob("*")):
        if path.is_file():
            raw_records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    write_json(reports / "raw_file_manifest.json", {"files": raw_records, "raw_source_preserved": True})
    result = {"ply_organ_observations": len(organs), "unique_ply_leaf_instances": len(first), "organ_csv_mismatches": len(mismatch), "reappearing_instances": len(reappearance), "manual_reference_joined": int(manual_join._merge.eq("both").sum()),
              "geometry": {"height_proxy_minus_csv_cm_max_abs": float(geometry.height_proxy_minus_csv_cm.abs().max()), "coordinate_unit": "m verified by z-minus-growth-point vs released [m] height and approximately 10m experiment layout", "axes": "z vertical positive, x/y registered greenhouse ground plane; global origin floor target frame, not a biological growth point", "normal_nonunit_fraction_max": float(geometry.normal_nonunit_fraction.max())},
              "anomalies": len(anomalies), "raw_files_hashed": len(raw_records), "cache_files": len(cache_rows)}
    write_json(reports / "detailed_data_audit.json", result)
    return metadata, result


def make_folds(metadata, mapping, protocol, destination):
    plants = metadata.drop_duplicates("plant_uid")[["plant_uid", "plant_id", "genotype", "split"]].copy()
    generator = np.random.default_rng(protocol["fold_seed"])
    bucket = {}
    totals = np.zeros(4, dtype=int)
    for genotype, group in plants.groupby("genotype", sort=True):
        order = generator.permutation(sorted(group.plant_uid))
        start = int(np.argmin(totals))
        for index, plant in enumerate(order):
            assigned = (start + index) % 4
            bucket[plant] = assigned
            totals[assigned] += 1
    if not np.array_equal(totals, [12, 12, 12, 12]):
        raise ValueError("Fold plant counts are not balanced")
    plans = {"official": dict(zip(plants.plant_uid, plants.split))}
    for fold in range(4):
        plans[f"fold{fold}"] = {plant: "test" if value == fold else "val" if value == (fold + 1) % 4 else "train" for plant, value in bucket.items()}
    distributions = []
    for name, assignment in plans.items():
        if len(assignment) != 48 or set(assignment.values()) != {"train", "val", "test"}:
            raise ValueError("Invalid fold partition")
        for split in ("train", "val", "test"):
            subset = plants.loc[plants.plant_uid.map(assignment).eq(split)]
            for genotype, group in subset.groupby("genotype"):
                distributions.append({"partition": name, "split": split, "genotype": genotype, "plants": len(group)})
    write_json(destination / "plant_splits.json", {"independent_unit": "plant_uid", "official_retained": True, "fold_seed": protocol["fold_seed"], "assignments": plans, "oof": "each plant held out once; val fold is cyclic successor; not unseen-genotype validation"})
    pd.DataFrame(distributions).to_csv(destination / "split_genotype_distribution.csv", index=False)
    return plans
