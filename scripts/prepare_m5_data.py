#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from datasets.sugar4d_ssl_dataset import current_transform
from m3.common import PLY_TYPES
from m4.common import sha256, write_json
from models.point_jepa_encoder import patch_indices


def read_vertices(content):
    stream = io.BytesIO(content)
    if stream.readline().strip() != b"ply":
        raise ValueError("Invalid external PLY")
    properties, count, active, encoding = [], None, None, None
    for _ in range(512):
        fields = stream.readline().decode("ascii").strip().split()
        if fields[:1] == ["format"]:
            encoding = fields[1]
        elif fields[:1] == ["element"]:
            active = fields[1]
            if active == "vertex":
                count = int(fields[2])
        elif fields[:1] == ["property"] and active == "vertex":
            if fields[1] not in PLY_TYPES:
                raise ValueError("Unsupported external PLY property")
            properties.append((fields[2], PLY_TYPES[fields[1]]))
        elif fields == ["end_header"]:
            break
    if not count or not properties:
        raise ValueError("External PLY has no vertices")
    if encoding == "ascii":
        matrix = np.loadtxt(stream, max_rows=count, ndmin=2)
        return {name: matrix[:, index].astype(kind) for index, (name, kind) in enumerate(properties)}
    endian = "<" if encoding == "binary_little_endian" else ">"
    dtype = np.dtype([(name, endian + kind) for name, kind in properties])
    vertices = np.frombuffer(content, dtype=dtype, count=count, offset=stream.tell())
    return {name: vertices[name] for name, _ in properties}


def cache_geometry(xyz, destination, identity):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 64 or not np.isfinite(xyz).all():
        raise ValueError("Invalid external XYZ")
    normalized, center, radius = current_transform(xyz)
    quantized = np.rint(normalized.astype(np.float64) * 1e5).astype(np.int64)
    ordered = np.unique(quantized, axis=0)
    digest = hashlib.sha256(ordered.tobytes()).hexdigest()
    generator = np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little"))
    indices = generator.choice(len(xyz), 2048, replace=len(xyz) < 2048)
    points = normalized[indices].astype(np.float32)
    selected, neighbors = patch_indices(torch.from_numpy(points)[None], 64, 32)
    np.savez(destination, points=points, center_indices=selected[0].numpy(), neighbors=neighbors[0].numpy(),
             source_indices=indices, native_center=center, native_radius=np.array(radius), native_extent=np.ptp(xyz, axis=0))
    return {"points_raw": len(xyz), "geometry_sha256": digest, "native_radius": radius,
            "native_extent": json.dumps(np.ptp(xyz, axis=0).tolist()), "cache_sha256": sha256(destination)}


def cross_source_duplicates(root, frame):
    from m3.common import read_ply, xyz_from_vertices

    destination = root / "reports/m5/sugar4d_geometry_fingerprints.csv"
    if destination.exists():
        sugar = pd.read_csv(destination)
    else:
        metadata = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
        fingerprints = []
        for row in metadata.itertuples(index=False):
            vertices, _ = read_ply(root / row.pointcloud_path)
            normalized, _, _ = current_transform(xyz_from_vertices(vertices))
            ordered = np.unique(np.rint(normalized.astype(np.float64) * 1e5).astype(np.int64), axis=0)
            fingerprints.append({"plant_uid": row.plant_uid, "DAS": row.DAS,
                                 "geometry_sha256": hashlib.sha256(ordered.tobytes()).hexdigest()})
        sugar = pd.DataFrame(fingerprints)
        sugar.to_csv(destination, index=False)
    overlap = frame.merge(sugar, on="geometry_sha256", how="inner", suffixes=("_external", "_sugar"))
    overlap.to_csv(root / "reports/m5/cross_source_duplicates.csv", index=False)
    if len(overlap):
        frame = frame.loc[~frame.geometry_sha256.isin(overlap.geometry_sha256)].copy()
    return frame, len(overlap)


def unwrap_crops(root):
    folder = root / "data/raw/m5/crops3d"
    outer = folder / "Crops3D_article_v1.zip"
    with zipfile.ZipFile(outer) as archive:
        members = archive.namelist()
        if len(members) == 1 and members[0].lower().endswith(".zip"):
            destination = folder / "Crops3D.zip"
            if not destination.exists():
                with archive.open(members[0]) as source, destination.open("wb") as output:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
            return destination
    return outer


def crops(root, output):
    path = unwrap_crops(root)
    records, texts = [], {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        write_json(root / "reports/m5/sources/crops_archive_inventory.json", [{"name": value.filename, "bytes": value.file_size, "crc32": value.CRC} for value in archive.infolist()])
        for name in names:
            if name.lower().endswith(".txt") and "/Crops3D/" in "/" + name.replace("\\", "/"):
                texts[name] = archive.read(name).decode("utf-8-sig", errors="replace")
        write_json(root / "reports/m5/sources/crops_split_texts.json", texts)
        train_text = "\n".join(text for name, text in texts.items() if "train" in Path(name).name.lower())
        if not train_text:
            raise ValueError("Could not identify official Crops3D training list")
        train_paths = [line.strip().replace("\\", "/").removeprefix("./").removeprefix("data/") for line in train_text.splitlines() if line.strip()]
        selected = []
        for name in names:
            canonical = name.replace("\\", "/")
            if not canonical.lower().endswith(".ply") or "/Crops3D/" not in "/" + canonical:
                continue
            matches = [entry for entry in train_paths if canonical.endswith(entry) or canonical.endswith(entry + ".ply")]
            if matches:
                selected.append(name)
        if not selected:
            raise ValueError("No canonical Crops3D training PLY matched")
        for index, name in enumerate(sorted(selected)):
            species = Path(name).parent.name
            identity = "Crops3D:" + name
            vertices = read_vertices(archive.read(name))
            xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")])
            cache_file = f"crops_{index:05d}.npz"
            record = cache_geometry(xyz, output / cache_file, identity)
            record["source_crc32"] = archive.getinfo(name).CRC
            components = Path(name).stem.split("_")
            if species == "Cabbage" and len(components) == 3 and components[0] in ("mvs", "sl"):
                plant_uid = "Crops3D:Cabbage:plant" + components[-1]
                identity_status = "filename plant suffix across dates and mvs/sl modes; 14 physical plants in paper"
            else:
                plant_uid = f"Crops3D:{species}:identity-unresolved"
                identity_status = "scan identifier only; repeated physical plants possible; independent count unknown"
            records.append(record | {"source": "Crops3D", "species": species, "scan_uid": identity,
                                     "plant_uid": plant_uid, "identity_status": identity_status,
                                     "split": "train", "cache_file": cache_file, "source_file": name,
                                     "source_unit": "not explicitly documented; native units retained, dimensionless radius normalization",
                                     "annotation_assisted": "released authors' single-plant crop; no additional label-based crop in M5"})
            if (index + 1) % 50 == 0:
                print(f"CROPS CACHE {index + 1}/{len(selected)}", flush=True)
    return records


def beet(root, output):
    path = root / "data/raw/m5/bonnbeet/BonnBeetClouds3Dv0.zip"
    metadata = json.loads((root / "reports/m5/sources/beet_metadata.json").read_text())
    expected = metadata["data"]["latestVersion"]["files"][0]["dataFile"]["checksum"]["value"]
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("Bonn official MD5 mismatch")
    records = []
    with zipfile.ZipFile(path) as archive:
        write_json(root / "reports/m5/sources/beet_archive_inventory.json", [{"name": value.filename, "bytes": value.file_size, "crc32": value.CRC} for value in archive.infolist()])
        names = [name for name in archive.namelist() if name.lower().endswith(".ply") and "train" in Path(name).parts]
        groups = {}
        for index, name in enumerate(sorted(names)):
            vertices = read_vertices(archive.read(name))
            write_json(root / "reports/m5/sources/beet_field_schema.json", {"fields": list(vertices), "example": name}) if index == 0 else None
            plant_key = next((key for key in vertices if key.lower().replace("_", " ") in ("plant id", "plantid", "plant ids")), None)
            plot_key = next((key for key in vertices if key.lower().replace("_", " ") in ("plot id", "plotid", "plot")), None)
            if plant_key is None or plot_key is None:
                raise ValueError("Bonn plant/plot identifiers not established; cannot safely crop")
            xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")])
            identifiers = np.column_stack((vertices[plot_key], vertices[plant_key])).astype(np.int64)
            for plot, plant in np.unique(identifiers, axis=0):
                if plant < 0 or plot < 0:
                    continue
                key = (int(plot), int(plant))
                mask = (identifiers == np.array(key)).all(1)
                fragment = xyz[mask]
                if key not in groups or len(fragment) > len(groups[key][0]):
                    groups[key] = (fragment.copy(), name)
            if (index + 1) % 25 == 0:
                print(f"BEET READ {index + 1}/{len(names)}", flush=True)
        for index, (identity, (fragment, source_file)) in enumerate(sorted(groups.items())):
            xyz = np.unique(fragment.astype(np.float32), axis=0)
            if len(xyz) < 64:
                continue
            uid = f"BonnBeet:plot{identity[0]}:plant{identity[1]}"
            cache_file = f"beet_{index:05d}.npz"
            record = cache_geometry(xyz, output / cache_file, uid)
            records.append(record | {"source": "BonnBeetClouds3D", "species": "sugar_beet", "scan_uid": uid,
                                     "plant_uid": uid, "identity_status": "released plant_ids; plot field constant 99; one largest train patch fragment per ID, no multiple-ID copies",
                                     "split": "train", "cache_file": cache_file, "source_file": source_file,
                                     "source_unit": "metre", "annotation_assisted": "yes: plant_ids used for cropping; largest single-patch partial plant, no leaf/organ labels in SSL target"})
    return records


def prepare(root, include_beet=False, stage="all"):
    root = Path(root)
    output = root / "outputs/m5/data/external_xyz"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if stage == "beet":
        records = beet(root, output)
        pd.DataFrame(records).to_csv(root / "reports/m5/beet_data_manifest.csv", index=False)
        print(f"BEET PREPARED {len(records)} merged plants", flush=True)
        return
    if stage == "finalize":
        previous = pd.read_csv(root / "reports/m5/external_data_manifest.csv")
        records = previous.loc[previous.source.eq("Crops3D")].to_dict("records")
    else:
        records = crops(root, output)
    errors = []
    if include_beet:
        try:
            saved_beet = root / "reports/m5/beet_data_manifest.csv"
            records.extend(pd.read_csv(saved_beet).to_dict("records") if saved_beet.exists() else beet(root, output))
        except Exception as error:
            errors.append({"source": "BonnBeetClouds3D", "reason": str(error)})
            print(f"BONN EXCLUDED: {error}", flush=True)
    frame = pd.DataFrame(records)
    duplicates = frame.loc[frame.duplicated("geometry_sha256", keep=False)]
    duplicates.to_csv(root / "reports/m5/external_duplicates.csv", index=False)
    frame = frame.drop_duplicates("geometry_sha256").reset_index(drop=True)
    frame, cross_duplicates = cross_source_duplicates(root, frame)
    frame.to_csv(root / "reports/m5/external_data_manifest.csv", index=False)
    archives = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in (root / "data/raw/m5").glob("*/*.zip")]
    write_json(root / "reports/m5/data_preparation.json", {"status": "COMPLETED", "admitted_scans": len(frame),
               "sources": frame.groupby("source").size().to_dict(), "species": frame.groupby(["source", "species"]).size().reset_index(name="scans").to_dict("records"),
               "duplicate_scan_rows": len(duplicates), "errors": errors, "archives": archives,
               "cross_source_exact_geometry_duplicates": cross_duplicates,
               "crops_plants": "physical independent plant count unknown, not 1230", "validation_test_external_labels_used": False,
               "unit_policy": "all input geometry dimensionless current-cloud mean/radius; no invented native-to-metre conversion for Crops3D",
               "integrity": "ZIP read validates per-member CRC; Bonn published MD5 checked; Figshare API MD5 unavailable; SHA256 and byte size fixed"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--include-beet", action="store_true")
    parser.add_argument("--stage", choices=("all", "beet", "finalize"), default="all")
    arguments = parser.parse_args()
    prepare(arguments.root, arguments.include_beet, arguments.stage)
