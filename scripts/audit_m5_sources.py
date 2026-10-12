#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from m4.common import sha256, write_json
from m5.report import markdown_table


def audit(root):
    root = Path(root)
    report = root / "reports/m5"
    prepared = json.loads((report / "data_preparation.json").read_text())
    frame = pd.read_csv(report / "external_data_manifest.csv")
    if set(frame.source) != {"Crops3D", "BonnBeetClouds3D"} or len(frame) != 951:
        raise ValueError("Expected 823 Crops3D train scans and 128 Bonn train identities")
    if set(frame.split) != {"train"} or frame.geometry_sha256.duplicated().any():
        raise ValueError("Invalid external split or duplicated geometry")
    failures = []
    for row in frame.itertuples(index=False):
        if sha256(root / "outputs/m5/data/external_xyz" / row.cache_file) != row.cache_sha256:
            failures.append(row.cache_file)
    if failures:
        raise ValueError("Cached external geometry differs from audited hashes")
    fields = []
    for source, group in frame.groupby("source"):
        for species, selected in group.groupby("species"):
            fields.append({"source": source, "species": species, "ssl_objects": len(selected),
                           "independent_physical_plants": "14 identified across repeated scans" if species == "Cabbage" else
                           "128 released plant IDs; one fragment each" if source == "BonnBeetClouds3D" else "unknown; not equal to scan count",
                           "method": "structured light / MVS" if species == "Cabbage" else "SfM greenhouse" if species == "Tomato" else
                           "UAV camera + SfM" if source == "BonnBeetClouds3D" else "FARO Focus S70 TLS",
                           "training_probability": .25 if source == "BonnBeetClouds3D" else .25 / 8})
    inventory = pd.DataFrame(fields)
    inventory.to_csv(report / "source_sampling_plan.csv", index=False)
    metadata = json.loads((report / "sources/beet_metadata.json").read_text())["data"]["latestVersion"]
    checksum = hashlib.md5()
    with (root / "data/raw/m5/bonnbeet/BonnBeetClouds3Dv0.zip").open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum.update(block)
    expected = metadata["files"][0]["dataFile"]["md5"]
    if checksum.hexdigest() != expected:
        raise ValueError("Bonn official MD5 mismatch")
    with zipfile.ZipFile(root / "data/raw/m5/crops3d/Crops3D.zip") as archive:
        test = archive.testzip()
        if test is not None:
            raise ValueError(f"Crops3D corrupted member {test}")
    write_json(report / "source_integrity.json", {"status": "PASS", "admitted_scans": len(frame), "cache_hashes": len(frame),
               "crops_full_inner_zip_crc": "PASS", "crops_native_zip_bytes": 9289027713, "bonn_md5": checksum.hexdigest(),
               "bonn_md5_matches_official": True, "cross_source_exact_duplicates": prepared["cross_source_exact_geometry_duplicates"],
               "unresolved": ["Crops3D non-cabbage physical identities", "Crops3D explicit native unit", "Figshare official MD5 API unavailable"]})
    revision = json.loads((report / "sources/crops_revision.json").read_text())["sha"]
    text = f"""# M5 data source audit

Date:2026-10-12. Real external acquisition completed; no augmentation copies are
counted as new independent plants. Data-source identity/unit completeness is
**PARTIAL**, while published-byte/CRC/cache integrity checks are PASS.

## Official sources and fixed versions

- Crops3D: https://github.com/clawCa/Crops3D , fixed utility revision
  `{revision}`. Official data DOI https://doi.org/10.6084/m9.figshare.27313272 .
  Article https://www.nature.com/articles/s41597-024-04290-0 . DataCite confirms
  **CC0-1.0**, distinct from article copyright. Pin downloaded **article v1**;
  do not claim it is the latest unobserved release. Native `Crops3D.zip` exactly
  9,289,027,713 bytes matches DataCite. Outer download 9,289,027,995 bytes contains
  that file; outer is not another dataset. Full inner ZIP CRC pass, archive
  SHA256 in data_preparation.json. Metadata API repeatedly403, official article
  downloader200. Published Figshare MD5 could not be retrieved; no invented MD5.
- BonnBeetClouds3D: https://doi.org/10.60507/FK2/34W30T ; official Bonndata **V1.0**
  released2024-01-29, file`BonnBeetClouds3Dv0.zip`, CC0-1.0. Official MD5
  `{expected}` matches actual. 11,149,363,555 bytes; SHA256 fixed in preparation.
  The paper https://arxiv.org/abs/2312.14706 is currentlyv2(2025-08-11), not a
  licence/version change to the downloaded dataset.
- Sugar4D: originalV4/CC BY4.0/DOI10.60507/FK2/IS8YBZ is not redownloaded or
  relabelled.48 physical plants×16 scans. Original official/fold assignments and
  raw files are protected. Terrestrial LiDAR,2021-10-26—2021-12-21,Göttingen.

## Admitted geometry and physical identity

{markdown_table(inventory)}

Total external951objects=823Crops3D Train scans+128Bonn Train identity fragments.
The Crops3D archive has1180canonical raw single-plant PLY, plus50plot PLY; these
explain1230total samples without implying1230independent plants. Canonical Train
list has823/1180.10k/FPS copies,corruption HDF5,plot-scale IS and all externalTest
files are excluded from SSL; no test annotations are used for model selection.

**Cabbage:**14plants were scanned weekly2022-09-01—2022-12-05 with Wiiboox
Reeyee-Pro structured light and enhanced CasMVSNet MVS/Canon77D. Filename
`mvs/sl_date_plantID` lets M5 group date/mode scans of the same plant.137Train
scans remain technical observations; physical IDs are sampled uniformly before
choosing a scan. Other crops lack a verified identity schema; numeric filename
identifiers must not be called independently verified biological replicates.
Tomato:83raw/58Train samples; repeated2020-08—10greenhouse acquisitions in
Liaocheng,Shandong via SfM/3DPhenoMVS. Other6species useFARO Focus S70TLS; maize
multi-stage acquisition can revisit plants. Record unknown identities, rather
than fabricate a full independent-plant total.

**Bonn:**UAV imagery/SfM, real field breeding trial,2022-07,Germany.149Train
1m-overlapping patch files contain128releasedplant_ids, matching official Train
count. Stored`plot` is constant99, therefore is not a useful varying identity
disambiguator in this release. Use plant_ids as released identities and at most
one point cloud per ID. Initial cross-patch merging produced6m-spanning objects
because overlapping/local fragments were not universally safe to fuse; before
formal locking it was replaced with each ID's largest single-patch XYZ fragment.
Resulting radii0.181—0.444m. These are possibly partial plants, not guaranteed
whole-plant reconstructions, and no patch fragments are counted independently.

## Annotation-assisted preprocessing

Bonn single-plant extraction is explicitly **annotation-assisted preprocessing**:
plant_ids provide the crop; point count only picks the largestTrain fragment.
No leaf_ids,organ labels,keypoints,phenotypic_traits or future labels enter JEPA
loss, inputs or patch sampling. The data preparation is **not fully unsupervised**.
Crops3D single-plant crop/denoise/annotations were produced by authors; M5 uses
the released authors' single-plant objects without further label-based cropping,
still does not characterize the entire upstream pipeline as label-free.

## Overlap and preprocessing

Within/admitted exact geometry duplicate rows:{prepared['duplicate_scan_rows']}.
Cross-Sugar4D exact normalized geometry matches:{prepared['cross_source_exact_geometry_duplicates']}.
Hashes sort uniqueXYZ rounded at1e-5after translation/radius normalization;
this catches exact/quantized identical geometry and label-order variants, not
all similar or resampled objects. Sugar4D2021GermanLiDAR and Bonn2022UAV field
trial, Crops3D China's documented2020tomato/2022cabbage differ in acquisition;
no official known overlap is reported. Shared Bonn authors are not proof of
shared plants. Absence of exact hashes does not prove no near duplicates; we
disclose unknownphysicalIDs and never calculate fictitious independent totals.
SugarVal/TestXYZ is read only for this audit, not training or selection.

XYZ-only2048uniform samples after current-cloud mean/max-radius normalization;
deterministic64FPS/32KNN. Cache has points/indices/native transform only, no
semantic/instance labels. Sugar4D and Bonn coordinates are metres; Crops3D
explicit native units are unresolved. Recordnative radius/extent without
inventing conversion; normalizedinput is dimensionless and comparable, but
no cross-sourcephysical-scale claim is made. Sugar4D downstream lawful current
metric descriptors and train-only standardization are unchanged.

Source weightsSugar0.5,Crops0.25,Bonn0.25, external species uniform (Crops each
0.03125). Actual exposure CSV/JSON is recordedperrun; largeCrops species cannot
overwhelm small species. Known cabbageIDsbalanced; unknown otherphysicalIDs
are not masqueraded as resolved. Sources have no external label-based tuning.

## Retrieval audit and reproducibility

Bocha searchHTTP403quota exhausted. Official URL pages/DOI APIs were directly
retrieved, preserved in sources/; no successful search is claimed. Context7
officialPyTorch2.8documentation saved. Network requests include no credentials
in recorded queries/URLs. Crops Range attempt was rejected as200rather than206;
subsequent ordinary complete download and CRC validated the final bytes.
Bonn exactRange downloads were206and officialMD5 passed. Acquisition logs are
not scientific measurements. Unresolved license of GitHub utility code (no
LICENSE) means code is audited/read, not copied into the M5 implementation.

Preparation reproduction:
`.venv-m4/bin/python scripts/prepare_m5_data.py --include-beet`;
already prepared caches retain hashes in external_data_manifest.csv and the
formal protocol will reject changes. Raw data/checkpoints stay local and are
not git source artifacts. External manifests/source snapshots are retained.
"""
    (report / "DATA_SOURCE_AUDIT.md").write_text(text)
    print("M5 SOURCES AUDITED:951objects, units/physicalIDs partially unresolved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    arguments = parser.parse_args()
    audit(arguments.root)
