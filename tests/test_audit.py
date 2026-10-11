from __future__ import annotations

import csv
import shutil
import zipfile

import pytest

from audit_dataset import audit_dataset
from download_data import safe_extract
from validate_m0 import read_csv, validate_windows


def test_independent_full_audit_checks_every_mask_and_archive_member(workflow):
    data_root = workflow["root"] / "data"
    archive_path = data_root / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((data_root / "leaf_dataset_colour").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(data_root))
    report = audit_dataset(workflow["metadata_dir"], workflow["root"], archive_path, workflow["root"] / "reports/m0", 2)
    assert report["status"] == "PASS"
    assert report["frames_audited"] == report["masks_independently_recomputed"] == 168
    assert report["archive_crc_comparison_enabled"]
    assert report["archive_sha256"]
    assert report["extra_archive_file_errors"] == []
    assert report["cross_plant_duplicate_rgb_groups"] == 0


def test_independent_audit_rejects_forged_phenotype(workflow):
    path = workflow["metadata_dir"] / "frames.csv"
    rows = read_csv(path)
    rows[0]["leaf_count"] = "30"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = audit_dataset(workflow["metadata_dir"], workflow["root"], workflow["root"] / "missing.zip", workflow["root"] / "reports/m0", 2)
    assert report["status"] == "FAIL"
    assert "leaf_count_mismatch" in report["frame_errors"][0]["errors"]


def test_independent_audit_detects_cross_plant_duplicate_pixels(workflow):
    rows = read_csv(workflow["metadata_dir"] / "frames.csv")
    source = rows[0]
    destination = next(row for row in rows if row["plant_uid"] != source["plant_uid"])
    shutil.copyfile(workflow["root"] / source["rgb_path"], workflow["root"] / destination["rgb_path"])
    report = audit_dataset(workflow["metadata_dir"], workflow["root"], workflow["root"] / "missing.zip", workflow["root"] / "reports/m0", 2)
    assert report["status"] == "FAIL"
    assert report["cross_plant_duplicate_rgb_groups"] >= 1


def test_archive_extraction_rejects_traversal_and_preserves_existing_raw(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    destination = tmp_path / "output"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escaped.txt", b"unsafe")
    with pytest.raises(ValueError, match="Unsafe archive"):
        safe_extract(archive_path, destination)
    assert not (tmp_path / "escaped.txt").exists()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("raw.txt", b"abc")
    destination.mkdir(exist_ok=True)
    (destination / "raw.txt").write_bytes(b"original raw")
    with pytest.raises(ValueError, match="not overwriting"):
        safe_extract(archive_path, destination)
    assert (destination / "raw.txt").read_bytes() == b"original raw"
    (destination / "raw.txt").write_bytes(b"xyz")
    with pytest.raises(ValueError, match="not overwriting"):
        safe_extract(archive_path, destination)
    assert (destination / "raw.txt").read_bytes() == b"xyz"


def test_full_window_validator_rejects_cross_plant_history(workflow):
    rows = read_csv(workflow["windows_path"])
    frames = read_csv(workflow["metadata_dir"] / "frames.csv")
    assert all(check["status"] == "PASS" for check in validate_windows(workflow["windows_path"], workflow["split_path"], frames))
    import json

    paths = json.loads(rows[0]["input_image_paths"])
    paths[0] = next(frame["rgb_path"] for frame in frames if frame["plant_uid"] != rows[0]["plant_uid"])
    rows[0]["input_image_paths"] = json.dumps(paths)
    with workflow["windows_path"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert any(check["status"] == "FAIL" for check in validate_windows(workflow["windows_path"], workflow["split_path"], frames))
