from __future__ import annotations

import csv
from collections import defaultdict

from prepare_metadata import scan_dataset


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_metadata_scans_all_plants_and_assigns_dense_order(workflow):
    frames = read_csv(workflow["metadata_dir"] / "frames.csv")
    plants = read_csv(workflow["metadata_dir"] / "plants.csv")
    assert len(frames) == 3 * 4 * 14
    assert len(plants) == 3 * 4
    assert len({row["genotype"] for row in frames}) == 3
    grouped = defaultdict(list)
    for row in frames:
        grouped[row["plant_uid"]].append(row)
        assert row["plant_uid"] == f"{row['genotype']}/{row['replicate']}"
        assert row["timestamp_reliable"] == "True"
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_idx"]))
        assert [int(row["frame_idx"]) for row in rows] == list(range(len(rows)))
        assert [row["timestamp"] for row in rows] == sorted(row["timestamp"] for row in rows)
        assert all(row["rgb_path"] and row["mask_path"] for row in rows)


def test_missing_mask_and_orphan_mask_are_not_silently_dropped(synthetic_project):
    raw_root = synthetic_project / "data/leaf_dataset_colour/01_raw_dataset"
    mask_root = synthetic_project / "data/leaf_dataset_colour/02_leaf_labels"
    path = sorted(raw_root.glob("*/*/*.png"))[0]
    mask = mask_root / path.relative_to(raw_root)
    mask.rename(mask.with_name("orphan.png"))
    output = synthetic_project / "data/metadata"
    stats = scan_dataset(raw_root.parent, output, project_root=synthetic_project)
    assert stats["counts"]["frame_rows"] == 168
    assert stats["issues"]["by_type"]["missing_mask"] == 1
    assert stats["issues"]["by_type"]["orphan_mask"] == 1
    assert sum(row["quality_status"] == "missing_mask" for row in read_csv(output / "frames.csv")) == 1


def test_corruption_and_dimension_mismatch_are_reported(synthetic_project):
    from PIL import Image

    raw_root = synthetic_project / "data/leaf_dataset_colour/01_raw_dataset"
    paths = sorted(raw_root.glob("*/*/*.png"))
    paths[0].write_bytes(b"not an image")
    mask = raw_root.parent / "02_leaf_labels" / paths[1].relative_to(raw_root)
    Image.new("RGB", (12, 9)).save(mask)
    output = synthetic_project / "data/metadata"
    stats = scan_dataset(raw_root.parent, output, project_root=synthetic_project)
    assert stats["issues"]["by_type"]["corrupt_rgb"] == 1
    assert stats["issues"]["by_type"]["dimension_mismatch"] == 1
    assert stats["counts"]["frame_rows"] == 168


def test_invalid_timestamp_quarantines_unverified_sequence(synthetic_project):
    raw_root = synthetic_project / "data/leaf_dataset_colour/01_raw_dataset"
    path = sorted(raw_root.glob("*/*/*.png"))[0]
    mask = raw_root.parent / "02_leaf_labels" / path.relative_to(raw_root)
    path.rename(path.with_name("unparseable.png"))
    mask.rename(mask.with_name("unparseable.png"))
    output = synthetic_project / "data/metadata"
    stats = scan_dataset(raw_root.parent, output, project_root=synthetic_project)
    assert stats["timestamp"]["unreliable_plants"] == 1
    assert sum(row["quality_status"] == "invalid_time_order" for row in read_csv(output / "frames.csv")) == 14


def test_duplicate_timestamps_are_an_error(synthetic_project):
    raw_root = synthetic_project / "data/leaf_dataset_colour/01_raw_dataset"
    path = sorted(raw_root.glob("*/*/*.png"))[0]
    copy = path.with_name(path.stem + "-duplicate.png")
    copy.write_bytes(path.read_bytes())
    mask_root = raw_root.parent / "02_leaf_labels"
    mask = mask_root / path.relative_to(raw_root)
    (mask_root / copy.relative_to(raw_root)).write_bytes(mask.read_bytes())
    output = synthetic_project / "data/metadata"
    stats = scan_dataset(raw_root.parent, output, project_root=synthetic_project)
    assert stats["issues"]["by_type"]["duplicate_timestamp"] == 1
    assert stats["counts"]["frame_rows"] == 169
