from __future__ import annotations

import csv
import json

from PIL import Image, ImageDraw

from datasets.phenoleaf_utils import OFFICIAL_PALETTE, compute_mask_phenotype
from build_phenotypes import build_phenotypes


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_exact_palette_labels_and_area_ratio(workflow):
    phenotypes = read_csv(workflow["metadata_dir"] / "phenotypes.csv")
    instances = read_csv(workflow["metadata_dir"] / "leaf_instances.csv")
    assert len(phenotypes) == 3 * 4 * 14
    assert len(instances) > len(phenotypes)
    first = next(row for row in phenotypes if row["frame_idx"] == "0")
    assert int(first["leaf_count"]) == 4
    assert int(first["invalid_color_pixel_count"]) == 0
    ratio = float(first["projected_area_ratio"])
    assert 0.0 <= ratio <= 1.0
    assert int(first["projected_area_px"]) == int(first["known_projected_area_px"])
    assert json.loads(first["invalid_colors"]) == []
    assert {row["growth_stage"] for row in phenotypes} >= {"early", "intermediate", "mature"}


def test_invalid_mask_color_is_recorded_without_being_counted_as_a_leaf(tmp_path):
    mask = Image.new("RGB", (8, 8), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((1, 1, 2, 2), fill=OFFICIAL_PALETTE[0])
    draw.point((7, 7), fill=(1, 2, 3))
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path)
    phenotype = compute_mask_phenotype(mask_path, OFFICIAL_PALETTE)
    assert phenotype["leaf_count"] == 1
    assert phenotype["invalid_color_pixel_count"] == 1
    assert phenotype["projected_area_px"] == phenotype["known_projected_area_px"] + 1
    assert "invalid_palette_colors" in phenotype["anomaly_flags"]
    assert phenotype["invalid_color_regions"][0]["first_pixel_xy"] == [7, 7]


def test_workbook_string_rgb_layout_matches_official_palette(tmp_path):
    from openpyxl import Workbook
    from datasets.phenoleaf_utils import load_palette

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["Colour", "RGB", "Leaf number"])
    for leaf_id, color in enumerate(OFFICIAL_PALETTE, 1):
        worksheet.append([None, str(color), leaf_id])
    path = tmp_path / "palette.xlsx"
    workbook.save(path)
    colors, source, issues = load_palette(path)
    assert colors == list(OFFICIAL_PALETTE)
    assert source == "xlsx" and issues == []


def test_phenotype_pipeline_is_byte_reproducible(workflow):
    paths = [workflow["metadata_dir"] / name for name in ("frames.csv", "plants.csv", "phenotypes.csv", "leaf_instances.csv", "data_issues.csv", "dataset_statistics.json")]
    before = [path.read_bytes() for path in paths]
    build_phenotypes(
        workflow["metadata_dir"] / "frames.csv", workflow["metadata_dir"],
        project_root=workflow["root"],
        palette_path=workflow["root"] / "data/leaf_dataset_colour/leaves_unique_colours.xlsx",
    )
    assert before == [path.read_bytes() for path in paths]


def test_exact_instances_match_official_extractor(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/PhenoLeaf-TS"))
    from phenoleaf_ts.data.prepare import extract_instances_from_mask

    mask = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((1, 1, 3, 3), fill=OFFICIAL_PALETTE[3])
    draw.point((7, 7), fill=OFFICIAL_PALETTE[15])
    path = tmp_path / "mask.png"
    mask.save(path)
    phenotype = compute_mask_phenotype(path, OFFICIAL_PALETTE)
    expected = extract_instances_from_mask(path)
    assert {item["leaf_id"]: item["area_px"] for item in phenotype["instances"]} == {item["leaf_id"]: item["area"] for item in expected}
