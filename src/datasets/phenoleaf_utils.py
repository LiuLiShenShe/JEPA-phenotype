"""Shared PhenoLeaf-TS parsing, palette, and phenotype utilities.

The official repository uses the same fixed 31-colour palette and exact RGB
matching.  This module keeps that behaviour explicit while adding validation
and structured issue reporting needed for plant-level temporal experiments.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


OFFICIAL_PALETTE: tuple[tuple[int, int, int], ...] = (
    (244, 64, 14),
    (48, 57, 249),
    (234, 250, 37),
    (24, 193, 65),
    (245, 130, 49),
    (231, 80, 219),
    (0, 182, 173),
    (115, 0, 218),
    (191, 239, 69),
    (255, 250, 200),
    (250, 190, 212),
    (66, 212, 244),
    (155, 99, 36),
    (220, 190, 255),
    (69, 158, 220),
    (255, 216, 177),
    (98, 2, 37),
    (227, 213, 12),
    (79, 159, 83),
    (170, 23, 101),
    (170, 255, 195),
    (169, 169, 169),
    (181, 111, 119),
    (144, 121, 171),
    (9, 125, 244),
    (184, 70, 30),
    (154, 35, 246),
    (229, 225, 238),
    (141, 254, 82),
    (31, 200, 209),
    (194, 217, 105),
)

BACKGROUND_RGB = (0, 0, 0)
TIMESTAMP_PATTERN = re.compile(
    r"^[^_]+_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})(?:[-_]|$)"
)


def json_dumps(value: Any) -> str:
    """Serialize a value consistently for CSV columns."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_timestamp(filename: str) -> tuple[datetime | None, str, str]:
    """Parse the timestamp format documented by the official repository.

    Official examples start with an acquisition id followed by
    ``YYYY_MM_DD_HH_MM_SS``.  The returned status is one of ``parsed``,
    ``invalid`` or ``missing``.
    """

    stem = Path(filename).stem
    match = TIMESTAMP_PATTERN.match(stem)
    if match is None:
        return None, "", "missing"
    try:
        value = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
        )
    except ValueError:
        return None, "", "invalid"
    return value, value.isoformat(timespec="seconds"), "parsed"


def load_palette(palette_path: str | Path | None) -> tuple[list[tuple[int, int, int]], str, list[str]]:
    """Load and validate the official palette workbook when it is available.

    The workbook layout is not assumed: rows containing three RGB values are
    accepted, and a leading leaf-id column is handled when present.  If the
    workbook is absent or malformed, the palette copied from the official
    ``prepare.py`` is used and the reason is returned as an issue.
    """

    issues: list[str] = []
    if palette_path is None:
        issues.append("palette_workbook_not_provided; using_official_prepare_py_palette")
        return list(OFFICIAL_PALETTE), "official_prepare_py", issues

    path = Path(palette_path)
    if not path.exists():
        issues.append(f"palette_workbook_missing:{path}; using_official_prepare_py_palette")
        return list(OFFICIAL_PALETTE), "official_prepare_py", issues

    try:
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        rows = []
        for worksheet in workbook.worksheets:
            rows.extend(worksheet.iter_rows(values_only=True))
        workbook.close()
    except Exception as exc:
        issues.append(f"palette_workbook_read_error:{type(exc).__name__}; using_official_prepare_py_palette")
        return list(OFFICIAL_PALETTE), "official_prepare_py", issues

    header_indices: dict[str, int] = {}
    for row in rows[:10]:
        for index, value in enumerate(row):
            if value is None:
                continue
            name = str(value).strip().lower()
            if name in {"r", "red"}:
                header_indices["r"] = index
            elif name in {"g", "green"}:
                header_indices["g"] = index
            elif name in {"b", "blue"}:
                header_indices["b"] = index
            elif name in {"rgb", "colour", "color", "colour rgb", "color rgb"}:
                header_indices["rgb"] = index
            elif "leaf number" in name or name in {"leaf_id", "leaf id", "id"}:
                header_indices["leaf_id"] = index

    parsed: list[tuple[int | None, tuple[int, int, int]]] = []
    for row in rows:
        values = list(row)
        candidate: tuple[int, int, int] | None = None
        leaf_id: int | None = None
        if "rgb" in header_indices and header_indices["rgb"] < len(values):
            raw_rgb = values[header_indices["rgb"]]
            if isinstance(raw_rgb, str):
                components = re.findall(r"\d+", raw_rgb)
                if len(components) == 3:
                    candidate = tuple(int(value) for value in components)  # type: ignore[assignment]
        if "leaf_id" in header_indices and header_indices["leaf_id"] < len(values):
            raw_leaf_id = values[header_indices["leaf_id"]]
            if isinstance(raw_leaf_id, (int, float)) and float(raw_leaf_id).is_integer():
                leaf_id = int(raw_leaf_id)
        if all(key in header_indices for key in ("r", "g", "b")):
            indices = [header_indices["r"], header_indices["g"], header_indices["b"]]
            if max(indices) < len(values):
                raw = [values[index] for index in indices]
                if all(isinstance(value, (int, float)) for value in raw):
                    candidate = tuple(int(value) for value in raw)  # type: ignore[assignment]
        numeric = [value for value in values if isinstance(value, (int, float))]
        if candidate is None and len(numeric) >= 3:
            rgb_values = numeric[-3:]
            candidate = tuple(int(value) for value in rgb_values)  # type: ignore[assignment]
            if len(numeric) >= 4 and float(numeric[0]).is_integer():
                leaf_id = int(numeric[0])
        if candidate is None:
            continue
        if not all(0 <= value <= 255 for value in candidate):
            continue
        parsed.append((leaf_id, candidate))

    if parsed:
        if any(leaf_id is not None for leaf_id, _ in parsed):
            parsed.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0))
        colors: list[tuple[int, int, int]] = []
        for _, color in parsed:
            if color not in colors:
                colors.append(color)
        if len(colors) == len(OFFICIAL_PALETTE):
            if colors != list(OFFICIAL_PALETTE):
                raise ValueError("Workbook palette/leaf-ID order disagrees with official prepare.py")
            return colors, "xlsx", issues
        issues.append(
            f"palette_workbook_expected_{len(OFFICIAL_PALETTE)}_colors_found_{len(colors)}; "
            "using_official_prepare_py_palette"
        )
    else:
        issues.append("palette_workbook_contains_no_rgb_rows; using_official_prepare_py_palette")
    return list(OFFICIAL_PALETTE), "official_prepare_py", issues


def inspect_image(path: str | Path) -> tuple[int, int, str]:
    """Verify an image and return width, height, and original mode."""

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.width, image.height, image.mode


def _rgb_array(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def growth_stage(leaf_count: int | None) -> tuple[int | None, str]:
    """Return the official leaf-count growth-stage label."""

    if leaf_count is None or leaf_count < 0:
        return None, "unknown"
    if leaf_count <= 6:
        return 0, "early"
    if leaf_count <= 10:
        return 1, "intermediate"
    return 2, "mature"


def compute_mask_phenotype(
    mask_path: str | Path,
    palette: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    """Extract exact-colour instances and plant-level phenotype measurements."""

    mask = _rgb_array(mask_path)
    height, width = mask.shape[:2]
    palette_array = np.asarray(palette, dtype=np.uint32)

    def encode_rgb(values: np.ndarray) -> np.ndarray:
        return (values[..., 0].astype(np.uint32) << 16) | (values[..., 1].astype(np.uint32) << 8) | values[..., 2].astype(np.uint32)

    mask_codes = encode_rgb(mask)
    palette_codes = (palette_array[:, 0] << 16) | (palette_array[:, 1] << 8) | palette_array[:, 2]
    order = np.argsort(palette_codes)
    sorted_codes = palette_codes[order]
    flat_codes = mask_codes.reshape(-1)
    insertion = np.searchsorted(sorted_codes, flat_codes, side="left")
    in_range = insertion < len(sorted_codes)
    known_flat = np.zeros(flat_codes.shape, dtype=bool)
    known_flat[in_range] = sorted_codes[insertion[in_range]] == flat_codes[in_range]
    label_flat = np.full(flat_codes.shape, -1, dtype=np.int16)
    label_flat[known_flat] = order[insertion[known_flat]].astype(np.int16) + 1
    label_image = label_flat.reshape(height, width)
    known_mask = known_flat.reshape(height, width)
    non_background = mask_codes != 0
    invalid_mask = non_background & ~known_mask
    invalid_codes = np.unique(mask_codes[invalid_mask])
    invalid_values = [
        (int(code >> 16) & 255, int(code >> 8) & 255, int(code) & 255)
        for code in invalid_codes
    ]
    invalid_color_regions = []
    for code, color in zip(invalid_codes, invalid_values):
        rows, columns = np.nonzero(mask_codes == code)
        invalid_color_regions.append(
            {
                "rgb": list(color),
                "pixel_count": int(len(rows)),
                "bbox_xyxy": [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())],
                "first_pixel_xy": [int(columns[0]), int(rows[0])],
            }
        )

    known_rows, known_columns = np.nonzero(known_mask)
    labels = label_image[known_mask]
    areas = np.bincount(labels, minlength=len(palette) + 1)
    min_rows = np.full(len(palette) + 1, height, dtype=np.int32)
    max_rows = np.full(len(palette) + 1, -1, dtype=np.int32)
    min_columns = np.full(len(palette) + 1, width, dtype=np.int32)
    max_columns = np.full(len(palette) + 1, -1, dtype=np.int32)
    if len(labels):
        np.minimum.at(min_rows, labels, known_rows)
        np.maximum.at(max_rows, labels, known_rows)
        np.minimum.at(min_columns, labels, known_columns)
        np.maximum.at(max_columns, labels, known_columns)
    instances: list[dict[str, Any]] = []
    for leaf_id, color in enumerate(palette, start=1):
        pixel_count = int(areas[leaf_id])
        if pixel_count == 0:
            continue
        instances.append(
            {
                "leaf_id": leaf_id,
                "colour_r": color[0],
                "colour_g": color[1],
                "colour_b": color[2],
                "area_px": pixel_count,
                "bbox_x": int(min_columns[leaf_id]),
                "bbox_y": int(min_rows[leaf_id]),
                "bbox_width": int(max_columns[leaf_id] - min_columns[leaf_id] + 1),
                "bbox_height": int(max_rows[leaf_id] - min_rows[leaf_id] + 1),
            }
        )

    invalid_pixel_count = int(invalid_mask.sum())
    leaf_count = len(instances)
    stage_id, stage_name = growth_stage(leaf_count)
    anomaly_flags: list[str] = []
    if leaf_count == 0:
        anomaly_flags.append("zero_leaf_count")
    if invalid_pixel_count:
        anomaly_flags.append("invalid_palette_colors")
    if int(non_background.sum()) == 0:
        anomaly_flags.append("zero_projected_area")
    if leaf_count > len(palette):
        anomaly_flags.append("leaf_count_exceeds_palette")

    return {
        "image_width": width,
        "image_height": height,
        "leaf_count": leaf_count,
        "projected_area_px": int(non_background.sum()),
        "known_projected_area_px": int(known_mask.sum()),
        "projected_area_ratio": float(non_background.mean()),
        "invalid_color_pixel_count": invalid_pixel_count,
        "invalid_colors": [list(color) for color in invalid_values],
        "invalid_color_regions": invalid_color_regions,
        "growth_stage_id": stage_id,
        "growth_stage": stage_name,
        "anomaly_flags": anomaly_flags,
        "instances": instances,
    }


def is_usable_quality_status(status: str) -> bool:
    """Return whether a frame may participate in split/window construction."""

    return status == "valid"


def resolve_project_path(path_value: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve a metadata path stored as absolute or project-relative."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    root = Path(project_root) if project_root is not None else Path.cwd()
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def serializable_path(path: str | Path, project_root: str | Path | None = None) -> str:
    """Store paths relative to the project when possible."""

    path_obj = Path(path).resolve()
    if project_root is not None:
        try:
            return path_obj.relative_to(Path(project_root).resolve()).as_posix()
        except ValueError:
            pass
    return str(path_obj)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write deterministic UTF-8 CSV output, including empty schema files."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, (list, tuple, dict)):
                    value = json_dumps(value)
                normalized[field] = value
            writer.writerow(normalized)
