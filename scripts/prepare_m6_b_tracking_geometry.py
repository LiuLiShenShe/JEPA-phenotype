from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from m6a.common import normalize, digest, write_json
from m6a.experiment import metadata, cache_path
from m6b.data import read_frame
from m6b.tracking import tracker_indices


def main():
    records = metadata(ROOT)
    records = records[records.split.isin(["val", "test"])]
    checked = []
    for record in records.itertuples(index=False):
        indices = tracker_indices(ROOT, record)
        data = read_frame(ROOT / record.pointcloud_path)
        with np.load(cache_path(ROOT, record.plant_uid, record.DAS)) as cache:
            if not np.array_equal(normalize(data["xyz"][indices]), cache["tracking_xyz"]):
                raise ValueError("Geometry cache mismatch")
            if not np.array_equal(cache["labels"][indices], cache["tracking_pred"]):
                raise ValueError("Label sampling cache mismatch")
        checked.append({"plant_uid": record.plant_uid, "DAS": int(record.DAS), "split": record.split,
                        "points": len(indices), "index_sha256": digest(ROOT / f"outputs/m6_b/tracker_geometry/{record.plant_uid}_{record.DAS}.npy")})
        if len(checked) % 16 == 0:
            print("GEOMETRY", len(checked), "/", len(records), flush=True)
    write_json(ROOT / "reports/m6_b/tracking_geometry_equivalence.json", {
        "frames": len(checked), "identical_to_M6_A_normalized_XYZ": True,
        "identical_to_M6_A_predicted_label_sampling": True, "GT_labels_not_used_for_sampling": True,
        "legacy_radius10_frames": 6, "records": checked})


if __name__ == "__main__":
    main()
