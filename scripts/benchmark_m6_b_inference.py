from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from m6a.common import digest, write_json
from m6a.experiment import metadata
from m6b.data import read_frame, transform
from m6b.evaluation import run_dir
from m6b.models import official_model


def main():
    torch.set_num_threads(4)
    rows = []
    records = metadata(ROOT).query("plant_uid == 'plant_001' and DAS in [37,79]")
    for method in ["S1", "S2", "S3"]:
        model, unused_loading = official_model(ROOT, "PSegNet" if method == "S3" else "PlantNet", False, "cuda:0")
        checkpoint_path = run_dir(ROOT, method, 17) / "best.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for record in records.itertuples(index=False):
            data = read_frame(ROOT / record.pointcloud_path)
            xyz, unused_center, unused_scale = transform(data["xyz"])
            seed = 20261012 + record.plant_id * 100 + record.DAS
            indices = np.random.default_rng(seed).choice(len(xyz), 4096, replace=False)
            tensor = torch.from_numpy(xyz[indices][None]).to("cuda:0")
            torch.manual_seed(seed)
            with torch.inference_mode():
                for unused_iteration in range(3):
                    model(tensor)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                repeated = []
                for unused_iteration in range(10):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    logits, embedding, unused_similarity = model(tensor)
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - started)
                    repeated.append((logits.detach().cpu().clone(), embedding.detach().cpu().clone()))
                identical = all(torch.equal(repeated[0][0], result[0]) and torch.equal(repeated[0][1], result[1]) for result in repeated[1:])
            rows.append({"method": method, "seed": 17, "plant_uid": record.plant_uid, "DAS": int(record.DAS),
                         "raw_points": len(xyz), "sample_points": 4096, "batch_size": 1, "warmup": 3, "repeats": 10,
                         "forward_seconds_median": float(np.median(elapsed)), "forward_seconds_mean": float(np.mean(elapsed)),
                         "frames_per_second_forward": float(1 / np.mean(elapsed)), "peak_allocated_vram_GB": torch.cuda.max_memory_allocated() / 1e9,
                         "exact_repeat_logits_embeddings": identical, "checkpoint_sha256": digest(checkpoint_path),
                         "scope": "forward-only same-device microbenchmark; excludes PLY IO, MeanShift and full-resolution nearest-neighbor",
                         "shared_host": "no exclusive GPU reservation; capture nvidia-smi separately"})
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / "reports/m6_b/inference_microbenchmark.csv", index=False)
    write_json(ROOT / "reports/m6_b/inference_repeatability.json", {"all_exact": all(row["exact_repeat_logits_embeddings"] for row in rows), "frames": rows})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
