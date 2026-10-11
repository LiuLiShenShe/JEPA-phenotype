#!/usr/bin/env python3
"""Run B0/B1/B1b on all three fixed horizons, fitting only train/val."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

from datasets.m1_feature_dataset import M1Index, TIME_NAMES
from models.growth_baseline import LinearGrowth, TimeOnly
from models.persistence import Persistence
from training.common import TrainStandardizer, file_hash, write_json
from training.metrics import evaluate, prediction_frame, selection_score


def main():
    index = M1Index(ROOT)
    for horizon in (2,4,8):
        start = time.perf_counter()
        train_selection = index.select("train",horizon)
        val_selection = index.select("val",horizon)
        train_rows = index.rows.iloc[train_selection]
        scale = TrainStandardizer.fit(index.targets[train_selection],train_rows).scale
        candidates = []
        fitted = []
        for alpha in (0.1,1.0,10.0,100.0):
            candidate = TimeOnly(alpha).fit(index.times[train_selection],index.targets[train_selection],train_rows)
            score = selection_score(index.rows.iloc[val_selection],index.targets[val_selection],candidate.predict(index.times[val_selection]),scale)
            fitted.append(candidate)
            candidates.append({"alpha":alpha,"val_selection_score":score})
        selected = int(np.argmin([candidate["val_selection_score"] for candidate in candidates]))
        time_model = fitted[selected]
        artifact_dir = ROOT / "outputs/m1/checkpoints" / f"B1b_h{horizon}"
        write_json(artifact_dir / "checkpoint.json",time_model.state())
        write_json(artifact_dir / "config.json",{"model":"B1b","horizon":horizon,"seed":42,"time_names":TIME_NAMES,
                   "degree":2,"solver":"float64_svd","alpha":time_model.alpha,"selection_split":"val","test_used_for_selection":False,
                   "candidates":candidates,"seconds":time.perf_counter()-start,"device":"cpu"})
        loaded = TimeOnly.from_state(json.loads((artifact_dir / "checkpoint.json").read_text()))
        if not np.allclose(loaded.predict(index.times[val_selection]),time_model.predict(index.times[val_selection]),rtol=0,atol=1e-7):
            raise RuntimeError("Time-only checkpoint reload differs")
        for model_name, model in (("B0",Persistence()),("B1",LinearGrowth()),("B1b",time_model)):
            if model_name != "B1b":
                model_dir = ROOT / "outputs/m1/checkpoints" / f"{model_name}_h{horizon}"
                write_json(model_dir / "config.json",{"model":model_name,"horizon":horizon,"seed":42,"condition":"Oracle-phenotype",
                                                       "device":"cpu","test_used_for_selection":False,"fit_required":False})
                write_json(model_dir / "checkpoint.json",{"model":model_name,"stateless":True,"version":1,"horizon":horizon})
            for split in ("val","test"):
                selection = index.select(split,horizon)
                rows = index.rows.iloc[selection]
                if model_name == "B1b":
                    predictions = model.predict(index.times[selection])
                else:
                    history = index.oracle_history(selection)
                    history_times = (index.timestamps[selection]-index.timestamps[selection,0,None])/np.timedelta64(1,"D")
                    predictions = model.predict(history=history,history_times=history_times,
                                                forecast_delta_days=rows.target_time_delta_seconds.to_numpy()/86400)
                targets = index.targets[selection]
                output_dir = ROOT / "outputs/m1/predictions" / f"{model_name}_h{horizon}"
                output_dir.mkdir(parents=True,exist_ok=True)
                prediction_frame(rows,targets,predictions,model=model_name,seed=42).to_csv(output_dir / f"{split}.csv",index=False)
                metrics, plants = evaluate(rows,targets,predictions)
                write_json(output_dir / f"{split}_metrics.json",metrics)
                plants.to_csv(output_dir / f"{split}_plants.csv",index=False)
            print(f"Completed {model_name} horizon {horizon}",flush=True)
    write_json(ROOT / "reports/m1/simple_baseline_selection.json",{
        "status":"locked_before_neural_test","test_used_for_selection":False,"selection_split":"val",
        "models":["B0","B1","B1b"],"horizons":[2,4,8],"time_only_train_fit":True,
        "split_sha256":file_hash(ROOT / "splits/phenoleaf_plant_split_seed42.json"),
    })


if __name__ == "__main__":
    main()
