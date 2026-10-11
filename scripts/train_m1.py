#!/usr/bin/env python3
"""Tune on horizon-4 validation only, lock configuration, train 18 RGB-only heads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

from datasets.m1_feature_dataset import FeatureStore, M1FeatureDataset, M1Index, TIME_NAMES
from training.common import file_hash, write_json
from training.trainer import train_head


def main(args):
    index = M1Index(ROOT)
    cache_info = json.loads((ROOT / "reports/m1/feature_cache.json").read_text())
    store = FeatureStore(ROOT / cache_info["directory"],root=ROOT)
    locked_path = ROOT / "reports/m1/model_selection.json"
    if args.stage == "tune":
        if locked_path.exists():
            raise FileExistsError("Model selection is already locked; do not retune against test results")
        selections = {}
        candidate_results = []
        for model_name, config_name in (("B2a","dinov2_mlp"),("B2b","dinov2_gru")):
            base = yaml.safe_load((ROOT / f"configs/m1/{config_name}.yaml").read_text())
            train = M1FeatureDataset(index,store,"train",4)
            val = M1FeatureDataset(index,store,"val",4)
            candidates = [(0.001,256),(0.0003,256),(0.001,512)]
            resources = []
            for candidate_number,(learning_rate,batch_size) in enumerate(candidates):
                config = {**base,"learning_rate":learning_rate,"batch_size":batch_size,"seed":42,"horizon":4,
                          "feature_cache_signature":store.metadata["features_sha256"],"time_names":list(TIME_NAMES),"role":"validation_only_tuning"}
                directory = ROOT / "outputs/m1/tuning" / f"{model_name}_candidate{candidate_number}"
                if (directory / "resources.json").exists():
                    saved_config = json.loads((directory / "config.json").read_text())
                    if saved_config != config:
                        raise ValueError("Existing tuning run config disagrees")
                    resource = json.loads((directory / "resources.json").read_text())
                else:
                    resource = train_head(train,val,config,directory,device_name=args.device)
                resources.append(resource)
                candidate_results.append({"model":model_name,"candidate":candidate_number,"learning_rate":learning_rate,
                                          "batch_size":batch_size,"horizon":4,"selection_split":"val",**resource})
            winner = min(range(len(resources)),key=lambda position:resources[position]["val_selection_score"])
            selections[model_name] = {**base,"learning_rate":candidates[winner][0],"batch_size":candidates[winner][1],
                                     "feature_cache_signature":store.metadata["features_sha256"],"time_names":list(TIME_NAMES),
                                     "winner":winner,"tuning_horizon":4,"tuning_seed":42}
        write_json(locked_path,{"status":"locked","test_used_for_selection":False,"selection_split":"val",
                               "selection_criterion":"mean of train-scale normalized target plant-macro MAEs",
                               "candidates":candidate_results,"selected":selections,"horizons":[2,4,8],"seeds":[42,43,44],
                               "split_sha256":file_hash(ROOT / "splits/phenoleaf_plant_split_seed42.json")})
    else:
        locked = json.loads(locked_path.read_text())
        if locked["test_used_for_selection"] or locked["status"] != "locked":
            raise ValueError("Main experiments require validation-only immutable selections")
        for model_name,base in locked["selected"].items():
            for horizon in locked["horizons"]:
                train = M1FeatureDataset(index,store,"train",horizon)
                val = M1FeatureDataset(index,store,"val",horizon)
                for seed in locked["seeds"]:
                    config = {**base,"seed":seed,"horizon":horizon,"role":"main","selection_registry_sha256":file_hash(locked_path)}
                    directory = ROOT / "outputs/m1/checkpoints" / f"{model_name}_h{horizon}_s{seed}"
                    if (directory / "resources.json").exists():
                        saved_config = json.loads((directory / "config.json").read_text())
                        if saved_config != config:
                            raise ValueError("Existing main run config disagrees")
                        print(f"Reusing completed run {directory.name}",flush=True)
                        continue
                    train_head(train,val,config,directory,device_name=args.device)
        write_json(ROOT / "reports/m1/training_complete.json",{"status":"complete","models":["B2a","B2b"],"runs":18,
                   "horizons":[2,4,8],"seeds":[42,43,44],"selection_registry_sha256":file_hash(locked_path),"test_used_for_selection":False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["tune","main"],required=True)
    parser.add_argument("--device",default="cuda:0")
    main(parser.parse_args())
