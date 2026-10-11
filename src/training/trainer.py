"""Validation-only deterministic GPU head training; no test input accepted."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.dinov2_gru import DINOv2GRU
from models.dinov2_mlp import DINOv2MLP
from training.common import TrainStandardizer, clip_predictions, require_device, seed_everything, write_json
from training.losses import standardized_huber
from training.metrics import selection_score


def make_model(config):
    constructor = {"B2a":DINOv2MLP,"B2b":DINOv2GRU}[config["model"]]
    return constructor(feature_dim=config["feature_dim"],time_dim=config["time_dim"],
                       hidden_dim=config["hidden_dim"],dropout=config["dropout"],gru_layers=1)


def tensor_inputs(dataset, scaler, device):
    return {
        "features":torch.as_tensor(np.array(dataset.features[dataset.feature_indices],copy=True),device=device),
        "time_conditions":torch.as_tensor(scaler.transform(dataset.times),device=device),
    }


def model_inference(model, inputs, *, batch_size=2048):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0,len(inputs["features"]),batch_size):
            outputs.append(model(inputs["features"][start:start+batch_size],inputs["time_conditions"][start:start+batch_size]).cpu().numpy())
    return np.concatenate(outputs)


def train_head(train, val, config, destination, *, device_name="cuda:0"):
    if set(train.rows.split) != {"train"} or set(val.rows.split) != {"val"}:
        raise ValueError("Trainer only accepts train and validation plants, never test")
    if set(train.rows.plant_uid) & set(val.rows.plant_uid):
        raise ValueError("Training and validation plants overlap")
    destination = Path(destination)
    if (destination / "checkpoint.pt").exists():
        raise FileExistsError("Refusing to overwrite a completed experiment")
    destination.mkdir(parents=True,exist_ok=True)
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    device = require_device(device_name)
    target_scaler = TrainStandardizer.fit(train.targets,train.rows)
    time_scaler = TrainStandardizer.fit(train.times,train.rows)
    train_inputs = tensor_inputs(train,time_scaler,device)
    val_inputs = tensor_inputs(val,time_scaler,device)
    train_targets = torch.as_tensor(target_scaler.transform(train.targets),device=device)
    val_targets = torch.as_tensor(target_scaler.transform(val.targets),device=device)
    counts = train.rows.groupby("plant_uid").window_id.transform("count").to_numpy()
    weights = torch.as_tensor(len(train.rows)/(train.rows.plant_uid.nunique()*counts),dtype=torch.float32,device=device)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["max_epochs"],eta_min=config["learning_rate"]*0.1)
    best_score = float("inf")
    best_epoch = 0
    history = []
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    for epoch in range(1,config["max_epochs"]+1):
        model.train()
        positions = torch.randperm(len(train),generator=generator).to(device)
        epoch_loss = 0
        for start in range(0,len(train),config["batch_size"]):
            batch = positions[start:start+config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(train_inputs["features"][batch],train_inputs["time_conditions"][batch])
            loss = standardized_huber(outputs,train_targets[batch],weights[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            optimizer.step()
            epoch_loss += float(loss.detach())*len(batch)
        normalized_val = model_inference(model,val_inputs)
        predictions = clip_predictions(target_scaler.inverse(normalized_val))
        score = selection_score(val.rows,val.targets,predictions,target_scaler.scale)
        val_loss = float(standardized_huber(torch.from_numpy(normalized_val),val_targets.cpu()))
        history.append({"epoch":epoch,"train_weighted_huber":epoch_loss/len(train),"val_huber":val_loss,
                        "val_plant_macro_normalized_mae":score,"learning_rate":optimizer.param_groups[0]["lr"]})
        if score < best_score-1e-7:
            best_score = score
            best_epoch = epoch
            checkpoint = {
                "config":config,"model_state":{name:value.detach().cpu().clone() for name,value in model.state_dict().items()},
                "target_scaler":target_scaler.state(),"time_scaler":time_scaler.state(),
                "best_epoch":best_epoch,"val_selection_score":best_score,"selection_split":"val",
                "test_used_for_selection":False,"train_plants":sorted(train.rows.plant_uid.unique().tolist()),
                "val_plants":sorted(val.rows.plant_uid.unique().tolist()),"feature_cache_signature":config["feature_cache_signature"],
            }
            torch.save(checkpoint,destination / "checkpoint.pt")
        scheduler.step()
        if epoch-best_epoch >= config["patience"]:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    resource = {"seconds":time.perf_counter()-start_time,"epochs":epoch,"best_epoch":best_epoch,
                "val_selection_score":best_score,"device":str(device),"model_parameter_device":str(next(model.parameters()).device),
                "trainable_parameters":sum(parameter.numel() for parameter in model.parameters()),
                "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "selection_split":"val","test_used_for_selection":False}
    pd.DataFrame(history).to_csv(destination / "training_history.csv",index=False)
    write_json(destination / "config.json",config)
    write_json(destination / "resources.json",resource)
    print(json.dumps({"run":destination.name,**resource}),flush=True)
    return resource


def load_head(path, device_name="cpu"):
    device = require_device(device_name)
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint["selection_split"] != "val" or checkpoint["test_used_for_selection"]:
        raise ValueError("Invalid checkpoint selection provenance")
    model = make_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"],strict=True)
    model.eval()
    return model,checkpoint


def predict_checkpoint(path, dataset, device_name="cpu"):
    model,checkpoint = load_head(path,device_name)
    inputs = tensor_inputs(dataset,TrainStandardizer.from_state(checkpoint["time_scaler"]),next(model.parameters()).device)
    normalized = model_inference(model,inputs)
    return clip_predictions(TrainStandardizer.from_state(checkpoint["target_scaler"]).inverse(normalized))
