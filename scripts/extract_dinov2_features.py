#!/usr/bin/env python3
"""Extract immutable, RGB-only official frozen DINOv2 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.common import file_hash, object_hash, require_device, seed_everything, write_json


PREPROCESSING = {
    "version":"m1_letterbox_bicubic_224_v1", "size":224,
    "resize":"preserve aspect ratio; round each dimension; bicubic PIL",
    "padding":"symmetric RGB (124,116,104), approximately ImageNet mean",
    "crop":False, "mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225],
    "output":"float32 normalized CHW", "augmentation":False,
}
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
EXPECTED_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def preprocess(image: Image.Image):
    image = image.convert("RGB")
    scale = 224 / max(image.size)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224,224), (124,116,104))
    canvas.paste(resized, ((224-resized.width)//2, (224-resized.height)//2))
    values = np.asarray(canvas, dtype=np.float32) / 255
    values = (values - np.asarray(PREPROCESSING["mean"], dtype=np.float32)) / np.asarray(PREPROCESSING["std"], dtype=np.float32)
    return torch.from_numpy(values.transpose(2,0,1).copy())


def load_encoder(root: Path, device):
    source = root / "third_party/dinov2"
    weights_path = root / "outputs/m1/pretrained/dinov2_vits14_pretrain.pth"
    if file_hash(weights_path) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Official DINOv2 weight identity mismatch")
    encoder = torch.hub.load(str(source), "dinov2_vits14", source="local", pretrained=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False)
    encoder.eval().to(device)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Encoder unexpectedly has trainable parameters")
    return encoder


class RGBImages(Dataset):
    def __init__(self, root, paths):
        self.root = root
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, position):
        with Image.open(self.root / self.paths[position]) as image:
            tensor = preprocess(image)
        return position, tensor


def extract(args):
    seed_everything(42)
    torch.set_num_threads(4)
    device = require_device(args.device)
    frames = pd.read_csv(ROOT / "data/metadata/frames.csv")
    frames = frames.loc[frames.quality_status.eq("valid")].sort_values("rgb_path").reset_index(drop=True)
    audit = pd.read_csv(ROOT / "data/metadata/file_manifest.csv").set_index("rgb_path")
    weights_hash = file_hash(ROOT / "outputs/m1/pretrained/dinov2_vits14_pretrain.pth")
    source_commit = (ROOT / "reports/m1/docs/dinov2_commit.txt").read_text().strip()
    signature = {"model":"dinov2_vits14", "weights_sha256":weights_hash,"source_commit":source_commit,
                 "preprocessing":PREPROCESSING, "frames_sha256":file_hash(ROOT / "data/metadata/frames.csv")}
    directory = ROOT / "outputs/m1/features" / object_hash(signature)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "metadata.json").exists():
        raise FileExistsError("Refusing to overwrite an existing cache; validate/reuse it instead")
    start = time.perf_counter()
    encoder = load_encoder(ROOT, device)
    with torch.inference_mode():
        probe = encoder(torch.zeros(2,3,224,224,device=device))
    if tuple(probe.shape) != (2,384) or probe.device != device:
        raise RuntimeError(f"Unexpected real encoder output: {probe.shape}, {probe.device}")
    matrix = np.lib.format.open_memmap(directory / "features.npy", mode="w+", dtype="float32", shape=(len(frames),384))
    loader = DataLoader(RGBImages(ROOT, frames.rgb_path.tolist()), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=device.type == "cuda", shuffle=False)
    seen = np.zeros(len(frames),dtype=bool)
    with torch.inference_mode():
        for positions, images in tqdm(loader, desc="Frozen DINOv2 RGBs"):
            result = encoder(images.to(device, non_blocking=True)).float().cpu().numpy()
            if not np.isfinite(result).all():
                raise RuntimeError("NaN/Inf in encoder output")
            matrix[positions.numpy()] = result
            seen[positions.numpy()] = True
    if not seen.all():
        raise RuntimeError("Not all RGB rows were extracted")
    matrix.flush()
    manifest = frames[["rgb_path","plant_uid","genotype","frame_idx","image_width","image_height"]].copy()
    manifest.insert(0,"feature_row",np.arange(len(frames)))
    manifest["rgb_sha256"] = manifest.rgb_path.map(audit.rgb_sha256)
    manifest["feature_sha256"] = [hashlib.sha256(row.tobytes()).hexdigest() for row in matrix]
    manifest.to_csv(directory / "manifest.csv",index=False)
    rng = np.random.default_rng(42)
    samples = np.sort(rng.choice(len(frames),size=32,replace=False))
    with torch.inference_mode():
        repeated = encoder(torch.stack([RGBImages(ROOT, frames.rgb_path.tolist())[int(position)][1] for position in samples]).to(device)).cpu().numpy()
    max_difference = float(np.abs(repeated - matrix[samples]).max())
    if not np.allclose(repeated,matrix[samples],rtol=2e-5,atol=2e-5):
        raise RuntimeError(f"Image-to-feature re-extraction mismatch: {max_difference}")
    metadata = {**signature,"status":"complete","weight_source_url":WEIGHTS_URL,"feature_dimension":384,
                "preprocessing_sha256":object_hash(PREPROCESSING),"images":len(frames),"dtype":"float32",
                "features_sha256":file_hash(directory / "features.npy"),"manifest_sha256":file_hash(directory / "manifest.csv"),
                "minimum":float(matrix.min()),"maximum":float(matrix.max()),"mean":float(matrix.mean()),"std":float(matrix.std()),
                "nonfinite_values":0,"reextraction_samples":samples.tolist(),"reextraction_max_abs_difference":max_difference,
                "device":str(device),"encoder_parameter_device":str(next(encoder.parameters()).device),
                "encoder_output_device":str(probe.device),"encoder_trainable_parameters":0,
                "encoder_parameters":sum(parameter.numel() for parameter in encoder.parameters()),
                "batch_size":args.batch_size,"workers":args.workers,"seconds":time.perf_counter()-start,
                "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device),"torch":torch.__version__,"pillow":Image.__version__}
    write_json(directory / "metadata.json", metadata)
    write_json(ROOT / "reports/m1/feature_cache.json", {"directory":str(directory.relative_to(ROOT)),**metadata})
    environment = {
        "python":sys.version,"executable":sys.executable,"platform":platform.platform(),"torch":torch.__version__,
        "torchvision":torchvision.__version__,"torch_cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
        "cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),
        "devices":[{"index":position,"name":torch.cuda.get_device_name(position),"total_memory":torch.cuda.get_device_properties(position).total_memory,
                    "tensor_test_sum":float((torch.ones(16,device=f"cuda:{position}") @ torch.ones(16,device=f"cuda:{position}")).item())} for position in range(torch.cuda.device_count())],
        "nvidia_smi":subprocess.check_output(["nvidia-smi"],text=True),"training_device":str(device),
        "actual_encoder_device":str(next(encoder.parameters()).device),"m0_environment_untouched":True,
    }
    write_json(ROOT / "reports/m1/environment.json", environment)
    print(json.dumps({"cache":str(directory),"seconds":metadata["seconds"],"dimension":384}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--workers",type=int,default=8)
    extract(parser.parse_args())
