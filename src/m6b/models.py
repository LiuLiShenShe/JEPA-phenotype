from __future__ import annotations

import importlib.util
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from m6a.common import digest


def official_model(root, name="PlantNet", pretrained=False, device="cuda:0"):
    base = root / f"data/raw/m6_a/official/plantnet/{name}/{name}_pytorch"
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils.") or key in ["model_pytorch", "torch_util"]:
            del sys.modules[key]
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(base / "utils"))
    spec = importlib.util.spec_from_file_location("m6b_official_" + name, base / "models/model_pytorch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.plantnet_model(3).to(device)
    for component in model.modules():
        if isinstance(component, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.xavier_normal_(component.weight)
            if component.bias is not None:
                nn.init.zeros_(component.bias)
    state = model.state_dict()
    loaded = []
    random = list(state)
    checkpoint_hash = None
    if pretrained:
        checkpoint_path = base / "models/checkpoints/model_epoch199.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        source = checkpoint["model_state_dict"]
        compatible = {key: value for key, value in source.items() if key in state and value.shape == state[key].shape}
        incompatible = model.load_state_dict(compatible, strict=False)
        loaded = sorted(compatible)
        random = sorted(incompatible.missing_keys)
        checkpoint_hash = digest(checkpoint_path)
        head = "CONV5" if name == "PlantNet" else "CONV10"
        if set(random) != {head + ".weight", head + ".bias"}:
            raise ValueError("Unexpected parameter adaptation: " + str(random))
    return model, {"official_architecture": name, "source_commit": "5efe9ced49473f0d59c618b7e140ffeeb692d9b1",
                   "classes": 3, "embedding_dimensions": 5, "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "loaded_state_keys": loaded, "randomly_initialized_state_keys": random,
                   "frozen_parameters": [], "trainable_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
                   "pretrained_checkpoint_sha256": checkpoint_hash, "embedding_adaptation": "unchanged five-dimensional head; loaded for finetuning"}


def discriminative_leaf_loss(embedding, labels):
    components = []
    for features, instance in zip(embedding, labels):
        valid = instance >= 0
        features, instance = features[valid], instance[valid]
        ids = torch.unique(instance)
        if not len(ids):
            components.append(embedding.sum() * 0)
            continue
        means = torch.stack([features[instance == label].mean(dim=0) for label in ids])
        variance = torch.stack([functional.relu(torch.linalg.vector_norm(features[instance == label] - mean, ord=1, dim=-1) - .5).square().mean()
                                for label, mean in zip(ids, means)]).mean()
        if len(ids) > 1:
            distances = torch.cdist(means, means, p=1)
            off_diagonal = ~torch.eye(len(ids), device=means.device, dtype=torch.bool)
            separation = functional.relu(3 - distances[off_diagonal]).square().mean()
        else:
            separation = means.sum() * 0
        regularization = .001 * torch.linalg.vector_norm(means, ord=1, dim=-1).mean()
        components.append(variance + separation + regularization)
    return torch.stack(components).mean()


def official_joint_loss(semantic_logits, embedding, similarity, semantic, instances):
    semantic_loss = functional.cross_entropy(semantic_logits.transpose(1, 2), semantic)
    instance_loss = discriminative_leaf_loss(embedding, instances)
    valid = instances >= 0
    same_group = instances[:, :, None] == instances[:, None, :]
    weights = torch.zeros_like(instances, dtype=similarity.dtype)
    for frame_index, instance in enumerate(instances):
        frame_valid = valid[frame_index]
        unique, counts = torch.unique(instance[frame_valid], return_counts=True)
        for label, count in zip(unique, counts):
            weights[frame_index, instance == label] = 1 - count / frame_valid.sum().clamp_min(1)
    pair_weights = weights[:, :, None] * weights[:, None, :]
    double_hinge = 10 * (~same_group) * functional.relu(10 - similarity) + same_group * functional.relu(80 - similarity)
    similarity_loss = (double_hinge * pair_weights).mean()
    total = 10 * semantic_loss + 10 * instance_loss + similarity_loss
    return total, {"semantic": semantic_loss, "discriminative": instance_loss, "similarity": similarity_loss}


def smoke(root, device="cuda:0"):
    from m6a.common import write_json
    import time

    records = {}
    torch.set_num_threads(4)
    for name in ["PlantNet", "PSegNet"]:
        torch.manual_seed(17)
        model, loading = official_model(root, name, False, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        xyz = torch.randn(4, 4096, 3, device=device)
        semantic = torch.full((4, 4096), 2, dtype=torch.long, device=device)
        semantic[:, :512] = 1
        semantic[:, :128] = 0
        instances = torch.arange(4096, device=device)[None].repeat(4, 1) // 512
        instances[:, :512] = -1
        torch.cuda.reset_peak_memory_stats()
        times = []
        for iteration in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model(xyz)
            loss, components = official_joint_loss(*output, semantic, instances)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite smoke loss")
        pretrained, load = official_model(root, name, True, device)
        records[name] = {"forward_backward_optimizer_success": True, "batch": 4, "points": 4096,
                         "step_seconds": times, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                         "parameters": loading["parameters"], "finite_loss": float(loss.detach()),
                         "pretrained_random_keys": load["randomly_initialized_state_keys"]}
        del model, pretrained, optimizer
        torch.cuda.empty_cache()
        print(name, records[name], flush=True)
    write_json(root / "reports/m6_b/runtime_smoke.json", records)
