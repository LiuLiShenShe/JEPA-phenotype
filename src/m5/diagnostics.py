from __future__ import annotations

import numpy as np
import torch


def feature_statistics(values):
    values = values.detach().double().cpu().reshape(-1, values.shape[-1])
    if not torch.isfinite(values).all() or len(values) < 2:
        raise ValueError("Invalid diagnostic feature matrix")
    centered = values - values.mean(0)
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total, 1e-30)
    rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())) if total > 1e-20 else 0.0
    diagonal = covariance.diagonal()
    off_diagonal = covariance - torch.diag(diagonal)
    mean = values.mean(0)
    return {"dimension": values.shape[-1], "samples": len(values), "std_mean": float(values.std(0, unbiased=False).mean()),
            "std_per_dimension": values.std(0, unbiased=False).tolist(), "variance_mean": float(diagonal.mean()),
            "variance_min": float(diagonal.min()), "variance_max": float(diagonal.max()),
            "covariance_trace": total, "covariance_offdiagonal_abs_mean": float(off_diagonal.abs().mean()),
            "covariance_offdiagonal_frobenius": float(off_diagonal.norm()), "effective_rank": rank,
            "participation_rank": total ** 2 / max(float(eigenvalues.square().sum()), 1e-30),
            "largest_eigenvalue_fraction": float(eigenvalues[-1] / max(total, 1e-30)),
            "covariance_eigenvalues": eigenvalues.flip(0).tolist(),
            "mean_energy_fraction": float(mean.square().sum() / values.square().sum(1).mean().clamp_min(1e-30)),
            "max_rank": min(len(values) - 1, values.shape[-1]),
            "numerical_rank_relative_1e_6": int((eigenvalues > max(float(eigenvalues[-1]) * 1e-6, 1e-20)).sum())}


def similarities(values, first, second):
    values = values.detach().float().cpu().reshape(-1, values.shape[-1])
    result = {}
    for name, features in (("uncentred", values), ("centred", values - values.mean(0))):
        normalized = torch.nn.functional.normalize(features, dim=-1, eps=1e-12)
        cosine = (normalized[first] * normalized[second]).sum(-1)
        result[name] = {"pairs": len(cosine), "mean": float(cosine.mean()),
                        "quantiles_0_05_50_95_100": torch.quantile(cosine, torch.tensor([0., .05, .5, .95, 1.])).tolist()}
    return result


def representation_diagnostics(tokens, pooled, plant_ids, seed=20261012):
    plant_ids = np.asarray(plant_ids)
    if len(tokens) != len(pooled) or len(tokens) != len(plant_ids):
        raise ValueError("Diagnostic identity/features mismatch")
    generator = np.random.default_rng(seed)
    first, second = np.triu_indices(len(pooled), 1)
    categories = {}
    for name, selection in (("different_plants", plant_ids[first] != plant_ids[second]),
                            ("same_plant_different_scans", plant_ids[first] == plant_ids[second])):
        available = np.flatnonzero(selection)
        if not len(available):
            continue
        chosen = generator.choice(available, min(8192, len(available)), replace=False)
        categories[name] = similarities(pooled, first[chosen], second[chosen])
    scans = generator.integers(len(tokens), size=8192)
    patch_first = generator.integers(tokens.shape[1], size=len(scans))
    patch_second = (patch_first + generator.integers(1, tokens.shape[1], size=len(scans))) % tokens.shape[1]
    categories["same_scan_different_patches"] = similarities(tokens, scans * tokens.shape[1] + patch_first,
                                                            scans * tokens.shape[1] + patch_second)
    token_first = generator.integers(len(tokens) * tokens.shape[1], size=8192)
    token_second = generator.integers(len(tokens) * tokens.shape[1], size=8192)
    categories["random_tokens"] = similarities(tokens, token_first, token_second)
    token_stats, pooled_stats = feature_statistics(tokens), feature_statistics(pooled)
    return {"token": token_stats, "pooled": pooled_stats, "similarities": categories,
            "independent_plants": len(set(plant_ids)), "scans": len(tokens), "diagnostic_split": "train",
            "hard_collapse": token_stats["std_mean"] < .01 and token_stats["effective_rank"] < 2,
            "interpretation": "cosine is not a collapse criterion; covariance eigenvalues are centred; repeated scans are technical observations"}


@torch.no_grad()
def diagnose_encoder(encoder, dataset, plant_ids, device, batch_size=64):
    encoder.eval()
    pooled, tokens = [], []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = {name: torch.stack([sample[name] for sample in samples]).to(device)
                 for name in ("points", "center_indices", "neighbors")}
        features, patches, _ = encoder(**batch, return_tokens=True)
        pooled.append(features.cpu())
        tokens.append(patches.cpu())
    return representation_diagnostics(torch.cat(tokens), torch.cat(pooled), plant_ids)


def degradation_stop(current, previous, initial, steps):
    if current["hard_collapse"]:
        return True
    if steps < 500:
        return False
    return all(current["pooled"][field] <= .5 * comparison["pooled"][field]
               for field in ("effective_rank", "variance_mean") for comparison in (previous, initial))
