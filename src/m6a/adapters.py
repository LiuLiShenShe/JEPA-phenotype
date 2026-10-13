from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from m6a.common import normalize


class OfficialSegmenter:
    def __init__(self, root, name="PlantNet", device="cuda:0"):
        import torch

        self.torch = torch
        self.device = device
        self.base = root / f"data/raw/m6_a/official/plantnet/{name}/{name}_pytorch"
        for key in list(sys.modules):
            if key == "utils" or key.startswith("utils.") or key in ["model_pytorch", "torch_util"]:
                del sys.modules[key]
        sys.path.insert(0, str(self.base))
        sys.path.insert(0, str(self.base / "utils"))
        source = self.base / "models/model_pytorch.py"
        spec = importlib.util.spec_from_file_location("m6_official_model", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.plantnet_model(6).to(device).eval()
        checkpoint = torch.load(self.base / "models/checkpoints/model_epoch199.pth", map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.epoch = int(checkpoint["epoch"])
        self.mean_size = np.loadtxt(self.base / "models/mean_ins_size.txt")[:6]

    def predict(self, xyz, seed, official_normalization=False):
        from sklearn.cluster import MeanShift

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("Segmenter accepts XYZ only, never label-bearing arrays")
        generator = np.random.default_rng(seed)
        indices = generator.choice(len(xyz), min(4096, len(xyz)), replace=False)
        if official_normalization:
            sampled = np.asarray(xyz[indices], dtype=np.float32).copy()
            midpoint = (sampled.min(axis=0) + sampled.max(axis=0)) / 2
            sampled -= midpoint
            sampled /= max(float(sampled.max()), 1e-8)
        else:
            sampled = normalize(xyz)[indices]
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        with self.torch.inference_mode():
            semantic, embedding, unused_similarity = self.model(self.torch.from_numpy(sampled[None]).to(self.device))
            probabilities = self.torch.softmax(semantic, dim=2)[0].cpu().numpy()
            features = embedding[0].cpu().numpy()
        semantics = probabilities.argmax(axis=1)
        clusters = MeanShift(bandwidth=.6, bin_seeding=True, n_jobs=1).fit_predict(features)
        labels = np.zeros(len(indices), dtype=np.int32)
        next_label = 1
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            majority = int(np.bincount(semantics[mask], minlength=6).argmax())
            if majority in [1, 3, 5] and mask.sum() > .01 * self.mean_size[majority]:
                labels[mask] = next_label
                next_label += 1
        neighbor = cKDTree(xyz[indices]).query(xyz, workers=4)[1]
        return {"labels": labels[neighbor], "confidence": probabilities.max(axis=1)[neighbor],
                "sample_indices": indices, "sample_labels": labels, "sample_semantics": semantics,
                "semantic_probability": probabilities, "embedding": features}


class OfficialTracker:
    def __init__(self, root):
        base = root / "data/raw/m6_a/official/trackplant3d"
        spec = importlib.util.spec_from_file_location("m6_track_utils", base / "utils.py")
        self.utilities = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.utilities)
        sys.path.insert(0, str(base / "registration"))
        from pycpd import DeformableRegistration

        self.registration = DeformableRegistration

    def sample_indices(self, cloud, seed):
        import open3d as o3d

        sampled_cloud = np.asarray(cloud, dtype=np.float32)
        pointcloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(sampled_cloud))
        pointcloud.estimate_normals(max_nn=20)
        neighbor_distance = cKDTree(sampled_cloud).query(sampled_cloud, k=20, workers=4)[0][:, -1]
        radius = min(10., float(neighbor_distance.max()) * 1.001 + 1e-7)
        _, boundary = pointcloud.compute_boundary_points(radius=radius, max_nn=20, angle_threshold=90)
        mask = boundary.numpy().reshape(-1).astype(bool)
        generator = np.random.default_rng(seed)
        results = []
        for subset in [np.flatnonzero(~mask), np.flatnonzero(mask)]:
            if not len(subset):
                subset = np.arange(len(cloud))
            candidates = generator.choice(subset, 4096, replace=True)
            positions = cloud[candidates]
            farthest = int(generator.integers(len(candidates)))
            minimum_distance = np.full(len(candidates), np.inf)
            chosen = []
            for unused_index in range(128):
                chosen.append(candidates[farthest])
                minimum_distance = np.minimum(minimum_distance, ((positions - positions[farthest]) ** 2).sum(axis=1))
                farthest = int(minimum_distance.argmax())
            results.extend(chosen)
        return np.asarray(results, dtype=int), {"boundary_fraction": float(mask.mean()), "empty_partition_fallback": bool(mask.all() or (~mask).all()),
                                               "equivalent_neighbor_radius": radius}

    def deformation(self, previous, current):
        registration = self.registration(X=previous, Y=current, low_rank=False)
        transformed, unused_parameters = registration.register()
        return transformed, {"iterations": int(registration.iteration), "objective_difference": float(registration.diff)}

    def assign(self, previous_xyz, previous_labels, current_xyz, current_labels, next_track):
        previous_ids = np.unique(previous_labels[previous_labels >= 0])
        current_ids = np.unique(current_labels[current_labels >= 0])
        if not len(current_ids):
            return {}, {}, set(), next_track
        if not len(previous_ids):
            result = {int(label): next_track + offset for offset, label in enumerate(current_ids)}
            return result, {int(label): 0.0 for label in current_ids}, set(result), next_track + len(result)
        previous_compact = np.searchsorted(previous_ids, previous_labels)
        current_compact = np.searchsorted(current_ids, current_labels)
        cost = self.utilities.build_matrix(np.column_stack([previous_xyz, previous_compact]),
                                           np.column_stack([current_xyz, current_compact]))
        extended = self.utilities.extend_matrix(cost)
        unused_rows, columns = linear_sum_assignment(extended)
        assigned = {}
        confidence = {}
        born = set()
        for row, current_id in enumerate(current_ids):
            column = columns[row]
            if column < len(previous_ids):
                assigned[int(current_id)] = int(previous_ids[column])
                alternatives = [float(extended[row, len(previous_ids) + row] + extended[len(current_ids) + column, column])]
                alternatives.extend(float(value) for index, value in enumerate(cost[row]) if index != column)
                alternative = min(alternatives)
                confidence[int(current_id)] = float(np.clip((alternative - cost[row, column]) / max(alternative, 1e-8), 0, 1))
            else:
                assigned[int(current_id)] = next_track
                next_track += 1
                born.add(int(current_id))
                confidence[int(current_id)] = float(np.clip((cost[row].min() - extended[row, column]) / max(cost[row].min(), 1e-8), 0, 1))
        return assigned, confidence, born, next_track


def simple_assign(previous_xyz, previous_labels, current_xyz, current_labels, next_track, gate=.25):
    previous_ids = np.unique(previous_labels[previous_labels >= 0])
    current_ids = np.unique(current_labels[current_labels >= 0])
    if not len(current_ids):
        return {}, {}, set(), next_track
    if not len(previous_ids):
        assigned = {int(label): next_track + offset for offset, label in enumerate(current_ids)}
        return assigned, {int(label): 0.0 for label in current_ids}, set(assigned), next_track + len(assigned)
    previous_centroids = np.array([previous_xyz[previous_labels == label].mean(axis=0) for label in previous_ids])
    current_centroids = np.array([current_xyz[current_labels == label].mean(axis=0) for label in current_ids])
    distances = np.linalg.norm(current_centroids[:, None] - previous_centroids[None], axis=2)
    current_count, previous_count = distances.shape
    extended = np.full((current_count + previous_count, current_count + previous_count), 1000.0)
    extended[:current_count, :previous_count] = np.where(distances <= gate, distances, 1000.)
    extended[current_count:, previous_count:] = 0.0
    extended[np.arange(current_count), previous_count + np.arange(current_count)] = gate / 2
    extended[current_count + np.arange(previous_count), np.arange(previous_count)] = gate / 2
    unused_rows, columns = linear_sum_assignment(extended)
    assigned = {}
    confidence = {}
    born = set()
    for row, current_id in enumerate(current_ids):
        column = columns[row]
        if column < previous_count:
            assigned[int(current_id)] = int(previous_ids[column])
            alternatives = [gate] + [float(value) for index, value in enumerate(distances[row]) if index != column]
            alternate = min(alternatives)
            confidence[int(current_id)] = float(np.clip((alternate - distances[row, column]) / max(alternate, 1e-8), 0, 1))
        else:
            assigned[int(current_id)] = next_track
            next_track += 1
            born.add(int(current_id))
            confidence[int(current_id)] = float(np.clip((distances[row].min() - gate) / max(distances[row].min(), 1e-8), 0, 1))
    return assigned, confidence, born, next_track
