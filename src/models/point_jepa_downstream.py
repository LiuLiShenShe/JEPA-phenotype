from __future__ import annotations

import torch
from torch import nn


def interpolate_tokens(points, centers, tokens):
    distances, indices = torch.cdist(points, centers, compute_mode="donot_use_mm_for_euclid_dist").topk(
        min(3, centers.shape[1]), dim=-1, largest=False, sorted=True)
    inverse = distances.clamp_min(1e-6).reciprocal()
    weights = inverse / inverse.sum(dim=-1, keepdim=True)
    batch_indices = torch.arange(len(points), device=points.device)[:, None, None]
    features = (tokens[batch_indices, indices] * weights[:, :, :, None]).sum(dim=2)
    offsets = points - centers[batch_indices[:, :, 0], indices[:, :, 0]]
    return torch.cat((features, points, offsets), dim=-1)


class PointJEPACount(nn.Module):
    def __init__(self, encoder, geometry_dim, frozen=False):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(2 * encoder.feature_dim + geometry_dim, 1)
        self.frozen = frozen
        if frozen:
            encoder.requires_grad_(False)
            encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, points, geometry, center_indices=None, neighbors=None):
        if self.frozen:
            with torch.no_grad():
                features = self.encoder(points, center_indices, neighbors)
        else:
            features = self.encoder(points, center_indices, neighbors)
        return self.head(torch.cat((features, geometry), dim=-1)).squeeze(-1)


class PointJEPASegmentation(nn.Module):
    def __init__(self, encoder, frozen=False):
        super().__init__()
        self.encoder, self.frozen = encoder, frozen
        self.head = nn.Sequential(nn.Linear(encoder.feature_dim + 6, 128), nn.GELU(), nn.Linear(128, 3))
        if frozen:
            encoder.requires_grad_(False)
            encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def point_features(self, points, center_indices=None, neighbors=None):
        _, tokens, centers = self.encoder(points, center_indices, neighbors, return_tokens=True)
        return interpolate_tokens(points, centers, tokens)

    def forward(self, points, center_indices=None, neighbors=None):
        if self.frozen:
            with torch.no_grad():
                features = self.point_features(points, center_indices, neighbors)
        else:
            features = self.point_features(points, center_indices, neighbors)
        return self.head(features)


class PointNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.head = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU(), nn.Conv1d(128, 3, 1))

    def forward(self, points, center_indices=None, neighbors=None):
        local = self.local(points.transpose(1, 2))
        maximum = local.max(dim=2, keepdim=True).values.expand_as(local)
        return self.head(torch.cat((local, maximum), dim=1)).transpose(1, 2)
