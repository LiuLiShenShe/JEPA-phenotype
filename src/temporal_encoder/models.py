from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class PointEncoder(nn.Module):
    def __init__(self, dimensions=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, dimensions), nn.ReLU())

    def forward(self, points):
        encoded = self.layers(points)
        return encoded.amax(dim=-2)


class BirthEncoder(nn.Module):
    def __init__(self, feature_dimensions, temporal=True):
        super().__init__()
        self.temporal = temporal
        self.point_encoder = PointEncoder(64)
        if temporal:
            self.gru = nn.GRU(66, 64, batch_first=True)
        self.features = nn.Sequential(nn.Linear(feature_dimensions, 64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, patches, timing, features):
        if self.temporal:
            encoded = self.point_encoder(patches)
            sequence = torch.cat([encoded, timing.to(dtype=encoded.dtype)], dim=-1)
            unused_output, hidden = self.gru(sequence)
            representation = hidden[-1]
        else:
            representation = self.point_encoder(patches[:, -1])
        return self.classifier(torch.cat([representation, self.features(features)], dim=-1)).squeeze(-1)


class PairMatcher(nn.Module):
    def __init__(self, dimensions, metric=False):
        super().__init__()
        self.metric = metric
        if metric:
            self.point_encoder = PointEncoder(32)
            self.projection = nn.Linear(32, 32)
        self.classifier = nn.Sequential(nn.Linear(dimensions + (2 if metric else 0), 96), nn.ReLU(),
                                        nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))

    def embedding(self, points):
        return functional.normalize(self.projection(self.point_encoder(points)), dim=-1)

    def forward(self, features, previous_points=None, current_points=None):
        distance = features.new_zeros(features.shape[0])
        if self.metric:
            previous = self.embedding(previous_points)
            current = self.embedding(current_points)
            distance = torch.linalg.vector_norm(previous - current, dim=-1)
            cosine = (previous * current).sum(dim=-1)
            features = torch.cat([features, distance[:, None], cosine[:, None]], dim=-1)
        return self.classifier(features).squeeze(-1), distance
