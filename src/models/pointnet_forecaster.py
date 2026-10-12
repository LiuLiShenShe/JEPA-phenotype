from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.pointwise = nn.Sequential(nn.Conv1d(3, 32, 1), nn.BatchNorm1d(32), nn.ReLU(),
                                       nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                       nn.Conv1d(64, feature_dim, 1), nn.BatchNorm1d(feature_dim), nn.ReLU())
        self.projection = nn.Sequential(nn.Linear(2 * feature_dim, feature_dim), nn.ReLU())

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 1:
            raise ValueError("Expected batch x points x XYZ")
        features = self.pointwise(points.transpose(1, 2))
        pooled = torch.cat((features.amax(dim=2), features.mean(dim=2)), dim=1)
        return self.projection(pooled)


class PointNetForecaster(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=128, time_dim=7, dropout=0.1):
        super().__init__()
        self.encoder = PointNetEncoder(feature_dim)
        self.time_dim = time_dim
        self.head = nn.Sequential(nn.Linear(feature_dim + time_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def forward(self, points, time_conditions):
        if points.ndim != 4 or points.shape[1] != 3 or time_conditions.shape != (points.shape[0], self.time_dim):
            raise ValueError("Expected three history clouds and seven legal time conditions")
        return self.head(torch.cat((self.encoder(points[:, -1]), time_conditions), dim=1))
