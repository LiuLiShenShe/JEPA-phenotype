from __future__ import annotations

import torch
from torch import nn

from models.pointnet_forecaster import PointNetEncoder


class PointNetGRUForecaster(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=128, time_dim=7, dropout=0.1, repeated_latest=False):
        super().__init__()
        self.encoder = PointNetEncoder(feature_dim)
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.time_dim = time_dim
        self.repeated_latest = repeated_latest
        self.head = nn.Sequential(nn.Linear(hidden_dim + time_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def forward(self, points, time_conditions):
        if points.ndim != 4 or points.shape[1] != 3 or time_conditions.shape != (points.shape[0], self.time_dim):
            raise ValueError("Expected three history clouds and seven legal time conditions")
        if self.repeated_latest:
            features = self.encoder(points[:, -1]).unsqueeze(1).expand(-1, 3, -1)
        else:
            features = self.encoder(points.flatten(0, 1)).reshape(points.shape[0], 3, -1)
        _, hidden = self.gru(features)
        return self.head(torch.cat((hidden[-1], time_conditions), dim=1))
