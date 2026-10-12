"""XYZ-only geometry predictors and current-state PointNet diagnostics."""

from __future__ import annotations

import torch
from torch import nn

from models.pointnet_forecaster import PointNetForecaster


class CurrentPointNetRegressor(nn.Module):
    def __init__(self, geometry_dim=0, feature_dim=128, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.geometry_dim = geometry_dim
        self.pointnet = PointNetForecaster(feature_dim, hidden_dim, 7, dropout)
        if geometry_dim:
            self.pointnet.head = nn.Sequential(nn.Linear(feature_dim + geometry_dim + 7, hidden_dim), nn.ReLU(),
                                               nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def forward(self, points, geometry):
        zero_times = torch.zeros((len(points), 7), dtype=points.dtype, device=points.device)
        if self.geometry_dim:
            encoded = self.pointnet.encoder(points)
            return self.pointnet.head(torch.cat((encoded, geometry, zero_times), dim=1))
        return self.pointnet(points[:, None].expand(-1, 3, -1, -1), zero_times)


class GeometryResidualForecaster(nn.Module):
    def __init__(self, geometry_dim, time_dim=7, hidden_dim=64, historical=False, dropout=0.1):
        super().__init__()
        self.historical = historical
        self.head = nn.Sequential(nn.Linear(geometry_dim * (3 if historical else 1) + time_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))

    def forward(self, sequence, time_conditions):
        geometry = sequence.flatten(1) if self.historical else sequence[:, -1]
        return self.head(torch.cat((geometry, time_conditions), dim=1))
