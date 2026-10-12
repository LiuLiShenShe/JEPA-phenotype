"""Small Oracle organ-state and first-observation event baselines."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def persistence(history):
    values = np.asarray(history, dtype=np.float64)
    return values[-1].copy()


def linear_growth(history, days, target_das):
    values, times = np.asarray(history, dtype=np.float64), np.asarray(days, dtype=np.float64)
    if len(values) != len(times) or len(times) < 2 or not (np.diff(times) > 0).all() or target_das <= times[-1]:
        raise ValueError("Linear trend requires strictly historical actual DAS")
    centered = times - times.mean()
    slope = (centered[:, None] * (values - values.mean(axis=0))).sum(axis=0) / np.square(centered).sum()
    return np.maximum(values.mean(axis=0) + slope * (target_das - times.mean()), 0)


class OrganStateMLP(nn.Module):
    def __init__(self, input_dim=5, time_dim=8, hidden_dim=64, output_dim=4, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim + time_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, sequence, time_conditions):
        return self.head(torch.cat((sequence[:, -1], time_conditions), dim=1))


class OrganStateGRU(nn.Module):
    def __init__(self, input_dim=5, time_dim=8, hidden_dim=64, output_dim=4, dropout=0.1, repeated_latest=False):
        super().__init__()
        self.repeated_latest = repeated_latest
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim + time_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, sequence, time_conditions):
        if self.repeated_latest:
            sequence = sequence[:, -1:].expand(-1, sequence.shape[1], -1)
        _, hidden = self.gru(sequence)
        return self.head(torch.cat((hidden[-1], time_conditions), dim=1))
