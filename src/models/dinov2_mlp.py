"""B2a: last RGB embedding plus the common time query, no historical labels."""

import torch
from torch import nn


class DINOv2MLP(nn.Module):
    def __init__(self, feature_dim=384, time_dim=10, hidden_dim=256, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.head = nn.Sequential(nn.Linear(feature_dim+time_dim,hidden_dim),nn.ReLU(),
                                  nn.Dropout(dropout),nn.Linear(hidden_dim,2))

    def forward(self, features, time_conditions):
        if features.ndim != 3 or features.shape[1:] != (3,self.feature_dim):
            raise ValueError("Expected three RGB embedding slots")
        if time_conditions.shape != (features.shape[0],self.time_dim):
            raise ValueError("Time condition dimensions disagree")
        return self.head(torch.cat([features[:, -1],time_conditions],dim=1))
