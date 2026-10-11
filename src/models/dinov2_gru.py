"""B2b: chronological frozen RGB embeddings, one GRU layer and two-layer head."""

import torch
from torch import nn


class DINOv2GRU(nn.Module):
    def __init__(self, feature_dim=384, time_dim=10, hidden_dim=256, gru_layers=1, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.gru = nn.GRU(feature_dim,hidden_dim,num_layers=gru_layers,batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim+time_dim,hidden_dim),nn.ReLU(),
                                  nn.Dropout(dropout),nn.Linear(hidden_dim,2))

    def forward(self, features, time_conditions):
        if features.ndim != 3 or features.shape[1:] != (3,self.feature_dim):
            raise ValueError("Expected three chronologically ordered RGB embeddings")
        if time_conditions.shape != (features.shape[0],self.time_dim):
            raise ValueError("Time condition dimensions disagree")
        _, hidden = self.gru(features)
        return self.head(torch.cat([hidden[-1],time_conditions],dim=1))
