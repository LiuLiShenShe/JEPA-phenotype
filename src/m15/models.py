"""Identical GRU capacity for genuine and repeated history, trained independently."""

import torch
from torch import nn

from models.dinov2_gru import DINOv2GRU
from models.dinov2_mlp import DINOv2MLP


class DifferenceMLP(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(384 * 5 + 10, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, features, time_conditions):
        differences = features[:, 1:] - features[:, :-1]
        design = torch.cat([features.flatten(1), differences.flatten(1), time_conditions], dim=1)
        return self.head(design)


class DiagnosticHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.strategy = config["strategy"]
        architecture = config["architecture"]
        if architecture == "diff":
            self.predictor = DifferenceMLP(config["hidden_dim"], config["dropout"])
        else:
            constructor = DINOv2GRU if architecture == "gru" else DINOv2MLP
            self.predictor = constructor(feature_dim=384, time_dim=10, hidden_dim=config["hidden_dim"], dropout=config["dropout"])

    def forward(self, features, time_conditions):
        if self.strategy == "repeat":
            features = features[:, -1:].expand(-1, 3, -1)
        elif self.strategy == "time":
            features = torch.zeros_like(features)
        return self.predictor(features, time_conditions)


def model_spec(name, protocol):
    training = protocol["training"]
    prefix, suffix = name.split("_", 1)
    architecture = "mlp" if prefix == "time" else prefix
    condition = "full" if suffix == "repeat" else suffix
    strategy = "repeat" if suffix == "repeat" else ("time" if prefix == "time" else "history")
    is_gru = architecture == "gru"
    return {
        "model": name, "architecture": architecture, "strategy": strategy, "condition": condition,
        "hidden_dim": training["difference_hidden_dim"] if architecture == "diff" else training["hidden_dim"],
        "dropout": training["dropout"], "learning_rate": training["gru_learning_rate"] if is_gru else training["mlp_learning_rate"],
        "batch_size": training["gru_batch_size"] if is_gru else training["mlp_batch_size"],
        "weight_decay": training["weight_decay"], "max_epochs": training["epochs"],
        "gradient_clip": training["gradient_clip"], "early_stopping": False,
    }
