"""Equal-target standardized Huber loss with inverse plant-window weighting."""

import torch
from torch.nn import functional


def standardized_huber(predicted, targets, weights=None):
    losses = functional.huber_loss(predicted,targets,reduction="none",delta=1).mean(dim=1)
    if weights is not None:
        losses = losses * weights
    return losses.mean()
