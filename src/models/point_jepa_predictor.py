from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional

from models.point_jepa_encoder import PointJEPAEncoder, Transformer, gather_tokens, greedy_order


class PointJEPAPredictor(nn.Module):
    def __init__(self, encoder_dim=384, predictor_dim=192, predictor_depth=6,
                 predictor_heads=6, dropout=0.1, attention_dropout=0.05, drop_path_rate=0.25, **unused):
        super().__init__()
        self.predictor_embed = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.positional_encoding = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, predictor_dim))
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.predictor_proj = nn.Linear(predictor_dim, encoder_dim)
        self.predictor = Transformer(predictor_dim, predictor_depth, predictor_heads, dropout,
                                     attention_dropout, drop_path_rate, add_pos_every_layer=False,
                                     constant_drop_path=True)

    def forward(self, context_tokens, context_centers, target_centers):
        context = self.predictor_embed(context_tokens)
        target_positions = self.positional_encoding(target_centers) + self.mask_token
        positions = torch.cat((self.positional_encoding(context_centers), target_positions), dim=1)
        empty = context.new_zeros((len(context), target_centers.shape[1], context.shape[-1]))
        predicted = self.predictor(torch.cat((context, empty), dim=1), positions)
        return self.predictor_proj(predicted[:, context.shape[1]:])


def sample_masks(group_count, device, target_ratio=(0.15, 0.2), context_ratio=(0.4, 0.75), blocks=4):
    ratio = float(torch.empty((), device=device).uniform_(*target_ratio))
    target_size = max(1, int(group_count * ratio))
    if target_size >= group_count:
        raise ValueError("Target mask leaves no legal context")
    for _ in range(128):
        starts = torch.randint(group_count - target_size + 1, (blocks,), device=device)
        targets = starts[:, None] + torch.arange(target_size, device=device)[None]
        valid = torch.ones(group_count, dtype=torch.bool, device=device)
        valid[targets.flatten()] = False
        available = valid.nonzero(as_tuple=False).flatten()
        if len(available):
            fraction = float(torch.empty((), device=device).uniform_(*context_ratio))
            count = max(1, int(len(available) * fraction))
            start = int(torch.randint(len(available) - count + 1, (), device=device))
            return available[start:start + count], targets
    raise RuntimeError("Could not sample disjoint nonempty context; no illegal fallback")


class PointJEPA(nn.Module):
    def __init__(self, model_config, ssl_config):
        super().__init__()
        self.encoder = PointJEPAEncoder(**model_config)
        self.predictor = PointJEPAPredictor(**model_config)
        self.target_encoder = copy.deepcopy(self.encoder.student)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.ssl_config = ssl_config
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def reset_target(self):
        self.target_encoder.load_state_dict(self.encoder.student.state_dict(), strict=True)

    @torch.no_grad()
    def update_target(self, momentum):
        if not 0 <= momentum <= 1:
            raise ValueError("Invalid EMA momentum")
        source = self.encoder.student.state_dict()
        for name, value in self.target_encoder.state_dict().items():
            if value.is_floating_point():
                value.mul_(momentum).add_(source[name], alpha=1 - momentum)
            else:
                value.copy_(source[name])
        self.ema_updates.add_(1)

    def forward(self, points, center_indices=None, neighbors=None, masks=None):
        tokens, centers = self.encoder.tokenize(points, center_indices, neighbors)
        order = greedy_order(centers)
        tokens, centers = gather_tokens(tokens, order), gather_tokens(centers, order)
        if masks is None:
            context_indices, targets = sample_masks(tokens.shape[1], tokens.device,
                                                     self.ssl_config["target_ratio"], self.ssl_config["context_ratio"],
                                                     self.ssl_config["target_blocks"])
        else:
            context_indices, targets = masks
        if not len(context_indices) or torch.isin(context_indices, targets).any():
            raise ValueError("Context/target overlap or empty context")
        positions = self.encoder.positional_encoding(centers)
        with torch.no_grad():
            target_features = functional.layer_norm(self.target_encoder(tokens.detach(), positions.detach(), raw_final=True),
                                                     (tokens.shape[-1],))
        context = self.encoder.encode_tokens(tokens[:, context_indices], centers[:, context_indices])
        predictions = torch.stack([self.predictor(context, centers[:, context_indices], centers[:, indices]) for indices in targets])
        desired = torch.stack([target_features[:, indices] for indices in targets])
        loss = functional.smooth_l1_loss(predictions, desired, beta=self.ssl_config["loss_beta"])
        return {"loss": loss, "prediction": predictions, "target": desired,
                "context_indices": context_indices, "target_indices": targets,
                "tokens": target_features, "centers": centers}
