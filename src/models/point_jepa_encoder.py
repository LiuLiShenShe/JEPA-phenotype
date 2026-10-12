from __future__ import annotations

import torch
from torch import nn


def gather_tokens(values, indices):
    return values[torch.arange(len(values), device=values.device)[:, None], indices]


@torch.no_grad()
def patch_indices(points, num_groups=64, group_size=32):
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < max(num_groups, group_size):
        raise ValueError("Expected finite batch x N x XYZ with N >= patch sizes")
    if not torch.isfinite(points).all():
        raise ValueError("Non-finite XYZ")
    batch_size, point_count, _ = points.shape
    minimum = torch.full((batch_size, point_count), torch.inf, device=points.device)
    selected = points.square().sum(dim=-1).argmax(dim=-1)
    batch_indices = torch.arange(batch_size, device=points.device)
    centers = []
    for _ in range(num_groups):
        centers.append(selected)
        squared = (points - points[batch_indices, selected][:, None]).square().sum(dim=-1)
        minimum = torch.minimum(minimum, squared)
        selected = minimum.argmax(dim=-1)
    center_indices = torch.stack(centers, dim=1)
    coordinates = gather_tokens(points, center_indices)
    distances = torch.cdist(coordinates, points, compute_mode="donot_use_mm_for_euclid_dist")
    neighbors = distances.topk(group_size, dim=-1, largest=False, sorted=True).indices
    return center_indices, neighbors


@torch.no_grad()
def greedy_order(centers):
    batch_size, group_count, _ = centers.shape
    distances = torch.cdist(centers, centers, compute_mode="donot_use_mm_for_euclid_dist")
    selected = centers.sum(dim=-1).argmin(dim=-1)
    visited = torch.zeros((batch_size, group_count), dtype=torch.bool, device=centers.device)
    batch_indices = torch.arange(batch_size, device=centers.device)
    order = []
    for _ in range(group_count):
        order.append(selected)
        visited[batch_indices, selected] = True
        selected = distances[batch_indices, selected].masked_fill(visited, torch.inf).argmin(dim=-1)
    return torch.stack(order, dim=1)


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, values):
        if not self.training or not self.probability:
            return values
        keep = 1.0 - self.probability
        shape = (len(values),) + (1,) * (values.ndim - 1)
        mask = (keep + torch.rand(shape, device=values.device, dtype=values.dtype)).floor()
        return values * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, values):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(values)))))


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, attention_dropout):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, values):
        batch_size, token_count, dimension = values.shape
        packed = self.qkv(values).reshape(batch_size, token_count, 3, self.num_heads, dimension // self.num_heads)
        query, key, value = packed.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        result = (self.attn_drop(attention) @ value).transpose(1, 2).reshape(batch_size, token_count, dimension)
        return self.proj_drop(self.proj(result))


class Block(nn.Module):
    def __init__(self, dim, heads, dropout, attention_dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, attention_dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dropout)

    def forward(self, values):
        values = values + self.drop_path(self.attn(self.norm1(values)))
        return values + self.drop_path(self.mlp(self.norm2(values)))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout=0.0, attention_dropout=0.0,
                 drop_path=0.0, add_pos_every_layer=True, constant_drop_path=False):
        super().__init__()
        rates = [drop_path] * depth if constant_drop_path else torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, attention_dropout, rate) for rate in rates])
        self.norm = nn.LayerNorm(dim)
        self.add_pos_at_every_layer = add_pos_every_layer

    def forward(self, values, positions, raw_final=False):
        if not self.add_pos_at_every_layer:
            values = values + positions
        for block in self.blocks:
            if self.add_pos_at_every_layer:
                values = values + positions
            values = block(values)
        return values if raw_final else self.norm(values)


class PatchPointNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.first_conv = nn.Sequential(nn.Conv1d(3, 128, 1, bias=False), nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(nn.Conv1d(512, 512, 1, bias=False), nn.BatchNorm1d(512),
                                         nn.ReLU(inplace=True), nn.Conv1d(512, dim, 1))

    def forward(self, points):
        features = self.first_conv(points.transpose(1, 2))
        maximum = features.max(dim=2, keepdim=True).values.expand(-1, -1, features.shape[-1])
        return self.second_conv(torch.cat((maximum, features), dim=1)).max(dim=2).values


class PatchEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = PatchPointNet(dim)

    def forward(self, groups):
        batch_size, group_count, group_size, _ = groups.shape
        return self.encoder(groups.reshape(batch_size * group_count, group_size, 3)).reshape(batch_size, group_count, -1)


class Tokenizer(nn.Module):
    def __init__(self, dim, num_groups, group_size):
        super().__init__()
        self.embedding = PatchEmbedding(dim)
        self.num_groups, self.group_size = num_groups, group_size

    def forward(self, points, center_indices=None, neighbors=None):
        if center_indices is None or neighbors is None:
            center_indices, neighbors = patch_indices(points, self.num_groups, self.group_size)
        centers = gather_tokens(points, center_indices)
        batch_indices = torch.arange(len(points), device=points.device)[:, None, None]
        groups = points[batch_indices, neighbors] - centers[:, :, None]
        return self.embedding(groups), centers


class PointJEPAEncoder(nn.Module):
    def __init__(self, num_groups=64, group_size=32, encoder_dim=384, encoder_depth=12,
                 encoder_heads=6, dropout=0.1, attention_dropout=0.05,
                 drop_path_rate=0.25, **unused):
        super().__init__()
        self.feature_dim = encoder_dim
        self.tokenizer = Tokenizer(encoder_dim, num_groups, group_size)
        self.positional_encoding = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, encoder_dim))
        self.student = Transformer(encoder_dim, encoder_depth, encoder_heads, dropout,
                                   attention_dropout, drop_path_rate, add_pos_every_layer=True)

    def tokenize(self, points, center_indices=None, neighbors=None):
        return self.tokenizer(points, center_indices, neighbors)

    def encode_tokens(self, tokens, centers):
        return self.student(tokens, self.positional_encoding(centers))

    def forward(self, points, center_indices=None, neighbors=None, return_tokens=False):
        tokens, centers = self.tokenize(points, center_indices, neighbors)
        tokens = self.encode_tokens(tokens, centers)
        pooled = torch.cat((tokens.mean(dim=1), tokens.max(dim=1).values), dim=-1)
        return (pooled, tokens, centers) if return_tokens else pooled


def load_encoder_checkpoint(encoder, path, report_path=None):
    from m4.common import write_json

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint.get("encoder", checkpoint.get("model", checkpoint)))
    required = encoder.state_dict()
    mapped, unexpected, mismatched = {}, [], []
    for name, value in source.items():
        candidate = name.removeprefix("module.").removeprefix("encoder.")
        if candidate not in required:
            unexpected.append(name)
        elif not isinstance(value, torch.Tensor) or value.shape != required[candidate].shape:
            mismatched.append({"name": name, "source_shape": list(value.shape) if isinstance(value, torch.Tensor) else None,
                               "required_shape": list(required[candidate].shape)})
        else:
            mapped[candidate] = value
    missing = sorted(set(required) - set(mapped))
    parameter_sizes = {name: value.numel() for name, value in encoder.named_parameters()}
    coverage = sum(parameter_sizes.get(name, 0) for name in mapped) / sum(parameter_sizes.values())
    report = {"path": str(path), "parameter_coverage": coverage, "state_tensor_coverage": len(mapped) / len(required),
              "loaded_keys": sorted(mapped), "missing_keys": missing, "shape_mismatches": mismatched,
              "ignored_non_encoder_keys": unexpected, "strict_encoder_match": not missing and not mismatched}
    if report_path is not None:
        write_json(report_path, report)
    if missing or mismatched:
        raise ValueError(f"Checkpoint is not a strict encoder match (parameter coverage {coverage:.3%}); see compatibility report")
    encoder.load_state_dict(mapped, strict=True)
    return report
