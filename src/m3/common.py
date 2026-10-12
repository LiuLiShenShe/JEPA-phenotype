from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


TARGETS = ("plant_height", "plant_width", "visible_organ_count")
UNITS = ("cm", "cm", "leaves")
TIME_NAMES = ("history_das_1", "history_das_2", "history_das_3", "history_gap_1", "history_gap_2", "forecast_delta", "target_das")


def write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
             "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


def read_ply(path):
    properties = []
    vertex_count = None
    active = None
    header = []
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY: {path}")
        for _ in range(512):
            raw = stream.readline()
            if not raw:
                raise ValueError("Truncated PLY header")
            line = raw.decode("ascii").strip()
            header.append(line)
            fields = line.split()
            if fields[:1] == ["format"]:
                encoding = fields[1]
            elif fields[:1] == ["element"]:
                active = fields[1]
                if active == "vertex":
                    vertex_count = int(fields[2])
                elif vertex_count is None:
                    raise ValueError("PLY vertices must be the first element")
            elif fields[:1] == ["property"] and active == "vertex":
                if fields[1] == "list" or fields[1] not in PLY_TYPES:
                    raise ValueError("Unsupported vertex property")
                properties.append((fields[2], PLY_TYPES[fields[1]]))
            elif line == "end_header":
                break
        else:
            raise ValueError("PLY header exceeds limit")
        offset = stream.tell()
    if not vertex_count or not properties:
        raise ValueError("Empty PLY vertex table")
    if encoding not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"Expected binary PLY, found {encoding}")
    endian = "<" if encoding == "binary_little_endian" else ">"
    dtype = np.dtype([(name, endian + kind) for name, kind in properties])
    if Path(path).stat().st_size < offset + vertex_count * dtype.itemsize:
        raise ValueError("Truncated PLY vertices")
    vertices = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(vertex_count,))
    return vertices, {"encoding": encoding, "vertex_count": vertex_count, "properties": [name for name, _ in properties], "header": header}


def xyz_from_vertices(vertices):
    return np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
