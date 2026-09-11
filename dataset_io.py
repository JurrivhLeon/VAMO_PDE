"""Dataset split I/O helpers shared by PDE modules."""

from __future__ import annotations

import os
from typing import Dict

import torch


def save_dataset_splits(splits: Dict[str, Dict[str, torch.Tensor]], path: str) -> None:
    """Save precomputed train/val/test splits to disk as a single .pt file."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(splits, path)


def load_dataset_splits(path: str, map_location: str = "cpu") -> Dict[str, Dict[str, torch.Tensor]]:
    """Load precomputed train/val/test splits from disk."""
    splits = torch.load(path, map_location=map_location)
    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            raise ValueError(f"Missing split {split_name!r} in {path}")
        for key in ("f", "u0", "u_traj"):
            if key not in splits[split_name]:
                raise ValueError(f"Missing key {split_name}.{key} in {path}")
    return splits
