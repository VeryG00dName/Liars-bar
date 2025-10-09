# src/training/weight_utils.py

"""Utilities for manipulating model weights for batched inference."""

from __future__ import annotations

from typing import Dict, List

import torch


def pack_weights(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a collection of state dicts into a batched weight dictionary."""

    if not state_dicts:
        raise ValueError("state_dicts must contain at least one entry")

    reference_keys = list(state_dicts[0].keys())
    packed: Dict[str, torch.Tensor] = {}

    for key in reference_keys:
        tensors = [sd[key] for sd in state_dicts]
        packed[key] = torch.stack(tensors, dim=0)

    return packed

