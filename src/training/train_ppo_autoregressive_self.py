# src/training/train_ppo_autoregressive_self.py

import os, logging, warnings
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple, Sequence
from collections import Counter
from numpy.random import Generator
import random
import numpy as np
import argparse
# Quiet Torch compile logs
os.environ.pop("TORCH_LOGS", None)           # disable extra compile logs
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Deterministic cuBLAS workspace requirement for CUDA
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
# Hide symbolic_shapes warnings printed via warnings module (belt-and-suspenders)
warnings.filterwarnings("ignore", message=".*symbolic_shapes.*")
warnings.filterwarnings(
    "ignore",
    message=".*does not have a deterministic implementation.*",
    category=UserWarning,
)
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.amp as amp
import torch.nn.functional as F

from src.misc import lb
from src import config
from src.model.ppo_reactive_model import PPOReactiveModel
from src.agents.learner_ar_agent import LearnerAutoregressiveAgent
from src.training.vec_ppo_rollout import PPOVecRolloutManager
from src.training.tracing_utils import trace_model_from_checkpoint
from src.training.train_extras import (
    _collate_batch,
    _to_device_batch,
    ppo_losses_batched,
    set_seed
)
import src.training.train_extras as train_extras

def _silence_torch_symbolic_logs():
    for name in ("torch.fx.experimental.symbolic_shapes", "torch._dynamo.symbolic_shapes", "torch._dynamo", "torch._inductor"):
        logging.getLogger(name).setLevel(logging.ERROR)
_silence_torch_symbolic_logs()

SEED = int(getattr(config, "SEED", 42))
set_seed(SEED)
_GLOBAL_RNG = np.random.default_rng(SEED)

PAD_BUCKET_BOUNDARIES = [32, 64, 160, 256]


def _select_bucket_length(length: int) -> int:
    for boundary in PAD_BUCKET_BOUNDARIES:
        if length <= boundary:
            return boundary
    return int(length)

FORCE_CUDA_SYNC_FOR_TIMING = bool(getattr(config, "FORCE_CUDA_SYNC_FOR_TIMING", False))
USE_HELDOUT_AGENT = bool(getattr(config, "USE_HELDOUT_AGENT", True))


# ==============================================================================
# SECTION 1: HELPER CLASSES AND FUNCTIONS
# ==============================================================================

class OpponentPoolManager:
    """Manages the opponent_pool.json file for persistent population state."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.pool = self._load()

    def _load(self) -> List[Dict]:
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Pool file '{self.filepath}' not found. Initializing with base C++ bots.")

            # Initialize with base C++ bots using fixed labels 0..6
            base_bots = [
                {"name": "Classic", "type": "cpp_bot", "model_type": "cpp_bot", "label": 0, "path": None, "status": "active"},
                {"name": "GreedyCardSpammer", "type": "cpp_bot", "model_type": "cpp_bot", "label": 1, "path": None, "status": "active"},
                {"name": "RandomAgent", "type": "cpp_bot", "model_type": "cpp_bot", "label": 2, "path": None, "status": "active"},
                {"name": "SelectiveTableConservativeChallenger", "type": "cpp_bot", "model_type": "cpp_bot", "label": 3, "path": None, "status": "active"},
                {"name": "StrategicChallenger", "type": "cpp_bot", "model_type": "cpp_bot", "label": 4, "path": None, "status": "active"},
                {"name": "TableFirstConservativeChallenger", "type": "cpp_bot", "model_type": "cpp_bot", "label": 5, "path": None, "status": "active"},
                {"name": "TableNonTableAgent", "type": "cpp_bot", "model_type": "cpp_bot", "label": 6, "path": None, "status": "active"},
            ]

            self._save(base_bots)
            return base_bots
        except FileNotFoundError:
            data = []

        changed = False
        for entry in data:
            if isinstance(entry, dict) and "status" not in entry:
                entry["status"] = "active"
                changed = True
        if changed:
            self._save(data)
        return data

    def _save(self, pool_data: List[Dict]):
        with open(self.filepath, 'w') as f:
            json.dump(pool_data, f, indent=4)

    def save(self) -> None:
        self._save(self.pool)

    def add_agent(self, name: str, model_type: str, path: str, **kwargs):
        """
        Adds a new agent to the pool, assigning the next available label.
        Accepts additional keyword arguments to store as metadata.
        """
        # The check for existence should be based on the primary .pth path.
        if any(a.get('path') == path for a in self.pool if a.get('path')):
            print(f"Agent at path '{path}' already in pool. Skipping.")
            return

        existing_labels = {a['label'] for a in self.pool if a['type'] != 'cpp_bot'}
        next_label = 7
        while next_label in existing_labels:
            next_label += 1
        
        if next_label >= 64:
            print("Warning: Opponent pool has reached the maximum size of 64.")
            return

        # Create the new agent dictionary
        new_agent_entry = {
            "name": name,
            "type": "historical",
            "model_type": model_type,
            "label": next_label,
            "path": path,  # The primary .pth path
            "status": "active",
        }

        # Add any extra metadata passed in, like path_pt
        new_agent_entry.update(kwargs)

        self.pool.append(new_agent_entry)
        self._save(self.pool)
        print(f"Added '{name}' to pool with label {next_label}.")

    def set_status(self, label: int, status: str, *, save: bool = True) -> None:
        label_int = int(label)
        updated = False
        for entry in self.pool:
            if int(entry.get("label", -1)) == label_int:
                if entry.get("status") != status:
                    entry["status"] = status
                    updated = True
                break
        if updated and save:
            self._save(self.pool)

    def get_entries(self, *, status: Optional[str] = None, include_cpp: bool = True) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for entry in self.pool:
            if not include_cpp and entry.get("type") == "cpp_bot":
                continue
            if status is not None and entry.get("status", "active") != status:
                continue
            entries.append(entry)
        return entries

    def build_sampling_weights(
        self,
        pressure_scores: Dict[int, float],
        *,
        exclude_label: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        labels: List[int] = []
        weights: List[float] = []
        exclude = int(exclude_label) if exclude_label is not None else None
        for entry in self.pool:
            label = entry.get("label")
            if label is None:
                continue
            try:
                label_int = int(label)
            except Exception:
                continue
            if exclude is not None and label_int == exclude:
                continue
            if entry.get("status", "active") != "active":
                continue
            base_weight = entry.get("sampling_weight", 1.0)
            try:
                base = float(base_weight)
            except Exception:
                base = 1.0
            pressure = pressure_scores.get(label_int)
            if pressure is not None:
                base = pressure
            labels.append(label_int)
            weights.append(base)
        return labels, weights


class OpponentStatsManager:
    """Tracks opponent pressure statistics via ridge regression."""

    def __init__(
        self,
        pool_manager: OpponentPoolManager,
        ridge_alpha: float = 1.0,
        ema_decay: float = 0.9,
    ) -> None:
        self.ema_decay = float(min(max(ema_decay, 0.0), 1.0))
        self.pool_manager = pool_manager
        self.ridge_alpha = float(max(ridge_alpha, 1e-6))
        self.ema_alpha = float(getattr(config, "GRAD_FPRINT_EMA_ALPHA", 0.1))
        self.current_scores: Dict[int, float] = {}
        self.ema_scores: Dict[int, float] = {}
        self.intercept: float = 0.0
        self.ema_grad_fingerprints: Dict[int, np.ndarray] = {}
        self.fingerprint_norms: Dict[int, float] = {}
        self.total_coplay_steps: Counter[int] = Counter()

    def _active_labels(self) -> List[int]:
        labels: List[int] = []
        for entry in self.pool_manager.get_entries(status="active", include_cpp=True):
            label = entry.get("label")
            if label is None:
                continue
            try:
                labels.append(int(label))
            except Exception:
                continue
        labels = sorted(set(labels))
        return labels

    def update_pressure_scores(
        self,
        opponent_lineups: Sequence[Tuple[int, ...]],
        targets: Sequence[float],
        *,
        sample_weights: Optional[Sequence[float]] = None,
        self_play_counts: Optional[Sequence[int]] = None,
    ) -> None:
        if not opponent_lineups or not targets:
            return

        active_labels = self._active_labels()
        if not active_labels:
            return

        label_to_index = {label: idx for idx, label in enumerate(active_labels)}
        num_features = len(active_labels) + 2  # opponents + self-play count + bias

        features: List[List[float]] = []
        y_vals: List[float] = []
        weights: List[float] = []

        for idx, lineup in enumerate(opponent_lineups):
            if idx >= len(targets):
                break
            lineup_labels = [int(l) for l in lineup if l is not None and int(l) >= 0]
            if not lineup_labels:
                continue
            row = [0.0 for _ in range(num_features)]
            for lab in lineup_labels:
                if lab in label_to_index:
                    row[label_to_index[lab]] += 1.0
            if all(v == 0.0 for v in row[:-2]):
                continue
            if self_play_counts is not None and idx < len(self_play_counts):
                row[-2] = float(self_play_counts[idx])
            else:
                row[-2] = 0.0
            row[-1] = 1.0  # bias term
            features.append(row)
            y_vals.append(float(targets[idx]))
            if sample_weights is not None and idx < len(sample_weights):
                weights.append(float(max(sample_weights[idx], 1e-6)))
            else:
                weights.append(1.0)

        if not features:
            return

        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(y_vals, dtype=np.float32).reshape(-1, 1)
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)

        if X.shape[0] <= X.shape[1]:
            ridge = self.ridge_alpha * np.eye(X.shape[1], dtype=np.float32)
        else:
            ridge = self.ridge_alpha * np.eye(X.shape[1], dtype=np.float32)

        weighted_X = X * w
        XtX = weighted_X.T @ X
        XtY = weighted_X.T @ y
        try:
            coef = np.linalg.solve(XtX + ridge, XtY)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(XtX + ridge, XtY, rcond=None)

        coef = coef.reshape(-1)
        self.intercept = float(coef[-1]) if coef.size > 0 else 0.0

        updated_scores: Dict[int, float] = {}
        for label, idx in label_to_index.items():
            updated_scores[label] = float(coef[idx])

        self.current_scores = updated_scores
        for label, value in updated_scores.items():
            prev = self.ema_scores.get(label, value)
            self.ema_scores[label] = prev * self.ema_decay + value * (1.0 - self.ema_decay)

    def get_pressure_scores(self, *, use_ema: bool = True) -> Dict[int, float]:
        if use_ema and self.ema_scores:
            return dict(self.ema_scores)
        return dict(self.current_scores)

    def update_fingerprints(self, batch_fingerprints: Dict[int, Tuple[torch.Tensor, int]]) -> None:
        if not batch_fingerprints:
            return

        ema_steps = float(getattr(config, "GRAD_FINGERPRINT_EMA_STEPS", 5000.0))
        ema_steps = max(ema_steps, 1.0)

        for label, (vector, step_count) in batch_fingerprints.items():
            if vector is None or step_count is None:
                continue

            steps_int = int(step_count)
            if steps_int <= 0:
                continue

            vec_tensor = vector.detach().to(torch.float32)
            if vec_tensor.numel() == 0:
                continue

            vec_norm = torch.linalg.norm(vec_tensor)
            if not torch.isfinite(vec_norm) or vec_norm.item() <= 0.0:
                continue

            normalized_vec = (vec_tensor / vec_norm.clamp_min(1e-8)).cpu()

            self.total_coplay_steps[label] += steps_int
            
            prev = self.ema_grad_fingerprints.get(label)
            if prev is not None and prev.size == normalized_vec.numel():
                prev_vec = torch.from_numpy(prev.astype(np.float32))
                # Standard EMA update
                blended = (1.0 - self.ema_alpha) * prev_vec + self.ema_alpha * normalized_vec
            else:
                # First time seeing this opponent, just take the new value
                blended = normalized_vec
            blended_norm = torch.linalg.norm(blended)
            if not torch.isfinite(blended_norm) or blended_norm.item() <= 0.0:
                self.fingerprint_norms[label] = 0.0
                continue

            self.fingerprint_norms[label] = float(blended_norm.item())
            blended = blended / blended_norm
            self.ema_grad_fingerprints[label] = blended.to(torch.float16).cpu().numpy()


def _compute_fingerprints_for_update(
    batch_cpu: Dict[str, Any],
    batch_gpu: Dict[str, Any],
    model_outs: Tuple[torch.Tensor, ...],
    projection_matrix: torch.Tensor,
) -> Dict[int, Tuple[torch.Tensor, int]]:
    if projection_matrix is None:
        return {}

    if len(model_outs) < 4:
        return {}

    action_logits = model_outs[0]
    state_values = model_outs[2].squeeze(-1).to(torch.float32)
    transformer_output = model_outs[-1]
    if transformer_output.dim() != 3:
        return {}

    our_idx = batch_gpu["our_idx"].long()
    our_mask = batch_gpu["mask"].bool()
    actions = batch_gpu["actions"].long()
    rewards = batch_gpu["rewards"].to(torch.float32)
    our_action_mask = batch_gpu.get("our_action_mask")

    device = action_logits.device
    proj = projection_matrix.to(device=device)

    B, T = our_idx.shape
    A = action_logits.size(-1)
    gather_idx = our_idx.unsqueeze(-1).expand(-1, -1, A)
    logits_at = torch.gather(action_logits, 1, gather_idx)

    if our_action_mask is not None:
        step_mask_full = our_action_mask.gather(1, gather_idx)
        invalid_rows = (~step_mask_full).all(dim=-1)
        if invalid_rows.any():
            fallback_cols = logits_at[invalid_rows].argmax(dim=-1)
            step_mask_full = step_mask_full.clone()
            step_mask_full[invalid_rows] = False
            step_mask_full[invalid_rows, fallback_cols] = True
        logits_at = logits_at.masked_fill(
            ~step_mask_full,
            torch.finfo(logits_at.dtype).min,
        )

    logits_at = torch.nan_to_num(
        logits_at,
        nan=0.0,
        posinf=0.0,
        neginf=float(torch.finfo(logits_at.dtype).min),
    )
    probs = F.softmax(logits_at, dim=-1).to(torch.float32)

    hidden_dim = transformer_output.size(-1)
    hidden_idx = our_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
    hidden_states = torch.gather(transformer_output, 1, hidden_idx).to(torch.float32)

    values_at = torch.gather(state_values, 1, our_idx)

    next_idx = torch.zeros_like(our_idx)
    if T > 1:
        next_idx[:, :-1] = our_idx[:, 1:]

    L = state_values.size(1)
    idx_safe = next_idx.clamp(0, max(L - 1, 0))
    next_values = torch.gather(state_values, 1, idx_safe)

    has_next = torch.zeros_like(our_mask)
    if T > 1:
        has_next[:, :-1] = our_mask[:, 1:]
    has_next = has_next & our_mask

    gap_steps = (next_idx - our_idx).clamp_min(1).to(torch.float32)
    gap_steps = torch.where(has_next, gap_steps, torch.zeros_like(gap_steps))

    gamma = float(getattr(config, "GAMMA", 0.99))
    gae_lambda = float(getattr(config, "GAE_LAMBDA", 0.95))
    if gamma <= 0.0:
        gamma = 0.0
    if gae_lambda <= 0.0:
        gae_lambda = 0.0

    log_gamma = math.log(gamma) if gamma > 0 else 0.0
    log_lambda = math.log(gae_lambda) if gae_lambda > 0 else 0.0
    gamma_gap = torch.where(has_next, torch.exp(log_gamma * gap_steps), torch.zeros_like(gap_steps))
    lambda_gap = torch.where(has_next, torch.exp(log_lambda * gap_steps), torch.zeros_like(gap_steps))

    next_values = torch.where(has_next, next_values, torch.zeros_like(next_values))
    delta = rewards + gamma_gap * next_values - values_at
    delta = torch.where(our_mask, delta, torch.zeros_like(delta))
    discount = gamma_gap * lambda_gap

    advantages = torch.zeros_like(values_at)
    lastgaelam = torch.zeros(B, device=device, dtype=torch.float32)
    for t in reversed(range(T)):
        lastgaelam = delta[:, t] + discount[:, t] * lastgaelam
        lastgaelam = torch.where(our_mask[:, t], lastgaelam, torch.zeros_like(lastgaelam))
        advantages[:, t] = lastgaelam

    _returns = advantages + values_at

    adv_valid = advantages[our_mask]
    advantages_norm = torch.zeros_like(advantages)
    if adv_valid.numel() > 0:
        adv_mean = adv_valid.mean()
        adv_std = adv_valid.std(unbiased=False).clamp_min(1e-6)
        advantages_norm = (advantages - adv_mean) / adv_std
    advantages_norm = torch.where(our_mask, advantages_norm, torch.zeros_like(advantages_norm))

    clip_val = float(getattr(config, "GRAD_FINGERPRINT_ADV_CLIP", 4.0))
    advantages_clip = advantages_norm.clamp(min=-clip_val, max=clip_val)
    advantages_clip = torch.where(our_mask, advantages_clip, torch.zeros_like(advantages_clip))

    one_hot_actions = torch.zeros_like(probs)
    valid_action_mask = our_mask & (actions >= 0) & (actions < A)
    if valid_action_mask.any():
        action_indices = actions.clamp(0, A - 1)
        one_hot_actions = F.one_hot(action_indices, num_classes=A).to(probs.dtype)
        one_hot_actions = one_hot_actions * valid_action_mask.unsqueeze(-1)

    policy_delta = (one_hot_actions - probs) * our_mask.unsqueeze(-1)
    hidden_states = hidden_states * our_mask.unsqueeze(-1)

    projected_phi = torch.einsum(
        "bt,bta,bth,ahd->btd",
        advantages_clip.to(torch.float32),
        policy_delta.to(torch.float32),
        hidden_states.to(torch.float32),
        proj.to(torch.float32),
    )
    projected_phi = projected_phi * our_mask.unsqueeze(-1)

    phi_cpu = projected_phi.detach().cpu()
    mask_cpu = our_mask.detach().cpu()

    opponent_sums: Dict[int, torch.Tensor] = {}
    opponent_counts: Dict[int, int] = {}

    lineup_labels = batch_cpu.get("lineup_opponent_labels", [])
    for b in range(phi_cpu.size(0)):
        if b >= len(lineup_labels):
            continue
        labels_tuple = lineup_labels[b]
        if not labels_tuple:
            continue

        valid_labels = [int(l) for l in labels_tuple if l is not None and int(l) >= 0]
        if not valid_labels:
            continue

        share = 1.0 / max(len(valid_labels), 1)
        for t in range(phi_cpu.size(1)):
            if not bool(mask_cpu[b, t]):
                continue
            contrib = phi_cpu[b, t] * share
            for label in valid_labels:
                acc = opponent_sums.get(label)
                if acc is None:
                    opponent_sums[label] = contrib.clone()
                else:
                    opponent_sums[label] = acc + contrib
                opponent_counts[label] = opponent_counts.get(label, 0) + 1

    return {
        label: (tensor, opponent_counts[label])
        for label, tensor in opponent_sums.items()
        if opponent_counts.get(label, 0) > 0
    }
def _sanitize_sampling_weights(
    labels: Sequence[int],
    raw_weights: Sequence[float],
    *,
    exploration_floor: float = 0.05,
) -> Tuple[List[int], List[float]]:
    label_list = [int(l) for l in labels]
    weight_array = np.asarray(list(raw_weights), dtype=np.float64)
    if len(label_list) != weight_array.size or weight_array.size == 0:
        return label_list, []

    if np.any(np.isnan(weight_array)):
        weight_array = np.nan_to_num(weight_array, nan=0.0, posinf=0.0, neginf=0.0)

    min_val = float(weight_array.min(initial=0.0))
    if min_val < 0.0:
        weight_array = weight_array - min_val

    weight_array += 1e-6
    weight_array = np.maximum(weight_array, 0.0)

    total = float(weight_array.sum())
    if not np.isfinite(total) or total <= 0.0:
        weight_array = np.ones_like(weight_array, dtype=np.float64)
        total = float(weight_array.sum())

    weight_array /= total

    floor = float(np.clip(exploration_floor, 0.0, 1.0))
    if floor > 0.0:
        uniform = np.ones_like(weight_array) / float(weight_array.size)
        weight_array = (1.0 - floor) * weight_array + floor * uniform

    weight_array /= float(weight_array.sum())
    return label_list, weight_array.tolist()


def _normalized_ranks(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    sorted_items = sorted(values.items(), key=lambda item: item[1])
    if len(sorted_items) == 1:
        return {sorted_items[0][0]: 1.0}
    denom = float(len(sorted_items) - 1)
    return {label: idx / denom for idx, (label, _val) in enumerate(sorted_items)}


def _perform_generational_culling(
    pool_manager: OpponentPoolManager,
    stats_manager: OpponentStatsManager,
    training_label: int,
) -> None:
    scores = stats_manager.get_pressure_scores()
    max_active = int(getattr(config, "MAX_ACTIVE_OPPONENTS", 32))
    min_pressure = float(getattr(config, "CULL_MIN_PRESSURE", float("-inf")))
    active_candidates = [
        (label, score)
        for label, score in scores.items()
        if score >= min_pressure and label != int(training_label)
    ]

    pressure_map = {label: score for label, score in active_candidates}

    min_norm = float(getattr(config, "CULL_MIN_FINGERPRINT_NORM", 1e-4))
    min_steps = int(getattr(config, "CULL_MIN_COPLAY_STEPS", 0))
    alpha = float(getattr(config, "CULL_SCORE_ALPHA", 0.7))
    alpha = min(max(alpha, 0.0), 1.0)

    fingerprint_vectors = getattr(stats_manager, "ema_grad_fingerprints", {})
    fingerprint_norms = getattr(stats_manager, "fingerprint_norms", {})
    coplay_steps = getattr(stats_manager, "total_coplay_steps", Counter())

    eligible_vectors: Dict[int, torch.Tensor] = {}
    for label in pressure_map:
        arr = fingerprint_vectors.get(label)
        if arr is None:
            continue
        norm_val = float(fingerprint_norms.get(label, 0.0))
        steps_val = int(coplay_steps.get(label, 0))
        if steps_val < min_steps or norm_val < min_norm:
            continue
        vec = torch.from_numpy(arr.astype(np.float32))
        vec_norm = torch.linalg.norm(vec)
        if not torch.isfinite(vec_norm) or vec_norm.item() <= 0.0:
            continue
        eligible_vectors[label] = vec / vec_norm

    redundancy_values: Dict[int, float] = {}
    default_redundancy = 0.5
    for label in pressure_map:
        vec_i = eligible_vectors.get(label)
        if vec_i is None:
            redundancy_values[label] = default_redundancy
            continue

        min_dist = None
        for other_label, vec_j in eligible_vectors.items():
            if other_label == label:
                continue
            cos_sim = torch.dot(vec_i, vec_j).clamp(-1.0, 1.0)
            dist = float(1.0 - cos_sim.item())
            if min_dist is None or dist < min_dist:
                min_dist = dist

        if min_dist is None:
            redundancy_values[label] = 1.0
        else:
            redundancy_values[label] = max(0.0, min_dist)

    pressure_ranks = _normalized_ranks(pressure_map)
    redundancy_ranks = _normalized_ranks(redundancy_values)
    blended_scores = {
        label: alpha * pressure_ranks.get(label, 0.0)
        + (1.0 - alpha) * redundancy_ranks.get(label, 0.0)
        for label in pressure_map
    }

    active_candidates.sort(
        key=lambda item: blended_scores.get(item[0], float("-inf")),
        reverse=True,
    )

    selected: Dict[int, str] = {}
    for entry in pool_manager.pool:
        label = entry.get("label")
        if label is None:
            continue
        label_int = int(label)
        if entry.get("type") == "cpp_bot" or label_int == int(training_label):
            selected[label_int] = "active"
        else:
            selected[label_int] = entry.get("status", "active")

    limit = max_active
    for idx, (label, _score) in enumerate(active_candidates):
        if idx < limit:
            selected[label] = "active"
        else:
            selected.setdefault(label, "inactive")

    updates_needed = False
    for entry in pool_manager.pool:
        label = entry.get("label")
        if label is None:
            continue
        label_int = int(label)
        desired = selected.get(label_int, entry.get("status", "active"))
        if entry.get("type") == "cpp_bot":
            desired = "active"
        if entry.get("status") != desired:
            entry["status"] = desired
            updates_needed = True

    if updates_needed:
        pool_manager.save()
def _create_new_agent(agent_type: str, device: torch.device) -> LearnerAutoregressiveAgent:
    """Creates a new agent and its corresponding model."""
    agent = LearnerAutoregressiveAgent(device, f"learner_{agent_type}")
    if agent_type == 'main':
        model = PPOReactiveModel(
            obs_dim=9,
            use_gradient_checkpointing=bool(getattr(config, "USE_GRADIENT_CHECKPOINTING", False)),
        )
    else:  # Future branches (e.g., exploiter) can be added here.
        raise ValueError(f"Unknown agent type for creation: {agent_type}")
    agent.model = model.to(device)
    agent.max_seq_length = getattr(model, "max_seq_length", None)
    agent.reset()
    return agent

def _load_agent_from_checkpoint(
    path: str,
    model_type: str,
    device: torch.device,
) -> LearnerAutoregressiveAgent:
    """Loads an agent's state from a checkpoint path. Optionally compiles its model."""
    agent = LearnerAutoregressiveAgent(device, f"loaded_{model_type}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    agent.load_from_state_dict(state_dict)
    return agent

def _clone_agent_from_agent(src_agent: LearnerAutoregressiveAgent,
                            device: torch.device) -> LearnerAutoregressiveAgent:
    """Clone an agent using the exact same path as checkpoint loading:
    build a fresh agent and load via load_models_from_checkpoint with an
    in-memory state_dict. This avoids architecture inference and stays
    robust to naming/wrapping changes (e.g., _orig_mod, compile)."""
    if src_agent is None or src_agent.model is None:
        raise ValueError("Source agent/model is None; cannot clone.")

    # get the unwrapped model for a clean state_dict (handles torch.compile)
    src_model = getattr(src_agent.model, "_orig_mod", src_agent.model)
    # take a CPU copy of the tensors to be device-agnostic
    src_state = {k: v.detach().cpu() for k, v in src_model.state_dict().items()}

    # make a fresh agent and load exactly like _load_agent_from_checkpoint
    clone = LearnerAutoregressiveAgent(device, f"clone_of_{src_agent.player_id}")
    clone.load_from_state_dict(src_state)

    # bookkeeping to mirror the source
    clone.label = getattr(src_agent, "label", -1)
    clone.max_seq_length = getattr(src_agent, "max_seq_length", getattr(clone, "max_seq_length", None))

    # ensure correct device/mode
    if hasattr(clone, "model") and clone.model is not None:
        clone.model.to(device)
        clone.model.eval()   # rollouts should be in eval mode by default

    return clone


def _episode_token_count(episode: Dict[str, Any]) -> int:
    """Return the number of autoregressive tokens contained in an episode."""
    model_input = episode.get("model_input")
    if isinstance(model_input, dict):
        valid_lengths = model_input.get("valid_lengths")
        if isinstance(valid_lengths, torch.Tensor) and valid_lengths.numel() > 0:
            try:
                return int(valid_lengths.view(-1)[0].item())
            except Exception:
                pass
        elif valid_lengths is not None:
            try:
                return int(valid_lengths)
            except Exception:
                pass

    rewards = episode.get("reward")
    if rewards is not None:
        try:
            return int(len(rewards))
        except Exception:
            pass

    actions = episode.get("our_action")
    if actions is not None:
        try:
            return int(len(actions))
        except Exception:
            pass

    return 0


def _prepare_episode_for_buffer(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Detach tensors to CPU memory before storing the episode in the buffer."""
    if not isinstance(episode, dict):
        return episode

    for key in list(episode.keys()):
        value = episode[key]
        if torch.is_tensor(value):
            episode[key] = value.detach().cpu()

    model_input = episode.get("model_input")
    if isinstance(model_input, dict):
        for key, value in list(model_input.items()):
            if torch.is_tensor(value):
                model_input[key] = value.detach().cpu()

    return episode

def _find_traced_artifact_for_checkpoint(checkpoint_path: str) -> Optional[Path]:
    """Return the TorchScript trace produced by ``train_utils.py`` if it exists."""

    ckpt_path = Path(os.path.abspath(checkpoint_path))
    candidate = ckpt_path.with_name(f"{ckpt_path.stem}_traced.pt")
    if candidate.exists():
        return candidate

    index_path = ckpt_path.parent / "traced_index.json"
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            entries = []

        if isinstance(entries, dict):
            entries = [entries]

        resolved_ckpt = str(ckpt_path.resolve(strict=False))
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            traced_name = entry.get("traced_module")
            if not traced_name:
                continue

            traced_candidate = (ckpt_path.parent / traced_name).resolve(strict=False)
            if not traced_candidate.exists():
                continue

            source = entry.get("source_checkpoint")
            if not source:
                return traced_candidate

            if source == resolved_ckpt or source == str(ckpt_path) or source.endswith(ckpt_path.name):
                return traced_candidate

    return candidate if candidate.exists() else None


# ==============================================================================
# SECTION 2: THE CORE TRAIN FUNCTION
# ==============================================================================

def train_generation(
    run_name: str,
    master_run_name: str,
    pool_manager: OpponentPoolManager,
    max_updates: int = 100,
    # New: pass a preloaded/compiled learner or a warm_start_path for backward-compat
    learner: Optional[LearnerAutoregressiveAgent] = None,
    warm_start_path: Optional[str] = None,
    # New: cache for already loaded opponents/agents to avoid reloading
    agent_cache: Optional[Dict[str, LearnerAutoregressiveAgent]] = None,
    rng: Optional[Generator] = None,
    collect_metrics: bool = False,
    metrics_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    projection_matrix: Optional[torch.Tensor] = None,
):
    """
    Trains a single generation of an agent for 100 updates.
    Saves the final model and adds it to the opponent pool.
    """
    # 1. SETUP
    run_log_dir = os.path.join("logs", master_run_name, run_name)
    run_ckpt_dir = os.path.join("checkpoints", master_run_name, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    
    device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    if projection_matrix is not None:
        projection_matrix = projection_matrix.to(device)
    writer = SummaryWriter(log_dir=run_log_dir)
    logging.info(f"--- Starting Training Run: '{run_name}' ---")
    logging.info(f"    TensorBoard Log Dir: {run_log_dir}")
    
    # 2. INITIALIZE LEARNER AND OPPONENTS
    if learner is None:
        if warm_start_path:
            # If a path IS provided, load/clone
            logging.info(f"Loading learner from warm_start_path: {warm_start_path}")
            cache_key = f"ckpt:{os.path.abspath(warm_start_path)}"
            if agent_cache is not None and cache_key in agent_cache:
                learner = _clone_agent_from_agent(agent_cache[cache_key], device)
            else:
                base_agent = _load_agent_from_checkpoint(warm_start_path, 'main', device)
                if agent_cache is not None:
                    agent_cache[cache_key] = base_agent  # keep a copy of the base
                learner = _clone_agent_from_agent(base_agent, device)
        else:
            # If no path is provided, create a new agent from scratch
            learner = _create_new_agent('main', device)
    else:
        # Ensure learner is on the correct device
        learner.model = learner.model.to(device)
    learner.device = device

    if hasattr(torch, "compile"):
        base_model = getattr(learner.model, "_orig_mod", learner.model)
        if learner.model is base_model:
            try:
                learner.model = torch.compile(base_model)
            except Exception as exc:
                logging.warning(f"torch.compile failed for learner model: {exc}")
                learner.model = base_model

    learner.model.train()

    # Create two lists to hold parameters for weight decay and no weight decay
    decay_params = []
    no_decay_params = []

    # Iterate through all named parameters of the model
    for name, param in learner.model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if the parameter is a bias, a LayerNorm weight/bias, or an embedding weight.
        # These are typically excluded from weight decay.
        if name.endswith(".bias") or "layernorm" in name.lower() or "embedding" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Create the optimizer with two parameter groups
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': 0.01}, # Apply weight decay to this group
            {'params': no_decay_params, 'weight_decay': 0.0}   # No weight decay for this group
        ],
        lr=float(config.LEARNING_RATE),
    )
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    all_params = list(learner.model.parameters())

    # Check if the model has an opponent action head to separate its parameters
    if hasattr(learner.model, 'opp_action_head'):
        opp_head_params = list(learner.model.opp_action_head.parameters())
        opp_head_param_ids = {id(p) for p in opp_head_params}
        
        # Main parameters are everything EXCEPT the opponent head
        main_params = [p for p in all_params if id(p) not in opp_head_param_ids]
    else:
        # If the model is purely reactive with no opponent head, all params are main params
        opp_head_params = []
        main_params = all_params
        
    # or use the main one. Let's use a new config for flexibility.
    opp_head_max_norm = float(getattr(config, "OPP_HEAD_MAX_NORM", config.MAX_NORM))
    
    existing_labels = {
        int(entry.get("label"))
        for entry in pool_manager.pool
        if isinstance(entry, dict) and entry.get("label") is not None
    }

    training_policy_id = 7
    while training_policy_id in existing_labels:
        training_policy_id += 1

    learner.label = training_policy_id
    policy_map: Dict[int, Any] = {training_policy_id: learner}

    logging.info(
        f"Assigned training policy id {training_policy_id}; existing opponent labels: {sorted(existing_labels)}"
    )

    # 3. INITIALIZE ROLLOUT MANAGER
    rollout_manager = PPOVecRolloutManager(
        policy_map,
        device,
        rng=(rng or _GLOBAL_RNG),
    )

    cpp_bot_names = {
        0: "Classic",
        1: "GreedyCardSpammer",
        2: "RandomAgent",
        3: "SelectiveTableConservativeChallenger",
        4: "StrategicChallenger",
        5: "TableFirstConservativeChallenger",
        6: "TableNonTableAgent",
    }
    registered_cpp_bots: List[int] = []
    for label, name in cpp_bot_names.items():
        if not hasattr(lb, name):
            logging.error(f"lb missing C++ bot class '{name}' — cannot register native bot for label {label}")
            continue
        try:
            rollout_manager.cpp_manager.register_cpp_bot(label, name)
            registered_cpp_bots.append(label)
        except Exception as exc:
            logging.exception(
                f"Failed to register native C++ bot '{name}' (label {label}) with rollout manager: {exc}"
            )

    loaded_historical_labels: List[int] = []
    for agent_def in pool_manager.pool:
        if agent_def.get('type') == 'cpp_bot':
            continue

        label = agent_def.get('label')
        if label is None:
            continue

        policy_id = int(label)
        traced_path = agent_def.get('path_pt')

        if traced_path and not os.path.exists(traced_path):
            traced_path = None

        if not traced_path and agent_def.get('path'):
            traced_candidate = _find_traced_artifact_for_checkpoint(agent_def['path'])
            if traced_candidate is not None and traced_candidate.exists():
                traced_path = str(traced_candidate)

        if not traced_path:
            logging.warning(
                f"Skipping historical opponent label {policy_id}: missing TorchScript trace."
            )
            continue

        try:
            rollout_manager.cpp_manager.load_historical_model(policy_id, str(traced_path))
            loaded_historical_labels.append(policy_id)
        except Exception as exc:
            logging.exception(
                f"Failed to load traced historical policy {policy_id} from {traced_path}: {exc}"
            )

    logging.info(
        "Native C++ bots registered: %s; historical TorchScript policies loaded: %s",
        sorted(registered_cpp_bots),
        sorted(loaded_historical_labels),
    )

    # Opponent pool is managed locally for sampling; no manager sync needed.
    stats_manager = OpponentStatsManager(pool_manager)

    # 4. MAIN TRAINING LOOP
    episodes_per_update = int(config.EPISODES_PER_UPDATE)
    k_epochs = int(config.K_EPOCHS)
    max_batch_envs = int(getattr(config, "EPISODES_PER_UPDATE", 512))
    num_players = int(getattr(config, "NUM_PLAYERS", 4))
    
    # The buffer stores tuples of (data, age)
    max_data_age = int(getattr(config, "OFFPOLICY_EP_BUFFER_MULT", 4))
    ep_buffer: List[Tuple[Dict[str, Any], int]] = [] 

    collected_updates: List[Dict[str, Any]] = []

    for update in range(1, max_updates + 1):
        # -------- Rollout --------
        t0 = time.time()
        learner.model.eval()
        
        # Set the number of games to collect for the update.
        games_to_collect = episodes_per_update

        # Guarantee at least one learner seat per game. The C++ backend will fill the
        # remaining seats by sampling from the weighted opponent pool.
        training_ids_for_rollout = [training_policy_id]

        # Get pressure scores for all active opponents from the pool.
        pressure_scores = stats_manager.get_pressure_scores()

        # The learner (e.g., gen_2) is not yet in the opponent pool, so it won't have a score.
        # We create a proxy score for it based on the average pressure of existing opponents.
        # This ensures it's always a candidate for self-play.
        if training_policy_id not in pressure_scores:
            if pressure_scores:
                avg_pressure = sum(pressure_scores.values()) / len(pressure_scores)
                pressure_scores[training_policy_id] = avg_pressure
            else:
                # Fallback if there are no opponents with scores yet (e.g., first ever run)
                pressure_scores[training_policy_id] = 1.0
        
        # Build the list of opponents and their weights from the pool.
        opp_labels_raw, opp_weights_raw = pool_manager.build_sampling_weights(
            pressure_scores
        )

        # --- START OF FIX for Missing Learner ---
        # Manually add the current learner to the sampling pool if it wasn't already found.
        # This is necessary because the learner is not yet saved to opponent_pool.json.
        if training_policy_id not in opp_labels_raw:
            learner_pressure = pressure_scores.get(training_policy_id, 1.0)
            opp_labels_raw.append(training_policy_id)
            opp_weights_raw.append(learner_pressure)
        # --- END OF FIX ---

        # Apply the self-play discount to the learner's sampling weight.
        try:
            learner_idx = opp_labels_raw.index(training_policy_id)
            learner_discount = float(getattr(config, "SELF_PLAY_LEARNER_DISCOUNT", 0.5))
            opp_weights_raw[learner_idx] *= learner_discount
        except ValueError:
            # This should no longer happen with the fix above, but we keep it as a safeguard.
            logging.warning(f"Learner ID {training_policy_id} not in labels for discounting. This is unexpected.")
            pass

        # Sanitize weights (make non-negative, normalize, add exploration floor).
        opp_labels_sanitized, opp_weights_sanitized = _sanitize_sampling_weights(
            opp_labels_raw,
            opp_weights_raw,
            exploration_floor=float(getattr(config, "OPPONENT_EXPLORATION_FLOOR", 0.05)),
        )

        if not opp_weights_sanitized:
            opp_labels_arg = None
            opp_weights_arg = None
        else:
            opp_labels_arg = opp_labels_sanitized
            opp_weights_arg = opp_weights_sanitized

        new_eps = rollout_manager.collect_episodes(
            num_episodes=games_to_collect,
            num_players=num_players,
            training_policy_ids=training_ids_for_rollout,
            max_batch_envs=max_batch_envs,
            opponent_labels=opp_labels_arg,
            opponent_weights=opp_weights_arg,
        )
        
        if device.type == "cuda" and FORCE_CUDA_SYNC_FOR_TIMING:
            torch.cuda.synchronize()
        t_roll = time.time()
        
        if not new_eps:
            logging.warning(f"Update {update}: No episodes collected. Skipping.")
            continue

        rollout_tokens = 0
        for ep in new_eps:
            _prepare_episode_for_buffer(ep)
            tokens = _episode_token_count(ep)
            ep["_token_count"] = tokens
            rollout_tokens += tokens

        fingerprint_enabled = bool(getattr(config, "USE_GRADIENT_FINGERPRINTING", False))
        if fingerprint_enabled and projection_matrix is not None:
            try:
                update_batch_cpu = _collate_batch(new_eps)
            except ValueError:
                update_batch_cpu = None
            if update_batch_cpu is not None:
                update_batch_gpu = _to_device_batch(update_batch_cpu, device)
                with torch.no_grad():
                    fp_outs = learner.model(
                        **update_batch_gpu["mi"],
                        return_policy_features=True,
                    )
                fingerprints = _compute_fingerprints_for_update(
                    update_batch_cpu,
                    update_batch_gpu,
                    fp_outs,
                    projection_matrix,
                )
                if fingerprints:
                    stats_manager.update_fingerprints(fingerprints)
                del update_batch_gpu, update_batch_cpu, fp_outs, fingerprints

        # --- AGE-BASED BUFFER MANAGEMENT ---
        for i in range(len(ep_buffer)):
            ep_buffer[i] = (ep_buffer[i][0], ep_buffer[i][1] + 1)
            
        for ep in new_eps:
            ep_buffer.append((ep, 1))

        ep_buffer = [item for item in ep_buffer if item[1] <= max_data_age]
        
        buffer_token_total = sum(item[0].get("_token_count", 0) for item in ep_buffer)
        
        # -------- Optimize (aggregate metrics) --------
        learner.model.train()
        agg = {"total_loss": 0.0}
        n_batches = 0
        opt_tokens_processed = 0
        regression_records: List[Dict[str, Any]] = []

        for _ in range(k_epochs):
            if not ep_buffer:
                continue

            # Determine the sampling fraction based on the age of the oldest data
            oldest_age = max(item[1] for item in ep_buffer)
            sampling_fraction = 1.0 / oldest_age
            
            num_to_sample = math.ceil(len(ep_buffer) * sampling_fraction)
            
            # Ensure we always sample at least a standard batch size's worth of trajectories
            num_to_sample = max(num_to_sample, episodes_per_update) 
            
            # And never more than what's available in the buffer
            num_to_sample = min(num_to_sample, len(ep_buffer))

            # The training batch is a NEW random sample for each epoch
            sampled_indices = random.sample(range(len(ep_buffer)), num_to_sample)
            batch_eps = [ep_buffer[i][0] for i in sampled_indices]

            if not batch_eps:
                continue

            bucket_to_indices: Dict[int, List[int]] = {}
            for idx, episode in enumerate(batch_eps):
                tokens = int(episode.get("_token_count", _episode_token_count(episode)))
                if tokens <= 0:
                    tokens = 1
                bucket_len = _select_bucket_length(tokens)
                bucket_to_indices.setdefault(bucket_len, []).append(idx)

            minibatch_target = int(getattr(config, "PPO_MINIBATCH_SIZE", len(batch_eps)))
            minibatch_size = max(1, min(minibatch_target, len(batch_eps)))

            bucket_batches: List[List[int]] = []
            for indices in bucket_to_indices.values():
                if not indices:
                    continue
                random.shuffle(indices)
                for start in range(0, len(indices), minibatch_size):
                    bucket_batches.append(indices[start : start + minibatch_size])

            if not bucket_batches:
                continue

            random.shuffle(bucket_batches)

            num_minibatches = len(bucket_batches)
            if num_minibatches <= 0:
                continue

            grad_accum_steps = max(1, int(getattr(config, "GRAD_ACCUM_STEPS", 1)))
            optimizer.zero_grad(set_to_none=True)
            group_target = min(grad_accum_steps, num_minibatches)
            group_count = 0
            processed_minibatches = 0

            for indices in bucket_batches:
                mini_eps = [batch_eps[i] for i in indices]
                mini_cpu = _collate_batch(mini_eps)
                valid_lengths_cpu = mini_cpu.get("mi", {}).get("valid_lengths")
                if isinstance(valid_lengths_cpu, torch.Tensor):
                    opt_tokens_processed += int(valid_lengths_cpu.sum().item())

                mini_gpu = _to_device_batch(mini_cpu, device)

                with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    total_loss, metrics, vector_metrics = ppo_losses_batched(
                        learner.model,
                        mini_gpu,
                        sl_teacher=None,
                        update_num=update,
                        return_vector_metrics=True,
                    )

                loss_denom = max(group_target, 1)
                scaler.scale(total_loss / loss_denom).backward()

                processed_minibatches += 1
                group_count += 1

                agg["total_loss"] += float(total_loss.detach().cpu())
                for k, v in metrics.items():
                    try:
                        agg[k] = agg.get(k, 0.0) + float(v.detach().cpu())
                    except Exception:
                        pass

                lineup_targets = vector_metrics.get("lineup_pressure_targets") if isinstance(vector_metrics, dict) else None
                if lineup_targets is not None:
                    targets_list = [float(x) for x in lineup_targets.detach().cpu().tolist()]
                    token_counts_tensor = vector_metrics.get("lineup_token_counts") if isinstance(vector_metrics, dict) else None
                    if token_counts_tensor is not None:
                        weights_list = [float(x) for x in token_counts_tensor.detach().cpu().tolist()]
                    else:
                        weights_list = [1.0 for _ in targets_list]
                    opp_lineups = mini_cpu.get("lineup_opponent_labels", []) or []
                    self_play_counts = mini_cpu.get("lineup_self_play_counts", []) or []
                    player_lineups = mini_cpu.get("lineup_player_labels", []) or []
                    for idx, target_val in enumerate(targets_list):
                        if idx >= len(opp_lineups):
                            break
                        opponents = tuple(int(x) for x in opp_lineups[idx])
                        self_play = int(self_play_counts[idx]) if idx < len(self_play_counts) else 0
                        weight_val = float(weights_list[idx]) if idx < len(weights_list) else 1.0
                        players_tuple = tuple(int(x) for x in player_lineups[idx]) if idx < len(player_lineups) else tuple()
                        regression_records.append(
                            {
                                "opponents": opponents,
                                "self_play": self_play,
                                "target": float(target_val),
                                "weight": max(weight_val, 1e-6),
                                "players": players_tuple,
                            }
                        )

                n_batches += 1

                should_step = (
                    group_count >= group_target
                    or processed_minibatches == num_minibatches
                )
                if should_step:
                    scaler.unscale_(optimizer)

                    # Clip gradients for the main part of the network
                    if main_params:
                        clip_grad_norm_(main_params, max_norm=float(config.MAX_NORM))

                    # Clip gradients for the auxiliary opponent head separately
                    if opp_head_params:
                        clip_grad_norm_(opp_head_params, max_norm=opp_head_max_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    remaining = num_minibatches - processed_minibatches
                    group_target = min(grad_accum_steps, remaining) if remaining > 0 else grad_accum_steps
                    group_count = 0

                del mini_gpu, mini_cpu

        if regression_records:
            stats_manager.update_pressure_scores(
                [tuple(rec["opponents"]) for rec in regression_records],
                [rec["target"] for rec in regression_records],
                sample_weights=[rec["weight"] for rec in regression_records],
                self_play_counts=[rec["self_play"] for rec in regression_records],
            )
        cull_frequency = max(1, int(getattr(config, "CULL_FREQUENCY", 5)))
        if regression_records and (update % cull_frequency == 0):
            _perform_generational_culling(pool_manager, stats_manager, training_policy_id)
            # Opponent pool is managed locally for sampling; no manager sync needed.
        if device.type == "cuda" and FORCE_CUDA_SYNC_FOR_TIMING:
            torch.cuda.synchronize()
        t_opt_end = time.time()
        # Timings (rollout + optimize). We'll measure logging separately below
        dur_roll = t_roll - t0
        dur_opt  = t_opt_end - t_roll
        # Note: do NOT finalize dur_tot yet; include logging time later

        avg_game_length = (rollout_tokens / len(new_eps)) if new_eps else 0.0
        rollout_tps = (rollout_tokens / dur_roll) if dur_roll > 0 else 0.0
        optimize_tps = (opt_tokens_processed / dur_opt) if dur_opt > 0 else 0.0

        # -------- Log metrics (timed) --------
        t_log_start = time.time()
        avg = {k: (v / max(n_batches, 1)) for k, v in agg.items()}
        win_rate = sum(ep["win"] for ep in new_eps) / max(len(new_eps), 1)
        per_opponent_totals: Dict[Any, List[float]] = {}
        for ep in new_eps:
            opp_labels = ep.get("true_opponent_labels", ())
            if not opp_labels:
                continue
            training_label = ep.get("training_agent_label")
            winner_label = ep.get("winner_label")
            training_won = bool(ep.get("win", 0))
            if training_label is not None and winner_label is not None:
                training_won = winner_label == training_label
            for label in set(l for l in opp_labels if l is not None):
                totals = per_opponent_totals.setdefault(label, [0.0, 0.0])
                if training_won:
                    totals[0] += 1.0
                totals[1] += 1.0

        candidate_labels = set()
        if opp_labels_arg:
            try:
                candidate_labels.update(int(lab) for lab in opp_labels_arg if lab is not None)
            except Exception:
                candidate_labels.update(lab for lab in opp_labels_arg if lab is not None)
        else:
            try:
                for entry in pool_manager.get_entries(status="active", include_cpp=True):
                    lab = entry.get("label")
                    if lab is None:
                        continue
                    try:
                        candidate_labels.add(int(lab))
                    except Exception:
                        candidate_labels.add(lab)
            except Exception:
                pass
        try:
            candidate_labels.add(int(training_policy_id))
        except Exception:
            candidate_labels.add(training_policy_id)

        for lab in candidate_labels:
            per_opponent_totals.setdefault(lab, [0.0, 0.0])
        per_opponent_win_rates = {}
        per_opponent_episode_counts = {}
        for label, (wins_vs, total) in per_opponent_totals.items():
            if total <= 0:
                continue
            label_int = int(label) if isinstance(label, (int, np.integer, str)) else label
            try:
                label_key = int(label_int)
            except Exception:
                label_key = label
            per_opponent_win_rates[label_key] = wins_vs / total
            per_opponent_episode_counts[label_key] = total

        update_summary = {
            "update": update,
            "win_rate": win_rate,
            "per_opponent_win_rates": per_opponent_win_rates,
            "per_opponent_episode_counts": per_opponent_episode_counts,
        }

        if collect_metrics:
            collected_updates.append(update_summary)

        writer.add_scalar("Loss/Total", avg.get("total_loss", 0.0), update)
        writer.add_scalar("Loss/Policy", avg.get("policy_loss", 0.0), update)
        writer.add_scalar("Loss/Value", avg.get("value_loss", 0.0), update)
        writer.add_scalar("Loss/Opponent", avg.get("opp_loss", 0.0), update)
        writer.add_scalar("Loss/L1Sparsity", avg.get("l1_sparsity_loss", 0.0), update)
        writer.add_scalar("Loss/UsageBalance", avg.get("usage_balance_loss", 0.0), update)
        writer.add_scalar("Loss/BrickDiversity", avg.get("brick_diversity_loss", 0.0), update)
        writer.add_scalar("Loss/BrickDecor", avg.get("brick_decorrelation_loss", 0.0), update)
        writer.add_scalar("Loss_DCP/Total", avg.get("dcp_total_loss", 0.0), update)
        writer.add_scalar("Loss_DCP/Policy", avg.get("dcp_policy_loss", 0.0), update)
        writer.add_scalar("Loss_DCP/Value", avg.get("dcp_value_loss", 0.0), update)
        writer.add_scalar("Loss_DCP/Opponent", avg.get("dcp_opp_loss", 0.0), update)
        writer.add_scalar("Policy/Entropy", avg.get("entropy", 0.0), update)
        writer.add_scalar("Policy/ApproxKL", avg.get("approx_kl", 0.0), update)
        writer.add_scalar("Policy/ClipFraction", avg.get("clip_fraction", 0.0), update)
        writer.add_scalar("Policy/TrinalClipNegFrac", avg.get("trinal_clip_neg_frac", 0.0), update)
        writer.add_scalar("Policy_DCP/Entropy", avg.get("dcp_entropy", 0.0), update)
        writer.add_scalar("Policy_DCP/ApproxKL", avg.get("dcp_approx_kl", 0.0), update)
        writer.add_scalar("Policy_DCP/ClipFraction", avg.get("dcp_clip_fraction", 0.0), update)
        writer.add_scalar("Value/ClipFrac", avg.get("value_clip_frac", 0.0), update)
        writer.add_scalar("Value_DCP/ClipFrac", avg.get("dcp_value_clip_frac", 0.0), update)
        writer.add_scalar("Diag/ReturnStdEMA", getattr(config, "RET_STD_EMA", 1.0), update)
        # Rollout stats
        writer.add_scalar("Rollout/WinRate", win_rate, update)
        # Sort once (same criterion you use below)
        sorted_items = sorted(per_opponent_totals.items(), key=lambda item: str(item[0]))

        # Log per-opponent metrics (always emit), with safe handling for zero totals
        for label, (wins_vs, total) in sorted_items:
            win_rate_val = (wins_vs / total) if total > 0 else 0.0
            writer.add_scalar(f"PerOpponent/win_rate_vs_{label}", win_rate_val, update)
            writer.add_scalar(f"PerOpponent/episodes_vs_{label}", total, update)

        BOT_MAX_ID = 6
        per_opponent_totals_int: Dict[int, List[float]] = {}
        for lab_any, (wins_vs, total) in per_opponent_totals.items():
            try:
                lab_int = int(lab_any)
            except Exception:
                continue
            acc = per_opponent_totals_int.setdefault(lab_int, [0.0, 0.0])
            acc[0] += float(wins_vs)
            acc[1] += float(total)
        
        heldout_candidates = [lab for lab in per_opponent_totals_int.keys() if lab > BOT_MAX_ID and lab != training_policy_id]
        if heldout_candidates:
            heldout_label = max(heldout_candidates)
            hw, ht = per_opponent_totals_int[heldout_label]
            if ht > 0:
                writer.add_scalar("PerOpponent/Win_rate_vs_heldout", hw / ht, update)
        writer.add_scalar("Rollout/AvgGameLength", avg_game_length, update)
        writer.add_scalar("Rollout/TokensCollected", rollout_tokens, update)
        writer.add_scalar("Rollout/TokensPerSecond", rollout_tps, update)
        writer.add_scalar("Optimize/TokensProcessed", opt_tokens_processed, update)
        writer.add_scalar("Optimize/TokensPerSecond", optimize_tps, update)
        writer.add_scalar("Buffer/Size", len(ep_buffer), update)
        writer.add_scalar("Buffer/Tokens", buffer_token_total, update)
        writer.add_scalar("Acc/OpponentAction", avg.get("opp_action_acc", 0.0), update)

        # OpponentStatsManager: Per-agent stats (pressure/redundancy/cull/fingerprint)
        try:
            scores = stats_manager.get_pressure_scores()
            min_pressure = float(getattr(config, "CULL_MIN_PRESSURE", float("-inf")))
            alpha = float(getattr(config, "CULL_SCORE_ALPHA", 0.7))
            alpha = min(max(alpha, 0.0), 1.0)

            # Filter to candidates, excluding the training policy itself
            pressure_map: Dict[int, float] = {
                int(label): float(score)
                for label, score in scores.items()
                if float(score) >= min_pressure and int(label) != int(training_policy_id)
            }

            # Normalized ranks of pressure
            pressure_ranks = _normalized_ranks(pressure_map)

            # Compute redundancy scores based on gradient fingerprint dissimilarity
            min_norm = float(getattr(config, "CULL_MIN_FINGERPRINT_NORM", 1e-4))
            min_steps = int(getattr(config, "CULL_MIN_COPLAY_STEPS", 0))
            fingerprint_vectors = getattr(stats_manager, "ema_grad_fingerprints", {})
            fingerprint_norms = getattr(stats_manager, "fingerprint_norms", {})
            coplay_steps = getattr(stats_manager, "total_coplay_steps", Counter())

            eligible_vectors: Dict[int, torch.Tensor] = {}
            for label in pressure_map:
                arr = fingerprint_vectors.get(label)
                if arr is None:
                    continue
                norm_val = float(fingerprint_norms.get(label, 0.0))
                steps_val = int(coplay_steps.get(label, 0))
                if steps_val < min_steps or norm_val < min_norm:
                    continue
                vec = torch.from_numpy(arr.astype(np.float32))
                vec_norm = torch.linalg.norm(vec)
                if not torch.isfinite(vec_norm) or vec_norm.item() <= 0.0:
                    continue
                eligible_vectors[label] = vec / vec_norm

            redundancy_values: Dict[int, float] = {}
            default_redundancy = 0.5
            for label in pressure_map:
                vec_i = eligible_vectors.get(label)
                if vec_i is None:
                    redundancy_values[label] = default_redundancy
                    continue
                min_dist = None
                for other_label, vec_j in eligible_vectors.items():
                    if other_label == label:
                        continue
                    cos_sim = torch.dot(vec_i, vec_j).clamp(-1.0, 1.0)
                    dist = float(1.0 - cos_sim.item())
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                if min_dist is None:
                    redundancy_values[label] = 1.0
                else:
                    redundancy_values[label] = max(0.0, min_dist)

            redundancy_ranks = _normalized_ranks(redundancy_values)
            cull_scores: Dict[int, float] = {
                label: alpha * pressure_ranks.get(label, 0.0)
                + (1.0 - alpha) * redundancy_ranks.get(label, 0.0)
                for label in pressure_map
            }

            # Log per-agent stats deterministically ordered by label
            for label in sorted(pressure_map.keys(), key=lambda x: str(x)):
                writer.add_scalar(
                    f"OpponentStatsManager/PerAgent/pressure_score_{label}",
                    pressure_map.get(label, 0.0),
                    update,
                )
                writer.add_scalar(
                    f"OpponentStatsManager/PerAgent/pressure_rank_{label}",
                    pressure_ranks.get(label, 0.0),
                    update,
                )
                writer.add_scalar(
                    f"OpponentStatsManager/PerAgent/redundancy_score_{label}",
                    redundancy_values.get(label, 0.0),
                    update,
                )
                writer.add_scalar(
                    f"OpponentStatsManager/PerAgent/redundancy_rank_{label}",
                    redundancy_ranks.get(label, 0.0),
                    update,
                )
                writer.add_scalar(
                    f"OpponentStatsManager/PerAgent/cull_score_{label}",
                    cull_scores.get(label, 0.0),
                    update,
                )
                writer.add_scalar(
                    f"OpponentStatsManager/PerAgent/fingerprint_norm_{label}",
                    float(fingerprint_norms.get(label, 0.0)),
                    update,
                )
        except Exception as e:
            logging.debug("OpponentStatsManager per-agent logging error: %s", e)

        model_call_stats = rollout_manager.get_last_model_call_stats()

        train_stats = model_call_stats.get(int(training_policy_id), {})
        train_count = int(train_stats.get("count", 0) or 0)
        train_total = float(train_stats.get("total_time", 0.0) or 0.0)
        train_min = float(train_stats.get("min", 0.0) or 0.0) if train_count else 0.0
        train_max = float(train_stats.get("max", 0.0) or 0.0) if train_count else 0.0
        train_avg = (train_total / train_count) if train_count else 0.0

        writer.add_scalar("ModelCalls/TrainCount", train_count, update)
        writer.add_scalar("ModelCalls/TrainAvgMs", train_avg * 1000.0, update)
        writer.add_scalar("ModelCalls/TrainMinMs", train_min * 1000.0, update)
        writer.add_scalar("ModelCalls/TrainMaxMs", train_max * 1000.0, update)

        # Finalize logging timings and write time scalars
        if device.type == "cuda" and FORCE_CUDA_SYNC_FOR_TIMING:
            torch.cuda.synchronize()
        t_log_end = time.time()
        dur_log = t_log_end - t_log_start
        dur_tot = t_log_end - t0

        writer.add_scalar("Time/Rollout",  dur_roll, update)
        writer.add_scalar("Time/Optimize", dur_opt,  update)
        writer.add_scalar("Time/Log",      dur_log,  update)
        writer.add_scalar("Time/Total",    dur_tot,  update)

        if update % int(config.CHECKPOINT_INTERVAL) == 0:
            path = os.path.join(run_ckpt_dir, f"update_{update}.pth")
            to_save = getattr(learner.model, "_orig_mod", learner.model)
            torch.save({"model_state_dict": to_save.state_dict()}, path)

    _perform_generational_culling(pool_manager, stats_manager, training_policy_id)
    # Opponent pool is managed locally for sampling; no manager sync needed.

    # 5. FINALIZE AND SAVE
    final_path_pth = os.path.join(run_ckpt_dir, "final.pth")
    final_path_pt = os.path.join(run_ckpt_dir, "final_traced.pt")

    # Save the standard PyTorch state_dict
    model_to_save = getattr(learner.model, "_orig_mod", learner.model)
    torch.save({"model_state_dict": model_to_save.state_dict()}, final_path_pth)
    logging.info(f"Saved standard PyTorch checkpoint to {final_path_pth}")


    traced_success = trace_model_from_checkpoint(final_path_pth, final_path_pt, device)

    extra_metadata = {}
    if traced_success and os.path.exists(final_path_pt):
        extra_metadata["path_pt"] = final_path_pt
    else:
        if traced_success:
            logging.warning(
                "TorchScript artifact %s missing after tracing; skipping pool registration.",
                final_path_pt,
            )
        else:
            logging.warning(
                "TorchScript tracing failed for %s; historical self-play will skip C++ loading.",
                run_name,
            )

    pool_manager.add_agent(
            name=run_name,
            model_type='main',
            # The 'path' should ALWAYS be the .pth file for cloning and warm-starting.
            path=final_path_pth,
            **extra_metadata,
        )
    writer.close()
    logging.info(f"Saved final model for '{run_name}' to {final_path_pth}")

    result: Dict[str, Any] = {
        "run_name": run_name,
        "final_model_path": final_path_pth,
    }
    if collect_metrics:
        result["update_metrics"] = collected_updates

    return result


# ==============================================================================
# SECTION 3: THE MASTER ORCHESTRATOR
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Master Self-Play Loop for PPO Autoregressive Agent")
    parser.add_argument("--pool-file", type=str, default="opponent_pool.json", help="Path to the opponent pool JSON file.")
    parser.add_argument("--sl-path", type=str, default=config.SL_TEACHER_CKPT, help="Path to the initial supervised learning checkpoint.")
    parser.add_argument("--max-gens", type=int, default=10, help="Total number of generations to train.")
    parser.add_argument("--challenger-freq", type=int, default=0, help="Inject a challenger from SL every N generations. Set to 0 to disable.")
    parser.add_argument("--master-run-name", type=str, default=None, help="Overall name for the self-play experiment folder.")
    parser.add_argument("--no-sl", action="store_true", help="Start generation 1 from scratch, without SL warm-start.")
    args = parser.parse_args()
    
    master_run_name = args.master_run_name or f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"Starting master self-play run: {master_run_name}")

    pool_manager = OpponentPoolManager(args.pool_file)
    # Keep agents/models in memory across generations
    agent_cache: Dict[str, LearnerAutoregressiveAgent] = {}
    initial_sl_path = None if args.no_sl else args.sl_path
    training_device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    projection_matrix = None
    if getattr(config, "USE_GRADIENT_FINGERPRINTING", False):
        action_dim = int(getattr(config, "OUTPUT_DIM", 7))
        hidden_dim = int(getattr(config, "HIDDEN_DIM", 256))
        fingerprint_dim = int(getattr(config, "GRAD_FINGERPRINT_DIM", 256))
        generator = torch.Generator(device=training_device)
        generator.manual_seed(SEED)
        projection_matrix = torch.randn(
            action_dim,
            hidden_dim,
            fingerprint_dim,
            generator=generator,
            device=training_device,
            dtype=torch.float32,
        )
        projection_matrix = F.normalize(projection_matrix, dim=-1)
    # --- Step 1: Bootstrap Generation 1 (if it doesn't exist) ---
    gen1_name = "gen_1"
    if not any(gen1_name in agent['name'] for agent in pool_manager.pool):
        logging.info("="*20 + " Training Generation 1 (Bootstrap) " + "="*20)
        train_generation(
            run_name=gen1_name,
            master_run_name=master_run_name,
            pool_manager=pool_manager,
            warm_start_path=initial_sl_path,
            agent_cache=agent_cache,
            rng=_GLOBAL_RNG,
            projection_matrix=projection_matrix,
        )

    # --- Step 2: The Main Generational Loop ---
    latest_gen_num = 1
    while any(f"gen_{latest_gen_num}" in a['name'] for a in pool_manager.pool):
        latest_gen_num += 1
    
    for gen in range(latest_gen_num, args.max_gens + 1):
        logging.info(f"\n{'='*20} Starting Generation {gen} {'='*20}\n")
        
        # --- Optional: Inject a Challenger ---
        if args.challenger_freq > 0 and gen % args.challenger_freq == 0:
            challenger_name = f"challenger_for_gen_{gen}"
            if not any(challenger_name in a['name'] for a in pool_manager.pool):
                logging.info("--- Training a new Challenger from SL ---")
                train_generation(
                    run_name=challenger_name,
                    master_run_name=master_run_name,
                    pool_manager=pool_manager,
                    warm_start_path=initial_sl_path,
                    agent_cache=agent_cache,
                    rng=_GLOBAL_RNG,
                    projection_matrix=projection_matrix,
                )
        
        # The new generation is a clone of the previous one
        prev_gen_name = f"gen_{gen - 1}"
        prev_gen_def = next((a for a in pool_manager.pool if a['name'] == prev_gen_name), None)
        if not prev_gen_def:
            logging.error(f"Could not find previous generation champion '{prev_gen_name}' in pool. Exiting.")
            break

        # Reuse compiled previous-gen model from cache if available, to avoid disk I/O
        prev_ckpt_key = f"ckpt:{os.path.abspath(prev_gen_def['path'])}"
        if prev_ckpt_key not in agent_cache:
            # Load once into cache (no compile)
            agent_cache[prev_ckpt_key] = _load_agent_from_checkpoint(prev_gen_def['path'], 'main', torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")))

        # Clone from cached prev champion for the new learner
        device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        new_learner = _clone_agent_from_agent(agent_cache[prev_ckpt_key], device)

        train_generation(
            run_name=f"gen_{gen}",
            master_run_name=master_run_name,
            pool_manager=pool_manager,
            learner=new_learner,
            agent_cache=agent_cache,
            rng=_GLOBAL_RNG,
            projection_matrix=projection_matrix,
        )
