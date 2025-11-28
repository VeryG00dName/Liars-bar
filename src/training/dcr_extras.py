import torch
import numpy as np
from typing import Tuple, Optional, List

class ReservoirBuffer:
    """
    Fixed-size buffer using reservoir sampling for replacement.
    Stores data on CPU to save GPU memory, moves to GPU on sample.
    """
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        max_seq_len: int,
        action_dim: int = 7,
        device: torch.device = torch.device("cpu"),
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_seq_len = max_seq_len
        self.action_dim = action_dim
        self.device = device
        
        self.ptr = 0
        self.size = 0
        self.total_seen = 0
        
        # Storage
        # We store everything as float32 or long tensors
        self.obs_seq = torch.zeros((capacity, max_seq_len, obs_dim), dtype=torch.float32)
        self.act_seq = torch.zeros((capacity, max_seq_len), dtype=torch.long)
        self.agent_types = torch.zeros((capacity, max_seq_len), dtype=torch.long)
        self.positions = torch.zeros((capacity, max_seq_len), dtype=torch.long)
        self.padding_mask = torch.ones((capacity, max_seq_len), dtype=torch.bool) # Default to all padded
        
        # Targets (Regrets or Strategy)
        # We assume we are training on the *last* step of the sequence usually, 
        # or maybe all steps? Deep CFR usually trains on specific information sets.
        # If we train on the whole sequence, target should be [capacity, max_seq_len, action_dim]
        # But usually we collect (InfoSet, Regret) pairs.
        # Let's assume we store one target vector per sequence (the regret/strategy at the decision point).
        # If we need multiple decision points from one game, we add them as separate entries.
        self.targets = torch.zeros((capacity, action_dim), dtype=torch.float32)
        
        # Weights (iteration t linear weighting)
        self.weights = torch.zeros((capacity,), dtype=torch.float32)

    def add(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
        target: torch.Tensor,
        weight: float = 1.0,
    ):
        """
        Add a single sample to the buffer.
        Inputs should be tensors of shape [seq_len, ...] or [action_dim].
        """
        # Reservoir sampling logic
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            # Random replacement
            r = np.random.randint(0, self.total_seen + 1)
            if r < self.capacity:
                idx = r
            else:
                self.total_seen += 1
                return # Discard
        
        self.total_seen += 1
        
        # Copy data
        # Ensure inputs are on CPU
        seq_len = obs_seq.size(0)
        if seq_len > self.max_seq_len:
            # Truncate or error? Let's truncate from the end (keep recent history)
            # Actually for Liar's Bar, recent history is most important but full history matters.
            # Let's assume caller handles truncation or we take last max_seq_len
            start = seq_len - self.max_seq_len
            obs_seq = obs_seq[start:]
            act_seq = act_seq[start:]
            agent_types = agent_types[start:]
            positions = positions[start:]
            padding_mask = padding_mask[start:]
            seq_len = self.max_seq_len
        
        self.obs_seq[idx, :seq_len] = obs_seq.cpu()
        self.act_seq[idx, :seq_len] = act_seq.cpu()
        self.agent_types[idx, :seq_len] = agent_types.cpu()
        self.positions[idx, :seq_len] = positions.cpu()
        self.padding_mask[idx, :seq_len] = padding_mask.cpu()
        
        # Pad the rest
        if seq_len < self.max_seq_len:
            self.obs_seq[idx, seq_len:] = 0
            self.act_seq[idx, seq_len:] = 0
            self.agent_types[idx, seq_len:] = 0
            self.positions[idx, seq_len:] = 0
            self.padding_mask[idx, seq_len:] = True
            
        self.targets[idx] = target.cpu()
        self.weights[idx] = weight

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if self.size == 0:
            return None
            
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.obs_seq[indices].to(self.device),
            self.act_seq[indices].to(self.device),
            self.agent_types[indices].to(self.device),
            self.positions[indices].to(self.device),
            self.padding_mask[indices].to(self.device),
            self.targets[indices].to(self.device),
            self.weights[indices].to(self.device),
        )

    def __len__(self):
        return self.size
