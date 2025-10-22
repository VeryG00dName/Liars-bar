"""
Generate PyTorch reference outputs for layer-by-layer C++ validation.

This script loads a checkpoint and runs ~512 game states through the PyTorch model,
saving intermediate outputs after each layer so we can compare against C++ implementation.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import numpy as np

from src.agents.learner_ar_agent import build_model_from_state
from src.model.ppo_reactive_model_debug import PPOReactiveModelDebug


def generate_random_inputs(
    batch_size: int,
    seq_length: int,
    obs_dim: int,
    action_dim: int = 7,
    pad_obs_to: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Generate random inputs that match the model's expected format.

    Args:
        pad_obs_to: Pad observation dimension to this value for CUDA alignment.
                    If obs_dim < pad_obs_to, observations will be zero-padded.

    Returns:
        Dictionary with keys: obs_sequence, action_sequence, agent_types,
        positions, action_masks, padding_mask
    """
    # Generate observations (random values in reasonable range)
    obs_sequence = torch.randn(batch_size, seq_length, obs_dim)

    # Pad observations to aligned dimension for CUDA efficiency
    if obs_dim < pad_obs_to:
        padding = torch.zeros(batch_size, seq_length, pad_obs_to - obs_dim)
        obs_sequence = torch.cat([obs_sequence, padding], dim=-1)
        print(f"Padded observations from {obs_dim} to {pad_obs_to} dimensions")

    # Generate action sequence (random valid action indices 0-10)
    action_sequence = torch.randint(0, 11, (batch_size, seq_length))

    # Generate agent types (0=our agent, 1-3=opponents)
    agent_types = torch.randint(0, 4, (batch_size, seq_length))

    # Generate positions (0 to seq_length-1)
    positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)

    # Generate action masks (random valid/invalid actions)
    action_masks = torch.randint(0, 2, (batch_size, seq_length, action_dim)).bool()

    # Generate padding mask (some sequences are shorter than max)
    # For simplicity, no padding in this version
    padding_mask = torch.zeros(batch_size, seq_length).bool()

    return {
        'obs_sequence': obs_sequence,
        'action_sequence': action_sequence,
        'agent_types': agent_types,
        'positions': positions,
        'action_masks': action_masks,
        'padding_mask': padding_mask,
    }


def pad_model_weights(model_state_dict: Dict[str, torch.Tensor], pad_obs_to: int = 16) -> Dict[str, torch.Tensor]:
    """
    Pad observation encoder weights for CUDA alignment.

    Args:
        model_state_dict: Model state dictionary
        pad_obs_to: Target dimension for padding (default 16)

    Returns:
        Modified state dict with padded weights
    """
    # Pad obs_encoder.0.weight from [hidden_dim, obs_dim] to [hidden_dim, pad_obs_to]
    obs_linear_weight_key = "obs_encoder.0.weight"
    if obs_linear_weight_key in model_state_dict:
        weight = model_state_dict[obs_linear_weight_key]
        hidden_dim, obs_dim = weight.shape

        if obs_dim < pad_obs_to:
            # Pad with zeros
            padding = torch.zeros(hidden_dim, pad_obs_to - obs_dim, dtype=weight.dtype, device=weight.device)
            padded_weight = torch.cat([weight, padding], dim=1)
            model_state_dict[obs_linear_weight_key] = padded_weight
            print(f"Padded {obs_linear_weight_key} from [{hidden_dim}, {obs_dim}] to [{hidden_dim}, {pad_obs_to}]")

    return model_state_dict


def save_debug_outputs(outputs_dict: Dict[str, torch.Tensor], path: Path):
    """Save debug outputs to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(outputs_dict, f)
    print(f"Saved reference outputs to {path}")


def main():
    parser = argparse.ArgumentParser(description='Generate PyTorch reference outputs')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=64, help='Sequence length')
    parser.add_argument('--output', type=str, default='reference_outputs.pkl',
                       help='Output path for reference file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--agent-key', type=str, default='self', help='Agent key in checkpoint')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Load model using existing infrastructure
    if 'policy_nets' in checkpoint and args.agent_key in checkpoint['policy_nets']:
        model_state_dict = checkpoint['policy_nets'][args.agent_key]
    elif 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        raise ValueError(f"Could not find model state dict in checkpoint (keys: {list(checkpoint.keys())})")

    # Infer original obs_dim before padding
    obs_linear_weight = model_state_dict.get("obs_encoder.0.weight")
    if obs_linear_weight is not None:
        original_obs_dim = obs_linear_weight.shape[1]
        print(f"Detected original obs_dim: {original_obs_dim}")
    else:
        raise ValueError("Could not find obs_encoder.0.weight in model state dict")

    # Pad model weights for CUDA alignment
    PAD_OBS_TO = 16
    print(f"Padding model weights for CUDA alignment...")
    model_state_dict = pad_model_weights(model_state_dict, pad_obs_to=PAD_OBS_TO)

    print(f"Building model from state dict...")
    device = torch.device('cpu')
    model = build_model_from_state(model_state_dict, device, debug=True)
    model.eval()

    # obs_dim is now the original dimension (before padding)
    obs_dim = original_obs_dim
    print(f"Will generate inputs with obs_dim={obs_dim}, then pad to {PAD_OBS_TO}")

    print(f"Generating random inputs: batch_size={args.batch_size}, seq_length={args.seq_length}")
    inputs = generate_random_inputs(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        obs_dim=obs_dim,
        pad_obs_to=PAD_OBS_TO,
    )

    print("Running debug forward pass...")
    with torch.no_grad():
        debug_outputs = model(
            obs_sequence=inputs['obs_sequence'],
            action_sequence=inputs['action_sequence'],
            agent_types=inputs['agent_types'],
            positions=inputs['positions'],
            action_masks=inputs['action_masks'],
            padding_mask=inputs['padding_mask'],
            return_debug_outputs=True,
        )

    # Save outputs
    output_path = Path(args.output)
    save_debug_outputs(debug_outputs, output_path)

    print("\nSummary of recorded tensors:")
    for key in sorted(debug_outputs.keys()):
        tensor = debug_outputs[key]
        print(f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == '__main__':
    main()
