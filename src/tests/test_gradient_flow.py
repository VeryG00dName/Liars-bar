#!/usr/bin/env python3
"""
Quick test to verify gradient flow through the training forward pass.
"""

import torch
import sys

def test_gradient_flow():
    print("=" * 60)
    print("Testing Gradient Flow with indexed_batched_linear_autograd")
    print("=" * 60)

    # Import model
    from src.model.ppo_reactive_model import PPOReactiveModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create small model
    model = PPOReactiveModel(
        obs_dim=16,
        action_dim=7,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.0,
        max_seq_length=32,
        num_agent_types=4,
        num_experts=4,
        top_k=2,
        expert_ffn_dim=128
    ).to(device)

    # Create dummy inputs
    B, T = 2, 8
    obs_sequence = torch.randn(B, T, 16, device=device)
    action_sequence = torch.randint(0, 7, (B, T), device=device)
    agent_types = torch.zeros(B, T, device=device, dtype=torch.long)
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    action_masks = torch.ones(B, T, 7, device=device, dtype=torch.bool)
    padding_mask = torch.zeros(B, T, device=device, dtype=torch.bool)

    print(f"\nInput shapes:")
    print(f"  obs_sequence: {obs_sequence.shape}")
    print(f"  action_sequence: {action_sequence.shape}")
    print(f"  agent_types: {agent_types.shape}")

    # Forward pass
    print("\n" + "=" * 60)
    print("Running forward pass...")
    print("=" * 60)

    try:
        outputs = model(
            obs_sequence=obs_sequence,
            action_sequence=action_sequence,
            agent_types=agent_types,
            positions=positions,
            action_masks=action_masks,
            padding_mask=padding_mask
        )

        action_logits, opp_logits, state_values, win_logits, gate_logits_tensor, routing = outputs

        print(f"\nForward pass successful!")
        print(f"  action_logits: {action_logits.shape}")
        print(f"  state_values: {state_values.shape}")
        print(f"  gate_logits_tensor: {gate_logits_tensor.shape}")

        # Check for NaN in forward outputs
        if torch.isnan(action_logits).any():
            print("  ❌ NaN detected in action_logits!")
            return False
        if torch.isnan(state_values).any():
            print("  ❌ NaN detected in state_values!")
            return False

        print("  ✓ No NaN in forward outputs")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Backward pass
    print("\n" + "=" * 60)
    print("Running backward pass...")
    print("=" * 60)

    try:
        # Create dummy loss
        loss = action_logits.sum() + state_values.sum() + win_logits.sum() + opp_logits.sum()

        print(f"Loss: {loss.item():.4f}")

        # Backward
        loss.backward()

        print("\nBackward pass successful!")

        # Check gradients for each layer
        print("\n" + "=" * 60)
        print("Checking gradients...")
        print("=" * 60)

        has_nan = False
        no_grad_count = 0

        for name, param in model.named_parameters():
            if param.grad is None:
                no_grad_count += 1
                print(f"⚠️  {name:50s} has no gradient")
                continue

            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            grad_mean = param.grad.mean().item()
            has_nan_param = torch.isnan(param.grad).any()

            # Only print problematic params or first few
            if has_nan_param or "transformer.layers.0" in name or "obs_encoder" in name:
                status = "❌ NaN" if has_nan_param else "✓"
                print(f"{status} {name:50s} grad: min={grad_min:+.4e}, max={grad_max:+.4e}, mean={grad_mean:+.4e}")

            if has_nan_param:
                has_nan = True

        if no_grad_count > 0:
            print(f"\n⚠️  {no_grad_count} parameters have no gradient")

        if has_nan:
            print("\n❌ FAILED: NaN gradients detected!")
            return False
        else:
            print("\n✅ SUCCESS: All gradients are finite!")
            return True

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradient_flow()
    sys.exit(0 if success else 1)
