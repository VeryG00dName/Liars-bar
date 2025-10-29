#!/usr/bin/env python3
"""
Quick test to verify gradient flow through the training forward pass,
both with and without gradient checkpointing.
"""

import torch
import sys
import traceback

# Import model
from src.model.ppo_reactive_model import PPOReactiveModel

def run_test(checkpointing: bool):
    header = f"Testing Gradient Flow (use_gradient_checkpointing={checkpointing})"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create small model and set the checkpointing flag
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
        expert_ffn_dim=128,
        use_gradient_checkpointing=checkpointing  # <-- Key change here
    ).to(device)
    model.train() # Ensure we are in training mode

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

    # --- Forward pass ---
    print("\n" + "-" * 20 + " Forward Pass " + "-" * 20)
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

        print(f"Forward pass successful!")
        if torch.isnan(action_logits).any() or torch.isnan(state_values).any():
            print("  ❌ NaN detected in forward outputs!")
            return False
        print("  ✓ No NaN in forward outputs")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False

    # --- Backward pass ---
    print("\n" + "-" * 20 + " Backward Pass " + "-" * 20)
    try:
        # Create dummy loss that uses all outputs
        loss = action_logits.sum() + state_values.sum() + win_logits.sum() + opp_logits.sum()
        print(f"Loss: {loss.item():.4f}")

        # Backward
        loss.backward()
        print("Backward pass successful!")

        # --- Check gradients ---
        print("\n" + "-" * 20 + " Gradient Check " + "-" * 20)
        has_nan = False
        no_grad_count = 0
        all_params = list(model.named_parameters())
        print(f"Checking {len(all_params)} parameters...")

        for name, param in all_params:
            if param.grad is None:
                no_grad_count += 1
                print(f"⚠️  {name:50s} has no gradient")
                continue

            if torch.isnan(param.grad).any():
                print(f"❌ {name:50s} has NaN gradients!")
                has_nan = True

        if no_grad_count > 0:
            print(f"\n⚠️  {no_grad_count} parameters have no gradient")
        
        if has_nan:
            print("\n❌ FAILED: NaN gradients detected!")
            return False
        elif no_grad_count > 0:
            print("\n❌ FAILED: Some parameters have no gradient!")
            return False
        else:
            print("\n✅ SUCCESS: All gradients are finite and present!")
            return True

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test without checkpointing first
    success_no_checkpoint = run_test(checkpointing=False)
    
    # Run the test with checkpointing
    success_checkpoint = run_test(checkpointing=True)

    print("\n" + "=" * 60)
    print("                 Test Summary")
    print("=" * 60)
    print(f"Standard Backward Pass (checkpointing=False): {'✅ SUCCESS' if success_no_checkpoint else '❌ FAILED'}")
    print(f"Checkpointing Backward Pass (checkpointing=True): {'✅ SUCCESS' if success_checkpoint else '❌ FAILED'}")
    print("=" * 60)

    # Exit with failure code if either test failed
    sys.exit(0 if (success_no_checkpoint and success_checkpoint) else 1)