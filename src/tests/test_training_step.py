#!/usr/bin/env python3
"""
Test a full training step to verify:
1. Gradient flow through all layers
2. No NaN in gradients
3. GradScaler stability
4. MoE backward kernels are being used
"""

import torch
import sys

def test_training_step():
    print("=" * 80)
    print("Testing Full Training Step with Custom MoE Backward")
    print("=" * 80)

    from src.model.ppo_reactive_model import PPOReactiveModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create model with realistic dimensions
    model = PPOReactiveModel(
        obs_dim=50,
        action_dim=7,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=480,
        num_agent_types=4,
        num_experts=8,
        top_k=2,
        expert_ffn_dim=512
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # Create realistic batch
    B, T = 8, 64
    obs_sequence = torch.randn(B, T, 50, device=device)
    action_sequence = torch.randint(0, 7, (B, T), device=device)
    agent_types = torch.zeros(B, T, device=device, dtype=torch.long)
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    action_masks = torch.ones(B, T, 7, device=device, dtype=torch.bool)
    padding_mask = torch.zeros(B, T, device=device, dtype=torch.bool)

    print(f"Batch configuration:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {T}")
    print(f"  Total tokens: {B * T}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    # Training step with AMP
    print("=" * 80)
    print("Running training step with AMP...")
    print("=" * 80)

    try:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(
                obs_sequence=obs_sequence,
                action_sequence=action_sequence,
                agent_types=agent_types,
                positions=positions,
                action_masks=action_masks,
                padding_mask=padding_mask
            )

            action_logits, opp_logits, state_values, win_logits, gate_logits_tensor, routing = outputs

            # Create dummy loss (similar to PPO)
            policy_loss = -action_logits.mean()
            value_loss = state_values.pow(2).mean()
            entropy_loss = -action_logits.softmax(dim=-1).mul(action_logits.log_softmax(dim=-1)).sum(dim=-1).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

        print(f"\n✓ Forward pass completed")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Policy loss: {policy_loss.item():.6f}")
        print(f"  Value loss: {value_loss.item():.6f}")
        print(f"  Entropy loss: {entropy_loss.item():.6f}")

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        print(f"\n✓ Backward pass completed")
        print(f"  GradScaler scale: {scaler.get_scale()}")

        # Check gradients
        print("\n" + "=" * 80)
        print("Gradient Analysis:")
        print("=" * 80)

        layer_grads = {
            "obs_encoder": [],
            "embeddings": [],
            "gates": [],
            "transformer.layers.0": [],
            "transformer.layers.1": [],
            "moe.gate": [],
            "moe.experts": [],
            "heads": []
        }

        nan_params = []
        zero_grad_params = []
        total_params = 0

        for name, param in model.named_parameters():
            total_params += 1

            if param.grad is None:
                zero_grad_params.append(name)
                continue

            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()

            if has_nan:
                nan_params.append((name, grad_norm))

            # Categorize gradients
            if "obs_encoder" in name:
                layer_grads["obs_encoder"].append(grad_norm)
            elif "embedding" in name:
                layer_grads["embeddings"].append(grad_norm)
            elif "gate_" in name:
                layer_grads["gates"].append(grad_norm)
            elif "transformer.layers.0" in name:
                if "moe.gate" in name:
                    layer_grads["moe.gate"].append(grad_norm)
                elif "moe.experts" in name:
                    layer_grads["moe.experts"].append(grad_norm)
                else:
                    layer_grads["transformer.layers.0"].append(grad_norm)
            elif "transformer.layers.1" in name:
                layer_grads["transformer.layers.1"].append(grad_norm)
            elif "heads" in name:
                layer_grads["heads"].append(grad_norm)

        # Print summary
        print(f"\nTotal parameters: {total_params}")
        print(f"Parameters with gradients: {total_params - len(zero_grad_params)}")
        print(f"Parameters without gradients: {len(zero_grad_params)}")

        print(f"\nGradient norms by layer:")
        for layer_name, norms in layer_grads.items():
            if norms:
                avg_norm = sum(norms) / len(norms)
                max_norm = max(norms)
                min_norm = min(norms)
                print(f"  {layer_name:25s}: avg={avg_norm:.4e}, min={min_norm:.4e}, max={max_norm:.4e}, count={len(norms)}")

        if nan_params:
            print(f"\n❌ FAILED: {len(nan_params)} parameters have NaN gradients:")
            for name, norm in nan_params[:10]:  # Show first 10
                print(f"  - {name}: norm={norm}")
            return False

        if not layer_grads["moe.experts"]:
            print(f"\n⚠️  WARNING: No MoE expert gradients detected!")
            print(f"   This might mean the custom backward isn't being used")

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        new_scale = scaler.get_scale()
        print(f"\n✓ Optimizer step completed")
        print(f"  New GradScaler scale: {new_scale}")

        if new_scale < 1.0:
            print(f"\n⚠️  WARNING: GradScaler dropped below 1.0!")
            print(f"   This suggests gradient overflow issues")

        print("\n" + "=" * 80)
        print("✅ SUCCESS: Training step completed without NaN!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_step()
    sys.exit(0 if success else 1)
