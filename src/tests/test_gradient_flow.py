#!/usr/bin/env python3
"""
Quick test to verify gradient flow through the training forward pass,
both with and without gradient checkpointing.
"""

import torch
import sys
import traceback
from src.tests import test_utils as tu

# Import model
from src.model.ppo_reactive_model import PPOReactiveModel


def run_one(model, inputs, checkpointing: bool):
    header = f"Testing Gradient Flow (use_gradient_checkpointing={checkpointing})"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Unpack inputs created outside so both runs share the same batch
    obs_sequence, action_sequence, agent_types, positions, action_masks, padding_mask = inputs

    # Print shapes for visibility
    print(f"\nInput shapes:")
    print(f"  obs_sequence: {obs_sequence.shape}")
    print(f"  action_sequence: {action_sequence.shape}")

    # Forward pass
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
            return False, {}
        print("  ✓ No NaN in forward outputs")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False, {}

    # Backward pass
    print("\n" + "-" * 20 + " Backward Pass " + "-" * 20)
    try:
        loss = action_logits.sum() + state_values.sum() + win_logits.sum() + opp_logits.sum()
        print(f"Loss: {loss.item():.4f}")
        loss.backward()
        print("Backward pass successful!")

        # Gradient check with per-parameter NaN counts
        print("\n" + "-" * 20 + " Gradient Check " + "-" * 20)
        has_nan = False
        no_grad_count = 0
        nan_params = {}
        all_params = list(model.named_parameters())
        print(f"Checking {len(all_params)} parameters...")

        for name, param in all_params:
            if param.grad is None:
                no_grad_count += 1
                print(f"⚠️  {name:50s} has no gradient")
                continue
            n_nans = torch.isnan(param.grad).sum().item()
            if n_nans > 0:
                has_nan = True
                nan_params[name] = int(n_nans)

        return (not has_nan and no_grad_count == 0), nan_params

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        traceback.print_exc()
        return False, {}


def run_test_suite(num_trials: int = 5, batch_size: int = 128, seq_len: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # One model instance toggling checkpoint flag for comparable params/grads
    def make_model(use_ckpt: bool):
        m = PPOReactiveModel(
            obs_dim=16,
            dropout_rate=0.1,
            use_gradient_checkpointing=use_ckpt
        ).to(device)
        m.train()
        return m

    results = []
    for trial in range(1, num_trials + 1):
        print(f"\n===== Trial {trial}/{num_trials} =====")
        # Seed per trial for reproducibility
        seed = 1337 + trial
        # Use padding_ratio=0.3 to mark last 30% as padding like before
        inputs = tu.create_dummy_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            obs_dim=16,
            device=str(device),
            seed=seed,
            padding_ratio=0.3,
            dtype=torch.float32,
        )

        # Non-checkpoint run
        model_no = make_model(False)
        ok_no, nans_no = run_one(model_no, inputs, checkpointing=False)

        # Checkpoint run on the exact same inputs
        model_ckpt = make_model(True)
        ok_ck, nans_ck = run_one(model_ckpt, inputs, checkpointing=True)

        results.append({
            "trial": trial,
            "seed": seed,
            "ok_no": ok_no,
            "ok_ck": ok_ck,
            "nan_count_no": sum(nans_no.values()) if nans_no else 0,
            "nan_count_ck": sum(nans_ck.values()) if nans_ck else 0,
            "nans_no": nans_no,
            "nans_ck": nans_ck,
        })

    # Summary
    print("\n" + "=" * 60)
    print("NaN Summary over trials (same inputs per trial)")
    print("=" * 60)
    total_no = sum(1 for r in results if r["ok_no"]) \
               , sum(r["nan_count_no"] for r in results)
    total_ck = sum(1 for r in results if r["ok_ck"]) \
               , sum(r["nan_count_ck"] for r in results)
    ok_no_trials, nan_no_total = total_no
    ok_ck_trials, nan_ck_total = total_ck
    print(f"Non-ckpt: ok={ok_no_trials}/{num_trials}, total_nan_elems={nan_no_total}")
    print(f"Ckpt    : ok={ok_ck_trials}/{num_trials}, total_nan_elems={nan_ck_total}")
    for r in results:
        print(f"Trial {r['trial']} (seed={r['seed']}): no-ckpt ok={r['ok_no']} (nan_elems={r['nan_count_no']}), "
              f"ckpt ok={r['ok_ck']} (nan_elems={r['nan_count_ck']})")
        # Print top offending parameters (up to 8) when NaNs present
        if r['nan_count_no']:
            top_no = sorted(r['nans_no'].items(), key=lambda kv: kv[1], reverse=True)[:8]
            print("  Non-ckpt NaNs (top):")
            for name, cnt in top_no:
                print(f"    {name}: {cnt}")
        if r['nan_count_ck']:
            top_ck = sorted(r['nans_ck'].items(), key=lambda kv: kv[1], reverse=True)[:8]
            print("  Ckpt NaNs (top):")
            for name, cnt in top_ck:
                print(f"    {name}: {cnt}")

    # Return overall pass/fail
    return (ok_no_trials == num_trials) and (ok_ck_trials == num_trials)

if __name__ == "__main__":
    # Smaller defaults to iterate fast; bump as needed
    all_ok = run_test_suite(num_trials=5, batch_size=64, seq_len=64)
    sys.exit(0 if all_ok else 1)
