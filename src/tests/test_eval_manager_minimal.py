"""Minimal unit tests for EvalManager to isolate the crash."""

import torch
import numpy as np
from src.misc import lb
from src.tests import test_utils as tu


def test_1_load_single_policy():
    """Test 1: Load a single policy and finalize."""
    print("\n=== TEST 1: Load Single Policy ===")

    eval_mgr = lb.EvalManager()
    eval_mgr.set_max_env_batch(4)
    eval_mgr.set_inference_batch_size(4)

    # Load a real checkpoint
    ckpt_path = "checkpoints/test76/gen_1/final.pth"
    print(f"Loading checkpoint: {ckpt_path}")

    data = torch.load(ckpt_path, map_location="cpu")
    state_dict = data if not isinstance(data, dict) else data.get("model_state_dict", data)
    try:
        tu.pad_model_weights(state_dict, pad_obs_to=16)
    except Exception as e:
        print(f"[WARN] Padding obs_encoder.0.weight failed for {ckpt_path}: {e}")

    policy_id = 10
    eval_mgr.load_model(policy_id, state_dict, ckpt_path)
    print(f"  ✓ Loaded policy {policy_id}")

    eval_mgr.finalize_model_loading()
    print("  ✓ Finalized model loading")

    return eval_mgr


def test_2_run_minimal_roles(eval_mgr):
    """Test 2: Run minimal game (1 quartet, 1 generation)."""
    print("\n=== TEST 2: Run Minimal Roles ===")

    # Single quartet: 4 players all using policy_id=10
    roles = [[10, 10, 10, 10]]
    lineup_indices = [0]
    num_players = 4
    seed = 42

    print(f"  Running 1 quartet with roles: {roles[0]}")

    try:
        outcome = eval_mgr.run_roles(roles, lineup_indices, num_players, seed)
        print(f"  ✓ Completed {outcome.total_games} games")

        # Print results
        for lineup_idx, lineup_result in enumerate(outcome.lineups):
            print(f"  Lineup {lineup_idx}:")
            for policy_id, stats in lineup_result.per_policy.items():
                print(f"    Policy {policy_id}: wins={stats.total_wins}, "
                      f"returns={stats.total_returns:.2f}, games={stats.num_games}")

        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_run_larger_batch(eval_mgr):
    """Test 3: Run larger batch (16 quartets)."""
    print("\n=== TEST 3: Run Larger Batch (16 quartets) ===")

    # 16 quartets
    roles = [[10, 10, 10, 10] for _ in range(16)]
    lineup_indices = list(range(16))
    num_players = 4
    seed = 42

    print(f"  Running {len(roles)} quartets")

    try:
        outcome = eval_mgr.run_roles(roles, lineup_indices, num_players, seed)
        print(f"  ✓ Completed {outcome.total_games} games")
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_mixed_policies():
    """Test 4: Load TWO policies and run mixed games."""
    print("\n=== TEST 4: Load Two Policies (Mixed Games) ===")

    eval_mgr = lb.EvalManager()
    eval_mgr.set_max_env_batch(4)
    eval_mgr.set_inference_batch_size(4)

    # Load gen_1 and gen_2
    for gen in [1, 2]:
        ckpt_path = f"checkpoints/test76/gen_{gen}/final.pth"
        print(f"Loading checkpoint: {ckpt_path}")

        data = torch.load(ckpt_path, map_location="cpu")
        state_dict = data if not isinstance(data, dict) else data.get("model_state_dict", data)
        try:
            _pad_obs_encoder_input_inplace(state_dict, target_in_dim=16)
        except Exception as e:
            print(f"[WARN] Padding obs_encoder.0.weight failed for {ckpt_path}: {e}")

        policy_id = 10 + gen
        eval_mgr.load_model(policy_id, state_dict, ckpt_path)
        print(f"  ✓ Loaded policy {policy_id}")

    eval_mgr.finalize_model_loading()
    print("  ✓ Finalized model loading")

    # Mixed quartet: gen_1 vs gen_2
    roles = [[11, 11, 12, 12]]
    lineup_indices = [0]
    num_players = 4
    seed = 42

    print(f"  Running mixed quartet: {roles[0]}")

    try:
        outcome = eval_mgr.run_roles(roles, lineup_indices, num_players, seed)
        print(f"  ✓ Completed {outcome.total_games} games")

        for lineup_idx, lineup_result in enumerate(outcome.lineups):
            print(f"  Lineup {lineup_idx}:")
            for policy_id, stats in lineup_result.per_policy.items():
                print(f"    Policy {policy_id}: wins={stats.total_wins}, "
                      f"returns={stats.total_returns:.2f}, games={stats.num_games}")

        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("EvalManager Minimal Unit Tests")
    print("=" * 60)

    # Test 1: Load single policy
    eval_mgr = test_1_load_single_policy()
    if eval_mgr is None:
        print("\n✗ TEST 1 FAILED - Stopping")
        exit(1)

    # Test 2: Run minimal roles (1 quartet)
    success = test_2_run_minimal_roles(eval_mgr)
    if not success:
        print("\n✗ TEST 2 FAILED - Device assert likely here")
        exit(1)

    # Test 3: Run larger batch
    success = test_3_run_larger_batch(eval_mgr)
    if not success:
        print("\n✗ TEST 3 FAILED")
        exit(1)

    # Test 4: Mixed policies
    success = test_4_mixed_policies()
    if not success:
        print("\n✗ TEST 4 FAILED")
        exit(1)

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
