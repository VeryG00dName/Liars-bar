"""
Test script for Deep CFR implementation.
Tests each component individually to isolate issues.
"""

import torch
import argparse
from pathlib import Path

from src.model.deep_cfr_model import DeepCFRNetwork
from src.misc.lb import CFRManager


def test_model_creation(obs_dim=16, hidden_dim=256):
    """Test 1: Create Deep CFR model"""
    print("\n=== Test 1: Model Creation ===")
    try:
        model = DeepCFRNetwork(
            obs_dim=obs_dim,
            action_dim=7,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout_rate=0.1,
            max_seq_length=480,
            num_agent_types=4,
            swiglu_ffn_dim=384,
            use_gradient_checkpointing=False,
        )
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_model_forward(model, batch_size=2, seq_len=10, obs_dim=16):
    """Test 2: Forward pass"""
    print("\n=== Test 2: Model Forward Pass ===")
    try:
        obs_sequence = torch.randn(batch_size, seq_len, obs_dim)
        action_sequence = torch.randint(0, 7, (batch_size, seq_len))
        agent_types = torch.randint(0, 4, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        model.eval()
        with torch.no_grad():
            output = model(obs_sequence, action_sequence, agent_types, positions, padding_mask)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {obs_sequence.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected: [{batch_size}, {seq_len}, 7]")
        assert output.shape == (batch_size, seq_len, 7), f"Shape mismatch: {output.shape}"
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_state_dict_export(model):
    """Test 3: Export state dict"""
    print("\n=== Test 3: State Dict Export ===")
    try:
        state_dict = model.state_dict()
        print(f"✓ State dict exported")
        print(f"  - Keys: {len(state_dict)}")
        
        # Check for critical keys
        critical_keys = ['gate_obs.0.weight', 'gate_action.0.weight', 'gate_agent.0.weight',
                        'output_head.weight', 'lut_act_kind', 'lut_count', 'lut_table_flag']
        for key in critical_keys:
            if key in state_dict:
                print(f"  ✓ Found {key}: {state_dict[key].shape}")
            else:
                print(f"  ✗ Missing {key}")
        
        return state_dict
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_cfr_manager_creation(num_players=4, max_depth=100, num_workers=32):
    """Test 4: Create CFR Manager"""
    print("\n=== Test 4: CFR Manager Creation ===")
    try:
        cfr_manager = CFRManager(num_players, max_depth, num_workers)
        print(f"✓ CFR Manager created")
        print(f"  - Players: {num_players}")
        print(f"  - Max depth: {max_depth}")
        print(f"  - Workers: {num_workers}")
        return cfr_manager
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_load_advantage_network(cfr_manager, state_dict):
    """Test 5: Load model into CFR manager"""
    print("\n=== Test 5: Load Advantage Network ===")
    try:
        # Convert to unordered_map format for C++
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
        cfr_manager.load_advantage_network(state_dict_cpu)
        print(f"✓ Model loaded into CFR manager")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_single_traversal(cfr_manager, num_traversals=5):
    """Test 6: Run CFR traversals"""
    print(f"\n=== Test 6: Run {num_traversals} CFR Traversals ===")
    try:
        nodes_visited = cfr_manager.run_cfr_traversal(num_traversals)
        print(f"✓ Traversals completed")
        print(f"  - Nodes visited: {nodes_visited}")
        return nodes_visited
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_collect_samples(cfr_manager):
    """Test 7: Collect training samples"""
    print("\n=== Test 7: Collect Samples ===")
    try:
        regret_samples, strategy_samples = cfr_manager.collect_samples()
        
        print(f"✓ Samples collected")
        print(f"  Regret samples:")
        for key in regret_samples:
            print(f"    - {key}: {regret_samples[key].shape}")
        
        print(f"  Strategy samples:")
        for key in strategy_samples:
            print(f"    - {key}: {strategy_samples[key].shape}")
        
        return regret_samples, strategy_samples
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_data_format(regret_samples, strategy_samples):
    """Test 8: Validate data format"""
    print("\n=== Test 8: Validate Data Format ===")
    try:
        # Check regret samples
        assert 'observations' in regret_samples, "Missing observations in regret_samples"
        assert 'targets' in regret_samples, "Missing targets in regret_samples"
        assert 'padding_mask' in regret_samples, "Missing padding_mask in regret_samples"
        
        # Check strategy samples
        assert 'observations' in strategy_samples, "Missing observations in strategy_samples"
        assert 'targets' in strategy_samples, "Missing targets in strategy_samples"
        assert 'padding_mask' in strategy_samples, "Missing padding_mask in strategy_samples"
        
        # Check shapes match
        reg_obs_shape = regret_samples['observations'].shape
        reg_tgt_shape = regret_samples['targets'].shape
        assert reg_obs_shape[0] == reg_tgt_shape[0], "Batch size mismatch in regret samples"
        
        print(f"✓ Data format valid")
        print(f"  - Regret batch: {reg_obs_shape[0]} samples")
        print(f"  - Strategy batch: {strategy_samples['observations'].shape[0]} samples")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def run_all_tests(args):
    """Run all tests in sequence"""
    print("=" * 60)
    print("DEEP CFR COMPONENT TESTING")
    print("=" * 60)
    
    try:
        # Test 1-3: Model creation and export
        model = test_model_creation(obs_dim=args.obs_dim, hidden_dim=args.hidden_dim)
        test_model_forward(model, batch_size=2, seq_len=10, obs_dim=args.obs_dim)
        state_dict = test_state_dict_export(model)
        
        # Test 4-5: CFR manager setup
        cfr_manager = test_cfr_manager_creation(
            num_players=args.num_players,
            max_depth=args.max_depth,
            num_workers=args.num_workers
        )
        test_load_advantage_network(cfr_manager, state_dict)
        
        # Test 6: Run traversals
        nodes_visited = test_single_traversal(cfr_manager, num_traversals=args.num_traversals)
        
        # Test 7-8: Collect and validate samples
        regret_samples, strategy_samples = test_collect_samples(cfr_manager)
        test_data_format(regret_samples, strategy_samples)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST SUITE FAILED")
        print("=" * 60)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--num_traversals", type=int, default=5)
    
    args = parser.parse_args()
    run_all_tests(args)
