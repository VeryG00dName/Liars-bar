#!/usr/bin/env python3
"""
Speed-focused profiler for C++ forward_packed vs PyTorch forward.

Usage:
    python -m src.tests.profile_cpp_forward <checkpoint> [batch] [seq_len] [num_policies] [mode]

    mode: 'cpp', 'pytorch', or 'both' (default: 'cpp')

Example:
    python -m src.tests.profile_cpp_forward checkpoints/test76/gen_58/final.pth 512 64 1 both
"""
import sys
from pathlib import Path
from typing import Dict, Tuple
import torch

from src.misc import lb
from src.tests import test_utils as tu

# ============================================================================
# Weight Loading
# ============================================================================

def split_attention_weights(wd: dict) -> dict:
    """Split fused in_proj tensors into q/k/v."""
    keys = list(wd.keys())
    for key in keys:
        if key.endswith('.self_attn.in_proj_weight') and key in wd:
            w = wd[key]
            if not torch.is_tensor(w) or w.dim() not in (2, 3):
                continue
            dim = 1 if w.dim() == 3 else 0
            if (w.size(dim) % 3) != 0:
                continue
            q, k, v = torch.chunk(w, 3, dim=dim)
            base = key.replace('.self_attn.in_proj_weight', '')
            wd[f"{base}.self_attn.q_proj.weight"] = q.contiguous()
            wd[f"{base}.self_attn.k_proj.weight"] = k.contiguous()
            wd[f"{base}.self_attn.v_proj.weight"] = v.contiguous()
            wd[f"{base}.q_proj.weight"] = q.contiguous()
            wd[f"{base}.k_proj.weight"] = k.contiguous()
            wd[f"{base}.v_proj.weight"] = v.contiguous()

        if key.endswith('.self_attn.in_proj_bias') and key in wd:
            b = wd[key]
            if not torch.is_tensor(b) or b.dim() not in (1, 2):
                continue
            dim = 1 if b.dim() == 2 else 0
            if (b.size(dim) % 3) != 0:
                continue
            q, k, v = torch.chunk(b, 3, dim=dim)
            base = key.replace('.self_attn.in_proj_bias', '')
            wd[f"{base}.self_attn.q_proj.bias"] = q.contiguous()
            wd[f"{base}.self_attn.k_proj.bias"] = k.contiguous()
            wd[f"{base}.self_attn.v_proj.bias"] = v.contiguous()
            wd[f"{base}.q_proj.bias"] = q.contiguous()
            wd[f"{base}.k_proj.bias"] = k.contiguous()
            wd[f"{base}.v_proj.bias"] = v.contiguous()

    return wd

def load_cpp_weights(checkpoint_path: Path, num_policies: int = 1, device: str = "cuda") -> Tuple[Dict, Dict]:
    """Load weights for C++ forward_packed (batched format)."""
    # Arch config
    arch = {
        "num_layers": 2,
        "num_heads": 4,
        "hidden_dim": 256,
        "num_experts": 8,
        "top_k": 2,
        "count_pad": 4,
        "tflag_pad": 3,
    }

    # Load and pad checkpoint
    sd = tu.load_checkpoint_weights(checkpoint_path)
    sd = tu.pad_model_weights(sd, pad_obs_to=16)

    # Prestack MoE
    sd = lb.prestack_moe_expert_weights(sd, arch["num_layers"], arch["num_experts"])

    # Split attention
    sd = split_attention_weights(sd)

    # Batch dimension handling
    LUT_KEYS = {"lut_act_kind", "lut_count", "lut_table_flag"}

    if num_policies == 1:
        # Single policy: add batch dim [1, ...]
        for k in list(sd.keys()):
            if k in LUT_KEYS:
                continue
            sd[k] = sd[k].unsqueeze(0)
    else:
        # Multi-policy: stack along dim 0
        sd_stacked = {}
        for k, v in sd.items():
            if k in LUT_KEYS:
                continue
            copies = [v for _ in range(num_policies)]
            sd_stacked[k] = torch.stack(copies, dim=0)
        sd = sd_stacked

    # Move to device + FP16
    for k in sd:
        if sd[k].is_floating_point():
            sd[k] = sd[k].to(device, dtype=torch.float16).contiguous()
        else:
            sd[k] = sd[k].to(device).contiguous()

    # Create MoE pointers
    sd = lb.create_moe_weight_pointers(sd, arch["num_layers"], arch["num_experts"])

    # Add LUTs
    sd = lb.add_fixed_buffers(sd, device)

    return sd, arch

def load_pytorch_model(checkpoint_path: Path, device: str = "cuda") -> torch.jit.ScriptModule:
    """Load TorchScript model for pure PyTorch performance comparison.

    Note: We load the TorchScript version (not PPOReactiveModel) because
    PPOReactiveModel has custom C++ forward/backward that would skew the
    PyTorch vs C++ comparison.
    """
    # Derive TorchScript path from checkpoint path
    script_path = checkpoint_path.parent / (checkpoint_path.stem + "_traced.pt")

    if not script_path.exists():
        raise FileNotFoundError(
            f"TorchScript model not found at {script_path}\n"
            f"Please create it first:\n"
            f"  python -m src.training.trace_checkpoint {checkpoint_path}"
        )

    # Load TorchScript model
    model = torch.jit.load(str(script_path), map_location=device)
    model.eval()

    return model

# ============================================================================
# Speed Profiling
# ============================================================================

def benchmark_cpp_forward(
    weights: Dict,
    arch: Dict,
    batch_size: int,
    seq_len: int,
    num_policies: int = 1,
    num_iters: int = 100,
    device: str = "cuda"
) -> Dict:
    """Benchmark C++ forward_packed.

    Note: C++ expects 16-dim obs (padded), but only first 9 dims are meaningful.
    We create 16-dim obs with zeros in dims 9-15 to match real inference.
    """
    # Create 9-dim inputs first (semantic content)
    # Enable C++ forward workspace cache so forward_packed reuses CUTLASS buffers
    lb.enable_forward_moe_workspace_cache(
        arch["num_layers"], batch_size, seq_len, arch["hidden_dim"], arch["top_k"]
    )
    obs_9d, acts, agent_types, positions, action_masks, padding_mask = tu.create_dummy_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        obs_dim=9,
        device=device,
        dtype=torch.float16,
    )

    # Pad obs from 9 to 16 dims (zeros in dims 9-15, matching real data)
    obs = torch.cat([
        obs_9d,
        torch.zeros(batch_size, seq_len, 7, dtype=torch.float16, device=device)
    ], dim=-1)

    # Policy indices
    if num_policies == 1:
        policy_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        policy_indices = torch.randint(0, num_policies, (batch_size,), dtype=torch.long, device=device)

    # Profile using test_utils
    timers_dict = {}
    
    def forward_fn():
        with torch.no_grad():
            result = lb.forward_packed(
                obs, acts, agent_types, positions,
                weights, policy_indices, padding_mask,
                arch["num_layers"], arch["num_heads"], arch["hidden_dim"],
                arch["num_experts"], arch["top_k"],
                arch["count_pad"], arch["tflag_pad"],
            )
            # Handle new return signature: (action_logits, opp_logits, state_values, win_logits, timers_dict)
            if isinstance(result, tuple) and len(result) == 5:
                # Accumulate timers across iterations
                for key, value in result[4].items():
                    timers_dict[key] = timers_dict.get(key, 0) + value
                return result[:4]  # Return just the tensors for profiling
            return result

    results = tu.profile_cuda_execution(forward_fn, warmup=10, iterations=num_iters)
    # Free cached workspaces after benchmark
    lb.disable_forward_moe_workspace_cache()
    
    # Add timers to results
    results["timers"] = timers_dict
    return results

def benchmark_pytorch_forward(
    model: torch.jit.ScriptModule,
    batch_size: int,
    seq_len: int,
    num_iters: int = 100,
    device: str = "cuda"
) -> Dict:
    """Benchmark PyTorch TorchScript model.forward().

    Note: TorchScript model was trained with 9-dim obs (before C++ padding).
    We create 9-dim inputs directly to match training.
    TorchScript model uses float32 (not float16 like C++).
    """
    # Create 9-dim inputs (TorchScript expects this)
    obs, acts, agent_types, positions, action_masks, padding_mask = tu.create_dummy_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        obs_dim=9,  # TorchScript expects original 9-dim obs
        device=device,
        dtype=torch.float32,  # TorchScript uses float32, not float16
    )

    # Profile using test_utils
    def forward_fn():
        with torch.no_grad():
            # TorchScript signature: forward(obs, acts, agent_types, positions, action_masks, padding_mask)
            return model(obs, acts, agent_types, positions, action_masks, padding_mask)

    results = tu.profile_cuda_execution(forward_fn, warmup=10, iterations=num_iters)
    return results

def print_benchmark_results(name: str, results: Dict, batch_size: int, seq_len: int):
    """Print benchmark results with throughput metrics."""
    print(f"\n{'='*80}")
    print(f"{name.upper()} BENCHMARK")
    print(f"{'='*80}")

    ms_per_iter = results["avg_time_ms"]
    iters_per_sec = results["throughput_per_sec"]
    tokens_per_batch = batch_size * seq_len
    tokens_per_sec = tokens_per_batch * iters_per_sec

    print(f"CUDA time per iteration: {ms_per_iter:.3f} ms")
    print(f"Throughput: {iters_per_sec:.1f} batches/sec")
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"Throughput: {tokens_per_sec/1e6:.2f} M tokens/sec")
    print(f"\nBatch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Tokens per batch: {tokens_per_batch:,}")
    
    # Print timer breakdown if available
    if "timers" in results and results["timers"]:
        print(f"\n{'='*80}")
        print("PER-LAYER TIMING BREAKDOWN (microseconds, per iteration)")
        print(f"{'='*80}")
        
        timers = results["timers"]
        num_iters = results.get("iterations", 100)
        
        # Average timers per iteration
        avg_timers = {k: v / num_iters for k, v in timers.items()}
        
        # Sort by time descending
        sorted_timers = sorted(avg_timers.items(), key=lambda x: -x[1])
        
        total_us = sum(avg_timers.values())
        
        for key, value_us in sorted_timers:
            ms = value_us / 1000.0
            percentage = (value_us / total_us * 100.0) if total_us > 0 else 0.0
            print(f"  {key:<40}: {value_us:>12.0f} us ({ms:>8.3f} ms) [{percentage:>5.1f}%]")
        
        total_ms = total_us / 1000.0
        print(f"\n  {'TOTAL':<40}: {total_us:>12.0f} us ({total_ms:>8.3f} ms)")

# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    num_policies = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    mode = sys.argv[5] if len(sys.argv) > 5 else "cpp"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = "cuda"
    num_iters = 100

    print(f"{'='*80}")
    print("C++ vs PYTORCH SPEED PROFILER")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Num policies: {num_policies}")
    print(f"Mode: {mode}")
    print(f"Iterations: {num_iters}")

    cpp_results = None
    pytorch_results = None

    # C++ benchmarking
    if mode in ("cpp", "both"):
        print(f"\n{'='*80}")
        print("Loading C++ weights...")
        print(f"{'='*80}")
        weights, arch = load_cpp_weights(checkpoint_path, num_policies, device)
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory after loading: {mem_gb:.2f} GB")

        print("\nBenchmarking C++ forward_packed...")
        cpp_results = benchmark_cpp_forward(weights, arch, batch_size, seq_len, num_policies, num_iters, device)
        print_benchmark_results("C++ forward_packed", cpp_results, batch_size, seq_len)

    # PyTorch benchmarking
    if mode in ("pytorch", "both"):
        print(f"\n{'='*80}")
        print("Loading PyTorch model...")
        print(f"{'='*80}")
        torch.cuda.reset_peak_memory_stats()
        model = load_pytorch_model(checkpoint_path, device)
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory after loading: {mem_gb:.2f} GB")

        print("\nBenchmarking PyTorch forward...")
        pytorch_results = benchmark_pytorch_forward(model, batch_size, seq_len, num_iters, device)
        print_benchmark_results("PyTorch forward", pytorch_results, batch_size, seq_len)

    # Comparison
    if mode == "both" and cpp_results and pytorch_results:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        cpp_ms = cpp_results["avg_time_ms"]
        pytorch_ms = pytorch_results["avg_time_ms"]
        speedup = pytorch_ms / cpp_ms

        print(f"C++ forward_packed:  {cpp_ms:.3f} ms/iter")
        print(f"PyTorch forward:     {pytorch_ms:.3f} ms/iter")
        print(f"Speedup (PyTorch/C++): {speedup:.2f}x")

        if speedup < 1.0:
            print(f"\n⚠️  C++ is {1/speedup:.2f}x SLOWER than PyTorch for this config!")
            print("    Consider using PyTorch forward for training (single policy).")
            print("    Keep C++ for rollouts (multi-policy batching).")
        else:
            print(f"\n✅ C++ is {speedup:.2f}x faster than PyTorch!")

if __name__ == "__main__":
    main()
