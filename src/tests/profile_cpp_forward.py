#!/usr/bin/env python3
"""
Profile C++ forward_packed_cpp to identify memory bottlenecks.
"""
import sys
from pathlib import Path
from collections.abc import Mapping
import torch

# Import torch first to load libraries
print("Loading PyTorch...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Import C++ extension
print("\nImporting C++ extension...")
from src.misc import lb
from src.model.ppo_reactive_model import PPOReactiveModel


# -----------------------------
# Utilities
# -----------------------------
def create_dummy_inputs(batch_size=128, seq_len=256, obs_dim=9, device="cuda"):
    """Create dummy inputs matching real inference patterns."""
    print(f"\nCreating dummy inputs: batch={batch_size}, seq_len={seq_len}, obs_dim={obs_dim}")

    obs_sequence    = torch.randn(batch_size, seq_len, obs_dim, dtype=torch.float16, device=device)
    action_sequence = torch.randint(0, 11, (batch_size, seq_len), dtype=torch.long, device=device)
    agent_types     = torch.randint(0, 3,  (batch_size, seq_len), dtype=torch.long, device=device)
    positions       = torch.randint(0, 4,  (batch_size, seq_len), dtype=torch.long, device=device)
    padding_mask    = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    # Mark some positions as padding (last 30% of sequence)
    pad_start = int(seq_len * 0.7)
    padding_mask[:, pad_start:] = True

    return obs_sequence, action_sequence, agent_types, positions, padding_mask


def _extract_state_dict(ckpt) -> Mapping:
    """
    Return ONLY the inner mapping that actually contains tensors.
    Supports your save format: {'model_state_dict': <real state_dict>}.
    """
    if hasattr(ckpt, "state_dict") and callable(getattr(ckpt, "state_dict")):
        return ckpt.state_dict()

    if not isinstance(ckpt, Mapping):
        raise TypeError("Expected a Mapping or nn.Module checkpoint.")

    # Your format first
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], Mapping):
        return ckpt["model_state_dict"]

    # Other common wrappers
    for k in ("state_dict", "model", "module"):
        if k in ckpt and isinstance(ckpt[k], Mapping):
            return ckpt[k]

    # policy_nets (pick first mapping if present)
    if "policy_nets" in ckpt and isinstance(ckpt["policy_nets"], Mapping):
        for _, sub in ckpt["policy_nets"].items():
            if isinstance(sub, Mapping):
                for k in ("model_state_dict", "state_dict", "model", "module"):
                    if k in sub and isinstance(sub[k], Mapping):
                        return sub[k]
                return sub
            if hasattr(sub, "state_dict"):
                return sub.state_dict()

    # If the top-level itself is a tensor mapping, use it
    if any(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt

    raise TypeError("Could not find a tensor mapping in checkpoint (try 'model_state_dict').")


def _to_device_fp16(sd: Mapping, device="cuda") -> dict:
    """
    Convert only floating-point tensors to fp16 and move all tensors to `device`.
    Returns a plain dict[str, Tensor or original non-tensor].
    """
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=torch.float16, non_blocking=True).contiguous()
            else:
                out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _infer_arch_cfg(ckpt: Mapping) -> dict:
    """
    Build the small arch config dict expected by the C++ path, with safe defaults.
    If your checkpoints store hyperparams, pick them up here.
    """
    defaults = dict(
        num_layers=2,
        num_heads=4,
        hidden_dim=256,
        num_experts=8,
        top_k=2,
        count_pad=4,
        tflag_pad=3,
    )

    for key in ("arch", "hparams", "config", "model_config"):
        if isinstance(ckpt, Mapping) and key in ckpt and isinstance(ckpt[key], Mapping):
            src = ckpt[key]
            for k in list(defaults.keys()):
                if k in src and isinstance(src[k], (int, float)):
                    defaults[k] = int(src[k])
    return defaults

LUT_KEYS = {"lut_act_kind", "lut_count", "lut_table_flag"}

def ensure_batched_(wd):
    for k, t in list(wd.items()):
        if k in LUT_KEYS:
            # keep LUTs 1-D Long[11]
            wd[k] = t.to(torch.long).contiguous().view(-1)
            continue

        # Embeddings: [vocab, dim] -> [1, vocab, dim]
        if k.endswith("embedding.weight") and t.ndim == 2:
            wd[k] = t.unsqueeze(0)
            continue

        # Linear weights: [out, in] -> [1, out, in]
        if k.endswith(".weight") and t.ndim == 2:
            wd[k] = t.unsqueeze(0)
            continue

        # Biases / LayerNorm params: [dim] -> [1, dim]
        if (k.endswith(".bias") or k.endswith(".weight")) and t.ndim == 1:
            wd[k] = t.unsqueeze(0)
            continue
    return wd

def _split_attention_weights(wd: dict) -> dict:
    """Split fused in-proj tensors into q/k/v and add alias keys.

    - For keys ending with '.self_attn.in_proj_weight': split [B, 3*H, H]
      into q_proj.weight/k_proj.weight/v_proj.weight (each [B, H, H]).
    - For keys ending with '.self_attn.in_proj_bias': split [B, 3*H]
      into q_proj.bias/k_proj.bias/v_proj.bias (each [B, H]).
    - Add aliases by removing the first occurrence of '.self_attn.' in keys.
    """
    def replace_suffix(s: str, from_s: str, to_s: str) -> str:
        idx = s.rfind(from_s)
        if idx == -1:
            return s
        return s[:idx] + to_s + s[idx + len(from_s):]

    def drop_self_attn(s: str) -> str:
        return s.replace('.self_attn.', '.', 1)

    keys = list(wd.keys())
    for key in keys:
        if key.endswith('.self_attn.in_proj_weight') and key in wd:
            w = wd[key]
            if not torch.is_tensor(w):
                continue
            if w.dim() not in (2, 3):
                continue
            dim = 1 if w.dim() == 3 else 0
            if (w.size(dim) % 3) != 0:
                continue
            q, k, v = torch.chunk(w, 3, dim=dim)
            q_key = replace_suffix(key, 'in_proj_weight', 'q_proj.weight')
            k_key = replace_suffix(key, 'in_proj_weight', 'k_proj.weight')
            v_key = replace_suffix(key, 'in_proj_weight', 'v_proj.weight')
            wd[q_key] = q.contiguous()
            wd[k_key] = k.contiguous()
            wd[v_key] = v.contiguous()
            wd[drop_self_attn(q_key)] = wd[q_key]
            wd[drop_self_attn(k_key)] = wd[k_key]
            wd[drop_self_attn(v_key)] = wd[v_key]
            continue

        if key.endswith('.self_attn.in_proj_bias') and key in wd:
            b = wd[key]
            if not torch.is_tensor(b):
                continue
            if b.dim() not in (1, 2):
                continue
            dim = 1 if b.dim() == 2 else 0
            if (b.size(dim) % 3) != 0:
                continue
            q, k, v = torch.chunk(b, 3, dim=dim)
            q_key = replace_suffix(key, 'in_proj_bias', 'q_proj.bias')
            k_key = replace_suffix(key, 'in_proj_bias', 'k_proj.bias')
            v_key = replace_suffix(key, 'in_proj_bias', 'v_proj.bias')
            wd[q_key] = q.contiguous()
            wd[k_key] = k.contiguous()
            wd[v_key] = v.contiguous()
            wd[drop_self_attn(q_key)] = wd[q_key]
            wd[drop_self_attn(k_key)] = wd[k_key]
            wd[drop_self_attn(v_key)] = wd[v_key]
            continue

        if '.self_attn.' in key and key in wd:
            alias = drop_self_attn(key)
            if alias != key and alias not in wd:
                wd[alias] = wd[key]
    return wd

# -----------------------------
# Checkpoint loading
# -----------------------------
def load_checkpoint_weights(checkpoint_path, device="cuda"):
    """Load weights from a checkpoint file and prepare a GPU FP16 weight dict."""
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Infer architecture config
    arch_cfg = _infer_arch_cfg(checkpoint)
    print(f"Model architecture: {PPOReactiveModel} (using arch_cfg: {arch_cfg})")

    # Extract and convert weights
    print("Converting weights to FP16 and moving to GPU...")
    state_dict = _extract_state_dict(checkpoint)
    
    processed_dict = _to_device_fp16(state_dict, device=device)

    # Ensure we pass ONLY tensors and a plain dict to the C++ helpers
    tensor_only = {k: v for k, v in processed_dict.items() if torch.is_tensor(v)}

    # Pre-stack MoE weights (returns modified dict)
    print("Pre-stacking MoE expert weights...")
    tensor_only = lb.prestack_moe_expert_weights(tensor_only, arch_cfg["num_layers"], arch_cfg["num_experts"])

    # This is what the C++ forward expects
    weight_dict = dict(tensor_only)

    for k in ("lut_act_kind", "lut_count", "lut_table_flag"):
        t = weight_dict[k]
    assert t.dtype == torch.long and t.dim() == 1 and t.numel() == 11, \
        f"{k} must be 1-D long[11], got {tuple(t.shape)}, {t.dtype}"
    weight_dict = ensure_batched_(weight_dict)
    for k in [
    "obs_encoder.0.weight", "obs_encoder.0.bias",
    "obs_encoder.1.weight", "obs_encoder.1.bias",
    "act_kind_embedding.weight", "count_embedding.weight",
    "table_flag_embedding.weight", "agent_embedding.weight",
    "position_embedding.weight",
    "gate_obs.0.weight", "gate_obs.0.bias", "gate_obs.2.weight", "gate_obs.2.bias",
    "action_heads.0.weight", "action_heads.0.bias",
    "opp_action_heads.0.weight", "opp_action_heads.0.bias",
]:
        t = weight_dict[k]
        print(k, tuple(t.shape), t.dtype)

    # Add fixed buffers (LUTs) AFTER adding batch dim
    # LUTs should NOT have batch dimension - they're shared lookup tables
    print("Adding fixed buffers (LUTs)...")
    lb.add_fixed_buffers(weight_dict, device)

    return weight_dict, arch_cfg


def replicate_weights_for_batch(weight_dict, batch_size, device="cuda"):
    """
    Replicate weights to simulate multiple samples using the same model.

    This mimics what EvalManager does: if you have 128 requests all for the same
    policy, it uses index_select to replicate the weights 128 times.
    """
    if batch_size == 1:
        return weight_dict

    print(f"Replicating weights for batch_size={batch_size} (simulates {batch_size} inference requests)...")

    # Create indices: [0, 0, 0, ..., 0] (batch_size times)
    # This simulates all requests using policy_id=0
    indices = torch.zeros(batch_size, dtype=torch.long, device=device)

    replicated = {}
    for key, tensor in weight_dict.items():
        # For LUTs (unbatched, 1D), don't replicate
        if tensor.ndim == 1:
            replicated[key] = tensor
        else:
            # Use index_select to replicate along batch dimension (dim=0)
            replicated[key] = tensor.index_select(0, indices)

    return replicated


def load_checkpoint_weights_old(checkpoint_path, device="cuda"):
    """Load weights from a checkpoint file and prepare a GPU FP16 weight dict."""
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Infer architecture config
    arch_cfg = _infer_arch_cfg(checkpoint)
    print(f"Model architecture: {PPOReactiveModel} (using arch_cfg: {arch_cfg})")

    # Extract and convert weights
    print("Converting weights to FP16 and moving to GPU...")
    state_dict = _extract_state_dict(checkpoint)

    processed_dict = _to_device_fp16(state_dict, device=device)

    # Ensure we pass ONLY tensors and a plain dict to the C++ helpers
    tensor_only = {k: v for k, v in processed_dict.items() if torch.is_tensor(v)}

    # Pre-stack MoE weights (returns modified dict)
    print("Pre-stacking MoE expert weights...")
    tensor_only = lb.prestack_moe_expert_weights(tensor_only, arch_cfg["num_layers"], arch_cfg["num_experts"])

    # This is what the C++ forward expects
    weight_dict = dict(tensor_only)

    # IMPORTANT: Add batch dimension (forward_packed_cpp expects batched weights)
    # Single model: [out_dim, in_dim] -> [1, out_dim, in_dim]
    # This represents having 1 model in the policy pool
    print("Adding batch dimension to all weights (simulates 1 model in pool)...")

    # Add fixed buffers (LUTs) AFTER adding batch dim
    # LUTs should NOT have batch dimension - they're shared lookup tables
    print("Adding fixed buffers (LUTs)...")
    lb.add_fixed_buffers(weight_dict, device)

    print(f"Total weight tensors: {len(weight_dict)}")

    # Calculate total memory
    total_params = sum(t.numel() for t in weight_dict.values())
    total_bytes  = sum(t.numel() * t.element_size() for t in weight_dict.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Total memory: {total_bytes / 1024**3:.2f} GB")

    return weight_dict, arch_cfg


# -----------------------------
# Profiling
# -----------------------------
def profile_forward_pass(checkpoint_path, batch_size=128, seq_len=256):
    """Profile the C++ forward pass with PyTorch profiler."""
    device = "cuda"

    # Load weights (keep at batch_size=1 for broadcasting)
    weight_dict, arch = load_checkpoint_weights(checkpoint_path, device)

    # NOTE: We do NOT replicate weights. The C++ forward function now supports
    # broadcasting when weight batch dim is 1 but input batch dim is larger.
    # This saves massive amounts of memory (10+ GB with batch_size=512).

    # Create dummy inputs
    obs_seq, act_seq, agent_types, positions, padding_mask = create_dummy_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    # Warm up
    print("\nWarming up (1 iteration)...")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        policy_indices = torch.zeros(obs_seq.size(0), dtype=torch.long, device=device)
        _ = lb.forward_packed_cpp(
            obs_seq, act_seq, agent_types, positions,
            weight_dict, policy_indices, padding_mask,
            arch["num_layers"],
            arch["num_heads"],
            arch["hidden_dim"],
            arch["num_experts"],
            arch["top_k"],
            arch["count_pad"],
            arch["tflag_pad"],
        )
    torch.cuda.synchronize()
    warmup_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Warmup peak memory: {warmup_mem:.2f} GB")

    # Profile
    print("\n" + "=" * 80)
    print("PROFILING FORWARD PASS")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            policy_indices = torch.zeros(obs_seq.size(0), dtype=torch.long, device=device)
            action_logits, opp_logits, state_values, win_logits = lb.forward_packed_cpp(
                obs_seq, act_seq, agent_types, positions,
                weight_dict, policy_indices, padding_mask,
                arch["num_layers"],
                arch["num_heads"],
                arch["hidden_dim"],
                arch["num_experts"],
                arch["top_k"],
                arch["count_pad"],
                arch["tflag_pad"],
            )

    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak memory during profiling: {peak_mem:.2f} GB")

    # Print results sorted by memory
    print("\n" + "=" * 80)
    print("TOP MEMORY-CONSUMING OPERATIONS")
    print("=" * 80)
    print(
        prof.key_averages().table(
            sort_by="self_cuda_memory_usage",
            row_limit=30,
            top_level_events_only=False,
        )
    )

    print("\n" + "=" * 80)
    print("TOP TIME-CONSUMING OPERATIONS")
    print("=" * 80)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
            top_level_events_only=False,
        )
    )

    # Export to chrome trace for visualization
    trace_path = "forward_profile_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    print("View it at: chrome://tracing")

    return prof


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_cpp_forward.py <checkpoint_path> [batch_size] [seq_len]")
        print("\nExample:")
        print("  python profile_cpp_forward.py checkpoints/test80/gen_0/final.pth 128 256")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    # IMPORTANT: batch_size must match weight batch dimension (1 for single model)
    # C++ forward_packed_cpp is designed for batched MODELS, not batched SAMPLES
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seq_len    = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 80)
    print("C++ FORWARD PASS MEMORY PROFILER")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")

    try:
        profile_forward_pass(checkpoint_path, batch_size, seq_len)
        print("\n✅ Profiling completed successfully!")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
