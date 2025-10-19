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
    """Ensure a consistent leading batch dim for all weights.

    - Standard params: [out,in] -> [1,out,in]; [dim] -> [1,dim]
    - Fused attention in-proj_*: treat like standard (get batch dim at 0)
    - Split attention q/k/v: ensure batch dim at 0 for weights/biases
    - MoE stacked tensors: ensure shape [E, 1, ...] so policy batch (W) is at dim=1
    - LUTs remain 1-D
    - Idempotent: calling multiple times won't add extra dimensions
    """
    for k, t in list(wd.items()):
        if not torch.is_tensor(t):
            continue

        # LUTs stay 1-D
        if k in LUT_KEYS:
            wd[k] = t.to(torch.long).contiguous().view(-1)
            continue

        # MoE stacked weights/biases: enforce [E, 1, ...]
        if ".moe.experts.w1" in k or ".moe.experts.w2" in k:
            if t.ndim == 3:  # [E, out, in] -> [E,1,out,in]
                wd[k] = t.unsqueeze(1).contiguous()
                continue
            if t.ndim == 4 and t.size(0) == 1 and t.size(1) > 1:
                # Reorder [1,E,out,in] -> [E,1,out,in]
                wd[k] = t.permute(1, 0, 2, 3).contiguous()
                continue
            if t.ndim == 4:
                # Already batched correctly [E, 1, out, in]
                continue

        if ".moe.experts.b1" in k or ".moe.experts.b2" in k:
            if t.ndim == 2:  # [E, dim] -> [E,1,dim]
                wd[k] = t.unsqueeze(1).contiguous()
                continue
            if t.ndim == 3 and t.size(0) == 1 and t.size(1) > 1:
                # Reorder [1,E,dim] -> [E,1,dim]
                wd[k] = t.permute(1, 0, 2).contiguous()
                continue
            if t.ndim == 3:
                # Already batched correctly [E, 1, dim]
                continue

        # Fused attention params: add batch dim at 0
        if k.endswith("in_proj_weight") and t.ndim == 2:
            wd[k] = t.unsqueeze(0).contiguous()
            continue
        if k.endswith("in_proj_weight") and t.ndim == 3:
            # Already batched [1, 3*H, H]
            continue
        if k.endswith("in_proj_bias") and t.ndim == 1:
            wd[k] = t.unsqueeze(0).contiguous()
            continue
        if k.endswith("in_proj_bias") and t.ndim == 2:
            # Already batched [1, 3*H]
            continue

        # Split attention q/k/v weights and biases
        if (k.endswith("q_proj.weight") or k.endswith("k_proj.weight") or k.endswith("v_proj.weight")) and t.ndim == 2:
            wd[k] = t.unsqueeze(0).contiguous()
            continue
        if (k.endswith("q_proj.weight") or k.endswith("k_proj.weight") or k.endswith("v_proj.weight")) and t.ndim == 3:
            # Already batched [1, H, H]
            continue
        if (k.endswith("q_proj.bias") or k.endswith("k_proj.bias") or k.endswith("v_proj.bias")) and t.ndim == 1:
            wd[k] = t.unsqueeze(0).contiguous()
            continue
        if (k.endswith("q_proj.bias") or k.endswith("k_proj.bias") or k.endswith("v_proj.bias")) and t.ndim == 2:
            # Already batched [1, H]
            continue

        # Embeddings: [vocab, dim] -> [1, vocab, dim]
        if k.endswith("embedding.weight") and t.ndim == 2:
            wd[k] = t.unsqueeze(0).contiguous()
            continue
        if k.endswith("embedding.weight") and t.ndim == 3:
            # Already batched [1, vocab, dim]
            continue

        # Standard linear/LayerNorm params
        # LayerNorm: 1D [H] -> 2D [1, H]
        # Linear: 2D [out, in] -> 3D [1, out, in]
        # Biases: 1D [dim] -> 2D [1, dim]

        if k.endswith(".weight") and t.ndim == 1:
            # LayerNorm weight: [H] -> [1, H]
            wd[k] = t.unsqueeze(0).contiguous()
            continue
        if k.endswith(".weight") and t.ndim == 2:
            # Could be: batched LayerNorm [1, H] or unbatched Linear [out, in]
            # Explicitly check if it's a known LayerNorm layer
            is_layernorm_layer = (
                "norm" in k.lower() or
                "obs_encoder.1" in k or  # obs_encoder.1 is LayerNorm
                ".1.weight" in k and "encoder" in k  # Common pattern: encoder.1 = LayerNorm
            )

            if is_layernorm_layer and t.size(0) == 1:
                # Batched LayerNorm [1, H] - already batched, skip
                continue
            elif t.size(0) == t.size(1):
                # Square matrix (attention): [H, H] -> [1, H, H]
                wd[k] = t.unsqueeze(0).contiguous()
                continue
            else:
                # Linear layer (including [1, in]): [out, in] -> [1, out, in]
                wd[k] = t.unsqueeze(0).contiguous()
                continue
        if k.endswith(".weight") and t.ndim == 3:
            # Already batched [1, out, in]
            continue

        if k.endswith(".bias") and t.ndim == 1:
            # Any bias: [dim] -> [1, dim]
            wd[k] = t.unsqueeze(0).contiguous()
            continue
        if k.endswith(".bias") and t.ndim == 2:
            # Already batched [1, dim]
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

    # --- DIAG: after _to_device_fp16 ---
    def _find_first_key(keys, predicate):
        for k in keys:
            if predicate(k):
                return k
        return None
    attn_fused_key_0 = _find_first_key(tensor_only.keys(),
                                        lambda k: k.endswith('.self_attn.in_proj_weight') and '.layers.0.' in k)
    moe_sample_w1_0 = _find_first_key(tensor_only.keys(),
                                      lambda k: '.layers.0.moe.experts.' in k and k.endswith('.0.weight'))
    if attn_fused_key_0:
        t = tensor_only[attn_fused_key_0]
        print(f"[DIAG] after _to_device_fp16: {attn_fused_key_0} -> {tuple(t.shape)}, {t.dtype}")
    if moe_sample_w1_0:
        t = tensor_only[moe_sample_w1_0]
        print(f"[DIAG] after _to_device_fp16: {moe_sample_w1_0} -> {tuple(t.shape)}, {t.dtype}")

    # 1) Pre-stack MoE on unbatched inputs
    print("Pre-stacking MoE expert weights...")
    weight_dict = lb.prestack_moe_expert_weights(dict(tensor_only), arch_cfg["num_layers"], arch_cfg["num_experts"])

    # --- DIAG: after prestack (unbatched), check stacked MoE tensors ---
    stacked_w1_key_0 = f"transformer.layers.0.moe.experts.w1"
    if stacked_w1_key_0 in weight_dict:
        t = weight_dict[stacked_w1_key_0]
        print(f"[DIAG] after prestack: {stacked_w1_key_0} -> {tuple(t.shape)}, {t.dtype}")

    # 2) Ensure batch dims:
    weight_dict = ensure_batched_(weight_dict)

    # --- DIAG: after ensure_batched_ ---
    if attn_fused_key_0 and attn_fused_key_0 in weight_dict:
        t = weight_dict[attn_fused_key_0]
        print(f"[DIAG] after ensure_batched_: {attn_fused_key_0} -> {tuple(t.shape)}, {t.dtype}")
    if stacked_w1_key_0 in weight_dict:
        t = weight_dict[stacked_w1_key_0]
        print(f"[DIAG] after ensure_batched_: {stacked_w1_key_0} -> {tuple(t.shape)}, {t.dtype}")

    # 3) Split fused attention weights into q/k/v and add aliases
    _split_attention_weights(weight_dict)

    # 4) Ensure batch dims again to cover newly added q/k/v params
    weight_dict = ensure_batched_(weight_dict)

    # --- DIAG: after split, check Q/K/V for layer 0 ---
    for base in ("transformer.layers.0.self_attn.", "transformer.layers.0."):
        for proj in ("q_proj", "k_proj", "v_proj"):
            wk = base + f"{proj}.weight"
            bk = base + f"{proj}.bias"
            if wk in weight_dict:
                tw = weight_dict[wk]
                print(f"[DIAG] after split: {wk} -> {tuple(tw.shape)}, {tw.dtype}")
            if bk in weight_dict:
                tb = weight_dict[bk]
                print(f"[DIAG] after split: {bk} -> {tuple(tb.shape)}, {tb.dtype}")
                
    print("\nlayers that don't have a batch dimension after processing:")
    for k, t in weight_dict.items():
        shape = tuple(t.shape)
        if len(shape) == 0 or shape[0] != 1:  # print only if first dim is not 1
            print(k, shape, t.dtype)

    print("\nLayerNorm weights (checking dimensions):")
    for k, t in weight_dict.items():
        if "norm" in k.lower() and ("weight" in k or "bias" in k):
            print(f"  {k}: {tuple(t.shape)}, {t.dtype}")

    print("\nLinear weights (should be 3D [W, out, in]):")
    for k, t in weight_dict.items():
        is_layernorm = "norm" in k.lower() or "obs_encoder.1" in k or (".1.weight" in k and "encoder" in k)
        if k.endswith(".weight") and not ("embedding" in k or is_layernorm or k in LUT_KEYS):
            if t.ndim != 3:
                print(f"  ❌ {k}: {tuple(t.shape)}, {t.dtype} (expected 3D)")
            # Don't print all the ✓ to reduce clutter
            # else:
            #     print(f"  ✓ {k}: {tuple(t.shape)}, {t.dtype}")

    # Add fixed buffers (LUTs) after other processing; keep them 1-D
    print("Adding fixed buffers (LUTs)...")
    weight_dict = lb.add_fixed_buffers(weight_dict, device)

    # Sanity-check LUTs
    for k in ("lut_act_kind", "lut_count", "lut_table_flag"):
        t = weight_dict[k]
        assert t.dtype == torch.long and t.dim() == 1 and t.numel() == 11, \
            f"{k} must be 1-D long[11], got {tuple(t.shape)}, {t.dtype}"

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
        # DIAG: Input tensor shapes prior to warm-up call
        print(f"[DIAG] obs_seq: {tuple(obs_seq.shape)}, {obs_seq.dtype}")
        print(f"[DIAG] act_seq: {tuple(act_seq.shape)}, {act_seq.dtype}")
        print(f"[DIAG] agent_types: {tuple(agent_types.shape)}, {agent_types.dtype}")
        print(f"[DIAG] positions: {tuple(positions.shape)}, {positions.dtype}")
        print(f"[DIAG] padding_mask: {tuple(padding_mask.shape)}, {padding_mask.dtype}")
        policy_indices = torch.zeros(obs_seq.size(0), dtype=torch.long, device=device)
        print(f"[DIAG] policy_indices: {tuple(policy_indices.shape)}, {policy_indices.dtype}")
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

    # ------------------------------------------------------------------
    # DIAG: Comprehensive weight dict logging before profiling
    # ------------------------------------------------------------------
    print("\n[DIAG] FINAL WEIGHT DICT SUMMARY (key -> shape, dtype)")
    suspicious = []
    for k in sorted(weight_dict.keys()):
        t = weight_dict[k]
        shp = tuple(t.shape)
        print(f"  {k}: {shp}, {t.dtype}")
        if any(dim == 171 for dim in shp):
            suspicious.append((k, shp))

    if suspicious:
        print("[DIAG][WARN] Tensors containing dimension 171 detected:")
        for k, shp in suspicious:
            print(f"  -> {k}: {shp}")

    # Check attention Q/K/V shapes against hidden_dim
    H = int(arch["hidden_dim"]) if "hidden_dim" in arch else None
    if H is not None:
        def _expect_weight(shp):
            return len(shp) == 3 and shp[0] == 1 and shp[1] == H and shp[2] == H
        def _expect_bias(shp):
            return len(shp) == 2 and shp[0] == 1 and shp[1] == H
        for k in sorted(weight_dict.keys()):
            if "q_proj.weight" in k or "k_proj.weight" in k or "v_proj.weight" in k:
                if not _expect_weight(tuple(weight_dict[k].shape)):
                    print(f"[DIAG][MISMATCH] {k} expected [1,{H},{H}], got {tuple(weight_dict[k].shape)}")
            if "q_proj.bias" in k or "k_proj.bias" in k or "v_proj.bias" in k:
                if not _expect_bias(tuple(weight_dict[k].shape)):
                    print(f"[DIAG][MISMATCH] {k} expected [1,{H}], got {tuple(weight_dict[k].shape)}")

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
            # DIAG: Input tensor shapes just before profiled call
            print(f"[DIAG] obs_seq: {tuple(obs_seq.shape)}, {obs_seq.dtype}")
            print(f"[DIAG] act_seq: {tuple(act_seq.shape)}, {act_seq.dtype}")
            print(f"[DIAG] agent_types: {tuple(agent_types.shape)}, {agent_types.dtype}")
            print(f"[DIAG] positions: {tuple(positions.shape)}, {positions.dtype}")
            policy_indices = torch.zeros(obs_seq.size(0), dtype=torch.long, device=device)
            print(f"[DIAG] policy_indices: {tuple(policy_indices.shape)}, {policy_indices.dtype}")
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
        print("  python profile_cpp_forward.py checkpoints/test75/gen_0/final.pth 128 256")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    # IMPORTANT: batch_size must match weight batch dimension (1 for single model)
    # C++ forward_packed_cpp is designed for batched MODELS and batched INPUTS.
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
