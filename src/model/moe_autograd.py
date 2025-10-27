"""
Custom autograd.Function for MoE FFN backward pass.

Wraps the C++ grouped_moe_backward implementation to enable training
with custom CUDA kernels that match rollout numerics exactly.

Phase 1: Weight gradients only (routing frozen).
Phase 2: Add routing gradients (not yet implemented).
"""

import torch
from typing import Tuple, List, Optional
from src.misc import lb  # C++ extension


class GroupedMoEFunction(torch.autograd.Function):
    """
    Custom autograd.Function for grouped MoE FFN layer.

    Forward: Computes Y = MoE(X) using C++ CUTLASS grouped GEMM.
    Backward: Computes gradients using custom CUDA kernels.

    Phase 1: Trains W1, W2, b1, b2 with frozen routing.
    Phase 2: Will add routing gradients (dr) when ready.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
        w1_weights: torch.Tensor,    # [num_policies, num_experts, ffn_dim, hidden_dim]
        w2_weights: torch.Tensor,    # [num_policies, num_experts, hidden_dim, ffn_dim]
        b1_biases: torch.Tensor,     # [num_policies, num_experts, ffn_dim]
        b2_biases: torch.Tensor,     # [num_policies, num_experts, hidden_dim]
        routing_scores: torch.Tensor,  # [batch_size, seq_len, num_experts]
        policy_indices: torch.Tensor,  # [batch_size, seq_len]
        top_k: int = 2,
        compute_routing_grads: bool = False  # Phase 1: False, Phase 2: True
    ) -> torch.Tensor:
        """
        Forward pass through grouped MoE FFN.

        Computes top-k expert routing, groups tokens by (policy, expert),
        and applies expert FFNs via CUTLASS grouped GEMM.

        Args:
            input_tensor: Input activations [B, T, H]
            w1_weights: First layer weights [P, E, F, H]
            w2_weights: Second layer weights [P, E, H, F]
            b1_biases: First layer biases [P, E, F]
            b2_biases: Second layer biases [P, E, H]
            routing_scores: Gating network scores [B, T, E]
            policy_indices: Policy ID per token [B, T]
            top_k: Number of experts to activate per token
            compute_routing_grads: If True, compute dr (Phase 2)

        Returns:
            output: MoE output [B, T, H]
        """
        batch_size, seq_len, hidden_dim = input_tensor.shape
        num_policies, num_experts, ffn_dim, _ = w1_weights.shape

        # Flatten batch and sequence dimensions
        input_flat = input_tensor.reshape(-1, hidden_dim)  # [B*T, H]
        routing_scores_flat = routing_scores.reshape(-1, num_experts)  # [B*T, E]
        policy_indices_flat = policy_indices.reshape(-1)  # [B*T]

        # Top-k routing
        topk_scores, topk_indices = torch.topk(routing_scores_flat, top_k, dim=-1)
        topk_weights = torch.softmax(topk_scores, dim=-1)  # [B*T, K]

        # Expand tokens: each token produces K routed tokens
        # Shape: [B*T*K, H], [B*T*K], [B*T*K], [B*T*K]
        num_tokens = input_flat.size(0)
        total_routed_tokens = num_tokens * top_k

        input_expanded = input_flat.unsqueeze(1).expand(-1, top_k, -1).reshape(total_routed_tokens, hidden_dim)
        expert_indices_flat = topk_indices.reshape(total_routed_tokens)
        routing_weights_flat = topk_weights.reshape(total_routed_tokens)
        policy_indices_expanded = policy_indices_flat.unsqueeze(1).expand(-1, top_k).reshape(total_routed_tokens)

        # Create original token indices for scatter-add in backward
        token_indices = torch.arange(num_tokens, device=input_flat.device, dtype=torch.long)
        token_indices_expanded = token_indices.unsqueeze(1).expand(-1, top_k).reshape(total_routed_tokens)

        # Sort by (policy_id, expert_id) for grouping
        sort_keys = policy_indices_expanded * num_experts + expert_indices_flat
        sorted_indices = torch.argsort(sort_keys)

        # Apply sorting
        input_grouped = input_expanded[sorted_indices]
        expert_indices_grouped = expert_indices_flat[sorted_indices]
        routing_weights_grouped = routing_weights_flat[sorted_indices]
        policy_indices_grouped = policy_indices_expanded[sorted_indices]
        token_indices_grouped = token_indices_expanded[sorted_indices]

        # Build group ranges: find start/count for each (policy, expert) pair
        sort_keys_sorted = sort_keys[sorted_indices]
        unique_keys, group_counts = torch.unique_consecutive(sort_keys_sorted, return_counts=True)

        # Decode unique keys back to (policy_id, expert_id)
        policy_ids = (unique_keys // num_experts).tolist()
        expert_ids = (unique_keys % num_experts).tolist()
        m_sizes = group_counts.tolist()

        # Compute token offsets
        token_offsets = [0]
        for count in m_sizes[:-1]:
            token_offsets.append(token_offsets[-1] + count)

        # Call C++ grouped forward binding and get both hidden_grouped and output_grouped
        input_grouped_half = input_grouped.to(torch.float16)
        routing_weights_f32 = routing_weights_grouped.to(torch.float32)
        hidden_grouped, output_grouped = lb.grouped_moe_forward(
            input_grouped_half,
            w1_weights,
            w2_weights,
            b1_biases,
            b2_biases,
            routing_weights_f32,
            m_sizes,
            policy_ids,
            expert_ids,
            token_offsets,
        )

        # Scatter-add back to original token order
        output_flat = torch.zeros(num_tokens, hidden_dim, dtype=torch.float16, device=input_tensor.device)
        output_flat.scatter_add_(0, token_indices_grouped.unsqueeze(-1).expand(-1, hidden_dim),
                                  output_grouped)

        # Reshape back to [B, T, H]
        output = output_flat.reshape(batch_size, seq_len, hidden_dim)

        # Save for backward
        ctx.save_for_backward(
            input_grouped,
            hidden_grouped,
            routing_weights_grouped,
            token_indices_grouped,
            expert_indices_grouped,
            w1_weights,
            w2_weights,
            b1_biases,
            b2_biases
        )
        ctx.m_sizes = m_sizes
        ctx.policy_ids = policy_ids
        ctx.expert_ids = expert_ids
        ctx.token_offsets = token_offsets
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.num_tokens = num_tokens
        ctx.compute_routing_grads = compute_routing_grads
        # Save top-k mapping for routing gradient backprop
        ctx.topk_indices = topk_indices  # [num_tokens, K]
        ctx.topk_weights = topk_weights  # [num_tokens, K]

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass using custom CUDA kernels.

        Calls C++ grouped_moe_backward to compute gradients.

        Args:
            grad_output: Upstream gradient [B, T, H]

        Returns:
            Tuple of gradients for all forward inputs.
            Phase 1: Only returns gradients for input, w1, w2, b1, b2.
            Phase 2: Will also return routing gradients.
        """
        # Retrieve saved tensors
        (input_grouped, hidden_grouped, routing_weights_grouped,
         token_indices_grouped, expert_indices_grouped,
         w1_weights, w2_weights, b1_biases, b2_biases) = ctx.saved_tensors

        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        num_tokens = ctx.num_tokens
        m_sizes = ctx.m_sizes
        policy_ids = ctx.policy_ids
        expert_ids = ctx.expert_ids
        token_offsets = ctx.token_offsets

        # Flatten grad_output (backward kernel expects FP16)
        grad_output_flat = grad_output.reshape(num_tokens, -1).to(torch.float16).contiguous()

        # Allocate gradient tensors (zero-initialized)
        grad_input = torch.zeros_like(grad_output_flat)
        grad_w1 = torch.zeros_like(w1_weights, dtype=torch.float32)
        grad_w2 = torch.zeros_like(w2_weights, dtype=torch.float32)
        grad_b1 = torch.zeros_like(b1_biases, dtype=torch.float32)
        grad_b2 = torch.zeros_like(b2_biases, dtype=torch.float32)

        # Call C++ backward
        lb.grouped_moe_backward(
            grad_output_flat,
            input_grouped,
            hidden_grouped,
            routing_weights_grouped,
            token_indices_grouped,
            w1_weights,
            w2_weights,
            b1_biases,
            b2_biases,
            m_sizes,
            policy_ids,
            expert_ids,
            token_offsets,
            grad_input,
            grad_w1,
            grad_w2,
            grad_b1,
            grad_b2
        )

        # Reshape grad_input back to [B, T, H]
        grad_input = grad_input.reshape(batch_size, seq_len, -1)

        # Optional: compute routing gradients and map to routing_scores
        grad_routing_scores = None
        if ctx.compute_routing_grads:
            # Build Gy in grouped order: each routed token uses gradient of its source token
            Gy_grouped = grad_output_flat.index_select(0, token_indices_grouped)

            # Recompute Y_pre_grouped = H @ W2^T + b2 per group (FP32 accumulate)
            hidden_dim = grad_output_flat.size(1)
            ffn_dim = hidden_grouped.size(1)
            device = hidden_grouped.device
            dtype_half = hidden_grouped.dtype

            # We'll compute in chunks per group to avoid extra kernels and cast at end
            Y_pre_grouped = torch.empty_like(Gy_grouped)
            offset = 0
            for g, M in enumerate(m_sizes):
                if M == 0:
                    continue
                pi = policy_ids[g]
                ei = expert_ids[g]
                # Slices
                H_g = hidden_grouped.narrow(0, offset, M).to(torch.float32)
                W2 = w2_weights[pi, ei].to(torch.float32)        # [H, F]
                b2 = b2_biases[pi, ei].to(torch.float32)         # [H]
                # Y_pre = H @ W2^T + b2
                Y_pre = torch.addmm(b2, H_g, W2.t())             # [M, H]
                Y_pre_grouped.narrow(0, offset, M).copy_(Y_pre.to(dtype=Y_pre_grouped.dtype))
                offset += M

            # dr per routed token: rowwise dot(Gy_grouped, Y_pre_grouped)
            dr_grouped = (Gy_grouped.to(torch.float32) * Y_pre_grouped.to(torch.float32)).sum(dim=1)

            # Map dr_grouped back to per-token K positions to apply softmax backward
            num_tokens = ctx.num_tokens
            topk_indices = ctx.topk_indices  # [num_tokens, K]
            topk_weights = ctx.topk_weights  # [num_tokens, K]
            K = topk_indices.size(1)

            # Build a quick map: for each (token, expert) -> k position
            # Since K is small (e.g., 2), linear search is fine
            # Prepare output tensor
            dr_token_k = torch.zeros((num_tokens, K), device=device, dtype=dr_grouped.dtype)

            # We have expert_indices_grouped and token_indices_grouped aligned with dr_grouped
            # Fill dr_token_k at the corresponding (token, kpos)
            # Compute kpos with a simple equality match per token
            # For efficiency, build a per-token dict via tensor ops
            # Expand for broadcasting
            # Note: using a Python loop over groups is acceptable since group_count << total tokens
            for i in range(dr_grouped.numel()):
                t = int(token_indices_grouped[i].item())
                e = int(expert_indices_grouped[i].item())
                # Find k position for expert e in topk_indices[t]
                kpos = (topk_indices[t] == e).nonzero(as_tuple=False)
                if kpos.numel() == 0:
                    continue
                k = int(kpos[0].item())
                dr_token_k[t, k] = dr_grouped[i]

            # Softmax backward on top-k: g_s = w * (g_w - sum_j w_j g_w_j)
            gw = dr_token_k
            w = topk_weights
            dot = (gw * w).sum(dim=1, keepdim=True)
            g_s = w * (gw - dot)

            # Scatter into full expert dim
            num_experts = w2_weights.size(1)
            grad_routing_scores_flat = torch.zeros((num_tokens, num_experts), device=device, dtype=g_s.dtype)
            # scatter to expert dimension using topk indices
            grad_routing_scores_flat.scatter_(1, topk_indices, g_s)
            grad_routing_scores = grad_routing_scores_flat.reshape(batch_size, seq_len, num_experts)

        # Return gradients for all forward inputs
        # Order matches forward signature:
        # input_tensor, w1, w2, b1, b2, routing_scores, policy_indices, top_k, compute_routing_grads
        return (
            grad_input,      # input_tensor
            grad_w1,         # w1_weights
            grad_w2,         # w2_weights
            grad_b1,         # b1_biases
            grad_b2,         # b2_biases
            grad_routing_scores,  # routing_scores gradient (optional)
            None,            # policy_indices (no gradient)
            None,            # top_k (no gradient)
            None             # compute_routing_grads (no gradient)
        )


def grouped_moe_ffn(
    input_tensor: torch.Tensor,
    w1_weights: torch.Tensor,
    w2_weights: torch.Tensor,
    b1_biases: torch.Tensor,
    b2_biases: torch.Tensor,
    routing_scores: torch.Tensor,
    policy_indices: torch.Tensor,
    top_k: int = 2,
    compute_routing_grads: bool = False
) -> torch.Tensor:
    """
    Functional API for grouped MoE FFN with custom backward.

    Args:
        input_tensor: Input activations [B, T, H]
        w1_weights: First layer weights [P, E, F, H]
        w2_weights: Second layer weights [P, E, H, F]
        b1_biases: First layer biases [P, E, F]
        b2_biases: Second layer biases [P, E, H]
        routing_scores: Gating network scores [B, T, E]
        policy_indices: Policy ID per token [B, T]
        top_k: Number of experts to activate (default: 2)
        compute_routing_grads: Enable routing gradients for Phase 2 (default: False)

    Returns:
        output: MoE FFN output [B, T, H]
    """
    return GroupedMoEFunction.apply(
        input_tensor,
        w1_weights,
        w2_weights,
        b1_biases,
        b2_biases,
        routing_scores,
        policy_indices,
        top_k,
        compute_routing_grads
    )
