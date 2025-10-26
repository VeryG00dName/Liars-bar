#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace lb {
namespace moe {

/**
 * Phase 1 Backward Pass for Grouped MoE FFN.
 *
 * Implements custom CUDA backward kernels for cutlass_grouped_moe_forward(),
 * ensuring exact numerical parity between training and rollout.
 *
 * Phase 1 Scope:
 *   - Weight gradients: dW1, dW2, db1, db2
 *   - Input gradient: dX
 *   - Routing weights FROZEN (dr = 0, routing is not trained)
 *
 * Save-Minimal Policy:
 *   - Forward saves: X, H, routing_weights
 *   - Backward recomputes: Z (pre-GELU activation)
 *
 * Mathematical Flow:
 *   Forward:  Z = X @ W1^T + b1
 *             H = GELU(Z)
 *             Y_pre = H @ W2^T + b2
 *             Y = Y_pre ⊙ r
 *
 *   Backward: Gỹ = Gy ⊙ r                           [route-scale kernel]
 *             dW2 = Gỹ^T @ H                         [GEMM #1]
 *             db2 = sum(Gỹ, dim=0)                   [bias reduction]
 *             dH = Gỹ @ W2                           [GEMM #2]
 *             Z_recompute = X @ W1^T + b1            [GEMM #3 - recompute]
 *             dZ = dH ⊙ GELU'(Z_recompute)           [GELU' kernel]
 *             dW1 = dZ^T @ X                         [GEMM #4]
 *             db1 = sum(dZ, dim=0)                   [bias reduction]
 *             dX_grouped = dZ @ W1                   [GEMM #5]
 *             dX = scatter_add(dX_grouped, indices)  [scatter-add kernel]
 *
 * All kernels use FP16 I/O with FP32 accumulation for numerical stability.
 */

/**
 * Route-scale kernel: Gỹ = Gy ⊙ r
 *
 * Applies per-token routing weights to upstream gradient.
 *
 * @param grad_output Upstream gradient [total_tokens, hidden_dim] FP16
 * @param routing_weights Routing weights [total_tokens] FP32
 * @param grad_y_tilde Output scaled gradient [total_tokens, hidden_dim] FP16
 * @param total_tokens Total number of tokens
 * @param hidden_dim Hidden dimension
 * @param stream CUDA stream for async execution
 */
void route_scale_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& routing_weights,
    torch::Tensor& grad_y_tilde,
    int64_t total_tokens,
    int64_t hidden_dim,
    cudaStream_t stream = 0
);

/**
 * GELU backward kernel: dZ = dH ⊙ GELU'(Z)
 *
 * Applies GELU derivative using exact erf-based formula from gelu_constants.h.
 * CRITICAL: Must use same GELU formula as forward LinearCombinationGELU epilogue.
 *
 * @param grad_hidden Upstream gradient from dH [total_tokens, ffn_dim] FP16
 * @param z_recomputed Recomputed pre-GELU activation [total_tokens, ffn_dim] FP16
 * @param grad_z Output gradient [total_tokens, ffn_dim] FP16
 * @param total_tokens Total number of tokens
 * @param ffn_dim FFN intermediate dimension
 * @param stream CUDA stream for async execution
 */
void gelu_backward_kernel(
    const torch::Tensor& grad_hidden,
    const torch::Tensor& z_recomputed,
    torch::Tensor& grad_z,
    int64_t total_tokens,
    int64_t ffn_dim,
    cudaStream_t stream = 0
);

/**
 * Bias reduction kernel: db = sum(grad, dim=0)
 *
 * Computes bias gradients via reduction over token dimension.
 * Handles grouped data (tokens may not be contiguous in batch).
 *
 * @param grad Input gradient [total_tokens, feature_dim] FP16
 * @param grad_bias Output bias gradient [feature_dim] FP32 (accumulated)
 * @param total_tokens Total number of tokens
 * @param feature_dim Feature dimension (ffn_dim or hidden_dim)
 * @param stream CUDA stream for async execution
 */
void bias_reduction_kernel(
    const torch::Tensor& grad,
    torch::Tensor& grad_bias,
    int64_t total_tokens,
    int64_t feature_dim,
    cudaStream_t stream = 0
);

/**
 * Scatter-add kernel: dX_original[indices[i]] += dX_grouped[i]
 *
 * Accumulates gradients from grouped/sorted order back to original token order.
 * Required because forward pass groups tokens by (policy, expert).
 *
 * @param grad_x_grouped Input gradients in grouped order [total_tokens, hidden_dim] FP16
 * @param indices Mapping from grouped to original indices [total_tokens] int64
 * @param grad_x_original Output gradients in original order [batch_size, hidden_dim] FP16
 * @param total_tokens Total number of tokens (grouped)
 * @param hidden_dim Hidden dimension
 * @param stream CUDA stream for async execution
 */
void scatter_add_backward(
    const torch::Tensor& grad_x_grouped,
    const torch::Tensor& indices,
    torch::Tensor& grad_x_original,
    int64_t total_tokens,
    int64_t hidden_dim,
    cudaStream_t stream = 0
);

/**
 * Grouped MoE backward entry point.
 *
 * Orchestrates all backward kernels and GEMMs for Phase 1 training.
 *
 * Inputs (saved from forward):
 * @param grad_output Upstream gradient [batch_size, hidden_dim] FP16
 * @param input_grouped Saved input (grouped) [total_tokens, hidden_dim] FP16
 * @param hidden_grouped Saved hidden (grouped) [total_tokens, ffn_dim] FP16
 * @param routing_weights_grouped Saved routing weights [total_tokens] FP32
 * @param w1_weights W1 weights [num_policies, num_experts, ffn_dim, hidden_dim] FP16 ColumnMajor
 * @param w2_weights W2 weights [num_policies, num_experts, hidden_dim, ffn_dim] FP16 ColumnMajor
 * @param b1_biases b1 biases [num_policies, num_experts, ffn_dim] FP16
 * @param b2_biases b2 biases [num_policies, num_experts, hidden_dim] FP16
 * @param indices_grouped Mapping to original order [total_tokens] int64
 * @param group_metadata Grouping information (policy_ids, expert_ids, m_sizes, etc.)
 *
 * Outputs (gradients):
 * @param grad_input Gradient w.r.t. input [batch_size, hidden_dim] FP16
 * @param grad_w1 Gradient w.r.t. W1 [num_policies, num_experts, ffn_dim, hidden_dim] FP32
 * @param grad_w2 Gradient w.r.t. W2 [num_policies, num_experts, hidden_dim, ffn_dim] FP32
 * @param grad_b1 Gradient w.r.t. b1 [num_policies, num_experts, ffn_dim] FP32
 * @param grad_b2 Gradient w.r.t. b2 [num_policies, num_experts, hidden_dim] FP32
 *
 * Note: Routing gradients (dr) are NOT computed in Phase 1 (routing frozen).
 */
void cutlass_grouped_moe_backward(
    // Upstream gradient
    const torch::Tensor& grad_output,

    // Saved tensors from forward
    const torch::Tensor& input_grouped,
    const torch::Tensor& hidden_grouped,
    const torch::Tensor& routing_weights_grouped,
    const torch::Tensor& indices_grouped,

    // Forward weights
    const torch::Tensor& w1_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& b1_biases,
    const torch::Tensor& b2_biases,

    // Group metadata
    const std::vector<int64_t>& m_sizes,
    const std::vector<int64_t>& policy_ids,
    const std::vector<int64_t>& expert_ids,
    const std::vector<int64_t>& token_offsets,

    // Output gradients
    torch::Tensor& grad_input,
    torch::Tensor& grad_w1,
    torch::Tensor& grad_w2,
    torch::Tensor& grad_b1,
    torch::Tensor& grad_b2
);

} // namespace moe
} // namespace lb
