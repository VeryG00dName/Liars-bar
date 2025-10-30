#pragma once

#include <torch/extension.h>
#include <unordered_map>
#include <chrono>

/**
 * lb_kernels.h - Raw CUDA/cuBLASLt kernel primitives
 *
 * This file contains pure computational kernels with no autograd.
 * These are the single source of truth for both training and inference.
 *
 * Naming convention: lb::kernels::function_name
 */

namespace lb {
namespace kernels {

// Fused linear epilogues supported by the indexed batched linear helper.
enum class IndexedLinearEpilogue {
    Bias,
    BiasGELU,
};

/**
 * Indexed Batched Embedding
 *
 * Performs batched embedding lookup where each batch element uses a different weight matrix.
 *
 * @param weight_cache [W, vocab, hidden] - Stacked embedding matrices for W policies
 * @param indices [B, T] - Token indices to look up
 * @param policy_indices [B] - Which policy's weights to use for each batch element
 * @return [B, T, hidden] - Embedding outputs
 *
 * Performance: Uses custom CUDA kernel with coalesced memory access.
 * Precision: FP16 I/O, no accumulation needed (simple lookup).
 */
torch::Tensor indexed_batched_embedding(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices);

/**
 * Indexed Batched Layer Normalization
 *
 * Performs batched layer norm where each batch element uses different gamma/beta parameters.
 *
 * @param input [B, T, H] - Input activations
 * @param gamma_cache [W, H] - Scale parameters for W policies
 * @param beta_cache [W, H] - Bias parameters for W policies
 * @param policy_indices [B] - Which policy's parameters to use for each batch
 * @param eps Layer norm epsilon (default: 1e-5)
 * @return [B, T, H] - Normalized outputs
 *
 * Performance: Custom CUDA kernel with welford online algorithm for mean/variance.
 * Precision: FP32 accumulation for mean/variance, FP16 I/O.
 * Numerical: Identical to PyTorch LayerNorm within FP16 tolerance.
 */
torch::Tensor indexed_batched_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps = 1e-5);

/**
 * Indexed Batched Linear
 *
 * Performs batched linear transformation where each batch element uses different weights.
 * This is the SINGLE SOURCE OF TRUTH for both training and inference.
 *
 * @param input [B, T, in_dim] - Input activations
 * @param weight_cache [W, out_dim, in_dim] - Weight matrices for W policies
 * @param bias_cache [W, out_dim] - Bias vectors for W policies
 * @param policy_indices [B] - Which policy's weights to use
 * @param timers Optional profiling timer map
 * @param epilogue Fused operation after GEMM (Bias or BiasGELU)
 * @return [B, T, out_dim] - Output activations
 *
 * Performance: Uses cuBLASLt grouped GEMM for maximum throughput.
 *              Benchmarked faster than torch::matmul for heterogeneous batches.
 * Precision: FP32 accumulation, FP16 I/O.
 * Epilogue: Bias add and optional GELU are fused into GEMM epilogue.
 *           GELU uses erf approximation with exact constants matching training.
 */
torch::Tensor indexed_batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, std::chrono::microseconds>& timers,
    IndexedLinearEpilogue epilogue = IndexedLinearEpilogue::Bias);

/**
 * Grouped FFN GEMM Forward (Low-level MoE kernel interface)
 *
 * Runs the grouped FFN GEMM forward pass using CUTLASS grouped kernels.
 * This is used internally by the MoE autograd function.
 *
 * Each array argument contains device pointers for the operands.
 * Applies per-row routing weights in the GEMM epilogue (fused).
 *
 * @param input_ptrs Device pointers to input matrices (one per group)
 * @param w1_ptrs Device pointers to W1 weight matrices
 * @param b1_ptrs Device pointers to b1 bias vectors
 * @param w2_ptrs Device pointers to W2 weight matrices
 * @param b2_ptrs Device pointers to b2 bias vectors
 * @param output_ptrs Device pointers to output matrices
 * @param routing_weight_ptrs Device pointers to routing weights
 * @param m_sizes Number of tokens per group
 * @param policy_ids Policy IDs per group (for logging)
 * @param expert_ids Expert IDs per group (for logging)
 * @param token_offsets Token offsets per group (for scatter)
 * @param group_count Number of groups
 * @param hidden_dim Hidden dimension (H)
 * @param ffn_dim FFN intermediate dimension (F)
 *
 * Performance: Uses CUTLASS grouped GEMM with Tensor Core acceleration.
 * Precision: FP32 accumulation, FP16 I/O.
 * Epilogue: W1 bias + GELU, then W2 bias + routing weight scaling.
 */
void grouped_ffn_gemm_forward(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* b2_ptrs,
    const uintptr_t* output_ptrs,
    const uintptr_t* routing_weight_ptrs,
    const int64_t* m_sizes,
    const int64_t* policy_ids,
    const int64_t* expert_ids,
    const int64_t* token_offsets,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim);

} // namespace kernels
} // namespace lb
