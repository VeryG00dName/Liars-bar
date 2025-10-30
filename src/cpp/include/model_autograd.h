#pragma once

#include <torch/extension.h>

/**
 * model_autograd.h - Autograd wrappers for model operations
 *
 * This file contains torch::autograd::Function wrappers that enable gradient flow
 * during training. All forward passes call the unified kernels from lb_kernels.h.
 *
 * Naming convention: lb::autograd::FunctionName
 */

namespace lb {
namespace autograd {

/**
 * Autograd wrapper for indexed batched embedding.
 *
 * Forward: Calls lb::kernels::indexed_batched_embedding
 * Backward: Accumulates gradients into weight_cache using index_add_
 *
 * @param weight_cache [W, vocab, H] - Embedding weight matrices
 * @param indices [B, T] - Token indices
 * @param policy_indices [B] - Policy selection indices
 * @return [B, T, H] - Embedded outputs
 */
torch::Tensor indexed_batched_embedding_autograd(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices);

/**
 * Autograd wrapper for indexed batched layer normalization.
 *
 * Forward: Calls lb::kernels::indexed_batched_layer_norm
 * Backward: Computes gradients for input, gamma_cache, beta_cache
 *
 * @param input [B, T, H] - Input activations
 * @param gamma_cache [W, H] - Scale parameters
 * @param beta_cache [W, H] - Bias parameters
 * @param policy_indices [B] - Policy selection indices
 * @param eps Layer norm epsilon
 * @return [B, T, H] - Normalized outputs
 */
torch::Tensor indexed_batched_layer_norm_autograd(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps = 1e-5);

/**
 * Autograd wrapper for indexed batched linear.
 *
 * Forward: Calls lb::kernels::indexed_batched_linear
 * Backward: Computes gradients for input, weight_cache, bias_cache
 *
 * @param input [B, T, in_dim] - Input activations
 * @param weight_cache [W, out_dim, in_dim] - Weight matrices
 * @param bias_cache [W, out_dim] - Bias vectors
 * @param policy_indices [B] - Policy selection indices
 * @return [B, T, out_dim] - Linear outputs
 */
torch::Tensor indexed_batched_linear_autograd(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices);

} // namespace autograd
} // namespace lb
