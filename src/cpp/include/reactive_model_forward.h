#pragma once

#include <torch/torch.h>
#include <tuple>
#include <string>

/**
 * Stateless forward pass for PPOReactiveModel with batched weights.
 *
 * This function replaces TorchScript inference by directly computing the forward pass
 * in C++ using the PyTorch C++ API and custom CUDA kernels. It accepts batched weights
 * to support efficient heterogeneous opponent inference.
 *
 * @param obs_sequence Observation sequences [B, T, obs_dim]
 * @param action_sequence Action sequences [B, T] (long tensor)
 * @param agent_types Agent type IDs [B, T] (long tensor)
 * @param positions Position IDs [B, T] (long tensor)
 * @param batched_weights Small batched weight cache [W, ...]. Keys follow the pattern:
 *                "module.submodule.weight" or "module.submodule.bias".
 * @param policy_indices Indices selecting a weight entry per sample [B].
 * @param padding_mask Optional padding mask [B, T] (bool tensor, True = padding)
 * @param num_layers Number of transformer layers
 * @param num_heads Number of attention heads
 * @param hidden_dim Hidden dimension size
 * @param num_experts Number of MoE experts
 * @param top_k Top-K experts to activate per token
 * @param count_pad Padding index for count embeddings (default: 4)
 * @param tflag_pad Padding index for table flag embeddings (default: 3)
 *
 * @return Tuple of (action_logits, opp_logits, state_values, win_logits)
 *         - action_logits: [B, T, action_dim]
 *         - opp_logits: [B, T, action_dim]
 *         - state_values: [B, T, 1]
 *         - win_logits: [B, T, 1]
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed_cpp(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights, // Shape [W, ...]
    const torch::Tensor& policy_indices, // Shape [B]
    const torch::optional<torch::Tensor>& padding_mask = torch::nullopt,
    int64_t num_layers = 2,
    int64_t num_heads = 4,
    int64_t hidden_dim = 256,
    int64_t num_experts = 8,
    int64_t top_k = 2,
    int64_t count_pad = 4,
    int64_t tflag_pad = 3
);

/**
 * Helper function: Batched linear transformation.
 *
 * Computes: output = input @ weight^T + bias
 * where weight and bias have a batch dimension.
 *
 * @param input Input tensor [B, T, in_dim]
 * @param weight Weight tensor [B, out_dim, in_dim]
 * @param bias Bias tensor [B, out_dim]
 * @return Output tensor [B, T, out_dim]
 */
torch::Tensor batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias
);

/**
 * Helper function: Batched layer normalization.
 *
 * Computes: output = (input - mean) / sqrt(var + eps) * weight + bias
 * where weight and bias have a batch dimension.
 *
 * @param input Input tensor [B, T, dim]
 * @param weight Weight tensor [B, dim]
 * @param bias Bias tensor [B, dim]
 * @param eps Epsilon for numerical stability
 * @return Output tensor [B, T, dim]
 */
torch::Tensor batched_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps = 1e-5
);

/**
 * Helper function: Batched embedding lookup.
 *
 * Performs embedding lookup where embedding weights have a batch dimension.
 *
 * @param weight Embedding weights [B, vocab_size, embed_dim]
 * @param indices Indices to lookup [B, T] (long tensor)
 * @return Embedded tensor [B, T, embed_dim]
 */
torch::Tensor batched_embedding(
    const torch::Tensor& weight,
    const torch::Tensor& indices
);

/**
 * Helper function: Reduce per-expert head outputs using MoE routing weights.
 *
 * Applies weighted combination of expert outputs based on Top-K routing decisions.
 *
 * @param stacked Stacked expert outputs [B, T, num_experts, out_dim]
 * @param topk_indices Top-K expert indices [B, T, K]
 * @param topk_scores Top-K routing weights [B, T, K]
 * @return Reduced output [B, T, out_dim]
 */
torch::Tensor reduce_expert_heads(
    const torch::Tensor& stacked,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_scores
);
