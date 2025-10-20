#pragma once

#include <torch/torch.h>
#include <tuple>
#include <string>
#include <unordered_map>
#include <chrono>

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

// Overload with detailed timers map for granular profiling
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed_cpp(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights, // Shape [W, ...]
    const torch::Tensor& policy_indices, // Shape [B]
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad,
    std::unordered_map<std::string, std::chrono::microseconds>& timers
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
