#pragma once

#include <torch/extension.h>
#include <tuple>
#include <unordered_map>
#include <chrono>

/**
 * model_layer.h - Stateless model layer implementations
 *
 * This file contains non-autograd model layer logic (attention, MoE, etc.).
 * These are pure computational functions that can be called from either
 * autograd contexts or non-autograd inference.
 *
 * Naming convention: lb::model::function_name
 */

namespace lb {
namespace model {

/**
 * Single transformer layer forward pass (non-autograd version).
 *
 * Used for inference. Training uses the autograd variant with checkpointing.
 *
 * @return Tuple of (output, gate_logits, topk_indices, topk_scores)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
transformer_layer(
    const torch::Tensor& x,
    const torch::Tensor& policy_indices,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& gate_weight,
    const torch::Tensor& gate_bias,
    const torch::Tensor& w1_all,
    const torch::Tensor& w2_all,
    const torch::Tensor& b1_all,
    const torch::Tensor& b2_all,
    const torch::Tensor& w1_ptrs,
    const torch::Tensor& w2_ptrs,
    const torch::Tensor& b1_ptrs,
    const torch::Tensor& b2_ptrs,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t top_k,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Compute all embeddings (observation, action factorized, agent, position).
 *
 * Handles action decomposition into (kind, count, table_flag) and sums them.
 */
c10::Dict<std::string, torch::Tensor>
compute_embeddings(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t count_pad,
    int64_t tflag_pad,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Reduce per-expert head outputs using MoE routing weights.
 *
 * @param stacked [B, T, num_experts, out_dim] - Expert outputs
 * @param topk_indices [B, T, K] - Selected expert indices
 * @param topk_scores [B, T, K] - Routing weights
 * @return [B, T, out_dim] - Weighted combination
 */
torch::Tensor reduce_expert_heads(
    const torch::Tensor& stacked,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_scores);

/**
 * Compute modality gates (obs/action/agent/position) using per-policy MLPs.
 *
 * Returns keys:
 *  - "g_obs", "g_action", "g_agent", "g_position" (each [B, T, H])
 */
c10::Dict<std::string, torch::Tensor>
gating(
    const torch::Tensor& obs_embed,
    const torch::Tensor& action_embed,
    const torch::Tensor& agent_embed,
    const torch::Tensor& position_embed,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Fuse gated embeddings and apply a final LayerNorm.
 *
 * Returns keys:
 *  - "fused_raw": before norm
 *  - "combined": after norm
 */
c10::Dict<std::string, torch::Tensor>
fuse_embeddings(
    const torch::Tensor& g_obs,
    const torch::Tensor& g_action,
    const torch::Tensor& g_agent,
    const torch::Tensor& g_position,
    const torch::Tensor& obs_embed,
    const torch::Tensor& action_embed,
    const torch::Tensor& agent_embed,
    const torch::Tensor& position_embed,
    int64_t hidden_dim);

} // namespace model
} // namespace lb
