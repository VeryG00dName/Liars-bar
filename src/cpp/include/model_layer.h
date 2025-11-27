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
namespace moe { struct MoEWorkspace; }
namespace model {

/**
 * Decompose actions into (kind, count, table_flag) using lookup tables.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
action_decomposition(
    const torch::Tensor& action_sequence,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t count_pad,
    int64_t tflag_pad);

/**
 * Compute all embeddings (observation, action factorized, agent, position).
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
 * Compute modality gates (obs/action/agent/position) using per-policy MLPs.
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

/**
 * Single transformer layer forward pass (non-autograd version).
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
    const torch::Tensor& w1_ptrs,
    const torch::Tensor& w2_ptrs,
    const torch::Tensor& b1_ptrs,
    const torch::Tensor& b2_ptrs,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t top_k,
    int64_t num_experts,
    int64_t num_policies,
    lb::moe::MoEWorkspace* workspace = nullptr,
    const torch::optional<torch::Tensor>& positions = torch::nullopt,
    bool use_rope = false,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Compute all per-expert heads.
 */
c10::Dict<std::string, torch::Tensor>
compute_heads(
    const torch::Tensor& transformer_output,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    int64_t num_experts,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Reduce per-expert head outputs using MoE routing weights.
 */
torch::Tensor reduce_expert_heads(
    const torch::Tensor& stacked,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_scores);

/**
 * Compute MoE group metadata (m_sizes, policy_ids, expert_ids, token_offsets)
 * from sorted (expert, policy) indices using GPU ops and return CPU tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_group_metadata(
    const torch::Tensor& sorted_expert_indices,
    const torch::Tensor& sorted_policy_indices);

/**
 * Device-side variant of moe_group_metadata that returns GPU tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_group_metadata_device(
    const torch::Tensor& sorted_expert_indices,
    const torch::Tensor& sorted_policy_indices);

// =============================================================================
// Dense (non-MoE) Architecture Support
// =============================================================================

/**
 * Check if the model is MoE-based or dense.
 *
 * @param batched_weights The batched weight dictionary
 * @return true if model has MoE layers, false for dense architecture
 */
bool is_moe_model(const c10::Dict<std::string, torch::Tensor>& batched_weights);

/**
 * Check if model uses RoPE. Prefers explicit RoPE buffers (inv_freq/cos_cache)
 * and falls back to absence of learned position embeddings.
 */
bool has_rope(const c10::Dict<std::string, torch::Tensor>& batched_weights);

/**
 * Check if model uses SwiGLU activation in FFN.
 */
bool has_swiglu(const c10::Dict<std::string, torch::Tensor>& batched_weights);

/**
 * Dense transformer layer forward pass (no MoE routing).
 * Uses standard dense FFN instead of MoE experts.
 *
 * @return Tuple of (output, dummy_gate_logits, dummy_topk_indices, dummy_topk_scores)
 *         Dummy values are returned for API compatibility with MoE version.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
transformer_layer_dense(
    const torch::Tensor& x,
    const torch::Tensor& policy_indices,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& linear1_weight,
    const torch::Tensor& linear1_bias,
    const torch::Tensor& linear2_weight,
    const torch::Tensor& linear2_bias,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    int64_t num_heads,
    int64_t hidden_dim,
    const torch::optional<torch::Tensor>& positions = torch::nullopt,
    bool use_rope = false,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Dense transformer layer forward pass using SwiGLU FFN.
 *
 * FFN path:
 *   gate = SiLU(W_gate x)
 *   up   = W_up x
 *   out  = W_down (gate ⊙ up)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
transformer_layer_dense_swiglu(
    const torch::Tensor& x,
    const torch::Tensor& policy_indices,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& w_gate_weight,
    const torch::Tensor& w_gate_bias,
    const torch::Tensor& w_up_weight,
    const torch::Tensor& w_up_bias,
    const torch::Tensor& w_down_weight,
    const torch::Tensor& w_down_bias,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    int64_t num_heads,
    int64_t hidden_dim,
    const torch::optional<torch::Tensor>& positions = torch::nullopt,
    bool use_rope = false,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Compute single (non-expert) output heads for dense models.
 *
 * @return Dict with keys: action_logits, opp_logits, state_values, win_logits
 */
c10::Dict<std::string, torch::Tensor>
compute_heads_dense(
    const torch::Tensor& transformer_output,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

} // namespace model
} // namespace lb
