#include "reactive_model_forward.h"
#include "indexed_kernels.h"

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <algorithm>

namespace {

// Helper to get weight from dict with error checking
torch::Tensor get_weight(
    const c10::Dict<std::string, torch::Tensor>& weights,
    const std::string& key
) {
    auto it = weights.find(key);
    if (it == weights.end()) {
        std::stringstream ss;
        ss << "Missing weight key: " << key;
        throw std::runtime_error(ss.str());
    }
    return it->value();
}

} // anonymous namespace

// ============================================================================
// Layer-by-Layer Testing Functions
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
test_action_decomposition(
    const torch::Tensor& action_sequence,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t count_pad,
    int64_t tflag_pad
) {
    int64_t batch_size = action_sequence.size(0);
    int64_t seq_len = action_sequence.size(1);
    auto device = action_sequence.device();

    // Load LUT buffers
    auto lut_act_kind = get_weight(batched_weights, "lut_act_kind").to(device);
    auto lut_count = get_weight(batched_weights, "lut_count").to(device);
    auto lut_table_flag = get_weight(batched_weights, "lut_table_flag").to(device);

    // Decompose actions using LUTs
    auto action_long = action_sequence.to(torch::kLong);
    auto action_flat = action_long.flatten();

    auto act_kind_flat = lut_act_kind.index({action_flat});
    auto count_flat = lut_count.index({action_flat});
    auto table_flag_flat = lut_table_flag.index({action_flat});

    auto act_kind_ids = act_kind_flat.view({batch_size, seq_len}).to(torch::kLong);
    auto count_ids = count_flat.view({batch_size, seq_len}).to(torch::kLong);
    auto table_flag_ids = table_flag_flat.view({batch_size, seq_len}).to(torch::kLong);

    // Apply padding mask if provided
    if (padding_mask.has_value()) {
        auto pad_mask = padding_mask.value().to(torch::kBool);
        auto zero_like = torch::zeros_like(act_kind_ids);
        auto count_pad_tensor = torch::full_like(count_ids, count_pad, torch::kLong);
        auto tflag_pad_tensor = torch::full_like(table_flag_ids, tflag_pad, torch::kLong);

        act_kind_ids = torch::where(pad_mask, zero_like, act_kind_ids);
        count_ids = torch::where(pad_mask, count_pad_tensor, count_ids);
        table_flag_ids = torch::where(pad_mask, tflag_pad_tensor, table_flag_ids);
    }

    return std::make_tuple(act_kind_ids, count_ids, table_flag_ids);
}

c10::Dict<std::string, torch::Tensor>
test_embeddings(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& act_kind_ids,
    const torch::Tensor& count_ids,
    const torch::Tensor& table_flag_ids,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices
) {
    auto device = obs_sequence.device();
    auto policy_indices_for_ops = policy_indices.to(device).to(torch::kLong).contiguous();

    c10::Dict<std::string, torch::Tensor> result;

    // Observation encoding: Linear -> LayerNorm -> GELU
    // Break down into individual steps for debugging
    std::unordered_map<std::string, std::chrono::microseconds> dummy_timers;

    // Step 1: Linear
    auto obs_linear = indexed_batched_linear(
        obs_sequence,
        get_weight(batched_weights, "obs_encoder.0.weight"),
        get_weight(batched_weights, "obs_encoder.0.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    result.insert("obs_linear", obs_linear);

    // Step 2: LayerNorm
    auto obs_layernorm = indexed_batched_layer_norm(
        obs_linear,
        get_weight(batched_weights, "obs_encoder.1.weight"),
        get_weight(batched_weights, "obs_encoder.1.bias"),
        policy_indices_for_ops,
        /* eps = */ 1e-5  // Match PyTorch default
    );
    result.insert("obs_layernorm", obs_layernorm);

    // Step 3: GELU
    auto obs_encoded = torch::gelu(obs_layernorm);
    result.insert("obs_embed", obs_encoded);

    // Action embeddings (factorized)
    auto act_kind_embed = indexed_batched_embedding(
        get_weight(batched_weights, "act_kind_embedding.weight"),
        act_kind_ids,
        policy_indices_for_ops
    );
    auto count_embed = indexed_batched_embedding(
        get_weight(batched_weights, "count_embedding.weight"),
        count_ids,
        policy_indices_for_ops
    );
    auto table_flag_embed = indexed_batched_embedding(
        get_weight(batched_weights, "table_flag_embedding.weight"),
        table_flag_ids,
        policy_indices_for_ops
    );
    auto action_embed = act_kind_embed + count_embed + table_flag_embed;

    result.insert("act_kind_embed", act_kind_embed);
    result.insert("count_embed", count_embed);
    result.insert("table_flag_embed", table_flag_embed);
    result.insert("action_embed", action_embed);

    // Agent type embedding
    auto agent_embed = indexed_batched_embedding(
        get_weight(batched_weights, "agent_embedding.weight"),
        agent_types.to(torch::kLong),
        policy_indices_for_ops
    );
    result.insert("agent_embed", agent_embed);

    // Position embedding
    auto position_embed = indexed_batched_embedding(
        get_weight(batched_weights, "position_embedding.weight"),
        positions.to(torch::kLong),
        policy_indices_for_ops
    );
    result.insert("position_embed", position_embed);

    return result;
}

c10::Dict<std::string, torch::Tensor>
test_gating(
    const torch::Tensor& obs_embed,
    const torch::Tensor& action_embed,
    const torch::Tensor& agent_embed,
    const torch::Tensor& position_embed,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices
) {
    auto device = obs_embed.device();
    auto policy_indices_for_ops = policy_indices.to(device).to(torch::kLong).contiguous();

    c10::Dict<std::string, torch::Tensor> result;
    std::unordered_map<std::string, std::chrono::microseconds> dummy_timers;

    // Gate for observations
    auto hidden_g_obs = indexed_batched_linear(
        obs_embed,
        get_weight(batched_weights, "gate_obs.0.weight"),
        get_weight(batched_weights, "gate_obs.0.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    hidden_g_obs = torch::tanh(hidden_g_obs);
    auto g_obs = indexed_batched_linear(
        hidden_g_obs,
        get_weight(batched_weights, "gate_obs.2.weight"),
        get_weight(batched_weights, "gate_obs.2.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    g_obs = torch::sigmoid(g_obs);
    result.insert("g_obs", g_obs);

    // Gate for actions
    auto hidden_g_action = indexed_batched_linear(
        action_embed,
        get_weight(batched_weights, "gate_action.0.weight"),
        get_weight(batched_weights, "gate_action.0.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    hidden_g_action = torch::tanh(hidden_g_action);
    auto g_action = indexed_batched_linear(
        hidden_g_action,
        get_weight(batched_weights, "gate_action.2.weight"),
        get_weight(batched_weights, "gate_action.2.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    g_action = torch::sigmoid(g_action);
    result.insert("g_action", g_action);

    // Gate for agent types
    auto hidden_g_agent = indexed_batched_linear(
        agent_embed,
        get_weight(batched_weights, "gate_agent.0.weight"),
        get_weight(batched_weights, "gate_agent.0.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    hidden_g_agent = torch::tanh(hidden_g_agent);
    auto g_agent = indexed_batched_linear(
        hidden_g_agent,
        get_weight(batched_weights, "gate_agent.2.weight"),
        get_weight(batched_weights, "gate_agent.2.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    g_agent = torch::sigmoid(g_agent);
    result.insert("g_agent", g_agent);

    // Gate for positions
    auto hidden_g_position = indexed_batched_linear(
        position_embed,
        get_weight(batched_weights, "gate_position.0.weight"),
        get_weight(batched_weights, "gate_position.0.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    hidden_g_position = torch::tanh(hidden_g_position);
    auto g_position = indexed_batched_linear(
        hidden_g_position,
        get_weight(batched_weights, "gate_position.2.weight"),
        get_weight(batched_weights, "gate_position.2.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    g_position = torch::sigmoid(g_position);
    result.insert("g_position", g_position);

    return result;
}

c10::Dict<std::string, torch::Tensor>
test_fusion(
    const torch::Tensor& g_obs,
    const torch::Tensor& g_action,
    const torch::Tensor& g_agent,
    const torch::Tensor& g_position,
    const torch::Tensor& obs_embed,
    const torch::Tensor& action_embed,
    const torch::Tensor& agent_embed,
    const torch::Tensor& position_embed,
    int64_t hidden_dim
) {
    c10::Dict<std::string, torch::Tensor> result;

    // Ensure dtype consistency for embeddings
    auto act_embed_casted = action_embed.scalar_type() != obs_embed.scalar_type()
        ? action_embed.to(obs_embed.scalar_type())
        : action_embed;
    auto agent_embed_casted = agent_embed.scalar_type() != obs_embed.scalar_type()
        ? agent_embed.to(obs_embed.scalar_type())
        : agent_embed;
    auto position_embed_casted = position_embed.scalar_type() != obs_embed.scalar_type()
        ? position_embed.to(obs_embed.scalar_type())
        : position_embed;

    // Fused embedding
    auto fused = g_obs * obs_embed
               + g_action * act_embed_casted
               + g_agent * agent_embed_casted
               + g_position * position_embed_casted;
    result.insert("fused_raw", fused);

    // Final layer norm
    auto combined = torch::layer_norm(fused, {hidden_dim});
    result.insert("combined", combined);

    return result;
}

c10::Dict<std::string, torch::Tensor>
test_attention_layer(
    const torch::Tensor& x,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t layer_idx,
    int64_t num_heads,
    int64_t hidden_dim
) {
    auto device = x.device();
    auto policy_indices_for_ops = policy_indices.to(device).to(torch::kLong).contiguous();
    int64_t batch_size = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t head_dim = hidden_dim / num_heads;

    c10::Dict<std::string, torch::Tensor> result;
    std::unordered_map<std::string, std::chrono::microseconds> dummy_timers;

    std::string layer_prefix = "transformer.layers." + std::to_string(layer_idx);

    // Get combined in_proj weight and bias, then split into Q, K, V
    auto in_proj_weight = get_weight(batched_weights, layer_prefix + ".self_attn.in_proj_weight");
    auto in_proj_bias = get_weight(batched_weights, layer_prefix + ".self_attn.in_proj_bias");

    // Determine chunking dimension
    int64_t weight_chunk_dim = (in_proj_weight.dim() == 3) ? 1 : 0;
    int64_t bias_chunk_dim = (in_proj_bias.dim() == 2) ? 1 : 0;

    // Split into Q, K, V weights
    auto qkv_weights = in_proj_weight.chunk(3, weight_chunk_dim);
    auto qkv_biases = in_proj_bias.chunk(3, bias_chunk_dim);

    // Project to Q, K, V
    auto q = indexed_batched_linear(x, qkv_weights[0], qkv_biases[0], policy_indices_for_ops, dummy_timers);
    auto k = indexed_batched_linear(x, qkv_weights[1], qkv_biases[1], policy_indices_for_ops, dummy_timers);
    auto v = indexed_batched_linear(x, qkv_weights[2], qkv_biases[2], policy_indices_for_ops, dummy_timers);

    // Reshape for multi-head attention
    q = q.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    k = k.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    v = v.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);

    // Apply scaled dot-product attention with causal masking
    auto attn_output = torch::scaled_dot_product_attention(
        q, k, v,
        torch::nullopt,
        0.0,
        true  // is_causal
    );

    // Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, hidden_dim});

    // Output projection
    attn_output = indexed_batched_linear(
        attn_output,
        get_weight(batched_weights, layer_prefix + ".self_attn.out_proj.weight"),
        get_weight(batched_weights, layer_prefix + ".self_attn.out_proj.bias"),
        policy_indices_for_ops,
        dummy_timers
    );
    result.insert("attn_output", attn_output);

    // Residual connection + LayerNorm
    auto residual = x + attn_output;
    auto post_attn = indexed_batched_layer_norm(
        residual,
        get_weight(batched_weights, layer_prefix + ".norm1.weight"),
        get_weight(batched_weights, layer_prefix + ".norm1.bias"),
        policy_indices_for_ops
    );
    result.insert("post_attn", post_attn);

    return result;
}

c10::Dict<std::string, torch::Tensor>
test_moe_layer(
    const torch::Tensor& x,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    int64_t layer_idx,
    int64_t num_experts,
    int64_t top_k,
    int64_t hidden_dim
) {
    auto device = x.device();
    auto policy_indices_for_ops = policy_indices.to(device).to(torch::kLong).contiguous();

    c10::Dict<std::string, torch::Tensor> result;
    std::unordered_map<std::string, std::chrono::microseconds> dummy_timers;

    std::string layer_prefix = "transformer.layers." + std::to_string(layer_idx);

    // Compute gate logits
    auto gate_logits = indexed_batched_linear(
        x,
        get_weight(batched_weights, layer_prefix + ".moe.gate.weight"),
        get_weight(batched_weights, layer_prefix + ".moe.gate.bias"),
        policy_indices_for_ops,
        dummy_timers
    );

    // Ensure dtype consistency
    if (gate_logits.scalar_type() != x.scalar_type()) {
        gate_logits = gate_logits.to(x.scalar_type());
    }
    result.insert("gate_logits", gate_logits);

    // Top-K selection
    auto topk_result = torch::topk(gate_logits, top_k, -1);
    auto topk_indices = std::get<1>(topk_result);
    result.insert("topk_indices", topk_indices);

    // Compute routing weights
    auto gate_probs = torch::softmax(gate_logits, -1);
    auto topk_scores = torch::gather(gate_probs, -1, topk_indices);
    auto topk_weights = topk_scores / topk_scores.sum(-1, true).clamp_min(1e-6);
    result.insert("topk_scores", topk_weights);

    // For now, use CPU fallback for MoE computation (can be optimized later)
    // This matches the PyTorch implementation
    int64_t batch_size = x.size(0);
    int64_t seq_len = x.size(1);
    auto y = torch::zeros_like(x);

    auto expert_w1 = get_weight(batched_weights, layer_prefix + ".moe.experts.w1");
    auto expert_b1 = get_weight(batched_weights, layer_prefix + ".moe.experts.b1");
    auto expert_w2 = get_weight(batched_weights, layer_prefix + ".moe.experts.w2");
    auto expert_b2 = get_weight(batched_weights, layer_prefix + ".moe.experts.b2");

    // Keep policy_indices on the same device as expert weights for indexing
    auto policy_indices_for_indexing = policy_indices_for_ops.to(torch::kLong).contiguous();

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        auto route_mask = (topk_indices == expert_idx).any(-1);

        if (!route_mask.any().item<bool>()) {
            continue;
        }

        auto mask_indices = torch::where(route_mask);
        auto batch_indices = mask_indices[0];
        auto time_indices = mask_indices[1];

        auto expert_inputs = x.index({batch_indices, time_indices});
        auto expert_mask = (topk_indices.index({batch_indices, time_indices}) == expert_idx);
        auto rank_in_topk = torch::where(expert_mask)[1];
        auto expert_routing_weights = topk_weights.index({batch_indices, time_indices, rank_in_topk}).unsqueeze(-1);

        auto policy_for_tokens = policy_indices_for_indexing.index({batch_indices});

        // Index with batch-first layout: [batch_size, num_experts, ...]
        auto w1 = expert_w1.index({policy_for_tokens, expert_idx});
        auto b1 = expert_b1.index({policy_for_tokens, expert_idx});
        auto w2 = expert_w2.index({policy_for_tokens, expert_idx});
        auto b2 = expert_b2.index({policy_for_tokens, expert_idx});

        auto hidden = torch::gelu(
            torch::bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1)
        );
        auto out = torch::bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1);
        auto weighted_out = out.squeeze(1) * expert_routing_weights;

        y.index_put_({batch_indices, time_indices},
                      y.index({batch_indices, time_indices}) + weighted_out);
    }

    result.insert("moe_output", y);

    return result;
}

c10::Dict<std::string, torch::Tensor>
test_heads(
    const torch::Tensor& transformer_output,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    int64_t num_experts
) {
    auto device = transformer_output.device();
    auto policy_indices_for_ops = policy_indices.to(device).to(torch::kLong).contiguous();

    c10::Dict<std::string, torch::Tensor> result;
    std::unordered_map<std::string, std::chrono::microseconds> dummy_timers;

    std::vector<torch::Tensor> action_logits_list, opp_logits_list, state_values_list, win_logits_list;

    for (int64_t i = 0; i < num_experts; ++i) {
        action_logits_list.push_back(
            indexed_batched_linear(
                transformer_output,
                get_weight(batched_weights, "action_heads." + std::to_string(i) + ".weight"),
                get_weight(batched_weights, "action_heads." + std::to_string(i) + ".bias"),
                policy_indices_for_ops,
                dummy_timers)
        );
        opp_logits_list.push_back(
            indexed_batched_linear(
                transformer_output,
                get_weight(batched_weights, "opp_action_heads." + std::to_string(i) + ".weight"),
                get_weight(batched_weights, "opp_action_heads." + std::to_string(i) + ".bias"),
                policy_indices_for_ops,
                dummy_timers)
        );
        state_values_list.push_back(
            indexed_batched_linear(
                transformer_output,
                get_weight(batched_weights, "reward_stream_heads." + std::to_string(i) + ".weight"),
                get_weight(batched_weights, "reward_stream_heads." + std::to_string(i) + ".bias"),
                policy_indices_for_ops,
                dummy_timers)
        );
        win_logits_list.push_back(
            indexed_batched_linear(
                transformer_output,
                get_weight(batched_weights, "win_prob_heads." + std::to_string(i) + ".weight"),
                get_weight(batched_weights, "win_prob_heads." + std::to_string(i) + ".bias"),
                policy_indices_for_ops,
                dummy_timers)
        );
    }

    // Stack along expert dimension
    result.insert("action_heads_stacked", torch::stack(action_logits_list, 2));
    result.insert("opp_heads_stacked", torch::stack(opp_logits_list, 2));
    result.insert("reward_heads_stacked", torch::stack(state_values_list, 2));
    result.insert("win_heads_stacked", torch::stack(win_logits_list, 2));

    return result;
}

// ============================================================================
// Helper Functions
// ============================================================================

torch::Tensor reduce_expert_heads(
    const torch::Tensor& stacked,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_scores
) {
    // stacked: [B, T, num_experts, out_dim]
    // topk_indices: [B, T, K]
    // topk_scores: [B, T, K]
    // output: [B, T, out_dim]

    int64_t B = stacked.size(0);
    int64_t T = stacked.size(1);
    int64_t K = topk_indices.size(2);
    int64_t out_dim = stacked.size(3);
    int64_t expert_dim = stacked.size(2);

    TORCH_CHECK(expert_dim > 0, "reduce_expert_heads: stacked tensor must have a non-zero expert dimension");

    // Ensure indices and scores are on the same device/dtype as stacked and contiguous
    auto indices = topk_indices.to(torch::kLong);
    if (!indices.is_contiguous()) {
        indices = indices.contiguous();
    }
    if (indices.device() != stacked.device()) {
        indices = indices.to(stacked.device());
    }

    auto scores = topk_scores;
    if (scores.device() != stacked.device()) {
        scores = scores.to(stacked.device());
    }
    if (scores.scalar_type() != stacked.scalar_type()) {
        scores = scores.to(stacked.scalar_type());
    }
    if (!scores.is_contiguous()) {
        scores = scores.contiguous();
    }

    auto min_max = torch::aminmax(indices);
    auto min_idx = std::get<0>(min_max).item<int64_t>();
    auto max_idx = std::get<1>(min_max).item<int64_t>();

    if (min_idx < 0 || max_idx >= expert_dim) {
        auto clamped_indices = torch::clamp(indices, 0, expert_dim - 1);
        auto invalid_mask = (indices != clamped_indices);
        auto invalid_count = invalid_mask.sum().item<int64_t>();

        std::ostringstream oss;
        oss << "MoE routing produced out-of-range expert indices (valid range [0, "
            << (expert_dim - 1) << "]). Observed min=" << min_idx
            << ", max=" << max_idx << ".";

        if (invalid_count > 0) {
            oss << " Clamping " << invalid_count << " offending indices and renormalizing weights.";
            auto invalid_coords = torch::nonzero(invalid_mask).to(torch::kCPU);
            auto preview = std::min<int64_t>(5, invalid_coords.size(0));
            if (preview > 0) {
                oss << " Example coordinates (batch, time, topk):";
                for (int64_t i = 0; i < preview; ++i) {
                    auto row = invalid_coords[i];
                    oss << " (" << row[0].item<int64_t>()
                        << "," << row[1].item<int64_t>()
                        << "," << row[2].item<int64_t>() << ")";
                }
            }

            std::cerr << "[WARN] " << oss.str() << std::endl;

            indices = clamped_indices;

            auto invalid_mask_scores = invalid_mask;
            if (invalid_mask_scores.device() != scores.device()) {
                invalid_mask_scores = invalid_mask_scores.to(scores.device());
            }

            scores = scores.clone();
            scores.masked_fill_(invalid_mask_scores, 0);
            auto denom = scores.sum(-1, /*keepdim=*/true).clamp_min(1e-6);
            scores = scores / denom;
        } else {
            std::cerr << "[WARN] " << oss.str() << " No indices were clamped due to empty mask." << std::endl;
            indices = clamped_indices;
        }
    }

    // Expand indices for gather: [B, T, K] -> [B, T, K, out_dim]
    auto gather_idx = indices.unsqueeze(-1).expand({B, T, K, out_dim});

    // Gather top-K expert outputs: [B, T, K, out_dim]
    auto top_outputs = torch::gather(stacked, /*dim=*/2, gather_idx);

    // Weight by routing scores: [B, T, K, 1] * [B, T, K, out_dim]
    auto weighted = top_outputs * scores.unsqueeze(-1);

    // Sum over K dimension: [B, T, out_dim]
    auto output = weighted.sum(/*dim=*/2);

    return output;
}

// ============================================================================
// Main Forward Function
// ============================================================================

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

// Wrapper to preserve original signature (e.g., Python bindings)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed_cpp(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights, // [W, ...]
    const torch::Tensor& policy_indices, // [B]
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad) {
    std::unordered_map<std::string, Microseconds> dummy;
    return forward_packed_cpp(
        obs_sequence,
        action_sequence,
        agent_types,
        positions,
        batched_weights,
        policy_indices,
        padding_mask,
        num_layers,
        num_heads,
        hidden_dim,
        num_experts,
        top_k,
        count_pad,
        tflag_pad,
        dummy
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed_cpp(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights, // [W, ...]
    const torch::Tensor& policy_indices, // [B]
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad,
    std::unordered_map<std::string, Microseconds>& timers
) {
    auto policy_indices_long = policy_indices.to(torch::kLong).contiguous();
    auto policy_indices_cpu = policy_indices_long.device().is_cpu()
        ? policy_indices_long
        : policy_indices_long.cpu();
    torch::Tensor policy_indices_device = policy_indices_long;
    if (obs_sequence.is_cuda() && policy_indices_device.device() != obs_sequence.device()) {
        policy_indices_device = policy_indices_device.to(obs_sequence.device());
    }
    const torch::Tensor& policy_indices_for_ops = obs_sequence.is_cuda() ? policy_indices_device
                                                                         : policy_indices_cpu;

    // ========================================================================
    // Input Validation
    // ========================================================================
    TORCH_CHECK(obs_sequence.dim() == 3, "obs_sequence must be 3D [B, T, obs_dim]");
    TORCH_CHECK(action_sequence.dim() == 2, "action_sequence must be 2D [B, T]");
    TORCH_CHECK(agent_types.dim() == 2, "agent_types must be 2D [B, T]");
    TORCH_CHECK(positions.dim() == 2, "positions must be 2D [B, T]");

    int64_t batch_size = obs_sequence.size(0);
    int64_t seq_len = obs_sequence.size(1);
    int64_t obs_dim = obs_sequence.size(2);
    int64_t action_seq_len = action_sequence.size(1);

    // DEBUG: Check policy indices distribution
    static int call_count = 0;
    if (call_count++ < 3) {  // Only print first 3 calls
        auto unique_policies = std::get<0>(torch::_unique(policy_indices_cpu));
        fprintf(stderr, "[DEBUG forward_packed_cpp call %d] batch_size=%ld, num_unique_policies=%ld\n",
                call_count, batch_size, unique_policies.size(0));
        if (unique_policies.size(0) <= 20 && unique_policies.size(0) > 0) {
            fprintf(stderr, "[DEBUG] Unique policy indices: ");
            auto ptr = unique_policies.data_ptr<int64_t>();
            for (int64_t i = 0; i < std::min<int64_t>(unique_policies.size(0), 20); ++i) {
                fprintf(stderr, "%ld ", ptr[i]);
            }
            fprintf(stderr, "\n");
        }
    }

    // Validate all sequences have matching batch and sequence dimensions
    // CRITICAL: This must happen BEFORE any tensor operations that assume matching shapes
    TORCH_CHECK(action_sequence.size(0) == batch_size,
        "action_sequence batch size mismatch: expected ", batch_size, ", got ", action_sequence.size(0));
    TORCH_CHECK(action_seq_len == seq_len,
        "SEQUENCE LENGTH MISMATCH: obs_sequence has seq_len=", seq_len,
        ", but action_sequence has seq_len=", action_seq_len,
        ". PolicyRequest data is inconsistent - check where PolicyRequests are created!");
    TORCH_CHECK(agent_types.size(0) == batch_size,
        "agent_types batch size mismatch: expected ", batch_size, ", got ", agent_types.size(0));
    TORCH_CHECK(agent_types.size(1) == seq_len,
        "agent_types seq_len mismatch: expected ", seq_len, ", got ", agent_types.size(1));
    TORCH_CHECK(positions.size(0) == batch_size,
        "positions batch size mismatch: expected ", batch_size, ", got ", positions.size(0));
    TORCH_CHECK(positions.size(1) == seq_len,
        "positions seq_len mismatch: expected ", seq_len, ", got ", positions.size(1));

    auto device = action_sequence.device();
    auto dtype = obs_sequence.scalar_type();

    // ========================================================================
    // Load LUT buffers (action decomposition tables)
    // ========================================================================
    // These are fixed buffers that decompose action IDs into (kind, count, flag)
    auto lut_act_kind = get_weight(batched_weights, "lut_act_kind").to(device);
    auto lut_count = get_weight(batched_weights, "lut_count").to(device);
    auto lut_table_flag = get_weight(batched_weights, "lut_table_flag").to(device);

    // ========================================================================
    // Action Decomposition
    // ========================================================================
    auto action_long = action_sequence.to(torch::kLong);

    // LUTs are 1D [11], action_long is 2D [B, T]
    // Flatten, index, reshape pattern for 1D LUT indexing
    auto action_flat = action_long.flatten();  // [B*T]

    auto act_kind_flat = lut_act_kind.index({action_flat});  // [B*T]
    auto count_flat = lut_count.index({action_flat});        // [B*T]
    auto table_flag_flat = lut_table_flag.index({action_flat}); // [B*T]

    // Reshape back to [B, T] and ensure Long dtype for embedding lookups
    auto act_kind_ids = act_kind_flat.view({batch_size, seq_len}).to(torch::kLong);
    auto count_ids = count_flat.view({batch_size, seq_len}).to(torch::kLong);
    auto table_flag_ids = table_flag_flat.view({batch_size, seq_len}).to(torch::kLong);


    // Apply padding mask if provided
    if (padding_mask.has_value()) {
        auto pad_mask = padding_mask.value().to(torch::kBool);
        auto zero_like = torch::zeros_like(act_kind_ids);
        auto count_pad_tensor = torch::full_like(count_ids, count_pad, torch::kLong);
        auto tflag_pad_tensor = torch::full_like(table_flag_ids, tflag_pad, torch::kLong);

        act_kind_ids = torch::where(pad_mask, zero_like, act_kind_ids);
        count_ids = torch::where(pad_mask, count_pad_tensor, count_ids);
        table_flag_ids = torch::where(pad_mask, tflag_pad_tensor, table_flag_ids);
    }

    // ========================================================================
    // Encoding Phase
    // ========================================================================

    // Observation encoding: Linear -> LayerNorm -> GELU
    auto enc_t0 = Clock::now();
    auto obs_encoded = indexed_batched_linear(
        obs_sequence,
        get_weight(batched_weights, "obs_encoder.0.weight"),
        get_weight(batched_weights, "obs_encoder.0.bias"),
        policy_indices_for_ops,
        timers
    );
    obs_encoded = indexed_batched_layer_norm(
        obs_encoded,
        get_weight(batched_weights, "obs_encoder.1.weight"),
        get_weight(batched_weights, "obs_encoder.1.bias"),
        policy_indices_for_ops
    );
    obs_encoded = torch::gelu(obs_encoded);

    // Action embeddings (factorized)
    auto act_embed =
        indexed_batched_embedding(get_weight(batched_weights, "act_kind_embedding.weight"), act_kind_ids, policy_indices_for_ops)
        + indexed_batched_embedding(get_weight(batched_weights, "count_embedding.weight"), count_ids, policy_indices_for_ops)
        + indexed_batched_embedding(get_weight(batched_weights, "table_flag_embedding.weight"), table_flag_ids, policy_indices_for_ops);

    // Agent type embedding
    auto agent_embed = indexed_batched_embedding(
        get_weight(batched_weights, "agent_embedding.weight"),
        agent_types.to(torch::kLong),
        policy_indices_for_ops
    );

    // Position embedding
    auto position_embed = indexed_batched_embedding(
        get_weight(batched_weights, "position_embedding.weight"),
        positions.to(torch::kLong),
        policy_indices_for_ops
    );

    // ========================================================================
    // Gated Fusion (4 independent gates)
    // ========================================================================

    // Gate for observations
    auto hidden_g_obs = indexed_batched_linear(
        obs_encoded,
        get_weight(batched_weights, "gate_obs.0.weight"),
        get_weight(batched_weights, "gate_obs.0.bias"),
        policy_indices_for_ops,
        timers
    );
    hidden_g_obs = torch::tanh(hidden_g_obs);
    auto g_obs = indexed_batched_linear(
        hidden_g_obs,
        get_weight(batched_weights, "gate_obs.2.weight"),
        get_weight(batched_weights, "gate_obs.2.bias"),
        policy_indices_for_ops,
        timers
    );
    g_obs = torch::sigmoid(g_obs);

    // Gate for actions
    auto hidden_g_action = indexed_batched_linear(
        act_embed,
        get_weight(batched_weights, "gate_action.0.weight"),
        get_weight(batched_weights, "gate_action.0.bias"),
        policy_indices_for_ops,
        timers
    );
    hidden_g_action = torch::tanh(hidden_g_action);
    auto g_action = indexed_batched_linear(
        hidden_g_action,
        get_weight(batched_weights, "gate_action.2.weight"),
        get_weight(batched_weights, "gate_action.2.bias"),
        policy_indices_for_ops,
        timers
    );
    g_action = torch::sigmoid(g_action);

    // Gate for agent types
    auto hidden_g_agent = indexed_batched_linear(
        agent_embed,
        get_weight(batched_weights, "gate_agent.0.weight"),
        get_weight(batched_weights, "gate_agent.0.bias"),
        policy_indices_for_ops,
        timers
    );
    hidden_g_agent = torch::tanh(hidden_g_agent);
    auto g_agent = indexed_batched_linear(
        hidden_g_agent,
        get_weight(batched_weights, "gate_agent.2.weight"),
        get_weight(batched_weights, "gate_agent.2.bias"),
        policy_indices_for_ops,
        timers
    );
    g_agent = torch::sigmoid(g_agent);

    // Gate for positions
    auto hidden_g_position = indexed_batched_linear(
        position_embed,
        get_weight(batched_weights, "gate_position.0.weight"),
        get_weight(batched_weights, "gate_position.0.bias"),
        policy_indices_for_ops,
        timers
    );
    hidden_g_position = torch::tanh(hidden_g_position);
    auto g_position = indexed_batched_linear(
        hidden_g_position,
        get_weight(batched_weights, "gate_position.2.weight"),
        get_weight(batched_weights, "gate_position.2.bias"),
        policy_indices_for_ops,
        timers
    );
    g_position = torch::sigmoid(g_position);

    // Ensure embeddings match compute dtype (FP32) for fused combine
    if (act_embed.scalar_type() != obs_encoded.scalar_type()) act_embed = act_embed.to(obs_encoded.scalar_type());
    if (agent_embed.scalar_type() != obs_encoded.scalar_type()) agent_embed = agent_embed.to(obs_encoded.scalar_type());
    if (position_embed.scalar_type() != obs_encoded.scalar_type()) position_embed = position_embed.to(obs_encoded.scalar_type());

    // Fused embedding
    auto fused = g_obs * obs_encoded
               + g_action * act_embed
               + g_agent * agent_embed
               + g_position * position_embed;

    // Final layer norm (uses torch::layer_norm, not batched version)
    auto encoded_inputs = torch::layer_norm(fused, {hidden_dim});
    
    auto enc_t1 = Clock::now();
    timers["fwd_input_encoding_us"] += std::chrono::duration_cast<Microseconds>(enc_t1 - enc_t0);

    // ========================================================================
    // Prepare Masks for Attention
    // ========================================================================

    // Causal mask creation
    // We'll use is_causal=true in scaled_dot_product_attention instead of explicit mask

    torch::optional<torch::Tensor> key_padding_mask = torch::nullopt;
    if (padding_mask.has_value()) {
        key_padding_mask = padding_mask.value().to(torch::kBool).contiguous();
    }

    // ========================================================================
    // Transformer Layers
    // ========================================================================

    auto x = encoded_inputs;
    torch::Tensor final_topk_indices;
    torch::Tensor final_topk_scores;

    int64_t head_dim = hidden_dim / num_heads;

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        std::string layer_prefix = "transformer.layers." + std::to_string(layer_idx);

        // ====================================================================
        // Attention Block
        // ====================================================================

        // Get combined in_proj weight and bias, then split into Q, K, V
        auto in_proj_weight = get_weight(batched_weights, layer_prefix + ".self_attn.in_proj_weight");
        auto in_proj_bias = get_weight(batched_weights, layer_prefix + ".self_attn.in_proj_bias");

        // Determine chunking dimension based on whether weights are batched
        // Batched: [B, 3*H, H] and [B, 3*H], chunk on dim=1
        // Unbatched: [3*H, H] and [3*H], chunk on dim=0
        int64_t weight_chunk_dim = (in_proj_weight.dim() == 3) ? 1 : 0;
        int64_t bias_chunk_dim = (in_proj_bias.dim() == 2) ? 1 : 0;

        // Split into Q, K, V weights
        auto qkv_weights = in_proj_weight.chunk(3, weight_chunk_dim);
        auto qkv_biases = in_proj_bias.chunk(3, bias_chunk_dim);

        auto q_weight = qkv_weights[0];
        auto k_weight = qkv_weights[1];
        auto v_weight = qkv_weights[2];
        auto q_bias = qkv_biases[0];
        auto k_bias = qkv_biases[1];
        auto v_bias = qkv_biases[2];

        // Project to Q, K, V
        auto t_attn_proj_0 = Clock::now();
        auto q = indexed_batched_linear(x, q_weight, q_bias, policy_indices_for_ops, timers);
        auto k = indexed_batched_linear(x, k_weight, k_bias, policy_indices_for_ops, timers);
        auto v = indexed_batched_linear(x, v_weight, v_bias, policy_indices_for_ops, timers);
        
        auto t_attn_proj_1 = Clock::now();
        timers["fwd_attn_proj_us"] += std::chrono::duration_cast<Microseconds>(t_attn_proj_1 - t_attn_proj_0);

        // Reshape for multi-head attention: [B, T, H] -> [B, num_heads, T, head_dim]
        q = q.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        k = k.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        v = v.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);

        // Apply FlashAttention (scaled dot-product attention)
        // Use is_causal=true for causal masking
        auto t_sdpa_0 = Clock::now();
        auto attn_output = torch::scaled_dot_product_attention(
            q, k, v,
            /*attn_mask=*/torch::nullopt,
            /*dropout_p=*/0.0,
            /*is_causal=*/true  // Enables causal masking
        );
        
        auto t_sdpa_1 = Clock::now();
        timers["fwd_attn_sdpa_us"] += std::chrono::duration_cast<Microseconds>(t_sdpa_1 - t_sdpa_0);

        // Reshape back: [B, num_heads, T, head_dim] -> [B, T, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, hidden_dim});

        // Output projection
        auto t_attn_out_0 = Clock::now();
        attn_output = indexed_batched_linear(
            attn_output,
            get_weight(batched_weights, layer_prefix + ".self_attn.out_proj.weight"),
            get_weight(batched_weights, layer_prefix + ".self_attn.out_proj.bias"),
            policy_indices_for_ops,
            timers
        );

        // Residual connection + LayerNorm
        auto residual = x + attn_output;
        x = indexed_batched_layer_norm(
            residual,
            get_weight(batched_weights, layer_prefix + ".norm1.weight"),
            get_weight(batched_weights, layer_prefix + ".norm1.bias"),
            policy_indices_for_ops
        );
        
        auto t_attn_out_1 = Clock::now();
        timers["fwd_attn_output_us"] += std::chrono::duration_cast<Microseconds>(t_attn_out_1 - t_attn_out_0);

        // ====================================================================
        // MoE Block
        // ====================================================================

        // Compute gate logits
        auto t_moe_0 = Clock::now();
        auto gate_logits = indexed_batched_linear(
            x,
            get_weight(batched_weights, layer_prefix + ".moe.gate.weight"),
            get_weight(batched_weights, layer_prefix + ".moe.gate.bias"),
            policy_indices_for_ops,
            timers
        );

        // Ensure dtype consistency
        if (gate_logits.scalar_type() != x.scalar_type()) {
            gate_logits = gate_logits.to(x.scalar_type());
        }

        // Top-K selection
        auto topk_result = torch::topk(gate_logits, top_k, /*dim=*/-1);
        auto topk_indices = std::get<1>(topk_result);

        // Compute routing weights (softmax + normalization)
        auto gate_probs = torch::softmax(gate_logits, /*dim=*/-1);
        auto topk_scores = torch::gather(gate_probs, /*dim=*/-1, topk_indices);
        auto topk_weights = topk_scores / topk_scores.sum(/*dim=*/-1, /*keepdim=*/true).clamp_min(1e-6);

        // Use fused MoE CUDA kernel for expert computation
        torch::Tensor moe_output;

        if (x.is_cuda()) {
            // Use pre-cached pointer tensors for MoE expert weights
            auto w1_ptrs_gpu = get_weight(batched_weights, layer_prefix + ".moe.experts.w1_ptrs");
            auto b1_ptrs_gpu = get_weight(batched_weights, layer_prefix + ".moe.experts.b1_ptrs");
            auto w2_ptrs_gpu = get_weight(batched_weights, layer_prefix + ".moe.experts.w2_ptrs");
            auto b2_ptrs_gpu = get_weight(batched_weights, layer_prefix + ".moe.experts.b2_ptrs");

            // Pointer tensors are tiny; move to CPU for quick indexed lookup
            auto w1_ptrs_cpu = w1_ptrs_gpu.cpu().contiguous();
            auto b1_ptrs_cpu = b1_ptrs_gpu.cpu().contiguous();
            auto w2_ptrs_cpu = w2_ptrs_gpu.cpu().contiguous();
            auto b2_ptrs_cpu = b2_ptrs_gpu.cpu().contiguous();

            const uint64_t* w1_ptr_data = w1_ptrs_cpu.data_ptr<uint64_t>();
            const uint64_t* b1_ptr_data = b1_ptrs_cpu.data_ptr<uint64_t>();
            const uint64_t* w2_ptr_data = w2_ptrs_cpu.data_ptr<uint64_t>();
            const uint64_t* b2_ptr_data = b2_ptrs_cpu.data_ptr<uint64_t>();
            const int64_t num_policies_in_cache = w1_ptrs_cpu.size(0);
            const int64_t num_experts_in_cache = w1_ptrs_cpu.size(1);


            auto orig_dtype = x.scalar_type();
            auto x_fp16 = x.to(torch::kHalf).contiguous();

            const int64_t num_tokens = batch_size * seq_len;
            auto x_flat = x_fp16.view({num_tokens, hidden_dim});

            std::cout << "[EARLY DEBUG] Starting MoE routing. num_tokens=" << num_tokens
                      << ", top_k=" << top_k << ", topk_indices.sizes()=" << topk_indices.sizes() << std::endl;

            auto topk_indices_long = topk_indices.to(torch::kLong).contiguous();
            auto flat_expert_indices = topk_indices_long.reshape({-1});
            auto flat_routing_weights = topk_weights.reshape({-1});

            std::cout << "[EARLY DEBUG] After reshape: flat_expert_indices.size=" << flat_expert_indices.size(0)
                      << ", flat_routing_weights.size=" << flat_routing_weights.size(0) << std::endl;

            auto token_indices = torch::arange(
                num_tokens,
                torch::dtype(torch::kLong).device(x.device())
            );
            auto expanded_token_indices = token_indices.unsqueeze(-1)
                                                  .expand({num_tokens, top_k})
                                                  .contiguous()
                                                  .reshape({-1});

            std::cout << "[EARLY DEBUG] After expand: expanded_token_indices.size=" << expanded_token_indices.size(0) << std::endl;

            auto sort_tuple = torch::sort(flat_expert_indices);
            auto sorted_expert_indices = std::get<0>(sort_tuple);
            auto sort_order = std::get<1>(sort_tuple);

            auto sorted_token_indices = expanded_token_indices.index_select(0, sort_order);
            auto sorted_routing_weights = flat_routing_weights.index_select(0, sort_order);

            auto policy_indices_long = policy_indices_for_ops.to(torch::kLong);
            auto policy_tokens = policy_indices_long.unsqueeze(1)
                                                     .expand({batch_size, seq_len})
                                                     .reshape({-1});
            auto flat_policy_indices = policy_tokens.index_select(0, expanded_token_indices);
            auto sorted_policy_indices = flat_policy_indices.index_select(0, sort_order);

            auto expert_inputs = x_flat.index_select(0, sorted_token_indices).contiguous();
            auto expert_outputs = torch::zeros_like(expert_inputs);

            // DEBUG: Check buffer sizes
            const int64_t expected_routes = num_tokens * top_k;
            std::cout << "[CRITICAL MoE] num_tokens=" << num_tokens << ", top_k=" << top_k
                      << ", expected_routes=" << expected_routes << std::endl;
            std::cout << "[CRITICAL MoE] sorted_token_indices.size=" << sorted_token_indices.size(0)
                      << ", expert_inputs.size=" << expert_inputs.sizes()
                      << ", expert_outputs.size=" << expert_outputs.sizes() << std::endl;
            std::cout << "[CRITICAL MoE] expert_outputs buffer: "
                      << (expert_outputs.numel() * expert_outputs.element_size()) << " bytes allocated, "
                      << (expected_routes * hidden_dim * 2) << " bytes needed" << std::endl;

            // Build grouped dispatch metadata on CPU to drive the grouped GEMM helper
            auto sorted_expert_cpu = sorted_expert_indices.to(torch::kCPU);
            auto sorted_policy_cpu = sorted_policy_indices.to(torch::kCPU);

            const auto* sorted_expert_ptr = sorted_expert_cpu.data_ptr<int64_t>();
            const auto* sorted_policy_ptr = sorted_policy_cpu.data_ptr<int64_t>();

            std::vector<uintptr_t> input_ptrs;
            std::vector<uintptr_t> output_ptrs;
            std::vector<uintptr_t> w1_ptrs;
            std::vector<uintptr_t> b1_ptrs;
            std::vector<uintptr_t> w2_ptrs;
            std::vector<uintptr_t> b2_ptrs;
            std::vector<int64_t> group_m_sizes;

            struct GroupRange {
                int64_t start;
                int64_t count;
                int64_t expert;
                int64_t policy;
            };
            std::vector<GroupRange> groups;

            const int64_t total_routes = sorted_expert_indices.size(0);
            const uintptr_t input_base = reinterpret_cast<uintptr_t>(expert_inputs.data_ptr<at::Half>());
            const uintptr_t output_base = reinterpret_cast<uintptr_t>(expert_outputs.data_ptr<at::Half>());
            const int64_t element_size = static_cast<int64_t>(expert_inputs.element_size());

            int64_t cursor = 0;
            while (cursor < total_routes) {
                const int64_t expert_id = sorted_expert_ptr[cursor];
                const int64_t policy_id = sorted_policy_ptr[cursor];

                int64_t end = cursor + 1;
                while (end < total_routes &&
                       sorted_expert_ptr[end] == expert_id &&
                       sorted_policy_ptr[end] == policy_id) {
                    ++end;
                }

                const int64_t count = end - cursor;
                const uintptr_t input_ptr = input_base + static_cast<uintptr_t>(cursor * hidden_dim * element_size);
                const uintptr_t output_ptr = output_base + static_cast<uintptr_t>(cursor * hidden_dim * element_size);

                // Bounds check: policy_id must be in [0, num_policies_in_cache)
                TORCH_CHECK(policy_id >= 0 && policy_id < num_policies_in_cache,
                    "Policy index out of range: policy_id=", policy_id,
                    ", num_policies_in_cache=", num_policies_in_cache);
                TORCH_CHECK(expert_id >= 0 && expert_id < num_experts,
                    "Expert index out of range: expert_id=", expert_id,
                    ", num_experts=", num_experts);

                input_ptrs.push_back(input_ptr);
                output_ptrs.push_back(output_ptr);

                // Look up pre-cached, stable GPU pointers with bounds checking
                const int64_t ptr_index = policy_id * num_experts_in_cache + expert_id;
                const int64_t max_ptr_index = num_policies_in_cache * num_experts_in_cache;
                TORCH_CHECK(ptr_index >= 0 && ptr_index < max_ptr_index,
                    "Pointer table index out of bounds: ptr_index=", ptr_index,
                    " (policy_id=", policy_id, ", expert_id=", expert_id, ")",
                    ", max_index=", max_ptr_index,
                    " (num_policies=", num_policies_in_cache, ", num_experts=", num_experts_in_cache, ")");

                w1_ptrs.push_back(w1_ptr_data[ptr_index]);
                b1_ptrs.push_back(b1_ptr_data[ptr_index]);
                w2_ptrs.push_back(w2_ptr_data[ptr_index]);
                b2_ptrs.push_back(b2_ptr_data[ptr_index]);

                group_m_sizes.push_back(count);

                cursor = end;
            }

            // Expert FFN dimension from one of the original tensors (shape only)
            const int64_t ffn_dim = get_weight(batched_weights, layer_prefix + ".moe.experts.w1").size(-2);

            if (!group_m_sizes.empty()) {
                grouped_ffn_gemm_forward(
                    input_ptrs.data(),
                    w1_ptrs.data(),
                    b1_ptrs.data(),
                    w2_ptrs.data(),
                    b2_ptrs.data(),
                    output_ptrs.data(),
                    group_m_sizes.data(),
                    static_cast<int64_t>(group_m_sizes.size()),
                    hidden_dim,
                    ffn_dim
                );
            }

            auto sorted_weights_half = sorted_routing_weights.to(expert_outputs.dtype()).unsqueeze(-1);
            auto weighted_outputs = expert_outputs * sorted_weights_half;

            auto moe_output_flat = torch::zeros({num_tokens, hidden_dim}, expert_outputs.options());
            moe_output_flat.index_add_(0, sorted_token_indices, weighted_outputs);
            auto moe_output_half = moe_output_flat.view({batch_size, seq_len, hidden_dim});

            moe_output = moe_output_half.to(orig_dtype);
        } else {
            // CPU fallback: Unrolled MoE computation
            auto y = torch::zeros_like(x);

            auto expert_w1 = get_weight(batched_weights, layer_prefix + ".moe.experts.w1");
            auto expert_b1 = get_weight(batched_weights, layer_prefix + ".moe.experts.b1");
            auto expert_w2 = get_weight(batched_weights, layer_prefix + ".moe.experts.w2");
            auto expert_b2 = get_weight(batched_weights, layer_prefix + ".moe.experts.b2");

            for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                auto route_mask = (topk_indices == expert_idx).any(/*dim=*/-1);

                if (!route_mask.any().item<bool>()) {
                    continue;
                }

                auto mask_indices = torch::where(route_mask);
                auto batch_indices = mask_indices[0];
                auto time_indices = mask_indices[1];

                auto expert_inputs = x.index({batch_indices, time_indices});
                auto expert_mask = (topk_indices.index({batch_indices, time_indices}) == expert_idx);
                auto rank_in_topk = torch::where(expert_mask)[1];
                auto expert_routing_weights = topk_weights.index({batch_indices, time_indices, rank_in_topk}).unsqueeze(-1);

                auto policy_for_tokens = policy_indices_cpu.index({batch_indices});

                auto w1 = expert_w1.index({expert_idx, policy_for_tokens});
                auto b1 = expert_b1.index({expert_idx, policy_for_tokens});
                auto w2 = expert_w2.index({expert_idx, policy_for_tokens});
                auto b2 = expert_b2.index({expert_idx, policy_for_tokens});

                auto hidden = torch::gelu(
                    torch::bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1)
                );
                auto out = torch::bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1);
                auto weighted_out = out.squeeze(1) * expert_routing_weights;

                y.index_put_({batch_indices, time_indices},
                              y.index({batch_indices, time_indices}) + weighted_out);
            }

            moe_output = y;
        }
        
        auto t_moe_1 = Clock::now();
        timers["fwd_moe_block_us"] += std::chrono::duration_cast<Microseconds>(t_moe_1 - t_moe_0);

        // Residual connection + LayerNorm
        auto t_moe_res_0 = Clock::now();
        auto residual2 = x + moe_output;
        x = indexed_batched_layer_norm(
            residual2,
            get_weight(batched_weights, layer_prefix + ".norm2.weight"),
            get_weight(batched_weights, layer_prefix + ".norm2.bias"),
            policy_indices_for_ops
        );
        
        auto t_moe_res_1 = Clock::now();
        timers["fwd_moe_residual_us"] += std::chrono::duration_cast<Microseconds>(t_moe_res_1 - t_moe_res_0);

        // Save final routing info for head reduction
        final_topk_indices = topk_indices;
        final_topk_scores = topk_weights;
    }

    // ========================================================================
    // Final Layer Norm
    // ========================================================================
    auto t_heads_0 = Clock::now();
    auto transformer_output = indexed_batched_layer_norm(
        x,
        get_weight(batched_weights, "transformer.norm.weight"),
        get_weight(batched_weights, "transformer.norm.bias"),
        policy_indices_for_ops
    );

    // ========================================================================
    // Per-Expert Heads (New, Robust Implementation)
    // ========================================================================

    std::vector<torch::Tensor> action_logits_list, opp_logits_list, state_values_list, win_logits_list;

    for (int64_t i = 0; i < num_experts; ++i) {
        // For each expert, compute its head output across the full (B, T, H) input
        // The output of each will be [B, T, out_dim]
        action_logits_list.push_back(
            indexed_batched_linear(
                transformer_output,
                get_weight(batched_weights, "action_heads." + std::to_string(i) + ".weight"),
                get_weight(batched_weights, "action_heads." + std::to_string(i) + ".bias"),
                policy_indices_for_ops,
                timers)
        );
        opp_logits_list.push_back(
            indexed_batched_linear(
                transformer_output,
                get_weight(batched_weights, "opp_action_heads." + std::to_string(i) + ".weight"),
                get_weight(batched_weights, "opp_action_heads." + std::to_string(i) + ".bias"),
                policy_indices_for_ops,
                timers)
        );
        state_values_list.push_back(
            indexed_batched_linear(
                transformer_output,
                get_weight(batched_weights, "reward_stream_heads." + std::to_string(i) + ".weight"),
                get_weight(batched_weights, "reward_stream_heads." + std::to_string(i) + ".bias"),
                policy_indices_for_ops,
                timers)
        );
        win_logits_list.push_back(
            indexed_batched_linear(
                transformer_output,
                get_weight(batched_weights, "win_prob_heads." + std::to_string(i) + ".weight"),
                get_weight(batched_weights, "win_prob_heads." + std::to_string(i) + ".bias"),
                policy_indices_for_ops,
                timers)
        );
    }

    // Stack the results along a new 'expert' dimension
    // Results in shape [B, T, num_experts, out_dim]
    auto action_stacked = torch::stack(action_logits_list, /*dim=*/2);
    auto opp_stacked = torch::stack(opp_logits_list, /*dim=*/2);
    auto reward_stacked = torch::stack(state_values_list, /*dim=*/2);
    auto win_stacked = torch::stack(win_logits_list, /*dim=*/2);

    // Reduce using MoE routing weights (this function is correct)
    auto action_logits = reduce_expert_heads(action_stacked, final_topk_indices, final_topk_scores);
    auto opp_logits = reduce_expert_heads(opp_stacked, final_topk_indices, final_topk_scores);
    auto state_values = reduce_expert_heads(reward_stacked, final_topk_indices, final_topk_scores);
    auto win_logits = reduce_expert_heads(win_stacked, final_topk_indices, final_topk_scores);
    
    auto t_heads_1 = Clock::now();
    timers["fwd_heads_us"] += std::chrono::duration_cast<Microseconds>(t_heads_1 - t_heads_0);

    return std::make_tuple(action_logits, opp_logits, state_values, win_logits);
}
