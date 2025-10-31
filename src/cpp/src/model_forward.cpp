/**
 * model_forward.cpp - High-level forward orchestrators
 *
 * Top-level forward pass implementations that delegate to model layers.
 */

#include "model_forward.h"
#include "model_layer.h"
#include "reactive_model_forward.h"
#include "lb_kernels.h"

namespace lb {
namespace forward {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad) {
    using Microseconds = std::chrono::microseconds;
    std::unordered_map<std::string, Microseconds> timers;

    auto device = obs_sequence.device();
    auto pol = policy_indices.to(device).to(torch::kLong).contiguous();

    // 1) Embeddings
    auto embeds = lb::model::compute_embeddings(
        obs_sequence, action_sequence, agent_types, positions,
        batched_weights, pol, padding_mask, count_pad, tflag_pad, &timers);

    // 2) Gating (reuse existing helper for now) and fuse
    auto gating = lb::model::gating(
        embeds.at("obs_embed"), embeds.at("action_embed"),
        embeds.at("agent_embed"), embeds.at("position_embed"),
        batched_weights, pol, &timers);

    auto fused = lb::model::fuse_embeddings(
        gating.at("g_obs"), gating.at("g_action"), gating.at("g_agent"), gating.at("g_position"),
        embeds.at("obs_embed"), embeds.at("action_embed"), embeds.at("agent_embed"), embeds.at("position_embed"),
        hidden_dim);

    auto x = fused.at("combined");

    // 3) Transformer layers
    torch::Tensor last_topk_idx, last_topk_scores;
    for (int64_t li = 0; li < num_layers; ++li) {
        std::string prefix = std::string("transformer.layers.") + std::to_string(li);
        auto in_proj_weight = batched_weights.at(prefix + ".self_attn.in_proj_weight");
        auto in_proj_bias   = batched_weights.at(prefix + ".self_attn.in_proj_bias");
        auto out_proj_weight= batched_weights.at(prefix + ".self_attn.out_proj.weight");
        auto out_proj_bias  = batched_weights.at(prefix + ".self_attn.out_proj.bias");
        auto norm1_weight   = batched_weights.at(prefix + ".norm1.weight");
        auto norm1_bias     = batched_weights.at(prefix + ".norm1.bias");
        auto gate_weight    = batched_weights.at(prefix + ".moe.gate.weight");
        auto gate_bias      = batched_weights.at(prefix + ".moe.gate.bias");
        auto w1_all         = batched_weights.at(prefix + ".moe.experts.w1");
        auto w2_all         = batched_weights.at(prefix + ".moe.experts.w2");
        auto b1_all         = batched_weights.at(prefix + ".moe.experts.b1");
        auto b2_all         = batched_weights.at(prefix + ".moe.experts.b2");
        auto w1_ptrs        = batched_weights.at(prefix + ".moe.experts.w1_ptrs");
        auto w2_ptrs        = batched_weights.at(prefix + ".moe.experts.w2_ptrs");
        auto b1_ptrs        = batched_weights.at(prefix + ".moe.experts.b1_ptrs");
        auto b2_ptrs        = batched_weights.at(prefix + ".moe.experts.b2_ptrs");
        auto norm2_weight   = batched_weights.at(prefix + ".norm2.weight");
        auto norm2_bias     = batched_weights.at(prefix + ".norm2.bias");

        auto tpl = lb::model::transformer_layer(
            x, pol,
            in_proj_weight, in_proj_bias,
            out_proj_weight, out_proj_bias,
            norm1_weight, norm1_bias,
            gate_weight, gate_bias,
            w1_all, w2_all, b1_all, b2_all,
            w1_ptrs, w2_ptrs, b1_ptrs, b2_ptrs,
            norm2_weight, norm2_bias,
            num_heads, hidden_dim, top_k,
            &timers);

        x = std::get<0>(tpl);
        last_topk_idx = std::get<2>(tpl);
        last_topk_scores = std::get<3>(tpl);
    }

    // 4) Heads per expert and reduction
    std::vector<torch::Tensor> action_logits_list, opp_logits_list, value_list, win_list;
    action_logits_list.reserve(num_experts);
    opp_logits_list.reserve(num_experts);
    value_list.reserve(num_experts);
    win_list.reserve(num_experts);

    for (int64_t ei = 0; ei < num_experts; ++ei) {
        auto idx = std::to_string(ei);
        action_logits_list.push_back(
            lb::kernels::indexed_batched_linear(
                x,
                batched_weights.at("action_heads." + idx + ".weight"),
                batched_weights.at("action_heads." + idx + ".bias"),
                pol,
                timers));
        opp_logits_list.push_back(
            lb::kernels::indexed_batched_linear(
                x,
                batched_weights.at("opp_action_heads." + idx + ".weight"),
                batched_weights.at("opp_action_heads." + idx + ".bias"),
                pol,
                timers));
        value_list.push_back(
            lb::kernels::indexed_batched_linear(
                x,
                batched_weights.at("reward_stream_heads." + idx + ".weight"),
                batched_weights.at("reward_stream_heads." + idx + ".bias"),
                pol,
                timers));
        win_list.push_back(
            lb::kernels::indexed_batched_linear(
                x,
                batched_weights.at("win_prob_heads." + idx + ".weight"),
                batched_weights.at("win_prob_heads." + idx + ".bias"),
                pol,
                timers));
    }

    auto action_stacked = torch::stack(action_logits_list, 2);
    auto opp_stacked    = torch::stack(opp_logits_list,    2);
    auto reward_stacked = torch::stack(value_list,         2);
    auto win_stacked    = torch::stack(win_list,           2);

    auto action_logits = lb::model::reduce_expert_heads(action_stacked, last_topk_idx, last_topk_scores);
    auto opp_logits    = lb::model::reduce_expert_heads(opp_stacked,    last_topk_idx, last_topk_scores);
    auto state_values  = lb::model::reduce_expert_heads(reward_stacked, last_topk_idx, last_topk_scores);
    auto win_logits    = lb::model::reduce_expert_heads(win_stacked,    last_topk_idx, last_topk_scores);

    return std::make_tuple(action_logits, opp_logits, state_values, win_logits);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
forward_packed_train(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad) {
    return ::forward_packed_train(obs_sequence, action_sequence, agent_types, positions,
                                  batched_weights, policy_indices, padding_mask,
                                  num_layers, num_heads, hidden_dim, num_experts, top_k,
                                  count_pad, tflag_pad);
}

} // namespace forward
} // namespace lb
