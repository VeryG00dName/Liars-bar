/**
 * model_layer.cpp - Stateless model layer implementations
 *
 * Contains pure computational model layers with no autograd.
 */

#include "model_layer.h"
#include "reactive_model_forward.h"
#include "lb_kernels.h"

namespace lb {
namespace model {

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
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t top_k,
    std::unordered_map<std::string, std::chrono::microseconds>* timers) {

    using Microseconds = std::chrono::microseconds;
    std::unordered_map<std::string, Microseconds> dummy;
    auto& t = timers ? *timers : dummy;

    auto device = x.device();
    auto B = x.size(0);
    auto T = x.size(1);
    auto H = hidden_dim;

    auto pol = policy_indices.to(device).to(torch::kLong).contiguous();

    // Attention projections (Q, K, V)
    int64_t weight_chunk_dim = (in_proj_weight.dim() == 3) ? 1 : 0;
    int64_t bias_chunk_dim   = (in_proj_bias.dim() == 2) ? 1 : 0;
    auto qkv_weights = in_proj_weight.chunk(3, weight_chunk_dim);
    auto qkv_biases  = in_proj_bias.chunk(3, bias_chunk_dim);

    auto q = lb::kernels::indexed_batched_linear(x, qkv_weights[0], qkv_biases[0], pol, t);
    auto k = lb::kernels::indexed_batched_linear(x, qkv_weights[1], qkv_biases[1], pol, t);
    auto v = lb::kernels::indexed_batched_linear(x, qkv_weights[2], qkv_biases[2], pol, t);

    int64_t head_dim = H / num_heads;
    q = q.view({B, T, num_heads, head_dim}).transpose(1, 2);
    k = k.view({B, T, num_heads, head_dim}).transpose(1, 2);
    v = v.view({B, T, num_heads, head_dim}).transpose(1, 2);

    auto attn_output = torch::scaled_dot_product_attention(q, k, v, torch::nullopt, 0.0, true);
    attn_output = attn_output.transpose(1, 2).contiguous().view({B, T, H});

    attn_output = lb::kernels::indexed_batched_linear(attn_output, out_proj_weight, out_proj_bias, pol, t);

    // Residual + Norm1
    auto residual1 = x + attn_output;
    auto x_norm = lb::kernels::indexed_batched_layer_norm(residual1, norm1_weight, norm1_bias, pol, 1e-5);

    // MoE gate logits (non-autograd)
    auto gate_logits = lb::kernels::indexed_batched_linear(x_norm, gate_weight, gate_bias, pol, t);
    auto probs = torch::softmax(gate_logits, -1);
    auto topk_vals_idx = torch::topk(gate_logits, top_k, -1);
    auto topk_indices = std::get<1>(topk_vals_idx);
    auto topk_scores = torch::gather(probs, -1, topk_indices);
    auto topk_weights = topk_scores / topk_scores.sum(-1, true).clamp_min(1e-6);

    // Grouping for CUTLASS grouped GEMM
    const int64_t num_tokens = B * T;
    auto x_norm_fp16 = x_norm.to(torch::kHalf).contiguous();
    auto x_flat = x_norm_fp16.view({num_tokens, H});

    auto topk_indices_long = topk_indices.to(torch::kLong).contiguous();
    auto flat_expert_indices = topk_indices_long.reshape({-1});
    auto flat_routing_weights = topk_weights.reshape({-1});

    auto token_indices = torch::arange(num_tokens, torch::dtype(torch::kLong).device(device));
    auto expanded_token_indices = token_indices.unsqueeze(-1).expand({num_tokens, top_k}).reshape({-1});

    auto policy_tokens = pol.unsqueeze(1).expand({B, T}).reshape({-1});
    auto flat_policy_indices = policy_tokens.index_select(0, expanded_token_indices);

    int64_t num_policies = w1_all.size(0);
    int64_t num_experts = w1_all.size(1);
    int64_t ffn_dim = w1_all.size(2);

    auto combined_key = flat_expert_indices * num_policies + flat_policy_indices;
    auto sort_order = torch::argsort(combined_key);
    auto sorted_expert_indices = flat_expert_indices.index_select(0, sort_order);
    auto sorted_token_indices = expanded_token_indices.index_select(0, sort_order);
    auto sorted_routing_weights = flat_routing_weights.index_select(0, sort_order).to(torch::kFloat32);
    auto sorted_policy_indices = flat_policy_indices.index_select(0, sort_order);

    auto expert_inputs = x_flat.index_select(0, sorted_token_indices).contiguous();
    auto expert_outputs = torch::zeros_like(expert_inputs);

    auto sorted_expert_cpu = sorted_expert_indices.to(torch::kCPU);
    auto sorted_policy_cpu = sorted_policy_indices.to(torch::kCPU);
    const auto* se = sorted_expert_cpu.data_ptr<int64_t>();
    const auto* sp = sorted_policy_cpu.data_ptr<int64_t>();

    // Base pointers for weights/biases
    const at::Half* w1_base = w1_all.data_ptr<at::Half>();
    const at::Half* w2_base = w2_all.data_ptr<at::Half>();
    const at::Half* b1_base = b1_all.data_ptr<at::Half>();
    const at::Half* b2_base = b2_all.data_ptr<at::Half>();
    const int64_t w1_stride_pe = ffn_dim * H;            // elements per (policy,expert) for W1
    const int64_t w2_stride_pe = H * ffn_dim;            // elements per (policy,expert) for W2
    const int64_t b1_stride_pe = ffn_dim;                // elements per (policy,expert) for b1
    const int64_t b2_stride_pe = H;                      // elements per (policy,expert) for b2
    const int64_t pe_stride = num_experts;               // experts per policy

    std::vector<uintptr_t> input_ptrs;
    std::vector<uintptr_t> output_ptrs;
    std::vector<uintptr_t> w1_ptrs;
    std::vector<uintptr_t> b1_ptrs;
    std::vector<uintptr_t> w2_ptrs;
    std::vector<uintptr_t> b2_ptrs;
    std::vector<int64_t> group_m_sizes;
    std::vector<int64_t> group_policy_ids;
    std::vector<int64_t> group_expert_ids;
    std::vector<int64_t> group_token_offsets;

    const int64_t total_routes = sorted_expert_indices.size(0);
    const uintptr_t input_base = reinterpret_cast<uintptr_t>(expert_inputs.data_ptr<at::Half>());
    const uintptr_t output_base = reinterpret_cast<uintptr_t>(expert_outputs.data_ptr<at::Half>());
    const int64_t element_size = static_cast<int64_t>(expert_inputs.element_size());

    int64_t cursor = 0;
    while (cursor < total_routes) {
        const int64_t ei = se[cursor];
        const int64_t pi = sp[cursor];
        int64_t end = cursor + 1;
        while (end < total_routes && se[end] == ei && sp[end] == pi) ++end;

        const int64_t count = end - cursor;
        const uintptr_t in_ptr  = input_base  + static_cast<uintptr_t>(cursor * H * element_size);
        const uintptr_t out_ptr = output_base + static_cast<uintptr_t>(cursor * H * element_size);

        input_ptrs.push_back(in_ptr);
        output_ptrs.push_back(out_ptr);

        const int64_t pe_index = pi * pe_stride + ei;
        w1_ptrs.push_back(reinterpret_cast<uintptr_t>(w1_base + pe_index * w1_stride_pe));
        b1_ptrs.push_back(reinterpret_cast<uintptr_t>(b1_base + pe_index * b1_stride_pe));
        w2_ptrs.push_back(reinterpret_cast<uintptr_t>(w2_base + pe_index * w2_stride_pe));
        b2_ptrs.push_back(reinterpret_cast<uintptr_t>(b2_base + pe_index * b2_stride_pe));

        group_m_sizes.push_back(count);
        group_policy_ids.push_back(pi);
        group_expert_ids.push_back(ei);
        group_token_offsets.push_back(cursor);
        cursor = end;
    }

    // Build routing weight pointer per group
    std::vector<uintptr_t> routing_weight_ptrs;
    routing_weight_ptrs.reserve(group_m_sizes.size());
    auto sorted_rw_f32 = sorted_routing_weights.to(torch::kFloat32).contiguous();
    const float* rw_base = sorted_rw_f32.data_ptr<float>();
    int64_t off = 0;
    for (size_t i = 0; i < group_m_sizes.size(); ++i) {
        routing_weight_ptrs.push_back(reinterpret_cast<uintptr_t>(rw_base + off));
        off += group_m_sizes[i];
    }

    if (!group_m_sizes.empty()) {
        lb::kernels::grouped_ffn_gemm_forward(
            input_ptrs.data(), w1_ptrs.data(), b1_ptrs.data(),
            w2_ptrs.data(), b2_ptrs.data(), output_ptrs.data(),
            routing_weight_ptrs.data(), group_m_sizes.data(),
            group_policy_ids.data(), group_expert_ids.data(),
            group_token_offsets.data(), static_cast<int64_t>(group_m_sizes.size()),
            H, ffn_dim);
    }

    // Scatter-add back to sequence order and finish layer
    auto moe_output_flat = torch::zeros({num_tokens, H}, expert_outputs.options());
    moe_output_flat.index_add_(0, sorted_token_indices, expert_outputs);
    auto moe_output = moe_output_flat.view({B, T, H}).to(x_norm.dtype());

    auto residual2 = x_norm + moe_output;
    auto x_next = lb::kernels::indexed_batched_layer_norm(residual2, norm2_weight, norm2_bias, pol, 1e-5);

    return std::make_tuple(x_next, gate_logits, topk_indices, topk_weights);
}

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
    std::unordered_map<std::string, std::chrono::microseconds>* timers) {

    using Microseconds = std::chrono::microseconds;
    std::unordered_map<std::string, Microseconds> dummy;
    auto& t = timers ? *timers : dummy;

    auto device = obs_sequence.device();
    auto pol = policy_indices.to(device).to(torch::kLong).contiguous();

    // Decompose actions into factor IDs
    auto tup_ids = test_action_decomposition(action_sequence, batched_weights, policy_indices, padding_mask, count_pad, tflag_pad);
    auto act_kind_ids = std::get<0>(tup_ids).to(torch::kLong);
    auto count_ids    = std::get<1>(tup_ids).to(torch::kLong);
    auto table_flag_ids = std::get<2>(tup_ids).to(torch::kLong);

    c10::Dict<std::string, torch::Tensor> result;

    // Obs encoder: Linear -> LayerNorm -> GELU
    auto obs_linear = lb::kernels::indexed_batched_linear(
        obs_sequence,
        batched_weights.at("obs_encoder.0.weight"),
        batched_weights.at("obs_encoder.0.bias"),
        pol,
        t
    );
    auto obs_layernorm = lb::kernels::indexed_batched_layer_norm(
        obs_linear,
        batched_weights.at("obs_encoder.1.weight"),
        batched_weights.at("obs_encoder.1.bias"),
        pol,
        1e-5
    );
    auto obs_embed = torch::gelu(obs_layernorm);
    result.insert("obs_embed", obs_embed);

    // Factorized action embeddings
    auto act_kind_embed = lb::kernels::indexed_batched_embedding(
        batched_weights.at("act_kind_embedding.weight"),
        act_kind_ids,
        pol
    );
    auto count_embed = lb::kernels::indexed_batched_embedding(
        batched_weights.at("count_embedding.weight"),
        count_ids,
        pol
    );
    auto table_flag_embed = lb::kernels::indexed_batched_embedding(
        batched_weights.at("table_flag_embedding.weight"),
        table_flag_ids,
        pol
    );
    auto action_embed = act_kind_embed + count_embed + table_flag_embed;

    result.insert("act_kind_embed", act_kind_embed);
    result.insert("count_embed", count_embed);
    result.insert("table_flag_embed", table_flag_embed);
    result.insert("action_embed", action_embed);

    // Agent + position embeddings
    auto agent_embed = lb::kernels::indexed_batched_embedding(
        batched_weights.at("agent_embedding.weight"),
        agent_types.to(torch::kLong),
        pol
    );
    auto position_embed = lb::kernels::indexed_batched_embedding(
        batched_weights.at("position_embedding.weight"),
        positions.to(torch::kLong),
        pol
    );
    result.insert("agent_embed", agent_embed);
    result.insert("position_embed", position_embed);

    return result;
}

torch::Tensor reduce_expert_heads(
    const torch::Tensor& stacked,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_scores) {
    return ::reduce_expert_heads(stacked, topk_indices, topk_scores);
}

} // namespace model
} // namespace lb
