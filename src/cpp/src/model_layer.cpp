#include "model_layer.h"
#include "lb_kernels.h"
#include "moe_cutlass_kernels.h"
#include <algorithm>
#include <sstream>
#include <torch/torch.h>

namespace lb {
namespace model {

namespace {
// Helper to get weight from dict with error checking
torch::Tensor get_weight(
    const c10::Dict<std::string, torch::Tensor>& weights,
    const std::string& key
) {
    auto it = weights.find(key);
    if (it == weights.end()) {
        std::stringstream ss;
        ss << "Missing weight key: " << key << "\n";
        ss << "Available keys (" << weights.size() << " total):\n";
        for (const auto& kv : weights) {
            ss << "  - " << kv.key() << "\n";
        }
        throw std::runtime_error(ss.str());
    }
    return it->value();
}

// Optimization wrappers for single-policy case
torch::Tensor optimized_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, std::chrono::microseconds>& timers) {
    
    try {
        if (weight.size(0) == 1) {
            // Single-policy optimization: use standard PyTorch functions
            static bool logged = false;
            if (!logged) {
                std::cout << "[INFO] Using single-policy optimized linear (weight batch dim = 1)" << std::endl;
                logged = true;
            }
            auto w = weight.squeeze(0);
            auto b = bias.defined() ? bias.squeeze(0) : torch::Tensor();
            // Standard functional API expects matching dtypes - cast input if needed
            auto input_cast = input.scalar_type() == w.scalar_type() ? input : input.to(w.scalar_type());
            return torch::nn::functional::linear(input_cast, w, b);
        }
        return lb::kernels::indexed_batched_linear(input, weight, bias, policy_indices, timers);
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Error in optimized_linear:\n";
        ss <<" input shape: [" << input.sizes() << "], dtype: " << input.scalar_type() << "\n";
        ss << "  weight shape: [" << weight.sizes() << "], dtype: " << weight.scalar_type() << "\n";
        ss << "  bias defined: " << (bias.defined() ? "yes" : "no");
        if (bias.defined()) ss << ", dtype: " << bias.scalar_type();
        ss << "\n  Original error: " << e.what();
        throw std::runtime_error(ss.str());
    }
}

torch::Tensor optimized_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& policy_indices,
    double eps) {
    
    try {
        if (weight.size(0) == 1) {
            auto w = weight.squeeze(0);
            auto b = bias.squeeze(0);
            // Cast input to match weight dtype if needed
            auto input_cast = input.scalar_type() == w.scalar_type() ? input : input.to(w.scalar_type());
            return torch::nn::functional::layer_norm(input_cast, torch::nn::functional::LayerNormFuncOptions({input_cast.size(-1)}).weight(w).bias(b).eps(eps));
        }
        return lb::kernels::indexed_batched_layer_norm(input, weight, bias, policy_indices, eps);
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Error in optimized_layer_norm:\n";
        ss << "  input shape: [" << input.sizes() << "], dtype: " << input.scalar_type() << "\n";
        ss << "  weight shape: [" << weight.sizes() << "], dtype: " << weight.scalar_type() << "\n";
        ss << "  bias shape: [" << bias.sizes() << "], dtype: " << bias.scalar_type() << "\n";
        ss << "  Original error: " << e.what();
        throw std::runtime_error(ss.str());
    }
}

torch::Tensor optimized_embedding(
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices) {
    
    try {
        if (weight.size(0) == 1) {
            auto w = weight.squeeze(0);
            return torch::nn::functional::embedding(indices, w);
        }
        return lb::kernels::indexed_batched_embedding(weight, indices, policy_indices);
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Error in optimized_embedding:\n";
        ss << "  weight shape: [" << weight.sizes() << "], dtype: " << weight.scalar_type() << "\n";
        ss << "  indices shape: [" << indices.sizes() << "], dtype: " << indices.scalar_type() << "\n";
        ss << "  Original error: " << e.what();
        throw std::runtime_error(ss.str());
    }
}

} // anonymous namespace

// -----------------------------
// Action decomposition (LUTs)
// -----------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
action_decomposition(
    const torch::Tensor& action_sequence,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& /*policy_indices*/, // Not used, but kept for API consistency
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t count_pad,
    int64_t tflag_pad) {

    const int64_t B = action_sequence.size(0);
    const int64_t T = action_sequence.size(1);
    auto device = action_sequence.device();

    auto lut_act_kind   = get_weight(batched_weights, "lut_act_kind").to(device);
    auto lut_count      = get_weight(batched_weights, "lut_count").to(device);
    auto lut_table_flag = get_weight(batched_weights, "lut_table_flag").to(device);

    auto action_long = action_sequence.to(torch::kLong);
    auto flat = action_long.reshape({-1});

    auto act_kind_flat   = lut_act_kind.index({flat});
    auto count_flat      = lut_count.index({flat});
    auto table_flag_flat = lut_table_flag.index({flat});

    auto act_kind_ids    = act_kind_flat.view({B, T}).to(torch::kLong);
    auto count_ids       = count_flat.view({B, T}).to(torch::kLong);
    auto table_flag_ids  = table_flag_flat.view({B, T}).to(torch::kLong);

    if (padding_mask.has_value()) {
        auto pm = padding_mask.value().to(torch::kBool);
        act_kind_ids    = torch::where(pm, torch::zeros_like(act_kind_ids), act_kind_ids);
        count_ids       = torch::where(pm, torch::full_like(count_ids, count_pad, torch::kLong), count_ids);
        table_flag_ids  = torch::where(pm, torch::full_like(table_flag_ids, tflag_pad, torch::kLong), table_flag_ids);
    }

    return std::make_tuple(act_kind_ids, count_ids, table_flag_ids);
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
    auto pol = policy_indices.to(obs_sequence.device()).to(torch::kLong).contiguous();

    // Decompose actions into factor IDs
    auto [act_kind_ids, count_ids, table_flag_ids] = action_decomposition(
        action_sequence, batched_weights, pol, padding_mask, count_pad, tflag_pad);

    c10::Dict<std::string, torch::Tensor> result;

    // Obs encoder: Linear -> LayerNorm -> GELU
    auto obs_linear = optimized_linear(
        obs_sequence,
        get_weight(batched_weights, "obs_encoder.0.weight"),
        get_weight(batched_weights, "obs_encoder.0.bias"),
        pol, t);
    auto obs_layernorm = optimized_layer_norm(
        obs_linear,
        get_weight(batched_weights, "obs_encoder.1.weight"),
        get_weight(batched_weights, "obs_encoder.1.bias"),
        pol, 1e-5);
    auto obs_embed = torch::gelu(obs_layernorm);
    result.insert("obs_embed", obs_embed);

    // Factorized action embeddings
    auto act_kind_embed = optimized_embedding(
        get_weight(batched_weights, "act_kind_embedding.weight"), act_kind_ids, pol);
    auto count_embed = optimized_embedding(
        get_weight(batched_weights, "count_embedding.weight"), count_ids, pol);
    auto table_flag_embed = optimized_embedding(
        get_weight(batched_weights, "table_flag_embedding.weight"), table_flag_ids, pol);
    auto action_embed = act_kind_embed + count_embed + table_flag_embed;
    result.insert("action_embed", action_embed);

    // Agent embedding (always present)
    auto agent_embed = optimized_embedding(
        get_weight(batched_weights, "agent_embedding.weight"), agent_types.to(torch::kLong), pol);
    result.insert("agent_embed", agent_embed);

    // Position embedding (optional - not present in RoPE models)
    if (batched_weights.contains("position_embedding.weight")) {
        auto position_embed = optimized_embedding(
            get_weight(batched_weights, "position_embedding.weight"), positions.to(torch::kLong), pol);
        result.insert("position_embed", position_embed);
    } else {
        // RoPE model: create zero tensor placeholder
        auto position_embed = torch::zeros_like(agent_embed);
        result.insert("position_embed", position_embed);
    }

    return result;
}

c10::Dict<std::string, torch::Tensor>
gating(
    const torch::Tensor& obs_embed,
    const torch::Tensor& action_embed,
    const torch::Tensor& agent_embed,
    const torch::Tensor& position_embed,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, std::chrono::microseconds>* timers) {

    using Microseconds = std::chrono::microseconds;
    std::unordered_map<std::string, Microseconds> dummy;
    auto& t = timers ? *timers : dummy;
    auto pol = policy_indices.to(obs_embed.device()).to(torch::kLong).contiguous();
    c10::Dict<std::string, torch::Tensor> result;

    auto lin = [&](const torch::Tensor& x, const std::string& base){
        auto h = optimized_linear(
            x, get_weight(batched_weights, base + ".0.weight"), get_weight(batched_weights, base + ".0.bias"), pol, t);
        h = torch::tanh(h);
        auto g = optimized_linear(
            h, get_weight(batched_weights, base + ".2.weight"), get_weight(batched_weights, base + ".2.bias"), pol, t);
        return torch::sigmoid(g);
    };

    result.insert("g_obs", lin(obs_embed, "gate_obs"));
    result.insert("g_action", lin(action_embed, "gate_action"));
    result.insert("g_agent", lin(agent_embed, "gate_agent"));
    
    // gate_position is optional (not present in RoPE models)
    if (batched_weights.contains("gate_position.0.weight")) {
        result.insert("g_position", lin(position_embed, "gate_position"));
    } else {
        // Return zeros for RoPE models (will be multiplied by zero position_embed anyway)
        result.insert("g_position", torch::zeros_like(agent_embed));
    }
    return result;
}

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
    int64_t hidden_dim) {
    c10::Dict<std::string, torch::Tensor> result;

    auto cast_like = [&](const torch::Tensor& t){
        return t.scalar_type() == obs_embed.scalar_type() ? t : t.to(obs_embed.scalar_type());
    };

    auto fused = g_obs * obs_embed
               + g_action * cast_like(action_embed)
               + g_agent * cast_like(agent_embed)
               + g_position * cast_like(position_embed);
    result.insert("fused_raw", fused);

    
    // layer_norm without weight/bias expects FP32, so cast if needed
    auto fused_for_norm = fused.scalar_type() == torch::kFloat32 ? fused : fused.to(torch::kFloat32);
    auto combined = torch::layer_norm(fused_for_norm, {hidden_dim});
    // Cast back to original dtype
    if (combined.scalar_type() != fused.scalar_type()) {
        combined = combined.to(fused.scalar_type());
    }
    result.insert("combined", combined);
    return result;
}

namespace {
using torch::indexing::Slice;

inline std::pair<torch::Tensor, torch::Tensor> apply_rope_bthd(
    const torch::Tensor& q_bthd, // [B, T, H, Hd]
    const torch::Tensor& k_bthd, // [B, T, H, Hd]
    const torch::Tensor& positions, // [B, T]
    int64_t head_dim,
    double base = 10000.0
) {
    auto device = q_bthd.device();
    auto dtype = q_bthd.scalar_type();

    // inv_freq: [Hd/2]
    auto half = head_dim / 2;
    auto inv_idx = torch::arange(0, half, torch::dtype(torch::kFloat32).device(device));
    auto inv_freq = 1.0 / torch::pow(torch::full_like(inv_idx, base), inv_idx / static_cast<float>(head_dim));

    // positions: [B, T] -> [B, T, 1]
    auto pos_f = positions.to(device, /*dtype=*/torch::kFloat32).unsqueeze(-1);
    // freqs: [B, T, Hd/2]
    auto freqs = pos_f * inv_freq.view({1, 1, half});
    auto cos = torch::cos(freqs);
    auto sin = torch::sin(freqs);
    // interleave to [B, T, Hd]
    auto cos_i = torch::stack({cos, cos}, -1).flatten(-2).to(dtype);
    auto sin_i = torch::stack({sin, sin}, -1).flatten(-2).to(dtype);

    // expand to [B, T, H, Hd]
    cos_i = cos_i.unsqueeze(2);
    sin_i = sin_i.unsqueeze(2);

    auto rot_half = [](const torch::Tensor& x) {
        auto parts = x.chunk(2, -1);
        return torch::cat({-parts[1], parts[0]}, -1);
    };

    auto q_rot = q_bthd * cos_i + rot_half(q_bthd) * sin_i;
    auto k_rot = k_bthd * cos_i + rot_half(k_bthd) * sin_i;
    return {q_rot, k_rot};
}
} // anonymous namespace

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
    lb::moe::MoEWorkspace* workspace,
    const torch::optional<torch::Tensor>& positions,
    bool use_rope,
    std::unordered_map<std::string, std::chrono::microseconds>* timers) {

    using Microseconds = std::chrono::microseconds;
    using Clock = std::chrono::high_resolution_clock;
    std::unordered_map<std::string, Microseconds> dummy_timers;
    auto& t = timers ? *timers : dummy_timers;

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t head_dim = hidden_dim / num_heads;

    // --- Attention Block ---
    auto t0 = Clock::now();
    auto qkv_weights = in_proj_weight.chunk(3, in_proj_weight.dim() == 3 ? 1 : 0);
    auto qkv_biases  = in_proj_bias.chunk(3, in_proj_bias.dim() == 2 ? 1 : 0);

    auto q = optimized_linear(x, qkv_weights[0], qkv_biases[0], policy_indices, t);
    auto k = optimized_linear(x, qkv_weights[1], qkv_biases[1], policy_indices, t);
    auto v = optimized_linear(x, qkv_weights[2], qkv_biases[2], policy_indices, t);

    q = q.view({B, T, num_heads, head_dim}).transpose(1, 2);
    k = k.view({B, T, num_heads, head_dim}).transpose(1, 2);
    v = v.view({B, T, num_heads, head_dim}).transpose(1, 2);

    if (use_rope && positions.has_value()) {
        auto q_bthd = q.transpose(1, 2);
        auto k_bthd = k.transpose(1, 2);
        auto [q_rot, k_rot] = apply_rope_bthd(q_bthd, k_bthd, positions.value(), head_dim);
        q = q_rot.transpose(1, 2);
        k = k_rot.transpose(1, 2);
    }

    auto attn_output = torch::scaled_dot_product_attention(q, k, v, torch::nullopt, 0.0, true);
    attn_output = attn_output.transpose(1, 2).contiguous().view({B, T, hidden_dim});
    attn_output = optimized_linear(attn_output, out_proj_weight, out_proj_bias, policy_indices, t);

    auto residual1 = x + attn_output;
    auto x_norm = optimized_layer_norm(residual1, norm1_weight, norm1_bias, policy_indices, 1e-5);
    
    auto t1 = Clock::now();
    t["layer_attention_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // --- MoE Block ---
    t0 = Clock::now();
    // Compute gate logits in FP16 (fast linear kernel)
    auto gate_logits_fp16 = lb::kernels::indexed_batched_linear(x_norm, gate_weight, gate_bias, policy_indices, t);
    // Convert to FP32 for routing to match reference precision and avoid topk rounding errors
    auto gate_logits = gate_logits_fp16.to(torch::kFloat);
    // Both topk and softmax now use FP32, ensuring identical expert selection to Python reference
    auto [topk_scores, topk_indices] = torch::topk(gate_logits, top_k, -1);
    auto probs = torch::softmax(gate_logits, -1);
    auto topk_probs = torch::gather(probs, -1, topk_indices);
    auto topk_weights = topk_probs / topk_probs.sum(-1, /*keepdim=*/true).clamp_min(1e-6);

    auto num_tokens = B * T;
    auto x_flat = x_norm.view({num_tokens, hidden_dim});

    auto flat_expert_indices = topk_indices.view({-1});
    auto flat_routing_weights = topk_weights.view({-1});
    auto token_indices = torch::arange(num_tokens, torch::dtype(torch::kLong).device(x.device()));
    auto expanded_token_indices = token_indices.unsqueeze(-1).expand({num_tokens, top_k}).reshape({-1});
    auto policy_tokens = policy_indices.unsqueeze(1).expand({B, T}).reshape({-1});
    auto flat_policy_indices = policy_tokens.index_select(0, expanded_token_indices);

    auto combined_key = flat_expert_indices * num_policies + flat_policy_indices;
    auto sort_order = torch::argsort(combined_key, /*stable=*/true);
    auto sorted_expert_indices = flat_expert_indices.index_select(0, sort_order);
    auto sorted_token_indices = expanded_token_indices.index_select(0, sort_order);
    auto sorted_routing_weights = flat_routing_weights.index_select(0, sort_order).to(torch::kFloat32);
    auto sorted_policy_indices = flat_policy_indices.index_select(0, sort_order);
    
    int64_t ffn_dim = gate_weight.size(-1) * 2; // Infer FFN dim from gate weight

    auto moe_output_flat = torch::zeros_like(x_flat);

    const auto element_size = static_cast<int64_t>(x_flat.element_size());
    bool has_workspace_buffer = workspace && workspace->hidden_buffer && workspace->hidden_buffer_size > 0;
    int64_t workspace_token_capacity = 0;
    if (has_workspace_buffer) {
        const int64_t bytes_per_token = element_size * (ffn_dim + 2 * hidden_dim);
        if (bytes_per_token <= 0) {
            has_workspace_buffer = false;
        } else {
            workspace_token_capacity = static_cast<int64_t>(workspace->hidden_buffer_size / static_cast<size_t>(bytes_per_token));
            if (workspace_token_capacity <= 0) {
                has_workspace_buffer = false;
            }
        }
    }

    const int64_t total_sorted_tokens = sorted_token_indices.size(0);
    if (total_sorted_tokens > 0) {
        auto tensor_options = x_flat.options();

        int64_t token_cursor = 0;
        while (token_cursor < total_sorted_tokens) {
            bool use_workspace_for_chunk = has_workspace_buffer;
            int64_t chunk_token_count;
            if (use_workspace_for_chunk) {
                chunk_token_count = std::min<int64_t>(workspace_token_capacity, total_sorted_tokens - token_cursor);
                if (chunk_token_count <= 0) {
                    use_workspace_for_chunk = false;
                    chunk_token_count = total_sorted_tokens - token_cursor;
                }
            } else {
                chunk_token_count = total_sorted_tokens - token_cursor;
            }

            auto sorted_token_indices_chunk = sorted_token_indices.narrow(0, token_cursor, chunk_token_count);
            auto routing_chunk = sorted_routing_weights.narrow(0, token_cursor, chunk_token_count);
            auto expert_indices_chunk = sorted_expert_indices.narrow(0, token_cursor, chunk_token_count);
            auto policy_indices_chunk = sorted_policy_indices.narrow(0, token_cursor, chunk_token_count);

            auto [m_sizes_dev_chunk, policy_ids_dev_chunk, expert_ids_dev_chunk, token_offsets_dev_chunk] =
                lb::model::moe_group_metadata_device(expert_indices_chunk, policy_indices_chunk);

            int64_t chunk_group_count = m_sizes_dev_chunk.size(0);
            if (chunk_group_count == 0) {
                token_cursor += chunk_token_count;
                continue;
            }

            torch::Tensor chunk_input;
            torch::Tensor chunk_hidden;
            torch::Tensor chunk_output;

            if (use_workspace_for_chunk) {
                size_t bytes_input = static_cast<size_t>(chunk_token_count) * static_cast<size_t>(hidden_dim) * static_cast<size_t>(element_size);
                size_t bytes_hidden = static_cast<size_t>(chunk_token_count) * static_cast<size_t>(ffn_dim) * static_cast<size_t>(element_size);
                size_t bytes_output = bytes_input;
                TORCH_CHECK(bytes_input + bytes_hidden + bytes_output <= workspace->hidden_buffer_size,
                            "MoE workspace hidden buffer too small for chunk");

                auto base_ptr = static_cast<char*>(workspace->hidden_buffer);
                auto noop_deleter = [](void*) {};
                chunk_input = torch::from_blob(base_ptr, {chunk_token_count, hidden_dim}, noop_deleter, tensor_options);
                chunk_hidden = torch::from_blob(base_ptr + bytes_input, {chunk_token_count, ffn_dim}, noop_deleter, tensor_options);
                chunk_output = torch::from_blob(base_ptr + bytes_input + bytes_hidden, {chunk_token_count, hidden_dim}, noop_deleter, tensor_options);
            } else {
                chunk_input = torch::empty({chunk_token_count, hidden_dim}, tensor_options);
                chunk_hidden = torch::empty({chunk_token_count, ffn_dim}, tensor_options);
                chunk_output = torch::empty({chunk_token_count, hidden_dim}, tensor_options);
            }

            torch::index_select_out(chunk_input, x_flat, 0, sorted_token_indices_chunk);

            lb::moe::cutlass_grouped_moe_forward(
                reinterpret_cast<uintptr_t>(chunk_input.data_ptr()),
                reinterpret_cast<uintptr_t>(chunk_hidden.data_ptr()),
                reinterpret_cast<uintptr_t>(chunk_output.data_ptr()),
                reinterpret_cast<uintptr_t>(routing_chunk.data_ptr()),
                w1_ptrs.data_ptr<uint64_t>(),
                w2_ptrs.data_ptr<uint64_t>(),
                b1_ptrs.data_ptr<uint64_t>(),
                b2_ptrs.data_ptr<uint64_t>(),
                num_policies, num_experts,
                m_sizes_dev_chunk.data_ptr<int64_t>(),
                policy_ids_dev_chunk.data_ptr<int64_t>(),
                expert_ids_dev_chunk.data_ptr<int64_t>(),
                token_offsets_dev_chunk.data_ptr<int64_t>(),
                chunk_group_count, hidden_dim, ffn_dim, workspace);

            moe_output_flat.index_add_(0, sorted_token_indices_chunk, chunk_output);

            chunk_input = torch::Tensor();
            chunk_hidden = torch::Tensor();
            chunk_output = torch::Tensor();

            token_cursor += chunk_token_count;
        }
    }

    auto moe_output = moe_output_flat.view({B, T, hidden_dim});

    auto residual2 = x_norm + moe_output;
    auto x_next = lb::kernels::indexed_batched_layer_norm(residual2, norm2_weight, norm2_bias, policy_indices, 1e-5);
    
    t1 = Clock::now();
    t["layer_moe_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    return std::make_tuple(x_next, gate_logits, topk_indices, topk_weights);
}

c10::Dict<std::string, torch::Tensor>
compute_heads(
    const torch::Tensor& transformer_output,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    int64_t num_experts,
    std::unordered_map<std::string, std::chrono::microseconds>* timers) {

    using Microseconds = std::chrono::microseconds;
    std::unordered_map<std::string, Microseconds> dummy;
    auto& t = timers ? *timers : dummy;
    auto pol = policy_indices.to(transformer_output.device()).to(torch::kLong).contiguous();
    c10::Dict<std::string, torch::Tensor> result;

    // Process all experts for each head type
    std::vector<torch::Tensor> action_logits_list, opp_logits_list, state_values_list, win_logits_list;
    action_logits_list.reserve(num_experts);
    opp_logits_list.reserve(num_experts);
    state_values_list.reserve(num_experts);
    win_logits_list.reserve(num_experts);

    for (int64_t i = 0; i < num_experts; ++i) {
        auto idx_str = std::to_string(i);
        // Process all 4 head types for this expert in sequence
        // They can't be batched due to different output dims, but we avoid extra overhead
        action_logits_list.push_back(lb::kernels::indexed_batched_linear(transformer_output,
            get_weight(batched_weights, "action_heads." + idx_str + ".weight"),
            get_weight(batched_weights, "action_heads." + idx_str + ".bias"), pol, t));
        opp_logits_list.push_back(lb::kernels::indexed_batched_linear(transformer_output,
            get_weight(batched_weights, "opp_action_heads." + idx_str + ".weight"),
            get_weight(batched_weights, "opp_action_heads." + idx_str + ".bias"), pol, t));
        state_values_list.push_back(lb::kernels::indexed_batched_linear(transformer_output,
            get_weight(batched_weights, "reward_stream_heads." + idx_str + ".weight"),
            get_weight(batched_weights, "reward_stream_heads." + idx_str + ".bias"), pol, t));
        win_logits_list.push_back(lb::kernels::indexed_batched_linear(transformer_output,
            get_weight(batched_weights, "win_prob_heads." + idx_str + ".weight"),
            get_weight(batched_weights, "win_prob_heads." + idx_str + ".bias"), pol, t));
    }

    result.insert("action_heads_stacked", torch::stack(action_logits_list, 2));
    result.insert("opp_heads_stacked", torch::stack(opp_logits_list, 2));
    result.insert("reward_heads_stacked", torch::stack(state_values_list, 2));
    result.insert("win_heads_stacked", torch::stack(win_logits_list, 2));

    return result;
}

torch::Tensor reduce_expert_heads(
    const torch::Tensor& stacked,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_scores
) {
    int64_t B = stacked.size(0), T = stacked.size(1), K = topk_indices.size(2);
    int64_t out_dim = stacked.size(3), expert_dim = stacked.size(2);
    TORCH_CHECK(expert_dim > 0, "reduce_expert_heads: stacked tensor must have a non-zero expert dimension");

    auto indices = topk_indices.to(torch::kLong).contiguous().to(stacked.device());
    auto scores = topk_scores.contiguous().to(stacked.device()).to(stacked.scalar_type());

    // Clamp out-of-range indices which can happen with buggy routing
    auto clamped_indices = torch::clamp(indices, 0, expert_dim - 1);
    if (!torch::equal(indices, clamped_indices)) {
        indices = clamped_indices;
        // Re-normalize scores if clamping occurred
        auto invalid_mask = (indices != clamped_indices);
        scores = scores.clone();
        scores.masked_fill_(invalid_mask, 0);
        scores = scores / scores.sum(-1, /*keepdim=*/true).clamp_min(1e-6);
    }

    auto gather_idx = indices.unsqueeze(-1).expand({B, T, K, out_dim});
    auto top_outputs = torch::gather(stacked, /*dim=*/2, gather_idx);
    auto weighted = top_outputs * scores.unsqueeze(-1);
    return weighted.sum(/*dim=*/2);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_group_metadata(
    const torch::Tensor& sorted_expert_indices,
    const torch::Tensor& sorted_policy_indices
) {
    TORCH_CHECK(sorted_expert_indices.scalar_type() == torch::kLong,
                "sorted_expert_indices must be int64");
    TORCH_CHECK(sorted_policy_indices.scalar_type() == torch::kLong,
                "sorted_policy_indices must be int64");
    TORCH_CHECK(sorted_expert_indices.dim() == 1 && sorted_policy_indices.dim() == 1,
                "sorted_* tensors must be 1-D");
    TORCH_CHECK(sorted_expert_indices.size(0) == sorted_policy_indices.size(0),
                "sorted_* tensors must have the same length");

    const auto N = sorted_expert_indices.size(0);
    if (N == 0) {
        auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
        return {
            torch::empty({0}, opts),  // m_sizes
            torch::empty({0}, opts),  // policy_ids
            torch::empty({0}, opts),  // expert_ids
            torch::empty({0}, opts)   // token_offsets
        };
    }

    auto device = sorted_expert_indices.device();
    using torch::indexing::Slice;
    using torch::indexing::None;

    auto se = sorted_expert_indices.contiguous();
    auto sp = sorted_policy_indices.contiguous();

    // starts[i] = true if i==0 or (se[i], sp[i]) != (se[i-1], sp[i-1])
    auto starts = torch::empty({N}, torch::dtype(torch::kBool).device(device));
    starts.index_put_({0}, true);
    auto se_prev = se.index({Slice(0, N - 1)});
    auto sp_prev = sp.index({Slice(0, N - 1)});
    auto se_cur = se.index({Slice(1, None)});
    auto sp_cur = sp.index({Slice(1, None)});
    auto same_prev = (se_cur.eq(se_prev)) & (sp_cur.eq(sp_prev));
    auto change = ~same_prev;
    starts.index_put_({Slice(1, None)}, change);

    // start positions of each group
    auto start_pos = torch::nonzero(starts).squeeze(-1).to(torch::kLong);

    // lengths = diff(cat(start_pos, [N]))
    auto N_tensor = torch::tensor({N}, torch::dtype(torch::kLong).device(device));
    auto cat_pos = torch::cat({start_pos, N_tensor}, /*dim=*/0);
    auto lengths = cat_pos.index({Slice(1, None)}) - cat_pos.index({Slice(0, -1)});

    // ids at starts
    auto pol_ids = sp.index_select(0, start_pos);
    auto exp_ids = se.index_select(0, start_pos);

    // Move to CPU for host-side usage
    auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    return {
        lengths.to(opts).contiguous(),     // m_sizes
        pol_ids.to(opts).contiguous(),     // policy_ids
        exp_ids.to(opts).contiguous(),     // expert_ids
        start_pos.to(opts).contiguous()    // token_offsets
    };
}

/**
 * Build pointer table on GPU for batched expert weights.
 *
 * For a stacked tensor [P, E, ...], computes data pointers for each [p, e] slice.
 * Returns GPU tensor of uint64 pointers.
 */
torch::Tensor build_ptr_table_device(const torch::Tensor& stacked) {
    TORCH_CHECK(stacked.is_cuda(), "Stacked expert weights must be CUDA tensors");
    TORCH_CHECK(stacked.dim() >= 2, "Stacked expert weights must be at least 2D");

    const int64_t P = stacked.size(0);
    const int64_t E = stacked.size(1);

    // Compute inner size (product of all dimensions after first 2)
    int64_t inner_size = 1;
    for (int64_t i = 2; i < stacked.dim(); ++i) {
        inner_size *= stacked.size(i);
    }

    // Create indices [P, E] grid
    auto p_idx = torch::arange(P, torch::dtype(torch::kLong).device(stacked.device()))
                    .unsqueeze(1).expand({P, E});
    auto e_idx = torch::arange(E, torch::dtype(torch::kLong).device(stacked.device()))
                    .unsqueeze(0).expand({P, E});

    // Compute linear offset: offset = (p * E * inner_size + e * inner_size)
    auto offsets = (p_idx * E * inner_size + e_idx * inner_size);

    // Get base pointer and element size
    auto base_ptr = reinterpret_cast<uintptr_t>(stacked.data_ptr());
    auto elem_size = stacked.element_size();

    // Compute final pointers: base_ptr + offset * elem_size
    auto ptr_table = (offsets * elem_size + base_ptr).to(torch::kUInt64);

    return ptr_table.contiguous();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_group_metadata_device(
    const torch::Tensor& sorted_expert_indices,
    const torch::Tensor& sorted_policy_indices
) {
    TORCH_CHECK(sorted_expert_indices.scalar_type() == torch::kLong,
                "sorted_expert_indices must be int64");
    TORCH_CHECK(sorted_policy_indices.scalar_type() == torch::kLong,
                "sorted_policy_indices must be int64");
    TORCH_CHECK(sorted_expert_indices.dim() == 1 && sorted_policy_indices.dim() == 1,
                "sorted_* tensors must be 1-D");
    TORCH_CHECK(sorted_expert_indices.size(0) == sorted_policy_indices.size(0),
                "sorted_* tensors must have the same length");

    const auto N = sorted_expert_indices.size(0);
    if (N == 0) {
        auto device = sorted_expert_indices.device();
        auto opts = torch::TensorOptions().dtype(torch::kLong).device(device);
        return {
            torch::empty({0}, opts),  // m_sizes
            torch::empty({0}, opts),  // policy_ids
            torch::empty({0}, opts),  // expert_ids
            torch::empty({0}, opts)   // token_offsets
        };
    }

    auto device = sorted_expert_indices.device();
    using torch::indexing::Slice;
    using torch::indexing::None;

    auto se = sorted_expert_indices.contiguous();
    auto sp = sorted_policy_indices.contiguous();

    // starts[i] = true if i==0 or (se[i], sp[i]) != (se[i-1], sp[i-1])
    auto starts = torch::empty({N}, torch::dtype(torch::kBool).device(device));
    starts.index_put_({0}, true);
    auto se_prev = se.index({Slice(0, N - 1)});
    auto sp_prev = sp.index({Slice(0, N - 1)});
    auto se_cur = se.index({Slice(1, None)});
    auto sp_cur = sp.index({Slice(1, None)});
    auto same_prev = (se_cur.eq(se_prev)) & (sp_cur.eq(sp_prev));
    auto change = ~same_prev;
    starts.index_put_({Slice(1, None)}, change);

    // start positions of each group
    auto start_pos = torch::nonzero(starts).squeeze(-1).to(torch::kLong);

    // lengths = diff(cat(start_pos, [N]))
    auto N_tensor = torch::tensor({N}, torch::dtype(torch::kLong).device(device));
    auto cat_pos = torch::cat({start_pos, N_tensor}, /*dim=*/0);
    auto lengths = cat_pos.index({Slice(1, None)}) - cat_pos.index({Slice(0, -1)});

    // ids at starts
    auto pol_ids = sp.index_select(0, start_pos);
    auto exp_ids = se.index_select(0, start_pos);

    // Return GPU tensors - NO CPU transfer
    return {
        lengths.contiguous(),     // m_sizes
        pol_ids.contiguous(),     // policy_ids
        exp_ids.contiguous(),     // expert_ids
        start_pos.contiguous()    // token_offsets
    };
}

// =============================================================================
// Architecture Detection Helpers
// =============================================================================

bool has_rope(const c10::Dict<std::string, torch::Tensor>& batched_weights) {
    // Prefer explicit RoPE buffers if present, otherwise fall back to absence of learned positions
    if (batched_weights.contains("transformer.layers.0.rope.inv_freq") ||
        batched_weights.contains("transformer_layers.0.rope.inv_freq") ||
        batched_weights.contains("transformer.layers.0.rope.cos_cache") ||
        batched_weights.contains("transformer_layers.0.rope.cos_cache")) {
        return true;
    }
    return !batched_weights.contains("position_embedding.weight");
}

bool has_swiglu(const c10::Dict<std::string, torch::Tensor>& batched_weights) {
    // SwiGLU models have w_gate/w_up/w_down instead of linear1/linear2
    return batched_weights.contains("transformer_layers.0.swiglu_ffn.w_gate.weight") ||
           batched_weights.contains("transformer.layers.0.swiglu_ffn.w_gate.weight") ||
           batched_weights.contains("transformer_layers.0.ffn.w_gate.weight") ||
           batched_weights.contains("transformer.layers.0.ffn.w_gate.weight");
}

// =============================================================================
// Dense (non-MoE) Architecture Support
// =============================================================================

bool is_moe_model(const c10::Dict<std::string, torch::Tensor>& batched_weights) {
    // Check if MoE-specific weights exist
    return batched_weights.contains("transformer.layers.0.moe.gate.weight");
}

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
    const torch::optional<torch::Tensor>& positions,
    bool use_rope,
    std::unordered_map<std::string, std::chrono::microseconds>* timers) {

    using Microseconds = std::chrono::microseconds;
    using Clock = std::chrono::high_resolution_clock;
    std::unordered_map<std::string, Microseconds> dummy_timers;
    auto& t = timers ? *timers : dummy_timers;

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t head_dim = hidden_dim / num_heads;

    // --- Attention Block (same as MoE version) ---
    auto t0 = Clock::now();
    auto qkv_weights = in_proj_weight.chunk(3, in_proj_weight.dim() == 3 ? 1 : 0);
    auto qkv_biases  = in_proj_bias.chunk(3, in_proj_bias.dim() == 2 ? 1 : 0);

    auto q = optimized_linear(x, qkv_weights[0], qkv_biases[0], policy_indices, t);
    auto k = optimized_linear(x, qkv_weights[1], qkv_biases[1], policy_indices, t);
    auto v = optimized_linear(x, qkv_weights[2], qkv_biases[2], policy_indices, t);

    q = q.view({B, T, num_heads, head_dim}).transpose(1, 2);
    k = k.view({B, T, num_heads, head_dim}).transpose(1, 2);
    v = v.view({B, T, num_heads, head_dim}).transpose(1, 2);

    if (use_rope && positions.has_value()) {
        auto q_bthd = q.transpose(1, 2);
        auto k_bthd = k.transpose(1, 2);
        auto [q_rot, k_rot] = apply_rope_bthd(q_bthd, k_bthd, positions.value(), head_dim);
        q = q_rot.transpose(1, 2);
        k = k_rot.transpose(1, 2);
    }

    auto attn_output = torch::scaled_dot_product_attention(q, k, v, torch::nullopt, 0.0, true);
    attn_output = attn_output.transpose(1, 2).contiguous().view({B, T, hidden_dim});
    attn_output = optimized_linear(attn_output, out_proj_weight, out_proj_bias, policy_indices, t);

    auto residual1 = x + attn_output;
    auto x_norm = optimized_layer_norm(residual1, norm1_weight, norm1_bias, policy_indices, 1e-5);

    auto t1 = Clock::now();
    t["layer_attention_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // --- Dense FFN Block (no MoE routing) ---
    t0 = Clock::now();

    // Z = X @ W1^T + b1
    auto z = lb::kernels::indexed_batched_linear(x_norm, linear1_weight, linear1_bias, policy_indices, t);

    // H = GELU(Z)
    auto h = torch::gelu(z);

    // Y = H @ W2^T + b2
    auto ffn_output = lb::kernels::indexed_batched_linear(h, linear2_weight, linear2_bias, policy_indices, t);

    // Residual + LayerNorm
    auto residual2 = x_norm + ffn_output;
    auto x_next = optimized_layer_norm(residual2, norm2_weight, norm2_bias, policy_indices, 1e-5);

    t1 = Clock::now();
    t["layer_ffn_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // Return dummy gate_logits, topk_indices, topk_weights for API compatibility
    auto dummy_gate_logits = torch::empty({B, T, 1}, x.options());
    auto dummy_topk_indices = torch::zeros({B, T, 1}, torch::dtype(torch::kLong).device(x.device()));
    auto dummy_topk_weights = torch::ones({B, T, 1}, x.options().dtype(torch::kFloat32));

    return std::make_tuple(x_next, dummy_gate_logits, dummy_topk_indices, dummy_topk_weights);
}

c10::Dict<std::string, torch::Tensor>
compute_heads_dense(
    const torch::Tensor& transformer_output,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, std::chrono::microseconds>* timers) {

    using Microseconds = std::chrono::microseconds;
    std::unordered_map<std::string, Microseconds> dummy;
    auto& t = timers ? *timers : dummy;
    auto pol = policy_indices.to(transformer_output.device()).to(torch::kLong).contiguous();
    c10::Dict<std::string, torch::Tensor> result;

    // Helper to get optional weight
    auto get_opt = [&](const std::string& key) -> torch::Tensor {
        if (batched_weights.contains(key)) return get_weight(batched_weights, key);
        return torch::Tensor();
    };

    // Action Logits (Policy)
    torch::Tensor action_logits;
    if (batched_weights.contains("action_head.weight")) {
        action_logits = optimized_linear(transformer_output,
            get_weight(batched_weights, "action_head.weight"),
            get_weight(batched_weights, "action_head.bias"), pol, t);
    } else if (batched_weights.contains("output_head.weight")) {
        // Fallback for Deep CFR model
        action_logits = optimized_linear(transformer_output,
            get_weight(batched_weights, "output_head.weight"),
            get_weight(batched_weights, "output_head.bias"), pol, t);
    } else {
        throw std::runtime_error("Missing action_head or output_head weights");
    }

    // Optional Heads (Opponent, Value, Win) - return zeros if missing
    auto make_zeros = [&](int64_t dim) {
        return torch::zeros({transformer_output.size(0), transformer_output.size(1), dim}, 
                            transformer_output.options());
    };

    torch::Tensor opp_logits;
    if (batched_weights.contains("opp_action_head.weight")) {
        opp_logits = optimized_linear(transformer_output,
            get_weight(batched_weights, "opp_action_head.weight"),
            get_weight(batched_weights, "opp_action_head.bias"), pol, t);
    } else {
        opp_logits = make_zeros(7); // Assuming 7 actions
    }

    torch::Tensor state_values;
    if (batched_weights.contains("reward_stream_head.weight")) {
        state_values = optimized_linear(transformer_output,
            get_weight(batched_weights, "reward_stream_head.weight"),
            get_weight(batched_weights, "reward_stream_head.bias"), pol, t);
    } else {
        state_values = make_zeros(1);
    }

    torch::Tensor win_logits;
    if (batched_weights.contains("win_prob_head.weight")) {
        win_logits = optimized_linear(transformer_output,
            get_weight(batched_weights, "win_prob_head.weight"),
            get_weight(batched_weights, "win_prob_head.bias"), pol, t);
    } else {
        win_logits = make_zeros(1);
    }

    result.insert("action_logits", action_logits);
    result.insert("opp_logits", opp_logits);
    result.insert("state_values", state_values);
    result.insert("win_logits", win_logits);

    return result;
}


} // namespace model
} // namespace lb

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
lb::model::transformer_layer_dense_swiglu(
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
    const torch::optional<torch::Tensor>& positions,
    bool use_rope,
    std::unordered_map<std::string, std::chrono::microseconds>* timers) {

    using Microseconds = std::chrono::microseconds;
    using Clock = std::chrono::high_resolution_clock;
    std::unordered_map<std::string, Microseconds> dummy_timers;
    auto& t = timers ? *timers : dummy_timers;

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t head_dim = hidden_dim / num_heads;

    // Attention block (same as dense)
    auto t0 = Clock::now();
    auto qkv_weights = in_proj_weight.chunk(3, in_proj_weight.dim() == 3 ? 1 : 0);
    auto qkv_biases  = in_proj_bias.chunk(3, in_proj_bias.dim() == 2 ? 1 : 0);

    auto q = optimized_linear(x, qkv_weights[0], qkv_biases[0], policy_indices, t);
    auto k = optimized_linear(x, qkv_weights[1], qkv_biases[1], policy_indices, t);
    auto v = optimized_linear(x, qkv_weights[2], qkv_biases[2], policy_indices, t);

    q = q.view({B, T, num_heads, head_dim}).transpose(1, 2);
    k = k.view({B, T, num_heads, head_dim}).transpose(1, 2);
    v = v.view({B, T, num_heads, head_dim}).transpose(1, 2);

    if (use_rope && positions.has_value()) {
        auto q_bthd = q.transpose(1, 2);
        auto k_bthd = k.transpose(1, 2);
        auto [q_rot, k_rot] = apply_rope_bthd(q_bthd, k_bthd, positions.value(), head_dim);
        q = q_rot.transpose(1, 2);
        k = k_rot.transpose(1, 2);
    }

    auto attn_output = torch::scaled_dot_product_attention(q, k, v, torch::nullopt, 0.0, true);
    attn_output = attn_output.transpose(1, 2).contiguous().view({B, T, hidden_dim});
    attn_output = optimized_linear(attn_output, out_proj_weight, out_proj_bias, policy_indices, t);

    auto residual1 = x + attn_output;
    auto x_norm = optimized_layer_norm(residual1, norm1_weight, norm1_bias, policy_indices, 1e-5);

    auto t1 = Clock::now();
    t["layer_attention_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // SwiGLU FFN block
    t0 = Clock::now();
    auto gate = optimized_linear(x_norm, w_gate_weight, w_gate_bias, policy_indices, t);
    gate = torch::silu(gate);
    auto up = optimized_linear(x_norm, w_up_weight, w_up_bias, policy_indices, t);
    auto gated = gate * up;
    auto ffn_output = optimized_linear(gated, w_down_weight, w_down_bias, policy_indices, t);

    auto residual2 = x_norm + ffn_output;
    auto x_next = optimized_layer_norm(residual2, norm2_weight, norm2_bias, policy_indices, 1e-5);

    t1 = Clock::now();
    t["layer_ffn_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // Return dummy routing tensors for API compatibility
    auto dummy_gate_logits = torch::empty({B, T, 1}, x.options());
    auto dummy_topk_indices = torch::zeros({B, T, 1}, torch::dtype(torch::kLong).device(x.device()));
    auto dummy_topk_weights = torch::ones({B, T, 1}, x.options().dtype(torch::kFloat32));

    return std::make_tuple(x_next, dummy_gate_logits, dummy_topk_indices, dummy_topk_weights);
}
