#include "reactive_model_forward.h"
#include "moe_kernel.h"

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <stdexcept>
#include <sstream>

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
// Helper Functions
// ============================================================================

torch::Tensor batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias
) {
    // input: [B, T, in_dim]
    // weight: [B, out_dim, in_dim]
    // bias: [B, out_dim]
    // output: [B, T, out_dim]

    // Ensure dtype compatibility
    auto x = input;
    if (x.scalar_type() != weight.scalar_type()) {
        x = x.to(weight.scalar_type());
    }

    auto b = bias;
    if (b.scalar_type() != weight.scalar_type()) {
        b = b.to(weight.scalar_type());
    }

    // Compute: x @ weight^T + bias
    // x: [B, T, in_dim], weight^T: [B, in_dim, out_dim]
    auto output = torch::matmul(x, weight.transpose(1, 2));  // [B, T, out_dim]
    output = output + b.unsqueeze(1);  // Broadcast bias over time dimension

    return output;
}

torch::Tensor batched_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
    // input: [B, T, dim]
    // weight: [B, dim]
    // bias: [B, dim]

    // Ensure dtype compatibility
    auto x = input;
    if (x.scalar_type() != weight.scalar_type()) {
        x = x.to(weight.scalar_type());
    }

    auto w = weight;
    auto b = bias;
    if (b.scalar_type() != w.scalar_type()) {
        b = b.to(w.scalar_type());
    }

    // Compute mean and variance along last dimension
    auto mean = x.mean(/*dim=*/-1, /*keepdim=*/true);
    auto var = (x - mean).pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto inv_std = torch::rsqrt(var + eps);

    // Normalize
    auto normalized = (x - mean) * inv_std;

    // Apply affine transformation
    // w and b: [B, dim] -> [B, 1, dim] for broadcasting
    auto output = normalized * w.unsqueeze(1) + b.unsqueeze(1);

    return output;
}

torch::Tensor batched_embedding(
    const torch::Tensor& weight,
    const torch::Tensor& indices
) {
    // weight: [B, vocab_size, embed_dim]
    // indices: [B, T]
    // output: [B, T, embed_dim]

    int64_t batch_size = weight.size(0);
    int64_t vocab_size = weight.size(1);
    int64_t embed_dim = weight.size(2);
    int64_t time_dim = indices.size(1);

    // Flatten weights: [B * vocab_size, embed_dim]
    auto weight_flat = weight.reshape({batch_size * vocab_size, embed_dim});

    // Create batch offsets for each example
    // offset[i] = i * vocab_size
    auto offset = torch::arange(0, batch_size, indices.options()) * vocab_size;

    // Offset indices: [B, T] + [B, 1] -> [B, T]
    auto indices_offset = indices + offset.unsqueeze(1);

    // Flatten indices: [B * T]
    auto indices_flat = indices_offset.reshape({-1});

    // Perform flat embedding lookup
    auto embedded_flat = torch::embedding(weight_flat, indices_flat);

    // Reshape back: [B, T, embed_dim]
    auto embedded = embedded_flat.reshape({batch_size, time_dim, embed_dim});

    return embedded;
}

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

    // Expand indices for gather: [B, T, K] -> [B, T, K, out_dim]
    auto gather_idx = topk_indices.unsqueeze(-1).expand({B, T, K, out_dim});

    // Gather top-K expert outputs: [B, T, K, out_dim]
    auto top_outputs = torch::gather(stacked, /*dim=*/2, gather_idx);

    // Weight by routing scores: [B, T, K, 1] * [B, T, K, out_dim]
    auto weighted = top_outputs * topk_scores.unsqueeze(-1);

    // Sum over K dimension: [B, T, out_dim]
    auto output = weighted.sum(/*dim=*/2);

    return output;
}

// ============================================================================
// Main Forward Function
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed_cpp(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& weights,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad
) {
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

    auto device = action_sequence.device();
    auto dtype = obs_sequence.scalar_type();

    // ========================================================================
    // Load LUT buffers (action decomposition tables)
    // ========================================================================
    // These are fixed buffers that decompose action IDs into (kind, count, flag)
    auto lut_act_kind = get_weight(weights, "lut_act_kind").to(device);
    auto lut_count = get_weight(weights, "lut_count").to(device);
    auto lut_table_flag = get_weight(weights, "lut_table_flag").to(device);

    // ========================================================================
    // Action Decomposition
    // ========================================================================
    auto action_long = action_sequence.to(torch::kLong);
    auto act_kind_ids = lut_act_kind.index({action_long});
    auto count_ids = lut_count.index({action_long});
    auto table_flag_ids = lut_table_flag.index({action_long});

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
    auto obs_encoded = batched_linear(
        obs_sequence,
        get_weight(weights, "obs_encoder.0.weight"),
        get_weight(weights, "obs_encoder.0.bias")
    );
    obs_encoded = batched_layer_norm(
        obs_encoded,
        get_weight(weights, "obs_encoder.1.weight"),
        get_weight(weights, "obs_encoder.1.bias")
    );
    obs_encoded = torch::gelu(obs_encoded);

    // Action embeddings (factorized)
    auto act_embed =
        batched_embedding(get_weight(weights, "act_kind_embedding.weight"), act_kind_ids)
        + batched_embedding(get_weight(weights, "count_embedding.weight"), count_ids)
        + batched_embedding(get_weight(weights, "table_flag_embedding.weight"), table_flag_ids);

    // Agent type embedding
    auto agent_embed = batched_embedding(
        get_weight(weights, "agent_embedding.weight"),
        agent_types.to(torch::kLong)
    );

    // Position embedding
    auto position_embed = batched_embedding(
        get_weight(weights, "position_embedding.weight"),
        positions.to(torch::kLong)
    );

    // ========================================================================
    // Gated Fusion (4 independent gates)
    // ========================================================================

    // Gate for observations
    auto hidden_g_obs = batched_linear(
        obs_encoded,
        get_weight(weights, "gate_obs.0.weight"),
        get_weight(weights, "gate_obs.0.bias")
    );
    hidden_g_obs = torch::tanh(hidden_g_obs);
    auto g_obs = batched_linear(
        hidden_g_obs,
        get_weight(weights, "gate_obs.2.weight"),
        get_weight(weights, "gate_obs.2.bias")
    );
    g_obs = torch::sigmoid(g_obs);

    // Gate for actions
    auto hidden_g_action = batched_linear(
        act_embed,
        get_weight(weights, "gate_action.0.weight"),
        get_weight(weights, "gate_action.0.bias")
    );
    hidden_g_action = torch::tanh(hidden_g_action);
    auto g_action = batched_linear(
        hidden_g_action,
        get_weight(weights, "gate_action.2.weight"),
        get_weight(weights, "gate_action.2.bias")
    );
    g_action = torch::sigmoid(g_action);

    // Gate for agent types
    auto hidden_g_agent = batched_linear(
        agent_embed,
        get_weight(weights, "gate_agent.0.weight"),
        get_weight(weights, "gate_agent.0.bias")
    );
    hidden_g_agent = torch::tanh(hidden_g_agent);
    auto g_agent = batched_linear(
        hidden_g_agent,
        get_weight(weights, "gate_agent.2.weight"),
        get_weight(weights, "gate_agent.2.bias")
    );
    g_agent = torch::sigmoid(g_agent);

    // Gate for positions
    auto hidden_g_position = batched_linear(
        position_embed,
        get_weight(weights, "gate_position.0.weight"),
        get_weight(weights, "gate_position.0.bias")
    );
    hidden_g_position = torch::tanh(hidden_g_position);
    auto g_position = batched_linear(
        hidden_g_position,
        get_weight(weights, "gate_position.2.weight"),
        get_weight(weights, "gate_position.2.bias")
    );
    g_position = torch::sigmoid(g_position);

    // Fused embedding
    auto fused = g_obs * obs_encoded
               + g_action * act_embed
               + g_agent * agent_embed
               + g_position * position_embed;

    // Final layer norm (uses torch::layer_norm, not batched version)
    auto encoded_inputs = torch::layer_norm(fused, {hidden_dim});

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

        // Project to Q, K, V
        auto q = batched_linear(
            x,
            get_weight(weights, layer_prefix + ".q_proj.weight"),
            get_weight(weights, layer_prefix + ".q_proj.bias")
        );
        auto k = batched_linear(
            x,
            get_weight(weights, layer_prefix + ".k_proj.weight"),
            get_weight(weights, layer_prefix + ".k_proj.bias")
        );
        auto v = batched_linear(
            x,
            get_weight(weights, layer_prefix + ".v_proj.weight"),
            get_weight(weights, layer_prefix + ".v_proj.bias")
        );

        // Reshape for multi-head attention: [B, T, H] -> [B, num_heads, T, head_dim]
        q = q.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        k = k.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        v = v.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);

        // Apply FlashAttention (scaled dot-product attention)
        // Use is_causal=true for causal masking
        auto attn_output = torch::scaled_dot_product_attention(
            q, k, v,
            /*attn_mask=*/torch::nullopt,
            /*dropout_p=*/0.0,
            /*is_causal=*/true  // Enables causal masking
        );

        // Reshape back: [B, num_heads, T, head_dim] -> [B, T, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, hidden_dim});

        // Output projection
        attn_output = batched_linear(
            attn_output,
            get_weight(weights, layer_prefix + ".out_proj.weight"),
            get_weight(weights, layer_prefix + ".out_proj.bias")
        );

        // Residual connection + LayerNorm
        auto residual = x + attn_output;
        x = batched_layer_norm(
            residual,
            get_weight(weights, layer_prefix + ".norm1.weight"),
            get_weight(weights, layer_prefix + ".norm1.bias")
        );

        // ====================================================================
        // MoE Block
        // ====================================================================

        // Compute gate logits
        auto gate_logits = batched_linear(
            x,
            get_weight(weights, layer_prefix + ".moe.gate.weight"),
            get_weight(weights, layer_prefix + ".moe.gate.bias")
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
            // Get pre-stacked expert weights
            auto expert_w1 = get_weight(weights, layer_prefix + ".moe.experts.w1");
            auto expert_b1 = get_weight(weights, layer_prefix + ".moe.experts.b1");
            auto expert_w2 = get_weight(weights, layer_prefix + ".moe.experts.w2");
            auto expert_b2 = get_weight(weights, layer_prefix + ".moe.experts.b2");

            // Ensure FP16 for CUDA kernel
            auto x_fp16 = x.to(torch::kHalf).contiguous();
            auto gate_logits_fp16 = gate_logits.to(torch::kHalf).contiguous();
            auto topk_indices_long = topk_indices.to(torch::kLong).contiguous();
            auto expert_w1_fp16 = expert_w1.to(torch::kHalf).contiguous();
            auto expert_b1_fp16 = expert_b1.to(torch::kHalf).contiguous();
            auto expert_w2_fp16 = expert_w2.to(torch::kHalf).contiguous();
            auto expert_b2_fp16 = expert_b2.to(torch::kHalf).contiguous();

            // Call custom CUDA kernel
            moe_output = moe_forward_cuda(
                x_fp16,
                gate_logits_fp16,
                topk_indices_long,
                expert_w1_fp16,
                expert_b1_fp16,
                expert_w2_fp16,
                expert_b2_fp16
            );

            // Convert back to original dtype if needed
            if (moe_output.scalar_type() != x.scalar_type()) {
                moe_output = moe_output.to(x.scalar_type());
            }
        } else {
            // CPU fallback: Unrolled MoE computation
            // This is slower but maintains numerical correctness for testing
            auto y = torch::zeros_like(x);

            for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                // Find tokens routed to this expert
                auto route_mask = (topk_indices == expert_idx).any(/*dim=*/-1);

                if (route_mask.any().item<bool>()) {
                    // Get batch and time indices where this expert is used
                    auto mask_indices = torch::where(route_mask);
                    auto batch_indices = mask_indices[0];
                    auto time_indices = mask_indices[1];

                    // Extract inputs for this expert
                    auto expert_inputs = x.index({batch_indices, time_indices});

                    // Find which rank in top-K this expert has for each token
                    auto expert_mask = (topk_indices.index({batch_indices, time_indices}) == expert_idx);
                    auto rank_in_topk = torch::where(expert_mask)[1];
                    auto expert_routing_weights = topk_weights.index({batch_indices, time_indices, rank_in_topk}).unsqueeze(-1);

                    // Get expert weights
                    std::string expert_prefix = layer_prefix + ".moe.experts." + std::to_string(expert_idx);
                    auto w1 = get_weight(weights, expert_prefix + ".0.weight").index({batch_indices});
                    auto b1 = get_weight(weights, expert_prefix + ".0.bias").index({batch_indices});
                    auto w2 = get_weight(weights, expert_prefix + ".3.weight").index({batch_indices});
                    auto b2 = get_weight(weights, expert_prefix + ".3.bias").index({batch_indices});

                    // Compute expert forward: x -> FFN -> output
                    auto hidden = torch::gelu(
                        torch::bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1)
                    );
                    auto out = torch::bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1);
                    auto weighted_out = out.squeeze(1) * expert_routing_weights;

                    // Accumulate to output
                    y.index_put_({batch_indices, time_indices},
                                  y.index({batch_indices, time_indices}) + weighted_out);
                }
            }

            moe_output = y;
        }

        // Residual connection + LayerNorm
        auto residual2 = x + moe_output;
        x = batched_layer_norm(
            residual2,
            get_weight(weights, layer_prefix + ".norm2.weight"),
            get_weight(weights, layer_prefix + ".norm2.bias")
        );

        // Save final routing info for head reduction
        final_topk_indices = topk_indices;
        final_topk_scores = topk_weights;
    }

    // ========================================================================
    // Final Layer Norm
    // ========================================================================
    auto transformer_output = batched_layer_norm(
        x,
        get_weight(weights, "transformer.norm.weight"),
        get_weight(weights, "transformer.norm.bias")
    );

    // ========================================================================
    // Per-Expert Heads
    // ========================================================================

    // Stack weights for each head type across experts
    std::vector<torch::Tensor> action_w_list, action_b_list;
    std::vector<torch::Tensor> opp_w_list, opp_b_list;
    std::vector<torch::Tensor> reward_w_list, reward_b_list;
    std::vector<torch::Tensor> win_w_list, win_b_list;

    for (int64_t i = 0; i < num_experts; ++i) {
        action_w_list.push_back(get_weight(weights, "action_heads." + std::to_string(i) + ".weight"));
        action_b_list.push_back(get_weight(weights, "action_heads." + std::to_string(i) + ".bias"));
        opp_w_list.push_back(get_weight(weights, "opp_action_heads." + std::to_string(i) + ".weight"));
        opp_b_list.push_back(get_weight(weights, "opp_action_heads." + std::to_string(i) + ".bias"));
        reward_w_list.push_back(get_weight(weights, "reward_stream_heads." + std::to_string(i) + ".weight"));
        reward_b_list.push_back(get_weight(weights, "reward_stream_heads." + std::to_string(i) + ".bias"));
        win_w_list.push_back(get_weight(weights, "win_prob_heads." + std::to_string(i) + ".weight"));
        win_b_list.push_back(get_weight(weights, "win_prob_heads." + std::to_string(i) + ".bias"));
    }

    // Stack along expert dimension: [B, num_experts, ...]
    auto action_w = torch::stack(action_w_list, /*dim=*/1);
    auto action_b = torch::stack(action_b_list, /*dim=*/1);
    auto opp_w = torch::stack(opp_w_list, /*dim=*/1);
    auto opp_b = torch::stack(opp_b_list, /*dim=*/1);
    auto reward_w = torch::stack(reward_w_list, /*dim=*/1);
    auto reward_b = torch::stack(reward_b_list, /*dim=*/1);
    auto win_w = torch::stack(win_w_list, /*dim=*/1);
    auto win_b = torch::stack(win_b_list, /*dim=*/1);

    // Expand transformer output for per-expert computation
    // [B, T, H] -> [B, num_experts, T, H]
    auto expanded_input = transformer_output.unsqueeze(1).expand({batch_size, num_experts, seq_len, hidden_dim});

    // Ensure dtype compatibility
    if (expanded_input.scalar_type() != action_w.scalar_type()) {
        expanded_input = expanded_input.to(action_w.scalar_type());
    }

    // Compute per-expert outputs: [B, num_experts, T, out_dim]
    auto action_stacked = torch::matmul(expanded_input, action_w.transpose(-1, -2)) + action_b.unsqueeze(2);
    auto opp_stacked = torch::matmul(expanded_input, opp_w.transpose(-1, -2)) + opp_b.unsqueeze(2);
    auto reward_stacked = torch::matmul(expanded_input, reward_w.transpose(-1, -2)) + reward_b.unsqueeze(2);
    auto win_stacked = torch::matmul(expanded_input, win_w.transpose(-1, -2)) + win_b.unsqueeze(2);

    // Permute to [B, T, num_experts, out_dim] for reduce_expert_heads
    action_stacked = action_stacked.permute({0, 2, 1, 3});
    opp_stacked = opp_stacked.permute({0, 2, 1, 3});
    reward_stacked = reward_stacked.permute({0, 2, 1, 3});
    win_stacked = win_stacked.permute({0, 2, 1, 3});

    // Reduce using MoE routing weights
    auto action_logits = reduce_expert_heads(action_stacked, final_topk_indices, final_topk_scores);
    auto opp_logits = reduce_expert_heads(opp_stacked, final_topk_indices, final_topk_scores);
    auto state_values = reduce_expert_heads(reward_stacked, final_topk_indices, final_topk_scores);
    auto win_logits = reduce_expert_heads(win_stacked, final_topk_indices, final_topk_scores);

    return std::make_tuple(action_logits, opp_logits, state_values, win_logits);
}
