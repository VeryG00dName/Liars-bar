#include "reactive_model_forward.h"
#include "moe_kernel.h"

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <iostream>
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
// Helper Functions (Simplified, assume matching batch dims)
// ============================================================================

torch::Tensor batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias
) {
    auto x = input.to(weight.scalar_type());
    return torch::matmul(x, weight.transpose(-1, -2)) + bias.unsqueeze(-2);
}

torch::Tensor batched_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps // manual LN supporting per-batch gamma/beta
) {
    auto x = input.to(weight.scalar_type()); // [B, T, H]
    // Compute mean/var over last dimension H
    auto mean = x.mean(-1, /*keepdim=*/true);
    auto var = x.var(-1, /*unbiased=*/false, /*keepdim=*/true);
    auto x_hat = (x - mean) / torch::sqrt(var + eps);
    // weight, bias: [B, H] -> [B, 1, H]
    auto gamma = weight.unsqueeze(1);
    auto beta = bias.unsqueeze(1);
    return x_hat * gamma + beta;
}

torch::Tensor batched_embedding(
    const torch::Tensor& weight,
    const torch::Tensor& indices
) {
    // weight: [B, vocab_size, embed_dim]
    // indices: [B, T] (long)
    int64_t B = indices.size(0);
    int64_t vocab_size = weight.size(1);
    int64_t embed_dim = weight.size(2);
    int64_t time_dim = indices.size(1);
    auto weight_flat = weight.reshape({B * vocab_size, embed_dim});
    auto offset = torch::arange(0, B, indices.options()) * vocab_size;
    auto indices_offset = indices + offset.unsqueeze(1);
    auto indices_flat = indices_offset.reshape({-1});
    auto embedded_flat = torch::embedding(weight_flat, indices_flat);
    return embedded_flat.reshape({B, time_dim, embed_dim});
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
    const c10::Dict<std::string, torch::Tensor>& batched_weights, // [W, ...]
    const torch::Tensor& policy_indices, // [B]
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad
) {
    // --- Definitive fix: unify batch dims by selecting weights per sample ---
    // Produce a new dictionary where each weight has batch size B and matches inputs.
    // Note: 1D tensors (LUT buffers) are shared and not indexed.
    c10::Dict<std::string, torch::Tensor> weights;
    weights.reserve(batched_weights.size());
    for (const auto& pair : batched_weights) {
        const auto& key = pair.key();
        const auto& tensor = pair.value();

        if (tensor.dim() == 1) { // LUTs are not batched
            weights.insert(key, tensor);
            continue;
        }

        // Check if this is a pre-stacked MoE expert weight tensor
        bool is_moe_expert_stack = (key.find(".moe.experts.") != std::string::npos);

        if (is_moe_expert_stack) {
            // For MoE weights [E, B, ...], select along batch dimension 1
            weights.insert(key, tensor.index_select(1, policy_indices));
        } else {
            // For standard weights [B, ...], select along batch dimension 0
            weights.insert(key, tensor.index_select(0, policy_indices));
        }
    }

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

        // Get combined in_proj weight and bias, then split into Q, K, V
        auto in_proj_weight = get_weight(weights, layer_prefix + ".self_attn.in_proj_weight");
        auto in_proj_bias = get_weight(weights, layer_prefix + ".self_attn.in_proj_bias");

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
        auto q = batched_linear(x, q_weight, q_bias);
        auto k = batched_linear(x, k_weight, k_bias);
        auto v = batched_linear(x, v_weight, v_bias);

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
            get_weight(weights, layer_prefix + ".self_attn.out_proj.weight"),
            get_weight(weights, layer_prefix + ".self_attn.out_proj.bias")
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
    // Per-Expert Heads (New, Robust Implementation)
    // ========================================================================

    std::vector<torch::Tensor> action_logits_list, opp_logits_list, state_values_list, win_logits_list;

    for (int64_t i = 0; i < num_experts; ++i) {
        // For each expert, compute its head output across the full (B, T, H) input
        // The output of each will be [B, T, out_dim]
        action_logits_list.push_back(
            batched_linear(transformer_output, 
                           get_weight(weights, "action_heads." + std::to_string(i) + ".weight"),
                           get_weight(weights, "action_heads." + std::to_string(i) + ".bias"))
        );
        opp_logits_list.push_back(
            batched_linear(transformer_output,
                           get_weight(weights, "opp_action_heads." + std::to_string(i) + ".weight"),
                           get_weight(weights, "opp_action_heads." + std::to_string(i) + ".bias"))
        );
        state_values_list.push_back(
            batched_linear(transformer_output,
                           get_weight(weights, "reward_stream_heads." + std::to_string(i) + ".weight"),
                           get_weight(weights, "reward_stream_heads." + std::to_string(i) + ".bias"))
        );
        win_logits_list.push_back(
            batched_linear(transformer_output,
                           get_weight(weights, "win_prob_heads." + std::to_string(i) + ".weight"),
                           get_weight(weights, "win_prob_heads." + std::to_string(i) + ".bias"))
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

    return std::make_tuple(action_logits, opp_logits, state_values, win_logits);
}
