#include "reactive_model_forward.h"
#include "indexed_kernels.h"
#include "indexed_autograd.h"
#include "moe_autograd_function.h"

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

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
transformer_layer_forward_impl(
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
    int64_t top_k) {
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t head_dim = hidden_dim / num_heads;

    // --- Attention Block ---
    int64_t weight_chunk_dim = (in_proj_weight.dim() == 3) ? 1 : 0;
    int64_t bias_chunk_dim   = (in_proj_bias.dim() == 2) ? 1 : 0;
    auto qkv_weights = in_proj_weight.chunk(3, weight_chunk_dim);
    auto qkv_biases  = in_proj_bias.chunk(3, bias_chunk_dim);

    auto q = indexed_batched_linear_autograd(x, qkv_weights[0], qkv_biases[0], policy_indices);
    auto k = indexed_batched_linear_autograd(x, qkv_weights[1], qkv_biases[1], policy_indices);
    auto v = indexed_batched_linear_autograd(x, qkv_weights[2], qkv_biases[2], policy_indices);

    q = q.view({B, T, num_heads, head_dim}).transpose(1, 2);
    k = k.view({B, T, num_heads, head_dim}).transpose(1, 2);
    v = v.view({B, T, num_heads, head_dim}).transpose(1, 2);

    auto attn_output = torch::scaled_dot_product_attention(q, k, v, torch::nullopt, 0.0, true);
    attn_output = attn_output.transpose(1, 2).contiguous().view({B, T, hidden_dim});
    attn_output = indexed_batched_linear_autograd(attn_output, out_proj_weight, out_proj_bias, policy_indices);

    auto residual1 = x + attn_output;
    auto x_norm = indexed_batched_layer_norm_autograd(residual1, norm1_weight, norm1_bias, policy_indices, 1e-5);

    // --- MoE Block ---
    auto gate_logits = indexed_batched_linear_autograd(x_norm, gate_weight, gate_bias, policy_indices);
    auto probs = torch::softmax(gate_logits, -1);
    auto topk_vals_idx = torch::topk(gate_logits, top_k, -1);
    auto topk_indices = std::get<1>(topk_vals_idx);
    auto topk_scores  = torch::gather(probs, -1, topk_indices);
    // Normalize selected top-k scores to sum to 1 for numerical stability
    auto topk_weights = topk_scores / topk_scores.sum(-1, /*keepdim=*/true).clamp_min(1e-6);

    auto num_tokens = B * T;
    
    // The input to the MoE kernels must be float16.
    auto x_norm_fp16 = x_norm.to(torch::kHalf).contiguous();
    auto x_flat = x_norm_fp16.view({num_tokens, hidden_dim});

    auto topk_indices_long = topk_indices.to(torch::kLong).contiguous();
    auto flat_expert_indices = topk_indices_long.reshape({-1});
    auto flat_routing_weights = topk_weights.reshape({-1});
    auto token_indices = torch::arange(num_tokens, torch::dtype(torch::kLong).device(x.device()));
    auto expanded_token_indices = token_indices.unsqueeze(-1).expand({num_tokens, top_k}).reshape({-1});
    auto policy_indices_long2 = policy_indices.to(torch::kLong);
    auto policy_tokens = policy_indices_long2.unsqueeze(1).expand({B, T}).reshape({-1});
    auto flat_policy_indices = policy_tokens.index_select(0, expanded_token_indices);
    // Stable grouping by (expert, policy): sort with combined key
    int64_t num_policies_for_key = w1_all.size(0);
    auto combined_key = flat_expert_indices * num_policies_for_key + flat_policy_indices;
    auto sort_order = torch::argsort(combined_key);
    auto sorted_expert_indices = flat_expert_indices.index_select(0, sort_order);
    auto sorted_token_indices = expanded_token_indices.index_select(0, sort_order);
    auto sorted_routing_weights = flat_routing_weights.index_select(0, sort_order).to(torch::kFloat32);
    auto sorted_policy_indices = flat_policy_indices.index_select(0, sort_order);
    
    auto expert_inputs = x_flat.index_select(0, sorted_token_indices).contiguous();
    // Clamp to finite FP16 range to avoid INF in kernels
    expert_inputs = expert_inputs.clamp(-65504.0, 65504.0);

    auto sorted_expert_cpu = sorted_expert_indices.to(torch::kCPU);
    auto sorted_policy_cpu = sorted_policy_indices.to(torch::kCPU);
    const auto* se = sorted_expert_cpu.data_ptr<int64_t>();
    const auto* sp = sorted_policy_cpu.data_ptr<int64_t>();
    std::vector<int64_t> m_sizes_v, policy_ids_v, expert_ids_v, token_offsets_v;
    int64_t cursor = 0;
    const int64_t total_expert_routes = sorted_expert_indices.size(0);
    while (cursor < total_expert_routes) {
        int64_t ei = se[cursor];
        int64_t pi = sp[cursor];
        int64_t end = cursor + 1;
        while (end < total_expert_routes && se[end] == ei && sp[end] == pi) {
            ++end;
        }
        m_sizes_v.push_back(end - cursor);
        policy_ids_v.push_back(pi);
        expert_ids_v.push_back(ei);
        token_offsets_v.push_back(cursor);
        cursor = end;
    }

    auto m_sizes_cpu = torch::from_blob(m_sizes_v.data(), {(int64_t)m_sizes_v.size()}, torch::TensorOptions().dtype(torch::kLong)).clone();
    auto policy_ids_cpu = torch::from_blob(policy_ids_v.data(), {(int64_t)policy_ids_v.size()}, torch::TensorOptions().dtype(torch::kLong)).clone();
    auto expert_ids_cpu = torch::from_blob(expert_ids_v.data(), {(int64_t)expert_ids_v.size()}, torch::TensorOptions().dtype(torch::kLong)).clone();
    auto token_offsets_cpu = torch::from_blob(token_offsets_v.data(), {(int64_t)token_offsets_v.size()}, torch::TensorOptions().dtype(torch::kLong)).clone();

    // The weights passed to the MoE autograd function must also be float16.
    auto expert_outputs = lb::moe::grouped_moe_autograd_forward(
        expert_inputs,
        w1_all.to(torch::kFloat16),
        w2_all.to(torch::kFloat16),
        b1_all.to(torch::kFloat16),
        b2_all.to(torch::kFloat16),
        sorted_routing_weights,
        sorted_token_indices,
        m_sizes_cpu,
        policy_ids_cpu,
        expert_ids_cpu,
        token_offsets_cpu,
        hidden_dim,
        w1_all.size(-2)
    );

    auto moe_output_flat = torch::zeros({num_tokens, hidden_dim}, expert_outputs.options());
    moe_output_flat.index_add_(0, sorted_token_indices, expert_outputs);
    auto moe_output = moe_output_flat.view({B, T, hidden_dim}).to(x_norm.dtype());

    auto residual2 = x_norm + moe_output;
    auto x_next = indexed_batched_layer_norm_autograd(residual2, norm2_weight, norm2_bias, policy_indices, 1e-5);

    return std::make_tuple(x_next, gate_logits, topk_indices, topk_weights);
}

struct TransformerLayerCheckpointFunction
    : public torch::autograd::Function<TransformerLayerCheckpointFunction> {
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
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
        int64_t top_k) {
        torch::NoGradGuard no_grad;

        auto outputs = transformer_layer_forward_impl(
            x,
            policy_indices,
            in_proj_weight,
            in_proj_bias,
            out_proj_weight,
            out_proj_bias,
            norm1_weight,
            norm1_bias,
            gate_weight,
            gate_bias,
            w1_all,
            w2_all,
            b1_all,
            b2_all,
            norm2_weight,
            norm2_bias,
            num_heads,
            hidden_dim,
            top_k);

        std::vector<int64_t> requires_grad_flags;
        requires_grad_flags.reserve(16);
        auto record_flag = [&](const torch::Tensor& t) {
            requires_grad_flags.push_back(t.requires_grad() ? 1 : 0);
        };

        record_flag(x);
        record_flag(policy_indices);
        record_flag(in_proj_weight);
        record_flag(in_proj_bias);
        record_flag(out_proj_weight);
        record_flag(out_proj_bias);
        record_flag(norm1_weight);
        record_flag(norm1_bias);
        record_flag(gate_weight);
        record_flag(gate_bias);
        record_flag(w1_all);
        record_flag(w2_all);
        record_flag(b1_all);
        record_flag(b2_all);
        record_flag(norm2_weight);
        record_flag(norm2_bias);

        ctx->saved_data["requires_grad_flags"] = requires_grad_flags;
        ctx->saved_data["num_heads"] = num_heads;
        ctx->saved_data["hidden_dim"] = hidden_dim;
        ctx->saved_data["top_k"] = top_k;

        ctx->save_for_backward({
            x.detach(),
            policy_indices.detach(),
            in_proj_weight.detach(),
            in_proj_bias.detach(),
            out_proj_weight.detach(),
            out_proj_bias.detach(),
            norm1_weight.detach(),
            norm1_bias.detach(),
            gate_weight.detach(),
            gate_bias.detach(),
            w1_all.detach(),
            w2_all.detach(),
            b1_all.detach(),
            b2_all.detach(),
            norm2_weight.detach(),
            norm2_bias.detach()
        });

        torch::autograd::tensor_list result(4);
        result[0] = std::get<0>(outputs);
        result[1] = std::get<1>(outputs);
        result[2] = std::get<2>(outputs);
        result[3] = std::get<3>(outputs);
        return result;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto requires_grad_flags = ctx->saved_data["requires_grad_flags"].toIntVector();
        int64_t num_heads = ctx->saved_data["num_heads"].toInt();
        int64_t hidden_dim = ctx->saved_data["hidden_dim"].toInt();
        int64_t top_k = ctx->saved_data["top_k"].toInt();

        // ========================================================================
        // STAGE 1: Prepare inputs for recomputation.
        // This is the core of the fix. We manually emulate what `autocast` would do
        // by upcasting all floating-point inputs to float32 for the recomputation.
        // This ensures the temporary graph used for the backward pass is numerically stable.
        // ========================================================================
        std::vector<torch::Tensor> inputs_for_recompute;
        std::vector<torch::Tensor> inputs_to_grad; // A filtered list of only the tensors that require gradients.
        inputs_for_recompute.reserve(saved.size());
        inputs_to_grad.reserve(saved.size());

        for (size_t i = 0; i < saved.size(); ++i) {
            auto tensor = saved[i].detach();
            // If it's a floating point tensor, cast it to float32 for stability during recomputation.
            // Integer tensors (like policy_indices) are left as-is.
            if (tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat32);
            }

            // If the original tensor required a gradient, set requires_grad on our new (potentially upcast) copy.
            if (requires_grad_flags[i]) {
                tensor.set_requires_grad(true);
                // We only add tensors to `inputs_to_grad` if they are differentiable.
                if (tensor.is_floating_point()) {
                    inputs_to_grad.push_back(tensor);
                }
            }
            inputs_for_recompute.push_back(tensor);
        }

        // ========================================================================
        // STAGE 2: Recompute the forward pass in a grad-enabled context.
        // This builds a new, temporary computation graph using our stable float32 tensors.
        // ========================================================================
        torch::AutoGradMode enable_grad(true);
        auto recomputed = transformer_layer_forward_impl(
            inputs_for_recompute[0],  // x
            inputs_for_recompute[1],  // policy_indices
            inputs_for_recompute[2],  // in_proj_weight
            inputs_for_recompute[3],  // in_proj_bias
            inputs_for_recompute[4],  // out_proj_weight
            inputs_for_recompute[5],  // out_proj_bias
            inputs_for_recompute[6],  // norm1_weight
            inputs_for_recompute[7],  // norm1_bias
            inputs_for_recompute[8],  // gate_weight
            inputs_for_recompute[9],  // gate_bias
            inputs_for_recompute[10], // w1_all
            inputs_for_recompute[11], // w2_all
            inputs_for_recompute[12], // b1_all
            inputs_for_recompute[13], // b2_all
            inputs_for_recompute[14], // norm2_weight
            inputs_for_recompute[15], // norm2_bias
            num_heads,
            hidden_dim,
            top_k);

        // ========================================================================
        // STAGE 3: Prepare the outputs and their corresponding upstream gradients for the grad() call.
        // The upstream gradients must also be cast to float32 to match the recomputed outputs' dtype.
        // ========================================================================
        std::vector<torch::Tensor> outputs_for_grad;
        std::vector<torch::Tensor> grad_outputs_for_grad;

        // Output 0: x_next
        outputs_for_grad.push_back(std::get<0>(recomputed));
        grad_outputs_for_grad.push_back(grad_outputs[0].defined() ? grad_outputs[0].to(torch::kFloat32) : torch::zeros_like(std::get<0>(recomputed)));

        // Output 1: gate_logits
        outputs_for_grad.push_back(std::get<1>(recomputed));
        grad_outputs_for_grad.push_back(grad_outputs[1].defined() ? grad_outputs[1].to(torch::kFloat32) : torch::zeros_like(std::get<1>(recomputed)));

        // Output 2: topk_indices is non-differentiable and not returned by forward_impl, so we skip it.

        // Output 3: topk_scores
        outputs_for_grad.push_back(std::get<3>(recomputed));
        grad_outputs_for_grad.push_back(grad_outputs[3].defined() ? grad_outputs[3].to(torch::kFloat32) : torch::zeros_like(std::get<3>(recomputed)));

        // ========================================================================
        // STAGE 4: Call torch::autograd::grad on the stable float32 graph.
        // This will produce float32 gradients.
        // ========================================================================
        auto grads = torch::autograd::grad(
            outputs_for_grad,
            inputs_to_grad, // Use the filtered list of tensors that require grad
            grad_outputs_for_grad,
            /*retain_graph=*/false,
            /*create_graph=*/false,
            /*allow_unused=*/true);

        // ========================================================================
        // STAGE 5: Map the computed gradients back to the full list of results.
        // Gradients are cast back to the original dtype of the parameters.
        // ========================================================================
        torch::autograd::tensor_list results(saved.size() + 3); // +3 for num_heads, hidden_dim, top_k
        int grad_idx = 0;
        for (size_t i = 0; i < saved.size(); ++i) {
            if (requires_grad_flags[i]) {
                // Check if the input was actually differentiable (floating point)
                if (saved[i].is_floating_point()) {
                    if (grad_idx < grads.size() && grads[grad_idx].defined()) {
                        // Cast the float32 gradient back to the original parameter's dtype (e.g., float16)
                        results[i] = grads[grad_idx].to(saved[i].scalar_type());
                    }
                    grad_idx++;
                }
                // If requires_grad was true but it wasn't a float tensor (e.g. a bug),
                // we do nothing, leaving an undefined tensor in results, which is correct.
            }
            // else, results[i] is already an undefined tensor (torch::Tensor())
        }

        return results;
    }
};

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
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, Microseconds>* timers
) {
    auto device = obs_sequence.device();
    auto policy_indices_for_ops = policy_indices.to(device).to(torch::kLong).contiguous();

    c10::Dict<std::string, torch::Tensor> result;
    std::unordered_map<std::string, Microseconds> dummy_timers;
    auto& timer_ref = timers ? *timers : dummy_timers;

    // Step 1: Linear
    auto obs_linear = indexed_batched_linear_autograd(
        obs_sequence,
        get_weight(batched_weights, "obs_encoder.0.weight"),
        get_weight(batched_weights, "obs_encoder.0.bias"),
        policy_indices_for_ops
    );
    result.insert("obs_linear", obs_linear);

    // Step 2: LayerNorm (autograd-enabled for training)
    auto obs_layernorm = indexed_batched_layer_norm_autograd(
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
    auto act_kind_embed = indexed_batched_embedding_autograd(
        get_weight(batched_weights, "act_kind_embedding.weight"),
        act_kind_ids,
        policy_indices_for_ops
    );
    auto count_embed = indexed_batched_embedding_autograd(
        get_weight(batched_weights, "count_embedding.weight"),
        count_ids,
        policy_indices_for_ops
    );
    auto table_flag_embed = indexed_batched_embedding_autograd(
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
    auto agent_embed = indexed_batched_embedding_autograd(
        get_weight(batched_weights, "agent_embedding.weight"),
        agent_types.to(torch::kLong),
        policy_indices_for_ops
    );
    result.insert("agent_embed", agent_embed);

    // Position embedding
    auto position_embed = indexed_batched_embedding_autograd(
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
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, Microseconds>* timers
) {
    auto device = obs_embed.device();
    auto policy_indices_for_ops = policy_indices.to(device).to(torch::kLong).contiguous();

    c10::Dict<std::string, torch::Tensor> result;
    std::unordered_map<std::string, Microseconds> dummy_timers;
    auto& timer_ref = timers ? *timers : dummy_timers;

    // Gate for observations
    auto hidden_g_obs = indexed_batched_linear_autograd(
        obs_embed,
        get_weight(batched_weights, "gate_obs.0.weight"),
        get_weight(batched_weights, "gate_obs.0.bias"),
        policy_indices_for_ops
    );
    hidden_g_obs = torch::tanh(hidden_g_obs);
    auto g_obs = indexed_batched_linear_autograd(
        hidden_g_obs,
        get_weight(batched_weights, "gate_obs.2.weight"),
        get_weight(batched_weights, "gate_obs.2.bias"),
        policy_indices_for_ops
    );
    g_obs = torch::sigmoid(g_obs);
    result.insert("g_obs", g_obs);

    // Gate for actions
    auto hidden_g_action = indexed_batched_linear_autograd(
        action_embed,
        get_weight(batched_weights, "gate_action.0.weight"),
        get_weight(batched_weights, "gate_action.0.bias"),
        policy_indices_for_ops
    );
    hidden_g_action = torch::tanh(hidden_g_action);
    auto g_action = indexed_batched_linear_autograd(
        hidden_g_action,
        get_weight(batched_weights, "gate_action.2.weight"),
        get_weight(batched_weights, "gate_action.2.bias"),
        policy_indices_for_ops
    );
    g_action = torch::sigmoid(g_action);
    result.insert("g_action", g_action);

    // Gate for agent types
    auto hidden_g_agent = indexed_batched_linear_autograd(
        agent_embed,
        get_weight(batched_weights, "gate_agent.0.weight"),
        get_weight(batched_weights, "gate_agent.0.bias"),
        policy_indices_for_ops
    );
    hidden_g_agent = torch::tanh(hidden_g_agent);
    auto g_agent = indexed_batched_linear_autograd(
        hidden_g_agent,
        get_weight(batched_weights, "gate_agent.2.weight"),
        get_weight(batched_weights, "gate_agent.2.bias"),
        policy_indices_for_ops
    );
    g_agent = torch::sigmoid(g_agent);
    result.insert("g_agent", g_agent);

    // Gate for positions
    auto hidden_g_position = indexed_batched_linear_autograd(
        position_embed,
        get_weight(batched_weights, "gate_position.0.weight"),
        get_weight(batched_weights, "gate_position.0.bias"),
        policy_indices_for_ops
    );
    hidden_g_position = torch::tanh(hidden_g_position);
    auto g_position = indexed_batched_linear_autograd(
        hidden_g_position,
        get_weight(batched_weights, "gate_position.2.weight"),
        get_weight(batched_weights, "gate_position.2.bias"),
        policy_indices_for_ops
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
    auto gate_logits = indexed_batched_linear_autograd(
        x,
        get_weight(batched_weights, layer_prefix + ".moe.gate.weight"),
        get_weight(batched_weights, layer_prefix + ".moe.gate.bias"),
        policy_indices_for_ops
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

    // MoE expert computation
    const int64_t batch_size = x.size(0);
    const int64_t seq_len = x.size(1);

    // Use fused MoE CUDA kernel path (grouped GEMMs)
    auto w1_ptrs_gpu = get_weight(batched_weights, layer_prefix + ".moe.experts.w1_ptrs");
    auto b1_ptrs_gpu = get_weight(batched_weights, layer_prefix + ".moe.experts.b1_ptrs");
    auto w2_ptrs_gpu = get_weight(batched_weights, layer_prefix + ".moe.experts.w2_ptrs");
    auto b2_ptrs_gpu = get_weight(batched_weights, layer_prefix + ".moe.experts.b2_ptrs");

    // Pointer tensors are tiny; move to CPU to read pointer values
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

    auto topk_indices_long = topk_indices.to(torch::kLong).contiguous();
    auto flat_expert_indices = topk_indices_long.reshape({-1});
    auto flat_routing_weights = topk_weights.reshape({-1});

    auto token_indices = torch::arange(
        num_tokens,
        torch::dtype(torch::kLong).device(x.device())
    );
    auto expanded_token_indices = token_indices.unsqueeze(-1)
                                            .expand({num_tokens, top_k})
                                            .contiguous()
                                            .reshape({-1});

    // Build per-token policy indices first (flat) for keying
    auto policy_indices_long = policy_indices_for_ops.to(torch::kLong);
    auto policy_tokens = policy_indices_long.unsqueeze(1)
                                            .expand({batch_size, seq_len})
                                            .reshape({-1});
    auto flat_policy_indices = policy_tokens.index_select(0, expanded_token_indices);

    // Sort by combined (expert, policy) key to ensure single group per pair
    int64_t num_policies_for_key = num_policies_in_cache;
    auto combined_key = flat_expert_indices * num_policies_for_key + flat_policy_indices;
    auto sort_order = torch::argsort(combined_key);
    auto sorted_expert_indices = flat_expert_indices.index_select(0, sort_order);

    auto sorted_token_indices = expanded_token_indices.index_select(0, sort_order);
    auto sorted_routing_weights = flat_routing_weights.index_select(0, sort_order);
    auto sorted_policy_indices = flat_policy_indices.index_select(0, sort_order);

    auto expert_inputs = x_flat.index_select(0, sorted_token_indices).contiguous();
    auto expert_outputs = torch::zeros_like(expert_inputs);

    // Build grouped dispatch metadata on CPU to drive grouped GEMM helper
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
    std::vector<int64_t> group_policy_ids;
    std::vector<int64_t> group_expert_ids;
    std::vector<int64_t> group_token_offsets;

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

        // Bounds check
        TORCH_CHECK(policy_id >= 0 && policy_id < num_policies_in_cache,
            "Policy index out of range: ", policy_id, " / ", num_policies_in_cache);
        TORCH_CHECK(expert_id >= 0 && expert_id < num_experts,
            "Expert index out of range: ", expert_id, " / ", num_experts);

        input_ptrs.push_back(input_ptr);
        output_ptrs.push_back(output_ptr);

        const int64_t ptr_index = policy_id * num_experts_in_cache + expert_id;
        w1_ptrs.push_back(w1_ptr_data[ptr_index]);
        b1_ptrs.push_back(b1_ptr_data[ptr_index]);
        w2_ptrs.push_back(w2_ptr_data[ptr_index]);
        b2_ptrs.push_back(b2_ptr_data[ptr_index]);

        group_m_sizes.push_back(count);
        group_policy_ids.push_back(policy_id);
        group_expert_ids.push_back(expert_id);
        group_token_offsets.push_back(cursor);
        cursor = end;
    }

    // Expert FFN dimension from one of the original tensors (shape only)
    const int64_t ffn_dim = get_weight(batched_weights, layer_prefix + ".moe.experts.w1").size(-2);

    // Build routing weight pointers for each group
    // Routing weights are applied per-row in the CUTLASS GEMM epilogue
    std::vector<uintptr_t> routing_weight_ptrs;
    routing_weight_ptrs.reserve(group_m_sizes.size());

    // Convert routing weights to float for the kernel
    auto sorted_routing_weights_f32 = sorted_routing_weights.to(torch::kFloat32).contiguous();
    const float* routing_base = sorted_routing_weights_f32.data_ptr<float>();

    cursor = 0;
    for (size_t i = 0; i < group_m_sizes.size(); ++i) {
        const uintptr_t routing_ptr = reinterpret_cast<uintptr_t>(routing_base + cursor);
        routing_weight_ptrs.push_back(routing_ptr);
        cursor += group_m_sizes[i];
    }

    if (!group_m_sizes.empty()) {
        grouped_ffn_gemm_forward(
            input_ptrs.data(),
            w1_ptrs.data(),
            b1_ptrs.data(),
            w2_ptrs.data(),
            b2_ptrs.data(),
            output_ptrs.data(),
            routing_weight_ptrs.data(),
            group_m_sizes.data(),
            group_policy_ids.data(),
            group_expert_ids.data(),
            group_token_offsets.data(),
            static_cast<int64_t>(group_m_sizes.size()),
            hidden_dim,
            ffn_dim
        );
    }

    // Routing weights already applied in CUTLASS epilogue - no separate scaling needed!
    // Just scatter-add the expert outputs back to token positions
    auto moe_output_flat = torch::zeros({num_tokens, hidden_dim}, expert_outputs.options());
    moe_output_flat.index_add_(0, sorted_token_indices, expert_outputs);
    auto moe_output_half = moe_output_flat.view({batch_size, seq_len, hidden_dim});

    auto moe_output = moe_output_half.to(orig_dtype);
    result.insert("moe_output", moe_output);
    return result;
}

c10::Dict<std::string, torch::Tensor>
test_moe_routing_sort(
    const torch::Tensor& x,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    int64_t layer_idx,
    int64_t num_experts,
    int64_t top_k
) {
    auto device = x.device();
    auto policy_indices_for_ops = policy_indices.to(device).to(torch::kLong).contiguous();
    int64_t batch_size = x.size(0);
    int64_t seq_len = x.size(1);
    std::unordered_map<std::string, std::chrono::microseconds> dummy_timers;

    std::string layer_prefix = "transformer.layers." + std::to_string(layer_idx);

    // Gate logits
    auto gate_logits = indexed_batched_linear_autograd(
        x,
        get_weight(batched_weights, layer_prefix + ".moe.gate.weight"),
        get_weight(batched_weights, layer_prefix + ".moe.gate.bias"),
        policy_indices_for_ops
    );
    if (gate_logits.scalar_type() != x.scalar_type()) gate_logits = gate_logits.to(x.scalar_type());

    // Top-K
    auto topk_result = torch::topk(gate_logits, top_k, /*dim=*/-1);
    auto topk_indices = std::get<1>(topk_result);
    auto gate_probs = torch::softmax(gate_logits, /*dim=*/-1);
    auto topk_scores = torch::gather(gate_probs, /*dim=*/-1, topk_indices);
    auto topk_weights = topk_scores / topk_scores.sum(/*dim=*/-1, /*keepdim=*/true).clamp_min(1e-6);

    // Flatten + sort by expert
    const int64_t num_tokens = batch_size * seq_len;
    auto topk_indices_long = topk_indices.to(torch::kLong).contiguous();
    auto flat_expert_indices = topk_indices_long.reshape({-1});
    auto flat_routing_weights = topk_weights.reshape({-1});

    auto token_indices = torch::arange(num_tokens, torch::dtype(torch::kLong).device(x.device()));
    auto expanded_token_indices = token_indices.unsqueeze(-1).expand({num_tokens, top_k}).contiguous().reshape({-1});

    auto policy_indices_long = policy_indices_for_ops.to(torch::kLong);
    auto policy_tokens = policy_indices_long.unsqueeze(1).expand({batch_size, seq_len}).reshape({-1});
    auto flat_policy_indices = policy_tokens.index_select(0, expanded_token_indices);

    // Sort by combined (expert, policy) key to ensure one group per pair
    int64_t num_policies_for_key = get_weight(batched_weights, layer_prefix + ".moe.experts.w1").size(0);
    auto combined_key = flat_expert_indices * num_policies_for_key + flat_policy_indices;
    auto sort_order = torch::argsort(combined_key);
    auto sorted_expert_indices = flat_expert_indices.index_select(0, sort_order);
    auto sorted_token_indices = expanded_token_indices.index_select(0, sort_order);
    auto sorted_routing_weights = flat_routing_weights.index_select(0, sort_order);
    auto sorted_policy_indices = flat_policy_indices.index_select(0, sort_order);

    c10::Dict<std::string, torch::Tensor> out;
    out.insert("gate_logits", gate_logits);
    out.insert("topk_indices", topk_indices);
    out.insert("topk_weights", topk_weights);
    out.insert("sorted_expert_indices", sorted_expert_indices);
    out.insert("sorted_token_indices", sorted_token_indices);
    out.insert("sorted_policy_indices", sorted_policy_indices);
    out.insert("sorted_routing_weights", sorted_routing_weights);
    return out;
}

torch::Tensor test_moe_group_ranges(
    const torch::Tensor& sorted_expert_indices,
    const torch::Tensor& sorted_policy_indices
) {
    auto sorted_expert_cpu = sorted_expert_indices.to(torch::kCPU);
    auto sorted_policy_cpu = sorted_policy_indices.to(torch::kCPU);
    const auto* se = sorted_expert_cpu.data_ptr<int64_t>();
    const auto* sp = sorted_policy_cpu.data_ptr<int64_t>();
    const int64_t N = sorted_expert_cpu.size(0);

    std::vector<int64_t> rows;
    rows.reserve(4 * 1024);
    int64_t cursor = 0;
    while (cursor < N) {
        const int64_t expert_id = se[cursor];
        const int64_t policy_id = sp[cursor];
        int64_t end = cursor + 1;
        while (end < N && se[end] == expert_id && sp[end] == policy_id) {
            ++end;
        }
        const int64_t count = end - cursor;
        rows.push_back(cursor);
        rows.push_back(count);
        rows.push_back(expert_id);
        rows.push_back(policy_id);
        cursor = end;
    }
    auto tensor = torch::from_blob(rows.data(), {static_cast<long long>(rows.size()/4), 4}, torch::dtype(torch::kInt64)).clone();
    return tensor;
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

// Wrapper to preserve original signature (e.g., Python bindings)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed(
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
  return forward_packed(
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

// Training variant that uses autograd MoE and returns routing info
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
    int64_t tflag_pad,
    bool use_gradient_checkpointing
) {
  std::unordered_map<std::string, Microseconds> timers_dummy;
  auto policy_indices_long = policy_indices.to(torch::kLong).contiguous();
  auto policy_indices_cpu = policy_indices_long.device().is_cpu()
      ? policy_indices_long
      : policy_indices_long.cpu();
  torch::Tensor policy_indices_device = policy_indices_long;
  if (obs_sequence.is_cuda() && policy_indices_device.device() != obs_sequence.device()) {
    policy_indices_device = policy_indices_device.to(obs_sequence.device());
  }
  const torch::Tensor& policy_indices_for_ops = obs_sequence.is_cuda() ? policy_indices_device : policy_indices_cpu;

  TORCH_CHECK(obs_sequence.dim() == 3);
  TORCH_CHECK(action_sequence.dim() == 2);
  TORCH_CHECK(agent_types.dim() == 2);
  TORCH_CHECK(positions.dim() == 2);
  int64_t B = obs_sequence.size(0);
  int64_t T = obs_sequence.size(1);

  auto decomp = test_action_decomposition(
      action_sequence, batched_weights, policy_indices_for_ops,
      padding_mask, count_pad, tflag_pad);
  auto act_kind_ids = std::get<0>(decomp);
  auto count_ids    = std::get<1>(decomp);
  auto table_flag_ids = std::get<2>(decomp);

  auto embeddings = test_embeddings(
      obs_sequence, act_kind_ids, count_ids, table_flag_ids,
      agent_types, positions, batched_weights, policy_indices_for_ops);
  auto obs_encoded    = embeddings.at("obs_embed");
  auto action_embed   = embeddings.at("action_embed");
  auto agent_embed    = embeddings.at("agent_embed");
  auto position_embed = embeddings.at("position_embed");

  auto gating = test_gating(
      obs_encoded, action_embed, agent_embed, position_embed,
      batched_weights, policy_indices_for_ops);

  auto fusion = test_fusion(
      gating.at("g_obs"), gating.at("g_action"), gating.at("g_agent"), gating.at("g_position"),
      obs_encoded, action_embed, agent_embed, position_embed, hidden_dim);
  auto x = fusion.at("combined");

  torch::Tensor final_topk_indices;  // [B,T,K]
  torch::Tensor final_topk_scores;   // [B,T,K]
  std::vector<torch::Tensor> gate_logits_list; // [L,B,T,E]

  for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    std::string layer_prefix = std::string("transformer.layers.") + std::to_string(layer_idx);

    auto in_proj_weight = get_weight(batched_weights, layer_prefix + ".self_attn.in_proj_weight");
    auto in_proj_bias   = get_weight(batched_weights, layer_prefix + ".self_attn.in_proj_bias");
    auto out_proj_weight = get_weight(batched_weights, layer_prefix + ".self_attn.out_proj.weight");
    auto out_proj_bias   = get_weight(batched_weights, layer_prefix + ".self_attn.out_proj.bias");
    auto norm1_weight = get_weight(batched_weights, layer_prefix + ".norm1.weight");
    auto norm1_bias   = get_weight(batched_weights, layer_prefix + ".norm1.bias");
    auto gate_weight  = get_weight(batched_weights, layer_prefix + ".moe.gate.weight");
    auto gate_bias    = get_weight(batched_weights, layer_prefix + ".moe.gate.bias");
    auto w1_all       = get_weight(batched_weights, layer_prefix + ".moe.experts.w1");
    auto w2_all       = get_weight(batched_weights, layer_prefix + ".moe.experts.w2");
    auto b1_all       = get_weight(batched_weights, layer_prefix + ".moe.experts.b1");
    auto b2_all       = get_weight(batched_weights, layer_prefix + ".moe.experts.b2");
    auto norm2_weight = get_weight(batched_weights, layer_prefix + ".norm2.weight");
    auto norm2_bias   = get_weight(batched_weights, layer_prefix + ".norm2.bias");

    torch::Tensor gate_logits;
    torch::Tensor topk_indices;
    torch::Tensor topk_scores;
    if (use_gradient_checkpointing) {
      auto layer_outputs = TransformerLayerCheckpointFunction::apply(
          x,
          policy_indices_for_ops,
          in_proj_weight,
          in_proj_bias,
          out_proj_weight,
          out_proj_bias,
          norm1_weight,
          norm1_bias,
          gate_weight,
          gate_bias,
          w1_all,
          w2_all,
          b1_all,
          b2_all,
          norm2_weight,
          norm2_bias,
          num_heads,
          hidden_dim,
          top_k);

      x = layer_outputs[0];
      gate_logits = layer_outputs[1];
      topk_indices = layer_outputs[2];
      topk_scores = layer_outputs[3];
    } else {
      auto layer_outputs = transformer_layer_forward_impl(
          x,
          policy_indices_for_ops,
          in_proj_weight,
          in_proj_bias,
          out_proj_weight,
          out_proj_bias,
          norm1_weight,
          norm1_bias,
          gate_weight,
          gate_bias,
          w1_all,
          w2_all,
          b1_all,
          b2_all,
          norm2_weight,
          norm2_bias,
          num_heads,
          hidden_dim,
          top_k);
      x = std::get<0>(layer_outputs);
      gate_logits = std::get<1>(layer_outputs);
      topk_indices = std::get<2>(layer_outputs);
      topk_scores = std::get<3>(layer_outputs);
    }

    gate_logits_list.push_back(gate_logits);
    final_topk_indices = topk_indices;
    final_topk_scores = topk_scores;
  }

  auto transformer_output = indexed_batched_layer_norm_autograd(
      x,
      get_weight(batched_weights, "transformer.norm.weight"),
      get_weight(batched_weights, "transformer.norm.bias"),
      policy_indices_for_ops,
      1e-5);

  std::vector<torch::Tensor> action_logits_list, opp_logits_list, state_values_list, win_logits_list;
  for (int64_t i = 0; i < num_experts; ++i) {
    action_logits_list.push_back(indexed_batched_linear_autograd(transformer_output,
      get_weight(batched_weights, "action_heads." + std::to_string(i) + ".weight"),
      get_weight(batched_weights, "action_heads." + std::to_string(i) + ".bias"),
      policy_indices_for_ops));
    opp_logits_list.push_back(indexed_batched_linear_autograd(transformer_output,
      get_weight(batched_weights, "opp_action_heads." + std::to_string(i) + ".weight"),
      get_weight(batched_weights, "opp_action_heads." + std::to_string(i) + ".bias"),
      policy_indices_for_ops));
    state_values_list.push_back(indexed_batched_linear_autograd(transformer_output,
      get_weight(batched_weights, "reward_stream_heads." + std::to_string(i) + ".weight"),
      get_weight(batched_weights, "reward_stream_heads." + std::to_string(i) + ".bias"),
      policy_indices_for_ops));
    win_logits_list.push_back(indexed_batched_linear_autograd(transformer_output,
      get_weight(batched_weights, "win_prob_heads." + std::to_string(i) + ".weight"),
      get_weight(batched_weights, "win_prob_heads." + std::to_string(i) + ".bias"),
      policy_indices_for_ops));
  }

  auto action_stacked = torch::stack(action_logits_list, 2);
  auto opp_stacked    = torch::stack(opp_logits_list,    2);
  auto reward_stacked = torch::stack(state_values_list,  2);
  auto win_stacked    = torch::stack(win_logits_list,    2);
  auto action_logits = reduce_expert_heads(action_stacked, final_topk_indices, final_topk_scores);
  auto opp_logits    = reduce_expert_heads(opp_stacked,    final_topk_indices, final_topk_scores);
  auto state_values  = reduce_expert_heads(reward_stacked, final_topk_indices, final_topk_scores);
  auto win_logits    = reduce_expert_heads(win_stacked,    final_topk_indices, final_topk_scores);

  torch::Tensor gate_logits_tensor;
  if (!gate_logits_list.empty()) gate_logits_tensor = torch::stack(gate_logits_list, 0);
  else gate_logits_tensor = transformer_output.new_zeros({0, B, T, num_experts});
  std::unordered_map<std::string, torch::Tensor> routing;
  routing["gate_logits"]  = gate_logits_tensor;
  routing["topk_indices"] = final_topk_indices;
  routing["topk_scores"]  = final_topk_scores;

  return std::make_tuple(action_logits, opp_logits, state_values, win_logits,
                         gate_logits_tensor, routing);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed(
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

    // Silent by default: remove noisy debug prints

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

    // ========================================================================
    // Action Decomposition (shared with diagnostics)
    // ========================================================================
    auto decomp = test_action_decomposition(
        action_sequence,
        batched_weights,
        policy_indices_for_ops,
        padding_mask,
        count_pad,
        tflag_pad
    );
    auto act_kind_ids = std::get<0>(decomp);
    auto count_ids = std::get<1>(decomp);
    auto table_flag_ids = std::get<2>(decomp);

    // ========================================================================
    // Encoding Phase (shared helpers)
    // ========================================================================
    auto enc_t0 = Clock::now();
    auto embeddings = test_embeddings(
        obs_sequence,
        act_kind_ids,
        count_ids,
        table_flag_ids,
        agent_types,
        positions,
        batched_weights,
        policy_indices_for_ops,
        &timers
    );
    auto obs_encoded = embeddings.at("obs_embed");
    auto action_embed = embeddings.at("action_embed");
    auto agent_embed = embeddings.at("agent_embed");
    auto position_embed = embeddings.at("position_embed");

    auto gating = test_gating(
        obs_encoded,
        action_embed,
        agent_embed,
        position_embed,
        batched_weights,
        policy_indices_for_ops,
        &timers
    );

    auto fusion = test_fusion(
        gating.at("g_obs"),
        gating.at("g_action"),
        gating.at("g_agent"),
        gating.at("g_position"),
        obs_encoded,
        action_embed,
        agent_embed,
        position_embed,
        hidden_dim
    );
    auto encoded_inputs = fusion.at("combined");
    
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

        auto topk_indices_long = topk_indices.to(torch::kLong).contiguous();
        auto flat_expert_indices = topk_indices_long.reshape({-1});
        auto flat_routing_weights = topk_weights.reshape({-1});

        auto token_indices = torch::arange(
            num_tokens,
            torch::dtype(torch::kLong).device(x.device())
        );
        auto expanded_token_indices = token_indices.unsqueeze(-1)
                                                .expand({num_tokens, top_k})
                                                .contiguous()
                                                .reshape({-1});

        auto policy_indices_long = policy_indices_for_ops.to(torch::kLong);
        auto policy_tokens = policy_indices_long.unsqueeze(1)
                                                    .expand({batch_size, seq_len})
                                                    .reshape({-1});
        auto flat_policy_indices = policy_tokens.index_select(0, expanded_token_indices);

        // Sort by combined (expert, policy) key to ensure single group per pair
        int64_t num_policies_for_key = num_policies_in_cache;
        auto combined_key = flat_expert_indices * num_policies_for_key + flat_policy_indices;
        auto sort_order = torch::argsort(combined_key);
        auto sorted_expert_indices = flat_expert_indices.index_select(0, sort_order);

        auto sorted_token_indices = expanded_token_indices.index_select(0, sort_order);
        auto sorted_routing_weights = flat_routing_weights.index_select(0, sort_order);
        auto sorted_policy_indices = flat_policy_indices.index_select(0, sort_order);

        auto expert_inputs = x_flat.index_select(0, sorted_token_indices).contiguous();
        // Clamp to finite FP16 range to avoid INF in kernels
        expert_inputs = expert_inputs.clamp(-65504.0, 65504.0);
        auto expert_outputs = torch::zeros_like(expert_inputs);

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
        std::vector<int64_t> group_policy_ids;
        std::vector<int64_t> group_expert_ids;
        std::vector<int64_t> group_token_offsets;

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
            group_policy_ids.push_back(policy_id);
            group_expert_ids.push_back(expert_id);
            group_token_offsets.push_back(cursor);

            cursor = end;
        }

        // Expert FFN dimension from one of the original tensors (shape only)
        const int64_t ffn_dim = get_weight(batched_weights, layer_prefix + ".moe.experts.w1").size(-2);

        // Build routing weight pointers for each group
        std::vector<uintptr_t> routing_weight_ptrs;
        routing_weight_ptrs.reserve(group_m_sizes.size());

        auto sorted_routing_weights_f32 = sorted_routing_weights.to(torch::kFloat32).contiguous();
        const float* routing_base = sorted_routing_weights_f32.data_ptr<float>();

        cursor = 0;
        for (size_t i = 0; i < group_m_sizes.size(); ++i) {
            const uintptr_t routing_ptr = reinterpret_cast<uintptr_t>(routing_base + cursor);
            routing_weight_ptrs.push_back(routing_ptr);
            cursor += group_m_sizes[i];
        }

        if (!group_m_sizes.empty()) {
            grouped_ffn_gemm_forward(
                input_ptrs.data(),
                w1_ptrs.data(),
                b1_ptrs.data(),
                w2_ptrs.data(),
                b2_ptrs.data(),
                output_ptrs.data(),
                routing_weight_ptrs.data(),
                group_m_sizes.data(),
                group_policy_ids.data(),
                group_expert_ids.data(),
                group_token_offsets.data(),
                static_cast<int64_t>(group_m_sizes.size()),
                hidden_dim,
                ffn_dim
            );
        }

        // Routing weights already applied in CUTLASS epilogue
        auto moe_output_flat = torch::zeros({num_tokens, hidden_dim}, expert_outputs.options());
        moe_output_flat.index_add_(0, sorted_token_indices, expert_outputs);
        auto moe_output_half = moe_output_flat.view({batch_size, seq_len, hidden_dim});

        moe_output = moe_output_half.to(orig_dtype);
        
        
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
