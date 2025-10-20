#include "reactive_model_forward.h"
#include "indexed_kernels.h"
#include "moe_kernel.h"

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <unordered_map>

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

    // Fused embedding
    auto fused = g_obs * obs_encoded
               + g_action * act_embed
               + g_agent * agent_embed
               + g_position * position_embed;

    // Final layer norm (uses torch::layer_norm, not batched version)
    auto encoded_inputs = torch::layer_norm(fused, {hidden_dim});
    if (obs_sequence.is_cuda()) { torch::cuda::synchronize(); }
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
        if (x.is_cuda()) { torch::cuda::synchronize(); }
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
        if (x.is_cuda()) { torch::cuda::synchronize(); }
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
        if (x.is_cuda()) { torch::cuda::synchronize(); }
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
            // Get pre-stacked expert weights and pointer metadata
            // NOTE: Weights are already FP16 and contiguous from eval_manager, don't call .to().contiguous()
            // because that creates a new tensor and invalidates the pointer tensors!
            auto expert_w1 = get_weight(batched_weights, layer_prefix + ".moe.experts.w1");
            auto expert_b1 = get_weight(batched_weights, layer_prefix + ".moe.experts.b1");
            auto expert_w2 = get_weight(batched_weights, layer_prefix + ".moe.experts.w2");
            auto expert_b2 = get_weight(batched_weights, layer_prefix + ".moe.experts.b2");

            static bool printed_expert_w1_addr = false;
            if (!printed_expert_w1_addr && layer_idx == 0) {
                uintptr_t expert_w1_addr = reinterpret_cast<uintptr_t>(expert_w1.data_ptr<at::Half>());
                fprintf(stderr, "[DEBUG forward_packed_cpp] Layer 0 expert_w1[0][0] address: 0x%lx\n", expert_w1_addr);
                printed_expert_w1_addr = true;
            }

            auto expert_w1_ptrs = get_weight(batched_weights, layer_prefix + ".moe.experts.w1_ptrs");
            auto expert_b1_ptrs = get_weight(batched_weights, layer_prefix + ".moe.experts.b1_ptrs");
            auto expert_w2_ptrs = get_weight(batched_weights, layer_prefix + ".moe.experts.w2_ptrs");
            auto expert_b2_ptrs = get_weight(batched_weights, layer_prefix + ".moe.experts.b2_ptrs");

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
                                                  .reshape({-1});

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

            // Build grouped dispatch metadata on CPU to drive the grouped GEMM helper
            auto sorted_expert_cpu = sorted_expert_indices.to(torch::kCPU);
            auto sorted_policy_cpu = sorted_policy_indices.to(torch::kCPU);

            const auto* sorted_expert_ptr = sorted_expert_cpu.data_ptr<int64_t>();
            const auto* sorted_policy_ptr = sorted_policy_cpu.data_ptr<int64_t>();

            auto w1_ptr_cpu = expert_w1_ptrs.to(torch::kCPU).contiguous();
            auto b1_ptr_cpu = expert_b1_ptrs.to(torch::kCPU).contiguous();
            auto w2_ptr_cpu = expert_w2_ptrs.to(torch::kCPU).contiguous();
            auto b2_ptr_cpu = expert_b2_ptrs.to(torch::kCPU).contiguous();

            // Verify pointer tensor dimensions
            TORCH_CHECK(w1_ptr_cpu.dim() == 2, "w1_ptrs must be 2D");
            const int64_t num_policies_in_cache = w1_ptr_cpu.size(0);
            const int64_t num_experts_in_ptrs = w1_ptr_cpu.size(1);

            // Debug: print pointer tensor shape on first call
            static bool printed_once = false;
            if (!printed_once) {
                fprintf(stderr, "[DEBUG MoE] Pointer tensor shape: [%ld, %ld]\n",
                        num_policies_in_cache, num_experts_in_ptrs);
                fprintf(stderr, "[DEBUG MoE] Expected: num_experts=%ld\n", num_experts);
                fprintf(stderr, "[DEBUG MoE] Stacked expert w1 shape: [%ld, %ld, %ld, %ld]\n",
                        expert_w1.size(0), expert_w1.size(1), expert_w1.size(2), expert_w1.size(3));
                printed_once = true;
            }

            TORCH_CHECK(num_experts_in_ptrs == num_experts,
                "Pointer tensor expert dimension mismatch: expected ", num_experts, ", got ", num_experts_in_ptrs);

            const auto* w1_ptr_data = w1_ptr_cpu.data_ptr<uint64_t>();
            const auto* b1_ptr_data = b1_ptr_cpu.data_ptr<uint64_t>();
            const auto* w2_ptr_data = w2_ptr_cpu.data_ptr<uint64_t>();
            const auto* b2_ptr_data = b2_ptr_cpu.data_ptr<uint64_t>();

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

                const int64_t ptr_idx = policy_id * num_experts + expert_id;
                input_ptrs.push_back(input_ptr);
                output_ptrs.push_back(output_ptr);
                w1_ptrs.push_back(static_cast<uintptr_t>(w1_ptr_data[ptr_idx]));
                b1_ptrs.push_back(static_cast<uintptr_t>(b1_ptr_data[ptr_idx]));
                w2_ptrs.push_back(static_cast<uintptr_t>(w2_ptr_data[ptr_idx]));
                b2_ptrs.push_back(static_cast<uintptr_t>(b2_ptr_data[ptr_idx]));
                group_m_sizes.push_back(count);
                groups.push_back({cursor, count, expert_id, policy_id});

                cursor = end;
            }

            const int64_t ffn_dim = expert_w1.size(-2);

            if (!groups.empty()) {
                grouped_ffn_gemm_forward(
                    input_ptrs.data(),
                    w1_ptrs.data(),
                    b1_ptrs.data(),
                    w2_ptrs.data(),
                    b2_ptrs.data(),
                    output_ptrs.data(),
                    group_m_sizes.data(),
                    static_cast<int64_t>(groups.size()),
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
        if (x.is_cuda()) { torch::cuda::synchronize(); }
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
        if (x.is_cuda()) { torch::cuda::synchronize(); }
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
    if (x.is_cuda()) { torch::cuda::synchronize(); }
    auto t_heads_1 = Clock::now();
    timers["fwd_heads_us"] += std::chrono::duration_cast<Microseconds>(t_heads_1 - t_heads_0);

    return std::make_tuple(action_logits, opp_logits, state_values, win_logits);
}
