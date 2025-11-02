#include "model_forward.h"
#include "model_layer.h"
#include "model_autograd.h"
#include "moe_cutlass_kernels.h"  // For lb::moe::MoEWorkspace
#include <cuda_runtime.h>
#include "lb_kernels.h"
#include "moe_autograd_function.h"

#include <torch/torch.h>

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

// THIS IS THE CRITICAL CHANGE: An autograd-enabled implementation for recomputation.
// It is kept private to this file as an implementation detail of the checkpointing function.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
transformer_layer_forward_impl_autograd(
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

    // --- Attention Block (using autograd functions) ---
    int64_t weight_chunk_dim = (in_proj_weight.dim() == 3) ? 1 : 0;
    int64_t bias_chunk_dim   = (in_proj_bias.dim() == 2) ? 1 : 0;
    auto qkv_weights = in_proj_weight.chunk(3, weight_chunk_dim);
    auto qkv_biases  = in_proj_bias.chunk(3, bias_chunk_dim);

    auto q = lb::autograd::indexed_batched_linear_autograd(x, qkv_weights[0], qkv_biases[0], policy_indices);
    auto k = lb::autograd::indexed_batched_linear_autograd(x, qkv_weights[1], qkv_biases[1], policy_indices);
    auto v = lb::autograd::indexed_batched_linear_autograd(x, qkv_weights[2], qkv_biases[2], policy_indices);

    q = q.view({B, T, num_heads, head_dim}).transpose(1, 2);
    k = k.view({B, T, num_heads, head_dim}).transpose(1, 2);
    v = v.view({B, T, num_heads, head_dim}).transpose(1, 2);

    auto attn_output = torch::scaled_dot_product_attention(q, k, v, torch::nullopt, 0.0, true);
    attn_output = attn_output.transpose(1, 2).contiguous().view({B, T, hidden_dim});
    attn_output = lb::autograd::indexed_batched_linear_autograd(attn_output, out_proj_weight, out_proj_bias, policy_indices);

    auto residual1 = x + attn_output;
    auto x_norm = lb::autograd::indexed_batched_layer_norm_autograd(residual1, norm1_weight, norm1_bias, policy_indices, 1e-5);

    // --- MoE Block (using autograd function) ---
    auto gate_logits = lb::autograd::indexed_batched_linear_autograd(x_norm, gate_weight, gate_bias, policy_indices);
    auto probs = torch::softmax(gate_logits, -1);
    auto topk_vals_idx = torch::topk(gate_logits, top_k, -1);
    auto topk_indices = std::get<1>(topk_vals_idx);
    auto topk_scores  = torch::gather(probs, -1, topk_indices);
    auto topk_weights = topk_scores / topk_scores.sum(-1, /*keepdim=*/true).clamp_min(1e-6);

    auto num_tokens = B * T;
    auto x_norm_fp16 = x_norm.to(torch::kHalf).contiguous();
    auto x_flat = x_norm_fp16.view({num_tokens, hidden_dim});

    auto topk_indices_long = topk_indices.to(torch::kLong).contiguous();
    auto flat_expert_indices = topk_indices_long.reshape({-1});
    auto flat_routing_weights = topk_weights.reshape({-1});
    auto token_indices = torch::arange(num_tokens, torch::dtype(torch::kLong).device(x.device()));
    auto expanded_token_indices = token_indices.unsqueeze(-1).expand({num_tokens, top_k}).reshape({-1});
    auto policy_indices_long = policy_indices.to(torch::kLong);
    auto policy_tokens = policy_indices_long.unsqueeze(1).expand({B, T}).reshape({-1});
    auto flat_policy_indices = policy_tokens.index_select(0, expanded_token_indices);

    int64_t num_policies_for_key = w1_all.size(0);
    auto combined_key = flat_expert_indices * num_policies_for_key + flat_policy_indices;
    auto sort_order = torch::argsort(combined_key);
    auto sorted_expert_indices = flat_expert_indices.index_select(0, sort_order);
    auto sorted_token_indices = expanded_token_indices.index_select(0, sort_order);
    auto sorted_routing_weights = flat_routing_weights.index_select(0, sort_order).to(torch::kFloat32);
    auto sorted_policy_indices = flat_policy_indices.index_select(0, sort_order);

    auto expert_inputs = x_flat.index_select(0, sorted_token_indices).contiguous();

    torch::Tensor expert_outputs;

    // Check if we need gradients (training/backward) or not (inference)
    bool needs_grad = expert_inputs.requires_grad() || w1_all.requires_grad() || w2_all.requires_grad();

    if (needs_grad) {
        // TRAINING PATH: Use autograd function to maintain gradient flow
        auto [m_sizes_cpu, policy_ids_cpu, expert_ids_cpu, token_offsets_cpu] =
            lb::model::moe_group_metadata(sorted_expert_indices, sorted_policy_indices);

        auto build_ptr_table = [&](const torch::Tensor& stacked) -> torch::Tensor {
            TORCH_CHECK(stacked.is_cuda(), "Stacked expert weights must be CUDA tensors");
            const int64_t P = stacked.size(0);
            const int64_t E = stacked.size(1);
            auto tbl_cpu = torch::empty({P, E}, torch::dtype(torch::kUInt64).device(torch::kCPU));
            auto acc = tbl_cpu.accessor<uint64_t, 2>();
            for (int64_t p = 0; p < P; ++p) {
                for (int64_t e = 0; e < E; ++e) {
                    auto slice = stacked.index({p, e});
                    acc[p][e] = reinterpret_cast<uint64_t>(slice.data_ptr());
                }
            }
            return tbl_cpu.to(stacked.device());
        };

        auto w1_tbl = build_ptr_table(w1_all.to(torch::kFloat16));
        auto w2_tbl = build_ptr_table(w2_all.to(torch::kFloat16));
        auto b1_tbl = build_ptr_table(b1_all.to(torch::kFloat16));
        auto b2_tbl = build_ptr_table(b2_all.to(torch::kFloat16));

        expert_outputs = lb::moe::grouped_moe_autograd_forward(
            expert_inputs, w1_all.to(torch::kFloat16), w2_all.to(torch::kFloat16),
            b1_all.to(torch::kFloat16), b2_all.to(torch::kFloat16),
            w1_tbl, w2_tbl, b1_tbl, b2_tbl,
            sorted_routing_weights, sorted_token_indices,
            m_sizes_cpu, policy_ids_cpu, expert_ids_cpu, token_offsets_cpu,
            hidden_dim, w1_all.size(-2), nullptr);
    } else {
        // INFERENCE PATH: Use optimized device-side path (no autograd overhead)
        auto [m_sizes_dev, policy_ids_dev, expert_ids_dev, token_offsets_dev] =
            lb::model::moe_group_metadata_device(sorted_expert_indices, sorted_policy_indices);

        auto w1_fp16 = w1_all.to(torch::kFloat16);
        auto w2_fp16 = w2_all.to(torch::kFloat16);
        auto b1_fp16 = b1_all.to(torch::kFloat16);
        auto b2_fp16 = b2_all.to(torch::kFloat16);

        auto w1_tbl = lb::model::build_ptr_table_device(w1_fp16);
        auto w2_tbl = lb::model::build_ptr_table_device(w2_fp16);
        auto b1_tbl = lb::model::build_ptr_table_device(b1_fp16);
        auto b2_tbl = lb::model::build_ptr_table_device(b2_fp16);

        int64_t total_tokens = expert_inputs.size(0);
        int64_t ffn_dim = w1_all.size(-2);
        expert_outputs = torch::empty({total_tokens, hidden_dim}, expert_inputs.options());
        auto hidden_tmp = torch::empty({total_tokens, ffn_dim}, expert_inputs.options());
        int64_t group_count = m_sizes_dev.size(0);

        lb::moe::cutlass_grouped_moe_forward_with_hidden_device(
            reinterpret_cast<uintptr_t>(expert_inputs.data_ptr<at::Half>()),
            reinterpret_cast<uintptr_t>(hidden_tmp.data_ptr<at::Half>()),
            reinterpret_cast<uintptr_t>(expert_outputs.data_ptr<at::Half>()),
            reinterpret_cast<uintptr_t>(sorted_routing_weights.data_ptr<float>()),
            w1_tbl.data_ptr<uint64_t>(),
            w2_tbl.data_ptr<uint64_t>(),
            b1_tbl.data_ptr<uint64_t>(),
            b2_tbl.data_ptr<uint64_t>(),
            w1_fp16.size(0), w1_fp16.size(1),
            m_sizes_dev.data_ptr<int64_t>(),
            policy_ids_dev.data_ptr<int64_t>(),
            expert_ids_dev.data_ptr<int64_t>(),
            token_offsets_dev.data_ptr<int64_t>(),
            group_count, hidden_dim, ffn_dim, nullptr);
    }

    auto moe_output_flat = torch::zeros({num_tokens, hidden_dim}, expert_outputs.options());
    moe_output_flat.index_add_(0, sorted_token_indices, expert_outputs);
    auto moe_output = moe_output_flat.view({B, T, hidden_dim}).to(x_norm.dtype());

    auto residual2 = x_norm + moe_output;
    auto x_next = lb::autograd::indexed_batched_layer_norm_autograd(residual2, norm2_weight, norm2_bias, policy_indices, 1e-5);

    return std::make_tuple(x_next, gate_logits, topk_indices, topk_weights);
}


// Gradient checkpointing implementation for a single transformer layer.
struct TransformerLayerCheckpointFunction
    : public torch::autograd::Function<TransformerLayerCheckpointFunction> {

    // Forward pass for checkpointing just calls the non-autograd layer implementation
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& x, const torch::Tensor& policy_indices,
        const torch::Tensor& in_proj_weight, const torch::Tensor& in_proj_bias,
        const torch::Tensor& out_proj_weight, const torch::Tensor& out_proj_bias,
        const torch::Tensor& norm1_weight, const torch::Tensor& norm1_bias,
        const torch::Tensor& gate_weight, const torch::Tensor& gate_bias,
        const torch::Tensor& w1_all, const torch::Tensor& w2_all,
        const torch::Tensor& b1_all, const torch::Tensor& b2_all,
        const torch::Tensor& norm2_weight, const torch::Tensor& norm2_bias,
        int64_t num_heads, int64_t hidden_dim, int64_t top_k) {
        
        torch::NoGradGuard no_grad;

        // Call the autograd-enabled implementation but with grads disabled.
        // This computes the forward values correctly without building a graph.
        auto outputs = transformer_layer_forward_impl_autograd(
            x, policy_indices, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
            norm1_weight, norm1_bias, gate_weight, gate_bias, w1_all, w2_all, b1_all, b2_all,
            norm2_weight, norm2_bias, num_heads, hidden_dim, top_k);

        // --- Save tensors and metadata for backward ---
        std::vector<torch::Tensor> to_save;
        std::vector<int64_t> requires_grad_flags;
        auto save_tensor = [&](const torch::Tensor& t) {
            to_save.push_back(t.detach());
            requires_grad_flags.push_back(t.requires_grad() ? 1 : 0);
        };

        save_tensor(x); save_tensor(policy_indices);
        save_tensor(in_proj_weight); save_tensor(in_proj_bias);
        save_tensor(out_proj_weight); save_tensor(out_proj_bias);
        save_tensor(norm1_weight); save_tensor(norm1_bias);
        save_tensor(gate_weight); save_tensor(gate_bias);
        save_tensor(w1_all); save_tensor(w2_all);
        save_tensor(b1_all); save_tensor(b2_all);
        save_tensor(norm2_weight); save_tensor(norm2_bias);
        
        ctx->save_for_backward(to_save);
        ctx->saved_data["requires_grad_flags"] = requires_grad_flags;
        ctx->saved_data["num_heads"] = num_heads;
        ctx->saved_data["hidden_dim"] = hidden_dim;
        ctx->saved_data["top_k"] = top_k;

        torch::autograd::tensor_list result(4);
        result[0] = std::get<0>(outputs);
        result[1] = std::get<1>(outputs);
        result[2] = std::get<2>(outputs);
        result[3] = std::get<3>(outputs);
        return result;
    }

    // Backward recomputes the forward pass to build a local graph for gradient calculation
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        
        auto saved = ctx->get_saved_variables();
        auto requires_grad_flags = ctx->saved_data["requires_grad_flags"].toIntVector();
        int64_t num_heads = ctx->saved_data["num_heads"].toInt();
        int64_t hidden_dim = ctx->saved_data["hidden_dim"].toInt();
        int64_t top_k = ctx->saved_data["top_k"].toInt();

        // --- Prepare inputs for recomputation ---
        std::vector<torch::Tensor> inputs_for_recompute;
        std::vector<torch::Tensor> inputs_to_grad;
        inputs_for_recompute.reserve(saved.size());
        inputs_to_grad.reserve(saved.size());

        for (size_t i = 0; i < saved.size(); ++i) {
            auto tensor = saved[i];
            if (tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat32); // Upcast for stability
            }
            if (requires_grad_flags[i]) {
                tensor.set_requires_grad(true);
                if (tensor.is_floating_point()) {
                    inputs_to_grad.push_back(tensor);
                }
            }
            inputs_for_recompute.push_back(tensor);
        }

        // --- Recompute forward pass with autograd enabled ---
        torch::AutoGradMode enable_grad(true);
        auto recomputed = transformer_layer_forward_impl_autograd(
            inputs_for_recompute[0], inputs_for_recompute[1],
            inputs_for_recompute[2], inputs_for_recompute[3], inputs_for_recompute[4], inputs_for_recompute[5],
            inputs_for_recompute[6], inputs_for_recompute[7], inputs_for_recompute[8], inputs_for_recompute[9],
            inputs_for_recompute[10], inputs_for_recompute[11], inputs_for_recompute[12], inputs_for_recompute[13],
            inputs_for_recompute[14], inputs_for_recompute[15],
            num_heads, hidden_dim, top_k);

        // --- Prepare outputs for torch::autograd::grad ---
        std::vector<torch::Tensor> outputs_for_grad;
        std::vector<torch::Tensor> grad_outputs_for_grad;

        outputs_for_grad.push_back(std::get<0>(recomputed));
        grad_outputs_for_grad.push_back(grad_outputs[0].defined() ? grad_outputs[0].to(torch::kFloat32) : torch::zeros_like(std::get<0>(recomputed)));

        outputs_for_grad.push_back(std::get<1>(recomputed));
        grad_outputs_for_grad.push_back(grad_outputs[1].defined() ? grad_outputs[1].to(torch::kFloat32) : torch::zeros_like(std::get<1>(recomputed)));

        outputs_for_grad.push_back(std::get<3>(recomputed)); // topk_weights
        grad_outputs_for_grad.push_back(grad_outputs[3].defined() ? grad_outputs[3].to(torch::kFloat32) : torch::zeros_like(std::get<3>(recomputed)));
        
        // --- Compute gradients ---
        auto grads = torch::autograd::grad(
            outputs_for_grad, inputs_to_grad, grad_outputs_for_grad,
            false, false, true);

        // --- Map gradients back to original inputs ---
        torch::autograd::tensor_list results(saved.size() + 3); // +3 for non-tensor args
        int grad_idx = 0;
        for (size_t i = 0; i < saved.size(); ++i) {
            if (requires_grad_flags[i] && saved[i].is_floating_point()) {
                if (grad_idx < grads.size() && grads[grad_idx].defined()) {
                    results[i] = grads[grad_idx].to(saved[i].scalar_type());
                }
                grad_idx++;
            }
        }
        return results;
    }
};

} // anonymous namespace

namespace lb {
namespace forward {

namespace {
// Thread-local workspace cache for direct forward_packed callers (e.g., profiling scripts)
thread_local bool g_tls_workspace_enabled = false;
thread_local std::vector<lb::moe::MoEWorkspace> g_tls_workspaces;

inline void free_workspace(lb::moe::MoEWorkspace& ws) {
    if (ws.hidden_buffer) { cudaFree(ws.hidden_buffer); ws.hidden_buffer = nullptr; ws.hidden_buffer_size = 0; }
    if (ws.workspace_w1) { cudaFree(ws.workspace_w1); ws.workspace_w1 = nullptr; ws.workspace_w1_size = 0; }
    if (ws.workspace_w2) { cudaFree(ws.workspace_w2); ws.workspace_w2 = nullptr; ws.workspace_w2_size = 0; }

    if (ws.problem_sizes_device_w1) { cudaFree(ws.problem_sizes_device_w1); ws.problem_sizes_device_w1 = nullptr; }
    if (ws.ptr_A_device_w1) { cudaFree(ws.ptr_A_device_w1); ws.ptr_A_device_w1 = nullptr; }
    if (ws.ptr_B_device_w1) { cudaFree(ws.ptr_B_device_w1); ws.ptr_B_device_w1 = nullptr; }
    if (ws.ptr_C_device_w1) { cudaFree(ws.ptr_C_device_w1); ws.ptr_C_device_w1 = nullptr; }
    if (ws.ptr_D_device_w1) { cudaFree(ws.ptr_D_device_w1); ws.ptr_D_device_w1 = nullptr; }
    if (ws.lda_device_w1) { cudaFree(ws.lda_device_w1); ws.lda_device_w1 = nullptr; }
    if (ws.ldb_device_w1) { cudaFree(ws.ldb_device_w1); ws.ldb_device_w1 = nullptr; }
    if (ws.ldc_device_w1) { cudaFree(ws.ldc_device_w1); ws.ldc_device_w1 = nullptr; }
    if (ws.ldd_device_w1) { cudaFree(ws.ldd_device_w1); ws.ldd_device_w1 = nullptr; }
    ws.descriptor_capacity_w1 = 0;

    if (ws.problem_sizes_device_w2) { cudaFree(ws.problem_sizes_device_w2); ws.problem_sizes_device_w2 = nullptr; }
    if (ws.ptr_A_device_w2) { cudaFree(ws.ptr_A_device_w2); ws.ptr_A_device_w2 = nullptr; }
    if (ws.ptr_B_device_w2) { cudaFree(ws.ptr_B_device_w2); ws.ptr_B_device_w2 = nullptr; }
    if (ws.ptr_C_device_w2) { cudaFree(ws.ptr_C_device_w2); ws.ptr_C_device_w2 = nullptr; }
    if (ws.ptr_D_device_w2) { cudaFree(ws.ptr_D_device_w2); ws.ptr_D_device_w2 = nullptr; }
    if (ws.lda_device_w2) { cudaFree(ws.lda_device_w2); ws.lda_device_w2 = nullptr; }
    if (ws.ldb_device_w2) { cudaFree(ws.ldb_device_w2); ws.ldb_device_w2 = nullptr; }
    if (ws.ldc_device_w2) { cudaFree(ws.ldc_device_w2); ws.ldc_device_w2 = nullptr; }
    if (ws.ldd_device_w2) { cudaFree(ws.ldd_device_w2); ws.ldd_device_w2 = nullptr; }
    ws.descriptor_capacity_w2 = 0;
}
} // anonymous namespace

void enable_forward_workspace_cache(
    int64_t num_layers,
    int64_t max_batch_size,
    int64_t max_seq_length,
    int64_t hidden_dim,
    int64_t top_k) {
    // Clean any existing cache first
    if (!g_tls_workspaces.empty()) {
        for (auto& ws : g_tls_workspaces) free_workspace(ws);
        g_tls_workspaces.clear();
    }

    g_tls_workspaces.resize(num_layers);
    const int64_t max_tokens = max_batch_size * max_seq_length;
    const int64_t max_groups = max_tokens * top_k;
    const int64_t ffn_dim = hidden_dim * 2; // typical expansion ratio

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        auto& ws = g_tls_workspaces[layer];

        // Hidden buffer
        ws.hidden_buffer_size = static_cast<size_t>(max_tokens) * static_cast<size_t>(ffn_dim) * sizeof(uint16_t);
        cudaMalloc(&ws.hidden_buffer, ws.hidden_buffer_size);

        // CUTLASS workspaces
        ws.workspace_w1_size = 4 * 1024 * 1024;
        cudaMalloc(&ws.workspace_w1, ws.workspace_w1_size);
        ws.workspace_w2_size = 4 * 1024 * 1024;
        cudaMalloc(&ws.workspace_w2, ws.workspace_w2_size);

        // Problem descriptors (capacity per GEMM group)
        ws.descriptor_capacity_w1 = max_groups;
        cudaMalloc(&ws.problem_sizes_device_w1, max_groups * sizeof(int) * 3);  // GemmCoord (M,N,K)
        cudaMalloc(&ws.ptr_A_device_w1, max_groups * sizeof(void*));
        cudaMalloc(&ws.ptr_B_device_w1, max_groups * sizeof(void*));
        cudaMalloc(&ws.ptr_C_device_w1, max_groups * sizeof(void*));
        cudaMalloc(&ws.ptr_D_device_w1, max_groups * sizeof(void*));
        cudaMalloc(&ws.lda_device_w1, max_groups * sizeof(int64_t));
        cudaMalloc(&ws.ldb_device_w1, max_groups * sizeof(int64_t));
        cudaMalloc(&ws.ldc_device_w1, max_groups * sizeof(int64_t));
        cudaMalloc(&ws.ldd_device_w1, max_groups * sizeof(int64_t));

        ws.descriptor_capacity_w2 = max_groups;
        cudaMalloc(&ws.problem_sizes_device_w2, max_groups * sizeof(int) * 3);
        cudaMalloc(&ws.ptr_A_device_w2, max_groups * sizeof(void*));
        cudaMalloc(&ws.ptr_B_device_w2, max_groups * sizeof(void*));
        cudaMalloc(&ws.ptr_C_device_w2, max_groups * sizeof(void*));
        cudaMalloc(&ws.ptr_D_device_w2, max_groups * sizeof(void*));
        cudaMalloc(&ws.lda_device_w2, max_groups * sizeof(int64_t));
        cudaMalloc(&ws.ldb_device_w2, max_groups * sizeof(int64_t));
        cudaMalloc(&ws.ldc_device_w2, max_groups * sizeof(int64_t));
        cudaMalloc(&ws.ldd_device_w2, max_groups * sizeof(int64_t));
    }

    g_tls_workspace_enabled = true;
}

void disable_forward_workspace_cache() {
    for (auto& ws : g_tls_workspaces) {
        free_workspace(ws);
    }
    g_tls_workspaces.clear();
    g_tls_workspace_enabled = false;
}

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
    int64_t tflag_pad,
    std::unordered_map<std::string, std::chrono::microseconds>* timers,
    const lb::moe::MoEWorkspace* moe_workspaces) {

    using Microseconds = std::chrono::microseconds;
    using Clock = std::chrono::high_resolution_clock;
    std::unordered_map<std::string, Microseconds> dummy_timers;
    auto& t = timers ? *timers : dummy_timers;
    
    auto pol = policy_indices.to(obs_sequence.device()).to(torch::kLong).contiguous();

    // 1. Embeddings
    auto t0 = Clock::now();
    auto embeds = lb::model::compute_embeddings(
        obs_sequence, action_sequence, agent_types, positions,
        batched_weights, pol, padding_mask, count_pad, tflag_pad, &t);
    auto t1 = Clock::now();
    t["forward_embeddings_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // 2. Gating and Fusion
    t0 = Clock::now();
    auto gating = lb::model::gating(
        embeds.at("obs_embed"), embeds.at("action_embed"),
        embeds.at("agent_embed"), embeds.at("position_embed"),
        batched_weights, pol, &t);
    t1 = Clock::now();
    t["forward_gating_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    t0 = Clock::now();
    auto fused = lb::model::fuse_embeddings(
        gating.at("g_obs"), gating.at("g_action"), gating.at("g_agent"), gating.at("g_position"),
        embeds.at("obs_embed"), embeds.at("action_embed"), embeds.at("agent_embed"), embeds.at("position_embed"),
        hidden_dim);
    t1 = Clock::now();
    t["forward_fusion_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    auto x = fused.at("combined");

    // 3. Transformer Layers
    torch::Tensor last_topk_idx, last_topk_scores;
    for (int64_t i = 0; i < num_layers; ++i) {
        std::string prefix = "transformer.layers." + std::to_string(i);
        std::string layer_key = "forward_layer_" + std::to_string(i) + "_us";
        
        // Select per-layer workspace if provided
        lb::moe::MoEWorkspace* ws_ptr = nullptr;
        if (moe_workspaces) {
            // Pointer is assumed to reference an array of length >= num_layers
            ws_ptr = const_cast<lb::moe::MoEWorkspace*>(moe_workspaces + i);
        } else if (g_tls_workspace_enabled && i < static_cast<int64_t>(g_tls_workspaces.size())) {
            ws_ptr = &g_tls_workspaces[i];
        }
        
        t0 = Clock::now();
        auto [x_next, gate_logits, topk_indices, topk_scores] = lb::model::transformer_layer(
            x, pol,
            get_weight(batched_weights, prefix + ".self_attn.in_proj_weight"),
            get_weight(batched_weights, prefix + ".self_attn.in_proj_bias"),
            get_weight(batched_weights, prefix + ".self_attn.out_proj.weight"),
            get_weight(batched_weights, prefix + ".self_attn.out_proj.bias"),
            get_weight(batched_weights, prefix + ".norm1.weight"),
            get_weight(batched_weights, prefix + ".norm1.bias"),
            get_weight(batched_weights, prefix + ".moe.gate.weight"),
            get_weight(batched_weights, prefix + ".moe.gate.bias"),
            get_weight(batched_weights, prefix + ".moe.experts.w1"),
            get_weight(batched_weights, prefix + ".moe.experts.w2"),
            get_weight(batched_weights, prefix + ".moe.experts.b1"),
            get_weight(batched_weights, prefix + ".moe.experts.b2"),
            get_weight(batched_weights, prefix + ".norm2.weight"),
            get_weight(batched_weights, prefix + ".norm2.bias"),
            num_heads, hidden_dim, top_k,
            ws_ptr, &t);
        t1 = Clock::now();
        t[layer_key] += std::chrono::duration_cast<Microseconds>(t1 - t0);
        
        x = x_next;
        last_topk_idx = topk_indices;
        last_topk_scores = topk_scores;
    }

    // 4. Final Norm
    t0 = Clock::now();
    auto transformer_output = lb::kernels::indexed_batched_layer_norm(
        x, get_weight(batched_weights, "transformer.norm.weight"),
        get_weight(batched_weights, "transformer.norm.bias"), pol);
    t1 = Clock::now();
    t["forward_final_norm_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // 5. Heads
    t0 = Clock::now();
    auto heads_stacked = lb::model::compute_heads(
        transformer_output, batched_weights, pol, num_experts, &t);
    t1 = Clock::now();
    t["forward_heads_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // 6. Reduce Heads
    t0 = Clock::now();
    auto action_logits = lb::model::reduce_expert_heads(
        heads_stacked.at("action_heads_stacked"), last_topk_idx, last_topk_scores);
    auto opp_logits = lb::model::reduce_expert_heads(
        heads_stacked.at("opp_heads_stacked"), last_topk_idx, last_topk_scores);
    auto state_values = lb::model::reduce_expert_heads(
        heads_stacked.at("reward_heads_stacked"), last_topk_idx, last_topk_scores);
    auto win_logits = lb::model::reduce_expert_heads(
        heads_stacked.at("win_heads_stacked"), last_topk_idx, last_topk_scores);
    t1 = Clock::now();
    t["forward_reduce_heads_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

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
    
    auto pol = policy_indices.to(obs_sequence.device()).to(torch::kLong).contiguous();

    // 1. Embeddings (Autograd version)
    auto [act_kind_ids, count_ids, table_flag_ids] = lb::model::action_decomposition(
        action_sequence, batched_weights, pol, padding_mask, count_pad, tflag_pad);

    auto obs_linear = lb::autograd::indexed_batched_linear_autograd(obs_sequence,
        get_weight(batched_weights, "obs_encoder.0.weight"), get_weight(batched_weights, "obs_encoder.0.bias"), pol);
    auto obs_layernorm = lb::autograd::indexed_batched_layer_norm_autograd(obs_linear,
        get_weight(batched_weights, "obs_encoder.1.weight"), get_weight(batched_weights, "obs_encoder.1.bias"), pol, 1e-5);
    auto obs_embed = torch::gelu(obs_layernorm);

    auto act_kind_embed = lb::autograd::indexed_batched_embedding_autograd(
        get_weight(batched_weights, "act_kind_embedding.weight"), act_kind_ids, pol);
    auto count_embed = lb::autograd::indexed_batched_embedding_autograd(
        get_weight(batched_weights, "count_embedding.weight"), count_ids, pol);
    auto table_flag_embed = lb::autograd::indexed_batched_embedding_autograd(
        get_weight(batched_weights, "table_flag_embedding.weight"), table_flag_ids, pol);
    auto action_embed = act_kind_embed + count_embed + table_flag_embed;

    auto agent_embed = lb::autograd::indexed_batched_embedding_autograd(
        get_weight(batched_weights, "agent_embedding.weight"), agent_types.to(torch::kLong), pol);
    auto position_embed = lb::autograd::indexed_batched_embedding_autograd(
        get_weight(batched_weights, "position_embedding.weight"), positions.to(torch::kLong), pol);

    // 2. Gating and Fusion (Autograd version)
    auto gate_mlp = [&](const torch::Tensor& src, const std::string& base) {
        auto h = lb::autograd::indexed_batched_linear_autograd(src, get_weight(batched_weights, base + ".0.weight"), get_weight(batched_weights, base + ".0.bias"), pol);
        h = torch::tanh(h);
        auto g = lb::autograd::indexed_batched_linear_autograd(h, get_weight(batched_weights, base + ".2.weight"), get_weight(batched_weights, base + ".2.bias"), pol);
        return torch::sigmoid(g);
    };
    auto g_obs = gate_mlp(obs_embed, "gate_obs"), g_action = gate_mlp(action_embed, "gate_action");
    auto g_agent = gate_mlp(agent_embed, "gate_agent"), g_position = gate_mlp(position_embed, "gate_position");

    auto fused = g_obs * obs_embed + g_action * action_embed + g_agent * agent_embed + g_position * position_embed;
    auto x = torch::layer_norm(fused, {hidden_dim});

    // 3. Transformer Layers with Gradient Checkpointing
    torch::Tensor final_topk_indices, final_topk_scores;
    std::vector<torch::Tensor> gate_logits_list;
    for (int64_t i = 0; i < num_layers; ++i) {
        std::string prefix = "transformer.layers." + std::to_string(i);
        auto layer_outputs = TransformerLayerCheckpointFunction::apply(
            x, pol,
            get_weight(batched_weights, prefix + ".self_attn.in_proj_weight"), get_weight(batched_weights, prefix + ".self_attn.in_proj_bias"),
            get_weight(batched_weights, prefix + ".self_attn.out_proj.weight"), get_weight(batched_weights, prefix + ".self_attn.out_proj.bias"),
            get_weight(batched_weights, prefix + ".norm1.weight"), get_weight(batched_weights, prefix + ".norm1.bias"),
            get_weight(batched_weights, prefix + ".moe.gate.weight"), get_weight(batched_weights, prefix + ".moe.gate.bias"),
            get_weight(batched_weights, prefix + ".moe.experts.w1"), get_weight(batched_weights, prefix + ".moe.experts.w2"),
            get_weight(batched_weights, prefix + ".moe.experts.b1"), get_weight(batched_weights, prefix + ".moe.experts.b2"),
            get_weight(batched_weights, prefix + ".norm2.weight"), get_weight(batched_weights, prefix + ".norm2.bias"),
            num_heads, hidden_dim, top_k);
        
        x = layer_outputs[0];
        gate_logits_list.push_back(layer_outputs[1]);
        final_topk_indices = layer_outputs[2];
        final_topk_scores = layer_outputs[3];
    }

    // 4. Final Norm (Autograd)
    auto transformer_output = lb::autograd::indexed_batched_layer_norm_autograd(
        x, get_weight(batched_weights, "transformer.norm.weight"),
        get_weight(batched_weights, "transformer.norm.bias"), pol, 1e-5);

    // 5. Heads (Autograd)
    std::vector<torch::Tensor> action_logits_list, opp_logits_list, state_values_list, win_logits_list;
    for (int64_t i = 0; i < num_experts; ++i) {
        auto idx = std::to_string(i);
        action_logits_list.push_back(lb::autograd::indexed_batched_linear_autograd(transformer_output,
            get_weight(batched_weights, "action_heads." + idx + ".weight"), get_weight(batched_weights, "action_heads." + idx + ".bias"), pol));
        opp_logits_list.push_back(lb::autograd::indexed_batched_linear_autograd(transformer_output,
            get_weight(batched_weights, "opp_action_heads." + idx + ".weight"), get_weight(batched_weights, "opp_action_heads." + idx + ".bias"), pol));
        state_values_list.push_back(lb::autograd::indexed_batched_linear_autograd(transformer_output,
            get_weight(batched_weights, "reward_stream_heads." + idx + ".weight"), get_weight(batched_weights, "reward_stream_heads." + idx + ".bias"), pol));
        win_logits_list.push_back(lb::autograd::indexed_batched_linear_autograd(transformer_output,
            get_weight(batched_weights, "win_prob_heads." + idx + ".weight"), get_weight(batched_weights, "win_prob_heads." + idx + ".bias"), pol));
    }

    auto action_stacked = torch::stack(action_logits_list, 2);
    auto opp_stacked = torch::stack(opp_logits_list, 2);
    auto reward_stacked = torch::stack(state_values_list, 2);
    auto win_stacked = torch::stack(win_logits_list, 2);

    // 6. Reduce Heads
    auto action_logits = lb::model::reduce_expert_heads(action_stacked, final_topk_indices, final_topk_scores);
    auto opp_logits = lb::model::reduce_expert_heads(opp_stacked, final_topk_indices, final_topk_scores);
    auto state_values = lb::model::reduce_expert_heads(reward_stacked, final_topk_indices, final_topk_scores);
    auto win_logits = lb::model::reduce_expert_heads(win_stacked, final_topk_indices, final_topk_scores);

    // 7. Package routing info
    torch::Tensor gate_logits_tensor;
    if (!gate_logits_list.empty()) {
        gate_logits_tensor = torch::stack(gate_logits_list, 0);
    } else {
        gate_logits_tensor = x.new_zeros({0, x.size(0), x.size(1), num_experts});
    }
    std::unordered_map<std::string, torch::Tensor> routing;
    routing["gate_logits"]  = gate_logits_tensor;
    routing["topk_indices"] = final_topk_indices;
    routing["topk_scores"]  = final_topk_scores;

    return std::make_tuple(action_logits, opp_logits, state_values, win_logits, gate_logits_tensor, routing);
}

} // namespace forward
} // namespace lb
