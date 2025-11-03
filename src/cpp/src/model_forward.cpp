#include "model_forward.h"
#include "model_layer.h"
#include "moe_cutlass_kernels.h"  // For lb::moe::MoEWorkspace
#include <cuda_runtime.h>
#include "lb_kernels.h"

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

} // namespace forward
} // namespace lb
