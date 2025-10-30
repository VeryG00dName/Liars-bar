// Compatibility shims for legacy global APIs.
//
// This file forwards the old free functions declared in indexed_kernels.h
// to the unified implementations in lb_kernels (pure compute) and
// model_autograd (autograd bridges). Keep these names to avoid touching
// existing call sites while ensuring there is a single source of truth.

#include "indexed_kernels.h"
#include "lb_kernels.h"
#include "model_autograd.h"

using TimerMap = std::unordered_map<std::string, std::chrono::microseconds>;

// ----------------------------------------------------------------------------
// Legacy global API -> lb::kernels
// ----------------------------------------------------------------------------

torch::Tensor indexed_batched_embedding(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices) {
    return lb::kernels::indexed_batched_embedding(weight_cache, indices, policy_indices);
}

torch::Tensor indexed_batched_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps) {
    return lb::kernels::indexed_batched_layer_norm(input, gamma_cache, beta_cache, policy_indices, eps);
}

torch::Tensor indexed_batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices,
    TimerMap& timers,
    IndexedLinearEpilogue epilogue) {
    // Map epilogue enum to lb::kernels equivalent
    lb::kernels::IndexedLinearEpilogue epi = lb::kernels::IndexedLinearEpilogue::Bias;
    if (epilogue == IndexedLinearEpilogue::BiasGELU) {
        epi = lb::kernels::IndexedLinearEpilogue::BiasGELU;
    }
    return lb::kernels::indexed_batched_linear(
        input, weight_cache, bias_cache, policy_indices, timers, epi);
}

void grouped_ffn_gemm_forward(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* b2_ptrs,
    const uintptr_t* output_ptrs,
    const uintptr_t* routing_weight_ptrs,
    const int64_t* m_sizes,
    const int64_t* policy_ids,
    const int64_t* expert_ids,
    const int64_t* token_offsets,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim) {
    lb::kernels::grouped_ffn_gemm_forward(
        input_ptrs, w1_ptrs, b1_ptrs, w2_ptrs, b2_ptrs, output_ptrs,
        routing_weight_ptrs, m_sizes, policy_ids, expert_ids, token_offsets,
        group_count, hidden_dim, ffn_dim);
}

// ----------------------------------------------------------------------------
// Legacy autograd helpers -> lb::autograd
// ----------------------------------------------------------------------------

torch::Tensor indexed_batched_linear_autograd(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices) {
    return lb::autograd::indexed_batched_linear_autograd(
        input, weight_cache, bias_cache, policy_indices);
}

