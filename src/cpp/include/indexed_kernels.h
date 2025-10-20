#pragma once

#include <torch/extension.h>
#include <unordered_map>
#include <chrono>

// Fused linear epilogues supported by the indexed batched linear helper.
enum class IndexedLinearEpilogue {
    Bias,
    BiasGELU,
};

// Launches a CUDA kernel that performs an indexed batched embedding lookup.
// weight_cache: [W, vocab, hidden]
// indices: [B, T]
// policy_indices: [B]
// Returns: [B, T, hidden]
torch::Tensor indexed_batched_embedding(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices);

// Launches a CUDA kernel that performs an indexed batched layer norm.
// input: [B, T, H]
// gamma_cache / beta_cache: [W, H]
// policy_indices: [B]
// Returns: [B, T, H]
torch::Tensor indexed_batched_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps = 1e-5);

// Runs a cuBLASLt powered indexed batched linear.
// input: [B, T, in_dim]
// weight_cache: [W, out_dim, in_dim]
// bias_cache: [W, out_dim]
// policy_indices: [B]
// Returns: [B, T, out_dim]
torch::Tensor indexed_batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, std::chrono::microseconds>& timers,
    IndexedLinearEpilogue epilogue = IndexedLinearEpilogue::Bias);

// Runs the grouped FFN GEMM forward pass using CUTLASS grouped kernels.
// Each array argument is expected to contain device pointers for the operands.
void grouped_ffn_gemm_forward(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* b2_ptrs,
    const uintptr_t* output_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim);

