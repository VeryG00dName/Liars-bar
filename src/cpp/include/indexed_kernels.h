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

