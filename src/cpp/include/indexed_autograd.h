#pragma once

#include <torch/extension.h>

// Autograd-enabled wrappers for indexed batched ops used in training forward.

// LayerNorm: input [B,T,H], gamma_cache [W,H], beta_cache [W,H], policy_indices [B]
// Returns output [B,T,H]. Backward computes grads for input, gamma_cache, beta_cache.
torch::Tensor indexed_batched_layer_norm_autograd(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps = 1e-5);

// Embedding: weight_cache [W, vocab, H], indices [B,T], policy_indices [B]
// Returns output [B,T,H]. Backward computes grad for weight_cache.
torch::Tensor indexed_batched_embedding_autograd(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices);

