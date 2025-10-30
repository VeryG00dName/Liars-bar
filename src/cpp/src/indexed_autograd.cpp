// Thin forwarders to the unified autograd bridges in model_autograd.

#include "indexed_autograd.h"
#include "model_autograd.h"

torch::Tensor indexed_batched_layer_norm_autograd(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps) {
    return lb::autograd::indexed_batched_layer_norm_autograd(
        input, gamma_cache, beta_cache, policy_indices, eps);
}

torch::Tensor indexed_batched_embedding_autograd(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices) {
    return lb::autograd::indexed_batched_embedding_autograd(
        weight_cache, indices, policy_indices);
}
