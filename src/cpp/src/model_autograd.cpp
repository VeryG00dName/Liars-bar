/**
 * model_autograd.cpp - Autograd function implementations
 *
 * This file implements torch::autograd::Function wrappers that enable training.
 * All forward passes delegate to the unified kernels in lb_kernels to ensure
 * numerical parity with inference.
 */

#include "model_autograd.h"
#include "lb_kernels.h"

namespace lb {
namespace autograd {

// ============================================================================
// IndexedLinearFunction - Autograd wrapper for batched linear transformation
// ============================================================================

struct IndexedLinearFunction : public torch::autograd::Function<IndexedLinearFunction> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,           // [B, T, in_dim]
        const torch::Tensor& weight_cache,    // [W, out_dim, in_dim]
        const torch::Tensor& bias_cache,      // [W, out_dim]
        const torch::Tensor& policy_indices   // [B]
    ) {
        // Save for backward
        ctx->save_for_backward({input, weight_cache, bias_cache, policy_indices});

        // Call the unified kernel (with dummy timers since we don't use them in training)
        std::unordered_map<std::string, std::chrono::microseconds> dummy_timers;
        return lb::kernels::indexed_batched_linear(
            input, weight_cache, bias_cache, policy_indices,
            dummy_timers, lb::kernels::IndexedLinearEpilogue::Bias);
    }

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<torch::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];           // [B, T, in_dim]
        auto weight_cache = saved[1];    // [W, out_dim, in_dim]
        auto bias_cache = saved[2];      // [W, out_dim]
        auto policy_indices = saved[3];  // [B]

        auto grad_output = grad_outputs[0];  // [B, T, out_dim]

        const int64_t B = input.size(0);
        const int64_t T = input.size(1);
        const int64_t in_dim = input.size(2);
        const int64_t out_dim = grad_output.size(2);

        // Convert policy indices to long
        auto policy_long = policy_indices.to(torch::kLong);

        // Select weights for each batch element
        auto weight_selected = weight_cache.index_select(0, policy_long);  // [B, out_dim, in_dim]

        // Compute grad_input: [B, T, in_dim]
        // grad_input = grad_output @ weight_selected
        // [B, T, out_dim] x [B, out_dim, in_dim] -> [B, T, in_dim]
        auto grad_input = torch::bmm(grad_output, weight_selected);

        // Compute grad_weight: [W, out_dim, in_dim]
        // Need to accumulate gradients for each policy
        auto grad_weight_cache = torch::zeros_like(weight_cache);

        // For each batch element: grad_weight[policy_idx] += grad_output[b]^T @ input[b]
        // grad_output[b]: [T, out_dim] -> transpose to [out_dim, T]
        // input[b]: [T, in_dim]
        // Result: [out_dim, in_dim]
        auto grad_output_bt = grad_output.transpose(1, 2);  // [B, out_dim, T]
        auto per_batch_grad_weight = torch::bmm(grad_output_bt, input);  // [B, out_dim, in_dim]

        // Accumulate into cache using index_add_
        grad_weight_cache.index_add_(0, policy_long, per_batch_grad_weight);

        // Compute grad_bias: [W, out_dim]
        auto grad_bias_cache = torch::zeros_like(bias_cache);

        // For each batch element: grad_bias[policy_idx] += sum over T of grad_output[b, :, :]
        auto per_batch_grad_bias = grad_output.sum(1);  // [B, out_dim]
        grad_bias_cache.index_add_(0, policy_long, per_batch_grad_bias);

        // Return gradients in order of forward parameters
        return {grad_input, grad_weight_cache, grad_bias_cache, torch::Tensor()};
    }
};

// ============================================================================
// LayerNormFunction - Autograd wrapper for batched layer normalization
// ============================================================================

struct LayerNormFunction : public torch::autograd::Function<LayerNormFunction> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,
        const torch::Tensor& gamma_cache,
        const torch::Tensor& beta_cache,
        const torch::Tensor& policy_indices,
        double eps
    ) {
        // Forward: call the unified kernel for numerical parity
        auto out = lb::kernels::indexed_batched_layer_norm(
            input, gamma_cache, beta_cache, policy_indices, eps);

        // Save for backward
        ctx->save_for_backward({input, gamma_cache, beta_cache, policy_indices});
        ctx->saved_data["eps"] = eps;

        return out;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];           // [B, T, H]
        auto gamma_cache = saved[1];     // [W, H]
        auto beta_cache  = saved[2];     // [W, H]
        auto policy_indices = saved[3];  // [B]
        double eps = ctx->saved_data["eps"].toDouble();

        auto grad_out = grad_outputs[0]; // [B, T, H]

        // Compute LN stats and grads in FP32 for numerical stability
        const auto B = input.size(0);
        const auto T = input.size(1);
        const auto H = input.size(2);

        auto input_f32 = input.to(torch::kFloat32);
        auto grad_out_f32 = grad_out.to(torch::kFloat32);

        // Compute per-token mean/var over H
        auto mean = input_f32.mean(-1, /*keepdim=*/true);                       // [B, T, 1]
        auto xc = input_f32 - mean;                                             // [B, T, H]
        auto var = (xc * xc).mean(-1, /*keepdim=*/true);                        // [B, T, 1]
        auto rstd = (var + eps).rsqrt();                                        // [B, T, 1]
        auto y = xc * rstd;                                                     // [B, T, H]

        // Gather gamma per batch and expand across T
        auto pol = policy_indices.to(gamma_cache.device()).to(torch::kLong);
        if (gamma_cache.device() != input.device()) {
            gamma_cache = gamma_cache.to(input.device());
            beta_cache  = beta_cache.to(input.device());
        }
        auto gamma_b = gamma_cache.index_select(0, pol).to(torch::kFloat32);    // [B, H]
        auto gamma_bt = gamma_b.unsqueeze(1).expand({B, T, H});                 // [B, T, H]

        // Compute gradients using standard LayerNorm backward formula
        auto sum_dout   = grad_out_f32.sum(-1, true);           // [B, T, 1]
        auto sum_dout_y = (grad_out_f32 * y).sum(-1, true);     // [B, T, 1]
        auto invH = 1.0f / static_cast<float>(H);

        // dx = (gamma * rstd) * (dout - mean(dout) - y * mean(dout*y))
        auto dx = (gamma_bt * rstd) * (grad_out_f32 - sum_dout * invH - y * (sum_dout_y * invH));
        auto grad_input = dx.to(input.scalar_type());

        // dgamma = sum(dout * y) over B, T
        // dbeta  = sum(dout) over B, T
        auto dgamma_bt = (grad_out_f32 * y).sum(1);    // [B, H]
        auto dbeta_bt  = grad_out_f32.sum(1);          // [B, H]

        // Accumulate into caches per policy row
        auto grad_gamma_cache = torch::zeros_like(gamma_cache);
        auto grad_beta_cache  = torch::zeros_like(beta_cache);
        grad_gamma_cache.index_add_(0, pol, dgamma_bt.to(gamma_cache.scalar_type()));
        grad_beta_cache.index_add_(0, pol,  dbeta_bt.to(beta_cache.scalar_type()));

        return {
            grad_input,           // input
            grad_gamma_cache,     // gamma_cache
            grad_beta_cache,      // beta_cache
            torch::Tensor(),      // policy_indices
            torch::Tensor(),      // eps
        };
    }
};

// ============================================================================
// EmbeddingFunction - Autograd wrapper for batched embedding lookup
// ============================================================================

struct EmbeddingFunction : public torch::autograd::Function<EmbeddingFunction> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& weight_cache,   // [W, vocab, H]
        const torch::Tensor& indices,        // [B, T]
        const torch::Tensor& policy_indices  // [B]
    ) {
        auto out = lb::kernels::indexed_batched_embedding(weight_cache, indices, policy_indices);
        ctx->save_for_backward({weight_cache, indices, policy_indices});
        return out;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto weight_cache = saved[0];              // [W, vocab, H]
        auto indices = saved[1].to(torch::kLong);  // [B, T]
        auto policy_indices = saved[2].to(torch::kLong); // [B]
        auto grad_out = grad_outputs[0];           // [B, T, H]

        auto grad_weight = torch::zeros_like(weight_cache);
        auto B = indices.size(0);

        // For each batch element, accumulate gradients into the appropriate policy's embedding table
        for (int64_t b = 0; b < B; ++b) {
            auto p = policy_indices[b].item<int64_t>();
            auto idx_b = indices[b];               // [T]
            auto grad_b = grad_out[b];             // [T, H]
            grad_weight[p].index_add_(0, idx_b, grad_b);
        }

        return {
            grad_weight,          // weight_cache
            torch::Tensor(),      // indices
            torch::Tensor()       // policy_indices
        };
    }
};

// ============================================================================
// Public API - Wrapper functions
// ============================================================================

torch::Tensor indexed_batched_embedding_autograd(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices) {
    return EmbeddingFunction::apply(weight_cache, indices, policy_indices);
}

torch::Tensor indexed_batched_layer_norm_autograd(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps) {
    return LayerNormFunction::apply(
        input, gamma_cache, beta_cache, policy_indices, eps);
}

torch::Tensor indexed_batched_linear_autograd(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices) {
    return IndexedLinearFunction::apply(input, weight_cache, bias_cache, policy_indices);
}

} // namespace autograd
} // namespace lb
