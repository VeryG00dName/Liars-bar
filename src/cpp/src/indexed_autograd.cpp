#include "indexed_autograd.h"
#include "indexed_kernels.h"

namespace lb_indexed_autograd {

struct LayerNormFunction : public torch::autograd::Function<LayerNormFunction> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,
        const torch::Tensor& gamma_cache,
        const torch::Tensor& beta_cache,
        const torch::Tensor& policy_indices,
        double eps
    ) {
        // Forward: call the existing fast kernel to keep rollout parity
        auto out = indexed_batched_layer_norm(
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
        auto input = saved[0];           // [B,T,H]
        auto gamma_cache = saved[1];     // [W,H]
        auto beta_cache  = saved[2];     // [W,H]
        auto policy_indices = saved[3];  // [B]
        double eps = ctx->saved_data["eps"].toDouble();

        auto grad_out = grad_outputs[0]; // [B,T,H]

        // Compute LN stats and grads in FP32 for stability
        const auto B = input.size(0);
        const auto T = input.size(1);
        const auto H = input.size(2);

        auto input_f32 = input.to(torch::kFloat32);
        auto grad_out_f32 = grad_out.to(torch::kFloat32);

        // Compute per-token mean/var over H
        auto mean = input_f32.mean(-1, /*keepdim=*/true);                       // [B,T,1]
        auto xc = input_f32 - mean;                                             // [B,T,H]
        auto var = (xc * xc).mean(-1, /*keepdim=*/true);                         // [B,T,1]
        auto rstd = (var + eps).rsqrt();                                        // [B,T,1]
        auto y = xc * rstd;                                                     // [B,T,H]

        // Gather gamma per batch and expand across T
        auto pol = policy_indices.to(gamma_cache.device()).to(torch::kLong);
        if (gamma_cache.device() != input.device()) {
            gamma_cache = gamma_cache.to(input.device());
            beta_cache  = beta_cache.to(input.device());
        }
        auto gamma_b = gamma_cache.index_select(0, pol).to(torch::kFloat32);    // [B,H]
        auto beta_b  = beta_cache.index_select(0,  pol).to(torch::kFloat32);    // [B,H]
        (void)beta_b; // not needed for backward formula directly, kept for parity
        auto gamma_bt = gamma_b.unsqueeze(1).expand({B, T, H});                 // [B,T,H]

        // Sums over H
        auto sum_dout   = grad_out_f32.sum(-1, true);           // [B,T,1]
        auto sum_dout_y = (grad_out_f32 * y).sum(-1, true);     // [B,T,1]
        auto invH = 1.0f / static_cast<float>(H);

        // dx = (gamma * rstd) * (dout - mean(dout) - y * mean(dout*y))
        auto dx = (gamma_bt * rstd) * (grad_out_f32 - sum_dout * invH - y * (sum_dout_y * invH));
        auto grad_input = dx.to(input.scalar_type());

        // dgamma = sum(dout * y) over B,T
        // dbeta  = sum(dout) over B,T
        auto dgamma_bt = (grad_out_f32 * y).sum(1);    // [B,H]
        auto dbeta_bt  = grad_out_f32.sum(1);          // [B,H]

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

struct EmbeddingFunction : public torch::autograd::Function<EmbeddingFunction> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& weight_cache,   // [W, vocab, H]
        const torch::Tensor& indices,        // [B, T]
        const torch::Tensor& policy_indices  // [B]
    ) {
        auto out = indexed_batched_embedding(weight_cache, indices, policy_indices);
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

} // namespace lb_indexed_autograd

torch::Tensor indexed_batched_layer_norm_autograd(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps) {
    return lb_indexed_autograd::LayerNormFunction::apply(
        input, gamma_cache, beta_cache, policy_indices, eps);
}

torch::Tensor indexed_batched_embedding_autograd(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices) {
    return lb_indexed_autograd::EmbeddingFunction::apply(
        weight_cache, indices, policy_indices);
}

