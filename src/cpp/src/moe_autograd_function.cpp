#include "moe_autograd_function.h"
#include "moe_cutlass_kernels.h"
#include "moe_backward_kernels.h"

#include <vector>

namespace lb {
namespace moe {

struct GroupedMoEAutograd : public torch::autograd::Function<GroupedMoEAutograd> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input_grouped,
        const torch::Tensor& w1_weights,
        const torch::Tensor& w2_weights,
        const torch::Tensor& b1_biases,
        const torch::Tensor& b2_biases,
        const torch::Tensor& routing_weights_grouped,
        const torch::Tensor& indices_grouped,
        const torch::Tensor& m_sizes_cpu,
        const torch::Tensor& policy_ids_cpu,
        const torch::Tensor& expert_ids_cpu,
        const torch::Tensor& token_offsets_cpu,
        int64_t hidden_dim,
        int64_t ffn_dim
    ) {
        TORCH_CHECK(input_grouped.scalar_type() == torch::kFloat16 && input_grouped.is_cuda());
        TORCH_CHECK(w1_weights.scalar_type() == torch::kFloat16 && w1_weights.is_cuda());
        TORCH_CHECK(w2_weights.scalar_type() == torch::kFloat16 && w2_weights.is_cuda());
        TORCH_CHECK(b1_biases.scalar_type() == torch::kFloat16 && b1_biases.is_cuda());
        TORCH_CHECK(b2_biases.scalar_type() == torch::kFloat16 && b2_biases.is_cuda());
        TORCH_CHECK(routing_weights_grouped.scalar_type() == torch::kFloat32 && routing_weights_grouped.is_cuda());
        TORCH_CHECK(indices_grouped.scalar_type() == torch::kLong && indices_grouped.is_cuda());
        TORCH_CHECK(m_sizes_cpu.device().is_cpu() && policy_ids_cpu.device().is_cpu());
        TORCH_CHECK(expert_ids_cpu.device().is_cpu() && token_offsets_cpu.device().is_cpu());

        auto m_sizes = m_sizes_cpu.to(torch::kCPU).contiguous();
        auto policy_ids = policy_ids_cpu.to(torch::kCPU).contiguous();
        auto expert_ids = expert_ids_cpu.to(torch::kCPU).contiguous();
        auto token_offsets = token_offsets_cpu.to(torch::kCPU).contiguous();

        const int64_t total_tokens = input_grouped.size(0);
        const int64_t group_count = m_sizes.size(0);
        auto opts_half = input_grouped.options().dtype(torch::kFloat16);
        torch::Tensor hidden_grouped = torch::empty({total_tokens, ffn_dim}, opts_half);
        torch::Tensor output_grouped = torch::empty({total_tokens, hidden_dim}, opts_half);

        // Build pointer arrays for CUTLASS forward
        std::vector<uintptr_t> input_ptrs(group_count);
        std::vector<uintptr_t> hidden_ptrs(group_count);
        std::vector<uintptr_t> output_ptrs(group_count);
        std::vector<uintptr_t> routing_ptrs(group_count);
        std::vector<uintptr_t> w1_ptrs(group_count), w2_ptrs(group_count), b1_ptrs(group_count), b2_ptrs(group_count);

        const uintptr_t input_base = reinterpret_cast<uintptr_t>(input_grouped.data_ptr<at::Half>());
        const uintptr_t hidden_base = reinterpret_cast<uintptr_t>(hidden_grouped.data_ptr<at::Half>());
        const uintptr_t output_base = reinterpret_cast<uintptr_t>(output_grouped.data_ptr<at::Half>());
        const uintptr_t routing_base = reinterpret_cast<uintptr_t>(routing_weights_grouped.data_ptr<float>());

        auto m_ptr = m_sizes.data_ptr<int64_t>();
        auto pi_ptr = policy_ids.data_ptr<int64_t>();
        auto ei_ptr = expert_ids.data_ptr<int64_t>();
        auto off_ptr = token_offsets.data_ptr<int64_t>();

        for (int64_t g = 0; g < group_count; ++g) {
            int64_t M = m_ptr[g];
            int64_t offset = off_ptr[g];
            int64_t pi = pi_ptr[g];
            int64_t ei = ei_ptr[g];
            input_ptrs[g]  = input_base  + static_cast<uintptr_t>(offset) * hidden_dim * sizeof(at::Half);
            hidden_ptrs[g] = hidden_base + static_cast<uintptr_t>(offset) * ffn_dim   * sizeof(at::Half);
            output_ptrs[g] = output_base + static_cast<uintptr_t>(offset) * hidden_dim * sizeof(at::Half);
            routing_ptrs[g]= routing_base+ static_cast<uintptr_t>(offset) * sizeof(float);
            w1_ptrs[g] = reinterpret_cast<uintptr_t>(w1_weights.index({pi, ei}).data_ptr<at::Half>());
            w2_ptrs[g] = reinterpret_cast<uintptr_t>(w2_weights.index({pi, ei}).data_ptr<at::Half>());
            b1_ptrs[g] = reinterpret_cast<uintptr_t>(b1_biases.index({pi, ei}).data_ptr<at::Half>());
            b2_ptrs[g] = reinterpret_cast<uintptr_t>(b2_biases.index({pi, ei}).data_ptr<at::Half>());
        }

        // Fused CUTLASS forward that also writes hidden_grouped
        cutlass_grouped_moe_forward_with_hidden(
            input_ptrs.data(),
            w1_ptrs.data(),
            b1_ptrs.data(),
            w2_ptrs.data(),
            b2_ptrs.data(),
            hidden_ptrs.data(),
            output_ptrs.data(),
            routing_ptrs.data(),
            m_ptr,
            pi_ptr,
            ei_ptr,
            off_ptr,
            group_count,
            hidden_dim,
            ffn_dim
        );

        // Save for backward
        ctx->save_for_backward({
            input_grouped,
            hidden_grouped,
            routing_weights_grouped,
            indices_grouped,
            w1_weights,
            w2_weights,
            b1_biases,
            b2_biases,
        });
        // Save CPU metadata
        ctx->saved_data["m_sizes"] = m_sizes;
        ctx->saved_data["policy_ids"] = policy_ids;
        ctx->saved_data["expert_ids"] = expert_ids;
        ctx->saved_data["token_offsets"] = token_offsets;
        ctx->saved_data["hidden_dim"] = hidden_dim;
        ctx->saved_data["ffn_dim"] = ffn_dim;

        return output_grouped;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        size_t idx = 0;
        auto input_grouped = saved[idx++];
        auto hidden_grouped = saved[idx++];
        auto routing_weights_grouped = saved[idx++];
        auto indices_grouped = saved[idx++];
        auto w1_weights = saved[idx++];
        auto w2_weights = saved[idx++];
        auto b1_biases = saved[idx++];
        auto b2_biases = saved[idx++];

        auto m_sizes = ctx->saved_data["m_sizes"].toTensor();
        auto policy_ids = ctx->saved_data["policy_ids"].toTensor();
        auto expert_ids = ctx->saved_data["expert_ids"].toTensor();
        auto token_offsets = ctx->saved_data["token_offsets"].toTensor();
        int64_t hidden_dim = ctx->saved_data["hidden_dim"].toInt();
        int64_t ffn_dim = ctx->saved_data["ffn_dim"].toInt();

        auto grad_out = grad_outputs[0];
        auto total_tokens = input_grouped.size(0);
        // Cast upstream grad to Half as kernels expect
        auto grad_out_half = grad_out.to(torch::kFloat16).contiguous();

        auto grad_input = torch::zeros_like(input_grouped);
        auto grad_w1 = torch::zeros_like(w1_weights, torch::dtype(torch::kFloat32));
        auto grad_w2 = torch::zeros_like(w2_weights, torch::dtype(torch::kFloat32));
        auto grad_b1 = torch::zeros_like(b1_biases, torch::dtype(torch::kFloat32));
        auto grad_b2 = torch::zeros_like(b2_biases, torch::dtype(torch::kFloat32));

        // Convert CPU meta to vectors
        std::vector<int64_t> m_v(m_sizes.data_ptr<int64_t>(), m_sizes.data_ptr<int64_t>() + m_sizes.size(0));
        std::vector<int64_t> pi_v(policy_ids.data_ptr<int64_t>(), policy_ids.data_ptr<int64_t>() + policy_ids.size(0));
        std::vector<int64_t> ei_v(expert_ids.data_ptr<int64_t>(), expert_ids.data_ptr<int64_t>() + expert_ids.size(0));
        std::vector<int64_t> off_v(token_offsets.data_ptr<int64_t>(), token_offsets.data_ptr<int64_t>() + token_offsets.size(0));

        // Call main backward orchestrator
        cutlass_grouped_moe_backward(
            grad_out_half,
            input_grouped,
            hidden_grouped,
            routing_weights_grouped,
            indices_grouped,
            w1_weights,
            w2_weights,
            b1_biases,
            b2_biases,
            m_v,
            pi_v,
            ei_v,
            off_v,
            grad_input,
            grad_w1,
            grad_w2,
            grad_b1,
            grad_b2
        );

        // Compute dr = rowwise_dot(Gy, Y_pre) to propagate into routing weights
        auto grad_out_f32 = grad_out.to(torch::kFloat32).contiguous();
        auto dr = torch::empty({total_tokens}, grad_out_f32.options());
        int64_t cursor = 0;
        for (int64_t g = 0; g < m_sizes.size(0); ++g) {
            int64_t M = m_sizes.data_ptr<int64_t>()[g];
            if (M == 0) continue;
            int64_t pi = policy_ids.data_ptr<int64_t>()[g];
            int64_t ei = expert_ids.data_ptr<int64_t>()[g];
            auto H_g = hidden_grouped.narrow(0, cursor, M).to(torch::kFloat32);
            auto W2 = w2_weights.index({pi, ei}).to(torch::kFloat32); // [H, F]
            auto b2 = b2_biases.index({pi, ei}).to(torch::kFloat32);  // [H]
            auto Y_pre = at::addmm(b2, H_g, W2.t());          // [M, H]
            auto Gy = grad_out_f32.narrow(0, cursor, M);
            auto dot = (Gy * Y_pre).sum(/*dim=*/1);
            dr.narrow(0, cursor, M).copy_(dot);
            cursor += M;
        }

        // Return gradients in the order of forward inputs
        return {
            grad_input,  // input_grouped
            grad_w1.to(w1_weights.dtype()),  // w1_weights
            grad_w2.to(w2_weights.dtype()),  // w2_weights
            grad_b1.to(b1_biases.dtype()),   // b1_biases
            grad_b2.to(b2_biases.dtype()),   // b2_biases
            dr.to(routing_weights_grouped.dtype()),  // routing_weights_grouped
            torch::Tensor(), // indices_grouped
            torch::Tensor(), // m_sizes_cpu
            torch::Tensor(), // policy_ids_cpu
            torch::Tensor(), // expert_ids_cpu
            torch::Tensor(), // token_offsets_cpu
            torch::Tensor(), // hidden_dim
            torch::Tensor()  // ffn_dim
        };
    }
};

torch::Tensor grouped_moe_autograd_forward(
    const torch::Tensor& input_grouped,
    const torch::Tensor& w1_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& b1_biases,
    const torch::Tensor& b2_biases,
    const torch::Tensor& routing_weights_grouped,
    const torch::Tensor& indices_grouped,
    const torch::Tensor& m_sizes_cpu,
    const torch::Tensor& policy_ids_cpu,
    const torch::Tensor& expert_ids_cpu,
    const torch::Tensor& token_offsets_cpu,
    int64_t hidden_dim,
    int64_t ffn_dim
) {
    return GroupedMoEAutograd::apply(
        input_grouped,
        w1_weights,
        w2_weights,
        b1_biases,
        b2_biases,
        routing_weights_grouped,
        indices_grouped,
        m_sizes_cpu,
        policy_ids_cpu,
        expert_ids_cpu,
        token_offsets_cpu,
        hidden_dim,
        ffn_dim
    );
}

} // namespace moe
} // namespace lb
