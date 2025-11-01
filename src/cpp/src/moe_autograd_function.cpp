#include "moe_autograd_function.h"
#include "moe_cutlass_kernels.h"
#include "moe_backward_kernels.h"

#include <vector>
#include <cstdlib>

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
        const torch::Tensor& w1_ptrs_table,
        const torch::Tensor& w2_ptrs_table,
        const torch::Tensor& b1_ptrs_table,
        const torch::Tensor& b2_ptrs_table,
        const torch::Tensor& routing_weights_grouped,
        const torch::Tensor& indices_grouped,
        const torch::Tensor& m_sizes_cpu,
        const torch::Tensor& policy_ids_cpu,
        const torch::Tensor& expert_ids_cpu,
        const torch::Tensor& token_offsets_cpu,
        int64_t hidden_dim,
        int64_t ffn_dim,
        lb::moe::MoEWorkspace* workspace
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

        // Device-only path: build metadata on device, no host descriptor staging

        // Build device-side group metadata
        auto device = input_grouped.device();
        auto m_sizes_dev = m_sizes.to(device).contiguous();
        auto policy_ids_dev = policy_ids.to(device).contiguous();
        auto expert_ids_dev = expert_ids.to(device).contiguous();
        auto token_offsets_dev = token_offsets.to(device).contiguous();

        // Base pointers
        auto input_base = reinterpret_cast<uintptr_t>(input_grouped.data_ptr<at::Half>());
        auto hidden_base = reinterpret_cast<uintptr_t>(hidden_grouped.data_ptr<at::Half>());
        auto output_base = reinterpret_cast<uintptr_t>(output_grouped.data_ptr<at::Half>());
        auto routing_base = reinterpret_cast<uintptr_t>(routing_weights_grouped.data_ptr<float>());

        // Pointer tables: use provided CUDA tables if defined; otherwise build from stacked weights
        auto build_ptr_table = [&](const torch::Tensor& stacked) -> torch::Tensor {
            TORCH_CHECK(stacked.is_cuda(), "Stacked expert weights must be CUDA tensors");
            TORCH_CHECK(stacked.dim() >= 2, "Stacked expert weights must be at least 2D");
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

        torch::Tensor w1_tbl = (w1_ptrs_table.defined() && w1_ptrs_table.numel() > 0)
                                    ? w1_ptrs_table
                                    : build_ptr_table(w1_weights);
        torch::Tensor w2_tbl = (w2_ptrs_table.defined() && w2_ptrs_table.numel() > 0)
                                    ? w2_ptrs_table
                                    : build_ptr_table(w2_weights);
        torch::Tensor b1_tbl = (b1_ptrs_table.defined() && b1_ptrs_table.numel() > 0)
                                    ? b1_ptrs_table
                                    : build_ptr_table(b1_biases);
        torch::Tensor b2_tbl = (b2_ptrs_table.defined() && b2_ptrs_table.numel() > 0)
                                    ? b2_ptrs_table
                                    : build_ptr_table(b2_biases);

        TORCH_CHECK(w1_tbl.is_cuda() && w2_tbl.is_cuda() && b1_tbl.is_cuda() && b2_tbl.is_cuda(),
                    "Pointer tables must be CUDA tensors");
        TORCH_CHECK(w1_tbl.scalar_type() == torch::kUInt64 && w2_tbl.scalar_type() == torch::kUInt64,
                    "Pointer tables must be UInt64");
        TORCH_CHECK(b1_tbl.scalar_type() == torch::kUInt64 && b2_tbl.scalar_type() == torch::kUInt64,
                    "Pointer tables must be UInt64");

        int64_t num_policies = w1_tbl.size(0);
        int64_t num_experts_tbl = w1_tbl.size(1);

        cutlass_grouped_moe_forward_with_hidden_device(
            input_base, hidden_base, output_base, routing_base,
            w1_tbl.data_ptr<uint64_t>(),
            w2_tbl.data_ptr<uint64_t>(),
            b1_tbl.data_ptr<uint64_t>(),
            b2_tbl.data_ptr<uint64_t>(),
            num_policies, num_experts_tbl,
            m_sizes_dev.data_ptr<int64_t>(),
            policy_ids_dev.data_ptr<int64_t>(),
            expert_ids_dev.data_ptr<int64_t>(),
            token_offsets_dev.data_ptr<int64_t>(),
            group_count,
            hidden_dim,
            ffn_dim,
            workspace
        );

        // Optional runtime guard: assert outputs are finite to catch instability early
        static bool assert_finite = std::getenv("LB_MOE_ASSERT_FINITE") != nullptr;
        if (assert_finite) {
            TORCH_CHECK(torch::isfinite(hidden_grouped).all().item<bool>(), "MoE forward hidden_grouped has non-finite values");
            TORCH_CHECK(torch::isfinite(output_grouped).all().item<bool>(), "MoE forward output_grouped has non-finite values");
        }

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
        const bool assert_finite = std::getenv("LB_MOE_ASSERT_FINITE") != nullptr;
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
        // Guard against FP16 overflow when converting upstream gradients
        // Clamp to representable FP16 range before cast to avoid inf->nan in kernels
        auto grad_out_f32_cast = grad_out.to(torch::kFloat32);
        auto grad_out_clamped = grad_out_f32_cast.clamp(-65504.0f, 65504.0f);
        auto grad_out_half = grad_out_clamped.to(torch::kFloat16).contiguous();

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

        if (assert_finite) {
            TORCH_CHECK(torch::isfinite(grad_input).all().item<bool>(), "MoE backward grad_input has non-finite values");
            TORCH_CHECK(torch::isfinite(grad_w1).all().item<bool>(), "MoE backward grad_w1 has non-finite values");
            TORCH_CHECK(torch::isfinite(grad_w2).all().item<bool>(), "MoE backward grad_w2 has non-finite values");
            TORCH_CHECK(torch::isfinite(grad_b1).all().item<bool>(), "MoE backward grad_b1 has non-finite values");
            TORCH_CHECK(torch::isfinite(grad_b2).all().item<bool>(), "MoE backward grad_b2 has non-finite values");
        }

        // Compute dr = rowwise_dot(Gy, Y_pre) to propagate into routing weights
        auto grad_out_f32_dr = grad_out.to(torch::kFloat32).contiguous();
        auto dr = torch::empty({total_tokens}, grad_out_f32_dr.options());
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
            auto Gy = grad_out_f32_dr.narrow(0, cursor, M);
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
            torch::Tensor(), // w1_ptrs_table
            torch::Tensor(), // w2_ptrs_table
            torch::Tensor(), // b1_ptrs_table
            torch::Tensor(), // b2_ptrs_table
            dr.to(routing_weights_grouped.dtype()),  // routing_weights_grouped
            torch::Tensor(), // indices_grouped
            torch::Tensor(), // m_sizes_cpu
            torch::Tensor(), // policy_ids_cpu
            torch::Tensor(), // expert_ids_cpu
            torch::Tensor(), // token_offsets_cpu
            torch::Tensor(), // hidden_dim
            torch::Tensor(), // ffn_dim
            torch::Tensor()  // workspace (no grad)
        };
    }
};

torch::Tensor grouped_moe_autograd_forward(
    const torch::Tensor& input_grouped,
    const torch::Tensor& w1_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& b1_biases,
    const torch::Tensor& b2_biases,
    const torch::Tensor& w1_ptrs_table,
    const torch::Tensor& w2_ptrs_table,
    const torch::Tensor& b1_ptrs_table,
    const torch::Tensor& b2_ptrs_table,
    const torch::Tensor& routing_weights_grouped,
    const torch::Tensor& indices_grouped,
    const torch::Tensor& m_sizes_cpu,
    const torch::Tensor& policy_ids_cpu,
    const torch::Tensor& expert_ids_cpu,
    const torch::Tensor& token_offsets_cpu,
    int64_t hidden_dim,
    int64_t ffn_dim,
    lb::moe::MoEWorkspace* workspace
) {
    return GroupedMoEAutograd::apply(
        input_grouped,
        w1_weights,
        w2_weights,
        b1_biases,
        b2_biases,
        w1_ptrs_table,
        w2_ptrs_table,
        b1_ptrs_table,
        b2_ptrs_table,
        routing_weights_grouped,
        indices_grouped,
        m_sizes_cpu,
        policy_ids_cpu,
        expert_ids_cpu,
        token_offsets_cpu,
        hidden_dim,
        ffn_dim,
        workspace
    );
}

} // namespace moe
} // namespace lb
