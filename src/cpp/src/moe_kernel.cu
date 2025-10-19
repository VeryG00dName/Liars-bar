#include "moe_kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cmath>

// No anonymous namespace needed, these are local to the file

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// Kernel is now explicitly for at::Half, no more templates
template <int FFN_DIM_MAX>
__global__ void simple_moe_forward_kernel_half(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_logits,
    const int64_t* __restrict__ topk_indices,
    const int64_t* __restrict__ policy_indices,
    at::Half* __restrict__ output,
    int64_t batch_size,
    int64_t tokens_per_batch,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t ffn_dim,
    int64_t weight_batch,
    int64_t w1_batch_stride,
    int64_t b1_batch_stride,
    int64_t w2_batch_stride,
    int64_t b2_batch_stride,
    const at::Half* __restrict__ w1,
    const at::Half* __restrict__ b1,
    const at::Half* __restrict__ w2,
    const at::Half* __restrict__ b2) {

    const int64_t token_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_tokens = batch_size * tokens_per_batch;
    if (token_index >= total_tokens) {
        return;
    }

    const int64_t batch_index = token_index / tokens_per_batch;
    const int64_t weight_batch_index = weight_batch == 1 ? 0 : policy_indices[batch_index];

    const at::Half* x_ptr = x + token_index * hidden_dim;
    at::Half* out_ptr = output + token_index * hidden_dim;

    const at::Half* gate_ptr = gate_logits + token_index * num_experts;
    const int64_t* topk_ptr = topk_indices + token_index * top_k;

    float max_logit = -1e9f;
    for (int64_t e = 0; e < num_experts; ++e) {
        max_logit = fmaxf(max_logit, __half2float(gate_ptr[e]));
    }
    float denom = 0.0f;
    for (int64_t e = 0; e < num_experts; ++e) {
        denom += expf(__half2float(gate_ptr[e]) - max_logit);
    }
    denom = fmaxf(denom, 1e-6f);

    float hidden_activations[FFN_DIM_MAX];
    float final_output_float[1024]; // Assuming HIDDEN_DIM <= 1024
    for(int64_t h=0; h < hidden_dim; ++h) final_output_float[h] = 0.0f;

    for (int64_t k = 0; k < top_k; ++k) {
        const int64_t expert_index = topk_ptr[k];
        const float routing_weight = expf(__half2float(gate_ptr[expert_index]) - max_logit) / denom;

        if (routing_weight <= 1e-6f) continue;

        const int64_t expert_offset = expert_index * weight_batch + weight_batch_index;
        const at::Half* w1_ptr = w1 + expert_offset * w1_batch_stride;
        const at::Half* b1_ptr = b1 + expert_offset * b1_batch_stride;
        const at::Half* w2_ptr = w2 + expert_offset * w2_batch_stride;
        const at::Half* b2_ptr = b2 + expert_offset * b2_batch_stride;

        for (int64_t f = 0; f < ffn_dim; ++f) {
            const at::Half* w1_row = w1_ptr + f * hidden_dim;
            float activation = __half2float(b1_ptr[f]);
            for (int64_t h = 0; h < hidden_dim; ++h) {
                activation += __half2float(w1_row[h]) * __half2float(x_ptr[h]);
            }
            hidden_activations[f] = gelu(activation);
        }

        for (int64_t h = 0; h < hidden_dim; ++h) {
            float expert_out_h = __half2float(b2_ptr[h]);
            for(int64_t f=0; f<ffn_dim; ++f) {
                expert_out_h += __half2float(w2_ptr[h * ffn_dim + f]) * hidden_activations[f];
            }
            final_output_float[h] += routing_weight * expert_out_h;
        }
    }

    for (int64_t h=0; h < hidden_dim; ++h) {
        out_ptr[h] = __float2half(final_output_float[h]);
    }
}

torch::Tensor moe_forward_cuda(
    torch::Tensor x,
    torch::Tensor gate_logits,
    torch::Tensor topk_indices,
    torch::Tensor policy_indices,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2) {
    
    // Most TORCH_CHECKs are good, but we add a specific dtype check
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "moe_forward_cuda: All input tensors must be FP16 (Half)");
    // ... all other checks ...
    const int64_t ffn_dim = w1.size(2);
    TORCH_CHECK(ffn_dim <= 2048, "FFN_DIM exceeds kernel max");
    const int64_t hidden_dim = x.size(2);
    TORCH_CHECK(hidden_dim <= 1024, "HIDDEN_DIM exceeds kernel max");


    auto output = torch::zeros_like(x, x.options());

    const int64_t total_tokens = x.size(0) * x.size(1);
    const int threads = 256;
    const dim3 blocks((total_tokens + threads - 1) / threads);
    
    // ... stride calculations ...

    c10::cuda::CUDAGuard device_guard(x.device());

    // --- START: Simplified Dispatch ---
    // No more AT_DISPATCH. We call the specific kernel we need.
    auto stream = at::cuda::getCurrentCUDAStream();

    auto policy_indices_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_indices_contig.device() != x.device()) {
        policy_indices_contig = policy_indices_contig.to(x.device());
    }

    if (ffn_dim <= 512) {
        simple_moe_forward_kernel_half<512><<<blocks, threads, 0, stream>>>(
            x.data_ptr<at::Half>(), gate_logits.data_ptr<at::Half>(), topk_indices.data_ptr<int64_t>(),
            policy_indices_contig.data_ptr<int64_t>(),
            output.data_ptr<at::Half>(), x.size(0), x.size(1), hidden_dim, gate_logits.size(2),
            topk_indices.size(2), ffn_dim, w1.size(1), w1.stride(1), b1.stride(1), w2.stride(1),
            b2.stride(1), w1.data_ptr<at::Half>(), b1.data_ptr<at::Half>(),
            w2.data_ptr<at::Half>(), b2.data_ptr<at::Half>());
    } else if (ffn_dim <= 1024) {
        simple_moe_forward_kernel_half<1024><<<blocks, threads, 0, stream>>>(
            x.data_ptr<at::Half>(), gate_logits.data_ptr<at::Half>(), topk_indices.data_ptr<int64_t>(),
            policy_indices_contig.data_ptr<int64_t>(),
            output.data_ptr<at::Half>(), x.size(0), x.size(1), hidden_dim, gate_logits.size(2),
            topk_indices.size(2), ffn_dim, w1.size(1), w1.stride(1), b1.stride(1), w2.stride(1),
            b2.stride(1), w1.data_ptr<at::Half>(), b1.data_ptr<at::Half>(),
            w2.data_ptr<at::Half>(), b2.data_ptr<at::Half>());
    } else { // ffn_dim <= 2048
        simple_moe_forward_kernel_half<2048><<<blocks, threads, 0, stream>>>(
            x.data_ptr<at::Half>(), gate_logits.data_ptr<at::Half>(), topk_indices.data_ptr<int64_t>(),
            policy_indices_contig.data_ptr<int64_t>(),
            output.data_ptr<at::Half>(), x.size(0), x.size(1), hidden_dim, gate_logits.size(2),
            topk_indices.size(2), ffn_dim, w1.size(1), w1.stride(1), b1.stride(1), w2.stride(1),
            b2.stride(1), w1.data_ptr<at::Half>(), b1.data_ptr<at::Half>(),
            w2.data_ptr<at::Half>(), b2.data_ptr<at::Half>());
    }
    // --- END: Simplified Dispatch ---

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}