#include "moe_backward_kernels.h"
#include "moe_cutlass_backward.h"
#include "gelu_constants.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <stdexcept>

namespace lb {
namespace moe {

// Environment variable guards for debug logging
static bool LB_MOE_LOG_BACKWARD = std::getenv("LB_MOE_LOG_BACKWARD") != nullptr;

/**
 * CUDA kernel: Gỹ[i,j] = Gy[i,j] * r[i]
 *
 * Applies per-token routing weights to upstream gradient.
 * This is the first operation in the backward pass.
 *
 * Thread organization: 1D grid over total elements (total_tokens * hidden_dim)
 * Each thread handles one element with grid-stride loop.
 */
__global__ void route_scale_backward_kernel(
    const __half* grad_output,      // [total_tokens, hidden_dim] FP16
    const float* routing_weights,   // [total_tokens] FP32
    __half* grad_y_tilde,           // [total_tokens, hidden_dim] FP16 output
    int64_t total_tokens,
    int64_t hidden_dim
) {
    // Grid-stride loop over all elements
    int64_t total_elements = total_tokens * hidden_dim;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        // Decode (token_idx, dim_idx) from linear index
        int64_t token_idx = i / hidden_dim;
        int64_t dim_idx = i % hidden_dim;

        // Load inputs
        float gy = __half2float(grad_output[i]);
        float r = routing_weights[token_idx];

        // Compute scaled gradient
        float gy_tilde = gy * r;

        // Store output
        grad_y_tilde[i] = __float2half(gy_tilde);
    }
}

/**
 * Host wrapper for route_scale_backward kernel.
 *
 * Validates inputs, computes launch configuration, and invokes kernel.
 */
void route_scale_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& routing_weights,
    torch::Tensor& grad_y_tilde,
    int64_t total_tokens,
    int64_t hidden_dim,
    cudaStream_t stream
) {
    // Input validation
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2D");
    TORCH_CHECK(grad_output.size(0) == total_tokens, "grad_output size mismatch");
    TORCH_CHECK(grad_output.size(1) == hidden_dim, "grad_output hidden_dim mismatch");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat16, "grad_output must be FP16");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");

    TORCH_CHECK(routing_weights.dim() == 1, "routing_weights must be 1D");
    TORCH_CHECK(routing_weights.size(0) == total_tokens, "routing_weights size mismatch");
    TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat32, "routing_weights must be FP32");
    TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be CUDA tensor");
    TORCH_CHECK(routing_weights.is_contiguous(), "routing_weights must be contiguous");

    TORCH_CHECK(grad_y_tilde.dim() == 2, "grad_y_tilde must be 2D");
    TORCH_CHECK(grad_y_tilde.size(0) == total_tokens, "grad_y_tilde size mismatch");
    TORCH_CHECK(grad_y_tilde.size(1) == hidden_dim, "grad_y_tilde hidden_dim mismatch");
    TORCH_CHECK(grad_y_tilde.scalar_type() == torch::kFloat16, "grad_y_tilde must be FP16");
    TORCH_CHECK(grad_y_tilde.is_cuda(), "grad_y_tilde must be CUDA tensor");
    TORCH_CHECK(grad_y_tilde.is_contiguous(), "grad_y_tilde must be contiguous");

    if (LB_MOE_LOG_BACKWARD) {
        printf("[MoE Backward] route_scale_backward:\n");
        printf("  total_tokens: %ld, hidden_dim: %ld\n", total_tokens, hidden_dim);
        printf("  total_elements: %ld\n", total_tokens * hidden_dim);
    }

    // Launch configuration
    int64_t total_elements = total_tokens * hidden_dim;
    const int threads_per_block = 256;
    int64_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Cap blocks to avoid excessive grid size
    const int64_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    if (LB_MOE_LOG_BACKWARD) {
        printf("  launch: blocks=%ld, threads=%d\n", num_blocks, threads_per_block);
    }

    // Get raw pointers
    const __half* grad_output_ptr = reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>());
    const float* routing_weights_ptr = routing_weights.data_ptr<float>();
    __half* grad_y_tilde_ptr = reinterpret_cast<__half*>(grad_y_tilde.data_ptr<at::Half>());

    // Launch kernel
    route_scale_backward_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        grad_output_ptr,
        routing_weights_ptr,
        grad_y_tilde_ptr,
        total_tokens,
        hidden_dim
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("route_scale_backward kernel launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    if (LB_MOE_LOG_BACKWARD) {
        // Synchronize to catch kernel errors
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("route_scale_backward kernel execution failed: ") +
                cudaGetErrorString(err)
            );
        }
        printf("  route_scale_backward completed successfully\n");
    }
}

/**
 * CUDA kernel: dZ[i,j] = dH[i,j] * GELU'(Z[i,j])
 *
 * Applies GELU derivative using exact erf-based formula.
 * CRITICAL: Uses same GELU constants as forward LinearCombinationGELU epilogue.
 *
 * Thread organization: 1D grid over total elements (total_tokens * ffn_dim)
 * Each thread handles one element with grid-stride loop.
 */
__global__ void gelu_backward_elementwise_kernel(
    const __half* grad_hidden,   // [total_tokens, ffn_dim] FP16
    const __half* z_recomputed,  // [total_tokens, ffn_dim] FP16
    __half* grad_z,              // [total_tokens, ffn_dim] FP16 output
    int64_t total_tokens,
    int64_t ffn_dim
) {
    // Grid-stride loop over all elements
    int64_t total_elements = total_tokens * ffn_dim;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        // Load inputs
        float dh = __half2float(grad_hidden[i]);
        float z = __half2float(z_recomputed[i]);

        // Compute GELU'(z) using shared constants from gelu_constants.h
        float gelu_grad = moe_activations::gelu_backward(z);

        // Apply chain rule: dZ = dH * GELU'(Z)
        float dz = dh * gelu_grad;

        // Store output
        grad_z[i] = __float2half(dz);
    }
}

/**
 * Host wrapper for GELU backward kernel.
 *
 * Validates inputs, computes launch configuration, and invokes kernel.
 */
void gelu_backward_kernel(
    const torch::Tensor& grad_hidden,
    const torch::Tensor& z_recomputed,
    torch::Tensor& grad_z,
    int64_t total_tokens,
    int64_t ffn_dim,
    cudaStream_t stream
) {
    // Input validation
    TORCH_CHECK(grad_hidden.dim() == 2, "grad_hidden must be 2D");
    TORCH_CHECK(grad_hidden.size(0) == total_tokens, "grad_hidden size mismatch");
    TORCH_CHECK(grad_hidden.size(1) == ffn_dim, "grad_hidden ffn_dim mismatch");
    TORCH_CHECK(grad_hidden.scalar_type() == torch::kFloat16, "grad_hidden must be FP16");
    TORCH_CHECK(grad_hidden.is_cuda(), "grad_hidden must be CUDA tensor");
    TORCH_CHECK(grad_hidden.is_contiguous(), "grad_hidden must be contiguous");

    TORCH_CHECK(z_recomputed.dim() == 2, "z_recomputed must be 2D");
    TORCH_CHECK(z_recomputed.size(0) == total_tokens, "z_recomputed size mismatch");
    TORCH_CHECK(z_recomputed.size(1) == ffn_dim, "z_recomputed ffn_dim mismatch");
    TORCH_CHECK(z_recomputed.scalar_type() == torch::kFloat16, "z_recomputed must be FP16");
    TORCH_CHECK(z_recomputed.is_cuda(), "z_recomputed must be CUDA tensor");
    TORCH_CHECK(z_recomputed.is_contiguous(), "z_recomputed must be contiguous");

    TORCH_CHECK(grad_z.dim() == 2, "grad_z must be 2D");
    TORCH_CHECK(grad_z.size(0) == total_tokens, "grad_z size mismatch");
    TORCH_CHECK(grad_z.size(1) == ffn_dim, "grad_z ffn_dim mismatch");
    TORCH_CHECK(grad_z.scalar_type() == torch::kFloat16, "grad_z must be FP16");
    TORCH_CHECK(grad_z.is_cuda(), "grad_z must be CUDA tensor");
    TORCH_CHECK(grad_z.is_contiguous(), "grad_z must be contiguous");

    if (LB_MOE_LOG_BACKWARD) {
        printf("[MoE Backward] gelu_backward_kernel:\n");
        printf("  total_tokens: %ld, ffn_dim: %ld\n", total_tokens, ffn_dim);
        printf("  total_elements: %ld\n", total_tokens * ffn_dim);
    }

    // Launch configuration
    int64_t total_elements = total_tokens * ffn_dim;
    const int threads_per_block = 256;
    int64_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Cap blocks to avoid excessive grid size
    const int64_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    if (LB_MOE_LOG_BACKWARD) {
        printf("  launch: blocks=%ld, threads=%d\n", num_blocks, threads_per_block);
    }

    // Get raw pointers
    const __half* grad_hidden_ptr = reinterpret_cast<const __half*>(grad_hidden.data_ptr<at::Half>());
    const __half* z_recomputed_ptr = reinterpret_cast<const __half*>(z_recomputed.data_ptr<at::Half>());
    __half* grad_z_ptr = reinterpret_cast<__half*>(grad_z.data_ptr<at::Half>());

    // Launch kernel
    gelu_backward_elementwise_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        grad_hidden_ptr,
        z_recomputed_ptr,
        grad_z_ptr,
        total_tokens,
        ffn_dim
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("gelu_backward_kernel launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    if (LB_MOE_LOG_BACKWARD) {
        // Synchronize to catch kernel errors
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("gelu_backward_kernel execution failed: ") +
                cudaGetErrorString(err)
            );
        }
        printf("  gelu_backward_kernel completed successfully\n");
    }
}

/**
 * CUDA kernel: db[j] = sum(grad[i,j] for i in range(total_tokens))
 *
 * Computes bias gradients via reduction over token dimension.
 * Uses shared memory for efficient within-block reduction, then atomic adds to global memory.
 *
 * Thread organization: 2D grid (blocks cover feature_dim, threads reduce over tokens)
 */
__global__ void bias_reduction_elementwise_kernel(
    const __half* grad,      // [total_tokens, feature_dim] FP16
    float* grad_bias,        // [feature_dim] FP32 output (accumulated)
    int64_t total_tokens,
    int64_t feature_dim
) {
    // Each block handles one feature dimension
    int64_t feature_idx = blockIdx.x;
    if (feature_idx >= feature_dim) return;

    // Shared memory for reduction within block
    extern __shared__ float shared_sum[];

    // Each thread accumulates a subset of tokens
    float local_sum = 0.0f;
    int64_t stride = blockDim.x;
    for (int64_t token_idx = threadIdx.x; token_idx < total_tokens; token_idx += stride) {
        int64_t idx = token_idx * feature_dim + feature_idx;
        local_sum += __half2float(grad[idx]);
    }

    // Store to shared memory
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce within block (tree reduction)
    for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes result to global memory (atomic add for accumulation)
    if (threadIdx.x == 0) {
        atomicAdd(&grad_bias[feature_idx], shared_sum[0]);
    }
}

/**
 * Host wrapper for bias reduction kernel.
 *
 * Validates inputs, computes launch configuration, and invokes kernel.
 */
void bias_reduction_kernel(
    const torch::Tensor& grad,
    torch::Tensor& grad_bias,
    int64_t total_tokens,
    int64_t feature_dim,
    cudaStream_t stream
) {
    // Input validation
    TORCH_CHECK(grad.dim() == 2, "grad must be 2D");
    TORCH_CHECK(grad.size(0) == total_tokens, "grad size mismatch");
    TORCH_CHECK(grad.size(1) == feature_dim, "grad feature_dim mismatch");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat16, "grad must be FP16");
    TORCH_CHECK(grad.is_cuda(), "grad must be CUDA tensor");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");

    TORCH_CHECK(grad_bias.dim() == 1, "grad_bias must be 1D");
    TORCH_CHECK(grad_bias.size(0) == feature_dim, "grad_bias size mismatch");
    TORCH_CHECK(grad_bias.scalar_type() == torch::kFloat32, "grad_bias must be FP32");
    TORCH_CHECK(grad_bias.is_cuda(), "grad_bias must be CUDA tensor");
    TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias must be contiguous");

    if (LB_MOE_LOG_BACKWARD) {
        printf("[MoE Backward] bias_reduction_kernel:\n");
        printf("  total_tokens: %ld, feature_dim: %ld\n", total_tokens, feature_dim);
    }

    // Launch configuration
    // One block per feature dimension, threads reduce over tokens
    const int threads_per_block = 256;
    int64_t num_blocks = feature_dim;

    // Shared memory size
    size_t shared_mem_bytes = threads_per_block * sizeof(float);

    if (LB_MOE_LOG_BACKWARD) {
        printf("  launch: blocks=%ld, threads=%d, shared_mem=%zu bytes\n",
               num_blocks, threads_per_block, shared_mem_bytes);
    }

    // Get raw pointers
    const __half* grad_ptr = reinterpret_cast<const __half*>(grad.data_ptr<at::Half>());
    float* grad_bias_ptr = grad_bias.data_ptr<float>();

    // Launch kernel
    bias_reduction_elementwise_kernel<<<num_blocks, threads_per_block, shared_mem_bytes, stream>>>(
        grad_ptr,
        grad_bias_ptr,
        total_tokens,
        feature_dim
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("bias_reduction_kernel launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    if (LB_MOE_LOG_BACKWARD) {
        // Synchronize to catch kernel errors
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("bias_reduction_kernel execution failed: ") +
                cudaGetErrorString(err)
            );
        }
        printf("  bias_reduction_kernel completed successfully\n");
    }
}

/**
 * CUDA kernel: dX_original[indices[i], j] += dX_grouped[i, j]
 *
 * Accumulates gradients from grouped/sorted order back to original token order.
 * Uses atomicAdd to handle potential conflicts when multiple grouped tokens map to same original index.
 *
 * Thread organization: 1D grid over total elements (total_tokens * hidden_dim)
 */
__global__ void scatter_add_backward_kernel(
    const __half* grad_x_grouped,   // [total_tokens, hidden_dim] FP16
    const int64_t* indices,         // [total_tokens] int64
    __half* grad_x_original,        // [batch_size, hidden_dim] FP16 output
    int64_t total_tokens,
    int64_t hidden_dim
) {
    // Grid-stride loop over all elements
    int64_t total_elements = total_tokens * hidden_dim;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        // Decode (token_idx, dim_idx) from linear index
        int64_t grouped_token_idx = i / hidden_dim;
        int64_t dim_idx = i % hidden_dim;

        // Get original token index
        int64_t original_token_idx = indices[grouped_token_idx];

        // Compute destination index
        int64_t dest_idx = original_token_idx * hidden_dim + dim_idx;

        // Load gradient value
        __half grad_val = grad_x_grouped[i];

        atomicAdd(&grad_x_original[dest_idx], grad_val);

    }
}

/**
 * Host wrapper for scatter-add backward kernel.
 *
 * Validates inputs, computes launch configuration, and invokes kernel.
 */
void scatter_add_backward(
    const torch::Tensor& grad_x_grouped,
    const torch::Tensor& indices,
    torch::Tensor& grad_x_original,
    int64_t total_tokens,
    int64_t hidden_dim,
    cudaStream_t stream
) {
    // Input validation
    TORCH_CHECK(grad_x_grouped.dim() == 2, "grad_x_grouped must be 2D");
    TORCH_CHECK(grad_x_grouped.size(0) == total_tokens, "grad_x_grouped size mismatch");
    TORCH_CHECK(grad_x_grouped.size(1) == hidden_dim, "grad_x_grouped hidden_dim mismatch");
    TORCH_CHECK(grad_x_grouped.scalar_type() == torch::kFloat16, "grad_x_grouped must be FP16");
    TORCH_CHECK(grad_x_grouped.is_cuda(), "grad_x_grouped must be CUDA tensor");
    TORCH_CHECK(grad_x_grouped.is_contiguous(), "grad_x_grouped must be contiguous");

    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");
    TORCH_CHECK(indices.size(0) == total_tokens, "indices size mismatch");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    TORCH_CHECK(grad_x_original.dim() == 2, "grad_x_original must be 2D");
    TORCH_CHECK(grad_x_original.size(1) == hidden_dim, "grad_x_original hidden_dim mismatch");
    TORCH_CHECK(grad_x_original.scalar_type() == torch::kFloat16, "grad_x_original must be FP16");
    TORCH_CHECK(grad_x_original.is_cuda(), "grad_x_original must be CUDA tensor");
    TORCH_CHECK(grad_x_original.is_contiguous(), "grad_x_original must be contiguous");

    if (LB_MOE_LOG_BACKWARD) {
        printf("[MoE Backward] scatter_add_backward:\n");
        printf("  total_tokens: %ld, hidden_dim: %ld\n", total_tokens, hidden_dim);
        printf("  batch_size: %ld\n", grad_x_original.size(0));
    }

    // Launch configuration
    int64_t total_elements = total_tokens * hidden_dim;
    const int threads_per_block = 256;
    int64_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Cap blocks to avoid excessive grid size
    const int64_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    if (LB_MOE_LOG_BACKWARD) {
        printf("  launch: blocks=%ld, threads=%d\n", num_blocks, threads_per_block);
    }

    // Get raw pointers
    const __half* grad_x_grouped_ptr = reinterpret_cast<const __half*>(grad_x_grouped.data_ptr<at::Half>());
    const int64_t* indices_ptr = indices.data_ptr<int64_t>();
    __half* grad_x_original_ptr = reinterpret_cast<__half*>(grad_x_original.data_ptr<at::Half>());

    // Launch kernel
    scatter_add_backward_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        grad_x_grouped_ptr,
        indices_ptr,
        grad_x_original_ptr,
        total_tokens,
        hidden_dim
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("scatter_add_backward kernel launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    if (LB_MOE_LOG_BACKWARD) {
        // Synchronize to catch kernel errors
        cudaStreamSynchronize(stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("scatter_add_backward kernel execution failed: ") +
                cudaGetErrorString(err)
            );
        }
        printf("  scatter_add_backward completed successfully\n");
    }
}

/**
 * Main orchestrator for MoE backward pass.
 *
 * Executes the full backward computation graph:
 *   1. Gỹ = Gy ⊙ r (route-scale)
 *   2. dW2 = Gỹᵀ @ H, db2 = sum(Gỹ)
 *   3. dH = Gỹ @ W2
 *   4. Z = X @ W1ᵀ + b1 (recompute)
 *   5. dZ = dH ⊙ GELU'(Z)
 *   6. dW1 = dZᵀ @ X, db1 = sum(dZ)
 *   7. dX_grouped = dZ @ W1
 *   8. dX = scatter_add(dX_grouped, indices)
 *
 * All intermediate buffers are allocated/freed within this function.
 * Device metadata (pointer arrays, problem sizes) is reused across GEMMs.
 */
void cutlass_grouped_moe_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_grouped,
    const torch::Tensor& hidden_grouped,
    const torch::Tensor& routing_weights_grouped,
    const torch::Tensor& indices_grouped,
    const torch::Tensor& w1_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& b1_biases,
    const torch::Tensor& b2_biases,
    const std::vector<int64_t>& m_sizes,
    const std::vector<int64_t>& policy_ids,
    const std::vector<int64_t>& expert_ids,
    const std::vector<int64_t>& token_offsets,
    torch::Tensor& grad_input,
    torch::Tensor& grad_w1,
    torch::Tensor& grad_w2,
    torch::Tensor& grad_b1,
    torch::Tensor& grad_b2
) {
    const int64_t group_count = m_sizes.size();
    if (group_count == 0) {
        if (LB_MOE_LOG_BACKWARD) {
            printf("[MoE Backward] No groups to process\n");
        }
        return;
    }

    const bool log_backward = LB_MOE_LOG_BACKWARD;

    // Extract dimensions
    const int64_t total_tokens_grouped = input_grouped.size(0);
    const int64_t hidden_dim = input_grouped.size(1);
    const int64_t ffn_dim = hidden_grouped.size(1);
    const int64_t batch_size = grad_output.size(0);
    const int64_t num_policies = w1_weights.size(0);
    const int64_t num_experts = w1_weights.size(1);

    if (log_backward) {
        printf("[MoE Backward] Starting backward pass:\n");
        printf("  group_count: %ld\n", group_count);
        printf("  total_tokens_grouped: %ld\n", total_tokens_grouped);
        printf("  hidden_dim: %ld, ffn_dim: %ld\n", hidden_dim, ffn_dim);
        printf("  batch_size: %ld\n", batch_size);
        printf("  num_policies: %ld, num_experts: %ld\n", num_policies, num_experts);
    }

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ========================================================================
    // Allocate intermediate buffers
    // ========================================================================

    // Gỹ: scaled gradient [total_tokens_grouped, hidden_dim] FP16
    torch::Tensor grad_y_tilde = torch::empty_like(grad_output.index({torch::indexing::Slice(0, total_tokens_grouped)}));

    // dH: hidden gradient [total_tokens_grouped, ffn_dim] FP16
    torch::Tensor grad_hidden = torch::empty({total_tokens_grouped, ffn_dim},
                                              torch::TensorOptions().dtype(torch::kFloat16).device(input_grouped.device()));

    // Z: recomputed pre-GELU activation [total_tokens_grouped, ffn_dim] FP16
    torch::Tensor z_recomputed = torch::empty_like(grad_hidden);

    // dZ: pre-GELU gradient [total_tokens_grouped, ffn_dim] FP16
    torch::Tensor grad_z = torch::empty_like(grad_hidden);

    // dX_grouped: input gradient in grouped order [total_tokens_grouped, hidden_dim] FP16
    torch::Tensor grad_x_grouped = torch::empty_like(input_grouped);

    // Initialize grad_input to zero (for scatter-add accumulation)
    grad_input.zero_();

    if (log_backward) {
        printf("  Allocated intermediate buffers\n");
    }

    // ========================================================================
    // Build per-group pointer arrays
    // ========================================================================

    std::vector<uintptr_t> input_ptrs, hidden_ptrs, routing_weight_ptrs;
    std::vector<uintptr_t> w1_ptrs, w2_ptrs, b1_ptrs, b2_ptrs;
    std::vector<uintptr_t> grad_y_tilde_ptrs, grad_hidden_ptrs, z_recomputed_ptrs, grad_z_ptrs, grad_x_grouped_ptrs;
    std::vector<uintptr_t> grad_w1_ptrs, grad_w2_ptrs, grad_b1_ptrs, grad_b2_ptrs;

    int64_t token_offset = 0;
    for (int64_t g = 0; g < group_count; ++g) {
        int64_t M = m_sizes[g];
        int64_t policy_id = policy_ids[g];
        int64_t expert_id = expert_ids[g];

        if (M == 0) {
            // Skip empty groups but maintain array structure
            token_offset += 0;
            continue;
        }

        // Input/saved tensors (grouped order)
        auto input_ptr = reinterpret_cast<uintptr_t>(input_grouped.data_ptr<at::Half>()) + token_offset * hidden_dim * sizeof(at::Half);
        auto hidden_ptr = reinterpret_cast<uintptr_t>(hidden_grouped.data_ptr<at::Half>()) + token_offset * ffn_dim * sizeof(at::Half);
        auto routing_weight_ptr = reinterpret_cast<uintptr_t>(routing_weights_grouped.data_ptr<float>()) + token_offset * sizeof(float);

        input_ptrs.push_back(input_ptr);
        hidden_ptrs.push_back(hidden_ptr);
        routing_weight_ptrs.push_back(routing_weight_ptr);

        // Weights/biases (indexed by policy, expert)
        auto w1_ptr = reinterpret_cast<uintptr_t>(w1_weights[policy_id][expert_id].data_ptr<at::Half>());
        auto w2_ptr = reinterpret_cast<uintptr_t>(w2_weights[policy_id][expert_id].data_ptr<at::Half>());
        auto b1_ptr = reinterpret_cast<uintptr_t>(b1_biases[policy_id][expert_id].data_ptr<at::Half>());
        auto b2_ptr = reinterpret_cast<uintptr_t>(b2_biases[policy_id][expert_id].data_ptr<at::Half>());

        w1_ptrs.push_back(w1_ptr);
        w2_ptrs.push_back(w2_ptr);
        b1_ptrs.push_back(b1_ptr);
        b2_ptrs.push_back(b2_ptr);

        // Intermediate buffers
        auto grad_y_tilde_ptr = reinterpret_cast<uintptr_t>(grad_y_tilde.data_ptr<at::Half>()) + token_offset * hidden_dim * sizeof(at::Half);
        auto grad_hidden_ptr = reinterpret_cast<uintptr_t>(grad_hidden.data_ptr<at::Half>()) + token_offset * ffn_dim * sizeof(at::Half);
        auto z_recomputed_ptr = reinterpret_cast<uintptr_t>(z_recomputed.data_ptr<at::Half>()) + token_offset * ffn_dim * sizeof(at::Half);
        auto grad_z_ptr = reinterpret_cast<uintptr_t>(grad_z.data_ptr<at::Half>()) + token_offset * ffn_dim * sizeof(at::Half);
        auto grad_x_grouped_ptr = reinterpret_cast<uintptr_t>(grad_x_grouped.data_ptr<at::Half>()) + token_offset * hidden_dim * sizeof(at::Half);

        grad_y_tilde_ptrs.push_back(grad_y_tilde_ptr);
        grad_hidden_ptrs.push_back(grad_hidden_ptr);
        z_recomputed_ptrs.push_back(z_recomputed_ptr);
        grad_z_ptrs.push_back(grad_z_ptr);
        grad_x_grouped_ptrs.push_back(grad_x_grouped_ptr);

        // Weight/bias gradients
        auto grad_w1_ptr = reinterpret_cast<uintptr_t>(grad_w1[policy_id][expert_id].data_ptr<float>());
        auto grad_w2_ptr = reinterpret_cast<uintptr_t>(grad_w2[policy_id][expert_id].data_ptr<float>());
        auto grad_b1_ptr = reinterpret_cast<uintptr_t>(grad_b1[policy_id][expert_id].data_ptr<float>());
        auto grad_b2_ptr = reinterpret_cast<uintptr_t>(grad_b2[policy_id][expert_id].data_ptr<float>());

        grad_w1_ptrs.push_back(grad_w1_ptr);
        grad_w2_ptrs.push_back(grad_w2_ptr);
        grad_b1_ptrs.push_back(grad_b1_ptr);
        grad_b2_ptrs.push_back(grad_b2_ptr);

        token_offset += M;
    }

    if (log_backward) {
        printf("  Built pointer arrays for %zu non-empty groups\n", input_ptrs.size());
    }

    // ========================================================================
    // Step 1: Route-scale kernel (Gỹ = Gy ⊙ r)
    // ========================================================================

    if (log_backward) {
        printf("  [Step 1] Route-scale: Gỹ = Gy ⊙ r\n");
    }

    // Gather grad_output into grouped order first
    // grad_output is in original order [batch_size, hidden_dim]
    // We need to gather it into grouped order using indices_grouped
    torch::Tensor grad_output_grouped = grad_output.index_select(0, indices_grouped);

    route_scale_backward(
        grad_output_grouped,
        routing_weights_grouped,
        grad_y_tilde,
        total_tokens_grouped,
        hidden_dim,
        stream
    );

    // ========================================================================
    // Step 2: dW2 GEMM + db2 reduction
    // ========================================================================

    if (log_backward) {
        printf("  [Step 2] dW2 = Gỹᵀ @ H, db2 = sum(Gỹ)\n");
    }

    // GEMM: dW2 = Gỹᵀ @ H
    lb::moe::cutlass_grouped_gemm_dW2(
        grad_y_tilde_ptrs.data(),
        hidden_ptrs.data(),
        grad_w2_ptrs.data(),
        m_sizes.data(),
        input_ptrs.size(),  // non-empty group count
        hidden_dim,
        ffn_dim
    );

    // Bias reduction: db2 = sum(Gỹ, dim=0) for each group
    for (size_t g = 0; g < input_ptrs.size(); ++g) {
        int64_t M = m_sizes[g];
        if (M == 0) continue;

        int64_t policy_id = policy_ids[g];
        int64_t expert_id = expert_ids[g];

        // Slice Gỹ for this group
        auto grad_y_tilde_slice = grad_y_tilde.narrow(0, token_offsets[g], M);

        // Reduce to grad_b2
        auto grad_b2_slice = grad_b2[policy_id][expert_id];

        bias_reduction_kernel(
            grad_y_tilde_slice,
            grad_b2_slice,
            M,
            hidden_dim,
            stream
        );
    }

    // ========================================================================
    // Step 3: dH GEMM (dH = Gỹ @ W2)
    // ========================================================================

    if (log_backward) {
        printf("  [Step 3] dH = Gỹ @ W2\n");
    }

    lb::moe::cutlass_grouped_gemm_dH(
        grad_y_tilde_ptrs.data(),
        w2_ptrs.data(),
        grad_hidden_ptrs.data(),
        m_sizes.data(),
        input_ptrs.size(),
        hidden_dim,
        ffn_dim
    );

    // ========================================================================
    // Step 4: Recompute Z (Z = X @ W1ᵀ + b1)
    // ========================================================================

    if (log_backward) {
        printf("  [Step 4] Recompute Z = X @ W1ᵀ + b1\n");
    }

    lb::moe::cutlass_grouped_gemm_recompute_Z(
        input_ptrs.data(),
        w1_ptrs.data(),
        b1_ptrs.data(),
        z_recomputed_ptrs.data(),
        m_sizes.data(),
        input_ptrs.size(),
        hidden_dim,
        ffn_dim
    );

    // ========================================================================
    // Step 5: GELU' kernel (dZ = dH ⊙ GELU'(Z))
    // ========================================================================

    if (log_backward) {
        printf("  [Step 5] dZ = dH ⊙ GELU'(Z)\n");
    }

    gelu_backward_kernel(
        grad_hidden,
        z_recomputed,
        grad_z,
        total_tokens_grouped,
        ffn_dim,
        stream
    );

    // ========================================================================
    // Step 6: dW1 GEMM + db1 reduction
    // ========================================================================

    if (log_backward) {
        printf("  [Step 6] dW1 = dZᵀ @ X, db1 = sum(dZ)\n");
    }

    // GEMM: dW1 = dZᵀ @ X
    lb::moe::cutlass_grouped_gemm_dW1(
        grad_z_ptrs.data(),
        input_ptrs.data(),
        grad_w1_ptrs.data(),
        m_sizes.data(),
        input_ptrs.size(),
        hidden_dim,
        ffn_dim
    );

    // Bias reduction: db1 = sum(dZ, dim=0) for each group
    for (size_t g = 0; g < input_ptrs.size(); ++g) {
        int64_t M = m_sizes[g];
        if (M == 0) continue;

        int64_t policy_id = policy_ids[g];
        int64_t expert_id = expert_ids[g];

        // Slice dZ for this group
        auto grad_z_slice = grad_z.narrow(0, token_offsets[g], M);

        // Reduce to grad_b1
        auto grad_b1_slice = grad_b1[policy_id][expert_id];

        bias_reduction_kernel(
            grad_z_slice,
            grad_b1_slice,
            M,
            ffn_dim,
            stream
        );
    }

    // ========================================================================
    // Step 7: dX GEMM (dX_grouped = dZ @ W1)
    // ========================================================================

    if (log_backward) {
        printf("  [Step 7] dX_grouped = dZ @ W1\n");
    }

    lb::moe::cutlass_grouped_gemm_dX(
        grad_z_ptrs.data(),
        w1_ptrs.data(),
        grad_x_grouped_ptrs.data(),
        m_sizes.data(),
        input_ptrs.size(),
        hidden_dim,
        ffn_dim
    );

    // ========================================================================
    // Step 8: Scatter-add (dX = scatter_add(dX_grouped, indices))
    // ========================================================================

    if (log_backward) {
        printf("  [Step 8] dX = scatter_add(dX_grouped, indices)\n");
    }

    scatter_add_backward(
        grad_x_grouped,
        indices_grouped,
        grad_input,
        total_tokens_grouped,
        hidden_dim,
        stream
    );

    // ========================================================================
    // Final synchronization
    // ========================================================================

    cudaStreamSynchronize(stream);

    if (log_backward) {
        printf("[MoE Backward] Backward pass completed successfully\n");
    }
}

} // namespace moe
} // namespace lb
