/**
 * lb_kernels.cu - Raw CUDA/cuBLASLt kernel primitives
 *
 * This file contains the SINGLE SOURCE OF TRUTH for core computational kernels.
 * All functions here are pure computation with no autograd - they are called by
 * both training (via autograd wrappers) and inference (directly).
 *
 * This ensures numerical parity between training and inference.
 */

#include "lb_kernels.h"
#include "moe_cutlass_kernels.h"

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <mutex>
#include <iostream>
#include <cstdlib>

// ============================================================================
// Environment-gated debug helpers
// ============================================================================
static inline bool lb_env_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    if (v[0] == '0' || v[0] == 'F' || v[0] == 'f' || v[0] == 'N' || v[0] == 'n') return false;
    return true;
}
static bool LB_DBG_STATS() {
    static int flag = -1;
    if (flag < 0) flag = lb_env_enabled("LB_DEBUG_STATS");
    return flag != 0;
}
static bool LB_KERNEL_DEBUG_ENABLED() {
    static int flag = -1;
    if (flag < 0) flag = lb_env_enabled("LB_ENABLE_KERNEL_DEBUG") ? 1 : 0;
    return flag != 0;
}

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

// ============================================================================
// CUDA Kernels
// ============================================================================

namespace {

// Embedding lookup kernel
template <typename scalar_t>
__global__ void indexed_batched_embedding_kernel(
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ policy_indices,
    const scalar_t* __restrict__ weight_cache,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t time_steps,
    int64_t hidden_dim,
    int64_t vocab_size) {
    const int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_tokens = batch_size * time_steps;
    if (linear_index >= total_tokens) {
        return;
    }

    const int64_t b = linear_index / time_steps;
    const int64_t t = linear_index % time_steps;

    const int64_t policy_idx = policy_indices[b];
    const int64_t token_idx = indices[linear_index];

    const scalar_t* table_ptr = weight_cache + policy_idx * vocab_size * hidden_dim;
    const scalar_t* src_ptr = table_ptr + token_idx * hidden_dim;
    scalar_t* dst_ptr = output + linear_index * hidden_dim;

#pragma unroll 4
    for (int64_t h = 0; h < hidden_dim; ++h) {
        dst_ptr[h] = src_ptr[h];
    }
}

// Warp reduction for LayerNorm
template <typename acc_t>
__device__ acc_t warp_reduce_sum(acc_t val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// LayerNorm kernel
template <typename scalar_t, typename acc_t>
__global__ void indexed_batched_layer_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma_cache,
    const scalar_t* __restrict__ beta_cache,
    const int64_t* __restrict__ policy_indices,
    int64_t batch_size,
    int64_t time_steps,
    int64_t hidden_dim,
    double eps,
    bool debug_enabled) {

    const int64_t token_index = blockIdx.x;
    if (token_index >= batch_size * time_steps) {
        return;
    }

    const int64_t b = token_index / time_steps;
    const int64_t policy_idx = policy_indices[b];

    const scalar_t* input_ptr = input + token_index * hidden_dim;
    scalar_t* output_ptr = output + token_index * hidden_dim;

    const scalar_t* gamma_ptr = gamma_cache + policy_idx * hidden_dim;
    const scalar_t* beta_ptr = beta_cache + policy_idx * hidden_dim;

    __shared__ acc_t shared_partial[32];
    __shared__ acc_t shared_sum;
    __shared__ acc_t shared_var;

    const int lane = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    // Compute mean
    acc_t thread_sum = static_cast<acc_t>(0);
    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        thread_sum += static_cast<acc_t>(input_ptr[h]);
    }
    acc_t warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        shared_partial[warp_id] = warp_sum;
    }
    __syncthreads();

    acc_t block_sum = (threadIdx.x < warp_count) ? shared_partial[threadIdx.x] : static_cast<acc_t>(0);
    block_sum = warp_reduce_sum(block_sum);
    if (threadIdx.x == 0) {
        shared_sum = block_sum;
    }
    __syncthreads();
    const acc_t mean = shared_sum / static_cast<acc_t>(hidden_dim);

    // Compute variance
    acc_t thread_var = static_cast<acc_t>(0);
    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        const acc_t diff = static_cast<acc_t>(input_ptr[h]) - mean;
        thread_var += diff * diff;
    }
    warp_sum = warp_reduce_sum(thread_var);
    if (lane == 0) {
        shared_partial[warp_id] = warp_sum;
    }
    __syncthreads();

    block_sum = (threadIdx.x < warp_count) ? shared_partial[threadIdx.x] : static_cast<acc_t>(0);
    block_sum = warp_reduce_sum(block_sum);
    if (threadIdx.x == 0) {
        shared_var = block_sum / static_cast<acc_t>(hidden_dim);
    }
    __syncthreads();

    const acc_t inv_std = rsqrt(shared_var + static_cast<acc_t>(eps));

    // Normalize and scale
    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        const acc_t norm = (static_cast<acc_t>(input_ptr[h]) - mean) * inv_std;
        const acc_t scaled = norm * static_cast<acc_t>(gamma_ptr[h]) + static_cast<acc_t>(beta_ptr[h]);
        output_ptr[h] = static_cast<scalar_t>(scaled);
    }
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

namespace lb {
namespace kernels {

torch::Tensor indexed_batched_embedding(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices) {

    TORCH_CHECK(weight_cache.dim() == 3, "Embedding cache must be [W, vocab, hidden]");
    TORCH_CHECK(indices.dim() == 2, "Indices must be [B, T]");
    TORCH_CHECK(policy_indices.dim() == 1, "policy_indices must be [B]");
    TORCH_CHECK(!indices.isnan().any().item<bool>(), "Embedding indices contain NaN");
    TORCH_CHECK(!indices.isinf().any().item<bool>(), "Embedding indices contain Inf");
    TORCH_CHECK((indices >= 0).all().item<bool>(), "Embedding indices contain negative values");

    auto vocab_size = weight_cache.size(1);
    auto hidden_dim = weight_cache.size(2);
    auto batch_size = indices.size(0);
    auto time_steps = indices.size(1);

    TORCH_CHECK(weight_cache.is_cuda(), "indexed_batched_embedding expects CUDA tensors for weight_cache");

    auto weight_contig = weight_cache.contiguous();
    auto indices_contig = indices.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != weight_cache.device()) {
        policy_contig = policy_contig.to(weight_cache.device());
    }

    auto output = torch::empty({batch_size, time_steps, hidden_dim}, weight_cache.options());

    const int threads = 256;
    const int64_t total_tokens = batch_size * time_steps;
    const dim3 blocks((total_tokens + threads - 1) / threads);

    c10::cuda::CUDAGuard guard(weight_cache.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weight_cache.scalar_type(), "indexed_batched_embedding_cuda", [&] {
            indexed_batched_embedding_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                indices_contig.data_ptr<int64_t>(),
                policy_contig.data_ptr<int64_t>(),
                weight_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                time_steps,
                hidden_dim,
                vocab_size);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor indexed_batched_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps) {

    TORCH_CHECK(input.dim() == 3, "input must be [B, T, H]");
    TORCH_CHECK(gamma_cache.dim() == 2, "gamma cache must be [W, H]");
    TORCH_CHECK(beta_cache.dim() == 2, "beta cache must be [W, H]");

    // Detailed input validation
    auto check_tensor = [](const torch::Tensor& t, const char* name) {
        if (t.isnan().any().item<bool>()) {
            TORCH_CHECK(false, name, " contains NaN values");
        }
        if (t.isinf().any().item<bool>()) {
            TORCH_CHECK(false, name, " contains Inf values");
        }
        if (LB_DBG_STATS()) {
            auto stats = t.to(torch::kFloat32);
            std::cerr << name << " stats - min: " << stats.min().item<float>()
                     << " max: " << stats.max().item<float>()
                     << " mean: " << stats.mean().item<float>()
                     << " std: " << stats.std().item<float>() << std::endl;
        }
    };

    check_tensor(input, "LayerNorm input");
    check_tensor(gamma_cache, "LayerNorm gamma_cache");
    check_tensor(beta_cache, "LayerNorm beta_cache");

    auto batch_size = input.size(0);
    auto time_steps = input.size(1);
    auto hidden_dim = input.size(2);

    TORCH_CHECK(input.is_cuda(), "indexed_batched_layer_norm expects CUDA tensors for input");

    auto input_contig = input.contiguous();
    auto gamma_t = gamma_cache;
    auto beta_t  = beta_cache;
    if (gamma_t.device() != input.device()) gamma_t = gamma_t.to(input.device());
    if (beta_t.device() != input.device())  beta_t  = beta_t.to(input.device());
    if (gamma_t.scalar_type() != input.scalar_type()) gamma_t = gamma_t.to(input.scalar_type());
    if (beta_t.scalar_type()  != input.scalar_type())  beta_t = beta_t.to(input.scalar_type());
    auto gamma_contig = gamma_t.contiguous();
    auto beta_contig  = beta_t.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != input.device()) {
        policy_contig = policy_contig.to(input.device());
    }

    // Validate policy indices
    const int64_t num_policies = gamma_contig.size(0);
    TORCH_CHECK(num_policies == beta_contig.size(0), "LayerNorm caches (gamma/beta) disagree on W dimension");
    TORCH_CHECK(gamma_contig.size(1) == hidden_dim && beta_contig.size(1) == hidden_dim,
                "LayerNorm caches must have H == hidden_dim");
    auto pol_min = policy_contig.min().item<int64_t>();
    auto pol_max = policy_contig.max().item<int64_t>();
    if (pol_min < 0 || pol_max >= num_policies) {
        auto clamped = torch::clamp(policy_contig, 0, num_policies - 1);
        std::cerr << "[WARN] indexed_batched_layer_norm: clamping policy indices to [0," << (num_policies-1)
                  << "] (observed min=" << pol_min << ", max=" << pol_max << ")" << std::endl;
        policy_contig = clamped;
    }

    auto output = torch::empty_like(input_contig, input_contig.options());

    const int threads = hidden_dim >= 256 ? 256 : 128;
    const int64_t total_tokens = batch_size * time_steps;

    c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool kernel_debug_enabled = LB_KERNEL_DEBUG_ENABLED();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "indexed_batched_layer_norm_cuda", [&] {
            using acc_t = at::acc_type<scalar_t, true>;
            indexed_batched_layer_norm_kernel<scalar_t, acc_t><<<total_tokens, threads, 0, stream>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma_contig.data_ptr<scalar_t>(),
                beta_contig.data_ptr<scalar_t>(),
                policy_contig.data_ptr<int64_t>(),
                batch_size,
                time_steps,
                hidden_dim,
                eps,
                kernel_debug_enabled);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor indexed_batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, Microseconds>& timers,
    IndexedLinearEpilogue epilogue) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "indexed_batched_linear expects CUDA tensors for input");
    TORCH_CHECK(weight_cache.is_cuda(), "weight_cache must be on CUDA device");
    TORCH_CHECK(bias_cache.is_cuda(), "bias_cache must be on CUDA device");

    TORCH_CHECK(input.dim()        == 3, "Input tensor must be 3D [B, T, in], got ", input.sizes());
    TORCH_CHECK(weight_cache.dim() == 3, "weight_cache must be 3D [W, out, in], got ", weight_cache.sizes());
    TORCH_CHECK(bias_cache.dim()   == 2, "bias_cache must be 2D [W, out], got ", bias_cache.sizes());
    TORCH_CHECK(policy_indices.dim() == 1, "policy_indices must be 1D [B], got ", policy_indices.sizes());

    const int64_t B = input.size(0);
    const int64_t T = input.size(1);
    const int64_t In_Dim = input.size(2);
    const int64_t W = weight_cache.size(0);
    const int64_t Out_Dim = weight_cache.size(1);

    TORCH_CHECK(policy_indices.size(0) == B, "policy_indices batch size must match input");
    TORCH_CHECK(weight_cache.size(2) == In_Dim, "weight_cache input dim must match input tensor");
    TORCH_CHECK(bias_cache.size(0) == W && bias_cache.size(1) == Out_Dim, "bias_cache shape mismatch");

    // Data preparation
    auto input_f32 = input.to(torch::kFloat32).contiguous();
    auto weight_f32 = weight_cache.to(torch::kFloat32).contiguous();
    auto bias_f32 = bias_cache.to(torch::kFloat32).contiguous();

    auto policy_indices_long = policy_indices.to(torch::kLong).contiguous();
    if (policy_indices_long.device() != input.device()) {
        policy_indices_long = policy_indices_long.to(input.device());
    }

    auto policy_min = policy_indices_long.min().item<int64_t>();
    auto policy_max = policy_indices_long.max().item<int64_t>();
    TORCH_CHECK(policy_min >= 0 && policy_max < W,
                "policy_indices are out of valid range. Got min=", policy_min, ", max=", policy_max,
                ", but valid range is [0, ", W - 1, "]");

    auto t0 = Clock::now();
    auto weight_batched = weight_f32.index_select(0, policy_indices_long).contiguous();
    auto bias_batched = bias_f32.index_select(0, policy_indices_long).contiguous();
    auto t1 = Clock::now();
    timers["linear_index_select_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // Batched matmul
    auto gemm_t0 = Clock::now();
    auto weight_transposed = weight_batched.transpose(1, 2);  // [B, In_Dim, Out_Dim]
    // Fused bias add + batched GEMM
    auto bias_expanded = bias_batched.unsqueeze(1).expand({B, T, Out_Dim});
    auto output = torch::baddbmm(bias_expanded, input_f32, weight_transposed, /*beta=*/1.0, /*alpha=*/1.0);
    auto gemm_t1 = Clock::now();
    timers["linear_cublas_matmul_us"] += std::chrono::duration_cast<Microseconds>(gemm_t1 - gemm_t0);

    // Optional GELU epilogue
    if (epilogue == IndexedLinearEpilogue::BiasGELU) {
        auto epi_t0 = Clock::now();
        output = torch::gelu(output);
        auto epi_t1 = Clock::now();
        timers["linear_epilogue_us"] += std::chrono::duration_cast<Microseconds>(epi_t1 - epi_t0);
    }

    return output.to(input.scalar_type());
}

void grouped_ffn_gemm_forward(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* b2_ptrs,
    const uintptr_t* output_ptrs,
    const uintptr_t* routing_weight_ptrs,
    const int64_t* m_sizes,
    const int64_t* policy_ids,
    const int64_t* expert_ids,
    const int64_t* token_offsets,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim) {

    TORCH_CHECK(m_sizes != nullptr, "grouped_ffn_gemm_forward: m_sizes must not be null");
    TORCH_CHECK(policy_ids != nullptr, "grouped_ffn_gemm_forward: policy_ids must not be null");
    TORCH_CHECK(expert_ids != nullptr, "grouped_ffn_gemm_forward: expert_ids must not be null");
    TORCH_CHECK(routing_weight_ptrs != nullptr, "grouped_ffn_gemm_forward: routing_weight_ptrs must not be null");

    // Delegate to CUTLASS grouped MoE implementation
    lb::moe::cutlass_grouped_moe_forward(
        input_ptrs,
        w1_ptrs,
        b1_ptrs,
        w2_ptrs,
        b2_ptrs,
        output_ptrs,
        routing_weight_ptrs,
        m_sizes,
        policy_ids,
        expert_ids,
        token_offsets,
        group_count,
        hidden_dim,
        ffn_dim
    );
}

} // namespace kernels
} // namespace lb
