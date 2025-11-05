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
    static const bool flag = lb_env_enabled("LB_DEBUG_STATS");
    return flag;
}
static bool LB_KERNEL_DEBUG_ENABLED() {
    static const bool flag = lb_env_enabled("LB_ENABLE_KERNEL_DEBUG");
    return flag;
}

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

// ============================================================================
// CUDA Kernels
// ============================================================================

namespace {

// ------------------------------------------------------------------
//  Helper to turn the PyTorch half type into the CUDA POD half
// ------------------------------------------------------------------
template <typename T>
struct CudaHalfHelper {
    using type = T;                     // default: identity
};

template <>
struct CudaHalfHelper<c10::Half> {
    using type = __half;                // map to CUDA half
};

// ------------------------------------------------------------------
//  Vector‑type traits (only POD types)
// ------------------------------------------------------------------
template <typename T> struct VecTraits;   // primary – left undefined

// float  → float4
template <> struct VecTraits<float> {
    using Vec   = float4;
    static constexpr int elems = 4;
};

// double → double2
template <> struct VecTraits<double> {
    using Vec   = double2;
    static constexpr int elems = 2;
};

// CUDA half → __half2 (2 × 2‑byte)
template <> struct VecTraits<__half> {
    using Vec   = __half2;
    static constexpr int elems = 2;
};

// ============================================================================
// Embedding lookup kernel (type-correct vector loads)
// ============================================================================
template <typename scalar_t>
__global__ void indexed_batched_embedding_kernel(
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ policy_indices,
    const scalar_t* __restrict__ weight_cache,   // may be c10::Half or POD
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t time_steps,
    int64_t hidden_dim,
    int64_t vocab_size) {

    // -------------------------------------------------------------
    //  1️⃣  Convert the scalar type to a CUDA POD half if needed
    // -------------------------------------------------------------
    using RealScalar = typename CudaHalfHelper<scalar_t>::type;   // __half for half, otherwise unchanged
    using Traits      = VecTraits<RealScalar>;
    using Vec         = typename Traits::Vec;
    constexpr int VEC_ELEMS = Traits::elems;   // 4 for float, 2 for double/half

    // -------------------------------------------------------------
    //  2️⃣  Thread-index bookkeeping: ONE BLOCK PER TOKEN
    //      All threads in the block cooperate to copy one token's embedding
    // -------------------------------------------------------------
    const int64_t token_index = blockIdx.x;  // One block handles one token
    const int64_t total_tokens = batch_size * time_steps;
    if (token_index >= total_tokens) return;

    const int64_t b          = token_index / time_steps;
    const int64_t policy_idx = policy_indices[b];
    const int64_t token_idx  = indices[token_index];

    // -------------------------------------------------------------
    //  3️⃣  Pointer arithmetic – reinterpret to the POD type
    // -------------------------------------------------------------
    const RealScalar* table_ptr = reinterpret_cast<const RealScalar*>(weight_cache)
                                + policy_idx * vocab_size * hidden_dim;
    const RealScalar* src_ptr   = table_ptr + token_idx * hidden_dim;
    RealScalar*       dst_ptr   = reinterpret_cast<RealScalar*>(output)
                                + token_index * hidden_dim;

    // -------------------------------------------------------------
    //  4️⃣  Vectorised copy (if hidden_dim is a multiple of VEC_ELEMS)
    // -------------------------------------------------------------
    int64_t vec_end = hidden_dim - (hidden_dim % VEC_ELEMS);
    for (int64_t h = threadIdx.x * VEC_ELEMS;
         h < vec_end;
         h += blockDim.x * VEC_ELEMS) {

        // Safe POD reinterpret‑cast – both src and dst are 4‑byte aligned
        *reinterpret_cast<Vec*>(&dst_ptr[h]) =
            *reinterpret_cast<const Vec*>(&src_ptr[h]);
    }

    // -------------------------------------------------------------
    //  5️⃣  Tail – scalar copy for the remainder (covers odd hidden_dim)
    // -------------------------------------------------------------
    for (int64_t h = vec_end + threadIdx.x; h < hidden_dim; h += blockDim.x) {
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

    // Launch configuration: one block per token, all threads cooperate on copying one embedding
    const int threads = 256;
    const int64_t total_tokens = batch_size * time_steps;
    const dim3 blocks(total_tokens);  // One block per token

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
    TORCH_CHECK(input.scalar_type() == gamma_cache.scalar_type() && input.scalar_type() == beta_cache.scalar_type(), "All inputs to layer_norm must have the same dtype");

    auto batch_size = input.size(0);
    auto time_steps = input.size(1);
    auto hidden_dim = input.size(2);

    TORCH_CHECK(input.is_cuda(), "indexed_batched_layer_norm expects CUDA tensors for input");

    auto input_contig = input.contiguous();
    auto gamma_contig = gamma_cache.contiguous();
    auto beta_contig  = beta_cache.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != input.device()) {
        policy_contig = policy_contig.to(input.device());
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
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "Input must be FP16 for optimized linear");
    TORCH_CHECK(weight_cache.scalar_type() == torch::kFloat16, "Weight cache must be FP16");
    TORCH_CHECK(bias_cache.scalar_type() == torch::kFloat16, "Bias cache must be FP16");

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

    // Data preparation (no dtype conversion)
    auto input_contig = input.contiguous();
    auto weight_contig = weight_cache.contiguous();
    auto bias_contig = bias_cache.contiguous();

    auto policy_indices_long = policy_indices.to(torch::kLong).contiguous();
    if (policy_indices_long.device() != input.device()) {
        policy_indices_long = policy_indices_long.to(input.device());
    }

    auto t0 = Clock::now();
    auto weight_batched = weight_contig.index_select(0, policy_indices_long).contiguous();
    auto bias_batched = bias_contig.index_select(0, policy_indices_long).contiguous();
    auto t1 = Clock::now();
    timers["linear_index_select_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // Batched matmul
    auto gemm_t0 = Clock::now();
    auto weight_transposed = weight_batched.transpose(1, 2);
    auto bias_expanded = bias_batched.unsqueeze(1).expand({B, T, Out_Dim});
    auto output = torch::baddbmm(bias_expanded, input_contig, weight_transposed, /*beta=*/1.0, /*alpha=*/1.0);
    auto gemm_t1 = Clock::now();
    timers["linear_cublas_matmul_us"] += std::chrono::duration_cast<Microseconds>(gemm_t1 - gemm_t0);

    // Optional GELU epilogue
    if (epilogue == IndexedLinearEpilogue::BiasGELU) {
        auto epi_t0 = Clock::now();
        output = torch::gelu(output);
        auto epi_t1 = Clock::now();
        timers["linear_epilogue_us"] += std::chrono::duration_cast<Microseconds>(epi_t1 - epi_t0);
    }

    return output;
}

} // namespace kernels
} // namespace lb