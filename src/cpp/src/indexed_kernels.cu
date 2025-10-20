#include "indexed_kernels.h"

#include <ATen/AccumulateType.h> // For at::acc_type
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <mutex>

namespace {

// -----------------------------------------------------------------------------
// GELU Activation
// -----------------------------------------------------------------------------

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// -----------------------------------------------------------------------------
// Embedding Kernel
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// LayerNorm Kernel
// -----------------------------------------------------------------------------

template <typename acc_t>
__device__ acc_t warp_reduce_sum(acc_t val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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
    double eps) {
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
        shared_sum = block_sum / static_cast<acc_t>(hidden_dim);
    }
    __syncthreads();
    const acc_t mean = shared_sum;

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

    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        const acc_t norm = (static_cast<acc_t>(input_ptr[h]) - mean) * inv_std;
        const acc_t scaled = norm * static_cast<acc_t>(gamma_ptr[h]) + static_cast<acc_t>(beta_ptr[h]);
        output_ptr[h] = static_cast<scalar_t>(scaled);
    }
}

// -----------------------------------------------------------------------------
// Helper utilities for cuBLASLt
// -----------------------------------------------------------------------------

cublasComputeType_t compute_type_for(const at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return CUBLAS_COMPUTE_32F;
        case at::kHalf:
            return CUBLAS_COMPUTE_32F_FAST_16F;
        default:
            TORCH_CHECK(false, "Unsupported dtype for indexed_batched_linear");
    }
}

cudaDataType_t cuda_dtype_for(const at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return CUDA_R_32F;
        case at::kHalf:
            return CUDA_R_16F;
        default:
            TORCH_CHECK(false, "Unsupported dtype for indexed_batched_linear");
    }
}

// Convert cuBLAS status codes to readable strings for diagnostics.
static const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

struct MatmulDescriptors {
    cublasLtMatmulDesc_t op_desc{nullptr};
    cublasLtMatrixLayout_t layout_a{nullptr};
    cublasLtMatrixLayout_t layout_b{nullptr};
    cublasLtMatrixLayout_t layout_c{nullptr};

    ~MatmulDescriptors() {
        if (layout_c) cublasLtMatrixLayoutDestroy(layout_c);
        if (layout_b) cublasLtMatrixLayoutDestroy(layout_b);
        if (layout_a) cublasLtMatrixLayoutDestroy(layout_a);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    }
};

} // anonymous namespace

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

torch::Tensor indexed_batched_embedding(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices) {
    TORCH_CHECK(weight_cache.dim() == 3, "Embedding cache must be [W, vocab, hidden]");
    TORCH_CHECK(indices.dim() == 2, "Indices must be [B, T]");
    TORCH_CHECK(policy_indices.dim() == 1, "policy_indices must be [B]");

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

// GELU activation kernel for intermediate tensors
template<typename scalar_t>
__global__ void gelu_inplace_kernel(
    scalar_t* data,
    int64_t total_elements
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float x = static_cast<float>(data[idx]);
    const float gelu_val = gelu(x);
    data[idx] = static_cast<scalar_t>(gelu_val);
}

void grouped_ffn_gemm_forward(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* b2_ptrs,
    const uintptr_t* output_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim) {

    if (group_count == 0) return;

    // Get cuBLAS handle and CUDA stream
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();

    // We'll use FP16 for all operations (input is already FP16 from caller)
    const cudaDataType_t data_type = CUDA_R_16F;
    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16F;

    // Scalars for GEMM (alpha=1.0, beta=0.0 for first GEMM, beta=1.0 for bias add)
    const __half alpha_half = __float2half(1.0f);
    const __half beta_zero = __float2half(0.0f);
    const __half beta_one = __float2half(1.0f);

    // Allocate workspace for intermediate hidden states
    // We'll allocate a single large buffer and partition it across groups
    int64_t max_m = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        max_m = std::max(max_m, m_sizes[i]);
    }

    // Allocate intermediate buffer: [max_m * ffn_dim] elements
    auto intermediate_buffer = torch::empty(
        {max_m * ffn_dim},
        torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)
    );
    auto* intermediate_ptr = reinterpret_cast<at::Half*>(intermediate_buffer.data_ptr());

    // Process each group
    for (int64_t group_idx = 0; group_idx < group_count; ++group_idx) {
        const int64_t M = m_sizes[group_idx];
        if (M == 0) continue;

        auto* input_ptr = reinterpret_cast<const at::Half*>(input_ptrs[group_idx]);
        auto* w1_ptr = reinterpret_cast<const at::Half*>(w1_ptrs[group_idx]);
        auto* b1_ptr = reinterpret_cast<const at::Half*>(b1_ptrs[group_idx]);
        auto* w2_ptr = reinterpret_cast<const at::Half*>(w2_ptrs[group_idx]);
        auto* b2_ptr = reinterpret_cast<const at::Half*>(b2_ptrs[group_idx]);
        auto* output_ptr = reinterpret_cast<at::Half*>(output_ptrs[group_idx]);

        // ====================================================================
        // First GEMM: hidden = input @ w1.T
        // input: [M, hidden_dim]
        // w1: [ffn_dim, hidden_dim] (stored as row-major, treat as transposed)
        // hidden: [M, ffn_dim]
        // ====================================================================

        // Using cublasSgemmEx-like operation via cublasGemmEx
        // C = alpha * A @ B + beta * C
        // We want: hidden = input @ w1.T
        // A = input [M, K=hidden_dim], op(A) = N
        // B = w1 [N=ffn_dim, K=hidden_dim], op(B) = T (transpose)
        // C = hidden [M, N=ffn_dim]

        auto status1 = cublasGemmEx(
            handle,
            CUBLAS_OP_T,           // op(B) = transpose w1
            CUBLAS_OP_N,           // op(A) = no-transpose input
            ffn_dim,               // M (rows of op(B)) = ffn_dim
            M,                     // N (cols of op(A)) = M (batch tokens)
            hidden_dim,            // K (common dimension) = hidden_dim
            &alpha_half,           // alpha
            w1_ptr,                // B: [ffn_dim, hidden_dim]
            data_type,
            hidden_dim,            // ldb (leading dim of B before transpose)
            input_ptr,             // A: [M, hidden_dim]
            data_type,
            hidden_dim,            // lda
            &beta_zero,            // beta = 0 (overwrite)
            intermediate_ptr,      // C: [M, ffn_dim]
            data_type,
            ffn_dim,               // ldc
            compute_type,
            CUBLAS_GEMM_DEFAULT
        );

        TORCH_CHECK(status1 == CUBLAS_STATUS_SUCCESS,
            "grouped_ffn_gemm_forward: First GEMM failed with cuBLAS status ",
            cublas_status_to_string(status1));

        // ====================================================================
        // Add bias b1 to each row
        // ====================================================================
        // b1: [ffn_dim]
        // We broadcast-add b1 to each of the M rows

        // Simple approach: use a kernel to add bias
        // Alternative: use cublas<t>axpy in a loop, but a custom kernel is cleaner

        auto b1_tensor = torch::from_blob(
            const_cast<at::Half*>(b1_ptr),
            {ffn_dim},
            torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)
        );
        auto intermediate_tensor = torch::from_blob(
            intermediate_ptr,
            {M, ffn_dim},
            torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)
        );

        // Broadcast add: intermediate_tensor += b1_tensor
        intermediate_tensor.add_(b1_tensor);

        // ====================================================================
        // Apply GELU activation in-place
        // ====================================================================

        const int64_t total_elements = M * ffn_dim;
        const int threads = 256;
        const int blocks = (total_elements + threads - 1) / threads;

        gelu_inplace_kernel<at::Half><<<blocks, threads, 0, stream>>>(
            intermediate_ptr,
            total_elements
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // ====================================================================
        // Second GEMM: output = hidden @ w2.T
        // hidden: [M, ffn_dim]
        // w2: [hidden_dim, ffn_dim] (stored as row-major, treat as transposed)
        // output: [M, hidden_dim]
        // ====================================================================

        auto status2 = cublasGemmEx(
            handle,
            CUBLAS_OP_T,           // op(B) = transpose w2
            CUBLAS_OP_N,           // op(A) = no-transpose hidden
            hidden_dim,            // M (rows of op(B)) = hidden_dim
            M,                     // N (cols of op(A)) = M (batch tokens)
            ffn_dim,               // K (common dimension) = ffn_dim
            &alpha_half,           // alpha
            w2_ptr,                // B: [hidden_dim, ffn_dim]
            data_type,
            ffn_dim,               // ldb (leading dim of B before transpose)
            intermediate_ptr,      // A: [M, ffn_dim]
            data_type,
            ffn_dim,               // lda
            &beta_zero,            // beta = 0 (overwrite)
            output_ptr,            // C: [M, hidden_dim]
            data_type,
            hidden_dim,            // ldc
            compute_type,
            CUBLAS_GEMM_DEFAULT
        );

        TORCH_CHECK(status2 == CUBLAS_STATUS_SUCCESS,
            "grouped_ffn_gemm_forward: Second GEMM failed with cuBLAS status ",
            cublas_status_to_string(status2));

        // ====================================================================
        // Add bias b2 to each row
        // ====================================================================

        auto b2_tensor = torch::from_blob(
            const_cast<at::Half*>(b2_ptr),
            {hidden_dim},
            torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)
        );
        auto output_tensor = torch::from_blob(
            output_ptr,
            {M, hidden_dim},
            torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)
        );

        // Broadcast add: output_tensor += b2_tensor
        output_tensor.add_(b2_tensor);
    }
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

    auto batch_size = input.size(0);
    auto time_steps = input.size(1);
    auto hidden_dim = input.size(2);

    TORCH_CHECK(input.is_cuda(), "indexed_batched_layer_norm expects CUDA tensors for input");

    auto input_contig = input.contiguous();
    auto gamma_contig = gamma_cache.contiguous();
    auto beta_contig = beta_cache.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != input.device()) {
        policy_contig = policy_contig.to(input.device());
    }

    auto output = torch::empty_like(input_contig, input_contig.options());

    const int threads = hidden_dim >= 256 ? 256 : 128;
    const int64_t total_tokens = batch_size * time_steps;

    c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

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
                eps);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

torch::Tensor indexed_batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, Microseconds>& timers,
    IndexedLinearEpilogue epilogue) {
    TORCH_CHECK(weight_cache.dim() == 3, "weight cache must be [W, out, in]");
    TORCH_CHECK(bias_cache.dim() == 2, "bias cache must be [W, out]");
    TORCH_CHECK(input.dim() == 3, "input must be [B, T, in]");

    const int64_t batch_size = input.size(0);
    const int64_t time_steps = input.size(1);
    const int64_t in_dim = input.size(2);
    const int64_t out_dim = weight_cache.size(1);

    TORCH_CHECK(weight_cache.size(2) == in_dim, "Input dim mismatch");

    if (!input.is_cuda()) {
        auto policy_cpu = policy_indices.cpu();
        auto t0 = Clock::now();
        auto weight = weight_cache.index_select(0, policy_cpu);
        auto bias = bias_cache.index_select(0, policy_cpu);
        auto t1 = Clock::now();
        timers["linear_index_select_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);
        auto x = input.to(weight.scalar_type());
        auto t2 = Clock::now();
        auto result = torch::matmul(x, weight.transpose(-1, -2)) + bias.unsqueeze(1);
        auto t3 = Clock::now();
        timers["linear_cublas_matmul_us"] += std::chrono::duration_cast<Microseconds>(t3 - t2);
        if (epilogue == IndexedLinearEpilogue::BiasGELU) {
            result = torch::gelu(result);
        }
        return result;
    }

    TORCH_CHECK(weight_cache.device().is_cuda(), "weight cache must be CUDA when input is CUDA");
    TORCH_CHECK(bias_cache.device().is_cuda(), "bias cache must be CUDA when input is CUDA");

    c10::cuda::CUDAGuard guard(input.device());

    auto input_cast = input.to(weight_cache.scalar_type()).contiguous();
    auto weight_contig = weight_cache.contiguous();
    auto bias_contig = bias_cache.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != input.device()) {
        policy_contig = policy_contig.to(input.device());
    }
    auto t0 = Clock::now();
    auto weight_batched = weight_contig.index_select(0, policy_contig).contiguous(); // [B, out, in]
    auto bias_batched = bias_contig.index_select(0, policy_contig).contiguous();     // [B, out]
    torch::cuda::synchronize();
    auto t1 = Clock::now();
    timers["linear_index_select_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    auto output = torch::empty({batch_size, time_steps, out_dim}, input_cast.options());

    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype = input_cast.scalar_type();

    // Create descriptors with current dynamic dimensions
    MatmulDescriptors desc{};
    const auto compute_type = compute_type_for(dtype);
    const auto data_type = cuda_dtype_for(dtype);
    TORCH_CHECK(
        cublasLtMatmulDescCreate(&desc.op_desc, compute_type, data_type) == CUBLAS_STATUS_SUCCESS,
        "cuBLASLt error");
    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_T; // B is [N, K] but treated as transposed
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(
        desc.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(
        desc.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    // Scalars alpha/beta are host pointers
    cublasLtPointerMode_t ab_pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(
        desc.op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &ab_pointer_mode, sizeof(ab_pointer_mode)) == CUBLAS_STATUS_SUCCESS,
        "cuBLASLt error");

    const int64_t M = time_steps;
    const int64_t K = in_dim;
    const int64_t N = out_dim;
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&desc.layout_a, data_type, M, K, K) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&desc.layout_b, data_type, N, K, K) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&desc.layout_c, data_type, M, N, N) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    // Configure strided batched A/B/C (base pointers + batch stride)
    const int32_t batch_count = static_cast<int32_t>(batch_size);
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    long long stride_a = static_cast<long long>(M) * static_cast<long long>(K);
    long long stride_b = static_cast<long long>(N) * static_cast<long long>(K);
    long long stride_c = static_cast<long long>(M) * static_cast<long long>(N);
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(desc.layout_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");

    size_t workspace_size = 1 << 22; // 4MB
    auto workspace = torch::empty({static_cast<long>(workspace_size)}, input_cast.options().dtype(torch::kByte));

    cublasLtMatmulPreference_t preference;
    TORCH_CHECK(cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");
    TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");

    // Use default epilogue; apply bias/GELU after GEMM for clarity/compat
    cublasLtEpilogue_t epilogue_attr = CUBLASLT_EPILOGUE_DEFAULT;
    {
        auto st = cublasLtMatmulDescSetAttribute(
            desc.op_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epilogue_attr,
            sizeof(epilogue_attr));
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
            "cublasLtMatmulDescSetAttribute(EPILOGUE) failed: ", cublas_status_to_string(st),
            " (", static_cast<int>(st), ")");
    }

    cublasLtMatmulHeuristicResult_t heuristic;
    int returned_results = 0;
    {
        auto st = cublasLtMatmulAlgoGetHeuristic(
            handle,
            desc.op_desc,
            desc.layout_a,
            desc.layout_b,
            desc.layout_c,
            desc.layout_c,
            preference,
            1,
            &heuristic,
            &returned_results);

        // Fallback to torch::matmul if cuBLASLt can't find a heuristic
        // This typically happens with small, misaligned dimensions (e.g., K=9 with FP16)
        if (st != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
            cublasLtMatmulPreferenceDestroy(preference);

            // Use PyTorch's matmul which handles any matrix size
            auto fb0 = Clock::now();
            auto result = torch::matmul(input_cast, weight_batched.transpose(-1, -2));
            result.add_(bias_batched.unsqueeze(1));
            if (epilogue == IndexedLinearEpilogue::BiasGELU) {
                result = torch::gelu(result);
            }
            if (input_cast.is_cuda()) { torch::cuda::synchronize(); }
            auto fb1 = Clock::now();
            timers["linear_matmul_fallback_us"] += std::chrono::duration_cast<Microseconds>(fb1 - fb0);
            return result;
        }
    }

    float alpha_float = 1.0f;
    float beta_float = 0.0f;
    const void* alpha_ptr = &alpha_float;
    const void* beta_ptr = &beta_float;

    // Strided-batched matmul with base pointers
    auto gemm_t0 = Clock::now();
    auto matmul_status = cublasLtMatmul(
        handle,
        desc.op_desc,
        alpha_ptr,
        input_cast.data_ptr(),        // A base: [B, M, K]
        desc.layout_a,
        weight_batched.data_ptr(),    // B base: [B, N, K]
        desc.layout_b,
        beta_ptr,
        output.data_ptr(),            // C base: [B, M, N]
        desc.layout_c,
        output.data_ptr(),
        desc.layout_c,
        &heuristic.algo,
        workspace_size ? workspace.data_ptr() : nullptr,
        workspace_size,
        stream);
    if (input_cast.is_cuda()) { torch::cuda::synchronize(); }
    auto gemm_t1 = Clock::now();
    timers["linear_cublas_matmul_us"] += std::chrono::duration_cast<Microseconds>(gemm_t1 - gemm_t0);
    const char* dtype_str = (dtype == at::kHalf) ? "f16" : (dtype == at::kFloat) ? "f32" : "other";
    TORCH_CHECK(
        matmul_status == CUBLAS_STATUS_SUCCESS,
        "cublasLtMatmul failed: ", cublas_status_to_string(matmul_status),
        " (", static_cast<int>(matmul_status), ")",
        "; M=", time_steps,
        ", N=", out_dim,
        ", K=", in_dim,
        ", batch=", batch_size,
        ", dtype=", dtype_str,
        ", epilogue=DEFAULT",
        ", workspace=", workspace_size
    );

    TORCH_CHECK(cublasLtMatmulPreferenceDestroy(preference) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error");

    // Apply bias and optional GELU epilogue per batch
    auto epi_t0 = Clock::now();
    output.add_(bias_batched.unsqueeze(1));
    if (epilogue == IndexedLinearEpilogue::BiasGELU) {
        output = torch::gelu(output);
    }
    if (output.is_cuda()) { torch::cuda::synchronize(); }
    auto epi_t1 = Clock::now();
    timers["linear_epilogue_us"] += std::chrono::duration_cast<Microseconds>(epi_t1 - epi_t0);

    return output;
}

