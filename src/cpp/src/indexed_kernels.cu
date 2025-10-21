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
#include <iostream>
// cuBLASLt debug dump (CUDA 12.9-compatible)
// Put this in a .cu/.cpp compiled with the same includes as your Lt code.

#include <cstdio>
#include <cinttypes>
#include <iostream>

static const char* dtypeStr(cudaDataType_t t) {
    switch (t) {
    case CUDA_R_32F: return "CUDA_R_32F";
    case CUDA_R_16F: return "CUDA_R_16F";
    case CUDA_R_16BF: return "CUDA_R_16BF";
    case CUDA_R_64F: return "CUDA_R_64F";
    case CUDA_R_8I:  return "CUDA_R_8I";
    case CUDA_R_8U:  return "CUDA_R_8U";
    default:         return "UNKNOWN";
    }
}

static const char* computeStr(cublasComputeType_t c) {
    switch (c) {
    case CUBLAS_COMPUTE_32F:           return "CUBLAS_COMPUTE_32F";
    case CUBLAS_COMPUTE_32F_FAST_16F:  return "CUBLAS_COMPUTE_32F_FAST_16F";
    case CUBLAS_COMPUTE_32F_FAST_TF32: return "CUBLAS_COMPUTE_32F_FAST_TF32";
    case CUBLAS_COMPUTE_16F:           return "CUBLAS_COMPUTE_16F";
    case CUBLAS_COMPUTE_64F:           return "CUBLAS_COMPUTE_64F";
    default:                           return "UNKNOWN";
    }
}

static const char* orderStr(cublasLtOrder_t o) {
    switch (o) {
    case CUBLASLT_ORDER_ROW:          return "ROW";
    case CUBLASLT_ORDER_COL:          return "COL";
    case CUBLASLT_ORDER_COL32:        return "COL32";
    case CUBLASLT_ORDER_COL4_4R2_8C:  return "COL4_4R2_8C";
    case CUBLASLT_ORDER_COL32_2R_4R4: return "COL32_2R_4R4";
    default:                          return "UNKNOWN";
    }
}

static const char* transStr(cublasOperation_t t) {
    switch (t) {
    case CUBLAS_OP_N: return "N";
    case CUBLAS_OP_T: return "T";
    case CUBLAS_OP_C: return "C";
    default:          return "?";
    }
}

static const char* epilogueStr(cublasLtEpilogue_t e) {
    switch (e) {
    case CUBLASLT_EPILOGUE_DEFAULT:         return "DEFAULT";
    case CUBLASLT_EPILOGUE_RELU:            return "RELU";
    case CUBLASLT_EPILOGUE_RELU_AUX:        return "RELU_AUX";
    case CUBLASLT_EPILOGUE_GELU:            return "GELU";
    case CUBLASLT_EPILOGUE_GELU_AUX:        return "GELU_AUX";
    case CUBLASLT_EPILOGUE_BIAS:            return "BIAS";
    case CUBLASLT_EPILOGUE_RELU_BIAS:       return "RELU_BIAS";
    case CUBLASLT_EPILOGUE_RELU_AUX_BIAS:   return "RELU_AUX_BIAS";
    case CUBLASLT_EPILOGUE_GELU_BIAS:       return "GELU_BIAS";
    case CUBLASLT_EPILOGUE_GELU_AUX_BIAS:   return "GELU_AUX_BIAS";
    default:                                 return "UNKNOWN";
    }
}

// minimal attr getters (12.9 symbols only)
template <typename T>
static void getDescAttr(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a, T& out) {
    size_t sz = sizeof(T);
    cublasLtMatmulDescGetAttribute(d, a, &out, sz, &sz);
}
template <typename T>
static size_t getLayoutAttrSZ(cublasLtMatrixLayout_t lay, cublasLtMatrixLayoutAttribute_t attr, T& out) {
    size_t written = 0;
    auto st = cublasLtMatrixLayoutGetAttribute(lay, attr, &out, sizeof(T), &written);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cout << "[LT] GetAttribute failed attr=" << (int)attr << " st=" << (int)st << "\n";
    }
    return written;
}
template <typename T>
static void getPrefAttr(cublasLtMatmulPreference_t p, cublasLtMatmulPreferenceAttributes_t a, T& out) {
    size_t sz = sizeof(T);
    cublasLtMatmulPreferenceGetAttribute(p, a, &out, sz, &sz);
}

static void dumpLayout(const char* tag, cublasLtMatrixLayout_t lay) {
    int64_t rows=0, cols=0, ld=0;
    int      batch=1;           // 32-bit
    long long stride_bytes=0;   // 64-bit
    cudaDataType_t dtype = CUDA_R_32F;
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;

    getLayoutAttrSZ(lay, CUBLASLT_MATRIX_LAYOUT_TYPE, dtype);
    getLayoutAttrSZ(lay, CUBLASLT_MATRIX_LAYOUT_ROWS, rows);
    getLayoutAttrSZ(lay, CUBLASLT_MATRIX_LAYOUT_COLS, cols);
    getLayoutAttrSZ(lay, CUBLASLT_MATRIX_LAYOUT_LD,   ld);
    getLayoutAttrSZ(lay, CUBLASLT_MATRIX_LAYOUT_ORDER, order);
    size_t bc_written  = getLayoutAttrSZ(lay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch);
    size_t sb_written  = getLayoutAttrSZ(lay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_bytes);

    std::cout << "  [" << tag << "] dtype=" << dtypeStr(dtype)
              << " rows=" << rows << " cols=" << cols
              << " ld="   << ld   << " order=" << orderStr(order)
              << " batch=" << batch << " (bytes=" << bc_written << ")"
              << " stride_bytes=" << stride_bytes << " (bytes=" << sb_written << ")"
              << "\n";
}

static void dumpOpDesc(cublasLtMatmulDesc_t op) {
    cublasOperation_t ta=CUBLAS_OP_N, tb=CUBLAS_OP_N;
    cudaDataType_t scaleType = CUDA_R_32F;
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DEFAULT;

    getDescAttr(op, CUBLASLT_MATMUL_DESC_TRANSA, ta);
    getDescAttr(op, CUBLASLT_MATMUL_DESC_TRANSB, tb);
    getDescAttr(op, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, compute);
    getDescAttr(op, CUBLASLT_MATMUL_DESC_SCALE_TYPE,   scaleType);
    getDescAttr(op, CUBLASLT_MATMUL_DESC_EPILOGUE,     epi);

    std::cout << "  [op_desc] transA=" << transStr(ta)
              << " transB=" << transStr(tb)
              << " compute=" << computeStr(compute)
              << " scaleType=" << dtypeStr(scaleType)
              << " epilogue=" << epilogueStr(epi) << "\n";

    // Bias pointer/type (12.9 has these)
    void* biasPtr = nullptr;
    size_t got = 0;
    if (cublasLtMatmulDescGetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &biasPtr, sizeof(biasPtr), &got) == CUBLAS_STATUS_SUCCESS && got == sizeof(biasPtr)) {
        cudaDataType_t biasType = CUDA_R_32F;
        cublasLtMatmulDescGetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                       &biasType, sizeof(biasType), &got);
        std::cout << "  [op_desc] bias_ptr=" << biasPtr
                  << " bias_type=" << dtypeStr(biasType) << "\n";
    }
}

static void dumpPreference(cublasLtMatmulPreference_t pref) {
    size_t maxWs=0;
    getPrefAttr(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, maxWs);
    std::cout << "  [preference] max_workspace=" << maxWs << "\n";
}

static void dumpHeuristic(const cublasLtMatmulHeuristicResult_t& h) {
    std::cout << "  [heuristic] state=" << (int)h.state
              << " wavesCount=" << h.wavesCount
              << " workspaceSize=" << h.workspaceSize << "\n";
}

static void dumpLtConfigLt(
    cublasLtHandle_t lt_handle,                    // lt handle (unused here; kept for signature parity)
    cublasLtMatmulDesc_t op_desc,
    cublasLtMatrixLayout_t a,
    cublasLtMatrixLayout_t b,
    cublasLtMatrixLayout_t c,
    cublasLtMatmulPreference_t pref)
{
    (void)lt_handle;
    int drv=0, rt=0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    auto vers=[&](int v){ return std::to_string(v/1000)+"."+std::to_string((v%1000)/10); };
    std::cout << "===== cuBLASLt Matmul Config (driver " << vers(drv)
            << ", runtime " << vers(rt) << ") =====\n";
    dumpOpDesc(op_desc);
    dumpLayout("A", a);
    dumpLayout("B", b);
    dumpLayout("C", c);
    dumpPreference(pref);
    std::cout << "==============================================\n";
}


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

    if (group_count == 0) {
        std::cout << "[DEBUG FFN GEMM] group_count=0, returning early" << std::endl;
        return;
    }
    std::cout << "test!!!!!!!!!!!!!!!!!!";
    std::cout << "[DEBUG FFN GEMM] Starting grouped FFN: group_count=" << group_count
              << ", hidden_dim=" << hidden_dim << ", ffn_dim=" << ffn_dim << std::endl;

    // Debug: print first few group sizes
    for (int64_t i = 0; i < std::min(group_count, 3L); ++i) {
        std::cout << "[DEBUG FFN GEMM] Group " << i << ": m_size=" << m_sizes[i]
                  << ", input_ptr=0x" << std::hex << input_ptrs[i]
                  << ", w1_ptr=0x" << w1_ptrs[i] << std::dec << std::endl;
    }

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

        // Validation: sample first weight value to verify pointer is valid
        static bool validated_weights = false;
        if (!validated_weights && group_idx == 0) {
            auto w1_sample = torch::from_blob(
                const_cast<at::Half*>(w1_ptr),
                {ffn_dim, hidden_dim},
                torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)
            );
            at::Half val = w1_sample[0][0].item<at::Half>();
            fprintf(stderr, "[DEBUG GEMM] Group 0 w1[0][0] = %f (ptr=0x%lx)\n",
                    static_cast<float>(val), reinterpret_cast<uintptr_t>(w1_ptr));
            validated_weights = true;
        }

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
    // Validate policy indices are within bounds before index_select
    auto policy_max = policy_contig.max();
    auto policy_min = policy_contig.min();
    
    // CLAMPING: Ensure indices are within the valid range of the weight_cache.
    // This is a safeguard against invalid policy IDs being passed from the Python side.
    auto clamped_policy_indices = torch::clamp(policy_contig, 0, weight_contig.size(0) - 1);

    if (!torch::equal(policy_contig, clamped_policy_indices)) {
        std::cerr << "[WARN] Clamping policy indices. Original min=" << policy_min.item<int64_t>()
                  << ", max=" << policy_max.item<int64_t>()
                  << ". Clamped to [0, " << weight_contig.size(0) - 1 << "]" << std::endl;
    }

    TORCH_CHECK(policy_min.item<int64_t>() >= 0,
        "indexed_batched_linear: policy indices contain negative values: min=", policy_min.item<int64_t>());
    TORCH_CHECK(policy_max.item<int64_t>() < weight_contig.size(0),
        "indexed_batched_linear: policy index out of bounds: max_index=", policy_max.item<int64_t>(),
        ", weight_cache_size=", weight_contig.size(0),
        " (batch_size=", batch_size, ", time_steps=", time_steps, ")");

    // DEBUG: Print policy indices and weight cache size
    std::cout << "[DBG idx_select] weight_cache_size=" << weight_contig.size(0)
              << ", policy_indices_size=" << policy_contig.size(0)
              << ", min=" << policy_min.item<int64_t>()
              << ", max=" << policy_max.item<int64_t>()
              << ", weight_cache_dims=" << weight_contig.dim()
              << ", weight_cache_shape=[";
    for (int64_t i = 0; i < weight_contig.dim(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << weight_contig.size(i);
    }
    std::cout << "]" << std::endl;
    if (policy_contig.size(0) <= 20) {
        std::cout << "[DBG idx_select] policy_indices=[";
        auto policy_cpu = policy_contig.cpu();
        auto policy_accessor = policy_cpu.accessor<int64_t, 1>();
        for (int64_t i = 0; i < policy_contig.size(0); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << policy_accessor[i];
        }
        std::cout << "]" << std::endl;
    }

    auto t0 = Clock::now();
    auto weight_batched = weight_contig.index_select(0, clamped_policy_indices).contiguous(); // [B, out, in]
    auto bias_batched = bias_contig.index_select(0, clamped_policy_indices).contiguous();     // [B, out]
    torch::cuda::synchronize();
    auto t1 = Clock::now();
    timers["linear_index_select_us"] += std::chrono::duration_cast<Microseconds>(t1 - t0);

    auto output = torch::empty({batch_size, time_steps, out_dim}, input_cast.options());

    // CRITICAL VALIDATION: Check that index_select didn't produce invalid tensors
    TORCH_CHECK(weight_batched.is_contiguous(),
        "weight_batched must be contiguous after index_select");
    TORCH_CHECK(weight_batched.size(0) == batch_size,
        "weight_batched size mismatch: expected ", batch_size, " got ", weight_batched.size(0));
    TORCH_CHECK(weight_batched.size(1) == out_dim && weight_batched.size(2) == in_dim,
        "weight_batched shape mismatch: expected [", batch_size, ",", out_dim, ",", in_dim, "]",
        " got [", weight_batched.size(0), ",", weight_batched.size(1), ",", weight_batched.size(2), "]");
    TORCH_CHECK(input_cast.is_contiguous(),
        "input_cast must be contiguous, got strides: ", input_cast.strides());
    TORCH_CHECK(input_cast.size(0) == batch_size && input_cast.size(1) == time_steps && input_cast.size(2) == in_dim,
        "input_cast shape mismatch: expected [", batch_size, ",", time_steps, ",", in_dim, "]",
        " got [", input_cast.size(0), ",", input_cast.size(1), ",", input_cast.size(2), "]");

    // Debug: print tensor info before cuBLASLt
    std::cout << "[DBG Linear] batch=" << batch_size << ", time_steps=" << time_steps
              << ", in_dim=" << in_dim << ", out_dim=" << out_dim << std::endl;
    std::cout << "[DBG Linear] input_ptr=0x" << std::hex
              << reinterpret_cast<uintptr_t>(input_cast.data_ptr()) << std::dec
              << ", input_size=" << (input_cast.numel() * input_cast.element_size()) << " bytes" << std::endl;
    std::cout << "[DBG Linear] weight_batched_ptr=0x" << std::hex
              << reinterpret_cast<uintptr_t>(weight_batched.data_ptr()) << std::dec
              << ", size=" << (weight_batched.numel() * weight_batched.element_size()) << " bytes" << std::endl;

    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t raw_stream = stream.stream();
    const auto dtype = input_cast.scalar_type();
    std::cout << "[DBG] raw_stream=" << raw_stream << "\n";
    // Create descriptors with current dynamic dimensions
    MatmulDescriptors desc{};
    const auto compute_type = compute_type_for(dtype);
    const auto data_type = cuda_dtype_for(dtype);
    TORCH_CHECK(
        cublasLtMatmulDescCreate(&desc.op_desc, compute_type, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS,
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
    // 1) Batch count: cuBLASLt expects a 32-bit int for this attribute on CUDA 12.x
    const int batch_count = static_cast<int>(batch_size);
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.layout_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) == CUBLAS_STATUS_SUCCESS, "A batch size cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.layout_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) == CUBLAS_STATUS_SUCCESS, "B batch size cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.layout_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) == CUBLAS_STATUS_SUCCESS, "C batch size cuBLASLt error");

    // 2) Strides must be **bytes**. Use the real tensor element size (handles FP16/BF16/FP32)
    const long long elem_bytes = static_cast<long long>(input_cast.element_size()); // 2 for FP16/BF16, 4 for FP32
    const long long stride_a_bytes = static_cast<long long>(M) * static_cast<long long>(K) * elem_bytes; // A: [M,K]
    const long long stride_b_bytes = static_cast<long long>(N) * static_cast<long long>(K) * elem_bytes; // B: [N,K] (transB=T)
    const long long stride_c_bytes = static_cast<long long>(M) * static_cast<long long>(N) * elem_bytes; // C: [M,N]

    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.layout_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_a_bytes, sizeof(stride_a_bytes)) == CUBLAS_STATUS_SUCCESS, "A stride cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.layout_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_b_bytes, sizeof(stride_b_bytes)) == CUBLAS_STATUS_SUCCESS, "B stride cuBLASLt error");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc.layout_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
    &stride_c_bytes, sizeof(stride_c_bytes)) == CUBLAS_STATUS_SUCCESS, "C stride cuBLASLt error");
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

    dumpLtConfigLt(handle, desc.op_desc, desc.layout_a, desc.layout_b, desc.layout_c, preference);

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;

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

    std::cout << "cublasLtMatmulAlgoGetHeuristic status=" << (int)st
            << " returned=" << returned_results << "\n";

    if (st == CUBLAS_STATUS_SUCCESS && returned_results > 0) {
        dumpHeuristic(heuristic);
    } else {
        std::cerr << "No heuristic. Check: ORDER vs LD, batch stride BYTES, compute/scale types, epilogue/bias.\n";
    }

    // NEW: No fallback. If a heuristic is not found, it's a fatal error.
    TORCH_CHECK(
        st == CUBLAS_STATUS_SUCCESS && returned_results > 0,
        "indexed_batched_linear: cublasLtMatmulAlgoGetHeuristic failed to find a valid algorithm. ",
        "This is a fatal error.",
        "M=", time_steps, ", N=", out_dim, ", K=", in_dim, ", batch=", batch_size,
        ", cuBLAS status: ", cublas_status_to_string(st)
    );

    // Fast path is now the ONLY path.
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
        raw_stream);
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
