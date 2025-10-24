#include "indexed_kernels.h"

#include <ATen/AccumulateType.h> // For at::acc_type
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <mutex>
#include <iostream>
#include <cstdlib>
// cuBLASLt debug dump
// Put this in a .cu/.cpp compiled with the same includes as your Lt code.

#include <cstdio>
#include <cinttypes>
#include <iostream>

// Simple RAII helper for cleanup actions
template<typename F>
struct ScopeGuard {
    F func;
    bool active;
    ScopeGuard(F f) : func(std::move(f)), active(true) {}
    ~ScopeGuard() { if (active) func(); }
    void dismiss() { active = false; }
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
};

template<typename F>
ScopeGuard<F> make_scope_guard(F f) {
    return ScopeGuard<F>(std::move(f));
}

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

// ----------------------------------------------------------------------------
// Debug gating helpers (runtime)
//  - Host-side verbose logs can be enabled via env vars:
//      LB_DEBUG_CUBLASLT=1   -> cuBLASLt config and heuristic dumps
//      LB_DEBUG_STATS=1      -> tensor stats/validation prints
//      LB_ENABLE_KERNEL_DEBUG=1 -> enable in-kernel debug prints
// ----------------------------------------------------------------------------
static inline bool lb_env_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    // Treat '0', 'false', 'no' as disabled; anything else enabled
    if (v[0] == '0' || v[0] == 'F' || v[0] == 'f' || v[0] == 'N' || v[0] == 'n') return false;
    return true;
}
static bool LB_DBG_CUBLASLT() { static int flag = -1; if (flag < 0) flag = lb_env_enabled("LB_DEBUG_CUBLASLT"); return flag != 0; }
static bool LB_DBG_STATS()    { static int flag = -1; if (flag < 0) flag = lb_env_enabled("LB_DEBUG_STATS");    return flag != 0; }
static bool LB_MOE_GUARD()    { static int flag = -1; if (flag < 0) flag = lb_env_enabled("LB_MOE_GUARD");    return flag != 0; }
static bool LB_MOE_LOG_GUARD(){ static int flag = -1; if (flag < 0) flag = lb_env_enabled("LB_MOE_LOG_GUARD"); return flag != 0; }
static bool LB_KERNEL_DEBUG_ENABLED() { static int flag = -1; if (flag < 0) flag = lb_env_enabled("LB_ENABLE_KERNEL_DEBUG") ? 1 : 0; return flag != 0; }
static int lb_env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    return static_cast<int>(parsed);
}
static int LB_MOE_VALIDATE_GROUPS() { static int value = lb_env_int("LB_MOE_VALIDATE_GROUPS", 0); return value; }
static int LB_MOE_LOG_GEMM_GROUPS() { static int value = lb_env_int("LB_MOE_LOG_GEMM", 0); return value; }

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

// minimal attr getters
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

    // Bias pointer/type
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
    if (!LB_DBG_CUBLASLT()) return;  // gated
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
using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;
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
    double eps,
    bool debug_enabled) {
    // Debug printing in kernel (runtime gated via host flag)
    if (debug_enabled && threadIdx.x == 0 && blockIdx.x == 0) {
        printf("LayerNorm Kernel Debug:\n");
        printf("Input ptr: %p\n", (void*)input);
        printf("Batch size: %ld, Time steps: %ld, Hidden dim: %ld\n",
               batch_size, time_steps, hidden_dim);
        printf("Epsilon: %e\n", eps);

        // Print first few input values
        printf("First few input values:\n");
        for (int i = 0; i < min(10, (int)hidden_dim); i++) {
            printf("%f ", (float)input[i]);
        }
        printf("\n");
    }
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
        // Store the raw sum for the block; divide once when computing mean
        shared_sum = block_sum;
    }
    __syncthreads();
    const acc_t mean = shared_sum / static_cast<acc_t>(hidden_dim);

    // Enhanced debug: Check input values and print statistics when enabled
    if (debug_enabled && blockIdx.x < 5 && threadIdx.x == 0) {
        acc_t max_abs_val = 0.0f;
        acc_t min_val = 1e9;
        acc_t max_val = -1e9;
        bool has_nan = false;
        bool has_inf = false;

        for (int64_t h = 0; h < hidden_dim; h++) {
            acc_t val = static_cast<acc_t>(input_ptr[h]);
            if (isnan(val)) has_nan = true;
            if (isinf(val)) has_inf = true;
            max_abs_val = max(max_abs_val, fabs(val));
            min_val = min(min_val, val);
            max_val = max(max_val, val);
        }

        printf("Token %d stats:\n", (int)blockIdx.x);
        printf("  Range: [%f, %f]\n", (float)min_val, (float)max_val);
        printf("  Max abs value: %f\n", (float)max_abs_val);
        printf("  Mean: %f\n", (float)mean);
        printf("  Has NaN: %s\n", has_nan ? "YES" : "NO");
        printf("  Has Inf: %s\n", has_inf ? "YES" : "NO");
    }

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

    // Debug: Print variance and inv_std for first few tokens
    if (debug_enabled && blockIdx.x < 5 && threadIdx.x == 0) {
        printf("Token %d - Variance: %f, Inv_std: %f\n",
               (int)blockIdx.x, (float)shared_var, (float)inv_std);
    }

    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        const acc_t norm = (static_cast<acc_t>(input_ptr[h]) - mean) * inv_std;
        const acc_t scaled = norm * static_cast<acc_t>(gamma_ptr[h]) + static_cast<acc_t>(beta_ptr[h]);
        output_ptr[h] = static_cast<scalar_t>(scaled);
    }
}

// -----------------------------------------------------------------------------
// Helper utilities for cuBLASLt
// -----------------------------------------------------------------------------

cublasComputeType_t compute_type_for(const at::ScalarType /*dtype*/) {
    // Force FP32 compute for numerical stability and consistency
    return CUBLAS_COMPUTE_32F;
}

cudaDataType_t cuda_dtype_for(const at::ScalarType /*dtype*/) {
    // Use FP32 tensor data type for GEMM inputs/outputs
    return CUDA_R_32F;
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
    // Debug: Check for NaN/Inf/negative values in indices (should be integer indices)
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

namespace {

__device__ inline float fast_gelu(float x) {
    const float kAlpha = 0.7978845608f;       // sqrt(2/pi)
    const float kGamma = 0.044715f;
    float x_cube = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kGamma * x_cube)));
}

__global__ void add_bias_gelu_and_cast_kernel(
    float* hidden,
    const __half* bias,
    __half* hidden_half,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;

    int64_t col = idx % cols;
    float val = hidden[idx] + __half2float(bias[col]);
    float gelu = fast_gelu(val);
    hidden[idx] = gelu;
    hidden_half[idx] = __float2half(gelu);
}

__global__ void add_bias_and_cast_kernel(
    float* data,
    const __half* bias,
    __half* dst_half,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;

    int64_t col = idx % cols;
    float val = data[idx] + __half2float(bias[col]);
    data[idx] = val;
    dst_half[idx] = __float2half(val);
}

__global__ void cast_float_to_half_kernel(
    const float* src,
    __half* dst,
    int64_t total
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = __float2half(src[idx]);
}

inline void launch_add_bias_gelu_and_cast(
    float* hidden,
    const at::Half* bias,
    at::Half* hidden_half,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream
) {
    if (rows == 0 || cols == 0) {
        return;
    }
    const int threads = 256;
    const int64_t total = rows * cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    add_bias_gelu_and_cast_kernel<<<blocks, threads, 0, stream>>>(
        hidden,
        reinterpret_cast<const __half*>(bias),
        reinterpret_cast<__half*>(hidden_half),
        rows,
        cols
    );
}

inline void launch_add_bias_and_cast(
    float* data,
    const at::Half* bias,
    at::Half* dst_half,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream
) {
    if (rows == 0 || cols == 0) {
        return;
    }
    const int threads = 256;
    const int64_t total = rows * cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    add_bias_and_cast_kernel<<<blocks, threads, 0, stream>>>(
        data,
        reinterpret_cast<const __half*>(bias),
        reinterpret_cast<__half*>(dst_half),
        rows,
        cols
    );
}

inline void launch_cast_float_to_half(
    float* src,
    at::Half* dst,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream
) {
    if (rows == 0 || cols == 0) {
        return;
    }
    const int threads = 256;
    const int64_t total = rows * cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cast_float_to_half_kernel<<<blocks, threads, 0, stream>>>(
        src,
        reinterpret_cast<__half*>(dst),
        total
    );
}

} // anonymous namespace

void grouped_ffn_gemm_forward(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* b2_ptrs,
    const uintptr_t* output_ptrs,
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

    const bool guard_enabled = LB_MOE_GUARD();
    const bool log_guard = LB_MOE_LOG_GUARD();

    if (group_count == 0) {
        if (log_guard) {
            std::cerr << "[LB][MOE_GUARD] grouped_ffn_gemm_forward invoked with group_count=0" << std::endl;
        }
        return;
    }

    struct GuardLogEntry {
        int64_t group_idx;
        int64_t policy_id;
        int64_t expert_id;
        int64_t token_offset;
        int64_t m;
        const char* stage;
    };

    std::vector<GuardLogEntry> guard_logs;
    if (log_guard) {
        guard_logs.reserve(group_count * 2);
    }
    int64_t fallback_hidden_groups = 0;
    int64_t fallback_output_groups = 0;
    constexpr const char* kStageHidden = "hidden_gemm";
    constexpr const char* kStageOutput = "output_gemm";
    const int validate_setting = LB_MOE_VALIDATE_GROUPS();
    const bool validate_all_groups = validate_setting < 0;
    int validated_groups = 0;
    int planned_validations = 0;
    const int log_gemm_setting = LB_MOE_LOG_GEMM_GROUPS();
    const bool log_gemm_enabled = log_gemm_setting != 0;
    const bool log_gemm_all = log_gemm_setting < 0;
    int logged_gemm_groups = 0;

    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;

    int64_t max_m = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        max_m = std::max(max_m, m_sizes[i]);
    }

    struct GroupTask {
        int64_t group_idx;
        int64_t m;
        int64_t policy_id;
        int64_t expert_id;
        int64_t token_offset;
        const at::Half* input_ptr;
        const at::Half* w1_ptr;
        const at::Half* b1_ptr;
        const at::Half* w2_ptr;
        const at::Half* b2_ptr;
        at::Half* output_ptr;
        bool validate;
    };

    std::vector<GroupTask> tasks;
    tasks.reserve(group_count);

    for (int64_t group_idx = 0; group_idx < group_count; ++group_idx) {
        const int64_t M = m_sizes[group_idx];
        if (M == 0) {
            continue;
        }

        const int64_t policy_id = policy_ids[group_idx];
        const int64_t expert_id = expert_ids[group_idx];
        const int64_t token_offset = token_offsets ? token_offsets[group_idx] : -1;

        const at::Half* input_ptr = reinterpret_cast<const at::Half*>(input_ptrs[group_idx]);
        const at::Half* w1_ptr = reinterpret_cast<const at::Half*>(w1_ptrs[group_idx]);
        const at::Half* b1_ptr = reinterpret_cast<const at::Half*>(b1_ptrs[group_idx]);
        const at::Half* w2_ptr = reinterpret_cast<const at::Half*>(w2_ptrs[group_idx]);
        const at::Half* b2_ptr = reinterpret_cast<const at::Half*>(b2_ptrs[group_idx]);
        at::Half* output_ptr = reinterpret_cast<at::Half*>(output_ptrs[group_idx]);

        bool validate_this_group = (validate_setting != 0) && (validate_all_groups || planned_validations < validate_setting);
        if (validate_this_group) {
            ++planned_validations;
        }
        bool log_this_group = log_gemm_enabled && (log_gemm_all || logged_gemm_groups < std::abs(log_gemm_setting));
        if (log_this_group) {
            ++logged_gemm_groups;
            std::cerr << "[LB][MOE_GEMM] group=" << group_idx
                      << " policy=" << policy_id
                      << " expert=" << expert_id;
            if (token_offset >= 0) {
                std::cerr << " offset=" << token_offset;
            }
            std::cerr << " M=" << M
                      << " hidden_dim=" << hidden_dim
                      << " ffn_dim=" << ffn_dim
                      << std::endl;
            std::cerr << "    first_gemm: transa=T transb=N"
                      << " m=" << ffn_dim
                      << " n=" << M
                      << " k=" << hidden_dim
                      << " lda=" << hidden_dim
                      << " ldb=" << hidden_dim
                      << " ldc=" << ffn_dim
                      << std::endl;
            std::cerr << "    second_gemm: transa=T transb=N"
                      << " m=" << hidden_dim
                      << " n=" << M
                      << " k=" << ffn_dim
                      << " lda=" << ffn_dim
                      << " ldb=" << ffn_dim
                      << " ldc=" << hidden_dim
                      << std::endl;
            std::cerr << "    ptrs: input=" << static_cast<const void*>(input_ptr)
                      << " w1=" << static_cast<const void*>(w1_ptr)
                      << " b1=" << static_cast<const void*>(b1_ptr)
                      << " w2=" << static_cast<const void*>(w2_ptr)
                      << " b2=" << static_cast<const void*>(b2_ptr)
                      << " out=" << static_cast<void*>(output_ptr)
                      << std::endl;
        }

        tasks.push_back({group_idx, M, policy_id, expert_id, token_offset,
                         input_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, output_ptr,
                         validate_this_group});
    }

    if (tasks.empty()) {
        if (log_guard && guard_enabled) {
            std::cerr << "[LB][MOE_GUARD] grouped_ffn_gemm_forward received only empty groups" << std::endl;
        }
        return;
    }

    const int num_streams = static_cast<int>(std::min<int64_t>(4, std::max<int64_t>(1, static_cast<int64_t>(tasks.size()))));

    auto base_stream = at::cuda::getCurrentCUDAStream();
    std::vector<at::cuda::CUDAStream> streams;
    streams.reserve(num_streams);
    streams.push_back(base_stream);
    const int device_index = base_stream.device_index();
    for (int i = 1; i < num_streams; ++i) {
        streams.push_back(at::cuda::getStreamFromPool(false, device_index));
    }

    std::vector<cublasHandle_t> handles(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        TORCH_CHECK(cublasCreate(&handles[i]) == CUBLAS_STATUS_SUCCESS,
                    "grouped_ffn_gemm_forward: failed to create cuBLAS handle");
        TORCH_CHECK(cublasSetStream(handles[i], streams[i].stream()) == CUBLAS_STATUS_SUCCESS,
                    "grouped_ffn_gemm_forward: failed to set cuBLAS stream");
    #if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
        cublasSetMathMode(handles[i], CUBLAS_TENSOR_OP_MATH);
    #endif
    }
    auto handle_guard = make_scope_guard([&]() {
        for (auto& h : handles) {
            if (h) {
                cublasDestroy(h);
            }
        }
    });

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto half_options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

    auto intermediate_buffer = torch::empty({num_streams, max_m * ffn_dim}, float_options);
    auto intermediate_half_buffer = torch::empty({num_streams, max_m * ffn_dim}, half_options);
    auto output_buffer = torch::empty({num_streams, max_m * hidden_dim}, float_options);

    float* intermediate_base = intermediate_buffer.data_ptr<float>();
    at::Half* intermediate_half_base = intermediate_half_buffer.data_ptr<at::Half>();
    float* output_base = output_buffer.data_ptr<float>();

    std::vector<std::vector<const GroupTask*>> stream_tasks(num_streams);
    for (size_t i = 0; i < tasks.size(); ++i) {
        stream_tasks[i % num_streams].push_back(&tasks[i]);
    }

    std::mutex guard_mutex;

    for (int stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
        if (stream_tasks[stream_idx].empty()) {
            continue;
        }

        at::cuda::CUDAStreamGuard guard(streams[stream_idx]);
        auto handle = handles[stream_idx];
        float* intermediate_ptr = intermediate_base + static_cast<int64_t>(stream_idx) * max_m * ffn_dim;
        at::Half* intermediate_half_ptr = intermediate_half_base + static_cast<int64_t>(stream_idx) * max_m * ffn_dim;
        float* output_float_ptr = output_base + static_cast<int64_t>(stream_idx) * max_m * hidden_dim;

        for (const GroupTask* task_ptr : stream_tasks[stream_idx]) {
            const GroupTask& task = *task_ptr;
            const int64_t M = task.m;
            if (M == 0) {
                continue;
            }

            auto status1 = cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                ffn_dim,
                M,
                hidden_dim,
                &alpha_one,
                task.w1_ptr,
                CUDA_R_16F,
                hidden_dim,
                task.input_ptr,
                CUDA_R_16F,
                hidden_dim,
                &beta_zero,
                intermediate_ptr,
                CUDA_R_32F,
                ffn_dim,
                compute_type,
                CUBLAS_GEMM_DEFAULT
            );

            TORCH_CHECK(status1 == CUBLAS_STATUS_SUCCESS,
                        "grouped_ffn_gemm_forward: First GEMM failed with cuBLAS status ",
                        cublas_status_to_string(status1));

            launch_add_bias_gelu_and_cast(
                intermediate_ptr,
                task.b1_ptr,
                intermediate_half_ptr,
                M,
                ffn_dim,
                streams[stream_idx].stream());

            torch::Tensor intermediate_tensor;
            if (guard_enabled || task.validate) {
                intermediate_tensor = torch::from_blob(
                    intermediate_ptr,
                    {M, ffn_dim},
                    [](void*) {},
                    float_options
                );
            }

            if (guard_enabled) {
                auto finite_mask = intermediate_tensor.isfinite();
                bool all_finite = finite_mask.all().item<bool>();
                if (!all_finite) {
                    ++fallback_hidden_groups;
                    if (log_guard) {
                        std::lock_guard<std::mutex> lock(guard_mutex);
                        guard_logs.push_back({task.group_idx, task.policy_id, task.expert_id, task.token_offset, M, kStageHidden});
                    }
                    auto input_tensor = torch::from_blob(
                        const_cast<at::Half*>(task.input_ptr),
                        {M, hidden_dim},
                        [](void*) {},
                        half_options
                    );
                    auto w1_tensor = torch::from_blob(
                        const_cast<at::Half*>(task.w1_ptr),
                        {ffn_dim, hidden_dim},
                        [](void*) {},
                        half_options
                    );
                    auto b1_tensor = torch::from_blob(
                        const_cast<at::Half*>(task.b1_ptr),
                        {ffn_dim},
                        [](void*) {},
                        half_options
                    ).to(torch::kFloat32);
                    auto hidden_ref = torch::matmul(input_tensor.to(torch::kFloat32), w1_tensor.to(torch::kFloat32).transpose(0, 1)) + b1_tensor;
                    hidden_ref = torch::gelu(hidden_ref);
                    intermediate_tensor.copy_(hidden_ref);
                    launch_cast_float_to_half(intermediate_ptr, intermediate_half_ptr, M, ffn_dim, streams[stream_idx].stream());
                }
            }

            auto status2 = cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                hidden_dim,
                M,
                ffn_dim,
                &alpha_one,
                task.w2_ptr,
                CUDA_R_16F,
                ffn_dim,
                reinterpret_cast<const at::Half*>(intermediate_half_ptr),
                CUDA_R_16F,
                ffn_dim,
                &beta_zero,
                output_float_ptr,
                CUDA_R_32F,
                hidden_dim,
                compute_type,
                CUBLAS_GEMM_DEFAULT
            );

            TORCH_CHECK(status2 == CUBLAS_STATUS_SUCCESS,
                        "grouped_ffn_gemm_forward: Second GEMM failed with cuBLAS status ",
                        cublas_status_to_string(status2));

            launch_add_bias_and_cast(
                output_float_ptr,
                task.b2_ptr,
                task.output_ptr,
                M,
                hidden_dim,
                streams[stream_idx].stream());

            torch::Tensor output_tensor_float;
            if (guard_enabled || task.validate) {
                output_tensor_float = torch::from_blob(
                    output_float_ptr,
                    {M, hidden_dim},
                    [](void*) {},
                    float_options
                );
            }

            if (guard_enabled) {
                auto finite_mask2 = output_tensor_float.isfinite();
                bool all_finite2 = finite_mask2.all().item<bool>();
                if (!all_finite2) {
                    ++fallback_output_groups;
                    if (log_guard) {
                        std::lock_guard<std::mutex> lock(guard_mutex);
                        guard_logs.push_back({task.group_idx, task.policy_id, task.expert_id, task.token_offset, M, kStageOutput});
                    }
                    auto w2_tensor = torch::from_blob(
                        const_cast<at::Half*>(task.w2_ptr),
                        {hidden_dim, ffn_dim},
                        [](void*) {},
                        half_options
                    );
                    auto inter_f32 = torch::from_blob(
                        intermediate_ptr,
                        {M, ffn_dim},
                        [](void*) {},
                        float_options
                    );
                    auto b2_tensor = torch::from_blob(
                        const_cast<at::Half*>(task.b2_ptr),
                        {hidden_dim},
                        [](void*) {},
                        half_options
                    ).to(torch::kFloat32);
                    auto out_ref = torch::matmul(inter_f32, w2_tensor.to(torch::kFloat32).transpose(0, 1)) + b2_tensor;
                    output_tensor_float.copy_(out_ref);
                    launch_cast_float_to_half(output_float_ptr, task.output_ptr, M, hidden_dim, streams[stream_idx].stream());
                }
            }

            if (task.validate) {
                ++validated_groups;
                auto half_opts = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

                auto input_tensor = torch::from_blob(
                    const_cast<at::Half*>(task.input_ptr),
                    {M, hidden_dim},
                    [](void*) {},
                    half_opts
                );
                auto w1_tensor = torch::from_blob(
                    const_cast<at::Half*>(task.w1_ptr),
                    {ffn_dim, hidden_dim},
                    [](void*) {},
                    half_opts
                );
                auto b1_tensor = torch::from_blob(
                    const_cast<at::Half*>(task.b1_ptr),
                    {ffn_dim},
                    [](void*) {},
                    half_opts
                );
                auto w2_tensor = torch::from_blob(
                    const_cast<at::Half*>(task.w2_ptr),
                    {hidden_dim, ffn_dim},
                    [](void*) {},
                    half_opts
                );
                auto b2_tensor = torch::from_blob(
                    const_cast<at::Half*>(task.b2_ptr),
                    {hidden_dim},
                    [](void*) {},
                    half_opts
                );

                auto input_f32 = input_tensor.to(torch::kFloat32);
                auto w1_f32 = w1_tensor.to(torch::kFloat32);
                auto b1_f32 = b1_tensor.to(torch::kFloat32);
                auto w2_f32 = w2_tensor.to(torch::kFloat32);
                auto b2_f32 = b2_tensor.to(torch::kFloat32);

                auto hidden_ref = torch::matmul(input_f32, w1_f32.transpose(0, 1)) + b1_f32;
                hidden_ref = torch::gelu(hidden_ref);
                auto output_ref = torch::matmul(hidden_ref, w2_f32.transpose(0, 1)) + b2_f32;

                auto diff = (output_tensor_float - output_ref).abs();
                auto diff_flat = diff.view({-1});
                auto max_abs_idx_tensor = diff_flat.argmax();
                float max_abs = diff_flat.index({max_abs_idx_tensor}).item<float>();
                float mean_abs = diff.mean().item<float>();

                auto denom = output_ref.abs().clamp_min(1e-6);
                auto rel = diff / denom;
                auto rel_flat = rel.view({-1});
                float max_rel = rel_flat.index({rel_flat.argmax()}).item<float>();
                float mean_rel = rel.mean().item<float>();

                int64_t worst_index = max_abs_idx_tensor.item<int64_t>();
                if (hidden_dim > 0 && M > 0) {
                    int64_t worst_m = worst_index / hidden_dim;
                    int64_t worst_h = worst_index % hidden_dim;
                    float actual_val = output_tensor_float.index({worst_m, worst_h}).item<float>();
                    float ref_val = output_ref.index({worst_m, worst_h}).item<float>();
                    float diff_val = actual_val - ref_val;

                    std::cerr << "[LB][MOE_VALIDATE] group=" << task.group_idx
                              << " policy=" << task.policy_id
                              << " expert=" << task.expert_id;
                    if (task.token_offset >= 0) {
                        std::cerr << " offset=" << task.token_offset;
                    }
                    std::cerr << " M=" << M
                              << " max_abs=" << max_abs
                              << " mean_abs=" << mean_abs
                              << " max_rel=" << max_rel
                              << " mean_rel=" << mean_rel
                              << std::endl;

                    if (max_abs > 5e-4f || max_rel > 1.0f || std::isnan(max_abs) || std::isnan(max_rel)) {
                        std::cerr << "    worst_coord=(" << worst_m << "," << worst_h << ")"
                                  << " actual=" << actual_val
                                  << " ref=" << ref_val
                                  << " diff=" << diff_val
                                  << std::endl;
                    }
                } else {
                    std::cerr << "[LB][MOE_VALIDATE] group=" << task.group_idx
                              << " policy=" << task.policy_id
                              << " expert=" << task.expert_id
                              << " M=" << M
                              << " -- skipped coord breakdown due to degenerate dims"
                              << " max_abs=" << max_abs
                              << " mean_abs=" << mean_abs
                              << " max_rel=" << max_rel
                              << " mean_rel=" << mean_rel
                              << std::endl;
                }
            }
        }
    }

    for (int i = 1; i < num_streams; ++i) {
        streams[i].synchronize();
    }

    if (log_guard) {
        if (!guard_enabled) {
            std::cerr << "[LB][MOE_GUARD] guard disabled; no fallback checks executed (group_count="
                      << group_count << ")." << std::endl;
        } else if (guard_logs.empty()) {
            std::cerr << "[LB][MOE_GUARD] guard enabled; no fallbacks triggered (group_count="
                      << group_count << ")." << std::endl;
        } else {
            std::cerr << "[LB][MOE_GUARD] guard fallback summary: entries="
                      << guard_logs.size()
                      << " (hidden=" << fallback_hidden_groups
                      << ", output=" << fallback_output_groups
                      << ", group_count=" << group_count << ")" << std::endl;
            for (const auto& entry : guard_logs) {
                std::cerr << "  group=" << entry.group_idx
                          << " policy=" << entry.policy_id
                          << " expert=" << entry.expert_id;
                if (entry.token_offset >= 0) {
                    std::cerr << " offset=" << entry.token_offset;
                }
                std::cerr << " M=" << entry.m
                          << " stage=" << entry.stage
                          << std::endl;
            }
        }
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
    
    // Detailed input validation
    auto fp32_input = input.to(torch::kFloat32);
    auto abs_input = fp32_input.abs();
    auto max_val = abs_input.max().item<float>();
    auto mean_val = fp32_input.mean().item<float>();
    auto std_val = fp32_input.std().item<float>();
    
    if (LB_DBG_STATS()) {
        std::cerr << "\nLayerNorm Input Validation:\n"
                  << "Max abs value: " << max_val << "\n"
                  << "Mean: " << mean_val << "\n"
                  << "Std dev: " << std_val << "\n";
    }
              
    // Check for potential numerical issues
    if (max_val > 1e4) {
        std::cerr << "Warning: Large input values detected in LayerNorm\n";
    }
    if (std_val < 1e-7) {
        std::cerr << "Warning: Very small standard deviation in LayerNorm input\n";
    }
    
    // Enhanced debug: Check for NaN/Inf and print detailed stats
    auto check_tensor = [](const torch::Tensor& t, const char* name) {
        if (t.isnan().any().item<bool>()) {
            auto nan_mask = t.isnan();
            auto nan_count = nan_mask.sum().item<int64_t>();
            std::cerr << name << " contains " << nan_count << " NaN values" << std::endl;
            // Print first NaN position
            auto nan_indices = nan_mask.nonzero();
            if (nan_indices.size(0) > 0) {
                auto first_nan = nan_indices[0];
                std::cerr << "First NaN at index: ";
                for (int64_t i = 0; i < first_nan.size(0); ++i) {
                    std::cerr << first_nan[i].item<int64_t>() << " ";
                }
                std::cerr << std::endl;
            }
            TORCH_CHECK(false, name, " contains NaN values");
        }
        if (t.isinf().any().item<bool>()) {
            auto inf_mask = t.isinf();
            auto inf_count = inf_mask.sum().item<int64_t>();
            std::cerr << name << " contains " << inf_count << " Inf values" << std::endl;
            TORCH_CHECK(false, name, " contains Inf values");
        }
        if (LB_DBG_STATS()) {
            auto stats = t.to(torch::kFloat32);  // Convert for accurate stats
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
    // Align parameter dtype with input to avoid mixed-type kernel access
    if (gamma_t.scalar_type() != input.scalar_type()) gamma_t = gamma_t.to(input.scalar_type());
    if (beta_t.scalar_type()  != input.scalar_type())  beta_t = beta_t.to(input.scalar_type());
    auto gamma_contig = gamma_t.contiguous();
    auto beta_contig  = beta_t.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != input.device()) {
        policy_contig = policy_contig.to(input.device());
    }
    // Validate and clamp policy indices to valid expert/policy range
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
    const torch::Tensor& weight_cache,  // [W, out, in]
    const torch::Tensor& bias_cache,    // [W, out]
    const torch::Tensor& policy_indices,// [B]
    std::unordered_map<std::string, Microseconds>& timers,
    IndexedLinearEpilogue epilogue)
{
    // === 1. Robust Input Validation ===
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

    TORCH_CHECK(policy_indices.size(0) == B, "policy_indices batch size must match input, expected ", B, ", got ", policy_indices.size(0));
    TORCH_CHECK(weight_cache.size(2) == In_Dim, "weight_cache input dim must match input tensor, expected ", In_Dim, ", got ", weight_cache.size(2));
    TORCH_CHECK(bias_cache.size(0) == W && bias_cache.size(1) == Out_Dim, "bias_cache shape mismatch");

    // === 2. Data Preparation and Sanitization ===
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

    // === 3. Use PyTorch's optimized batched matmul ===
    // Reshape for batched matmul: [B, T, In_Dim] @ [B, In_Dim, Out_Dim] -> [B, T, Out_Dim]
    // weight_batched is [B, Out_Dim, In_Dim], so we need to transpose the last two dims
    auto gemm_t0 = Clock::now();
    auto weight_transposed = weight_batched.transpose(1, 2);  // [B, In_Dim, Out_Dim]
    auto output = torch::matmul(input_f32, weight_transposed);  // [B, T, Out_Dim]
    auto gemm_t1 = Clock::now();
    timers["linear_cublas_matmul_us"] += std::chrono::duration_cast<Microseconds>(gemm_t1 - gemm_t0);

    // === 4. Epilogue: Bias and Activation ===
    auto epi_t0 = Clock::now();
    output.add_(bias_batched.unsqueeze(1));
    if (epilogue == IndexedLinearEpilogue::BiasGELU) {
        output = torch::gelu(output);
    }
    auto epi_t1 = Clock::now();
    timers["linear_epilogue_us"] += std::chrono::duration_cast<Microseconds>(epi_t1 - epi_t0);

    return output.to(input.scalar_type());
}
