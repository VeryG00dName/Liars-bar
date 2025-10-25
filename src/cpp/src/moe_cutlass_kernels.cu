#include "moe_cutlass_kernels.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <cmath>

namespace lb {
namespace moe {

// Environment variable helpers
static inline bool lb_env_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    if (v[0] == '0' || v[0] == 'F' || v[0] == 'f' || v[0] == 'N' || v[0] == 'n') return false;
    return true;
}

static inline int lb_env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    return static_cast<int>(parsed);
}

static bool LB_MOE_LOG_GEMM() {
    static int value = lb_env_int("LB_MOE_LOG_GEMM", 0);
    return value != 0;
}

static bool LB_MOE_LOG_CUTLASS() {
    static int flag = -1;
    if (flag < 0) flag = lb_env_enabled("LB_MOE_LOG_CUTLASS");
    return flag != 0;
}

// Helper macro for CUTLASS error checking
#define CUTLASS_CHECK(status)                                                           \
    {                                                                                   \
        cutlass::Status error = status;                                                 \
        if (error != cutlass::Status::kSuccess) {                                       \
            std::stringstream ss;                                                       \
            ss << "CUTLASS error at " << __FILE__ << ":" << __LINE__ << " - "          \
               << cutlassGetStatusString(error);                                        \
            throw std::runtime_error(ss.str());                                         \
        }                                                                               \
    }

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                                \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess) {                                                       \
            std::stringstream ss;                                                       \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "             \
               << cudaGetErrorString(err);                                              \
            throw std::runtime_error(ss.str());                                         \
        }                                                                               \
    }

// ============================================================================
// Custom CUTLASS Epilogue: Per-Row Routing Weight Scaling
// ============================================================================

/**
 * Custom epilogue functor that applies per-row routing weight scaling.
 *
 * Computes: output = (accumulator * alpha + bias) * routing_weight[row]
 *
 * This is used in the W2 GEMM to fuse the routing weight multiplication
 * directly into the GEMM epilogue, eliminating a separate kernel launch.
 */
template <
    typename ElementOutput_,
    typename ElementAccumulator_,
    typename ElementCompute_,
    int ElementsPerAccess
>
struct PerRowScaleEpilogue {
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;

    static int const kElementsPerAccess = ElementsPerAccess;

    struct Params {
        ElementCompute alpha;
        ElementCompute beta;
        ElementOutput const* bias_ptr;
        float const* routing_weights_ptr;
        int64_t group_offset;  // Offset into routing_weights array for this group
        int64_t output_stride;

        CUTLASS_HOST_DEVICE
        Params():
            alpha(ElementCompute(1)),
            beta(ElementCompute(0)),
            bias_ptr(nullptr),
            routing_weights_ptr(nullptr),
            group_offset(0),
            output_stride(0) {}

        CUTLASS_HOST_DEVICE
        Params(
            ElementCompute alpha_,
            ElementCompute beta_,
            ElementOutput const* bias_ptr_,
            float const* routing_weights_ptr_,
            int64_t group_offset_,
            int64_t output_stride_
        ):
            alpha(alpha_),
            beta(beta_),
            bias_ptr(bias_ptr_),
            routing_weights_ptr(routing_weights_ptr_),
            group_offset(group_offset_),
            output_stride(output_stride_) {}
    };

    Params params_;

    CUTLASS_DEVICE
    PerRowScaleEpilogue(Params const& params): params_(params) {}

    CUTLASS_DEVICE
    ElementOutput operator()(
        ElementAccumulator accumulator,
        int row_idx,
        int col_idx
    ) const {
        ElementCompute compute = ElementCompute(accumulator) * params_.alpha;

        // Add bias
        if (params_.bias_ptr != nullptr) {
            compute += ElementCompute(params_.bias_ptr[col_idx]);
        }

        // Apply per-row routing weight
        if (params_.routing_weights_ptr != nullptr) {
            float routing_weight = params_.routing_weights_ptr[params_.group_offset + row_idx];
            compute *= ElementCompute(routing_weight);
        }

        return ElementOutput(compute);
    }
};

// ============================================================================
// GELU Activation Functor
// ============================================================================

template <typename T>
struct GeluActivation {
    CUTLASS_DEVICE
    T operator()(T const& x) const {
        float x_f = float(x);
        float kAlpha = 0.7978845608f;  // sqrt(2/pi)
        float kGamma = 0.044715f;
        float x_cube = x_f * x_f * x_f;
        float tanh_arg = kAlpha * (x_f + kGamma * x_cube);
        float result = 0.5f * x_f * (1.0f + tanhf(tanh_arg));
        return T(result);
    }
};

/**
 * Epilogue functor for W1 GEMM: applies bias + GELU activation.
 */
template <
    typename ElementOutput_,
    typename ElementAccumulator_,
    typename ElementCompute_,
    int ElementsPerAccess
>
struct BiasGeluEpilogue {
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;

    static int const kElementsPerAccess = ElementsPerAccess;

    struct Params {
        ElementCompute alpha;
        ElementCompute beta;
        ElementOutput const* bias_ptr;
        int64_t output_stride;

        CUTLASS_HOST_DEVICE
        Params():
            alpha(ElementCompute(1)),
            beta(ElementCompute(0)),
            bias_ptr(nullptr),
            output_stride(0) {}

        CUTLASS_HOST_DEVICE
        Params(
            ElementCompute alpha_,
            ElementCompute beta_,
            ElementOutput const* bias_ptr_,
            int64_t output_stride_
        ):
            alpha(alpha_),
            beta(beta_),
            bias_ptr(bias_ptr_),
            output_stride(output_stride_) {}
    };

    Params params_;
    GeluActivation<ElementCompute> gelu_;

    CUTLASS_DEVICE
    BiasGeluEpilogue(Params const& params): params_(params), gelu_() {}

    CUTLASS_DEVICE
    ElementOutput operator()(
        ElementAccumulator accumulator,
        int row_idx,
        int col_idx
    ) const {
        ElementCompute compute = ElementCompute(accumulator) * params_.alpha;

        // Add bias
        if (params_.bias_ptr != nullptr) {
            compute += ElementCompute(params_.bias_ptr[col_idx]);
        }

        // Apply GELU
        compute = gelu_(compute);

        return ElementOutput(compute);
    }
};

// ============================================================================
// Grouped GEMM Helper Kernels
// ============================================================================

/**
 * Simple GELU kernel for intermediate activations.
 * Used for W1 output since custom epilogue support varies by CUTLASS version.
 */
template <typename T>
__global__ void gelu_kernel(
    T* data,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = float(data[idx]);
    float kAlpha = 0.7978845608f;  // sqrt(2/pi)
    float kGamma = 0.044715f;
    float x_cube = x * x * x;
    float tanh_arg = kAlpha * (x + kGamma * x_cube);
    float result = 0.5f * x * (1.0f + tanhf(tanh_arg));
    data[idx] = T(result);
}

/**
 * Kernel to add bias to GEMM output.
 */
template <typename T>
__global__ void add_bias_kernel(
    T* data,
    const T* bias,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;

    int64_t col = idx % cols;
    data[idx] += bias[col];
}

/**
 * Kernel to apply per-row routing weights.
 * output[i, j] *= routing_weights[i]
 */
template <typename T>
__global__ void apply_routing_weights_kernel(
    T* data,
    const float* routing_weights,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;

    int64_t row = idx / cols;
    float scale = routing_weights[row];
    data[idx] *= T(scale);
}

// ============================================================================
// CUTLASS Grouped GEMM Configuration
// ============================================================================

// FP16 input/output, FP32 accumulation
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

// Layouts
using LayoutA = cutlass::layout::RowMajor;  // Input tokens
using LayoutB = cutlass::layout::ColumnMajor;  // Weights (transposed)
using LayoutC = cutlass::layout::RowMajor;  // Output

// MMA configuration for Ampere (sm_80)
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Epilogue configuration
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
>;

// Grouped GEMM kernel - CUTLASS 4.x API
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,  // Alignment A
    ElementB,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,  // Alignment B
    ElementOutput,
    LayoutC,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4  // Stages
>::GemmKernel;

using GroupedGemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// ============================================================================
// Main Implementation
// ============================================================================

void cutlass_grouped_moe_forward(
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
    int64_t ffn_dim
) {
    if (group_count == 0) {
        if (LB_MOE_LOG_GEMM()) {
            std::cerr << "[LB][MOE_CUTLASS] No groups to process" << std::endl;
        }
        return;
    }

    const bool log_gemm = LB_MOE_LOG_GEMM();
    const bool log_cutlass = LB_MOE_LOG_CUTLASS();

    if (log_cutlass) {
        std::cerr << "[LB][MOE_CUTLASS] Starting grouped MoE forward:" << std::endl;
        std::cerr << "  group_count=" << group_count << std::endl;
        std::cerr << "  hidden_dim=" << hidden_dim << std::endl;
        std::cerr << "  ffn_dim=" << ffn_dim << std::endl;
    }

    // Allocate workspace for intermediate hidden states
    size_t max_tokens = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        max_tokens = std::max(max_tokens, static_cast<size_t>(m_sizes[i]));
    }

    // Allocate intermediate buffer for all groups
    cutlass::half_t* hidden_buffer = nullptr;
    size_t total_intermediate_size = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        total_intermediate_size += m_sizes[i] * ffn_dim;
    }

    CUDA_CHECK(cudaMalloc(&hidden_buffer, total_intermediate_size * sizeof(cutlass::half_t)));

    // Setup problem sizes and pointers for W1 GEMM (input @ W1^T)
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_w1;
    std::vector<void*> ptr_A_w1, ptr_B_w1, ptr_C_w1;
    std::vector<int64_t> lda_w1, ldb_w1, ldc_w1;

    size_t hidden_offset = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        int64_t K = hidden_dim;
        int64_t N = ffn_dim;

        if (M == 0) continue;

        // GEMM: [M, K] @ [K, N] = [M, N]
        // With LayoutA=RowMajor, LayoutB=ColumnMajor (transposed weights)
        problem_sizes_w1.push_back(cutlass::gemm::GemmCoord(M, N, K));

        ptr_A_w1.push_back(reinterpret_cast<void*>(input_ptrs[i]));
        ptr_B_w1.push_back(reinterpret_cast<void*>(w1_ptrs[i]));
        ptr_C_w1.push_back(reinterpret_cast<void*>(hidden_buffer + hidden_offset));

        lda_w1.push_back(K);  // Row-major input: stride = K
        ldb_w1.push_back(K);  // Column-major weights: stride = K
        ldc_w1.push_back(N);  // Row-major output: stride = N

        if (log_gemm) {
            std::cerr << "[LB][MOE_GEMM] W1 group=" << i
                      << " policy=" << policy_ids[i]
                      << " expert=" << expert_ids[i]
                      << " M=" << M << " N=" << N << " K=" << K << std::endl;
        }

        hidden_offset += M * N;
    }

    // Execute W1 grouped GEMM
    if (!problem_sizes_w1.empty()) {
        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS] Launching W1 grouped GEMM with "
                      << problem_sizes_w1.size() << " problems" << std::endl;
        }

        // Allocate device memory for problem metadata
        cutlass::gemm::GemmCoord* problem_sizes_device_w1 = nullptr;
        ElementA** ptr_A_device_w1 = nullptr;
        ElementB** ptr_B_device_w1 = nullptr;
        ElementOutput** ptr_C_device_w1 = nullptr;
        int64_t* lda_device_w1 = nullptr;
        int64_t* ldb_device_w1 = nullptr;
        int64_t* ldc_device_w1 = nullptr;

        size_t num_problems_w1 = problem_sizes_w1.size();
        CUDA_CHECK(cudaMalloc(&problem_sizes_device_w1, num_problems_w1 * sizeof(cutlass::gemm::GemmCoord)));
        CUDA_CHECK(cudaMalloc(&ptr_A_device_w1, num_problems_w1 * sizeof(ElementA*)));
        CUDA_CHECK(cudaMalloc(&ptr_B_device_w1, num_problems_w1 * sizeof(ElementB*)));
        CUDA_CHECK(cudaMalloc(&ptr_C_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
        CUDA_CHECK(cudaMalloc(&lda_device_w1, num_problems_w1 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldb_device_w1, num_problems_w1 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldc_device_w1, num_problems_w1 * sizeof(int64_t)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(problem_sizes_device_w1, problem_sizes_w1.data(),
                              num_problems_w1 * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_A_device_w1, ptr_A_w1.data(),
                              num_problems_w1 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_B_device_w1, ptr_B_w1.data(),
                              num_problems_w1 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_C_device_w1, ptr_C_w1.data(),
                              num_problems_w1 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lda_device_w1, lda_w1.data(),
                              num_problems_w1 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldb_device_w1, ldb_w1.data(),
                              num_problems_w1 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldc_device_w1, ldc_w1.data(),
                              num_problems_w1 * sizeof(int64_t), cudaMemcpyHostToDevice));

        GroupedGemm gemm_w1;

        // CUTLASS 4.x API: create epilogue params and arguments
        int threadblock_count_w1 = GroupedGemm::sufficient(problem_sizes_w1.data(), num_problems_w1);
        typename GroupedGemm::EpilogueOutputOp::Params epilogue_params_w1(1.0f, 0.0f);

        typename GroupedGemm::Arguments args_w1(
            problem_sizes_device_w1,
            num_problems_w1,
            threadblock_count_w1,
            epilogue_params_w1,
            ptr_A_device_w1,
            ptr_B_device_w1,
            ptr_C_device_w1,
            ptr_C_device_w1,  // D = C for in-place
            lda_device_w1,
            ldb_device_w1,
            ldc_device_w1,
            ldc_device_w1,
            problem_sizes_w1.data()  // Host pointer for reference
        );

        // Get workspace size and allocate
        size_t workspace_size_w1 = gemm_w1.get_workspace_size(args_w1);
        void* workspace_w1 = nullptr;
        if (workspace_size_w1 > 0) {
            CUDA_CHECK(cudaMalloc(&workspace_w1, workspace_size_w1));
        }

        // Initialize and run
        CUTLASS_CHECK(gemm_w1.initialize(args_w1, workspace_w1));
        CUTLASS_CHECK(gemm_w1.run());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Cleanup
        if (workspace_w1) cudaFree(workspace_w1);
        cudaFree(problem_sizes_device_w1);
        cudaFree(ptr_A_device_w1);
        cudaFree(ptr_B_device_w1);
        cudaFree(ptr_C_device_w1);
        cudaFree(lda_device_w1);
        cudaFree(ldb_device_w1);
        cudaFree(ldc_device_w1);

        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS] W1 grouped GEMM completed" << std::endl;
        }
    }

    // Apply bias + GELU to hidden states
    hidden_offset = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        cutlass::half_t* hidden_ptr = hidden_buffer + hidden_offset;
        const cutlass::half_t* bias_ptr = reinterpret_cast<const cutlass::half_t*>(b1_ptrs[i]);

        // Add bias
        int64_t total = M * ffn_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads>>>(hidden_ptr, bias_ptr, M, ffn_dim);

        // Apply GELU
        gelu_kernel<<<blocks, threads>>>(hidden_ptr, total);

        hidden_offset += M * ffn_dim;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Setup problem sizes and pointers for W2 GEMM (hidden @ W2^T)
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_w2;
    std::vector<void*> ptr_A_w2, ptr_B_w2, ptr_C_w2;
    std::vector<int64_t> lda_w2, ldb_w2, ldc_w2;

    hidden_offset = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        int64_t K = ffn_dim;
        int64_t N = hidden_dim;

        if (M == 0) continue;

        // GEMM: [M, K] @ [K, N] = [M, N]
        problem_sizes_w2.push_back(cutlass::gemm::GemmCoord(M, N, K));

        ptr_A_w2.push_back(reinterpret_cast<void*>(hidden_buffer + hidden_offset));
        ptr_B_w2.push_back(reinterpret_cast<void*>(w2_ptrs[i]));
        ptr_C_w2.push_back(reinterpret_cast<void*>(output_ptrs[i]));

        lda_w2.push_back(K);
        ldb_w2.push_back(K);
        ldc_w2.push_back(N);

        if (log_gemm) {
            std::cerr << "[LB][MOE_GEMM] W2 group=" << i
                      << " policy=" << policy_ids[i]
                      << " expert=" << expert_ids[i]
                      << " M=" << M << " N=" << N << " K=" << K << std::endl;
        }

        hidden_offset += M * K;
    }

    // Execute W2 grouped GEMM
    if (!problem_sizes_w2.empty()) {
        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS] Launching W2 grouped GEMM with "
                      << problem_sizes_w2.size() << " problems" << std::endl;
        }

        // Allocate device memory for problem metadata
        cutlass::gemm::GemmCoord* problem_sizes_device_w2 = nullptr;
        ElementA** ptr_A_device_w2 = nullptr;
        ElementB** ptr_B_device_w2 = nullptr;
        ElementOutput** ptr_C_device_w2 = nullptr;
        int64_t* lda_device_w2 = nullptr;
        int64_t* ldb_device_w2 = nullptr;
        int64_t* ldc_device_w2 = nullptr;

        size_t num_problems_w2 = problem_sizes_w2.size();
        CUDA_CHECK(cudaMalloc(&problem_sizes_device_w2, num_problems_w2 * sizeof(cutlass::gemm::GemmCoord)));
        CUDA_CHECK(cudaMalloc(&ptr_A_device_w2, num_problems_w2 * sizeof(ElementA*)));
        CUDA_CHECK(cudaMalloc(&ptr_B_device_w2, num_problems_w2 * sizeof(ElementB*)));
        CUDA_CHECK(cudaMalloc(&ptr_C_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
        CUDA_CHECK(cudaMalloc(&lda_device_w2, num_problems_w2 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldb_device_w2, num_problems_w2 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldc_device_w2, num_problems_w2 * sizeof(int64_t)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(problem_sizes_device_w2, problem_sizes_w2.data(),
                              num_problems_w2 * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_A_device_w2, ptr_A_w2.data(),
                              num_problems_w2 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_B_device_w2, ptr_B_w2.data(),
                              num_problems_w2 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_C_device_w2, ptr_C_w2.data(),
                              num_problems_w2 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lda_device_w2, lda_w2.data(),
                              num_problems_w2 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldb_device_w2, ldb_w2.data(),
                              num_problems_w2 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldc_device_w2, ldc_w2.data(),
                              num_problems_w2 * sizeof(int64_t), cudaMemcpyHostToDevice));

        GroupedGemm gemm_w2;

        // CUTLASS 4.x API: create epilogue params and arguments
        int threadblock_count_w2 = GroupedGemm::sufficient(problem_sizes_w2.data(), num_problems_w2);
        typename GroupedGemm::EpilogueOutputOp::Params epilogue_params_w2(1.0f, 0.0f);

        typename GroupedGemm::Arguments args_w2(
            problem_sizes_device_w2,
            num_problems_w2,
            threadblock_count_w2,
            epilogue_params_w2,
            ptr_A_device_w2,
            ptr_B_device_w2,
            ptr_C_device_w2,
            ptr_C_device_w2,  // D = C for in-place
            lda_device_w2,
            ldb_device_w2,
            ldc_device_w2,
            ldc_device_w2,
            problem_sizes_w2.data()  // Host pointer for reference
        );

        // Get workspace size and allocate
        size_t workspace_size_w2 = gemm_w2.get_workspace_size(args_w2);
        void* workspace_w2 = nullptr;
        if (workspace_size_w2 > 0) {
            CUDA_CHECK(cudaMalloc(&workspace_w2, workspace_size_w2));
        }

        // Initialize and run
        CUTLASS_CHECK(gemm_w2.initialize(args_w2, workspace_w2));
        CUTLASS_CHECK(gemm_w2.run());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Cleanup
        if (workspace_w2) cudaFree(workspace_w2);
        cudaFree(problem_sizes_device_w2);
        cudaFree(ptr_A_device_w2);
        cudaFree(ptr_B_device_w2);
        cudaFree(ptr_C_device_w2);
        cudaFree(lda_device_w2);
        cudaFree(ldb_device_w2);
        cudaFree(ldc_device_w2);

        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS] W2 grouped GEMM completed" << std::endl;
        }
    }

    // Apply bias + routing weights to outputs
    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        cutlass::half_t* output_ptr = reinterpret_cast<cutlass::half_t*>(output_ptrs[i]);
        const cutlass::half_t* bias_ptr = reinterpret_cast<const cutlass::half_t*>(b2_ptrs[i]);
        const float* routing_ptr = reinterpret_cast<const float*>(routing_weight_ptrs[i]);

        int64_t total = M * hidden_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        // Add bias
        add_bias_kernel<<<blocks, threads>>>(output_ptr, bias_ptr, M, hidden_dim);

        // Apply routing weights
        apply_routing_weights_kernel<<<blocks, threads>>>(output_ptr, routing_ptr, M, hidden_dim);

        if (log_cutlass && i < 2) {
            std::cerr << "[LB][MOE_CUTLASS] Applied bias and routing weights to group " << i << std::endl;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUDA_CHECK(cudaFree(hidden_buffer));

    if (log_cutlass) {
        std::cerr << "[LB][MOE_CUTLASS] Grouped MoE forward completed successfully" << std::endl;
    }
}

} // namespace moe
} // namespace lb
