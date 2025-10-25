#include "moe_cutlass_kernels.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/activation.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <array>

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
// CUTLASS Type Definitions
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
using LayoutC = cutlass::layout::RowMajor;  // Output/Bias

// MMA configuration for Ampere (sm_80)
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Swizzle
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

// Stages
constexpr int Stages = 4;

// ============================================================================
// W1 GEMM: Epilogue with Bias + GELU
// ============================================================================

// W1 Epilogue: D = GELU(alpha * AB + beta * C)
// Use erf-based GELU (CUTLASS default)
// We set alpha=1, beta=1, and pass bias as C with stride to broadcast per-column
using EpilogueOpW1 = cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // Elements per vector access
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::Default  // Use both alpha and beta
>;

// W1 Grouped GEMM Kernel
using GemmKernelW1 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,  // A
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,  // B
    ElementOutput, LayoutC,  // C/D
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOpW1,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemmW1 = cutlass::gemm::device::GemmGrouped<GemmKernelW1>;

// ============================================================================
// W2 GEMM: Standard Epilogue (bias + routing weights done separately for now)
// ============================================================================

// W2 Epilogue: Standard linear combination
using EpilogueOpW2 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
>;

// W2 Grouped GEMM Kernel
using GemmKernelW2 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,  // A
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,  // B
    ElementOutput, LayoutC,  // C/D
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOpW2,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemmW2 = cutlass::gemm::device::GemmGrouped<GemmKernelW2>;

// ============================================================================
// Helper Kernels for W2 Post-Processing
// ============================================================================

/**
 * Fused kernel: bias + per-row routing weight scaling
 * output[i, j] = (output[i, j] + bias[j]) * routing_weight[i]
 */
template <typename T>
__global__ void fused_bias_routing_kernel(
    T* data,
    const T* bias,
    const float* routing_weights,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;

    int64_t row = idx / cols;
    int64_t col = idx % cols;

    // Add bias and apply per-row routing weight in one pass
    float val = float(data[idx]) + float(bias[col]);
    val *= routing_weights[row];
    data[idx] = T(val);
}

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
            std::cerr << "[LB][MOE_CUTLASS_FUSED] No groups to process" << std::endl;
        }
        return;
    }

    const bool log_gemm = LB_MOE_LOG_GEMM();
    const bool log_cutlass = LB_MOE_LOG_CUTLASS();

    if (log_cutlass) {
        std::cerr << "[LB][MOE_CUTLASS_FUSED] Starting fused grouped MoE forward:" << std::endl;
        std::cerr << "  group_count=" << group_count << std::endl;
        std::cerr << "  hidden_dim=" << hidden_dim << std::endl;
        std::cerr << "  ffn_dim=" << ffn_dim << std::endl;
    }

    // Allocate workspace for intermediate hidden states
    cutlass::half_t* hidden_buffer = nullptr;
    size_t total_intermediate_size = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        total_intermediate_size += m_sizes[i] * ffn_dim;
    }
    CUDA_CHECK(cudaMalloc(&hidden_buffer, total_intermediate_size * sizeof(cutlass::half_t)));

    // ========================================================================
    // W1 GEMM with BIAS + GELU Fusion
    // ========================================================================

    // Setup problem sizes and pointers for W1 GEMM
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_w1;
    std::vector<ElementA*> ptr_A_w1;
    std::vector<ElementB*> ptr_B_w1;
    std::vector<ElementOutput*> ptr_C_w1;  // Bias
    std::vector<ElementOutput*> ptr_D_w1;  // Output
    std::vector<int64_t> lda_w1, ldb_w1, ldc_w1, ldd_w1;

    size_t hidden_offset = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        int64_t K = hidden_dim;
        int64_t N = ffn_dim;

        if (M == 0) continue;

        // GEMM: [M, K] @ [K, N] = [M, N]
        problem_sizes_w1.push_back(cutlass::gemm::GemmCoord(M, N, K));

        ptr_A_w1.push_back(reinterpret_cast<ElementA*>(input_ptrs[i]));
        ptr_B_w1.push_back(reinterpret_cast<ElementB*>(w1_ptrs[i]));
        ptr_C_w1.push_back(reinterpret_cast<ElementOutput*>(b1_ptrs[i]));  // Bias
        ptr_D_w1.push_back(reinterpret_cast<ElementOutput*>(hidden_buffer + hidden_offset));

        lda_w1.push_back(K);  // Row-major input
        ldb_w1.push_back(K);  // Column-major weights
        ldc_w1.push_back(0);  // Bias stride = 0 (broadcast across rows)
        ldd_w1.push_back(N);  // Row-major output

        if (log_gemm) {
            std::cerr << "[LB][MOE_GEMM_FUSED] W1 group=" << i
                      << " M=" << M << " N=" << N << " K=" << K << std::endl;
        }

        hidden_offset += M * N;
    }

    // Execute W1 grouped GEMM with fused BIAS + GELU
    if (!problem_sizes_w1.empty()) {
        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS_FUSED] Launching W1 fused GEMM (bias+GELU) with "
                      << problem_sizes_w1.size() << " problems" << std::endl;
        }

        // Device memory for problem metadata
        cutlass::gemm::GemmCoord* problem_sizes_device_w1 = nullptr;
        ElementA** ptr_A_device_w1 = nullptr;
        ElementB** ptr_B_device_w1 = nullptr;
        ElementOutput** ptr_C_device_w1 = nullptr;
        ElementOutput** ptr_D_device_w1 = nullptr;
        int64_t* lda_device_w1 = nullptr;
        int64_t* ldb_device_w1 = nullptr;
        int64_t* ldc_device_w1 = nullptr;
        int64_t* ldd_device_w1 = nullptr;

        size_t num_problems_w1 = problem_sizes_w1.size();
        CUDA_CHECK(cudaMalloc(&problem_sizes_device_w1, num_problems_w1 * sizeof(cutlass::gemm::GemmCoord)));
        CUDA_CHECK(cudaMalloc(&ptr_A_device_w1, num_problems_w1 * sizeof(ElementA*)));
        CUDA_CHECK(cudaMalloc(&ptr_B_device_w1, num_problems_w1 * sizeof(ElementB*)));
        CUDA_CHECK(cudaMalloc(&ptr_C_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
        CUDA_CHECK(cudaMalloc(&ptr_D_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
        CUDA_CHECK(cudaMalloc(&lda_device_w1, num_problems_w1 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldb_device_w1, num_problems_w1 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldc_device_w1, num_problems_w1 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldd_device_w1, num_problems_w1 * sizeof(int64_t)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(problem_sizes_device_w1, problem_sizes_w1.data(),
                              num_problems_w1 * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_A_device_w1, ptr_A_w1.data(),
                              num_problems_w1 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_B_device_w1, ptr_B_w1.data(),
                              num_problems_w1 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_C_device_w1, ptr_C_w1.data(),
                              num_problems_w1 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_D_device_w1, ptr_D_w1.data(),
                              num_problems_w1 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lda_device_w1, lda_w1.data(),
                              num_problems_w1 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldb_device_w1, ldb_w1.data(),
                              num_problems_w1 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldc_device_w1, ldc_w1.data(),
                              num_problems_w1 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldd_device_w1, ldd_w1.data(),
                              num_problems_w1 * sizeof(int64_t), cudaMemcpyHostToDevice));

        GroupedGemmW1 gemm_w1;

        int threadblock_count_w1 = GroupedGemmW1::sufficient(problem_sizes_w1.data(), num_problems_w1);

        // Epilogue params: alpha=1, beta=1 for D = GELU(1*AB + 1*C)
        typename GroupedGemmW1::EpilogueOutputOp::Params epilogue_params_w1(1.0f, 1.0f);

        typename GroupedGemmW1::Arguments args_w1(
            problem_sizes_device_w1,
            num_problems_w1,
            threadblock_count_w1,
            epilogue_params_w1,
            ptr_A_device_w1,
            ptr_B_device_w1,
            ptr_C_device_w1,  // Bias (with stride=0)
            ptr_D_device_w1,  // Output
            lda_device_w1,
            ldb_device_w1,
            ldc_device_w1,
            ldd_device_w1,
            problem_sizes_w1.data()
        );

        size_t workspace_size_w1 = gemm_w1.get_workspace_size(args_w1);
        void* workspace_w1 = nullptr;
        if (workspace_size_w1 > 0) {
            CUDA_CHECK(cudaMalloc(&workspace_w1, workspace_size_w1));
        }

        CUTLASS_CHECK(gemm_w1.initialize(args_w1, workspace_w1));
        CUTLASS_CHECK(gemm_w1.run());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Cleanup
        if (workspace_w1) cudaFree(workspace_w1);
        cudaFree(problem_sizes_device_w1);
        cudaFree(ptr_A_device_w1);
        cudaFree(ptr_B_device_w1);
        cudaFree(ptr_C_device_w1);
        cudaFree(ptr_D_device_w1);
        cudaFree(lda_device_w1);
        cudaFree(ldb_device_w1);
        cudaFree(ldc_device_w1);
        cudaFree(ldd_device_w1);

        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS_FUSED] W1 fused GEMM completed (2 kernels → 1)" << std::endl;
        }
    }

    // ========================================================================
    // W2 GEMM + Fused Bias+Routing Weight Post-Processing
    // ========================================================================

    // Setup problem sizes for W2 GEMM
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_w2;
    std::vector<ElementA*> ptr_A_w2;
    std::vector<ElementB*> ptr_B_w2;
    std::vector<ElementOutput*> ptr_C_w2;
    std::vector<ElementOutput*> ptr_D_w2;
    std::vector<int64_t> lda_w2, ldb_w2, ldc_w2, ldd_w2;

    hidden_offset = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        int64_t K = ffn_dim;
        int64_t N = hidden_dim;

        if (M == 0) continue;

        problem_sizes_w2.push_back(cutlass::gemm::GemmCoord(M, N, K));

        ptr_A_w2.push_back(reinterpret_cast<ElementA*>(hidden_buffer + hidden_offset));
        ptr_B_w2.push_back(reinterpret_cast<ElementB*>(w2_ptrs[i]));
        ptr_C_w2.push_back(nullptr);  // No C matrix
        ptr_D_w2.push_back(reinterpret_cast<ElementOutput*>(output_ptrs[i]));

        lda_w2.push_back(K);
        ldb_w2.push_back(K);
        ldc_w2.push_back(0);
        ldd_w2.push_back(N);

        if (log_gemm) {
            std::cerr << "[LB][MOE_GEMM_FUSED] W2 group=" << i
                      << " M=" << M << " N=" << N << " K=" << K << std::endl;
        }

        hidden_offset += M * K;
    }

    // Execute W2 grouped GEMM
    if (!problem_sizes_w2.empty()) {
        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS_FUSED] Launching W2 GEMM with "
                      << problem_sizes_w2.size() << " problems" << std::endl;
        }

        // Device memory
        cutlass::gemm::GemmCoord* problem_sizes_device_w2 = nullptr;
        ElementA** ptr_A_device_w2 = nullptr;
        ElementB** ptr_B_device_w2 = nullptr;
        ElementOutput** ptr_C_device_w2 = nullptr;
        ElementOutput** ptr_D_device_w2 = nullptr;
        int64_t* lda_device_w2 = nullptr;
        int64_t* ldb_device_w2 = nullptr;
        int64_t* ldc_device_w2 = nullptr;
        int64_t* ldd_device_w2 = nullptr;

        size_t num_problems_w2 = problem_sizes_w2.size();
        CUDA_CHECK(cudaMalloc(&problem_sizes_device_w2, num_problems_w2 * sizeof(cutlass::gemm::GemmCoord)));
        CUDA_CHECK(cudaMalloc(&ptr_A_device_w2, num_problems_w2 * sizeof(ElementA*)));
        CUDA_CHECK(cudaMalloc(&ptr_B_device_w2, num_problems_w2 * sizeof(ElementB*)));
        CUDA_CHECK(cudaMalloc(&ptr_C_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
        CUDA_CHECK(cudaMalloc(&ptr_D_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
        CUDA_CHECK(cudaMalloc(&lda_device_w2, num_problems_w2 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldb_device_w2, num_problems_w2 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldc_device_w2, num_problems_w2 * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&ldd_device_w2, num_problems_w2 * sizeof(int64_t)));

        CUDA_CHECK(cudaMemcpy(problem_sizes_device_w2, problem_sizes_w2.data(),
                              num_problems_w2 * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_A_device_w2, ptr_A_w2.data(),
                              num_problems_w2 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_B_device_w2, ptr_B_w2.data(),
                              num_problems_w2 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_C_device_w2, ptr_C_w2.data(),
                              num_problems_w2 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptr_D_device_w2, ptr_D_w2.data(),
                              num_problems_w2 * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lda_device_w2, lda_w2.data(),
                              num_problems_w2 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldb_device_w2, ldb_w2.data(),
                              num_problems_w2 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldc_device_w2, ldc_w2.data(),
                              num_problems_w2 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ldd_device_w2, ldd_w2.data(),
                              num_problems_w2 * sizeof(int64_t), cudaMemcpyHostToDevice));

        GroupedGemmW2 gemm_w2;

        int threadblock_count_w2 = GroupedGemmW2::sufficient(problem_sizes_w2.data(), num_problems_w2);
        typename GroupedGemmW2::EpilogueOutputOp::Params epilogue_params_w2(1.0f, 0.0f);

        typename GroupedGemmW2::Arguments args_w2(
            problem_sizes_device_w2,
            num_problems_w2,
            threadblock_count_w2,
            epilogue_params_w2,
            ptr_A_device_w2,
            ptr_B_device_w2,
            ptr_C_device_w2,
            ptr_D_device_w2,
            lda_device_w2,
            ldb_device_w2,
            ldc_device_w2,
            ldd_device_w2,
            problem_sizes_w2.data()
        );

        size_t workspace_size_w2 = gemm_w2.get_workspace_size(args_w2);
        void* workspace_w2 = nullptr;
        if (workspace_size_w2 > 0) {
            CUDA_CHECK(cudaMalloc(&workspace_w2, workspace_size_w2));
        }

        CUTLASS_CHECK(gemm_w2.initialize(args_w2, workspace_w2));
        CUTLASS_CHECK(gemm_w2.run());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Cleanup
        if (workspace_w2) cudaFree(workspace_w2);
        cudaFree(problem_sizes_device_w2);
        cudaFree(ptr_A_device_w2);
        cudaFree(ptr_B_device_w2);
        cudaFree(ptr_C_device_w2);
        cudaFree(ptr_D_device_w2);
        cudaFree(lda_device_w2);
        cudaFree(ldb_device_w2);
        cudaFree(ldc_device_w2);
        cudaFree(ldd_device_w2);

        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS_FUSED] W2 GEMM completed" << std::endl;
        }
    }

    // Apply fused bias + routing weights to W2 outputs (1 kernel instead of 2)
    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        cutlass::half_t* output_ptr = reinterpret_cast<cutlass::half_t*>(output_ptrs[i]);
        const cutlass::half_t* bias_ptr = reinterpret_cast<const cutlass::half_t*>(b2_ptrs[i]);
        const float* routing_ptr = reinterpret_cast<const float*>(routing_weight_ptrs[i]);

        int64_t total = M * hidden_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        // Fused bias + routing weight kernel (replaces 2 separate kernels)
        fused_bias_routing_kernel<<<blocks, threads>>>(output_ptr, bias_ptr, routing_ptr, M, hidden_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUDA_CHECK(cudaFree(hidden_buffer));

    if (log_cutlass) {
        std::cerr << "[LB][MOE_CUTLASS_FUSED] Fused MoE forward completed:" << std::endl;
        std::cerr << "  W1: 3 kernels → 1 (GEMM+bias+GELU fused)" << std::endl;
        std::cerr << "  W2: 3 kernels → 2 (GEMM + fused bias+routing)" << std::endl;
        std::cerr << "  Total: 6 kernels → 3 (50% reduction)" << std::endl;
    }
}

} // namespace moe
} // namespace lb
