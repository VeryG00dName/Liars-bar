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

// Optional log throttling for readability
static int LB_MOE_LOG_LIMIT() {
    // -1 means unlimited
    static int value = lb_env_int("LB_MOE_LOG_LIMIT", -1);
    return value;
}

static bool lb_moe_cutlass_should_log() {
    if (!LB_MOE_LOG_CUTLASS()) return false;
    static int remaining = LB_MOE_LOG_LIMIT();
    if (remaining < 0) return true;  // unlimited
    if (remaining == 0) return false;
    --remaining;
    if (remaining == 0) {
        std::cerr << "[LB][MOE_CUTLASS] Reached LB_MOE_LOG_LIMIT; suppressing further logs." << std::endl;
    }
    return true;
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
    int64_t ffn_dim,
    MoEWorkspace* workspace
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

    // Allocate or reuse workspace for intermediate hidden states
    cutlass::half_t* hidden_buffer = nullptr;
    bool owns_hidden_buffer = false;
    size_t total_intermediate_size = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        total_intermediate_size += m_sizes[i] * ffn_dim;
    }
    size_t required_hidden_size = total_intermediate_size * sizeof(cutlass::half_t);

    if (workspace && workspace->hidden_buffer && workspace->hidden_buffer_size >= required_hidden_size) {
        // Use pre-allocated workspace
        hidden_buffer = reinterpret_cast<cutlass::half_t*>(workspace->hidden_buffer);
    } else if (workspace && workspace->hidden_buffer) {
        // Workspace too small - grow it
        cudaFree(workspace->hidden_buffer);
        CUDA_CHECK(cudaMalloc(&workspace->hidden_buffer, required_hidden_size));
        workspace->hidden_buffer_size = required_hidden_size;
        hidden_buffer = reinterpret_cast<cutlass::half_t*>(workspace->hidden_buffer);
    } else if (workspace) {
        // Workspace provided but no buffer allocated yet
        CUDA_CHECK(cudaMalloc(&workspace->hidden_buffer, required_hidden_size));
        workspace->hidden_buffer_size = required_hidden_size;
        hidden_buffer = reinterpret_cast<cutlass::half_t*>(workspace->hidden_buffer);
    } else {
        // No workspace - allocate temporary buffer
        CUDA_CHECK(cudaMalloc(&hidden_buffer, required_hidden_size));
        owns_hidden_buffer = true;
    }

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
        bool owns_w1_descriptors = false;

        size_t num_problems_w1 = problem_sizes_w1.size();

        if (workspace && workspace->descriptor_capacity_w1 >= num_problems_w1) {
            // Use pre-allocated workspace
            problem_sizes_device_w1 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w1);
            ptr_A_device_w1 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w1);
            ptr_B_device_w1 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w1);
            ptr_C_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w1);
            ptr_D_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w1);
            lda_device_w1 = reinterpret_cast<int64_t*>(workspace->lda_device_w1);
            ldb_device_w1 = reinterpret_cast<int64_t*>(workspace->ldb_device_w1);
            ldc_device_w1 = reinterpret_cast<int64_t*>(workspace->ldc_device_w1);
            ldd_device_w1 = reinterpret_cast<int64_t*>(workspace->ldd_device_w1);
        } else if (workspace) {
            // Workspace too small - grow it
            if (workspace->problem_sizes_device_w1) {
                cudaFree(workspace->problem_sizes_device_w1);
                cudaFree(workspace->ptr_A_device_w1);
                cudaFree(workspace->ptr_B_device_w1);
                cudaFree(workspace->ptr_C_device_w1);
                cudaFree(workspace->ptr_D_device_w1);
                cudaFree(workspace->lda_device_w1);
                cudaFree(workspace->ldb_device_w1);
                cudaFree(workspace->ldc_device_w1);
                cudaFree(workspace->ldd_device_w1);
            }
            CUDA_CHECK(cudaMalloc(&workspace->problem_sizes_device_w1, num_problems_w1 * sizeof(cutlass::gemm::GemmCoord)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_A_device_w1, num_problems_w1 * sizeof(ElementA*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_B_device_w1, num_problems_w1 * sizeof(ElementB*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_C_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_D_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&workspace->lda_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldb_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldc_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldd_device_w1, num_problems_w1 * sizeof(int64_t)));
            workspace->descriptor_capacity_w1 = num_problems_w1;

            problem_sizes_device_w1 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w1);
            ptr_A_device_w1 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w1);
            ptr_B_device_w1 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w1);
            ptr_C_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w1);
            ptr_D_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w1);
            lda_device_w1 = reinterpret_cast<int64_t*>(workspace->lda_device_w1);
            ldb_device_w1 = reinterpret_cast<int64_t*>(workspace->ldb_device_w1);
            ldc_device_w1 = reinterpret_cast<int64_t*>(workspace->ldc_device_w1);
            ldd_device_w1 = reinterpret_cast<int64_t*>(workspace->ldd_device_w1);
        } else {
            // No workspace - allocate temporary buffers
            CUDA_CHECK(cudaMalloc(&problem_sizes_device_w1, num_problems_w1 * sizeof(cutlass::gemm::GemmCoord)));
            CUDA_CHECK(cudaMalloc(&ptr_A_device_w1, num_problems_w1 * sizeof(ElementA*)));
            CUDA_CHECK(cudaMalloc(&ptr_B_device_w1, num_problems_w1 * sizeof(ElementB*)));
            CUDA_CHECK(cudaMalloc(&ptr_C_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&ptr_D_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&lda_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldb_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldc_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldd_device_w1, num_problems_w1 * sizeof(int64_t)));
            owns_w1_descriptors = true;
        }

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
        bool owns_workspace_w1 = false;

        if (workspace && workspace->workspace_w1 && workspace->workspace_w1_size >= workspace_size_w1) {
            // Use pre-allocated workspace
            workspace_w1 = workspace->workspace_w1;
        } else if (workspace && workspace->workspace_w1) {
            // Workspace too small - grow it
            cudaFree(workspace->workspace_w1);
            CUDA_CHECK(cudaMalloc(&workspace->workspace_w1, workspace_size_w1));
            workspace->workspace_w1_size = workspace_size_w1;
            workspace_w1 = workspace->workspace_w1;
        } else if (workspace && workspace_size_w1 > 0) {
            // Workspace provided but no buffer allocated yet
            CUDA_CHECK(cudaMalloc(&workspace->workspace_w1, workspace_size_w1));
            workspace->workspace_w1_size = workspace_size_w1;
            workspace_w1 = workspace->workspace_w1;
        } else if (workspace_size_w1 > 0) {
            // No workspace - allocate temporary buffer
            CUDA_CHECK(cudaMalloc(&workspace_w1, workspace_size_w1));
            owns_workspace_w1 = true;
        }

        CUTLASS_CHECK(gemm_w1.initialize(args_w1, workspace_w1));
        CUTLASS_CHECK(gemm_w1.run());

        // Cleanup (only if we own the buffers)
        if (owns_workspace_w1 && workspace_w1) cudaFree(workspace_w1);
        if (owns_w1_descriptors) {
            cudaFree(problem_sizes_device_w1);
            cudaFree(ptr_A_device_w1);
            cudaFree(ptr_B_device_w1);
            cudaFree(ptr_C_device_w1);
            cudaFree(ptr_D_device_w1);
            cudaFree(lda_device_w1);
            cudaFree(ldb_device_w1);
            cudaFree(ldc_device_w1);
            cudaFree(ldd_device_w1);
        }

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
        bool owns_w2_descriptors = false;

        size_t num_problems_w2 = problem_sizes_w2.size();

        if (workspace && workspace->descriptor_capacity_w2 >= num_problems_w2) {
            // Use pre-allocated workspace
            problem_sizes_device_w2 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w2);
            ptr_A_device_w2 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w2);
            ptr_B_device_w2 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w2);
            ptr_C_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w2);
            ptr_D_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w2);
            lda_device_w2 = reinterpret_cast<int64_t*>(workspace->lda_device_w2);
            ldb_device_w2 = reinterpret_cast<int64_t*>(workspace->ldb_device_w2);
            ldc_device_w2 = reinterpret_cast<int64_t*>(workspace->ldc_device_w2);
            ldd_device_w2 = reinterpret_cast<int64_t*>(workspace->ldd_device_w2);
        } else if (workspace) {
            // Workspace too small - grow it
            if (workspace->problem_sizes_device_w2) {
                cudaFree(workspace->problem_sizes_device_w2);
                cudaFree(workspace->ptr_A_device_w2);
                cudaFree(workspace->ptr_B_device_w2);
                cudaFree(workspace->ptr_C_device_w2);
                cudaFree(workspace->ptr_D_device_w2);
                cudaFree(workspace->lda_device_w2);
                cudaFree(workspace->ldb_device_w2);
                cudaFree(workspace->ldc_device_w2);
                cudaFree(workspace->ldd_device_w2);
            }
            CUDA_CHECK(cudaMalloc(&workspace->problem_sizes_device_w2, num_problems_w2 * sizeof(cutlass::gemm::GemmCoord)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_A_device_w2, num_problems_w2 * sizeof(ElementA*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_B_device_w2, num_problems_w2 * sizeof(ElementB*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_C_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_D_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&workspace->lda_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldb_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldc_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldd_device_w2, num_problems_w2 * sizeof(int64_t)));
            workspace->descriptor_capacity_w2 = num_problems_w2;

            problem_sizes_device_w2 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w2);
            ptr_A_device_w2 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w2);
            ptr_B_device_w2 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w2);
            ptr_C_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w2);
            ptr_D_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w2);
            lda_device_w2 = reinterpret_cast<int64_t*>(workspace->lda_device_w2);
            ldb_device_w2 = reinterpret_cast<int64_t*>(workspace->ldb_device_w2);
            ldc_device_w2 = reinterpret_cast<int64_t*>(workspace->ldc_device_w2);
            ldd_device_w2 = reinterpret_cast<int64_t*>(workspace->ldd_device_w2);
        } else {
            // No workspace - allocate temporary buffers
            CUDA_CHECK(cudaMalloc(&problem_sizes_device_w2, num_problems_w2 * sizeof(cutlass::gemm::GemmCoord)));
            CUDA_CHECK(cudaMalloc(&ptr_A_device_w2, num_problems_w2 * sizeof(ElementA*)));
            CUDA_CHECK(cudaMalloc(&ptr_B_device_w2, num_problems_w2 * sizeof(ElementB*)));
            CUDA_CHECK(cudaMalloc(&ptr_C_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&ptr_D_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&lda_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldb_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldc_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldd_device_w2, num_problems_w2 * sizeof(int64_t)));
            owns_w2_descriptors = true;
        }

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
        bool owns_workspace_w2 = false;

        if (workspace && workspace->workspace_w2 && workspace->workspace_w2_size >= workspace_size_w2) {
            // Use pre-allocated workspace
            workspace_w2 = workspace->workspace_w2;
        } else if (workspace && workspace->workspace_w2) {
            // Workspace too small - grow it
            cudaFree(workspace->workspace_w2);
            CUDA_CHECK(cudaMalloc(&workspace->workspace_w2, workspace_size_w2));
            workspace->workspace_w2_size = workspace_size_w2;
            workspace_w2 = workspace->workspace_w2;
        } else if (workspace && workspace_size_w2 > 0) {
            // Workspace provided but no buffer allocated yet
            CUDA_CHECK(cudaMalloc(&workspace->workspace_w2, workspace_size_w2));
            workspace->workspace_w2_size = workspace_size_w2;
            workspace_w2 = workspace->workspace_w2;
        } else if (workspace_size_w2 > 0) {
            // No workspace - allocate temporary buffer
            CUDA_CHECK(cudaMalloc(&workspace_w2, workspace_size_w2));
            owns_workspace_w2 = true;
        }

        CUTLASS_CHECK(gemm_w2.initialize(args_w2, workspace_w2));
        CUTLASS_CHECK(gemm_w2.run());

        // Cleanup (only if we own the buffers)
        if (owns_workspace_w2 && workspace_w2) cudaFree(workspace_w2);
        if (owns_w2_descriptors) {
            cudaFree(problem_sizes_device_w2);
            cudaFree(ptr_A_device_w2);
            cudaFree(ptr_B_device_w2);
            cudaFree(ptr_C_device_w2);
            cudaFree(ptr_D_device_w2);
            cudaFree(lda_device_w2);
            cudaFree(ldb_device_w2);
            cudaFree(ldc_device_w2);
            cudaFree(ldd_device_w2);
        }

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

        // Debug: print first few routing weights for first group
        if (log_cutlass && i == 0 && M > 0) {
            std::vector<float> routing_cpu(std::min(M, int64_t(8)));
            cudaMemcpy(routing_cpu.data(), routing_ptr, routing_cpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::cerr << "[LB][MOE_CUTLASS_FUSED] Group 0 routing weights (M=" << M << ", hidden_dim=" << hidden_dim << "): [";
            for (size_t j = 0; j < routing_cpu.size(); ++j) {
                std::cerr << routing_cpu[j];
                if (j < routing_cpu.size() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;
        }

        int64_t total = M * hidden_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        // Fused bias + routing weight kernel (replaces 2 separate kernels)
        fused_bias_routing_kernel<<<blocks, threads>>>(output_ptr, bias_ptr, routing_ptr, M, hidden_dim);
    }

    // Cleanup (only if we own the buffer)
    if (owns_hidden_buffer && hidden_buffer) {
        CUDA_CHECK(cudaFree(hidden_buffer));
    }

    if (log_cutlass) {
        std::cerr << "[LB][MOE_CUTLASS_FUSED] Fused MoE forward completed:" << std::endl;
        std::cerr << "  W1: 3 kernels → 1 (GEMM+bias+GELU fused)" << std::endl;
        std::cerr << "  W2: 3 kernels → 2 (GEMM + fused bias+routing)" << std::endl;
        std::cerr << "  Total: 6 kernels → 3 (50% reduction)" << std::endl;
    }
}

// Variant that writes hidden to external buffer (no internal allocation)
void cutlass_grouped_moe_forward_with_hidden(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* b2_ptrs,
    const uintptr_t* hidden_ptrs,
    const uintptr_t* output_ptrs,
    const uintptr_t* routing_weight_ptrs,
    const int64_t* m_sizes,
    const int64_t* policy_ids,
    const int64_t* expert_ids,
    const int64_t* token_offsets,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim,
    MoEWorkspace* workspace
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
        std::cerr << "[LB][MOE_CUTLASS_FUSED] Starting fused grouped MoE forward (ext hidden):" << std::endl;
        std::cerr << "  group_count=" << group_count << std::endl;
        std::cerr << "  hidden_dim=" << hidden_dim << std::endl;
        std::cerr << "  ffn_dim=" << ffn_dim << std::endl;
    }

    // ========================================================================
    // W1 GEMM with BIAS + GELU Fusion
    // ========================================================================

    // Setup problem sizes and pointers for W1 GEMM
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_w1;
    std::vector<ElementA*> ptr_A_w1;
    std::vector<ElementB*> ptr_B_w1;
    std::vector<ElementOutput*> ptr_C_w1;  // Bias
    std::vector<ElementOutput*> ptr_D_w1;  // Output -> external hidden buffer
    std::vector<int64_t> lda_w1, ldb_w1, ldc_w1, ldd_w1;

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
        ptr_D_w1.push_back(reinterpret_cast<ElementOutput*>(hidden_ptrs[i]));

        lda_w1.push_back(K);  // Row-major input
        ldb_w1.push_back(K);  // Column-major weights
        ldc_w1.push_back(0);  // Bias stride = 0 (broadcast across rows)
        ldd_w1.push_back(N);  // Row-major output

        if (log_gemm) {
            std::cerr << "[LB][MOE_GEMM_FUSED] W1 group=" << i
                      << " M=" << M << " N=" << N << " K=" << K << std::endl;
        }
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
        bool owns_w1_descriptors = false;

        size_t num_problems_w1 = problem_sizes_w1.size();

        if (workspace && workspace->descriptor_capacity_w1 >= num_problems_w1) {
            // Use pre-allocated workspace
            problem_sizes_device_w1 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w1);
            ptr_A_device_w1 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w1);
            ptr_B_device_w1 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w1);
            ptr_C_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w1);
            ptr_D_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w1);
            lda_device_w1 = reinterpret_cast<int64_t*>(workspace->lda_device_w1);
            ldb_device_w1 = reinterpret_cast<int64_t*>(workspace->ldb_device_w1);
            ldc_device_w1 = reinterpret_cast<int64_t*>(workspace->ldc_device_w1);
            ldd_device_w1 = reinterpret_cast<int64_t*>(workspace->ldd_device_w1);
        } else if (workspace) {
            // Workspace too small - grow it
            if (workspace->problem_sizes_device_w1) {
                cudaFree(workspace->problem_sizes_device_w1);
                cudaFree(workspace->ptr_A_device_w1);
                cudaFree(workspace->ptr_B_device_w1);
                cudaFree(workspace->ptr_C_device_w1);
                cudaFree(workspace->ptr_D_device_w1);
                cudaFree(workspace->lda_device_w1);
                cudaFree(workspace->ldb_device_w1);
                cudaFree(workspace->ldc_device_w1);
                cudaFree(workspace->ldd_device_w1);
            }
            CUDA_CHECK(cudaMalloc(&workspace->problem_sizes_device_w1, num_problems_w1 * sizeof(cutlass::gemm::GemmCoord)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_A_device_w1, num_problems_w1 * sizeof(ElementA*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_B_device_w1, num_problems_w1 * sizeof(ElementB*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_C_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_D_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&workspace->lda_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldb_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldc_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldd_device_w1, num_problems_w1 * sizeof(int64_t)));
            workspace->descriptor_capacity_w1 = num_problems_w1;

            problem_sizes_device_w1 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w1);
            ptr_A_device_w1 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w1);
            ptr_B_device_w1 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w1);
            ptr_C_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w1);
            ptr_D_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w1);
            lda_device_w1 = reinterpret_cast<int64_t*>(workspace->lda_device_w1);
            ldb_device_w1 = reinterpret_cast<int64_t*>(workspace->ldb_device_w1);
            ldc_device_w1 = reinterpret_cast<int64_t*>(workspace->ldc_device_w1);
            ldd_device_w1 = reinterpret_cast<int64_t*>(workspace->ldd_device_w1);
        } else {
            // No workspace - allocate temporary buffers
            CUDA_CHECK(cudaMalloc(&problem_sizes_device_w1, num_problems_w1 * sizeof(cutlass::gemm::GemmCoord)));
            CUDA_CHECK(cudaMalloc(&ptr_A_device_w1, num_problems_w1 * sizeof(ElementA*)));
            CUDA_CHECK(cudaMalloc(&ptr_B_device_w1, num_problems_w1 * sizeof(ElementB*)));
            CUDA_CHECK(cudaMalloc(&ptr_C_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&ptr_D_device_w1, num_problems_w1 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&lda_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldb_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldc_device_w1, num_problems_w1 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldd_device_w1, num_problems_w1 * sizeof(int64_t)));
            owns_w1_descriptors = true;
        }

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
            ptr_C_device_w1,
            ptr_D_device_w1,
            lda_device_w1,
            ldb_device_w1,
            ldc_device_w1,
            ldd_device_w1,
            problem_sizes_w1.data()
        );

        size_t workspace_size_w1 = gemm_w1.get_workspace_size(args_w1);
        void* workspace_w1 = nullptr;
        bool owns_workspace_w1 = false;

        if (workspace && workspace->workspace_w1 && workspace->workspace_w1_size >= workspace_size_w1) {
            // Use pre-allocated workspace
            workspace_w1 = workspace->workspace_w1;
        } else if (workspace && workspace->workspace_w1) {
            // Workspace too small - grow it
            cudaFree(workspace->workspace_w1);
            CUDA_CHECK(cudaMalloc(&workspace->workspace_w1, workspace_size_w1));
            workspace->workspace_w1_size = workspace_size_w1;
            workspace_w1 = workspace->workspace_w1;
        } else if (workspace && workspace_size_w1 > 0) {
            // Workspace provided but no buffer allocated yet
            CUDA_CHECK(cudaMalloc(&workspace->workspace_w1, workspace_size_w1));
            workspace->workspace_w1_size = workspace_size_w1;
            workspace_w1 = workspace->workspace_w1;
        } else if (workspace_size_w1 > 0) {
            // No workspace - allocate temporary buffer
            CUDA_CHECK(cudaMalloc(&workspace_w1, workspace_size_w1));
            owns_workspace_w1 = true;
        }

        CUTLASS_CHECK(gemm_w1.initialize(args_w1, workspace_w1));
        CUTLASS_CHECK(gemm_w1.run());

        // Cleanup (only if we own the buffers)
        if (owns_workspace_w1 && workspace_w1) cudaFree(workspace_w1);
        if (owns_w1_descriptors) {
            cudaFree(problem_sizes_device_w1);
            cudaFree(ptr_A_device_w1);
            cudaFree(ptr_B_device_w1);
            cudaFree(ptr_C_device_w1);
            cudaFree(ptr_D_device_w1);
            cudaFree(lda_device_w1);
            cudaFree(ldb_device_w1);
            cudaFree(ldc_device_w1);
            cudaFree(ldd_device_w1);
        }

        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS_FUSED] W1 fused GEMM completed" << std::endl;
        }
    }

    // ========================================================================
    // W2 GEMM
    // ========================================================================

    std::vector<cutlass::gemm::GemmCoord> problem_sizes_w2;
    std::vector<ElementA*> ptr_A_w2;
    std::vector<ElementB*> ptr_B_w2;
    std::vector<ElementOutput*> ptr_C_w2;  // Bias (unused, beta=0)
    std::vector<ElementOutput*> ptr_D_w2;  // Output
    std::vector<int64_t> lda_w2, ldb_w2, ldc_w2, ldd_w2;

    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        int64_t K = ffn_dim;
        int64_t N = hidden_dim;

        if (M == 0) continue;

        problem_sizes_w2.push_back(cutlass::gemm::GemmCoord(M, N, K));
        ptr_A_w2.push_back(reinterpret_cast<ElementA*>(hidden_ptrs[i]));
        ptr_B_w2.push_back(reinterpret_cast<ElementB*>(w2_ptrs[i]));
        ptr_C_w2.push_back(nullptr);  // No C matrix
        ptr_D_w2.push_back(reinterpret_cast<ElementOutput*>(output_ptrs[i]));

        lda_w2.push_back(K);
        ldb_w2.push_back(K);
        ldc_w2.push_back(N);
        ldd_w2.push_back(N);
    }

    if (!problem_sizes_w2.empty()) {
        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS_FUSED] Launching W2 GEMM with "
                      << problem_sizes_w2.size() << " problems" << std::endl;
        }

        cutlass::gemm::GemmCoord* problem_sizes_device_w2 = nullptr;
        ElementA** ptr_A_device_w2 = nullptr;
        ElementB** ptr_B_device_w2 = nullptr;
        ElementOutput** ptr_C_device_w2 = nullptr;
        ElementOutput** ptr_D_device_w2 = nullptr;
        int64_t* lda_device_w2 = nullptr;
        int64_t* ldb_device_w2 = nullptr;
        int64_t* ldc_device_w2 = nullptr;
        int64_t* ldd_device_w2 = nullptr;
        bool owns_w2_descriptors = false;

        size_t num_problems_w2 = problem_sizes_w2.size();

        if (workspace && workspace->descriptor_capacity_w2 >= num_problems_w2) {
            // Use pre-allocated workspace
            problem_sizes_device_w2 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w2);
            ptr_A_device_w2 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w2);
            ptr_B_device_w2 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w2);
            ptr_C_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w2);
            ptr_D_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w2);
            lda_device_w2 = reinterpret_cast<int64_t*>(workspace->lda_device_w2);
            ldb_device_w2 = reinterpret_cast<int64_t*>(workspace->ldb_device_w2);
            ldc_device_w2 = reinterpret_cast<int64_t*>(workspace->ldc_device_w2);
            ldd_device_w2 = reinterpret_cast<int64_t*>(workspace->ldd_device_w2);
        } else if (workspace) {
            // Workspace too small - grow it
            if (workspace->problem_sizes_device_w2) {
                cudaFree(workspace->problem_sizes_device_w2);
                cudaFree(workspace->ptr_A_device_w2);
                cudaFree(workspace->ptr_B_device_w2);
                cudaFree(workspace->ptr_C_device_w2);
                cudaFree(workspace->ptr_D_device_w2);
                cudaFree(workspace->lda_device_w2);
                cudaFree(workspace->ldb_device_w2);
                cudaFree(workspace->ldc_device_w2);
                cudaFree(workspace->ldd_device_w2);
            }
            CUDA_CHECK(cudaMalloc(&workspace->problem_sizes_device_w2, num_problems_w2 * sizeof(cutlass::gemm::GemmCoord)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_A_device_w2, num_problems_w2 * sizeof(ElementA*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_B_device_w2, num_problems_w2 * sizeof(ElementB*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_C_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&workspace->ptr_D_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&workspace->lda_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldb_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldc_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&workspace->ldd_device_w2, num_problems_w2 * sizeof(int64_t)));
            workspace->descriptor_capacity_w2 = num_problems_w2;

            problem_sizes_device_w2 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w2);
            ptr_A_device_w2 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w2);
            ptr_B_device_w2 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w2);
            ptr_C_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w2);
            ptr_D_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w2);
            lda_device_w2 = reinterpret_cast<int64_t*>(workspace->lda_device_w2);
            ldb_device_w2 = reinterpret_cast<int64_t*>(workspace->ldb_device_w2);
            ldc_device_w2 = reinterpret_cast<int64_t*>(workspace->ldc_device_w2);
            ldd_device_w2 = reinterpret_cast<int64_t*>(workspace->ldd_device_w2);
        } else {
            // No workspace - allocate temporary buffers
            CUDA_CHECK(cudaMalloc(&problem_sizes_device_w2, num_problems_w2 * sizeof(cutlass::gemm::GemmCoord)));
            CUDA_CHECK(cudaMalloc(&ptr_A_device_w2, num_problems_w2 * sizeof(ElementA*)));
            CUDA_CHECK(cudaMalloc(&ptr_B_device_w2, num_problems_w2 * sizeof(ElementB*)));
            CUDA_CHECK(cudaMalloc(&ptr_C_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&ptr_D_device_w2, num_problems_w2 * sizeof(ElementOutput*)));
            CUDA_CHECK(cudaMalloc(&lda_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldb_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldc_device_w2, num_problems_w2 * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&ldd_device_w2, num_problems_w2 * sizeof(int64_t)));
            owns_w2_descriptors = true;
        }

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
        bool owns_workspace_w2 = false;

        if (workspace && workspace->workspace_w2 && workspace->workspace_w2_size >= workspace_size_w2) {
            // Use pre-allocated workspace
            workspace_w2 = workspace->workspace_w2;
        } else if (workspace && workspace->workspace_w2) {
            // Workspace too small - grow it
            cudaFree(workspace->workspace_w2);
            CUDA_CHECK(cudaMalloc(&workspace->workspace_w2, workspace_size_w2));
            workspace->workspace_w2_size = workspace_size_w2;
            workspace_w2 = workspace->workspace_w2;
        } else if (workspace && workspace_size_w2 > 0) {
            // Workspace provided but no buffer allocated yet
            CUDA_CHECK(cudaMalloc(&workspace->workspace_w2, workspace_size_w2));
            workspace->workspace_w2_size = workspace_size_w2;
            workspace_w2 = workspace->workspace_w2;
        } else if (workspace_size_w2 > 0) {
            // No workspace - allocate temporary buffer
            CUDA_CHECK(cudaMalloc(&workspace_w2, workspace_size_w2));
            owns_workspace_w2 = true;
        }

        CUTLASS_CHECK(gemm_w2.initialize(args_w2, workspace_w2));
        CUTLASS_CHECK(gemm_w2.run());

        // Cleanup (only if we own the buffers)
        if (owns_workspace_w2 && workspace_w2) cudaFree(workspace_w2);
        if (owns_w2_descriptors) {
            cudaFree(problem_sizes_device_w2);
            cudaFree(ptr_A_device_w2);
            cudaFree(ptr_B_device_w2);
            cudaFree(ptr_C_device_w2);
            cudaFree(ptr_D_device_w2);
            cudaFree(lda_device_w2);
            cudaFree(ldb_device_w2);
            cudaFree(ldc_device_w2);
            cudaFree(ldd_device_w2);
        }

        if (log_cutlass) {
            std::cerr << "[LB][MOE_CUTLASS_FUSED] W2 GEMM completed" << std::endl;
        }
    }

    // Apply fused bias + routing weights to W2 outputs
    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        cutlass::half_t* output_ptr = reinterpret_cast<cutlass::half_t*>(output_ptrs[i]);
        const cutlass::half_t* bias_ptr = reinterpret_cast<const cutlass::half_t*>(b2_ptrs[i]);
        const float* routing_ptr = reinterpret_cast<const float*>(routing_weight_ptrs[i]);

        int64_t total = M * hidden_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        fused_bias_routing_kernel<<<blocks, threads>>>(output_ptr, bias_ptr, routing_ptr, M, hidden_dim);
    }

    if (log_cutlass) {
        std::cerr << "[LB][MOE_CUTLASS_FUSED] Fused MoE forward (ext hidden) completed" << std::endl;
    }
}

} // namespace moe
} // namespace lb

// Re-open namespace for device-only descriptor path and helpers
namespace lb {
namespace moe {

// Host-pinned descriptor buffers (moved inside namespace)
// (Old copies removed; now defined within reopened namespace above)
static void ensure_host_pinned_capacity_w1(MoEWorkspace* ws, size_t n) {
    if (!ws) return;
    if (ws->host_descriptor_capacity_w1 >= n) return;
    // Free existing
    if (ws->host_problem_sizes_w1) cudaFreeHost(ws->host_problem_sizes_w1);
    if (ws->host_ptr_A_w1) cudaFreeHost(ws->host_ptr_A_w1);
    if (ws->host_ptr_B_w1) cudaFreeHost(ws->host_ptr_B_w1);
    if (ws->host_ptr_C_w1) cudaFreeHost(ws->host_ptr_C_w1);
    if (ws->host_ptr_D_w1) cudaFreeHost(ws->host_ptr_D_w1);
    if (ws->host_lda_w1) cudaFreeHost(ws->host_lda_w1);
    if (ws->host_ldb_w1) cudaFreeHost(ws->host_ldb_w1);
    if (ws->host_ldc_w1) cudaFreeHost(ws->host_ldc_w1);
    if (ws->host_ldd_w1) cudaFreeHost(ws->host_ldd_w1);

    CUDA_CHECK(cudaHostAlloc(&ws->host_problem_sizes_w1, n * sizeof(cutlass::gemm::GemmCoord), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ptr_A_w1, n * sizeof(ElementA*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ptr_B_w1, n * sizeof(ElementB*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ptr_C_w1, n * sizeof(ElementOutput*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ptr_D_w1, n * sizeof(ElementOutput*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_lda_w1, n * sizeof(int64_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ldb_w1, n * sizeof(int64_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ldc_w1, n * sizeof(int64_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ldd_w1, n * sizeof(int64_t), cudaHostAllocDefault));
    ws->host_descriptor_capacity_w1 = n;
}

static void ensure_host_pinned_capacity_w2(MoEWorkspace* ws, size_t n) {
    if (!ws) return;
    if (ws->host_descriptor_capacity_w2 >= n) return;
    if (ws->host_problem_sizes_w2) cudaFreeHost(ws->host_problem_sizes_w2);
    if (ws->host_ptr_A_w2) cudaFreeHost(ws->host_ptr_A_w2);
    if (ws->host_ptr_B_w2) cudaFreeHost(ws->host_ptr_B_w2);
    if (ws->host_ptr_C_w2) cudaFreeHost(ws->host_ptr_C_w2);
    if (ws->host_ptr_D_w2) cudaFreeHost(ws->host_ptr_D_w2);
    if (ws->host_lda_w2) cudaFreeHost(ws->host_lda_w2);
    if (ws->host_ldb_w2) cudaFreeHost(ws->host_ldb_w2);
    if (ws->host_ldc_w2) cudaFreeHost(ws->host_ldc_w2);
    if (ws->host_ldd_w2) cudaFreeHost(ws->host_ldd_w2);

    CUDA_CHECK(cudaHostAlloc(&ws->host_problem_sizes_w2, n * sizeof(cutlass::gemm::GemmCoord), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ptr_A_w2, n * sizeof(ElementA*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ptr_B_w2, n * sizeof(ElementB*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ptr_C_w2, n * sizeof(ElementOutput*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ptr_D_w2, n * sizeof(ElementOutput*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_lda_w2, n * sizeof(int64_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ldb_w2, n * sizeof(int64_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ldc_w2, n * sizeof(int64_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&ws->host_ldd_w2, n * sizeof(int64_t), cudaHostAllocDefault));
    ws->host_descriptor_capacity_w2 = n;
}
__global__ void compute_hidden_offsets(const int64_t* __restrict__ m_sizes,
                                       int64_t G, int64_t N,
                                       int64_t* __restrict__ offsets) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int64_t acc = 0;
        for (int64_t g = 0; g < G; ++g) {
            offsets[g] = acc;
            acc += m_sizes[g] * N;
        }
    }
}

__global__ void build_w1_descriptors_kernel(
    const int64_t* __restrict__ m_sizes,
    const int64_t* __restrict__ policy_ids,
    const int64_t* __restrict__ expert_ids,
    const int64_t* __restrict__ token_offsets,
    int64_t G, int64_t H, int64_t F,
    uintptr_t input_base,
    uintptr_t hidden_base,
    const int64_t* __restrict__ hidden_offsets,
    const uint64_t* __restrict__ w1_tbl,
    const uint64_t* __restrict__ b1_tbl,
    int64_t P, int64_t E,
    cutlass::gemm::GemmCoord* __restrict__ problem_sizes,
    ElementA** __restrict__ ptr_A,
    ElementB** __restrict__ ptr_B,
    ElementOutput** __restrict__ ptr_C,
    ElementOutput** __restrict__ ptr_D,
    int64_t* __restrict__ lda,
    int64_t* __restrict__ ldb,
    int64_t* __restrict__ ldc,
    int64_t* __restrict__ ldd) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    int64_t M = m_sizes[g];
    int64_t pi = policy_ids[g];
    int64_t ei = expert_ids[g];
    int64_t off = token_offsets[g];
    int64_t hidden_off = hidden_offsets[g];
    problem_sizes[g] = cutlass::gemm::GemmCoord(M, F, H);
    ptr_A[g] = reinterpret_cast<ElementA*>(input_base + static_cast<uintptr_t>(off) * H * sizeof(ElementA));
    ptr_B[g] = reinterpret_cast<ElementB*>(w1_tbl[pi * E + ei]);
    ptr_C[g] = reinterpret_cast<ElementOutput*>(b1_tbl[pi * E + ei]);
    ptr_D[g] = reinterpret_cast<ElementOutput*>(hidden_base + static_cast<uintptr_t>(hidden_off) * sizeof(ElementOutput));
    lda[g] = H; ldb[g] = H; ldc[g] = 0; ldd[g] = F;
}

__global__ void build_w2_descriptors_kernel(
    const int64_t* __restrict__ m_sizes,
    const int64_t* __restrict__ policy_ids,
    const int64_t* __restrict__ expert_ids,
    const int64_t* __restrict__ token_offsets,
    int64_t G, int64_t H, int64_t F,
    uintptr_t hidden_base,
    uintptr_t output_base,
    const int64_t* __restrict__ hidden_offsets,
    const uint64_t* __restrict__ w2_tbl,
    int64_t P, int64_t E,
    cutlass::gemm::GemmCoord* __restrict__ problem_sizes,
    ElementA** __restrict__ ptr_A,
    ElementB** __restrict__ ptr_B,
    ElementOutput** __restrict__ ptr_C,
    ElementOutput** __restrict__ ptr_D,
    int64_t* __restrict__ lda,
    int64_t* __restrict__ ldb,
    int64_t* __restrict__ ldc,
    int64_t* __restrict__ ldd) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    int64_t M = m_sizes[g];
    int64_t pi = policy_ids[g];
    int64_t ei = expert_ids[g];
    int64_t off = token_offsets[g];
    int64_t hidden_off = hidden_offsets[g];
    problem_sizes[g] = cutlass::gemm::GemmCoord(M, H, F);
    ptr_A[g] = reinterpret_cast<ElementA*>(hidden_base + static_cast<uintptr_t>(hidden_off) * sizeof(ElementA));
    ptr_B[g] = reinterpret_cast<ElementB*>(w2_tbl[pi * E + ei]);
    ptr_C[g] = nullptr;
    ptr_D[g] = reinterpret_cast<ElementOutput*>(output_base + static_cast<uintptr_t>(off) * H * sizeof(ElementOutput));
    lda[g] = F; ldb[g] = F; ldc[g] = 0; ldd[g] = H;
}

template <typename T>
__global__ void fused_bias_routing_groups_kernel(
    T* __restrict__ out_base,
    const float* __restrict__ routing_base,
    const uint64_t* __restrict__ b2_tbl,
    const int64_t* __restrict__ m_sizes,
    const int64_t* __restrict__ token_offsets,
    const int64_t* __restrict__ policy_ids,
    const int64_t* __restrict__ expert_ids,
    int64_t G,
    int64_t H,
    int64_t E)
{
    int g = blockIdx.y;
    if (g >= G) return;
    int64_t M = m_sizes[g];
    int64_t off = token_offsets[g];
    int64_t pi = policy_ids[g];
    int64_t ei = expert_ids[g];
    const T* __restrict__ bias = reinterpret_cast<const T*>(b2_tbl[pi * E + ei]);

    int64_t total = M * H;
    int idx = threadIdx.x;
    for (; idx < total; idx += blockDim.x) {
        int64_t row = idx / H;
        int64_t col = idx % H;
        T* data_row = out_base + (off + row) * H;
        float val = float(data_row[col]) + float(bias[col]);
        float rw = routing_base[off + row];
        data_row[col] = T(val * rw);
    }
}

void cutlass_grouped_moe_forward_with_hidden_device(
    uintptr_t input_base,
    uintptr_t hidden_base,
    uintptr_t output_base,
    uintptr_t routing_base,
    const uint64_t* w1_ptrs_table,
    const uint64_t* w2_ptrs_table,
    const uint64_t* b1_ptrs_table,
    const uint64_t* b2_ptrs_table,
    int64_t num_policies,
    int64_t num_experts,
    const int64_t* m_sizes_dev,
    const int64_t* policy_ids_dev,
    const int64_t* expert_ids_dev,
    const int64_t* token_offsets_dev,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim,
    MoEWorkspace* workspace) {

    if (group_count == 0) return;
    const bool log_cutlass = lb_moe_cutlass_should_log();

    // Prepare device descriptor buffers (use workspace if available, else allocate temporary)
    cutlass::gemm::GemmCoord* problem_sizes_device_w1 = nullptr;
    ElementA** ptr_A_device_w1 = nullptr;
    ElementB** ptr_B_device_w1 = nullptr;
    ElementOutput** ptr_C_device_w1 = nullptr;
    ElementOutput** ptr_D_device_w1 = nullptr;
    int64_t* lda_device_w1 = nullptr;
    int64_t* ldb_device_w1 = nullptr;
    int64_t* ldc_device_w1 = nullptr;
    int64_t* ldd_device_w1 = nullptr;

    cutlass::gemm::GemmCoord* problem_sizes_device_w2 = nullptr;
    ElementA** ptr_A_device_w2 = nullptr;
    ElementB** ptr_B_device_w2 = nullptr;
    ElementOutput** ptr_C_device_w2 = nullptr;
    ElementOutput** ptr_D_device_w2 = nullptr;
    int64_t* lda_device_w2 = nullptr;
    int64_t* ldb_device_w2 = nullptr;
    int64_t* ldc_device_w2 = nullptr;
    int64_t* ldd_device_w2 = nullptr;

    bool owns_desc_w1 = false, owns_desc_w2 = false;
    if (workspace && workspace->descriptor_capacity_w1 >= static_cast<size_t>(group_count) &&
        workspace->descriptor_capacity_w2 >= static_cast<size_t>(group_count)) {
        problem_sizes_device_w1 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w1);
        ptr_A_device_w1 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w1);
        ptr_B_device_w1 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w1);
        ptr_C_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w1);
        ptr_D_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w1);
        lda_device_w1 = reinterpret_cast<int64_t*>(workspace->lda_device_w1);
        ldb_device_w1 = reinterpret_cast<int64_t*>(workspace->ldb_device_w1);
        ldc_device_w1 = reinterpret_cast<int64_t*>(workspace->ldc_device_w1);
        ldd_device_w1 = reinterpret_cast<int64_t*>(workspace->ldd_device_w1);

        problem_sizes_device_w2 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w2);
        ptr_A_device_w2 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w2);
        ptr_B_device_w2 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w2);
        ptr_C_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w2);
        ptr_D_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w2);
        lda_device_w2 = reinterpret_cast<int64_t*>(workspace->lda_device_w2);
        ldb_device_w2 = reinterpret_cast<int64_t*>(workspace->ldb_device_w2);
        ldc_device_w2 = reinterpret_cast<int64_t*>(workspace->ldc_device_w2);
        ldd_device_w2 = reinterpret_cast<int64_t*>(workspace->ldd_device_w2);
    } else {
        // Allocate temporary descriptor buffers
        CUDA_CHECK(cudaMalloc(&problem_sizes_device_w1, sizeof(cutlass::gemm::GemmCoord) * group_count));
        CUDA_CHECK(cudaMalloc(&ptr_A_device_w1, sizeof(ElementA*) * group_count));
        CUDA_CHECK(cudaMalloc(&ptr_B_device_w1, sizeof(ElementB*) * group_count));
        CUDA_CHECK(cudaMalloc(&ptr_C_device_w1, sizeof(ElementOutput*) * group_count));
        CUDA_CHECK(cudaMalloc(&ptr_D_device_w1, sizeof(ElementOutput*) * group_count));
        CUDA_CHECK(cudaMalloc(&lda_device_w1, sizeof(int64_t) * group_count));
        CUDA_CHECK(cudaMalloc(&ldb_device_w1, sizeof(int64_t) * group_count));
        CUDA_CHECK(cudaMalloc(&ldc_device_w1, sizeof(int64_t) * group_count));
        CUDA_CHECK(cudaMalloc(&ldd_device_w1, sizeof(int64_t) * group_count));
        owns_desc_w1 = true;

        CUDA_CHECK(cudaMalloc(&problem_sizes_device_w2, sizeof(cutlass::gemm::GemmCoord) * group_count));
        CUDA_CHECK(cudaMalloc(&ptr_A_device_w2, sizeof(ElementA*) * group_count));
        CUDA_CHECK(cudaMalloc(&ptr_B_device_w2, sizeof(ElementB*) * group_count));
        CUDA_CHECK(cudaMalloc(&ptr_C_device_w2, sizeof(ElementOutput*) * group_count));
        CUDA_CHECK(cudaMalloc(&ptr_D_device_w2, sizeof(ElementOutput*) * group_count));
        CUDA_CHECK(cudaMalloc(&lda_device_w2, sizeof(int64_t) * group_count));
        CUDA_CHECK(cudaMalloc(&ldb_device_w2, sizeof(int64_t) * group_count));
        CUDA_CHECK(cudaMalloc(&ldc_device_w2, sizeof(int64_t) * group_count));
        CUDA_CHECK(cudaMalloc(&ldd_device_w2, sizeof(int64_t) * group_count));
        owns_desc_w2 = true;
    }

    // Allocate device offsets
    int64_t* hidden_offsets_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&hidden_offsets_dev, sizeof(int64_t) * group_count));
    // Compute exclusive offsets of hidden blocks
    compute_hidden_offsets<<<1,1>>>(m_sizes_dev, group_count, ffn_dim, hidden_offsets_dev);

    const int threads = 256;
    const int blocks = static_cast<int>((group_count + threads - 1) / threads);

    // Build W1 descriptors on device
    build_w1_descriptors_kernel<<<blocks, threads>>>(
        m_sizes_dev, policy_ids_dev, expert_ids_dev, token_offsets_dev,
        group_count, hidden_dim, ffn_dim,
        input_base, hidden_base, hidden_offsets_dev,
        w1_ptrs_table, b1_ptrs_table, num_policies, num_experts,
        problem_sizes_device_w1,
        ptr_A_device_w1,
        ptr_B_device_w1,
        ptr_C_device_w1,
        ptr_D_device_w1,
        lda_device_w1,
        ldb_device_w1,
        ldc_device_w1,
        ldd_device_w1);

    // Run W1 GEMM
    GroupedGemmW1 gemm_w1;
    int threadblock_count_w1 = static_cast<int>(group_count);
    typename GroupedGemmW1::EpilogueOutputOp::Params epilogue_params_w1(1.0f, 1.0f);
    typename GroupedGemmW1::Arguments args_w1(
        problem_sizes_device_w1,
        group_count,
        threadblock_count_w1,
        epilogue_params_w1,
        ptr_A_device_w1,
        ptr_B_device_w1,
        ptr_C_device_w1,
        ptr_D_device_w1,
        lda_device_w1,
        ldb_device_w1,
        ldc_device_w1,
        ldd_device_w1,
        nullptr);

    size_t workspace_size_w1 = gemm_w1.get_workspace_size(args_w1);
    void* dev_workspace_w1 = nullptr;
    bool owns_ws_w1 = false;
    if (workspace && workspace->workspace_w1 && workspace->workspace_w1_size >= workspace_size_w1) {
        dev_workspace_w1 = workspace->workspace_w1;
    } else if (workspace_size_w1 > 0) {
        CUDA_CHECK(cudaMalloc(&dev_workspace_w1, workspace_size_w1));
        owns_ws_w1 = true;
    }
    CUTLASS_CHECK(gemm_w1.initialize(args_w1, dev_workspace_w1));
    CUTLASS_CHECK(gemm_w1.run());
    if (owns_ws_w1 && dev_workspace_w1) cudaFree(dev_workspace_w1);

    // Build W2 descriptors on device
    build_w2_descriptors_kernel<<<blocks, threads>>>(
        m_sizes_dev, policy_ids_dev, expert_ids_dev, token_offsets_dev,
        group_count, hidden_dim, ffn_dim,
        hidden_base, output_base, hidden_offsets_dev,
        w2_ptrs_table, num_policies, num_experts,
        problem_sizes_device_w2,
        ptr_A_device_w2,
        ptr_B_device_w2,
        ptr_C_device_w2,
        ptr_D_device_w2,
        lda_device_w2,
        ldb_device_w2,
        ldc_device_w2,
        ldd_device_w2);

    // Run W2 GEMM
    GroupedGemmW2 gemm_w2;
    int threadblock_count_w2 = static_cast<int>(group_count);
    typename GroupedGemmW2::EpilogueOutputOp::Params epilogue_params_w2(1.0f, 0.0f);
    typename GroupedGemmW2::Arguments args_w2(
        problem_sizes_device_w2,
        group_count,
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
        nullptr);

    size_t workspace_size_w2 = gemm_w2.get_workspace_size(args_w2);
    void* dev_workspace_w2 = nullptr;
    bool owns_ws_w2 = false;
    if (workspace && workspace->workspace_w2 && workspace->workspace_w2_size >= workspace_size_w2) {
        dev_workspace_w2 = workspace->workspace_w2;
    } else if (workspace_size_w2 > 0) {
        CUDA_CHECK(cudaMalloc(&dev_workspace_w2, workspace_size_w2));
        owns_ws_w2 = true;
    }
    CUTLASS_CHECK(gemm_w2.initialize(args_w2, dev_workspace_w2));
    CUTLASS_CHECK(gemm_w2.run());
    if (owns_ws_w2 && dev_workspace_w2) cudaFree(dev_workspace_w2);

    // Apply fused bias + routing scaling across all groups
    {
        dim3 block(256, 1, 1);
        dim3 grid(1, static_cast<unsigned>(group_count), 1);
        fused_bias_routing_groups_kernel<ElementOutput><<<grid, block>>>(
            reinterpret_cast<ElementOutput*>(output_base),
            reinterpret_cast<const float*>(routing_base),
            b2_ptrs_table,
            m_sizes_dev,
            token_offsets_dev,
            policy_ids_dev,
            expert_ids_dev,
            group_count,
            hidden_dim,
            num_experts);
    }

    // Free temp buffers if owned
    cudaFree(hidden_offsets_dev);
    if (owns_desc_w1) {
        cudaFree(problem_sizes_device_w1);
        cudaFree(ptr_A_device_w1);
        cudaFree(ptr_B_device_w1);
        cudaFree(ptr_C_device_w1);
        cudaFree(ptr_D_device_w1);
        cudaFree(lda_device_w1);
        cudaFree(ldb_device_w1);
        cudaFree(ldc_device_w1);
        cudaFree(ldd_device_w1);
    }
    if (owns_desc_w2) {
        cudaFree(problem_sizes_device_w2);
        cudaFree(ptr_A_device_w2);
        cudaFree(ptr_B_device_w2);
        cudaFree(ptr_C_device_w2);
        cudaFree(ptr_D_device_w2);
        cudaFree(lda_device_w2);
        cudaFree(ldb_device_w2);
        cudaFree(ldc_device_w2);
        cudaFree(ldd_device_w2);
    }

    if (log_cutlass) {
        std::cerr << "[LB][MOE_CUTLASS_FUSED] Device-meta MoE forward completed" << std::endl;
    }
}

} // namespace moe
} // namespace lb
