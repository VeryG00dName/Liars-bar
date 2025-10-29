#include "moe_cutlass_backward.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <vector>

namespace lb {
namespace moe {

// ============================================================================
// Environment Variable Helpers
// ============================================================================

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

static bool LB_MOE_LOG_BACKWARD_GEMM() {
    static int value = lb_env_int("LB_MOE_LOG_BACKWARD", 0);
    return value != 0;
}

static bool LB_MOE_SYNC() {
    static int value = lb_env_int("LB_MOE_SYNC", 0);
    return value != 0;
}

// ============================================================================
// Error Checking Macros
// ============================================================================

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
// CUTLASS Type Definitions (Common)
// ============================================================================

// FP16 input/output, FP32 accumulation (matches forward)
using ElementInputFP16 = cutlass::half_t;
using ElementOutputFP16 = cutlass::half_t;
using ElementOutputFP32 = float;  // For weight gradients
using ElementAccumulator = float;
using ElementCompute = float;

// MMA configuration for Ampere (sm_80)
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
constexpr int Stages = 4;

// ============================================================================
// Backward GEMM #1: dW2 = Gỹᵀ @ H
// ============================================================================

// dW2: Output is FP32 for weight gradient accumulation
// A: H [M,F] viewed as [F,M] ColumnMajor (transposed)
// B: Gỹ [M,H] RowMajor
// D: dW2 [F,H] ColumnMajor (to match forward W2 storage)

using LayoutA_dW2 = cutlass::layout::ColumnMajor;  // H transposed
using LayoutB_dW2 = cutlass::layout::RowMajor;     // Gỹ
using LayoutC_dW2 = cutlass::layout::ColumnMajor;  // dW2 output

using EpilogueOp_dW2 = cutlass::epilogue::thread::LinearCombination<
    ElementOutputFP32,
    128 / cutlass::sizeof_bits<ElementOutputFP32>::value,
    ElementAccumulator,
    ElementCompute
>;

using GemmKernel_dW2 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInputFP16, LayoutA_dW2, cutlass::ComplexTransform::kNone, 8,
    ElementInputFP16, LayoutB_dW2, cutlass::ComplexTransform::kNone, 8,
    ElementOutputFP32, LayoutC_dW2,
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp_dW2,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemm_dW2 = cutlass::gemm::device::GemmGrouped<GemmKernel_dW2>;

void cutlass_grouped_gemm_dW2(
    const uintptr_t* gy_tilde_ptrs,
    const uintptr_t* hidden_ptrs,
    const uintptr_t* grad_w2_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
) {
    if (group_count == 0) return;

    const bool log_gemm = LB_MOE_LOG_BACKWARD_GEMM();

    if (log_gemm) {
        std::cerr << "[MoE Backward] dW2 GEMM: group_count=" << group_count
                  << " H=" << hidden_dim << " F=" << ffn_dim << std::endl;
    }

    // Setup problem sizes and pointers
    // Problem size: {F, H, M} (dW2 is [F,H])
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<ElementInputFP16*> ptr_A;  // H (transposed)
    std::vector<ElementInputFP16*> ptr_B;  // Gỹ
    std::vector<ElementOutputFP32*> ptr_C;  // Nullptr (no C matrix)
    std::vector<ElementOutputFP32*> ptr_D;  // dW2 output
    std::vector<int64_t> lda, ldb, ldc, ldd;

    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        int64_t F = ffn_dim;
        int64_t H = hidden_dim;

        // Problem: {F, H, M}
        problem_sizes.push_back(cutlass::gemm::GemmCoord(F, H, M));

        // A: H [M,F] viewed as [F,M] ColumnMajor
        // For ColumnMajor [F,M], lda = F (stride between columns)
        ptr_A.push_back(reinterpret_cast<ElementInputFP16*>(hidden_ptrs[i]));
        lda.push_back(F);

        // B: Gỹ [M,H] RowMajor, ldb = H
        ptr_B.push_back(reinterpret_cast<ElementInputFP16*>(gy_tilde_ptrs[i]));
        ldb.push_back(H);

        // C: Nullptr (no bias/accumulation in this GEMM)
        ptr_C.push_back(nullptr);
        ldc.push_back(0);

        // D: dW2 [F,H] ColumnMajor, ldd = F
        ptr_D.push_back(reinterpret_cast<ElementOutputFP32*>(grad_w2_ptrs[i]));
        ldd.push_back(F);

        if (log_gemm) {
            std::cerr << "  Group " << i << ": M=" << M << " F=" << F << " H=" << H << std::endl;
        }
    }

    if (problem_sizes.empty()) return;

    // Allocate device memory for problem metadata
    size_t num_problems = problem_sizes.size();
    cutlass::gemm::GemmCoord* problem_sizes_device = nullptr;
    ElementInputFP16** ptr_A_device = nullptr;
    ElementInputFP16** ptr_B_device = nullptr;
    ElementOutputFP32** ptr_C_device = nullptr;
    ElementOutputFP32** ptr_D_device = nullptr;
    int64_t* lda_device = nullptr;
    int64_t* ldb_device = nullptr;
    int64_t* ldc_device = nullptr;
    int64_t* ldd_device = nullptr;

    CUDA_CHECK(cudaMalloc(&problem_sizes_device, num_problems * sizeof(cutlass::gemm::GemmCoord)));
    CUDA_CHECK(cudaMalloc(&ptr_A_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_B_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_C_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_D_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&lda_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldb_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldc_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldd_device, num_problems * sizeof(int64_t)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(problem_sizes_device, problem_sizes.data(),
                          num_problems * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_A_device, ptr_A.data(),
                          num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_B_device, ptr_B.data(),
                          num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_C_device, ptr_C.data(),
                          num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_D_device, ptr_D.data(),
                          num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lda_device, lda.data(),
                          num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldb_device, ldb.data(),
                          num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldc_device, ldc.data(),
                          num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldd_device, ldd.data(),
                          num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Initialize GEMM
    GroupedGemm_dW2 gemm;
    int threadblock_count = GroupedGemm_dW2::sufficient(problem_sizes.data(), num_problems);

    // Epilogue params: alpha=1, beta=0 (no accumulation, just compute AB)
    typename GroupedGemm_dW2::EpilogueOutputOp::Params epilogue_params(1.0f, 0.0f);

    typename GroupedGemm_dW2::Arguments args(
        problem_sizes_device,
        num_problems,
        threadblock_count,
        epilogue_params,
        ptr_A_device,
        ptr_B_device,
        ptr_C_device,
        ptr_D_device,
        lda_device,
        ldb_device,
        ldc_device,
        ldd_device,
        problem_sizes.data()
    );

    size_t workspace_size = gemm.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    CUTLASS_CHECK(gemm.initialize(args, workspace));
    CUTLASS_CHECK(gemm.run());
    if (LB_MOE_SYNC()) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudaFree(problem_sizes_device);
    cudaFree(ptr_A_device);
    cudaFree(ptr_B_device);
    cudaFree(ptr_C_device);
    cudaFree(ptr_D_device);
    cudaFree(lda_device);
    cudaFree(ldb_device);
    cudaFree(ldc_device);
    cudaFree(ldd_device);

    if (log_gemm) {
        std::cerr << "[MoE Backward] dW2 GEMM completed" << std::endl;
    }
}

// ============================================================================
// Backward GEMM #2: dH = Gỹ @ W2
// ============================================================================

// Standard GEMM, reuses forward layouts
// A: Gỹ [M,H] RowMajor
// B: W2 [H,F] ColumnMajor
// D: dH [M,F] RowMajor

using LayoutA_dH = cutlass::layout::RowMajor;
using LayoutB_dH = cutlass::layout::ColumnMajor;
using LayoutC_dH = cutlass::layout::RowMajor;

using EpilogueOp_dH = cutlass::epilogue::thread::LinearCombination<
    ElementOutputFP16,
    128 / cutlass::sizeof_bits<ElementOutputFP16>::value,
    ElementAccumulator,
    ElementCompute
>;

using GemmKernel_dH = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInputFP16, LayoutA_dH, cutlass::ComplexTransform::kNone, 8,
    ElementInputFP16, LayoutB_dH, cutlass::ComplexTransform::kNone, 8,
    ElementOutputFP16, LayoutC_dH,
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp_dH,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemm_dH = cutlass::gemm::device::GemmGrouped<GemmKernel_dH>;

void cutlass_grouped_gemm_dH(
    const uintptr_t* gy_tilde_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* grad_hidden_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
) {
    if (group_count == 0) return;

    const bool log_gemm = LB_MOE_LOG_BACKWARD_GEMM();

    if (log_gemm) {
        std::cerr << "[MoE Backward] dH GEMM: group_count=" << group_count
                  << " H=" << hidden_dim << " F=" << ffn_dim << std::endl;
    }

    // Problem size: {M, F, H}
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<ElementInputFP16*> ptr_A;  // Gỹ
    std::vector<ElementInputFP16*> ptr_B;  // W2
    std::vector<ElementOutputFP16*> ptr_C;
    std::vector<ElementOutputFP16*> ptr_D;  // dH
    std::vector<int64_t> lda, ldb, ldc, ldd;

    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        int64_t H = hidden_dim;
        int64_t F = ffn_dim;

        problem_sizes.push_back(cutlass::gemm::GemmCoord(M, F, H));

        // A: Gỹ [M,H] RowMajor, lda=H
        ptr_A.push_back(reinterpret_cast<ElementInputFP16*>(gy_tilde_ptrs[i]));
        lda.push_back(H);

        // B: W2 [H,F] ColumnMajor, ldb=F
        ptr_B.push_back(reinterpret_cast<ElementInputFP16*>(w2_ptrs[i]));
        ldb.push_back(F);

        // C: Nullptr
        ptr_C.push_back(nullptr);
        ldc.push_back(0);

        // D: dH [M,F] RowMajor, ldd=F
        ptr_D.push_back(reinterpret_cast<ElementOutputFP16*>(grad_hidden_ptrs[i]));
        ldd.push_back(F);
    }

    if (problem_sizes.empty()) return;

    size_t num_problems = problem_sizes.size();
    cutlass::gemm::GemmCoord* problem_sizes_device = nullptr;
    ElementInputFP16** ptr_A_device = nullptr;
    ElementInputFP16** ptr_B_device = nullptr;
    ElementOutputFP16** ptr_C_device = nullptr;
    ElementOutputFP16** ptr_D_device = nullptr;
    int64_t* lda_device = nullptr;
    int64_t* ldb_device = nullptr;
    int64_t* ldc_device = nullptr;
    int64_t* ldd_device = nullptr;

    CUDA_CHECK(cudaMalloc(&problem_sizes_device, num_problems * sizeof(cutlass::gemm::GemmCoord)));
    CUDA_CHECK(cudaMalloc(&ptr_A_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_B_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_C_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_D_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&lda_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldb_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldc_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldd_device, num_problems * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(problem_sizes_device, problem_sizes.data(),
                          num_problems * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_A_device, ptr_A.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_B_device, ptr_B.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_C_device, ptr_C.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_D_device, ptr_D.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lda_device, lda.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldb_device, ldb.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldc_device, ldc.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldd_device, ldd.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));

    GroupedGemm_dH gemm;
    int threadblock_count = GroupedGemm_dH::sufficient(problem_sizes.data(), num_problems);
    typename GroupedGemm_dH::EpilogueOutputOp::Params epilogue_params(1.0f, 0.0f);

    typename GroupedGemm_dH::Arguments args(
        problem_sizes_device, num_problems, threadblock_count, epilogue_params,
        ptr_A_device, ptr_B_device, ptr_C_device, ptr_D_device,
        lda_device, ldb_device, ldc_device, ldd_device,
        problem_sizes.data()
    );

    size_t workspace_size = gemm.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    CUTLASS_CHECK(gemm.initialize(args, workspace));
    CUTLASS_CHECK(gemm.run());
    if (LB_MOE_SYNC()) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (workspace) cudaFree(workspace);
    cudaFree(problem_sizes_device);
    cudaFree(ptr_A_device);
    cudaFree(ptr_B_device);
    cudaFree(ptr_C_device);
    cudaFree(ptr_D_device);
    cudaFree(lda_device);
    cudaFree(ldb_device);
    cudaFree(ldc_device);
    cudaFree(ldd_device);

    if (log_gemm) {
        std::cerr << "[MoE Backward] dH GEMM completed" << std::endl;
    }
}

// ============================================================================
// Backward GEMM #3: Z_recompute = X @ W1ᵀ + b1
// ============================================================================

// Identical to forward W1, but WITHOUT GELU epilogue (we need raw Z for GELU' kernel)
// A: X [M,H] RowMajor
// B: W1 [H,F] ColumnMajor
// C: b1 [F] (broadcast)
// D: Z [M,F] RowMajor

using LayoutA_recompute = cutlass::layout::RowMajor;
using LayoutB_recompute = cutlass::layout::ColumnMajor;
using LayoutC_recompute = cutlass::layout::RowMajor;

using EpilogueOp_recompute = cutlass::epilogue::thread::LinearCombination<
    ElementOutputFP16,
    128 / cutlass::sizeof_bits<ElementOutputFP16>::value,
    ElementAccumulator,
    ElementCompute
>;

using GemmKernel_recompute = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInputFP16, LayoutA_recompute, cutlass::ComplexTransform::kNone, 8,
    ElementInputFP16, LayoutB_recompute, cutlass::ComplexTransform::kNone, 8,
    ElementOutputFP16, LayoutC_recompute,
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp_recompute,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemm_recompute = cutlass::gemm::device::GemmGrouped<GemmKernel_recompute>;

void cutlass_grouped_gemm_recompute_Z(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* z_recomputed_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
) {
    if (group_count == 0) return;

    const bool log_gemm = LB_MOE_LOG_BACKWARD_GEMM();

    if (log_gemm) {
        std::cerr << "[MoE Backward] Recompute Z: group_count=" << group_count
                  << " H=" << hidden_dim << " F=" << ffn_dim << std::endl;
    }

    // Problem size: {M, F, H} (same as forward W1)
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<ElementInputFP16*> ptr_A;  // X
    std::vector<ElementInputFP16*> ptr_B;  // W1
    std::vector<ElementOutputFP16*> ptr_C;  // b1 (bias)
    std::vector<ElementOutputFP16*> ptr_D;  // Z
    std::vector<int64_t> lda, ldb, ldc, ldd;

    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        int64_t H = hidden_dim;
        int64_t F = ffn_dim;

        problem_sizes.push_back(cutlass::gemm::GemmCoord(M, F, H));

        // A: X [M,H] RowMajor, lda=H
        ptr_A.push_back(reinterpret_cast<ElementInputFP16*>(input_ptrs[i]));
        lda.push_back(H);

        // B: W1 [H,F] ColumnMajor, ldb=H
        ptr_B.push_back(reinterpret_cast<ElementInputFP16*>(w1_ptrs[i]));
        ldb.push_back(H);

        // C: b1 [F] (broadcast via ldc=0)
        ptr_C.push_back(reinterpret_cast<ElementOutputFP16*>(b1_ptrs[i]));
        ldc.push_back(0);  // Stride 0 = broadcast across rows

        // D: Z [M,F] RowMajor, ldd=F
        ptr_D.push_back(reinterpret_cast<ElementOutputFP16*>(z_recomputed_ptrs[i]));
        ldd.push_back(F);
    }

    if (problem_sizes.empty()) return;

    size_t num_problems = problem_sizes.size();
    cutlass::gemm::GemmCoord* problem_sizes_device = nullptr;
    ElementInputFP16** ptr_A_device = nullptr;
    ElementInputFP16** ptr_B_device = nullptr;
    ElementOutputFP16** ptr_C_device = nullptr;
    ElementOutputFP16** ptr_D_device = nullptr;
    int64_t* lda_device = nullptr;
    int64_t* ldb_device = nullptr;
    int64_t* ldc_device = nullptr;
    int64_t* ldd_device = nullptr;

    CUDA_CHECK(cudaMalloc(&problem_sizes_device, num_problems * sizeof(cutlass::gemm::GemmCoord)));
    CUDA_CHECK(cudaMalloc(&ptr_A_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_B_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_C_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_D_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&lda_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldb_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldc_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldd_device, num_problems * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(problem_sizes_device, problem_sizes.data(),
                          num_problems * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_A_device, ptr_A.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_B_device, ptr_B.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_C_device, ptr_C.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_D_device, ptr_D.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lda_device, lda.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldb_device, ldb.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldc_device, ldc.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldd_device, ldd.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));

    GroupedGemm_recompute gemm;
    int threadblock_count = GroupedGemm_recompute::sufficient(problem_sizes.data(), num_problems);

    // Epilogue: D = alpha*AB + beta*C = 1*AB + 1*C (bias add)
    typename GroupedGemm_recompute::EpilogueOutputOp::Params epilogue_params(1.0f, 1.0f);

    typename GroupedGemm_recompute::Arguments args(
        problem_sizes_device, num_problems, threadblock_count, epilogue_params,
        ptr_A_device, ptr_B_device, ptr_C_device, ptr_D_device,
        lda_device, ldb_device, ldc_device, ldd_device,
        problem_sizes.data()
    );

    size_t workspace_size = gemm.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    CUTLASS_CHECK(gemm.initialize(args, workspace));
    CUTLASS_CHECK(gemm.run());
    if (LB_MOE_SYNC()) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (workspace) cudaFree(workspace);
    cudaFree(problem_sizes_device);
    cudaFree(ptr_A_device);
    cudaFree(ptr_B_device);
    cudaFree(ptr_C_device);
    cudaFree(ptr_D_device);
    cudaFree(lda_device);
    cudaFree(ldb_device);
    cudaFree(ldc_device);
    cudaFree(ldd_device);

    if (log_gemm) {
        std::cerr << "[MoE Backward] Recompute Z completed" << std::endl;
    }
}

// ============================================================================
// Backward GEMM #4: dW1 = dZᵀ @ X
// ============================================================================

// A: X [M,H] viewed as [H,M] ColumnMajor (transposed)
// B: dZ [M,F] RowMajor
// D: dW1 [F,H] ColumnMajor (to match forward W1 storage)

using LayoutA_dW1 = cutlass::layout::ColumnMajor;  // X transposed
using LayoutB_dW1 = cutlass::layout::RowMajor;     // dZ
using LayoutC_dW1 = cutlass::layout::ColumnMajor;  // dW1 output

using EpilogueOp_dW1 = cutlass::epilogue::thread::LinearCombination<
    ElementOutputFP32,
    128 / cutlass::sizeof_bits<ElementOutputFP32>::value,
    ElementAccumulator,
    ElementCompute
>;

using GemmKernel_dW1 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInputFP16, LayoutA_dW1, cutlass::ComplexTransform::kNone, 8,
    ElementInputFP16, LayoutB_dW1, cutlass::ComplexTransform::kNone, 8,
    ElementOutputFP32, LayoutC_dW1,
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp_dW1,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemm_dW1 = cutlass::gemm::device::GemmGrouped<GemmKernel_dW1>;

void cutlass_grouped_gemm_dW1(
    const uintptr_t* grad_z_ptrs,
    const uintptr_t* input_ptrs,
    const uintptr_t* grad_w1_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
) {
    if (group_count == 0) return;

    const bool log_gemm = LB_MOE_LOG_BACKWARD_GEMM();

    if (log_gemm) {
        std::cerr << "[MoE Backward] dW1 GEMM: group_count=" << group_count
                  << " H=" << hidden_dim << " F=" << ffn_dim << std::endl;
    }

    // Problem size: {F, H, M}
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<ElementInputFP16*> ptr_A;  // X (transposed)
    std::vector<ElementInputFP16*> ptr_B;  // dZ
    std::vector<ElementOutputFP32*> ptr_C;
    std::vector<ElementOutputFP32*> ptr_D;  // dW1
    std::vector<int64_t> lda, ldb, ldc, ldd;

    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        int64_t F = ffn_dim;
        int64_t H = hidden_dim;

        problem_sizes.push_back(cutlass::gemm::GemmCoord(F, H, M));

        // A: X [M,H] viewed as [H,M] ColumnMajor, lda=H
        ptr_A.push_back(reinterpret_cast<ElementInputFP16*>(input_ptrs[i]));
        lda.push_back(H);

        // B: dZ [M,F] RowMajor, ldb=F
        ptr_B.push_back(reinterpret_cast<ElementInputFP16*>(grad_z_ptrs[i]));
        ldb.push_back(F);

        // C: Nullptr
        ptr_C.push_back(nullptr);
        ldc.push_back(0);

        // D: dW1 [F,H] ColumnMajor, ldd=F
        ptr_D.push_back(reinterpret_cast<ElementOutputFP32*>(grad_w1_ptrs[i]));
        ldd.push_back(F);
    }

    if (problem_sizes.empty()) return;

    size_t num_problems = problem_sizes.size();
    cutlass::gemm::GemmCoord* problem_sizes_device = nullptr;
    ElementInputFP16** ptr_A_device = nullptr;
    ElementInputFP16** ptr_B_device = nullptr;
    ElementOutputFP32** ptr_C_device = nullptr;
    ElementOutputFP32** ptr_D_device = nullptr;
    int64_t* lda_device = nullptr;
    int64_t* ldb_device = nullptr;
    int64_t* ldc_device = nullptr;
    int64_t* ldd_device = nullptr;

    CUDA_CHECK(cudaMalloc(&problem_sizes_device, num_problems * sizeof(cutlass::gemm::GemmCoord)));
    CUDA_CHECK(cudaMalloc(&ptr_A_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_B_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_C_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_D_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&lda_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldb_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldc_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldd_device, num_problems * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(problem_sizes_device, problem_sizes.data(),
                          num_problems * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_A_device, ptr_A.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_B_device, ptr_B.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_C_device, ptr_C.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_D_device, ptr_D.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lda_device, lda.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldb_device, ldb.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldc_device, ldc.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldd_device, ldd.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));

    GroupedGemm_dW1 gemm;
    int threadblock_count = GroupedGemm_dW1::sufficient(problem_sizes.data(), num_problems);
    typename GroupedGemm_dW1::EpilogueOutputOp::Params epilogue_params(1.0f, 0.0f);

    typename GroupedGemm_dW1::Arguments args(
        problem_sizes_device, num_problems, threadblock_count, epilogue_params,
        ptr_A_device, ptr_B_device, ptr_C_device, ptr_D_device,
        lda_device, ldb_device, ldc_device, ldd_device,
        problem_sizes.data()
    );

    size_t workspace_size = gemm.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    CUTLASS_CHECK(gemm.initialize(args, workspace));
    CUTLASS_CHECK(gemm.run());
    if (LB_MOE_SYNC()) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (workspace) cudaFree(workspace);
    cudaFree(problem_sizes_device);
    cudaFree(ptr_A_device);
    cudaFree(ptr_B_device);
    cudaFree(ptr_C_device);
    cudaFree(ptr_D_device);
    cudaFree(lda_device);
    cudaFree(ldb_device);
    cudaFree(ldc_device);
    cudaFree(ldd_device);

    if (log_gemm) {
        std::cerr << "[MoE Backward] dW1 GEMM completed" << std::endl;
    }
}

// ============================================================================
// Backward GEMM #5: dX = dZ @ W1
// ============================================================================

// Standard GEMM, reuses forward layouts
// A: dZ [M,F] RowMajor
// B: W1 [F,H] ColumnMajor
// D: dX [M,H] RowMajor

using LayoutA_dX = cutlass::layout::RowMajor;
using LayoutB_dX = cutlass::layout::ColumnMajor;
using LayoutC_dX = cutlass::layout::RowMajor;

using EpilogueOp_dX = cutlass::epilogue::thread::LinearCombination<
    ElementOutputFP16,
    128 / cutlass::sizeof_bits<ElementOutputFP16>::value,
    ElementAccumulator,
    ElementCompute
>;

using GemmKernel_dX = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInputFP16, LayoutA_dX, cutlass::ComplexTransform::kNone, 8,
    ElementInputFP16, LayoutB_dX, cutlass::ComplexTransform::kNone, 8,
    ElementOutputFP16, LayoutC_dX,
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp_dX,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemm_dX = cutlass::gemm::device::GemmGrouped<GemmKernel_dX>;

void cutlass_grouped_gemm_dX(
    const uintptr_t* grad_z_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* grad_x_grouped_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
) {
    if (group_count == 0) return;

    const bool log_gemm = LB_MOE_LOG_BACKWARD_GEMM();

    if (log_gemm) {
        std::cerr << "[MoE Backward] dX GEMM: group_count=" << group_count
                  << " H=" << hidden_dim << " F=" << ffn_dim << std::endl;
    }

    // Problem size: {M, H, F}
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<ElementInputFP16*> ptr_A;  // dZ
    std::vector<ElementInputFP16*> ptr_B;  // W1
    std::vector<ElementOutputFP16*> ptr_C;
    std::vector<ElementOutputFP16*> ptr_D;  // dX
    std::vector<int64_t> lda, ldb, ldc, ldd;

    for (int64_t i = 0; i < group_count; ++i) {
        int64_t M = m_sizes[i];
        if (M == 0) continue;

        int64_t H = hidden_dim;
        int64_t F = ffn_dim;

        problem_sizes.push_back(cutlass::gemm::GemmCoord(M, H, F));

        // A: dZ [M,F] RowMajor, lda=F
        ptr_A.push_back(reinterpret_cast<ElementInputFP16*>(grad_z_ptrs[i]));
        lda.push_back(F);

        // B: W1 [F,H] ColumnMajor, ldb=H
        ptr_B.push_back(reinterpret_cast<ElementInputFP16*>(w1_ptrs[i]));
        ldb.push_back(H);

        // C: Nullptr
        ptr_C.push_back(nullptr);
        ldc.push_back(0);

        // D: dX [M,H] RowMajor, ldd=H
        ptr_D.push_back(reinterpret_cast<ElementOutputFP16*>(grad_x_grouped_ptrs[i]));
        ldd.push_back(H);
    }

    if (problem_sizes.empty()) return;

    size_t num_problems = problem_sizes.size();
    cutlass::gemm::GemmCoord* problem_sizes_device = nullptr;
    ElementInputFP16** ptr_A_device = nullptr;
    ElementInputFP16** ptr_B_device = nullptr;
    ElementOutputFP16** ptr_C_device = nullptr;
    ElementOutputFP16** ptr_D_device = nullptr;
    int64_t* lda_device = nullptr;
    int64_t* ldb_device = nullptr;
    int64_t* ldc_device = nullptr;
    int64_t* ldd_device = nullptr;

    CUDA_CHECK(cudaMalloc(&problem_sizes_device, num_problems * sizeof(cutlass::gemm::GemmCoord)));
    CUDA_CHECK(cudaMalloc(&ptr_A_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_B_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_C_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&ptr_D_device, num_problems * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&lda_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldb_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldc_device, num_problems * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&ldd_device, num_problems * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(problem_sizes_device, problem_sizes.data(),
                          num_problems * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_A_device, ptr_A.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_B_device, ptr_B.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_C_device, ptr_C.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr_D_device, ptr_D.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lda_device, lda.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldb_device, ldb.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldc_device, ldc.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ldd_device, ldd.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));

    GroupedGemm_dX gemm;
    int threadblock_count = GroupedGemm_dX::sufficient(problem_sizes.data(), num_problems);
    typename GroupedGemm_dX::EpilogueOutputOp::Params epilogue_params(1.0f, 0.0f);

    typename GroupedGemm_dX::Arguments args(
        problem_sizes_device, num_problems, threadblock_count, epilogue_params,
        ptr_A_device, ptr_B_device, ptr_C_device, ptr_D_device,
        lda_device, ldb_device, ldc_device, ldd_device,
        problem_sizes.data()
    );

    size_t workspace_size = gemm.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    CUTLASS_CHECK(gemm.initialize(args, workspace));
    CUTLASS_CHECK(gemm.run());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (workspace) cudaFree(workspace);
    cudaFree(problem_sizes_device);
    cudaFree(ptr_A_device);
    cudaFree(ptr_B_device);
    cudaFree(ptr_C_device);
    cudaFree(ptr_D_device);
    cudaFree(lda_device);
    cudaFree(ldb_device);
    cudaFree(ldc_device);
    cudaFree(ldd_device);

    if (log_gemm) {
        std::cerr << "[MoE Backward] dX GEMM completed" << std::endl;
    }
}

} // namespace moe
} // namespace lb
