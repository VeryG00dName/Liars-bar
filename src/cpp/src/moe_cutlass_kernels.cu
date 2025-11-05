#include "moe_cutlass_kernels.h"

#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

// Device defaults
#include <cutlass/gemm/device/default_gemm_configuration.h>

// ✅ Swizzles (defines GemmBatchedIdentityThreadblockSwizzle)
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

// Grouped GEMM + kernels (must come AFTER the alias block)
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm.h>

// Epilogues
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/epilogue/thread/activation.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

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
    static const int value = lb_env_int("LB_MOE_LOG_GEMM", 0);
    return value != 0;
}

static bool LB_MOE_LOG_CUTLASS() {
    static const bool flag = lb_env_enabled("LB_MOE_LOG_CUTLASS");
    return flag;
}

// Helper macro for CUTLASS error checking
#define CUTLASS_CHECK(status)                                                           \
    {\
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
    {\
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

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// MMA configuration for Ampere (sm_86 for RTX 30xx)
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80; // Updated for RTX 3080

// Let CUTLASS auto-select optimal shapes for Sm86
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Swizzle for better load balancing with irregular group sizes
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

// Stages
constexpr int Stages = 2; // Optimal for 2-stage double-buffered pipeline

// ============================================================================ 
// W1 GEMM: Epilogue with Bias + GELU
// ============================================================================ 
using EpilogueOpW1 = cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::Default
>;

using GemmKernelW1 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementOutput, LayoutC,
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOpW1,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemmW1 = cutlass::gemm::device::GemmGrouped<GemmKernelW1>;

// ============================================================================ 
// W2 GEMM: Standard Epilogue
// ============================================================================ 
using EpilogueOpW2 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
>;

using GemmKernelW2 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementOutput, LayoutC,
    ElementAccumulator,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOpW2,
    ThreadblockSwizzle,
    Stages
>::GemmKernel;

using GroupedGemmW2 = cutlass::gemm::device::GemmGrouped<GemmKernelW2>;

// ============================================================================ 
// Helper Kernels
// ============================================================================ 

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
    if (M == 0) return;

    int64_t off = token_offsets[g];
    int64_t pi = policy_ids[g];
    int64_t ei = expert_ids[g];
    const T* __restrict__ bias = reinterpret_cast<const T*>(b2_tbl[pi * E + ei]);

    int64_t row_start = blockIdx.x * blockDim.x;
    for (int64_t row = row_start + threadIdx.x; row < M; row += gridDim.x * blockDim.x) {
        T* data_row = out_base + (off + row) * H;
        float rw = routing_base[off + row];
        #pragma unroll
        for (int64_t col = 0; col < H; ++col) {
            float val = float(data_row[col]) + float(bias[col]);
            data_row[col] = T(val * rw);
        }
    }
}

// ============================================================================ 
// Main Implementation
// ============================================================================ 
void cutlass_grouped_moe_forward(
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

    // Prepare device descriptor buffers
    cutlass::gemm::GemmCoord* problem_sizes_device_w1 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w1);
    ElementA** ptr_A_device_w1 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w1);
    ElementB** ptr_B_device_w1 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w1);
    ElementOutput** ptr_C_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w1);
    ElementOutput** ptr_D_device_w1 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w1);
    int64_t* lda_device_w1 = reinterpret_cast<int64_t*>(workspace->lda_device_w1);
    int64_t* ldb_device_w1 = reinterpret_cast<int64_t*>(workspace->ldb_device_w1);
    int64_t* ldc_device_w1 = reinterpret_cast<int64_t*>(workspace->ldc_device_w1);
    int64_t* ldd_device_w1 = reinterpret_cast<int64_t*>(workspace->ldd_device_w1);

    cutlass::gemm::GemmCoord* problem_sizes_device_w2 = reinterpret_cast<cutlass::gemm::GemmCoord*>(workspace->problem_sizes_device_w2);
    ElementA** ptr_A_device_w2 = reinterpret_cast<ElementA**>(workspace->ptr_A_device_w2);
    ElementB** ptr_B_device_w2 = reinterpret_cast<ElementB**>(workspace->ptr_B_device_w2);
    ElementOutput** ptr_C_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_C_device_w2);
    ElementOutput** ptr_D_device_w2 = reinterpret_cast<ElementOutput**>(workspace->ptr_D_device_w2);
    int64_t* lda_device_w2 = reinterpret_cast<int64_t*>(workspace->lda_device_w2);
    int64_t* ldb_device_w2 = reinterpret_cast<int64_t*>(workspace->ldb_device_w2);
    int64_t* ldc_device_w2 = reinterpret_cast<int64_t*>(workspace->ldc_device_w2);
    int64_t* ldd_device_w2 = reinterpret_cast<int64_t*>(workspace->ldd_device_w2);

    int64_t* hidden_offsets_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&hidden_offsets_dev, sizeof(int64_t) * group_count));
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
        ptr_A_device_w1, ptr_B_device_w1, ptr_C_device_w1, ptr_D_device_w1,
        lda_device_w1, ldb_device_w1, ldc_device_w1, ldd_device_w1);

    // Run W1 GEMM
    GroupedGemmW1 gemm_w1;
    typename GroupedGemmW1::EpilogueOutputOp::Params epilogue_params_w1(1.0f, 1.0f);
    typename GroupedGemmW1::Arguments args_w1(
        problem_sizes_device_w1, group_count, static_cast<int>(group_count), epilogue_params_w1,
        ptr_A_device_w1, ptr_B_device_w1, ptr_C_device_w1, ptr_D_device_w1,
        lda_device_w1, ldb_device_w1, ldc_device_w1, ldd_device_w1, nullptr);

    CUTLASS_CHECK(gemm_w1.initialize(args_w1, workspace->workspace_w1));
    CUTLASS_CHECK(gemm_w1.run());

    // Build W2 descriptors on device
    build_w2_descriptors_kernel<<<blocks, threads>>>(
        m_sizes_dev, policy_ids_dev, expert_ids_dev, token_offsets_dev,
        group_count, hidden_dim, ffn_dim,
        hidden_base, output_base, hidden_offsets_dev,
        w2_ptrs_table, num_policies, num_experts,
        problem_sizes_device_w2,
        ptr_A_device_w2, ptr_B_device_w2, ptr_C_device_w2, ptr_D_device_w2,
        lda_device_w2, ldb_device_w2, ldc_device_w2, ldd_device_w2);

    // Run W2 GEMM
    GroupedGemmW2 gemm_w2;
    typename GroupedGemmW2::EpilogueOutputOp::Params epilogue_params_w2(1.0f, 0.0f);
    typename GroupedGemmW2::Arguments args_w2(
        problem_sizes_device_w2, group_count, static_cast<int>(group_count), epilogue_params_w2,
        ptr_A_device_w2, ptr_B_device_w2, ptr_C_device_w2, ptr_D_device_w2,
        lda_device_w2, ldb_device_w2, ldc_device_w2, ldd_device_w2, nullptr);

    CUTLASS_CHECK(gemm_w2.initialize(args_w2, workspace->workspace_w2));
    CUTLASS_CHECK(gemm_w2.run());

    // Apply fused bias + routing scaling
    {
        const int threads_per_block = 256;
        const int max_blocks_per_group = 128; // Heuristic to prevent too many small blocks
        dim3 block(threads_per_block, 1, 1);
        dim3 grid(max_blocks_per_group, static_cast<unsigned>(group_count), 1);
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

    CUDA_CHECK(cudaFree(hidden_offsets_dev));
}

} // namespace moe
} // namespace lb