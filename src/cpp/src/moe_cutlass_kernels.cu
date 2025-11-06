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
}

} // namespace moe
} // namespace lb