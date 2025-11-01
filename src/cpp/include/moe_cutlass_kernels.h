#pragma once

#include <cstdint>
#include <vector>

namespace lb {
namespace moe {

/**
 * Workspace buffers for CUTLASS grouped MoE operations.
 * Pre-allocate once and reuse to avoid malloc/free overhead.
 */
struct MoEWorkspace {
    void* hidden_buffer = nullptr;
    size_t hidden_buffer_size = 0;

    void* workspace_w1 = nullptr;
    size_t workspace_w1_size = 0;

    void* workspace_w2 = nullptr;
    size_t workspace_w2_size = 0;

    // Problem descriptor buffers for W1
    void* problem_sizes_device_w1 = nullptr;
    void* ptr_A_device_w1 = nullptr;
    void* ptr_B_device_w1 = nullptr;
    void* ptr_C_device_w1 = nullptr;
    void* ptr_D_device_w1 = nullptr;
    void* lda_device_w1 = nullptr;
    void* ldb_device_w1 = nullptr;
    void* ldc_device_w1 = nullptr;
    void* ldd_device_w1 = nullptr;
    size_t descriptor_capacity_w1 = 0;

    // Problem descriptor buffers for W2
    void* problem_sizes_device_w2 = nullptr;
    void* ptr_A_device_w2 = nullptr;
    void* ptr_B_device_w2 = nullptr;
    void* ptr_C_device_w2 = nullptr;
    void* ptr_D_device_w2 = nullptr;
    void* lda_device_w2 = nullptr;
    void* ldb_device_w2 = nullptr;
    void* ldc_device_w2 = nullptr;
    void* ldd_device_w2 = nullptr;
    size_t descriptor_capacity_w2 = 0;
};

/**
 * Performs grouped MoE FFN forward pass using CUTLASS Grouped GEMM.
 *
 * Single kernel launch handles all expert groups concurrently using persistent CTAs.
 * Applies per-row routing weight scaling in the GEMM epilogue for maximum efficiency.
 *
 * For each group (policy_id, expert_id):
 *   hidden = GELU(input @ W1^T + b1)
 *   output = (hidden @ W2^T + b2) * routing_weight[row]  <-- custom epilogue
 *
 * @param input_ptrs Array of input pointers (one per group) [M_i, hidden_dim] in FP16
 * @param w1_ptrs Array of W1 weight pointers [ffn_dim, hidden_dim] in FP16
 * @param b1_ptrs Array of b1 bias pointers [ffn_dim] in FP16
 * @param w2_ptrs Array of W2 weight pointers [hidden_dim, ffn_dim] in FP16
 * @param b2_ptrs Array of b2 bias pointers [hidden_dim] in FP16
 * @param output_ptrs Array of output pointers [M_i, hidden_dim] in FP16
 * @param routing_weight_ptrs Array of routing weight pointers [M_i] in FP32
 * @param m_sizes Array of M dimensions (number of tokens per group)
 * @param policy_ids Array of policy IDs (for logging/debugging)
 * @param expert_ids Array of expert IDs (for logging/debugging)
 * @param token_offsets Array of token offsets into sorted arrays (for logging/debugging)
 * @param group_count Number of expert groups
 * @param hidden_dim Hidden dimension
 * @param ffn_dim FFN intermediate dimension
 *
 * @throws std::runtime_error on CUTLASS errors or invalid inputs
 */
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
    MoEWorkspace* workspace = nullptr  // Optional pre-allocated workspace
);

/**
 * Variant of grouped MoE forward that writes the intermediate hidden buffer
 * (post-GELU, pre-W2) into a caller-provided tensor so Python can save it
 * for backward without recomputation.
 */
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
    MoEWorkspace* workspace = nullptr  // Optional pre-allocated workspace
);

} // namespace moe
} // namespace lb
