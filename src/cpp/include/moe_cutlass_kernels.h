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

    // Host-pinned descriptor buffers for faster async uploads (W1)
    void* host_problem_sizes_w1 = nullptr;
    void* host_ptr_A_w1 = nullptr;
    void* host_ptr_B_w1 = nullptr;
    void* host_ptr_C_w1 = nullptr;
    void* host_ptr_D_w1 = nullptr;
    void* host_lda_w1 = nullptr;
    void* host_ldb_w1 = nullptr;
    void* host_ldc_w1 = nullptr;
    void* host_ldd_w1 = nullptr;
    size_t host_descriptor_capacity_w1 = 0;

    // Host-pinned descriptor buffers for faster async uploads (W2)
    void* host_problem_sizes_w2 = nullptr;
    void* host_ptr_A_w2 = nullptr;
    void* host_ptr_B_w2 = nullptr;
    void* host_ptr_C_w2 = nullptr;
    void* host_ptr_D_w2 = nullptr;
    void* host_lda_w2 = nullptr;
    void* host_ldb_w2 = nullptr;
    void* host_ldc_w2 = nullptr;
    void* host_ldd_w2 = nullptr;
    size_t host_descriptor_capacity_w2 = 0;
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
 * Device-metadata variant: builds descriptor arrays entirely on device
 * using pointer tables and base pointers, avoiding host-side staging.
 *
 * @param input_base Base pointer of input_grouped [T,H] (Half)
 * @param hidden_base Base pointer of hidden_grouped [T,F] (Half)
 * @param output_base Base pointer of output_grouped [T,H] (Half)
 * @param routing_base Base pointer of routing_weights_grouped [T] (Float)
 * @param w1_ptrs_table [P*E] device pointer table for W1 weights
 * @param w2_ptrs_table [P*E] device pointer table for W2 weights
 * @param b1_ptrs_table [P*E] device pointer table for b1 biases
 * @param b2_ptrs_table [P*E] device pointer table for b2 biases
 * @param num_policies Number of policies (P)
 * @param num_experts Number of experts (E)
 * @param m_sizes_dev [G] on device - number of tokens per group
 * @param policy_ids_dev [G] on device - policy IDs (for logging/debugging)
 * @param expert_ids_dev [G] on device - expert IDs (for logging/debugging)
 * @param token_offsets_dev [G] on device - token offsets into sorted arrays
 * @param group_count Number of expert groups (G)
 * @param hidden_dim Hidden dimension (H)
 * @param ffn_dim FFN intermediate dimension (F)
 * @param workspace Pre-allocated workspace for descriptor/device buffer reuse
 *
 * @throws std::runtime_error on CUTLASS errors or invalid inputs
 */
void cutlass_grouped_moe_forward(
    uintptr_t input_base,                 // base pointer of input_grouped [T,H] (Half)
    uintptr_t hidden_base,                // base pointer of hidden_grouped [T,F] (Half)
    uintptr_t output_base,                // base pointer of output_grouped [T,H] (Half)
    uintptr_t routing_base,               // base pointer of routing_weights_grouped [T] (Float)
    const uint64_t* w1_ptrs_table,        // [P*E] device pointer table
    const uint64_t* w2_ptrs_table,        // [P*E]
    const uint64_t* b1_ptrs_table,        // [P*E]
    const uint64_t* b2_ptrs_table,        // [P*E]
    int64_t num_policies,                 // P
    int64_t num_experts,                  // E
    const int64_t* m_sizes_dev,           // [G] on device
    const int64_t* policy_ids_dev,        // [G] on device
    const int64_t* expert_ids_dev,        // [G] on device
    const int64_t* token_offsets_dev,     // [G] on device
    int64_t group_count,                  // G
    int64_t hidden_dim,                   // H
    int64_t ffn_dim,                      // F
    MoEWorkspace* workspace               // for descriptor/device buffer reuse
);

} // namespace moe
} // namespace lb
