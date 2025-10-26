#pragma once

#include <cstdint>
#include <vector>

namespace lb {
namespace moe {

/**
 * Individual CUTLASS grouped GEMM functions for MoE backward pass.
 *
 * Each function handles one of the 5 GEMMs in the backward computation graph.
 * All GEMMs use:
 * - FP16 input/output with FP32 accumulation
 * - Grouped execution (one problem per expert group)
 * - Layouts specified in moe_backward_gemm_layouts.md
 */

/**
 * Backward GEMM #1: dW2 = Gỹᵀ @ H
 *
 * Computes weight gradient for W2.
 * Problem size: {F, H, M} for each group
 * A: H [M,F] viewed as ColumnMajor (transposed), lda=F
 * B: Gỹ [M,H] RowMajor, ldb=H
 * D: dW2 [F,H] ColumnMajor (matches W2 storage), ldd=F
 *
 * Output layout MUST match forward W2 storage for gradient accumulation.
 *
 * @param gy_tilde_ptrs Array of Gỹ pointers [M_i, H] FP16
 * @param hidden_ptrs Array of H pointers [M_i, F] FP16
 * @param grad_w2_ptrs Array of dW2 output pointers [F, H] FP32
 * @param m_sizes Array of M dimensions (tokens per group)
 * @param group_count Number of expert groups
 * @param hidden_dim H dimension
 * @param ffn_dim F dimension
 */
void cutlass_grouped_gemm_dW2(
    const uintptr_t* gy_tilde_ptrs,
    const uintptr_t* hidden_ptrs,
    const uintptr_t* grad_w2_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
);

/**
 * Backward GEMM #2: dH = Gỹ @ W2
 *
 * Computes activation gradient (hidden layer).
 * Problem size: {M, F, H} for each group
 * A: Gỹ [M,H] RowMajor, lda=H
 * B: W2 [H,F] ColumnMajor, ldb=F
 * D: dH [M,F] RowMajor, ldd=F
 *
 * Reuses W2 weights directly from forward (no transpose needed).
 *
 * @param gy_tilde_ptrs Array of Gỹ pointers [M_i, H] FP16
 * @param w2_ptrs Array of W2 weight pointers [H, F] FP16 ColumnMajor
 * @param grad_hidden_ptrs Array of dH output pointers [M_i, F] FP16
 * @param m_sizes Array of M dimensions
 * @param group_count Number of expert groups
 * @param hidden_dim H dimension
 * @param ffn_dim F dimension
 */
void cutlass_grouped_gemm_dH(
    const uintptr_t* gy_tilde_ptrs,
    const uintptr_t* w2_ptrs,
    const uintptr_t* grad_hidden_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
);

/**
 * Backward GEMM #3: Z_recompute = X @ W1ᵀ + b1
 *
 * Recomputes pre-GELU activation for save-minimal policy.
 * IDENTICAL configuration to forward W1 GEMM (including GELU epilogue).
 *
 * Problem size: {M, F, H} for each group
 * A: X [M,H] RowMajor, lda=H
 * B: W1 [H,F] ColumnMajor, ldb=H
 * C: b1 [F] (broadcast via ldc=0)
 * D: Z_recomputed [M,F] RowMajor, ldd=F
 *
 * Epilogue: LinearCombination with bias add (alpha=1, beta=1)
 * NOTE: Do NOT apply GELU here (we need raw Z for GELU' computation)
 *
 * @param input_ptrs Array of X pointers [M_i, H] FP16
 * @param w1_ptrs Array of W1 weight pointers [H, F] FP16 ColumnMajor
 * @param b1_ptrs Array of b1 bias pointers [F] FP16
 * @param z_recomputed_ptrs Array of Z output pointers [M_i, F] FP16
 * @param m_sizes Array of M dimensions
 * @param group_count Number of expert groups
 * @param hidden_dim H dimension
 * @param ffn_dim F dimension
 */
void cutlass_grouped_gemm_recompute_Z(
    const uintptr_t* input_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* b1_ptrs,
    const uintptr_t* z_recomputed_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
);

/**
 * Backward GEMM #4: dW1 = dZᵀ @ X
 *
 * Computes weight gradient for W1.
 * Problem size: {F, H, M} for each group
 * A: X [M,H] viewed as ColumnMajor (transposed), lda=H
 * B: dZ [M,F] RowMajor, ldb=F
 * D: dW1 [F,H] ColumnMajor (matches W1 storage), ldd=F
 *
 * Output layout MUST match forward W1 storage for gradient accumulation.
 *
 * @param grad_z_ptrs Array of dZ pointers [M_i, F] FP16
 * @param input_ptrs Array of X pointers [M_i, H] FP16
 * @param grad_w1_ptrs Array of dW1 output pointers [F, H] FP32
 * @param m_sizes Array of M dimensions
 * @param group_count Number of expert groups
 * @param hidden_dim H dimension
 * @param ffn_dim F dimension
 */
void cutlass_grouped_gemm_dW1(
    const uintptr_t* grad_z_ptrs,
    const uintptr_t* input_ptrs,
    const uintptr_t* grad_w1_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
);

/**
 * Backward GEMM #5: dX = dZ @ W1
 *
 * Computes input gradient (data gradient).
 * Problem size: {M, H, F} for each group
 * A: dZ [M,F] RowMajor, lda=F
 * B: W1 [F,H] ColumnMajor, ldb=H
 * D: dX [M,H] RowMajor, ldd=H
 *
 * Reuses W1 weights directly from forward (no transpose needed).
 * Output is in grouped/sorted order; must scatter-add back to token order.
 *
 * @param grad_z_ptrs Array of dZ pointers [M_i, F] FP16
 * @param w1_ptrs Array of W1 weight pointers [F, H] FP16 ColumnMajor
 * @param grad_x_grouped_ptrs Array of dX output pointers [M_i, H] FP16
 * @param m_sizes Array of M dimensions
 * @param group_count Number of expert groups
 * @param hidden_dim H dimension
 * @param ffn_dim F dimension
 */
void cutlass_grouped_gemm_dX(
    const uintptr_t* grad_z_ptrs,
    const uintptr_t* w1_ptrs,
    const uintptr_t* grad_x_grouped_ptrs,
    const int64_t* m_sizes,
    int64_t group_count,
    int64_t hidden_dim,
    int64_t ffn_dim
);

} // namespace moe
} // namespace lb
