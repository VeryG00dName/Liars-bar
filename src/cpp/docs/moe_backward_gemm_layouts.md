# MoE Backward GEMM Layout Specification

**Purpose**: Explicit specification of GEMM configurations for all backward passes to prevent layout bugs.

**Critical**: All GEMMs must match weight storage layout from forward to avoid transpose overhead.

## Notation

- **M**: Batch/token dimension
- **K**: Input/hidden dimension
- **N**: Output/FFN dimension
- **H**: `hidden_dim` (256)
- **F**: `ffn_dim` (512, typically 2*H)
- **RowMajor**: Contiguous in last dimension (C-style)
- **ColumnMajor**: Contiguous in first dimension (Fortran-style)

## Forward GEMMs (Reference)

### Forward W1: Z = X @ W1ᵀ + b1

```
Logical operation: Z[M,F] = X[M,H] @ W1ᵀ[H,F]

CUTLASS configuration:
  problemSize: {M, F, H}
  A: X
    - shape: [M, H]
    - layout: RowMajor
    - lda: H
  B: W1
    - shape: [H, F] stored as [F, H] transposed
    - layout: ColumnMajor (represents W1ᵀ)
    - ldb: H
  C/D: Z (output)
    - shape: [M, F]
    - layout: RowMajor
    - ldd: F
```

**Storage**: `W1` weights stored as `[num_policies, num_experts, F, H]` ColumnMajor

### Forward W2: Y = H @ W2ᵀ + b2

```
Logical operation: Y[M,H] = H[M,F] @ W2ᵀ[F,H]

CUTLASS configuration:
  problemSize: {M, H, F}
  A: H
    - shape: [M, F]
    - layout: RowMajor
    - lda: F
  B: W2
    - shape: [F, H] stored as [H, F] transposed
    - layout: ColumnMajor (represents W2ᵀ)
    - ldb: F
  C/D: Y (output)
    - shape: [M, H]
    - layout: RowMajor
    - ldd: H
```

**Storage**: `W2` weights stored as `[num_policies, num_experts, H, F]` ColumnMajor

## Backward GEMMs

### Backward GEMM #1: dW2 = Gỹᵀ @ H

**Goal**: Compute weight gradient for W2

```
Logical operation: dW2[F,H] = Gỹᵀ[F,M] @ H[M,F]
                   dW2[F,H] = (Hᵀ[F,M] @ Gỹ[M,H])ᵀ

Mathematical equivalent forms:
  Option A: dW2 = Gỹᵀ @ H          (explicit transpose)
  Option B: dW2 = (Hᵀ @ Gỹ)ᵀ      (implicit via layout)

CUTLASS configuration (Option B - avoids explicit transpose):
  problemSize: {F, H, M}
  A: H
    - shape: [M, F]
    - layout: RowMajor, viewed as [F, M] ColumnMajor (transposed)
    - opA: OpTranspose or use ColumnMajor layout
    - lda: F (stride between columns in original M×F row-major)
  B: Gỹ
    - shape: [M, H]
    - layout: RowMajor
    - ldb: H
  D: dW2 (output)
    - shape: [F, H] (must match W2 storage: ColumnMajor [H,F] transposed)
    - layout: ColumnMajor (to match forward W2 storage)
    - ldd: F

Notes:
  - Output dW2 must have same layout as forward W2 for in-place gradient accumulation
  - No explicit transpose needed if we configure operand layouts correctly
  - Accumulation: FP32, output cast to FP16
```

### Backward GEMM #2: dH = Gỹ @ W2

**Goal**: Compute activation gradient (hidden layer)

```
Logical operation: dH[M,F] = Gỹ[M,H] @ W2[H,F]

CUTLASS configuration:
  problemSize: {M, F, H}
  A: Gỹ
    - shape: [M, H]
    - layout: RowMajor
    - lda: H
  B: W2
    - shape: [H, F] (stored as [F, H] transposed, so ColumnMajor)
    - layout: ColumnMajor
    - ldb: F (stride in original column-major storage)
  D: dH (output)
    - shape: [M, F]
    - layout: RowMajor
    - ldd: F

Notes:
  - Reuses W2 weights directly from forward (no transpose needed)
  - W2 already in ColumnMajor [H,F] format from forward storage
```

### Backward GEMM #3 (Recompute): Z = X @ W1ᵀ + b1

**Goal**: Recompute pre-GELU activation (save-minimal policy)

```
Logical operation: Z[M,F] = X[M,H] @ W1ᵀ[H,F] + b1

CUTLASS configuration: IDENTICAL to forward W1 GEMM
  problemSize: {M, F, H}
  A: X (saved from forward, grouped)
    - shape: [M, H]
    - layout: RowMajor
    - lda: H
  B: W1
    - shape: [H, F] stored as [F, H] transposed
    - layout: ColumnMajor
    - ldb: H
  D: Z_recomputed (output)
    - shape: [M, F]
    - layout: RowMajor
    - ldd: F

Epilogue: LinearCombination with bias add
  - alpha = 1.0
  - beta = 1.0 (accumulate bias)
  - C = broadcast(b1) [shape: [M, F], each row is b1]
  - D = alpha * (A @ B) + beta * C = (X @ W1ᵀ) + b1

Notes:
  - Must produce IDENTICAL numerical results to forward Z
  - Use same dtype (FP16 I/O, FP32 accumulation)
  - Same bias add mechanism as forward
```

### Backward GEMM #4: dW1 = dZᵀ @ X

**Goal**: Compute weight gradient for W1

```
Logical operation: dW1[F,H] = dZᵀ[F,M] @ X[M,H]
                   dW1[F,H] = (Xᵀ[H,M] @ dZ[M,F])ᵀ

CUTLASS configuration (implicit transpose via layout):
  problemSize: {F, H, M}
  A: X
    - shape: [M, H]
    - layout: RowMajor, viewed as [H, M] ColumnMajor (transposed)
    - opA: OpTranspose or use ColumnMajor layout
    - lda: H (stride in original row-major)
  B: dZ
    - shape: [M, F]
    - layout: RowMajor
    - ldb: F
  D: dW1 (output)
    - shape: [F, H] (must match W1 storage: ColumnMajor [H,F] transposed)
    - layout: ColumnMajor
    - ldd: F

Notes:
  - Same pattern as dW2: transpose via layout, not explicit operation
  - Output must match forward W1 storage layout
```

### Backward GEMM #5: dX = dZ @ W1

**Goal**: Compute input gradient (data gradient)

```
Logical operation: dX[M,H] = dZ[M,F] @ W1[F,H]

CUTLASS configuration:
  problemSize: {M, H, F}
  A: dZ
    - shape: [M, F]
    - layout: RowMajor
    - lda: F
  B: W1
    - shape: [F, H] (stored as [H, F] transposed, so ColumnMajor)
    - layout: ColumnMajor
    - ldb: H (stride in original column-major storage)
  D: dX (output, grouped/sorted order)
    - shape: [M, H]
    - layout: RowMajor
    - ldd: H

Notes:
  - Reuses W1 weights directly from forward
  - Output is in grouped/sorted order; must scatter-add back to token order
  - No transpose needed (W1 already in correct layout)
```

## Summary Table

| GEMM | Operation | Problem Size | A Layout | B Layout | D Layout | Notes |
|------|-----------|--------------|----------|----------|----------|-------|
| Forward W1 | Z = X @ W1ᵀ + b1 | {M, F, H} | Row | Col | Row | GELU epilogue |
| Forward W2 | Y = H @ W2ᵀ + b2 | {M, H, F} | Row | Col | Row | Routing scale epilogue |
| **Backward #1** | dW2 = Gỹᵀ @ H | {F, H, M} | Col (X^T) | Row | Col | Match W2 storage |
| **Backward #2** | dH = Gỹ @ W2 | {M, F, H} | Row | Col | Row | Reuse W2 weights |
| **Backward #3** | Z = X @ W1ᵀ + b1 | {M, F, H} | Row | Col | Row | Identical to forward W1 |
| **Backward #4** | dW1 = dZᵀ @ X | {F, H, M} | Col (X^T) | Row | Col | Match W1 storage |
| **Backward #5** | dX = dZ @ W1 | {M, H, F} | Row | Col | Row | Reuse W1 weights |

## Implementation Checklist

Before coding each GEMM:

- [ ] Verify problem size {M, N, K} matches specification
- [ ] Verify A/B layouts match table above
- [ ] Verify leading dimensions (lda/ldb/ldd) are correct for layout
- [ ] For weight gradients (dW1, dW2): verify output layout matches forward weight storage
- [ ] For data gradients (dH, dX): verify output can be used by next operation
- [ ] Verify accumulation dtype (FP32) and output dtype (FP16) match forward
- [ ] Add logging guards (LB_MOE_LOG_GEMM) to print shapes/layouts for debugging

## Validation

Test each GEMM in isolation:

```python
# Example: Test dW2 GEMM
M, H, F = 32, 256, 512
Gy_tilde = torch.randn(M, H, dtype=torch.float16, device='cuda')
H_tensor = torch.randn(M, F, dtype=torch.float16, device='cuda')

# Reference (PyTorch)
dW2_ref = (Gy_tilde.T @ H_tensor).T  # [F, H]

# Custom CUTLASS
dW2_custom = grouped_gemm_dW2(Gy_tilde, H_tensor, ...)

# Compare
assert torch.allclose(dW2_ref, dW2_custom, rtol=1e-2, atol=1e-3)
```
