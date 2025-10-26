#pragma once

#include <cmath>
#include <cuda_runtime.h>

/**
 * Shared GELU activation constants and implementations.
 *
 * CRITICAL: These must match exactly between forward (CUTLASS epilogue) and backward kernels
 * to ensure numerical parity. Any divergence will cause training/rollout policy mismatch.
 *
 * Formula: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * This is the "exact" erf-based GELU, matching:
 * - PyTorch's torch.nn.functional.gelu() (default)
 * - CUTLASS LinearCombinationGELU epilogue
 */

namespace moe_activations {

// Mathematical constants (high precision)
constexpr float GELU_SQRT_2 = 1.41421356237309504880f;
constexpr float GELU_SQRT_2PI = 2.50662827463100050242f;  // sqrt(2*pi)

/**
 * GELU forward activation (erf-based).
 *
 * Used in CUTLASS epilogue via LinearCombinationGELU.
 * Formula: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * @param x Input value
 * @return Activated value
 */
__host__ __device__ __forceinline__ float gelu_forward(float x) {
    return 0.5f * x * (1.0f + erff(x / GELU_SQRT_2));
}

/**
 * GELU backward gradient (derivative of erf-based GELU).
 *
 * Formula: GELU'(x) = cdf + x * pdf
 * where:
 *   cdf = 0.5 * (1 + erf(x / sqrt(2)))        [cumulative distribution]
 *   pdf = (1 / sqrt(2π)) * exp(-x² / 2)       [probability density]
 *
 * This is the chain rule gradient: d/dx[GELU(x)]
 *
 * @param x Input value (same x used in forward pass)
 * @return Gradient multiplier
 */
__host__ __device__ __forceinline__ float gelu_backward(float x) {
    // CDF component: 0.5 * (1 + erf(x / sqrt(2)))
    float cdf = 0.5f * (1.0f + erff(x / GELU_SQRT_2));

    // PDF component: (1 / sqrt(2π)) * exp(-x² / 2)
    float pdf = (1.0f / GELU_SQRT_2PI) * expf(-0.5f * x * x);

    // Full derivative
    return cdf + x * pdf;
}

/**
 * Apply GELU backward gradient element-wise with upstream gradient.
 *
 * Given upstream gradient dy and saved input x, computes:
 *   dx = dy * GELU'(x)
 *
 * @param dy Upstream gradient (from next layer)
 * @param x Input value (from forward pass, recomputed or saved)
 * @return Local gradient dx
 */
__host__ __device__ __forceinline__ float gelu_backward_with_grad(float dy, float x) {
    return dy * gelu_backward(x);
}

// Double precision versions for numerical testing/validation
__host__ __device__ __forceinline__ double gelu_forward_double(double x) {
    constexpr double SQRT_2 = 1.41421356237309504880;
    return 0.5 * x * (1.0 + erf(x / SQRT_2));
}

__host__ __device__ __forceinline__ double gelu_backward_double(double x) {
    constexpr double SQRT_2 = 1.41421356237309504880;
    constexpr double SQRT_2PI = 2.50662827463100050242;

    double cdf = 0.5 * (1.0 + erf(x / SQRT_2));
    double pdf = (1.0 / SQRT_2PI) * exp(-0.5 * x * x);
    return cdf + x * pdf;
}

} // namespace moe_activations
