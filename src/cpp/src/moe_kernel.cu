#include "moe_kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cmath>
#include <vector>

// No anonymous namespace needed, these are local to the file

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

template <int FFN_DIM_MAX>
__global__ void simple_moe_forward_kernel_half(
    const at::Half*,
    const at::Half*,
    const int64_t*,
    const int64_t*,
    at::Half*,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    const at::Half*,
    const at::Half*,
    const at::Half*,
    const at::Half*) {
    TORCH_CHECK(false, "simple_moe_forward_kernel_half is not implemented");
}

torch::Tensor moe_forward_cuda(
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor) {
    TORCH_CHECK(false, "moe_forward_cuda is not implemented");
}
