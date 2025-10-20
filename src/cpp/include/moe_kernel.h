#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <vector>

// This function declaration makes the C++ function `moe_forward_cuda` (defined in moe_kernel.cu)
// visible to other C++ files that include this header, like the bindings file.
torch::Tensor moe_forward_cuda(torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor);
