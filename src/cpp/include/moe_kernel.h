#pragma once

#include <torch/extension.h>

// This function declaration makes the C++ function `moe_forward_cuda` (defined in moe_kernel.cu)
// visible to other C++ files that include this header, like the bindings file.
torch::Tensor moe_forward_cuda(torch::Tensor x,
                               torch::Tensor gate_logits,
                               torch::Tensor topk_indices,
                               torch::Tensor policy_indices,
                               torch::Tensor w1,
                               torch::Tensor b1,
                               torch::Tensor w2,
                               torch::Tensor b2);