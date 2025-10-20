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

// Temporary grouped FFN entry point. The actual implementation will be provided
// by a fused CUDA kernel; for now, this stub allows the forward path to compile
// while we finish wiring up the dispatch logic in C++.
void grouped_ffn_gemm_forward(
    const std::vector<uintptr_t>& input_ptrs,
    const std::vector<uintptr_t>& w1_ptrs,
    const std::vector<uintptr_t>& b1_ptrs,
    const std::vector<uintptr_t>& w2_ptrs,
    const std::vector<uintptr_t>& b2_ptrs,
    const std::vector<uintptr_t>& output_ptrs,
    const std::vector<int64_t>& m_sizes,
    int64_t hidden_dim,
    int64_t ffn_dim);