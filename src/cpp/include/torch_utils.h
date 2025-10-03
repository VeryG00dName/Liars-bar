#pragma once

#include "vec_arena.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <torch/torch.h>

struct ModelInputBatch {
    torch::Tensor obs_sequence;
    torch::Tensor action_sequence;
    torch::Tensor agent_types;
    torch::Tensor positions;
    torch::Tensor action_masks;
    torch::Tensor padding_mask;
    torch::Tensor valid_lengths;
};

ModelInputBatch prepare_inference_batch(const std::vector<PolicyRequest>& requests,
                                        const torch::Device& device);

ModelInputBatch prepare_inference_batch(const std::vector<PolicyRequest>& requests,
                                        int64_t target_pad_len,
                                        const torch::Device& device,
                                        const std::vector<size_t>& indices);
