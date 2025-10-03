#include "torch_utils.h"

#include <algorithm>
#include <cstring>

namespace {

struct CpuBatchTensors {
    torch::Tensor obs_sequence;
    torch::Tensor action_sequence;
    torch::Tensor agent_types;
    torch::Tensor positions;
    torch::Tensor action_masks;
    torch::Tensor padding_mask;
    torch::Tensor valid_lengths;
};

CpuBatchTensors allocate_cpu_batch(int64_t batch_size,
                                   int64_t pad_len,
                                   bool pin_memory) {
    auto float_opts = torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(torch::kCPU)
                          .pinned_memory(pin_memory);
    auto long_opts = torch::TensorOptions()
                         .dtype(torch::kInt64)
                         .device(torch::kCPU)
                         .pinned_memory(pin_memory);
    auto bool_opts = torch::TensorOptions()
                         .dtype(torch::kBool)
                         .device(torch::kCPU)
                         .pinned_memory(pin_memory);

    CpuBatchTensors batch;
    batch.obs_sequence = torch::zeros({batch_size, pad_len, OBS_DIM}, float_opts);
    batch.action_sequence = torch::zeros({batch_size, pad_len}, long_opts);
    batch.agent_types = torch::zeros({batch_size, pad_len}, long_opts);
    batch.positions = torch::zeros({batch_size, pad_len}, long_opts);
    batch.action_masks = torch::zeros({batch_size, pad_len, 7}, bool_opts);
    batch.padding_mask = torch::zeros({batch_size, pad_len}, bool_opts);
    batch.valid_lengths = torch::zeros({batch_size}, long_opts);
    return batch;
}

void fill_batch_from_requests(CpuBatchTensors& batch,
                              const std::vector<PolicyRequest>& requests,
                              const std::vector<size_t>& indices,
                              int64_t pad_len) {
    const int64_t batch_size = static_cast<int64_t>(indices.size());
    for (int64_t b = 0; b < batch_size; ++b) {
        const auto& req = requests[indices[static_cast<size_t>(b)]];
        const int64_t requested_len = std::max<int64_t>(0, req.valid_len);
        const int64_t used_len = std::max<int64_t>(1, std::min<int64_t>(requested_len, pad_len));

        batch.valid_lengths[b] = used_len;

        float* obs_ptr = batch.obs_sequence[b].data_ptr<float>();
        int64_t* action_ptr = batch.action_sequence[b].data_ptr<int64_t>();
        int64_t* agent_ptr = batch.agent_types[b].data_ptr<int64_t>();
        int64_t* pos_ptr = batch.positions[b].data_ptr<int64_t>();
        bool* mask_ptr = batch.action_masks[b].data_ptr<bool>();
        bool* pad_ptr = batch.padding_mask[b].data_ptr<bool>();

        const float* req_obs_ptr = req.obs_sequence.empty() ? nullptr : req.obs_sequence.data();
        const int64_t* req_action_ptr =
            req.action_sequence.empty() ? nullptr : req.action_sequence.data();
        const int64_t* req_agent_ptr =
            req.agent_type_sequence.empty() ? nullptr : req.agent_type_sequence.data();
        const int64_t* req_pos_ptr =
            req.position_sequence.empty() ? nullptr : req.position_sequence.data();
        const uint8_t* req_mask_ptr =
            req.action_mask_sequence.empty() ? nullptr : req.action_mask_sequence.data();

        const int64_t obs_rows =
            req_obs_ptr ? static_cast<int64_t>(req.obs_sequence.size()) / OBS_DIM : 0;
        const int64_t mask_rows =
            req_mask_ptr ? static_cast<int64_t>(req.action_mask_sequence.size()) / 7 : 0;
        const int64_t action_rows =
            req_action_ptr ? static_cast<int64_t>(req.action_sequence.size()) : 0;
        const int64_t agent_rows =
            req_agent_ptr ? static_cast<int64_t>(req.agent_type_sequence.size()) : 0;
        const int64_t pos_rows =
            req_pos_ptr ? static_cast<int64_t>(req.position_sequence.size()) : 0;

        for (int64_t t = 0; t < std::min<int64_t>(requested_len, pad_len); ++t) {
            float* dst_obs = obs_ptr + t * OBS_DIM;
            if (req_obs_ptr && t < obs_rows) {
                std::memcpy(dst_obs, req_obs_ptr + t * OBS_DIM, sizeof(float) * OBS_DIM);
            } else {
                std::memset(dst_obs, 0, sizeof(float) * OBS_DIM);
            }

            action_ptr[t] = (req_action_ptr && t < action_rows) ? req_action_ptr[t] : 0;
            agent_ptr[t] = (req_agent_ptr && t < agent_rows) ? req_agent_ptr[t] : 0;
            pos_ptr[t] = (req_pos_ptr && t < pos_rows) ? req_pos_ptr[t] : t;

            bool* step_mask = mask_ptr + t * 7;
            if (req_mask_ptr && t < mask_rows) {
                const uint8_t* src_mask = req_mask_ptr + t * 7;
                for (int j = 0; j < 7; ++j) {
                    step_mask[j] = src_mask[j] != 0;
                }
            } else {
                for (int j = 0; j < 7; ++j) {
                    step_mask[j] = false;
                }
            }
        }

        if (requested_len <= 0) {
            bool* step_mask = mask_ptr;
            for (int j = 0; j < 7; ++j) {
                step_mask[j] = req.mask[j] != 0;
            }
        }

        for (int64_t t = std::max<int64_t>(0, std::min<int64_t>(requested_len, pad_len));
             t < used_len;
             ++t) {
            pos_ptr[t] = t;
        }

        for (int64_t t = used_len; t < pad_len; ++t) {
            pad_ptr[t] = true;
        }
    }
}

std::vector<size_t> default_indices(size_t count) {
    std::vector<size_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = i;
    }
    return indices;
}

ModelInputBatch transfer_to_device(CpuBatchTensors&& cpu_batch,
                                   const torch::Device& device,
                                   bool non_blocking) {
    ModelInputBatch result;
    if (device.is_cuda()) {
        result.obs_sequence = cpu_batch.obs_sequence.to(device, non_blocking, /*copy=*/true);
        result.action_sequence = cpu_batch.action_sequence.to(device, non_blocking, true);
        result.agent_types = cpu_batch.agent_types.to(device, non_blocking, true);
        result.positions = cpu_batch.positions.to(device, non_blocking, true);
        result.action_masks = cpu_batch.action_masks.to(device, non_blocking, true);
        result.padding_mask = cpu_batch.padding_mask.to(device, non_blocking, true);
        result.valid_lengths = cpu_batch.valid_lengths.to(device, non_blocking, true);
    } else {
        result.obs_sequence = std::move(cpu_batch.obs_sequence);
        result.action_sequence = std::move(cpu_batch.action_sequence);
        result.agent_types = std::move(cpu_batch.agent_types);
        result.positions = std::move(cpu_batch.positions);
        result.action_masks = std::move(cpu_batch.action_masks);
        result.padding_mask = std::move(cpu_batch.padding_mask);
        result.valid_lengths = std::move(cpu_batch.valid_lengths);
    }
    return result;
}

}  // namespace

ModelInputBatch prepare_inference_batch(const std::vector<PolicyRequest>& requests,
                                        const torch::Device& device) {
    ModelInputBatch batch;
    if (requests.empty()) {
        return batch;
    }

    const int64_t batch_size = static_cast<int64_t>(requests.size());
    int64_t pad_len = 1;
    for (const auto& req : requests) {
        const int64_t clamped = std::max<int64_t>(1, req.valid_len);
        pad_len = std::max(pad_len, clamped);
    }

    const bool use_cuda = device.is_cuda();
    auto indices = default_indices(static_cast<size_t>(batch_size));
    CpuBatchTensors cpu_batch = allocate_cpu_batch(batch_size, pad_len, use_cuda);
    fill_batch_from_requests(cpu_batch, requests, indices, pad_len);
    return transfer_to_device(std::move(cpu_batch), device, use_cuda);
}

ModelInputBatch prepare_inference_batch(const std::vector<PolicyRequest>& requests,
                                        int64_t target_pad_len,
                                        const torch::Device& device,
                                        const std::vector<size_t>& indices) {
    ModelInputBatch batch;
    if (indices.empty()) {
        return batch;
    }

    const int64_t pad_len = std::max<int64_t>(1, target_pad_len);
    const int64_t batch_size = static_cast<int64_t>(indices.size());
    const bool use_cuda = device.is_cuda();

    CpuBatchTensors cpu_batch = allocate_cpu_batch(batch_size, pad_len, use_cuda);
    fill_batch_from_requests(cpu_batch, requests, indices, pad_len);
    return transfer_to_device(std::move(cpu_batch), device, use_cuda);
}
