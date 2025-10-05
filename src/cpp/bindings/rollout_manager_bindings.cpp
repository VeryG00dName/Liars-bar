#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rollout_manager.h"

namespace py = pybind11;

// HELPER FUNCTION
namespace {
py::dict policy_request_to_dict(const PolicyRequest& req) {
    py::dict out;
    out["env"] = req.env;
    out["seat"] = req.seat;
    out["done"] = req.done;

    out["mask"] = py::array_t<uint8_t>({7}, req.mask.data());

    out["classic_obs"] =
        py::array_t<float>({(py::ssize_t)req.classic_obs_len}, req.classic_obs.data());
    out["classic_obs_len"] = req.classic_obs_len;

    const py::ssize_t valid_len = req.valid_len;

    const float* obs_ptr = req.obs_sequence.empty() ? nullptr : req.obs_sequence.data();
    const int64_t* action_ptr =
        req.action_sequence.empty() ? nullptr : req.action_sequence.data();
    const int64_t* agent_ptr =
        req.agent_type_sequence.empty() ? nullptr : req.agent_type_sequence.data();
    const int64_t* pos_ptr =
        req.position_sequence.empty() ? nullptr : req.position_sequence.data();
    const uint8_t* mask_ptr =
        req.action_mask_sequence.empty() ? nullptr : req.action_mask_sequence.data();

    out["obs_sequence"] =
        py::array_t<float>({valid_len, static_cast<py::ssize_t>(OBS_DIM)}, obs_ptr);
    out["action_sequence"] = py::array_t<int64_t>({valid_len}, action_ptr);
    out["agent_type_sequence"] = py::array_t<int64_t>({valid_len}, agent_ptr);
    out["position_sequence"] = py::array_t<int64_t>({valid_len}, pos_ptr);

    // Explicitly describe the 2D layout to avoid stride issues.
    py::buffer_info action_mask_buf(
        (void*)mask_ptr,
        sizeof(uint8_t),
        py::format_descriptor<uint8_t>::format(),
        2,
        {valid_len, (py::ssize_t)7},
        {sizeof(uint8_t) * 7, sizeof(uint8_t)});
    out["action_mask_sequence"] = py::array_t<uint8_t>(action_mask_buf);

    out["valid_len"] = req.valid_len;
    return out;
}

py::dict prepared_batch_to_dict(const PreparedBatch& batch) {
    py::dict tensors;
    tensors["obs_sequence"] = batch.obs_sequence;
    tensors["action_sequence"] = batch.action_sequence;
    tensors["agent_types"] = batch.agent_types;
    tensors["positions"] = batch.positions;
    tensors["action_masks"] = batch.action_masks;
    tensors["padding_mask"] = batch.padding_mask;
    tensors["valid_lengths"] = batch.valid_lengths;
    tensors["env_indices"] = batch.env_indices;
    tensors["seat_indices"] = batch.seat_indices;
    return tensors;
}
} // end anonymous namespace

void bind_rollout_manager(py::module_& m) {
    py::class_<TrajectoryData>(m, "TrajectoryData")
        .def_readonly("env_index", &TrajectoryData::env_index)
        .def_readonly("training_policy_id", &TrajectoryData::training_policy_id)
        .def_readonly("training_agent_seat", &TrajectoryData::training_agent_seat)
        .def_readonly("player_policy_ids", &TrajectoryData::player_policy_ids)
        .def_readonly("agent_id", &TrajectoryData::agent_id)
        .def_readonly("our_action", &TrajectoryData::our_action)
        .def_readonly("log_prob", &TrajectoryData::log_prob)
        .def_readonly("value", &TrajectoryData::value)
        .def_readonly("reward", &TrajectoryData::reward)
        .def_readonly("done", &TrajectoryData::done)
        .def_readonly("opp_target_action", &TrajectoryData::opp_target_action)
        .def_readonly("penalties_used", &TrajectoryData::penalties_used)
        .def_readonly("episode_return", &TrajectoryData::episode_return)
        .def_readonly("win", &TrajectoryData::win);

    py::class_<RolloutManager>(m, "RolloutManager")
        .def(py::init<>())

        .def("start_rollouts", &RolloutManager::start_rollouts,
             py::arg("num_episodes"),
             py::arg("num_players"),
             py::arg("training_policy_ids"),
             py::arg("max_batch_envs"),
             py::arg("seed"),
             py::arg("opponent_labels") = std::vector<int>{},
             py::arg("opponent_weights") = std::vector<double>{},
             py::arg("opponent_triplets") = std::vector<std::vector<int>>{})
        
        .def("collect_requests_for_inference", [](RolloutManager& self) {
            auto grouped = self.collect_requests_for_inference();
            py::dict out;
            for (auto const& [policy_id, req_vec] : grouped) {
                py::dict payload;
                py::list req_list;
                for (const auto& req : req_vec) {
                    req_list.append(policy_request_to_dict(req));
                }
                payload["requests"] = req_list;
                if (self.is_training_policy(policy_id) && !req_vec.empty()) {
                    auto prepared = self.prepare_training_batch(req_vec, policy_id);
                    payload["tensors"] = prepared_batch_to_dict(prepared);
                } else {
                    payload["tensors"] = py::none();
                }
                out[py::int_(policy_id)] = payload;
            }
            return out;
        })

        .def("submit_inference_results",
             [](RolloutManager& self,
                int policy_id,
                py::object actions_obj,
                py::object log_probs_obj,
                py::object values_obj) {
                 auto actions_arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(actions_obj);
                 if (!actions_arr) {
                     throw py::type_error("actions must be convertible to a uint8 numpy array");
                 }
                 const uint8_t* actions_ptr = actions_arr.size() > 0
                                                 ? static_cast<const uint8_t*>(actions_arr.data())
                                                 : nullptr;
                 const size_t action_count = static_cast<size_t>(actions_arr.size());

                 const float* log_ptr = nullptr;
                 size_t log_count = 0;
                 py::array_t<float, py::array::c_style | py::array::forcecast> log_arr;
                 if (!log_probs_obj.is_none()) {
                     log_arr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(log_probs_obj);
                     if (!log_arr) {
                         throw py::type_error("log_probs must be convertible to a float32 numpy array");
                     }
                     log_ptr = log_arr.size() > 0 ? static_cast<const float*>(log_arr.data()) : nullptr;
                     log_count = static_cast<size_t>(log_arr.size());
                 }

                 const float* val_ptr = nullptr;
                 size_t val_count = 0;
                 py::array_t<float, py::array::c_style | py::array::forcecast> val_arr;
                 if (!values_obj.is_none()) {
                     val_arr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(values_obj);
                     if (!val_arr) {
                         throw py::type_error("values must be convertible to a float32 numpy array");
                     }
                     val_ptr = val_arr.size() > 0 ? static_cast<const float*>(val_arr.data()) : nullptr;
                     val_count = static_cast<size_t>(val_arr.size());
                 }

                 self.submit_inference_results_array(
                     policy_id,
                     actions_ptr,
                     action_count,
                     log_ptr,
                     log_count,
                     val_ptr,
                     val_count);
             },
             py::arg("policy_id"),
             py::arg("actions"),
             py::arg("log_probs") = py::none(),
             py::arg("values") = py::none())
             
        .def("get_completed_episodes", &RolloutManager::get_completed_episodes)
        .def("load_historical_model",
            [](RolloutManager& self, int policy_id, const std::string& path) {
                self.load_historical_model(policy_id, path);
            },
            py::arg("policy_id"),
            py::arg("path"))
        .def("register_cpp_bot",
             [](RolloutManager& self, int policy_id, const std::string& name) {
                 self.register_cpp_bot(policy_id, name);
             },
             py::arg("policy_id"),
             py::arg("name"))
        .def("set_training_device", &RolloutManager::set_training_device,
             py::arg("device"))
        .def("set_max_sequence_length", &RolloutManager::set_max_sequence_length,
             py::arg("max_seq_length"))
        .def("set_policy_max_sequence_length", &RolloutManager::set_policy_max_sequence_length,
             py::arg("policy_id"), py::arg("max_seq_length"));
}
