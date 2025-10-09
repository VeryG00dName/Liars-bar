#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rollout_manager.h"

namespace py = pybind11;

namespace {
c10::Dict<c10::IValue, c10::IValue> py_weights_to_ivalue_dict(const py::dict& weights) {
    c10::Dict<c10::IValue, c10::IValue> dict(c10::StringType::get(), c10::TensorType::get());
    dict.reserve(weights.size());
    for (auto item : weights) {
        std::string key = py::cast<std::string>(item.first);
        torch::Tensor tensor = py::cast<torch::Tensor>(item.second);
        tensor = tensor.detach().to(torch::kCPU).contiguous();
        dict.insert(c10::IValue(key), c10::IValue(tensor));
    }
    return dict;
}
} // namespace

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
        .def_readonly("opp_target_action", &TrajectoryData::opp_target_action)
        .def_readonly("win", &TrajectoryData::win);

    py::class_<RolloutManager>(m, "RolloutManager")
        .def(py::init<>())

        // Consolidated helper to run and collect rollouts in one call
        .def("get_rollouts", &RolloutManager::get_rollouts,
             py::arg("num_episodes"),
             py::arg("num_players"),
             py::arg("training_policy_ids"),
             py::arg("max_batch_envs"),
             py::arg("seed"),
             py::arg("opponent_triplets"))

        .def("start_rollouts", &RolloutManager::start_rollouts,
             py::arg("num_episodes"),
             py::arg("num_players"),
             py::arg("training_policy_ids"),
             py::arg("max_batch_envs"),
             py::arg("seed"),
             py::arg("opponent_labels") = std::vector<int>{},
             py::arg("opponent_weights") = std::vector<double>{},
             py::arg("opponent_triplets") = std::vector<std::vector<int>>{})
        .def("run_rollouts_step", &RolloutManager::run_rollouts_step)
        .def("all_episodes_complete", &RolloutManager::all_episodes_complete)
        .def("get_completed_episodes", &RolloutManager::get_completed_episodes)
        .def("load_model_architecture", &RolloutManager::load_model_architecture,
             py::arg("path"))
        .def("load_policy_weights", &RolloutManager::load_policy_weights,
             py::arg("policy_id"),
             py::arg("path"))
        .def("update_learner_weights",
             [](RolloutManager& self, int policy_id, const py::dict& weights) {
                 self.update_learner_weights(policy_id, py_weights_to_ivalue_dict(weights));
             },
             py::arg("policy_id"),
             py::arg("weights"))
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
