#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rollout_manager.h"

namespace py = pybind11;

// HELPER FUNCTION
namespace {
py::dict policy_request_to_dict(const PolicyRequest& req) {
    py::dict out;
    out["env"] = req.env;
    out["seat"] = req.seat;
    out["done"] = req.done;

    // This one is fine as is
    out["mask"] = py::array_t<uint8_t>({7}, req.mask);

    out["classic_obs"] = py::array_t<float>({(py::ssize_t)req.classic_obs_len}, req.classic_obs);
    out["classic_obs_len"] = req.classic_obs_len;
    
    const py::ssize_t valid_len = req.valid_len;
    
    // Create numpy arrays from raw pointers.
    // The C-style arrays are contiguous in memory, so this is safe.
    out["obs_sequence"] = py::array_t<float>({valid_len, static_cast<py::ssize_t>(OBS_DIM)}, &req.obs_sequence[0][0]);
    out["action_sequence"] = py::array_t<int64_t>({valid_len}, &req.action_sequence[0]);
    out["agent_type_sequence"] = py::array_t<int64_t>({valid_len}, &req.agent_type_sequence[0]);
    out["position_sequence"] = py::array_t<int64_t>({valid_len}, &req.position_sequence[0]);
    
    // *** THE MAIN FIX IS HERE ***
    // For 2D C-style arrays, it's safest to create a buffer_info object
    // to explicitly describe the memory layout (shape and strides).
    py::buffer_info action_mask_buf(
        (void*)req.action_mask_sequence,              // Pointer to buffer
        sizeof(uint8_t),                              // Size of one scalar
        py::format_descriptor<uint8_t>::format(),     // Python struct-style format descriptor
        2,                                            // Number of dimensions
        { valid_len, (py::ssize_t)7 },                // Shape of the matrix
        { sizeof(uint8_t) * 7, sizeof(uint8_t) }      // Strides (in bytes) for each index
    );
    out["action_mask_sequence"] = py::array_t<uint8_t>(action_mask_buf);

    out["valid_len"] = req.valid_len;
    return out;
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
             py::arg("training_policy_id"),
             py::arg("max_batch_envs"),
             py::arg("seed"),
             py::arg("cpp_bots"),
             py::arg("latest_historical_agents"),
             py::arg("active_shadow_agents"),
             py::arg("front_mass"),
             py::arg("shadow_mass"))
        
        .def("collect_requests_for_inference", [](RolloutManager& self) {
            auto grouped = self.collect_requests_for_inference();
            py::dict out;
            for (auto const& [policy_id, req_vec] : grouped) {
                py::list req_list;
                for (const auto& req : req_vec) {
                    req_list.append(policy_request_to_dict(req));
                }
                out[py::int_(policy_id)] = req_list;
            }
            return out;
        })

        .def("submit_inference_results", &RolloutManager::submit_inference_results,
             py::arg("policy_id"),
             py::arg("actions"),
             py::arg("log_probs") = std::vector<float>(),
             py::arg("values") = std::vector<float>())
             
        .def("get_completed_episodes", &RolloutManager::get_completed_episodes);
}