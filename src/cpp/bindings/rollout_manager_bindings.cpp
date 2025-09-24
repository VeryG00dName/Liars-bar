#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rollout_manager.h"

namespace py = pybind11;

// HELPER FUNCTION COPIED FROM AGENT 1's SUBMISSION
namespace {
py::dict policy_request_to_dict(const PolicyRequest& req) {
    py::dict out;
    out["env"] = req.env;
    out["seat"] = req.seat;
    out["done"] = req.done;

    // Correctly copies fixed-size array to numpy array
    out["mask"] = py::array_t<uint8_t>({7}, req.mask);

    // Correctly copies variable-length array using len field
    out["classic_obs"] = py::array_t<float>({(py::ssize_t)req.classic_obs_len}, req.classic_obs);
    out["classic_obs_len"] = req.classic_obs_len;
    
    const py::ssize_t valid_len = req.valid_len;
    // Note: This part can be slow due to element-wise copying.
    // However, it's safer than dealing with complex buffer protocols for now.
    out["obs_sequence"] = py::array_t<float>({valid_len, static_cast<py::ssize_t>(OBS_DIM)}, (float*)req.obs_sequence);
    out["action_sequence"] = py::array_t<int64_t>({valid_len}, req.action_sequence);
    out["agent_type_sequence"] = py::array_t<int64_t>({valid_len}, req.agent_type_sequence);
    out["position_sequence"] = py::array_t<int64_t>({valid_len}, req.position_sequence);
    out["action_mask_sequence"] = py::array_t<uint8_t>({valid_len, 7}, (uint8_t*)req.action_mask_sequence);
    out["valid_len"] = req.valid_len;
    return out;
}
} // end anonymous namespace

// MAIN BINDING FUNCTION (FROM AGENT 3)
void bind_rollout_manager(py::module_& m) {
    // This binding from Agent 3 is clean and correct
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
        .def("start_rollouts", &RolloutManager::start_rollouts, /* ... args ... */ ) // Rest of Agent 3's bindings are good
        
        // MODIFIED BINDING using the helper function
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

        .def("submit_inference_results", &RolloutManager::submit_inference_results, /* ... args ... */)
        .def("get_completed_episodes", &RolloutManager::get_completed_episodes);
}