#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "vec_arena.h"

namespace py = pybind11;

void bind_vec_arena(py::module_& m) {
    py::class_<PolicyRequest>(m, "PolicyRequest")
        .def_readonly("env", &PolicyRequest::env)
        .def_readonly("seat", &PolicyRequest::seat)
        .def_property_readonly("mask", [](PolicyRequest& r) { return py::array_t<uint8_t>({7}, r.mask.data()); })
        .def_readonly("done", &PolicyRequest::done)
        .def_property_readonly("classic_obs", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.classic_obs_len);
            return py::array_t<float>({len}, r.classic_obs.data());
        })
        .def_readonly("classic_obs_len", &PolicyRequest::classic_obs_len)
        .def_property_readonly("obs_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            const float* ptr = r.obs_sequence.empty() ? nullptr : r.obs_sequence.data();
            return py::array_t<float>({len, static_cast<py::ssize_t>(OBS_DIM)}, ptr);
        })
        .def_property_readonly("action_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            const int64_t* ptr = r.action_sequence.empty() ? nullptr : r.action_sequence.data();
            return py::array_t<int64_t>({len}, ptr);
        })
        .def_property_readonly("agent_type_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            const int64_t* ptr = r.agent_type_sequence.empty() ? nullptr : r.agent_type_sequence.data();
            return py::array_t<int64_t>({len}, ptr);
        })
        .def_property_readonly("position_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            const int64_t* ptr = r.position_sequence.empty() ? nullptr : r.position_sequence.data();
            return py::array_t<int64_t>({len}, ptr);
        })
        .def_property_readonly("action_mask_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            const uint8_t* ptr = r.action_mask_sequence.empty() ? nullptr : r.action_mask_sequence.data();
            return py::array_t<uint8_t>({len, static_cast<py::ssize_t>(7)}, ptr);
        })
        .def_readonly("valid_len", &PolicyRequest::valid_len);

    py::class_<VecArena>(m, "VecArena")
        .def(py::init<>())
        .def_readonly("B", &VecArena::B, "Batch size (number of environments)")
        .def_readonly("n_players", &VecArena::n_players, "Number of players per environment")
        .def_property_readonly("done", [](VecArena& self) {
            return py::array_t<uint8_t>(self.done.size(), self.done.data());
        }, "Get the done status (0 or 1) for each environment in the batch.")
        .def("reset", &VecArena::reset, py::arg("batch"), py::arg("players"), py::arg("seed"))
        .def("obs_dim", &VecArena::obs_dim)
        .def("get_env", [](VecArena& self, int i) -> Env& {
            if (i < 0 || i >= self.B) {
                throw py::index_error("Environment index out of range");
            }
            return self.envs.at(i);
        }, py::return_value_policy::reference_internal, "Get a reference to an environment by index.")
        .def("set_roles", [](VecArena& arena, const py::list& roles_list) {
            // roles_list is a list of dicts: [{agent_index: Role}, ...]
            // Each Role is a dict with "policy_id" (agent_index is the map key, no trajectory_id needed)
            std::vector<std::unordered_map<int, Role>> roles_by_agent_index;
            roles_by_agent_index.resize(py::len(roles_list));
            for (size_t b = 0; b < roles_by_agent_index.size(); ++b) {
                auto env_roles = roles_list[b].cast<py::dict>();
                for (auto item : env_roles) {
                    int agent_idx = py::cast<int>(item.first);
                    py::dict role_dict = py::cast<py::dict>(item.second);
                    Role role;
                    role.policy_id = py::cast<int>(role_dict["policy_id"]);
                    // No trajectory_id needed - we use agent_index from the map key!
                    roles_by_agent_index[b][agent_idx] = role;
                }
            }
            arena.set_roles(roles_by_agent_index);
        }, py::arg("roles"))
        .def("collect_requests", [](VecArena& arena) {
            py::dict out;
            const auto& grouped = arena.collect_requests();
            for (auto& kv : grouped) {
                py::list lst;
                auto& reqs = kv.second;
                for (auto& r : reqs) {
                    lst.append(py::cast(&r, py::return_value_policy::reference));
                }
                out[py::int_(kv.first)] = lst;
            }
            return out;
        })
        .def("submit_actions", [](VecArena& arena, int policy_id, const py::array_t<uint8_t>& actions) {
            std::vector<uint8_t> v(actions.size());
            std::copy(actions.data(), actions.data() + actions.size(), v.begin());
            arena.submit_actions(policy_id, v);
        }, py::arg("policy_id"), py::arg("actions"));
}
