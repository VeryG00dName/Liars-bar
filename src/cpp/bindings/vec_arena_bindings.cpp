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
    py::class_<PolicyRequest, std::unique_ptr<PolicyRequest, py::nodelete>>(m, "PolicyRequest")
        .def_readonly("env", &PolicyRequest::env)
        .def_readonly("seat", &PolicyRequest::seat)
        .def_property_readonly("mask", [](PolicyRequest& r) { return py::array_t<uint8_t>({7}, r.mask); })
        .def_readonly("done", &PolicyRequest::done)
        .def_property_readonly("classic_obs", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.classic_obs_len);
            return py::array_t<float>({len}, r.classic_obs);
        })
        .def_readonly("classic_obs_len", &PolicyRequest::classic_obs_len)
        .def_property_readonly("obs_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            return py::array_t<float>({len, static_cast<py::ssize_t>(OBS_DIM)}, r.obs_sequence[0]);
        })
        .def_property_readonly("action_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            return py::array_t<int64_t>({len}, r.action_sequence);
        })
        .def_property_readonly("agent_type_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            return py::array_t<int64_t>({len}, r.agent_type_sequence);
        })
        .def_property_readonly("position_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            return py::array_t<int64_t>({len}, r.position_sequence);
        })
        .def_property_readonly("action_mask_sequence", [](PolicyRequest& r) {
            const py::ssize_t len = std::max(0, r.valid_len);
            return py::array_t<uint8_t>({len, static_cast<py::ssize_t>(7)}, r.action_mask_sequence[0]);
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
            std::vector<std::vector<int>> roles;
            roles.resize(py::len(roles_list));
            for (size_t b = 0; b < roles.size(); ++b) {
                auto row = roles_list[b].cast<py::list>();
                roles[b].resize(py::len(row));
                for (size_t s = 0; s < py::len(row); ++s) {
                    roles[b][s] = py::cast<int>(row[s]);
                }
            }
            arena.set_roles(roles);
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
