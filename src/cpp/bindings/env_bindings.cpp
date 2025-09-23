#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bare_env.h"

namespace py = pybind11;

namespace {

std::vector<int> env_valid_actions_py(const Env& e) {
    uint8_t mask[7];
    e.valid_actions(mask);
    std::vector<int> out(7);
    for (int i = 0; i < 7; ++i) {
        out[i] = static_cast<int>(mask[i]);
    }
    return out;
}

std::vector<float> env_observe_vector_py(const Env& e) {
    float buf[3 + Env::MAX_PLAYERS];
    int n = e.observe_vector(buf);
    int want = 3 + e.num_players();
    int take = std::min(n, want);
    std::vector<float> out(take);
    for (int i = 0; i < take; ++i) {
        out[i] = buf[i];
    }
    return out;
}

std::vector<float> env_observe_newerest_py(const Env& e, int agent_index) {
    if (agent_index < 0 || agent_index >= e.num_players()) {
        throw std::runtime_error("observe_newerest: agent_index out of range");
    }
    float buf[2 * Env::MAX_PLAYERS + 1];
    int n = e.observe_vector_newerest(agent_index, buf);
    if (n < 0) {
        n = 0;
    }
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = buf[i];
    }
    return out;
}

std::vector<int> env_penalties_list(const Env& e) {
    std::vector<int> values(e.num_players());
    for (int i = 0; i < e.num_players(); ++i) {
        values[i] = static_cast<int>(e.penalties[i]);
    }
    return values;
}

std::vector<int> env_terminations_list(const Env& e) {
    std::vector<int> values(e.num_players());
    for (int i = 0; i < e.num_players(); ++i) {
        values[i] = static_cast<int>(e.terminations[i]);
    }
    return values;
}

py::dict history_entry_to_py(const HistoryEntry& h) {
    py::dict obs_map;
    for (size_t p = 0; p < h.observations.size(); ++p) {
        const auto& values = h.observations[p];
        py::list py_values;
        for (float x : values) {
            py_values.append(x);
        }
        obs_map[py::int_(p)] = py_values;
    }
    py::list mask7;
    for (int i = 0; i < 7; ++i) {
        mask7.append(static_cast<int>(h.mask[i]));
    }
    py::dict d;
    d["player"] = h.player;
    d["action"] = static_cast<int>(h.action);
    d["step"] = h.step;
    d["observations"] = obs_map;
    d["mask"] = mask7;
    return d;
}

py::list env_game_history_py(const Env& e) {
    py::list out;
    for (const auto& entry : e.game_history) {
        out.append(history_entry_to_py(entry));
    }
    return out;
}

py::list env_game_history_slice_py(const Env& e, int start_index, int end_index) {
    const int total = e.get_total_history_entries();
    if (start_index < 0) {
        start_index = 0;
    }
    if (end_index < start_index) {
        end_index = start_index;
    }
    if (end_index > total) {
        end_index = total;
    }

    py::list out;
    for (int i = start_index; i < end_index; ++i) {
        out.append(history_entry_to_py(e.game_history[i]));
    }
    return out;
}

py::array_t<int32_t> env_game_history_slice_basic(const Env& e, int start_index, int end_index) {
    const int total = e.get_total_history_entries();
    if (start_index < 0) {
        start_index = 0;
    }
    if (end_index < start_index) {
        end_index = start_index;
    }
    if (end_index > total) {
        end_index = total;
    }

    const int count = end_index - start_index;
    py::array_t<int32_t> out({std::max(0, count), 2});
    auto buf = out.mutable_unchecked<2>();
    for (int i = 0; i < count; ++i) {
        const auto& entry = e.game_history[start_index + i];
        buf(i, 0) = entry.player;
        buf(i, 1) = static_cast<int32_t>(entry.action);
    }
    return out;
}

}  // namespace

void bind_env(py::module_& m) {
    py::class_<Env>(m, "Env")
        .def(py::init<>())
        .def("reset", &Env::reset, py::arg("players"), py::arg("seed") = 0u)
        .def("set_seed", &Env::set_seed, py::arg("seed"))
        .def("valid_actions", &env_valid_actions_py)
        .def("observe_vector", &env_observe_vector_py)
        .def("observe_newerest", &env_observe_newerest_py, py::arg("agent_index"))
        .def("step", &Env::step, py::arg("action"))
        .def("current_player", &Env::current_player)
        .def("num_players", &Env::num_players)
        .def("game_history", &env_game_history_py)
        .def("total_history_entries", &Env::get_total_history_entries)
        .def("history_slice", &env_game_history_slice_py, py::arg("start_index"), py::arg("end_index"))
        .def("history_slice_basic", &env_game_history_slice_basic, py::arg("start_index"), py::arg("end_index"))
        .def_property_readonly("penalties", &env_penalties_list)
        .def_property_readonly("terminations", &env_terminations_list);
}
