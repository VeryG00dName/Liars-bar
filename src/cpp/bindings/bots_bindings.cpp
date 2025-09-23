#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bots.h"

namespace py = pybind11;

namespace {

void seq_to_mask_7(const py::sequence& s, uint8_t out[7]) {
    if (static_cast<size_t>(py::len(s)) < 7) {
        throw std::runtime_error("mask needs length >= 7");
    }
    for (int i = 0; i < 7; ++i) {
        out[i] = static_cast<uint8_t>(py::cast<int>(s[i]));
    }
}

}  // namespace

void bind_bots(py::module_& m) {
    py::class_<bots::GreedyCardSpammer>(m, "GreedyCardSpammer")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::GreedyCardSpammer& self, py::sequence obs, int length, py::sequence mask) -> int {
            std::vector<float> v;
            v.reserve(length);
            for (auto item : obs) {
                v.push_back(py::cast<float>(item));
            }
            uint8_t mask7[7];
            seq_to_mask_7(mask, mask7);
            return static_cast<int>(self.act(v.data(), static_cast<int>(v.size()), mask7));
        });

    py::class_<bots::TableFirstConservativeChallenger>(m, "TableFirstConservativeChallenger")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::TableFirstConservativeChallenger& self, py::sequence obs, int length, py::sequence mask) -> int {
            std::vector<float> v;
            v.reserve(length);
            for (auto item : obs) {
                v.push_back(py::cast<float>(item));
            }
            uint8_t mask7[7];
            seq_to_mask_7(mask, mask7);
            return static_cast<int>(self.act(v.data(), static_cast<int>(v.size()), mask7));
        });

    py::class_<bots::SelectiveTableConservativeChallenger>(m, "SelectiveTableConservativeChallenger")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::SelectiveTableConservativeChallenger& self, py::sequence obs, int length, py::sequence mask) -> int {
            std::vector<float> v;
            v.reserve(length);
            for (auto item : obs) {
                v.push_back(py::cast<float>(item));
            }
            uint8_t mask7[7];
            seq_to_mask_7(mask, mask7);
            return static_cast<int>(self.act(v.data(), static_cast<int>(v.size()), mask7));
        });

    py::class_<bots::TableNonTableAgent>(m, "TableNonTableAgent")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::TableNonTableAgent& self, py::sequence obs, int length, py::sequence mask) -> int {
            std::vector<float> v;
            v.reserve(length);
            for (auto item : obs) {
                v.push_back(py::cast<float>(item));
            }
            uint8_t mask7[7];
            seq_to_mask_7(mask, mask7);
            return static_cast<int>(self.act(v.data(), static_cast<int>(v.size()), mask7));
        });

    py::class_<bots::Classic>(m, "Classic")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::Classic& self, py::sequence obs, int length, py::sequence mask) -> int {
            std::vector<float> v;
            v.reserve(length);
            for (auto item : obs) {
                v.push_back(py::cast<float>(item));
            }
            uint8_t mask7[7];
            seq_to_mask_7(mask, mask7);
            return static_cast<int>(self.act(v.data(), static_cast<int>(v.size()), mask7));
        });

    py::class_<bots::StrategicChallenger>(m, "StrategicChallenger")
        .def(py::init<const char*, int, int>(), py::arg("name"), py::arg("num_players"), py::arg("agent_index"))
        .def("act", [](bots::StrategicChallenger& self, py::sequence obs, int length, py::sequence mask) -> int {
            std::vector<float> v;
            v.reserve(length);
            for (auto item : obs) {
                v.push_back(py::cast<float>(item));
            }
            uint8_t mask7[7];
            seq_to_mask_7(mask, mask7);
            return static_cast<int>(self.act(v.data(), static_cast<int>(v.size()), mask7));
        });

    py::class_<bots::RandomAgent>(m, "RandomAgent")
        .def(py::init<const char*>(), py::arg("name"))
        .def("set_seed", &bots::RandomAgent::set_seed)
        .def("act", [](bots::RandomAgent& self, py::sequence obs, int length, py::sequence mask) -> int {
            std::vector<float> v;
            v.reserve(length);
            for (auto item : obs) {
                v.push_back(py::cast<float>(item));
            }
            uint8_t mask7[7];
            seq_to_mask_7(mask, mask7);
            return static_cast<int>(self.act(v.data(), static_cast<int>(v.size()), mask7));
        });
}
