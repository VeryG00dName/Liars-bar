#include <torch/extension.h>
#include "cfr_manager.h"

namespace py = pybind11;

void bind_cfr_manager(py::module& m) {
    py::class_<CFRManager>(m, "CFRManager")
        .def(py::init<int, int, int>(), 
             py::arg("num_players"), 
             py::arg("max_depth")=100, 
             py::arg("num_workers")=1024)
        .def("load_advantage_network", &CFRManager::load_advantage_network, "Load Advantage Network weights")
        .def("run_cfr_traversal", &CFRManager::run_cfr_traversal, py::arg("num_traversals"), "Run CFR traversals")
        .def("collect_samples", &CFRManager::collect_samples, "Collect generated samples from buffers");
}
