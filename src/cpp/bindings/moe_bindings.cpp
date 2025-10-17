#include <pybind11/pybind11.h>

#include "moe_kernel.h"

namespace py = pybind11;

void bind_moe(py::module_& m) {
    m.def("moe_forward",
          &moe_forward_cuda,
          "Fused mixture-of-experts forward pass",
          py::arg("x"),
          py::arg("gate_logits"),
          py::arg("topk_indices"),
          py::arg("w1"),
          py::arg("b1"),
          py::arg("w2"),
          py::arg("b2"));
}

