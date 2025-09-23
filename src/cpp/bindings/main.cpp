#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_env(py::module_& m);
void bind_bots(py::module_& m);
void bind_ps(py::module_& m);
void bind_vec_arena(py::module_& m);

PYBIND11_MODULE(lb, m) {
    m.doc() = "Liar's Bar: Env, Bots, PerfectSearch, VecArena";

    bind_env(m);
    bind_bots(m);
    bind_ps(m);
    bind_vec_arena(m);
}
