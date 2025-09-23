#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "bare_env.h"
#include "bots.h"
#include "ps.h"

namespace py = pybind11;

namespace {

PerfectSearch::BotFn make_botfn_from_cpp(bots::GreedyCardSpammer* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        return ptr->act(obs, len, mask);
    };
}

PerfectSearch::BotFn make_botfn_from_cpp(bots::TableFirstConservativeChallenger* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        return ptr->act(obs, len, mask);
    };
}

PerfectSearch::BotFn make_botfn_from_cpp(bots::SelectiveTableConservativeChallenger* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        return ptr->act(obs, len, mask);
    };
}

PerfectSearch::BotFn make_botfn_from_cpp(bots::TableNonTableAgent* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        return ptr->act(obs, len, mask);
    };
}

PerfectSearch::BotFn make_botfn_from_cpp(bots::Classic* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        return ptr->act(obs, len, mask);
    };
}

PerfectSearch::BotFn make_botfn_from_cpp(bots::StrategicChallenger* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        return ptr->act(obs, len, mask);
    };
}

PerfectSearch::BotFn make_botfn_from_cpp(bots::RandomAgent* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        return ptr->act(obs, len, mask);
    };
}

PerfectSearch::BotFn make_botfn_from_py(py::object f) {
    if (f.is_none()) {
        return PerfectSearch::BotFn();
    }
    return [f](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        py::gil_scoped_acquire gil;
        py::list py_obs(len);
        for (int i = 0; i < len; ++i) {
            py_obs[i] = obs[i];
        }
        py::list py_mask(7);
        for (int i = 0; i < 7; ++i) {
            py_mask[i] = static_cast<int>(mask[i]);
        }
        int a = py::cast<int>(f(py_obs, len, py_mask));
        return (a < 0 || a > 6) ? static_cast<uint8_t>(6) : static_cast<uint8_t>(a);
    };
}

}  // namespace

void bind_ps(py::module_& m) {
    py::class_<PerfectSearch>(m, "PerfectSearch")
        .def(py::init([](int my_index, py::list bot_list) {
            std::vector<PerfectSearch::BotFn> fns;
            fns.reserve(py::len(bot_list));
            for (py::handle h : bot_list) {
                py::object o = py::reinterpret_borrow<py::object>(h);
                PerfectSearch::BotFn fn;
                if (o.is_none()) {
                    fn = PerfectSearch::BotFn{};
                } else if (py::isinstance<bots::GreedyCardSpammer>(o)) {
                    fn = make_botfn_from_cpp(o.cast<bots::GreedyCardSpammer*>());
                } else if (py::isinstance<bots::TableFirstConservativeChallenger>(o)) {
                    fn = make_botfn_from_cpp(o.cast<bots::TableFirstConservativeChallenger*>());
                } else if (py::isinstance<bots::SelectiveTableConservativeChallenger>(o)) {
                    fn = make_botfn_from_cpp(o.cast<bots::SelectiveTableConservativeChallenger*>());
                } else if (py::isinstance<bots::TableNonTableAgent>(o)) {
                    fn = make_botfn_from_cpp(o.cast<bots::TableNonTableAgent*>());
                } else if (py::isinstance<bots::Classic>(o)) {
                    fn = make_botfn_from_cpp(o.cast<bots::Classic*>());
                } else if (py::isinstance<bots::StrategicChallenger>(o)) {
                    fn = make_botfn_from_cpp(o.cast<bots::StrategicChallenger*>());
                } else if (py::isinstance<bots::RandomAgent>(o)) {
                    fn = make_botfn_from_cpp(o.cast<bots::RandomAgent*>());
                } else {
                    fn = make_botfn_from_py(o);
                }
                fns.push_back(std::move(fn));
            }
            return std::make_unique<PerfectSearch>(my_index, fns);
        }), py::arg("my_index"), py::arg("bot_fns"))
        .def("search", [](PerfectSearch& ps, const Env& env) {
            float v = 0.f;
            uint8_t a = ps.search(env, &v);
            return py::make_tuple(static_cast<int>(a), v);
        })
        .def("next_planned_action", [](PerfectSearch& ps, int agent, const Env& env) {
            uint8_t a = 0;
            bool ok = ps.next_planned_action(agent, env, &a);
            return py::make_tuple(ok, static_cast<int>(a));
        })
        .def("set_sim_order", [](PerfectSearch& ps, const std::vector<int>& order) {
            std::vector<uint8_t> o;
            o.reserve(order.size());
            for (int x : order) {
                if (0 <= x && x <= 6) {
                    o.push_back(static_cast<uint8_t>(x));
                }
            }
            ps.set_sim_order(o);
        })
        .def("set_swap_heuristic", &PerfectSearch::set_swap_heuristic)
        .def("set_v5_penalty", &PerfectSearch::set_v5_penalty);
}
