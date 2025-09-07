#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "bare_env.h"
#include "bots.h"
#include "ps.h"
#include "vec_arena.h"
#include "roles.h"

namespace py = pybind11;

// ---------------- Env helpers ----------------
static std::vector<int> env_valid_actions_py(const Env& e) {
    uint8_t m[7]; e.valid_actions(m);
    std::vector<int> out(7);
    for (int i = 0; i < 7; ++i) out[i] = (int)m[i];
    return out;
}

static std::vector<float> env_observe_vector_py(const Env& e) {
    float buf[3 + Env::MAX_PLAYERS];
    int n = e.observe_vector(buf);
    int want = 3 + e.num_players();
    int take = std::min(n, want);
    std::vector<float> out(take);
    for (int i = 0; i < take; ++i) out[i] = buf[i];
    return out;
}

static std::vector<float> env_observe_newerest_py(const Env& e, int agent_index) {
    if (agent_index < 0 || agent_index >= e.num_players())
        throw std::runtime_error("observe_newerest: agent_index out of range");
    float buf[2 * Env::MAX_PLAYERS + 1];
    int n = e.observe_vector_newerest(agent_index, buf);
    if (n < 0) n = 0;
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) out[i] = buf[i];
    return out;
}

static std::vector<int> env_penalties_list(const Env& e) {
    std::vector<int> v(e.num_players());
    for (int i = 0; i < e.num_players(); ++i) v[i] = (int)e.penalties[i];
    return v;
}

static std::vector<int> env_terminations_list(const Env& e) {
    std::vector<int> v(e.num_players());
    for (int i = 0; i < e.num_players(); ++i) v[i] = (int)e.terminations[i];
    return v;
}

static py::list env_game_history_py(const Env& e) {
    py::list out;
    for (const auto& h : e.game_history) {
        py::dict obs_map;
        for (size_t p = 0; p < h.observations.size(); ++p) {
            const auto& v = h.observations[p];
            py::list pyv;
            for (float x : v) pyv.append(x);
            obs_map[py::int_(p)] = pyv;
        }
        py::list mask7;
        for (int i = 0; i < 7; ++i) mask7.append((int)h.mask[i]);
        py::dict d;
        d["player"] = h.player;
        d["action"] = (int)h.action;
        d["step"] = h.step;
        d["observations"] = obs_map;
        d["mask"] = mask7;
        out.append(d);
    }
    return out;
}

static void seq_to_mask_7(const py::sequence& s, uint8_t out[7]) {
    if ((size_t)py::len(s) < 7) throw std::runtime_error("mask needs length >= 7");
    for (int i = 0; i < 7; ++i) out[i] = (uint8_t)py::cast<int>(s[i]);
}

// --------------- PerfectSearch helpers ----------------
static PerfectSearch::BotFn make_botfn_from_cpp(bots::GreedyCardSpammer* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t { return ptr->act(obs, len, mask); };
}
static PerfectSearch::BotFn make_botfn_from_cpp(bots::TableFirstConservativeChallenger* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t { return ptr->act(obs, len, mask); };
}
static PerfectSearch::BotFn make_botfn_from_cpp(bots::SelectiveTableConservativeChallenger* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t { return ptr->act(obs, len, mask); };
}
static PerfectSearch::BotFn make_botfn_from_cpp(bots::TableNonTableAgent* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t { return ptr->act(obs, len, mask); };
}
static PerfectSearch::BotFn make_botfn_from_cpp(bots::Classic* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t { return ptr->act(obs, len, mask); };
}
static PerfectSearch::BotFn make_botfn_from_cpp(bots::StrategicChallenger* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t { return ptr->act(obs, len, mask); };
}
static PerfectSearch::BotFn make_botfn_from_cpp(bots::RandomAgent* ptr) {
    return [ptr](const float* obs, int len, const uint8_t mask[7]) -> uint8_t { return ptr->act(obs, len, mask); };
}

static PerfectSearch::BotFn make_botfn_from_py(py::object f) {
    if (f.is_none()) return PerfectSearch::BotFn();
    return [f](const float* obs, int len, const uint8_t mask[7]) -> uint8_t {
        py::gil_scoped_acquire gil;
        py::list py_obs(len); for (int i = 0; i < len; ++i) py_obs[i] = obs[i];
        py::list py_mask(7); for (int i = 0; i < 7; ++i) py_mask[i] = (int)mask[i];
        int a = py::cast<int>(f(py_obs, len, py_mask));
        return (a < 0 || a>6) ? (uint8_t)6 : (uint8_t)a;
        };
}

PYBIND11_MODULE(lb, m) {
    m.doc() = "Liar's Bar: Env, Bots, PerfectSearch, VecArena";

    // ---------------- Env ----------------
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
        .def_property_readonly("penalties", &env_penalties_list)
        .def_property_readonly("terminations", &env_terminations_list)
        ;

    // ---------------- Bots ----------------
    py::class_<bots::GreedyCardSpammer>(m, "GreedyCardSpammer")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::GreedyCardSpammer& self, py::sequence obs, int length, py::sequence mask)->int {
        std::vector<float> v; v.reserve(length); for (auto item : obs) v.push_back(py::cast<float>(item));
        uint8_t m7[7]; seq_to_mask_7(mask, m7); return (int)self.act(v.data(), (int)v.size(), m7); });
    py::class_<bots::TableFirstConservativeChallenger>(m, "TableFirstConservativeChallenger")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::TableFirstConservativeChallenger& self, py::sequence obs, int length, py::sequence mask)->int {
        std::vector<float> v; v.reserve(length); for (auto item : obs) v.push_back(py::cast<float>(item));
        uint8_t m7[7]; seq_to_mask_7(mask, m7); return (int)self.act(v.data(), (int)v.size(), m7); });
    py::class_<bots::SelectiveTableConservativeChallenger>(m, "SelectiveTableConservativeChallenger")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::SelectiveTableConservativeChallenger& self, py::sequence obs, int length, py::sequence mask)->int {
        std::vector<float> v; v.reserve(length); for (auto item : obs) v.push_back(py::cast<float>(item));
        uint8_t m7[7]; seq_to_mask_7(mask, m7); return (int)self.act(v.data(), (int)v.size(), m7); });
    py::class_<bots::TableNonTableAgent>(m, "TableNonTableAgent")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::TableNonTableAgent& self, py::sequence obs, int length, py::sequence mask)->int {
        std::vector<float> v; v.reserve(length); for (auto item : obs) v.push_back(py::cast<float>(item));
        uint8_t m7[7]; seq_to_mask_7(mask, m7); return (int)self.act(v.data(), (int)v.size(), m7); });
    py::class_<bots::Classic>(m, "Classic")
        .def(py::init<const char*>(), py::arg("name"))
        .def("act", [](bots::Classic& self, py::sequence obs, int length, py::sequence mask)->int {
        std::vector<float> v; v.reserve(length); for (auto item : obs) v.push_back(py::cast<float>(item));
        uint8_t m7[7]; seq_to_mask_7(mask, m7); return (int)self.act(v.data(), (int)v.size(), m7); });
    py::class_<bots::StrategicChallenger>(m, "StrategicChallenger")
        .def(py::init<const char*, int, int>(), py::arg("name"), py::arg("num_players"), py::arg("agent_index"))
        .def("act", [](bots::StrategicChallenger& self, py::sequence obs, int length, py::sequence mask)->int {
        std::vector<float> v; v.reserve(length); for (auto item : obs) v.push_back(py::cast<float>(item));
        uint8_t m7[7]; seq_to_mask_7(mask, m7); return (int)self.act(v.data(), (int)v.size(), m7); });
    py::class_<bots::RandomAgent>(m, "RandomAgent")
        .def(py::init<const char*>(), py::arg("name"))
        .def("set_seed", &bots::RandomAgent::set_seed)
        .def("act", [](bots::RandomAgent& self, py::sequence obs, int length, py::sequence mask)->int {
        std::vector<float> v; v.reserve(length); for (auto item : obs) v.push_back(py::cast<float>(item));
        uint8_t m7[7]; seq_to_mask_7(mask, m7); return (int)self.act(v.data(), (int)v.size(), m7); });

    // ---------------- PerfectSearch ----------------
    py::class_<PerfectSearch>(m, "PerfectSearch")
        .def(py::init([](int my_index, py::list bot_list) {
        std::vector<PerfectSearch::BotFn> fns; fns.reserve(py::len(bot_list));
        for (py::handle h : bot_list) {
            py::object o = py::reinterpret_borrow<py::object>(h); PerfectSearch::BotFn fn;
            if (o.is_none()) { fn = PerfectSearch::BotFn{}; }
            else if (py::isinstance<bots::GreedyCardSpammer>(o)) { fn = make_botfn_from_cpp(o.cast<bots::GreedyCardSpammer*>()); }
            else if (py::isinstance<bots::TableFirstConservativeChallenger>(o)) { fn = make_botfn_from_cpp(o.cast<bots::TableFirstConservativeChallenger*>()); }
            else if (py::isinstance<bots::SelectiveTableConservativeChallenger>(o)) { fn = make_botfn_from_cpp(o.cast<bots::SelectiveTableConservativeChallenger*>()); }
            else if (py::isinstance<bots::TableNonTableAgent>(o)) { fn = make_botfn_from_cpp(o.cast<bots::TableNonTableAgent*>()); }
            else if (py::isinstance<bots::Classic>(o)) { fn = make_botfn_from_cpp(o.cast<bots::Classic*>()); }
            else if (py::isinstance<bots::StrategicChallenger>(o)) { fn = make_botfn_from_cpp(o.cast<bots::StrategicChallenger*>()); }
            else if (py::isinstance<bots::RandomAgent>(o)) { fn = make_botfn_from_cpp(o.cast<bots::RandomAgent*>()); }
            else { fn = make_botfn_from_py(o); }
            fns.push_back(std::move(fn));
        }
        return std::make_unique<PerfectSearch>(my_index, fns);
            }), py::arg("my_index"), py::arg("bot_fns"))
        .def("search", [](PerfectSearch& ps, const Env& env) {
        float v = 0.f; uint8_t a = ps.search(env, &v); return py::make_tuple((int)a, v); })
        .def("next_planned_action", [](PerfectSearch& ps, int agent, const Env& env) {
        uint8_t a = 0; bool ok = ps.next_planned_action(agent, env, &a); return py::make_tuple(ok, (int)a); })
        .def("set_sim_order", [](PerfectSearch& ps, const std::vector<int>& order) {
        std::vector<uint8_t> o; o.reserve(order.size());
        for (int x : order) if (0 <= x && x <= 6) o.push_back((uint8_t)x); ps.set_sim_order(o); })
        .def("set_swap_heuristic", &PerfectSearch::set_swap_heuristic)
        .def("set_v5_penalty", &PerfectSearch::set_v5_penalty);

    // ---- Roles, VecArena ----
    py::enum_<RoleType>(m, "RoleType")
        .value("BotCpp", RoleType::BotCpp).value("Policy", RoleType::Policy);
    py::enum_<BotKind>(m, "BotKind")
        .value("Classic", BotKind::Classic).value("GreedyCardSpammer", BotKind::GreedyCardSpammer)
        .value("TableFirstConservativeChallenger", BotKind::TableFirstConservativeChallenger)
        .value("SelectiveTableConservativeChallenger", BotKind::SelectiveTableConservativeChallenger)
        .value("TableNonTableAgent", BotKind::TableNonTableAgent)
        .value("StrategicChallenger", BotKind::StrategicChallenger)
        .value("RandomAgent", BotKind::RandomAgent);
    py::class_<Role>(m, "Role")
        .def(py::init<>()).def_readwrite("type", &Role::type)
        .def_readwrite("bot_kind", &Role::bot_kind).def_readwrite("policy_id", &Role::policy_id);

    py::class_<VecArena>(m, "VecArena")
        .def(py::init<>())
        // --- Expose public members ---
        .def_readonly("B", &VecArena::B, "Batch size (number of environments)")
        .def_readonly("n_players", &VecArena::n_players, "Number of players per environment")
        .def_property_readonly("done", [](VecArena& self) {
        return py::array_t<uint8_t>(self.done.size(), self.done.data());
            }, "Get the done status (0 or 1) for each environment in the batch.")

        // --- Expose methods ---
        .def("reset", &VecArena::reset, py::arg("batch"), py::arg("players"), py::arg("seed"))
        .def("obs_dim", &VecArena::obs_dim)

        .def("get_env", [](VecArena& self, int i) -> Env& {
        if (i < 0 || i >= self.B) throw py::index_error("Environment index out of range");
        return self.envs.at(i);
            }, py::return_value_policy::reference_internal, "Get a reference to an environment by index.")

        .def("set_roles", [](VecArena& A, const py::list& roles_list) {
        std::vector<std::vector<Role>> R; R.resize(py::len(roles_list));
        for (size_t b = 0; b < R.size(); ++b) {
            auto row = roles_list[b].cast<py::list>(); R[b].resize(py::len(row));
            for (size_t s = 0; s < py::len(row); ++s) { R[b][s] = row[s].cast<Role>(); }
        }
        A.set_roles(R);
            }, py::arg("roles"))

        .def("collect_requests", [](VecArena& A) {
        py::dict out; auto grouped = A.collect_requests(); const int D = A.obs_dim();
        for (auto& kv : grouped) {
            int pid = kv.first; auto& reqs = kv.second; const int K = (int)reqs.size();
            if (K == 0) continue;
            py::array_t<float> obs({ K, D }); py::array_t<uint8_t> mask({ K, 7 });
            py::array_t<int> envs({ K }); py::array_t<int> seats({ K }); py::array_t<uint8_t> dones({ K });
            auto O = obs.mutable_unchecked<2>(); auto M = mask.mutable_unchecked<2>();
            auto E = envs.mutable_unchecked<1>(); auto S = seats.mutable_unchecked<1>(); auto Dn = dones.mutable_unchecked<1>();
            for (int i = 0; i < K; ++i) {
                E(i) = reqs[i].env; S(i) = reqs[i].seat; Dn(i) = reqs[i].done;
                for (int j = 0; j < D; ++j) O(i, j) = reqs[i].obs[j];
                for (int j = 0; j < 7; ++j) M(i, j) = reqs[i].mask[j];
            }
            out[py::int_(pid)] = py::make_tuple(obs, mask, envs, seats, dones);
        }
        return out;
            })
        .def("submit_actions", [](VecArena& A, int policy_id, const py::array_t<uint8_t>& actions) {
        std::vector<uint8_t> v(actions.size());
        std::copy(actions.data(), actions.data() + actions.size(), v.begin());
        A.submit_actions(policy_id, v);
            }, py::arg("policy_id"), py::arg("actions"));
}