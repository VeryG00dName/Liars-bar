#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "eval_manager.h"

namespace py = pybind11;

void bind_eval_manager(py::module_& m) {
    py::class_<EvalAgentStats>(m, "EvalAgentStats")
        .def_readonly("total_wins", &EvalAgentStats::total_wins)
        .def_readonly("total_returns", &EvalAgentStats::total_returns)
        .def_readonly("num_games", &EvalAgentStats::num_games);

    py::class_<EvalLineupResult>(m, "EvalLineupResult")
        .def_readonly("per_policy", &EvalLineupResult::per_policy);

    py::class_<EvalManager>(m, "EvalManager")
        .def(py::init<>())
        .def("set_max_env_batch", &EvalManager::set_max_env_batch, py::arg("max_batch"))
        .def("set_inference_batch_size", &EvalManager::set_inference_batch_size, py::arg("batch_size"))
        .def("load_model", &EvalManager::load_model, py::arg("policy_id"), py::arg("path"))
        .def("finalize_model_loading", &EvalManager::finalize_model_loading)
        .def("register_cpp_bot", &EvalManager::register_cpp_bot,
             py::arg("policy_id"), py::arg("name"))
        .def("get_last_performance_stats", &EvalManager::get_last_performance_stats)
        .def("run_roles",
             [](EvalManager& self,
                const std::vector<std::vector<int>>& roles,
                const std::vector<int>& lineup_indices,
                int num_players,
                uint32_t seed) {
                 auto outcome = self.run_roles(roles, lineup_indices, num_players, seed);

                 py::list py_lineups;
                 for (const auto& lineup : outcome.lineups) {
                     py::dict stats_dict;
                     for (const auto& kv : lineup.per_policy) {
                         py::dict entry;
                         entry["total_wins"] = kv.second.total_wins;
                         entry["total_returns"] = kv.second.total_returns;
                         entry["num_games"] = kv.second.num_games;
                         entry["expert_data"] = py::dict();
                         stats_dict[py::int_(kv.first)] = std::move(entry);
                     }
                     py_lineups.append(std::move(stats_dict));
                 }

                 py::dict counts;
                 for (const auto& kv : outcome.h2h_counts) {
                     counts[py::make_tuple(kv.first[0], kv.first[1])] = kv.second;
                 }

                 py::dict wins;
                 for (const auto& kv : outcome.h2h_wins) {
                     wins[py::make_tuple(kv.first[0], kv.first[1])] = kv.second;
                 }

                py::dict perf_stats;
                for (const auto& kv : self.get_last_performance_stats()) {
                    perf_stats[py::str(kv.first)] = kv.second;
                }

                return py::make_tuple(
                    std::move(py_lineups),
                    std::move(counts),
                    std::move(wins),
                    outcome.total_games,
                    std::move(perf_stats));
             },
             py::arg("roles"),
             py::arg("lineup_indices"),
             py::arg("num_players"),
             py::arg("seed"));
}

