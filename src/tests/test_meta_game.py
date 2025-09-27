import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

misc_module = types.ModuleType("src.misc")
lb_stub = mock.MagicMock()
misc_module.lb = lb_stub
sys.modules.setdefault("src.misc", misc_module)
sys.modules.setdefault("src.misc.lb", lb_stub)

gym_stub = types.ModuleType("gymnasium")
gym_stub.spaces = types.SimpleNamespace()
sys.modules.setdefault("gymnasium", gym_stub)

torch_stub = mock.MagicMock()
torch_stub.is_tensor = lambda _obj: False
torch_stub.cuda = mock.MagicMock()
torch_stub.cuda.is_available.return_value = False
torch_stub.device = lambda *args, **kwargs: mock.MagicMock()
torch_stub.float32 = object()
torch_stub.uint8 = object()
torch_stub.float16 = object()
torch_stub.int64 = object()
torch_stub.bool = object()
torch_stub.Tensor = mock.MagicMock
torch_stub.utils = mock.MagicMock()
sys.modules.setdefault("torch", torch_stub)

from src.training.meta_game_sampler import MetaGameSampler, SamplerConfig
from src.training.meta_game_solvers import MetaGameSolver, SolverConfig
from src.training.training_meta_game import MetaGameStore
from src.training.vec_ppo_rollout import PPOVecRolloutManager


def test_meta_game_store_records_and_interval(tmp_path):
    store = MetaGameStore()
    store.record_match(1, 2, seat_permutation=(1, 2, 3, 4))
    store.record_match(1, 2, seat_permutation=(1, 3, 2, 4))
    store.record_match(2, 1, seat_permutation=(2, 1, 3, 4))

    stats = store.get_ordered_stats(1, 2)
    assert stats.wins == 2
    assert stats.losses == 1
    assert len(stats.seat_permutations) == 3

    lower, upper = stats.wilson_interval()
    assert 0.0 <= lower <= upper <= 1.0

    store.save_incremental(tmp_path)
    loaded = MetaGameStore.load_from_directory(tmp_path)
    loaded_stats = loaded.get_ordered_stats(1, 2)
    assert loaded_stats.wins == 2
    assert len(loaded_stats.seat_permutations) == 3


def test_meta_game_solver_fallbacks():
    store = MetaGameStore()
    for _ in range(10):
        store.record_match(1, 2)
    solver = MetaGameSolver(
        store,
        SolverConfig(
            solver_type="alpha_rank",
            heldout_floor=0.2,
            exploration_epsilon=0.0,
        ),
    )
    distribution = solver.solve(candidates=[1, 2], held_out=[2])
    assert distribution[2] >= 0.2
    assert math.isclose(sum(distribution.values()), 1.0, rel_tol=1e-6)


def test_meta_game_sampler_archival(tmp_path):
    store = MetaGameStore()
    for _ in range(5):
        store.record_match(1, 2)
    solver_cfg = SolverConfig(solver_type="alpha_rank", exploration_epsilon=0.0)
    sampler_cfg = SamplerConfig(
        solver=solver_cfg,
        store_path=tmp_path,
        archive_threshold=0.4,
        archive_patience=1,
    )
    sampler = MetaGameSampler(sampler_cfg)
    sampler._last_distribution = {1: 0.8, 2: 0.1}
    sampler.refresh_store()
    metadata = sampler.store.metadata()
    assert 2 in metadata.get("archived", [])


def test_rollout_manager_uses_meta_sampler(monkeypatch):
    store = MetaGameStore()
    solver_cfg = SolverConfig(solver_type="alpha_rank", exploration_epsilon=0.0)
    sampler_cfg = SamplerConfig(solver=solver_cfg)
    sampler = MetaGameSampler(sampler_cfg)
    sampler._last_distribution = {1: 0.7, 2: 0.3}

    class DummyRollout:
        def __init__(self):
            self.opponent_labels = None
            self.opponent_weights = None
            self.newest_label = None

        def set_training_device(self, device):
            self.device = device

        def start_rollouts(
            self,
            *_args,
        ):
            if len(_args) >= 4:
                self.opponent_labels = _args[-3]
                self.opponent_weights = _args[-2]
                self.newest_label = _args[-1]

        def collect_requests_for_inference(self):
            return {}

        def get_completed_episodes(self):
            return []

    dummy = DummyRollout()
    monkeypatch.setattr("src.training.vec_ppo_rollout.lb.RolloutManager", lambda: dummy)

    policy = SimpleNamespace(reset=lambda: None)
    mgr = PPOVecRolloutManager({0: policy}, mock.MagicMock(), meta_sampler=sampler)
    mgr._cpp_bots = [1]
    mgr._newest_historical_agent = 2
    mgr._other_historical_agents = []
    mgr.collect_episodes(1, 4, training_policy_id=0)
    assert dummy.opponent_labels == [1, 2]
    assert dummy.newest_label == 2
    assert dummy.opponent_weights == pytest.approx([0.7, 0.3])

