# src/training/vec_ppo_rollout.py

from typing import Dict, Any, List, Optional
import logging
import time
import numpy as np
import torch
from src.misc import lb
from src import config

from src.agents.batch_autoregressive_ppo_agent import (
    BatchPPOAutoregressiveAgent,
    PreparedPolicyBatch,
)

class PPOVecRolloutManager:
    """
    Manages batched rollouts using the C++ VecArena for high-throughput
    data collection. Designed to be compatible with multiple policies for PBT.
    """
    def __init__(self,
                 arena: lb.VecArena,
                 policies: Dict[int, BatchPPOAutoregressiveAgent],
                 device: torch.device,
                 pool_manager: Optional[Any] = None,
                 rng: Optional[np.random.Generator] = None):
        self.arena = arena
        self.policies = policies
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pool_manager = pool_manager
        self._all_shadow_pool_labels: List[int] = []
        self._current_shadow_rotation_queue: List[int] = []
        self._shadow_pool_reference: List[int] = []
        self._cpp_bots: List[int] = []
        self._latest_historical_agents: List[int] = []
        self._shadow_historical_agents: List[int] = []
        self._opponent_pool_snapshot: List[Dict[str, Any]] = []
        self._partitions_ready: bool = False

    def _update_internal_agent_partitions(self, all_opponent_pool_data: List[Dict[str, Any]]) -> None:
        """Update cached opponent partitions and refresh the rotating shadow queue."""
        cpp_bots: List[int] = []
        historical: List[int] = []

        for entry in all_opponent_pool_data:
            label = entry.get("label")
            if label is None:
                continue
            label_int = int(label)
            if label_int <= config.CPP_BOT_MAX_LABEL:
                cpp_bots.append(label_int)
            else:
                historical.append(label_int)

        cpp_bots = sorted(set(cpp_bots))
        historical_sorted = sorted(set(historical))

        latest_historical_agents = (
            historical_sorted[-config.LATEST_K:]
            if config.LATEST_K > 0
            else []
        )
        shadow_historical_agents = [
            label for label in historical_sorted if label not in latest_historical_agents
        ]

        shadow_reference = list(shadow_historical_agents)
        refresh_shadow_rotation = shadow_reference != self._shadow_pool_reference

        self._cpp_bots = list(cpp_bots)
        self._latest_historical_agents = list(latest_historical_agents)
        self._shadow_historical_agents = list(shadow_historical_agents)

        if refresh_shadow_rotation:
            self._shadow_pool_reference = shadow_reference
            self._all_shadow_pool_labels = list(shadow_reference)
            if self._all_shadow_pool_labels:
                self.rng.shuffle(self._all_shadow_pool_labels)
                self._current_shadow_rotation_queue = list(self._all_shadow_pool_labels)
            else:
                self._current_shadow_rotation_queue = []
        elif not self._current_shadow_rotation_queue and self._all_shadow_pool_labels:
            self._current_shadow_rotation_queue = list(self._all_shadow_pool_labels)

        self._partitions_ready = True

    def set_opponent_pool(self, pool_data: List[Dict[str, Any]]) -> None:
        """Store the latest opponent pool definition and refresh cached partitions."""
        self._opponent_pool_snapshot = list(pool_data)
        self._update_internal_agent_partitions(self._opponent_pool_snapshot)

    def _reset_policy_state(self):
        for policy in self.policies.values():
            try:
                policy.reset()
            except Exception:
                logging.exception("Failed to reset policy %s", getattr(policy, 'player_id', '<unknown>'))

    def _setup_roles(self,
                 batch_size: int,
                 num_players: int,
                 training_policy_id: int = 0,
                 cpp_bots: Optional[List[int]] = None,
                 latest_historical_agents: Optional[List[int]] = None,
                 active_shadow_agents_for_this_update: Optional[List[int]] = None) -> List[List[int]]:
        cpp_bots = cpp_bots or []
        latest_historical_agents = latest_historical_agents or []
        active_shadow_agents_for_this_update = active_shadow_agents_for_this_update or []

        front = sorted(set(cpp_bots).union(latest_historical_agents))
        shadow = sorted(set(active_shadow_agents_for_this_update) - set(front))

        opponent_pool = np.array(front + shadow, dtype=np.int64)
        if opponent_pool.size == 0:
            opponent_pool = np.array([training_policy_id], dtype=np.int64)

        masses = {
            "front": config.FRONT_P_ADJUSTED if front else 0.0,
            "shadow": config.SHADOW_P_NEW if shadow else 0.0,
        }

        s = masses["front"] + masses["shadow"]

        if s <= 0.0 or opponent_pool.size == 0:
            probs = np.full(opponent_pool.shape[0], 1.0 / max(1, opponent_pool.shape[0]), dtype=np.float64)
        else:
            masses["front"] /= s
            masses["shadow"] /= s

            front_set = set(front)

            def bucket(x: int) -> str:
                return "front" if x in front_set else "shadow"

            sizes = {
                "front": max(1, len(front)),
                "shadow": max(1, len(shadow)),
            }
            probs = np.array([
                masses[bucket(int(x))] / sizes[bucket(int(x))] for x in opponent_pool
            ], dtype=np.float64)
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = np.full(opponent_pool.shape[0], 1.0 / max(1, opponent_pool.shape[0]), dtype=np.float64)

        num_opponents = max(0, num_players - 1)
        if batch_size <= 0:
            return []

        seat_order = np.argsort(self.rng.random((batch_size, num_players)), axis=1)
        training_seats = seat_order[:, 0]

        if opponent_pool.size == 0:
            opponent_samples = np.full((batch_size, num_opponents), training_policy_id, dtype=np.int64)
        else:
            opponent_samples = self.rng.choice(
                opponent_pool,
                size=(batch_size, num_opponents),
                replace=True,
                p=probs,
            )

        env_roles = np.full((batch_size, num_players), training_policy_id, dtype=np.int64)
        if num_opponents > 0:
            env_roles[np.arange(batch_size)[:, None], seat_order[:, 1:]] = opponent_samples

        env_roles[np.arange(batch_size), training_seats] = np.int64(training_policy_id)

        return env_roles.astype(int).tolist()

    def collect_episodes(self,
                         num_episodes: int,
                         num_players: int,
                         training_policy_id: int = 0,
                         max_batch_envs: int = None) -> List[Dict[str, Any]]:
        batch_guess = self.arena.B if self.arena.B > 0 else num_episodes
        if max_batch_envs is not None:
            batch_size = int(min(batch_guess, max_batch_envs))
        else:
            batch_size = int(batch_guess)
        self.arena.reset(batch=batch_size, players=num_players, seed=int(self.rng.integers(0, 2**31)))
        self._reset_policy_state()

        if self.pool_manager is not None and not self._partitions_ready:
            self.set_opponent_pool(self.pool_manager.pool)

        cpp_bots: List[int] = list(self._cpp_bots)
        latest_historical_agents: List[int] = list(self._latest_historical_agents)

        active_shadow_agents_for_this_update: List[int] = []
        if self._all_shadow_pool_labels:
            for _ in range(int(config.NUM_ACTIVE_SHADOW_AGENTS_PER_UPDATE)):
                if not self._current_shadow_rotation_queue and self._all_shadow_pool_labels:
                    refreshed = list(self._all_shadow_pool_labels)
                    self.rng.shuffle(refreshed)
                    self._current_shadow_rotation_queue = refreshed
                if not self._current_shadow_rotation_queue:
                    break
                active_shadow_agents_for_this_update.append(int(self._current_shadow_rotation_queue.pop(0)))

        roles = self._setup_roles(
            batch_size,
            num_players,
            training_policy_id,
            cpp_bots=cpp_bots,
            latest_historical_agents=latest_historical_agents,
            active_shadow_agents_for_this_update=active_shadow_agents_for_this_update,
        )
        self.arena.set_roles(roles)

        episodes = [self._new_episode_tracker(b, roles[b], training_policy_id) for b in range(batch_size)]
        completed_episodes = []

        # keyed by env_idx only
        pending_data: Dict[int, Dict[str, Any]] = {}

        iter_count = 0
        last_done_count = 0
        while len(completed_episodes) < num_episodes:
            t0 = time.time() if 'time' in globals() else None
            requests_by_policy = self.arena.collect_requests()
            if t0 is not None:
                dt = time.time() - t0
                if dt > 2.0:
                    logging.warning(f"collect_requests took {dt:.2f}s (batch={batch_size})")
            if not requests_by_policy:
                break

            self._log_rewards_and_dones(episodes, pending_data)

            # Safety: ensure every returned policy id has a handler
            missing = [pid for pid in requests_by_policy.keys() if pid not in self.policies]
            if missing:
                logging.error(f"Missing policy handlers for ids: {missing}. Available: {list(self.policies.keys())}")
                raise RuntimeError(f"No policy object for ids: {missing}")

            # Progress watchdog
            iter_count += 1
            done_now = sum(1 for ep in episodes if ep['done'])
            if iter_count % 500 == 0:
                logging.info(f"[rollout] iter={iter_count} done={done_now}/{batch_size}")
            if iter_count % 5000 == 0 and done_now == last_done_count:
                logging.warning(f"[rollout] no progress for 5000 iters; still {done_now} done. Keys: {list(requests_by_policy.keys())}")
            last_done_count = done_now

            ppo_requests: List[lb.PolicyRequest] = []
            ppo_indices_by_policy: Dict[int, List[int]] = {}
            penalties_snapshot: List[int] = []
            non_ppo_requests: List[tuple[int, List[lb.PolicyRequest]]] = []

            for policy_id, reqs in requests_by_policy.items():
                if policy_id not in self.policies:
                    continue
                agent = self.policies[policy_id]
                if isinstance(agent, BatchPPOAutoregressiveAgent):
                    indices: List[int] = []
                    for req in reqs:
                        indices.append(len(ppo_requests))
                        ppo_requests.append(req)
                        env = self.arena.get_env(int(req.env))
                        penalties_snapshot.append(int(env.penalties[int(req.seat)]))
                    ppo_indices_by_policy[policy_id] = indices
                else:
                    non_ppo_requests.append((policy_id, reqs))

            prepared_batch: Optional[PreparedPolicyBatch] = None
            if ppo_requests:
                prepared_batch = BatchPPOAutoregressiveAgent.build_prepared_batch(ppo_requests)

            for policy_id, indices in ppo_indices_by_policy.items():
                agent = self.policies[policy_id]
                assert prepared_batch is not None
                actions, log_probs, values = agent.get_actions_from_prepared(prepared_batch, indices)

                self.arena.submit_actions(policy_id, actions)

                if policy_id == training_policy_id:
                    for offset, global_idx in enumerate(indices):
                        req = ppo_requests[global_idx]
                        env_idx = req.env
                        ep = episodes[env_idx]
                        if ep['done']:
                            continue
                        step_idx = self._append_step_row(ep, ep['training_agent_seat'])
                        ep['data']['our_action'][step_idx] = actions[offset]
                        pending_data[env_idx] = {
                            "log_prob": log_probs[offset],
                            "value": values[offset],
                            "penalties_used": penalties_snapshot[global_idx],
                        }

            for policy_id, reqs in non_ppo_requests:
                agent = self.policies[policy_id]

                penalties_snapshot_np = [
                    int(self.arena.get_env(int(req.env)).penalties[int(req.seat)])
                    for req in reqs
                ]

                actions, log_probs, values = agent.get_actions_batch(reqs)

                self.arena.submit_actions(policy_id, actions)

                if policy_id == training_policy_id:
                    for i, req in enumerate(reqs):
                        env_idx = req.env
                        ep = episodes[env_idx]
                        if ep['done']:
                            continue
                        step_idx = self._append_step_row(ep, ep['training_agent_seat'])
                        ep['data']['our_action'][step_idx] = actions[i]
                        pending_data[env_idx] = {
                            "log_prob": log_probs[i],
                            "value": values[i],
                            "penalties_used": penalties_snapshot_np[i],
                        }

            requests_by_policy.clear()
            del requests_by_policy

            requests_by_policy.clear()
            del requests_by_policy

        # flush last chunk
        self._log_rewards_and_dones(episodes, pending_data)

        for ep_tracker in episodes:
            if not ep_tracker['done']:
                self._finalize_episode(ep_tracker, pending_data)
            if ep_tracker['is_training_episode']:
                completed_episodes.append(ep_tracker['data'])

        return completed_episodes[:num_episodes]

    def _log_rewards_and_dones(self, episodes: List[Dict], pending_data: Dict[int, Dict[str, Any]]):
        done_statuses = self.arena.done
        for env_idx, ep_tracker in enumerate(episodes):
            if ep_tracker['done']:
                continue

            env = self.arena.get_env(env_idx)

            total_history_len = env.total_history_entries()
            start_idx = ep_tracker.get('last_processed_cxx_history_len', 0)

            if ep_tracker['last_history_len'] < start_idx:
                ep_tracker['last_history_len'] = start_idx

            history_chunk = (
                env.history_slice_basic(start_idx, total_history_len)
                if start_idx < total_history_len
                else None
            )

            if history_chunk is not None and history_chunk.size > 0:
                players = history_chunk[:, 0].astype(int, copy=False)
                actions = history_chunk[:, 1].astype(int, copy=False)

                for idx in range(players.shape[0]):
                    ep_tracker['last_history_len'] += 1
                    actor = int(players[idx])
                    action = int(actions[idx])

                    if actor == ep_tracker['training_agent_seat']:
                        data = pending_data.pop(env_idx, None)
                        if data:
                            step_idx = len(ep_tracker['data']['our_action']) - 1
                            ep_tracker['data']['log_prob'][step_idx] = data['log_prob']
                            ep_tracker['data']['value'][step_idx] = data['value']
                            ep_tracker['data']['penalties_used'][step_idx] = int(data['penalties_used'])
                    else:
                        step_idx = self._append_step_row(ep_tracker, actor)
                        ep_tracker['data']['opp_target_action'][step_idx] = action

            ep_tracker['last_processed_cxx_history_len'] = total_history_len

            self._update_penalty_rewards(ep_tracker, env.penalties)

            if done_statuses[env_idx]:
                self._finalize_episode(ep_tracker, pending_data)

    def _finalize_episode(self, ep_tracker: Dict, pending_data: Dict[int, Dict[str, Any]]):
        if ep_tracker['done']:
            return
        ep_tracker['done'] = True

        env_idx = ep_tracker['env_idx']
        seat    = ep_tracker['training_agent_seat']
        pol_id  = ep_tracker['training_policy_id']

        env     = self.arena.get_env(env_idx)
        ep_data = ep_tracker['data']

        self._update_penalty_rewards(ep_tracker, env.penalties)

        # If our action ended the game and pending data still exists, flush it
        data = pending_data.pop(env_idx, None)
        if data and ep_data['agent_id'] and ep_data['agent_id'][-1] == seat:
            step_idx = len(ep_data['our_action']) - 1
            ep_data['log_prob'][step_idx] = data['log_prob']
            ep_data['value'][step_idx]    = data['value']
            ep_data['penalties_used'][step_idx] = int(data['penalties_used'])

        # --- FIX: ROBUST TERMINAL REWARD ASSIGNMENT ---
        # Find our last step in the episode to assign the terminal reward,
        # regardless of who made the final move.
        our_last_step_idx = -1
        try:
            our_last_step_idx = (
                len(ep_data['agent_id']) - 1 - ep_data['agent_id'][::-1].index(seat)
            )
        except ValueError:
            # Our agent never acted in this episode.
            pass

        # Win/lose bookkeeping
        player_labels = ep_data.get('player_labels')
        winner_label = None
        if player_labels:
            active_players = [idx for idx, terminated in enumerate(env.terminations) if not terminated]
            if active_players:
                winner_label = player_labels[active_players[0]]
        ep_data['winner_label'] = winner_label

        is_winner = (not env.terminations[seat]) and (sum(env.terminations) == env.num_players() - 1)
        ep_data['win'] = 1 if is_winner else 0

        # If we actually took an action, assign the final reward to our last step.
        if our_last_step_idx != -1:
            ep_data['reward'][our_last_step_idx] += 1.0 if is_winner else -1.0
        # --- END FIX ---

        ep_data['episode_return'] = float(sum(ep_data['reward']))

        # Persist the exact model_input used for the final forward on this env/seat
        agent  = self.policies[pol_id]
        mi_last = agent.pop_last_model_input(env_idx, seat)
        if mi_last is None:
            raise RuntimeError(f"Missing final model input for env {env_idx}, seat {seat}")
        ep_data['model_input'] = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in mi_last.items()
        }

    def _new_episode_tracker(self, env_idx: int, roles: List[int], training_policy_id: int) -> Dict[str, Any]:
        training_seats = [s for s, pid in enumerate(roles) if pid == training_policy_id]
        is_training_episode = len(training_seats) > 0
        training_agent_seat = training_seats[0] if is_training_episode else -1

        n_players = len(roles)
        if is_training_episode:
            opp_seats = [(training_agent_seat + r) % n_players for r in range(1, n_players)]
        else:
            opp_seats = list(range(n_players))

        player_labels = []
        for seat_idx, pid in enumerate(roles):
            agent = self.policies.get(pid)
            player_labels.append(getattr(agent, 'label', pid))

        training_agent_label = player_labels[training_agent_seat] if training_agent_seat != -1 else None
        true_opp_labels = tuple(player_labels[s] for s in opp_seats if s != training_agent_seat)
        ep_data = {
            "training_agent_seat": training_agent_seat,
            "training_agent_label": training_agent_label,
            "player_labels": tuple(player_labels),
            "true_opponent_labels": true_opp_labels,
            "agent_id": [],
            "our_action": [],
            "log_prob": [],
            "value": [],
            "reward": [],
            "done": [],
            "opp_target_action": [],
            "penalties_used": [],
            "model_input": None,
            "episode_return": 0.0,
            "win": 0,
            "winner_label": None,
        }
        return {
            "env_idx": env_idx,
            "done": False,
            "is_training_episode": is_training_episode,
            "training_agent_seat": training_agent_seat,
            "training_policy_id": training_policy_id,
            "last_history_len": 0,
            "last_processed_cxx_history_len": 0,
            "global_step": -1,
            "last_training_step_idx": -1,
            "last_penalties": None,
            "data": ep_data,
        }

    def _append_step_row(self, ep_tracker: Dict[str, Any], agent_seat: int) -> int:
        ep = ep_tracker['data']
        ep['agent_id'].append(agent_seat)
        ep['our_action'].append(None)
        ep['log_prob'].append(None)
        ep['value'].append(None)
        ep['reward'].append(0.0)
        ep['done'].append(False)
        ep['opp_target_action'].append(None)
        ep['penalties_used'].append(None)
        idx = len(ep['agent_id']) - 1
        if agent_seat == ep_tracker.get('training_agent_seat'):
            ep_tracker['last_training_step_idx'] = idx
        return idx

    def _update_penalty_rewards(self, ep_tracker: Dict[str, Any], penalties: Any) -> None:
        """Update cached penalty state and adjust the latest training reward."""
        if not ep_tracker.get('is_training_episode', False):
            ep_tracker['last_penalties'] = [int(p) for p in penalties]
            return

        penalties_list = [int(p) for p in penalties]
        last_penalties = ep_tracker.get('last_penalties')
        ep_tracker['last_penalties'] = penalties_list

        if last_penalties is None:
            return

        seat = ep_tracker.get('training_agent_seat', -1)
        last_idx = ep_tracker.get('last_training_step_idx', -1)
        ep_data = ep_tracker.get('data') or {}

        if last_idx < 0 or last_idx >= len(ep_data.get('reward', [])):
            return

        delta_total = 0.0
        for i, (prev, cur) in enumerate(zip(last_penalties, penalties_list)):
            diff = int(cur) - int(prev)
            if diff <= 0:
                continue
            if i == seat:
                delta_total -= 0.1 * diff
            else:
                delta_total += 0.033 * diff

        if delta_total != 0.0:
            ep_data['reward'][last_idx] += float(delta_total)
