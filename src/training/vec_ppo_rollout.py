# src/training/vec_ppo_rollout.py

from typing import Dict, Any, List
import logging
import time
import numpy as np
import torch
from src.misc import lb

from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent

class PPOVecRolloutManager:
    """
    Manages batched rollouts using the C++ VecArena for high-throughput
    data collection. Designed to be compatible with multiple policies for PBT.
    """
    def __init__(self,
                 arena: lb.VecArena,
                 policies: Dict[int, BatchPPOAutoregressiveAgent],
                 device: torch.device):
        self.arena = arena
        self.policies = policies
        self.device = device

    def _setup_roles(self,
                 batch_size: int,
                 num_players: int,
                 training_policy_id: int = 0,
                 opponent_pool: List[int] = None) -> List[List[int]]:
        if opponent_pool is None:
            opponent_pool = []

        BOT_MAX_ID = 6
        LATEST_K   = 4

        FRONT_P   = 0.95  # bots ∪ latest
        SHADOW_P  = 0.05  # shadow

        # Build the sampling pool (preserving any weighting duplicates) and always
        # include the current learner.
        pool: List[int] = [int(pid) for pid in opponent_pool]
        if training_policy_id not in pool:
            pool.append(training_policy_id)

        if not pool:
            pool = [training_policy_id]

        # Partition pool (excluding learner for historical buckets)
        pool_without_learner = [pid for pid in pool if pid != training_policy_id]

        bots    = [i for i in pool_without_learner if i <= BOT_MAX_ID]
        frozens = sorted([i for i in pool_without_learner if i > BOT_MAX_ID])
        latest  = frozens[-LATEST_K:] if LATEST_K > 0 else []
        shadow  = [i for i in frozens if i not in latest]

        # Build the front bucket, always placing the learner first without
        # shrinking the historical "latest" window.
        front: List[int] = []

        def _add_front(pid: int):
            if pid not in front:
                front.append(pid)

        _add_front(training_policy_id)
        for pid in bots + latest:
            if pid != training_policy_id:
                _add_front(pid)

        front_set = set(front)
        shadow_set = set(shadow)

        # Bucket masses (renormalize if some buckets are empty)
        masses = {
            "front":  FRONT_P  if front  else 0.0,
            "shadow": SHADOW_P if shadow else 0.0,
        }
        s = masses["front"] + masses["shadow"]

        if s <= 0.0:
            # Fallback: uniform over entire pool
            probs = np.full(len(pool), 1.0 / max(1, len(pool)))
        else:
            # Renormalize to 1.0 if a bucket was empty
            masses["front"]  /= s
            masses["shadow"] /= s

            def bucket(x: int) -> str:
                return "front" if x in front_set else "shadow"

            front_count = sum(1 for x in pool if x in front_set)
            shadow_count = sum(1 for x in pool if x in shadow_set)
            sizes = {"front": max(1, front_count), "shadow": max(1, shadow_count)}
            probs = np.array([masses[bucket(x)] / sizes[bucket(x)] for x in pool], dtype=np.float64)
            probs /= probs.sum()

        all_env_roles = []
        for _ in range(batch_size):
            env_roles = [0 for _ in range(num_players)]
            seats = list(range(num_players))
            np.random.shuffle(seats)
            # Guarantee at least one learner seat per game.
            primary_seat = seats.pop()
            env_roles[primary_seat] = training_policy_id

            num_remaining = len(seats)
            if num_remaining > 0:
                chosen = np.random.choice(pool, size=num_remaining, replace=True, p=probs).tolist()
                for idx, seat in enumerate(seats):
                    env_roles[seat] = int(chosen[idx])
            all_env_roles.append(env_roles)

        return all_env_roles

    def collect_episodes(self,
                     num_episodes: int,
                     num_players: int,
                     training_policy_id: int = 0,
                     opponent_pool: List[int] = None,
                     max_batch_envs: int = None) -> List[Dict[str, Any]]:
        batch_guess = self.arena.B if self.arena.B > 0 else num_episodes
        batch_size = int(min(batch_guess, max_batch_envs)) if max_batch_envs is not None else int(batch_guess)
        self.arena.reset(batch=batch_size, players=num_players, seed=np.random.randint(0, 2**31))

        roles = self._setup_roles(batch_size, num_players, training_policy_id, opponent_pool)
        self.arena.set_roles(roles)

        episodes = [self._new_episode_tracker(b, roles[b], training_policy_id) for b in range(batch_size)]
        completed_episodes = []

        # pending_data[env_idx][seat] -> {"step_idx", "log_prob", "value", "penalties_used"}
        pending_data: Dict[int, Dict[int, Dict[str, Any]]] = {}

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

            for policy_id, reqs in requests_by_policy.items():
                if policy_id not in self.policies:
                    continue
                agent = self.policies[policy_id]

                env_indices = np.array([r.env for r in reqs], dtype=int)
                seat_indices = np.array([r.seat for r in reqs], dtype=int)

                penalties_snapshot: List[int] = []
                for env_idx, seat in zip(env_indices, seat_indices):
                    env = self.arena.get_env(int(env_idx))
                    penalties_snapshot.append(int(env.penalties[int(seat)]))

                actions, log_probs, values = agent.get_actions_batch(reqs)

                self.arena.submit_actions(policy_id, actions)

                # Only log our action rows when the training policy acted
                if policy_id == training_policy_id:
                    for i, req in enumerate(reqs):
                        env_idx = req.env
                        ep = episodes[env_idx]
                        if ep['done']:
                            continue
                        seat = req.seat
                        step_idx = self._append_step_row(ep, seat)  # append row now; remember its index
                        seat_data = ep['seat_data'].get(seat)
                        if seat_data is None:
                            continue
                        seat_data['our_action'][step_idx] = actions[i]

                        env_pending = pending_data.setdefault(env_idx, {})
                        env_pending[seat] = {
                            "step_idx": step_idx,                     # store exact row index
                            "log_prob": log_probs[i],
                            "value": values[i],
                            "penalties_used": penalties_snapshot[i],
                        }

        # flush last chunk
        self._log_rewards_and_dones(episodes, pending_data)

        for ep_tracker in episodes:
            if not ep_tracker['done']:
                self._finalize_episode(ep_tracker, pending_data)
            for seat in ep_tracker['training_agent_seats']:
                data = ep_tracker['seat_data'].get(seat)
                if data is not None:
                    completed_episodes.append(data)

        return completed_episodes[:num_episodes]

    def _log_rewards_and_dones(self,
                           episodes: List[Dict],
                           pending_data: Dict[int, Dict[int, Dict[str, Any]]]):
        done_statuses = self.arena.done
        for env_idx, ep_tracker in enumerate(episodes):
            if ep_tracker['done']:
                continue

            env = self.arena.get_env(env_idx)
            history = env.game_history()

            # consume new history entries
            while ep_tracker['last_history_len'] < len(history):
                entry_idx = ep_tracker['last_history_len']
                entry = history[entry_idx]
                ep_tracker['last_history_len'] += 1

                seat_played = int(entry['player'])
                is_training_turn = seat_played in ep_tracker['seat_data']

                if is_training_turn:
                    # Fill from pending using the exact step_idx we stored at action time.
                    env_pending = pending_data.get(env_idx, {})
                    data = env_pending.pop(seat_played, None)
                    if not env_pending:
                        pending_data.pop(env_idx, None)
                    if data:
                        step_idx = int(data.get("step_idx", -1))
                        seat_data = ep_tracker['seat_data'].get(seat_played)
                        if seat_data is not None and 0 <= step_idx < len(seat_data['our_action']):
                            seat_data['log_prob'][step_idx] = data['log_prob']
                            seat_data['value'][step_idx]    = data['value']
                            seat_data['penalties_used'][step_idx] = int(data['penalties_used'])
                            # Everyone else sees this as an opponent action at the same row.
                            if 'action' in entry:
                                ep_tracker['common']['opp_target_action'][step_idx] = entry['action']
                else:
                    # Opponent (non-learner) turn: create a new row and write that action into the shared stream.
                    step_idx = self._append_step_row(ep_tracker, seat_played)
                    if 'action' in entry:
                        ep_tracker['common']['opp_target_action'][step_idx] = entry['action']

            if done_statuses[env_idx]:
                self._finalize_episode(ep_tracker, pending_data)

    def _finalize_episode(self,
                      ep_tracker: Dict,
                      pending_data: Dict[int, Dict[int, Dict[str, Any]]]):
        if ep_tracker['done']:
            return
        ep_tracker['done'] = True

        env_idx = ep_tracker['env_idx']
        pol_id  = ep_tracker['training_policy_id']

        env = self.arena.get_env(env_idx)
        env_pending = pending_data.pop(env_idx, {})

        for seat, seat_data in ep_tracker['seat_data'].items():
            # Flush any leftover pending row to the correct step index
            data = env_pending.pop(seat, None)
            if data:
                step_idx = int(data.get("step_idx", len(seat_data['our_action']) - 1))
                if 0 <= step_idx < len(seat_data['our_action']):
                    seat_data['log_prob'][step_idx] = data['log_prob']
                    seat_data['value'][step_idx]    = data['value']
                    seat_data['penalties_used'][step_idx] = int(data['penalties_used'])

            # --- Terminal reward per learner seat ---
            our_last_step_idx = -1
            try:
                our_last_step_idx = (
                    len(seat_data['agent_id']) - 1 - seat_data['agent_id'][::-1].index(seat)
                )
            except ValueError:
                pass  # learner never acted

            # Winner/loser bookkeeping (independent per seat)
            player_labels = seat_data.get('player_labels')
            winner_label = None
            if player_labels:
                active_players = [idx for idx, terminated in enumerate(env.terminations) if not terminated]
                if active_players:
                    winner_label = player_labels[active_players[0]]
            seat_data['winner_label'] = winner_label

            is_winner = (not env.terminations[seat]) and (sum(env.terminations) == env.num_players() - 1)
            seat_data['win'] = 1 if is_winner else 0

            if our_last_step_idx != -1:
                seat_data['reward'][our_last_step_idx] = 1.0 if is_winner else -1.0

            seat_data['episode_return'] = float(sum(seat_data['reward']))

            # Persist the exact model_input used for the final forward on this env/seat
            agent  = self.policies[pol_id]
            mi_last = agent.pop_last_model_input(env_idx, seat)
            if mi_last is None:
                raise RuntimeError(f"Missing final model input for env {env_idx}, seat {seat}")
            seat_data['model_input'] = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in mi_last.items()
            }

    def _new_episode_tracker(self, env_idx: int, roles: List[int], training_policy_id: int) -> Dict[str, Any]:
        training_seats = [s for s, pid in enumerate(roles) if pid == training_policy_id]
        is_training_episode = len(training_seats) > 0

        n_players = len(roles)
        player_labels = []
        for seat_idx, pid in enumerate(roles):
            agent = self.policies.get(pid)
            player_labels.append(getattr(agent, 'label', pid))

        common = {
            "agent_id": [],
            "opp_target_action": [],
            "done": [],
        }

        seat_data: Dict[int, Dict[str, Any]] = {}
        if is_training_episode:
            for seat in training_seats:
                opp_seats = [ (seat + r) % n_players for r in range(1, n_players) ]
                true_opp_labels = tuple(player_labels[s] for s in opp_seats if s != seat)
                seat_data[seat] = {
                    "training_agent_seat": seat,
                    "training_agent_label": player_labels[seat],
                    "player_labels": tuple(player_labels),
                    "true_opponent_labels": true_opp_labels,

                    "agent_id": common["agent_id"],
                    "our_action": [],
                    "log_prob": [],
                    "value": [],
                    "reward": [],
                    "done": common["done"],
                    "opp_target_action": common["opp_target_action"],

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
            "training_agent_seats": training_seats,
            "training_policy_id": training_policy_id,
            "last_history_len": 0,
            "global_step": -1,
            "seat_data": seat_data,
            "common": common,
            "player_labels": tuple(player_labels),
        }

    def _append_step_row(self, ep_tracker: Dict[str, Any], agent_seat: int) -> int:
        common = ep_tracker['common']
        common['agent_id'].append(agent_seat)
        common['opp_target_action'].append(None)
        common['done'].append(False)

        step_idx = len(common['agent_id']) - 1

        # Grow all per-seat arrays to the same length
        for seat_data in ep_tracker['seat_data'].values():
            seat_data['our_action'].append(None)
            seat_data['log_prob'].append(None)
            seat_data['value'].append(None)
            seat_data['reward'].append(0.0)
            seat_data['penalties_used'].append(None)

        return step_idx
