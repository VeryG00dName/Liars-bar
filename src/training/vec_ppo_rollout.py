# src/training/vec_ppo_rollout.py

from typing import Dict, Any, List
import logging
import time
import numpy as np
import torch
from src.agents.cpp_bot_wrapper import CppBotWrapper
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
            opponent_pool = [0]
        all_env_roles = []
        for b in range(batch_size):
            env_roles = [0 for _ in range(num_players)]
            num_opponents = num_players - 1
            chosen = np.random.choice(opponent_pool, size=num_opponents).tolist()
            seats = list(range(num_players))
            np.random.shuffle(seats)
            training_seat = seats.pop()
            env_roles[training_seat] = training_policy_id
            for i in range(num_opponents):
                env_roles[seats[i]] = chosen[i]
            all_env_roles.append(env_roles)
        return all_env_roles

    def collect_episodes(self,
                         num_episodes: int,
                         num_players: int,
                         training_policy_id: int = 0,
                         opponent_pool: List[int] = None,
                         max_batch_envs: int = None) -> List[Dict[str, Any]]:
        batch_guess = self.arena.B if self.arena.B > 0 else num_episodes
        if max_batch_envs is not None:
            batch_size = int(min(batch_guess, max_batch_envs))
        else:
            batch_size = int(batch_guess)
        if batch_size <= 0: return []
        self.arena.reset(batch=batch_size, players=num_players, seed=np.random.randint(0, 2**31))

        roles = self._setup_roles(batch_size, num_players, training_policy_id, opponent_pool)
        self.arena.set_roles(roles)

        episodes = [self._new_episode_tracker(b, roles[b], training_policy_id) for b in range(batch_size)]
        completed_episodes = []

        # Keyed by (env_idx, seat) tuple for uniqueness
        pending_data: Dict[tuple, Dict[str, Any]] = {}

        iter_count = 0
        last_done_count = 0
        while len(completed_episodes) < num_episodes:
            requests_by_policy = self.arena.collect_requests()
            if not requests_by_policy:
                break

            self._log_rewards_and_dones(episodes, pending_data)

            missing = [pid for pid in requests_by_policy if pid not in self.policies]
            if missing:
                raise RuntimeError(f"Missing policy handlers for ids: {missing}")

            # Watchdog
            iter_count += 1
            done_now = sum(1 for ep in episodes if ep['done'])
            if iter_count > 0 and iter_count % 5000 == 0:
                if done_now == last_done_count:
                    logging.warning(f"[rollout] no progress for 5000 iters; still {done_now} done.")
                last_done_count = done_now

            for policy_id, reqs in requests_by_policy.items():
                agent = self.policies[policy_id]
                is_ai_agent = not isinstance(agent, CppBotWrapper)
                
                actions, log_probs, values, beliefs = agent.get_actions_batch(reqs)
                self.arena.submit_actions(policy_id, actions)

                for i, req in enumerate(reqs):
                    env_idx, seat = req.env, req.seat
                    if episodes[env_idx]['done']: continue
                    
                    pending_key = (env_idx, seat)
                    # Initialize with belief predictions, which we *always* get now
                    if is_ai_agent:
                        if beliefs and i < len(beliefs):
                            pending_data[pending_key] = {"belief_preds": beliefs[i]}
                    else: # C++ Bot
                        bot_label = agent.label
                        pending_data[pending_key] = {"belief_preds": [bot_label, 0, 0]}
                    
                    if policy_id == training_policy_id:
                        # PPO data is only stored for the learner
                        if "log_prob" not in pending_data[pending_key]: pending_data[pending_key] = {}
                        if log_probs is not None: pending_data[pending_key]["log_prob"] = log_probs[i]
                        if values is not None: pending_data[pending_key]["value"] = values[i]
                        penalties = int(self.arena.get_env(env_idx).penalties[seat])
                        pending_data[pending_key]["penalties_used"] = penalties

        self._log_rewards_and_dones(episodes, pending_data)
        for ep_tracker in episodes:
            if not ep_tracker['done']:
                self._finalize_episode(ep_tracker, pending_data)
            if ep_tracker['is_training_episode']:
                completed_episodes.append(ep_tracker['data'])

        return completed_episodes[:num_episodes]

    def _log_rewards_and_dones(self, episodes: List[Dict], pending_data: Dict[tuple, Dict[str, Any]]):
        done_statuses = self.arena.done
        for env_idx, ep_tracker in enumerate(episodes):
            if ep_tracker['done']: continue

            env = self.arena.get_env(env_idx)
            history = env.game_history()

            while ep_tracker['last_history_len'] < len(history):
                entry_idx = ep_tracker['last_history_len']
                entry = history[entry_idx]
                actor_seat = entry['player']
                ep_tracker['last_history_len'] += 1

                step_idx = self._append_step_row(ep_tracker, actor_seat)
                is_our_turn = (actor_seat == ep_tracker['training_agent_seat'])
                
                pending_key = (env_idx, actor_seat)
                data = pending_data.pop(pending_key, None)

                if is_our_turn:
                    # This was the learner's turn. Populate its specific data.
                    ep_tracker['data']['our_action'][step_idx] = entry['action']
                    if data:
                        ep_tracker['data']['log_prob'][step_idx] = data.get('log_prob')
                        ep_tracker['data']['value'][step_idx] = data.get('value')
                        ep_tracker['data']['penalties_used'][step_idx] = data.get('penalties_used')
                        
                        beliefs = data.get('belief_preds')
                        if beliefs and len(beliefs) >= 3:
                            ep_tracker['data']['belief_pred0'][step_idx] = beliefs[0]
                            ep_tracker['data']['belief_pred1'][step_idx] = beliefs[1]
                            ep_tracker['data']['belief_pred2'][step_idx] = beliefs[2]
                else:
                    # This was an opponent's turn.
                    ep_tracker['data']['opp_target_action'][step_idx] = entry['action']
                    # Log their beliefs as ground truth for the oracle.
                    if data and data.get('belief_preds'):
                        beliefs = data.get('belief_preds')
                        padded_beliefs = list(beliefs) + [-100] * (3 - len(beliefs))
                        ep_tracker['data']['opp_belief_tgt0'][step_idx] = padded_beliefs[0]
                        ep_tracker['data']['opp_belief_tgt1'][step_idx] = padded_beliefs[1]
                        ep_tracker['data']['opp_belief_tgt2'][step_idx] = padded_beliefs[2]

            if done_statuses[env_idx]:
                self._finalize_episode(ep_tracker, pending_data)

    def _finalize_episode(self, ep_tracker: Dict, pending_data: Dict[tuple, Dict[str, Any]]):
        if ep_tracker['done']: return
        ep_tracker['done'] = True

        env_idx, seat, pol_id = ep_tracker['env_idx'], ep_tracker['training_agent_seat'], ep_tracker['training_policy_id']
        env, ep_data = self.arena.get_env(env_idx), ep_tracker['data']

        pending_key = (env_idx, seat)
        data = pending_data.pop(pending_key, None)
        if data and ep_data['agent_id'] and ep_data['agent_id'][-1] == seat:
            step_idx = len(ep_data['agent_id']) - 1
            ep_data['log_prob'][step_idx] = data.get('log_prob')
            ep_data['value'][step_idx] = data.get('value')
            ep_data['penalties_used'][step_idx] = data.get('penalties_used')
            beliefs = data.get('belief_preds')
            if beliefs and len(beliefs) >= 3:
                ep_data['belief_pred0'][step_idx], ep_data['belief_pred1'][step_idx], ep_data['belief_pred2'][step_idx] = beliefs[0], beliefs[1], beliefs[2]
        
        our_last_step_idx = -1
        try:
            our_last_step_idx = len(ep_data['agent_id']) - 1 - ep_data['agent_id'][::-1].index(seat)
        except ValueError:
            pass

        is_winner = (not env.terminations[seat]) and (sum(env.terminations) == env.num_players() - 1)
        ep_data['win'] = 1 if is_winner else 0
        if our_last_step_idx != -1:
            ep_data['reward'][our_last_step_idx] = 1.0 if is_winner else -1.0
        ep_data['episode_return'] = float(sum(ep_data['reward']))

        agent = self.policies[pol_id]
        mi_last = agent.pop_last_model_input(env_idx, seat)
        if mi_last is None:
            dummy_req = lb.PolicyRequest()
            self.arena.prepare_ai_sequence(env, seat, dummy_req)
            agent.get_actions_batch([dummy_req]) # This populates the last_model_input cache
            mi_last = agent.pop_last_model_input(env_idx, seat)
            if mi_last is None: raise RuntimeError(f"Could not generate final model input for env {env_idx}")

        ep_data['model_input'] = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in mi_last.items()}

    def _new_episode_tracker(self, env_idx: int, roles: List[int], training_policy_id: int):
        training_seats = [s for s, pid in enumerate(roles) if pid == training_policy_id]
        is_training_episode = len(training_seats) > 0
        training_agent_seat = training_seats[0] if is_training_episode else -1
        opp_seats = sorted([s for s in range(len(roles)) if s != training_agent_seat])
        true_opp_labels = [getattr(self.policies.get(roles[s]), 'label', roles[s]) for s in opp_seats]

        ep_data = {
            "training_agent_seat": training_agent_seat,
            "true_opponent_labels": tuple(true_opp_labels),
            "agent_id": [], "our_action": [], "log_prob": [], "value": [], "reward": [],
            "done": [], "opp_target_action": [],
            "belief_pred0": [], "belief_pred1": [], "belief_pred2": [],
            "belief_tgt0": [], "belief_tgt1": [], "belief_tgt2": [],
            "opp_belief_tgt0": [], "opp_belief_tgt1": [], "opp_belief_tgt2": [],
            "penalties_used": [], "model_input": None,
            "episode_return": 0.0, "win": 0,
        }
        return {"env_idx": env_idx, "done": False, "is_training_episode": is_training_episode,
                "training_agent_seat": training_agent_seat, "training_policy_id": training_policy_id,
                "last_history_len": 0, "data": ep_data}

    def _append_step_row(self, ep_tracker: Dict[str, Any], agent_seat: int) -> int:
        ep = ep_tracker['data']
        ep['agent_id'].append(agent_seat)
        for key in ["our_action", "log_prob", "value", "opp_target_action", 
                    "belief_pred0", "belief_pred1", "belief_pred2",
                    "belief_tgt0", "belief_tgt1", "belief_tgt2",
                    "opp_belief_tgt0", "opp_belief_tgt1", "opp_belief_tgt2",
                    "penalties_used"]:
            ep[key].append(None)
        ep['reward'].append(0.0)
        ep['done'].append(False)

        if agent_seat == ep['training_agent_seat']:
            tl = ep["true_opponent_labels"]
            ep['belief_tgt0'][-1] = tl[0] if len(tl) > 0 else None
            ep['belief_tgt1'][-1] = tl[1] if len(tl) > 1 else None
            ep['belief_tgt2'][-1] = tl[2] if len(tl) > 2 else None

        return len(ep['agent_id']) - 1