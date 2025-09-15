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
                # This can happen if all environments are done
                break

            # Process completed steps from the previous iteration
            self._log_rewards_and_dones(episodes, pending_data)

            # Safety check
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
                
                # CppBotWrappers don't have belief predictions, handle them separately.
                is_ai_agent = not isinstance(agent, CppBotWrapper)
                
                # Get actions from the agent (this will be a mix of AI and bot policies)
                actions, log_probs, values, beliefs = agent.get_actions_batch(reqs)
                
                # Submit actions to the arena
                self.arena.submit_actions(policy_id, actions)

                # Now, log the data from this batch of requests
                for i, req in enumerate(reqs):
                    env_idx, seat = req.env, req.seat
                    
                    # Skip if the episode for this env is already marked as done
                    if episodes[env_idx]['done']:
                        continue
                        
                    pending_key = (env_idx, seat)
                    
                    # Initialize a dictionary for this pending step
                    pending_data[pending_key] = {}
                    
                    # If it was an AI agent, we have belief predictions to store
                    if is_ai_agent:
                        # For AI agents, log their actual belief prediction
                        if beliefs and i < len(beliefs):
                            pending_data[pending_key]["belief_preds"] = beliefs[i]
                    else:
                        # For C++ bots, create a "ground truth" belief vector
                        # The bot's belief is its own label, and it has no beliefs about others.
                        # We use 0 as a placeholder for the other two slots.
                        bot_label = agent.label
                        pending_data[pending_key]["belief_preds"] = [bot_label, 0, 0]
                    # If this agent is the LEARNER, we also need to store PPO-specific data
                    if policy_id == training_policy_id:
                        # Append a new row to our learner-centric data structure
                        step_idx = self._append_step_row(episodes[env_idx], seat)
                        episodes[env_idx]['data']['our_action'][step_idx] = actions[i]
                        
                        # Add the PPO data to the pending dict
                        if log_probs is not None and i < len(log_probs):
                            pending_data[pending_key]["log_prob"] = log_probs[i]
                        if values is not None and i < len(values):
                            pending_data[pending_key]["value"] = values[i]
                        
                        # Get penalties at the moment of decision
                        penalties_snapshot = int(self.arena.get_env(env_idx).penalties[seat])
                        pending_data[pending_key]["penalties_used"] = penalties_snapshot


        # Final flush of any remaining data and finalization of episodes
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
            history = env.game_history()

            while ep_tracker['last_history_len'] < len(history):
                entry_idx = ep_tracker['last_history_len']
                entry = history[entry_idx]
                actor_seat = entry['player']
                ep_tracker['last_history_len'] += 1

                # --- Process pending data for ANY actor ---
                pending_key = (env_idx, actor_seat)
                data = pending_data.pop(pending_key, None)
                
                if data:
                    # This was an AI agent's turn. Log its beliefs as ground truth.
                    beliefs = data.get('belief_preds')
                    step_idx = self._append_step_row(ep_tracker, actor_seat)
                    
                    if beliefs and len(beliefs) > 0:
                        # Pad the list to length 3 with a placeholder if it's short
                        padded_beliefs = list(beliefs) + [-100] * (3 - len(beliefs))
                        
                        ep_tracker['data']['opp_belief_tgt0'][step_idx] = padded_beliefs[0]
                        ep_tracker['data']['opp_belief_tgt1'][step_idx] = padded_beliefs[1]
                        ep_tracker['data']['opp_belief_tgt2'][step_idx] = padded_beliefs[2]
                        
                    # If this was also the learner's turn, log its specific data
                    if actor_seat == ep_tracker['training_agent_seat']:
                        ep_tracker['data']['our_action'][step_idx] = entry['action']
                        ep_tracker['data']['log_prob'][step_idx] = data.get('log_prob')
                        ep_tracker['data']['value'][step_idx] = data.get('value')
                        ep_tracker['data']['penalties_used'][step_idx] = data.get('penalties_used')
                        
                        # Also log the learner's own predictions
                        if beliefs and len(beliefs) >= 3:
                            ep_tracker['data']['belief_pred0'][step_idx] = beliefs[0]
                            ep_tracker['data']['belief_pred1'][step_idx] = beliefs[1]
                            ep_tracker['data']['belief_pred2'][step_idx] = beliefs[2]
                    else:
                        # This was an opponent's turn
                        ep_tracker['data']['opp_target_action'][step_idx] = entry['action']

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

        # If our action ended the game and pending data still exists, flush it
        seat = ep_tracker['training_agent_seat']
        pending_key = (ep_tracker['env_idx'], seat)
        data = pending_data.pop(pending_key, None)
        if data and ep_data['agent_id'] and ep_data['agent_id'][-1] == seat:
            step_idx = len(ep_data['our_action']) - 1
            ep_data['log_prob'][step_idx] = data['log_prob']
            ep_data['value'][step_idx]    = data['value']
            beliefs = data['belief_preds']
            if beliefs and len(beliefs) > 0: ep_data['belief_pred0'][step_idx] = beliefs[0]
            if beliefs and len(beliefs) > 1: ep_data['belief_pred1'][step_idx] = beliefs[1]
            if beliefs and len(beliefs) > 2: ep_data['belief_pred2'][step_idx] = beliefs[2]
            ep_data['penalties_used'][step_idx] = int(data['penalties_used'])

        # --- ROBUST TERMINAL REWARD ASSIGNMENT ---
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
        is_winner = (not env.terminations[seat]) and (sum(env.terminations) == env.num_players() - 1)
        ep_data['win'] = 1 if is_winner else 0

        # If we actually took an action, assign the final reward to our last step.
        if our_last_step_idx != -1:
            ep_data['reward'][our_last_step_idx] = 1.0 if is_winner else -1.0

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
        opp_seats = sorted([s for s in range(len(roles)) if s != training_agent_seat])
        true_opp_labels = []
        for s in opp_seats:
            pid = roles[s]
            agent = self.policies.get(pid)
            true_opp_labels.append(getattr(agent, 'label', pid))
        ep_data = {
            "training_agent_seat": training_agent_seat,
            "true_opponent_labels": tuple(true_opp_labels),

            "agent_id": [],
            "our_action": [],
            "log_prob": [],
            "value": [],
            "reward": [],
            "done": [],
            "opp_target_action": [],

            "belief_pred0": [], "belief_pred1": [], "belief_pred2": [],
            "belief_tgt0": [], "belief_tgt1": [], "belief_tgt2": [],
            
            "opp_belief_tgt0": [], "opp_belief_tgt1": [], "opp_belief_tgt2": [],
            
            "penalties_used": [],

            "model_input": None,
            "episode_return": 0.0,
            "win": 0,
        }
        return {
            "env_idx": env_idx,
            "done": False,
            "is_training_episode": is_training_episode,
            "training_agent_seat": training_agent_seat,
            "training_policy_id": training_policy_id,
            "last_history_len": 0,
            "global_step": -1,
            "data": ep_data
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

        ep['belief_pred0'].append(None)
        ep['belief_pred1'].append(None)
        ep['belief_pred2'].append(None)

        # belief targets only on our steps
        if agent_seat == ep['training_agent_seat']:
            tl = ep["true_opponent_labels"]
            ep['belief_tgt0'].append(tl[0] if len(tl) > 0 else None)
            ep['belief_tgt1'].append(tl[1] if len(tl) > 1 else None)
            ep['belief_tgt2'].append(tl[2] if len(tl) > 2 else None)
        else:
            ep['belief_tgt0'].append(None)
            ep['belief_tgt1'].append(None)
            ep['belief_tgt2'].append(None)

        ep['opp_belief_tgt0'].append(None)
        ep['opp_belief_tgt1'].append(None)
        ep['opp_belief_tgt2'].append(None)

        # align penalties_used length with rows; will be filled on our steps
        ep['penalties_used'].append(None)

        return len(ep['agent_id']) - 1