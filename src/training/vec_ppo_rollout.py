# src/training/vec_ppo_rollout.py
import torch
import numpy as np
from src.misc import lb
from typing import Dict, Any, List, Tuple

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
        self._target_episodes = 0
        self._completed_episodes = 0

    def _setup_roles(self,
                     batch_size: int,
                     num_players: int,
                     training_policy_id: int = 0,
                     opponent_pool: List[lb.BotKind] = None) -> List[List[lb.Role]]:
        if opponent_pool is None:
            opponent_pool = [lb.BotKind.Classic]
        all_env_roles = []
        for b in range(batch_size):
            env_roles = [lb.Role() for _ in range(num_players)]
            num_opponents = num_players - 1
            chosen_opponents = np.random.choice(opponent_pool, size=num_opponents).tolist()
            seats = list(range(num_players))
            np.random.shuffle(seats)
            training_seat = seats.pop()
            env_roles[training_seat].type = lb.RoleType.Policy
            env_roles[training_seat].policy_id = training_policy_id
            for i in range(num_opponents):
                opp_seat = seats[i]
                env_roles[opp_seat].type = lb.RoleType.BotCpp
                env_roles[opp_seat].bot_kind = chosen_opponents[i]
            all_env_roles.append(env_roles)
        return all_env_roles

    def collect_episodes(self,
                         num_episodes: int,
                         num_players: int,
                         training_policy_id: int = 0,
                         opponent_pool: List[lb.BotKind] = None) -> List[Dict[str, Any]]:
        batch_size = self.arena.B if self.arena.B > 0 else num_episodes
        self.arena.reset(batch=batch_size, players=num_players, seed=np.random.randint(0, 2**31))
        
        roles = self._setup_roles(batch_size, num_players, training_policy_id, opponent_pool)
        self.arena.set_roles(roles)
        episodes = [self._new_episode_tracker(b, roles[b], training_policy_id) for b in range(batch_size)]
        completed_episodes = []
        # --- [FIX] pending_data is now keyed only by env_idx ---
        pending_data: Dict[int, Dict] = {}
        self._target_episodes = num_episodes
        self._completed_episodes = 0
        while len(completed_episodes) < num_episodes:
            requests_by_policy = self.arena.collect_requests()
            if not requests_by_policy: break
            self._log_rewards_and_dones(episodes, pending_data)
            for policy_id, (obs, mask, env_indices, seat_indices, dones) in requests_by_policy.items():
                if policy_id not in self.policies: continue
                agent = self.policies[policy_id]
                actions, log_probs, values, beliefs = agent.get_actions_batch(
                    self.arena, env_indices, seat_indices, obs, mask
                )
                self.arena.submit_actions(policy_id, actions)
                for i in range(len(env_indices)):
                    env_idx = env_indices[i]
                    ep = episodes[env_idx]
                    if ep['done']: continue
                    step_idx = self._append_step_row(ep, ep['training_agent_seat'])
                    ep['data']['our_action'][step_idx] = actions[i]
                    # --- [FIX] Use the simpler env_idx key ---
                    pending_data[env_idx] = {
                        "log_prob": log_probs[i], "value": values[i], "belief_preds": beliefs[i]
                    }

        self._log_rewards_and_dones(episodes, pending_data)
        for ep_tracker in episodes:
            if not ep_tracker['done']:
                self._finalize_episode(ep_tracker, pending_data)
            if ep_tracker['is_training_episode']:
                completed_episodes.append(ep_tracker['data'])
        
        return completed_episodes[:num_episodes]

    def _log_rewards_and_dones(self, episodes: List[Dict], pending_data: Dict):
        done_statuses = self.arena.done
        for env_idx, ep_tracker in enumerate(episodes):
            if ep_tracker['done']: continue
            env = self.arena.get_env(env_idx)
            history = env.game_history()
            
            while ep_tracker['last_history_len'] < len(history):
                entry_idx = ep_tracker['last_history_len']
                entry = history[entry_idx]
                ep_tracker['last_history_len'] += 1
                is_our_turn = (entry['player'] == ep_tracker['training_agent_seat'])
                
                if is_our_turn:
                    # --- [FIX] Use the simpler pop(env_idx) ---
                    data = pending_data.pop(env_idx, None)
                    if data:
                        step_idx = len(ep_tracker['data']['our_action']) - 1
                        ep_tracker['data']['log_prob'][step_idx] = data['log_prob']
                        ep_tracker['data']['value'][step_idx] = data['value']
                        beliefs = data['belief_preds']
                        if beliefs and len(beliefs) > 0: ep_tracker['data']['belief_pred0'][step_idx] = beliefs[0]
                        if beliefs and len(beliefs) > 1: ep_tracker['data']['belief_pred1'][step_idx] = beliefs[1]
                        if beliefs and len(beliefs) > 2: ep_tracker['data']['belief_pred2'][step_idx] = beliefs[2]
                        if done_statuses[env_idx]:
                            if env.terminations[ep_tracker['training_agent_seat']]:
                                ep_tracker['data']['reward'][step_idx] = -1.0
                else:
                    step_idx = self._append_step_row(ep_tracker, entry['player'])
                    ep_tracker['data']['opp_target_action'][step_idx] = entry['action']
            
            if done_statuses[env_idx]:
                self._finalize_episode(ep_tracker, pending_data)

    def _finalize_episode(self, ep_tracker: Dict, pending_data: Dict):
        if ep_tracker['done']:
            return
        ep_tracker['done'] = True

        env_idx = ep_tracker['env_idx']
        seat    = ep_tracker['training_agent_seat']
        pol_id  = ep_tracker['training_policy_id']

        env     = self.arena.get_env(env_idx)
        ep_data = ep_tracker['data']

        # --- Flush any lingering per-step data for the final action (if our turn ended the episode)
        data = pending_data.pop(env_idx, None)
        if data and ep_data['agent_id'] and ep_data['agent_id'][-1] == seat:
            step_idx = len(ep_data['our_action']) - 1
            ep_data['log_prob'][step_idx] = data['log_prob']
            ep_data['value'][step_idx]    = data['value']
            beliefs = data['belief_preds']
            if beliefs and len(beliefs) > 0: ep_data['belief_pred0'][step_idx] = beliefs[0]
            if beliefs and len(beliefs) > 1: ep_data['belief_pred1'][step_idx] = beliefs[1]
            if beliefs and len(beliefs) > 2: ep_data['belief_pred2'][step_idx] = beliefs[2]

        # --- Win/lose bookkeeping
        is_winner = (not env.terminations[seat]) and (sum(env.terminations) == env.num_players() - 1)
        ep_data['win'] = 1 if is_winner else 0

        if ep_data['agent_id'] and ep_data['agent_id'][-1] == seat:
            ep_data['reward'][-1] = 1.0 if is_winner else -1.0

        ep_data['episode_return'] = sum(ep_data['reward'])

        # --- Use the exact model_input the model last saw for this env/seat
        agent  = self.policies[pol_id]
        mi_last = agent.pop_last_model_input(env_idx, seat)
        if mi_last is None:
            raise RuntimeError(f"Missing final model input for env {env_idx}, seat {seat}")
        # Store on CPU to keep episodes light
        ep_data['model_input'] = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in mi_last.items()
        }
        
        # --- NEW: print which episode finished and running count ---
        self._completed_episodes += 1
        steps = len(ep_data['reward'])
        try:
            hist_sz = env.game_history_size()  # cheap if you add the binding
        except Exception:
            hist_sz = len(env.game_history())  # fallback (expensive)
        print(f"[ROLLOUT] finished env {env_idx}  "
            f"({self._completed_episodes}/{self._target_episodes})  "
            f"win={ep_data['win']}  steps={steps}")

    def _new_episode_tracker(self, env_idx: int, roles: List[lb.Role], training_policy_id: int) -> Dict:
        training_seats = [s for s, r in enumerate(roles) if r.policy_id == training_policy_id]
        is_training_episode = len(training_seats) > 0
        training_agent_seat = training_seats[0] if is_training_episode else -1
        opp_seats = sorted([s for s, r in enumerate(roles) if s != training_agent_seat])
        true_opp_labels = [roles[s].bot_kind.value for s in opp_seats]
        ep_data = {
            "training_agent_seat": training_agent_seat, "true_opponent_labels": tuple(true_opp_labels),
            "agent_id": [], "our_action": [], "log_prob": [], "value": [], "reward": [], "done": [],
            "opp_target_action": [], "belief_pred0": [], "belief_pred1": [], "belief_pred2": [],
            "belief_tgt0": [], "belief_tgt1": [], "belief_tgt2": [],
            "model_input": None, "episode_return": 0.0, "win": 0,
        }
        return {
            "env_idx": env_idx, "done": False, "is_training_episode": is_training_episode,
            "training_agent_seat": training_agent_seat, "training_policy_id": training_policy_id,
            "last_history_len": 0, "global_step": -1, "data": ep_data
        }

    def _append_step_row(self, ep_tracker: Dict, agent_seat: int) -> int:
        ep = ep_tracker['data']
        ep['agent_id'].append(agent_seat)
        ep['our_action'].append(None); ep['log_prob'].append(None); ep['value'].append(None)
        ep['reward'].append(0.0); ep['done'].append(False); ep['opp_target_action'].append(None)
        ep['belief_pred0'].append(None); ep['belief_pred1'].append(None); ep['belief_pred2'].append(None)
        if agent_seat == ep['training_agent_seat']:
            tl = ep["true_opponent_labels"]
            ep['belief_tgt0'].append(tl[0] if len(tl) > 0 else None)
            ep['belief_tgt1'].append(tl[1] if len(tl) > 1 else None)
            ep['belief_tgt2'].append(tl[2] if len(tl) > 2 else None)
        else:
            ep['belief_tgt0'].append(None); ep['belief_tgt1'].append(None); ep['belief_tgt2'].append(None)
        return len(ep['agent_id']) - 1