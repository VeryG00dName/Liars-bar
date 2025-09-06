# src/agents/batch_autoregressive_ppo_agent.py
import torch
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple

from src.agents.base_agent import BaseAgent
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.model_factory import ModelFactory as MFactoryUtil
from src.misc import lb  # Correct import path
import torch.amp as amp
logger = logging.getLogger(__name__)

class BatchPPOAutoregressiveAgent(BaseAgent):
    """
    An agent that uses the PPOAutoregressiveModel and is specifically designed to
    work with the batched C++ `VecArena` for efficient data collection.

    It processes batches of requests by padding sequences to the same length,
    allowing for a single model forward pass.
    """
    def __init__(self, device: torch.device, player_id: str):
        super().__init__(device, player_id)
        self.model: Optional[PPOAutoregressiveModel] = None

        # Model dimensions (inferred during loading)
        self.obs_dim: int = 2 + (4 - 1) + 4 # Default for 4 players, updated on load
        self.action_dim: int = 7
        self.max_seq_length: Optional[int] = 255
        self.num_players: int = 4
        self._last_model_input = {}

    def reset(self):
        self._last_model_input = {}
        pass

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """
        Load model state dict and re-instantiate AutoregressiveGameModelFull
        using inferred dimensions.
        """
        if "policy_nets" not in checkpoint or agent_key not in checkpoint["policy_nets"]:
            raise ValueError(f"Checkpoint missing model state for agent '{agent_key}' in 'policy_nets'.")

        model_state_dict = checkpoint["policy_nets"][agent_key]

        if not MFactoryUtil.is_ppo_autoregressive_model(model_state_dict):
            raise ValueError(f"The model state for '{agent_key}' is not a valid PPO autoregressive model.")
        
        logger.debug(f"[{self.player_id}] Model state dict extracted. Inferring dimensions...")

        try:
            inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
            inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'action_head')
            inferred_hidden_dim = MFactoryUtil.get_hidden_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
            inferred_belief_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'belief_head_op0')
            inferred_max_seq = model_state_dict.get('position_embedding.weight', torch.zeros(320)).shape[0]
            self.max_seq_length = inferred_max_seq - 1
            
            num_heads = 4  # Assuming this is fixed, common practice.
            
            if inferred_hidden_dim % num_heads != 0:
                logger.warning(f"Inferred hidden_dim ({inferred_hidden_dim}) not divisible by num_heads ({num_heads}). Check model architecture.")
        
        except ValueError as e:
            logger.error(f"Failed to infer dimensions for PPOAutoregressiveAgent {self.player_id}: {e}", exc_info=True)
            raise

        logger.info(f"Instantiating PPOAutoregressiveModel for {self.player_id} with dims: "
                    f"obs={inferred_obs_dim}, action={inferred_action_dim}, belief={inferred_belief_dim}, "
                    f"hidden={inferred_hidden_dim}, max_seq={inferred_max_seq}")

        self.model = PPOAutoregressiveModel(
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            belief_dim=inferred_belief_dim,
            hidden_dim=inferred_hidden_dim,
            num_heads=num_heads,
            max_seq_length=inferred_max_seq,
        ).to(self.device)

        try:
            self.model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            logger.error(f"Failed to load state dict for {self.player_id}: {e}", exc_info=True)
            raise

        self.model.eval()
        self.reset()
        logger.info(f"Successfully loaded PPOAutoregressiveModel for agent {self.player_id}.")

    def pop_last_model_input(self, env_idx: int, my_seat: int):
        return self._last_model_input.pop((env_idx, my_seat), None)

    def _get_relative_agent_map(self, num_players: int, my_seat: int) -> Dict[int, int]:
        """Creates a mapping from absolute seat index to relative position (0=me)."""
        return {(my_seat + i) % num_players: i for i in range(num_players)}

    def _prepare_single_sequence(self, env: lb.Env, my_seat: int):
        """
        ONE env → tensors for the AR model.

        - obs: always from *our* seat's perspective
        - actions: opponent plays hidden via action-id only (7/8/9), challenge=6, PAD=10
        - action_masks: real on *our* rows, zeros on opponent rows
        - left-shift inputs, tail-truncate to max_seq_length
        """
        ACTION_PAD = 10
        HIDDEN_BASE = 7  # 1→7, 2→8, 3→9

        rel_map = self._get_relative_agent_map(env.num_players(), my_seat)

        def hide_from_action_id(a_id: int) -> int:
            if a_id in (6, 7, 8, 9, ACTION_PAD):
                return a_id
            if 0 <= a_id <= 5:
                return HIDDEN_BASE + (a_id % 3)  # 0..5 → 7/8/9 by count
            return exit

        history = env.game_history()

        obs_list, action_list, agent_type_list, action_mask_list = [], [], [], []

        # --- past rows (already in game_history) ---
        for entry in history:
            actor = entry["player"]

            # obs always from *our* perspective
            obs_me = entry["observations"][my_seat]
            obs_list.append(torch.tensor(obs_me, dtype=torch.float32))

            # agent type in our-relative space (0==me)
            agent_type_list.append(rel_map[actor])

            # transform action purely from action id
            a = int(entry["action"])
            if actor != my_seat:
                a = hide_from_action_id(a)
            action_list.append(a)

            # mask: only meaningful on our rows; zeros on opponents
            if actor == my_seat:
                m = entry["mask"]
                action_mask_list.append(torch.tensor(m, dtype=torch.bool))
            else:
                action_mask_list.append(torch.zeros(self.action_dim, dtype=torch.bool))

        # --- current (not yet stepped) row we want to act on ---
        # obs from our seat
        curr_obs_me = env.observe_newerest(my_seat)
        obs_list.append(torch.tensor(curr_obs_me, dtype=torch.float32))
        action_list.append(ACTION_PAD)
        agent_type_list.append(0)  # me

        # grab the *current* legal mask from env info
        curr_mask = env.valid_actions()
        
        action_mask_list.append(torch.tensor(curr_mask, dtype=torch.bool))

        # --- tail truncate to model context length ---
        L = len(action_list)
        if L > self.max_seq_length:
            start = L - self.max_seq_length
            obs_list         = obs_list[start:]
            action_list      = action_list[start:]
            agent_type_list  = agent_type_list[start:]
            action_mask_list = action_mask_list[start:]
            L = self.max_seq_length

        # left-shift inputs (logits[i] predict unshifted action_list[i])
        input_actions = [ACTION_PAD] + action_list[:-1]

        # --- stack to tensors ---
        obs_t         = torch.stack(obs_list)                               # [L, obs_dim]
        in_actions_t  = torch.tensor(input_actions,   dtype=torch.long)     # [L]
        agent_types_t = torch.tensor(agent_type_list, dtype=torch.long)     # [L] 0==me
        positions_t   = torch.arange(L, dtype=torch.long)                   # [L]
        action_masks  = torch.stack(action_mask_list)                       # [L, action_dim] bool

        # return masks alongside the others so the trainer can use them directly
        return obs_t, in_actions_t, agent_types_t, positions_t, action_masks

    def _prepare_model_input_from_env(self, env: lb.Env, my_seat: int) -> Dict[str, torch.Tensor]:
        """
        Prepares the model input dictionary for a single environment's state.
        This is used for the final state snapshot and for loss calculation.
        """
        obs_seq, action_seq, agent_seq, pos_seq, action_masks = self._prepare_single_sequence(env, my_seat)

        # The model's forward pass expects a batch dimension, so we add one.
        return {
            'obs_sequence': obs_seq.unsqueeze(0).to(self.device),
            'action_sequence': action_seq.unsqueeze(0).to(self.device),
            'agent_types': agent_seq.unsqueeze(0).to(self.device),
            'positions': pos_seq.unsqueeze(0).to(self.device),
            'action_masks': action_masks.unsqueeze(0).to(self.device),
            'padding_mask': None, # This is a single sequence, no padding needed.
        }
    
    @torch.inference_mode()
    def get_actions_batch(
        self,
        arena: lb.VecArena,
        env_indices: np.ndarray,
        seat_indices: np.ndarray,
        obs_batch: np.ndarray,   # Unused (kept for interface parity)
        mask_batch: np.ndarray   # Fallback: env-provided current-step masks
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a batch of action requests from the VecArena by preparing
        and padding sequences for a single, efficient model forward pass.
        Now threads per-timestep action masks (our turns real, opponent turns zero).
        """
        
        batch_size = len(env_indices)
        if batch_size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        with amp.autocast(device_type=self.device.type, dtype=torch.float16), torch.no_grad():
            # 1) Build per-env sequences (includes per-step action_masks)
            all_obs, all_actions, all_agents, all_pos, all_masks = [], [], [], [], []
            valid_lengths = []
            for i in range(batch_size):
                env_idx = env_indices[i]
                my_seat = seat_indices[i]
                obs_seq, action_seq, agent_seq, pos_seq, action_masks = self._prepare_single_sequence(
                    arena.get_env(env_idx), my_seat
                )
                all_obs.append(obs_seq)            # [Li, obs_dim]
                all_actions.append(action_seq)     # [Li]
                all_agents.append(agent_seq)       # [Li]
                all_pos.append(pos_seq)            # [Li]
                all_masks.append(action_masks)     # [Li, A] (bool; our turns real, opponent turns zeros)
                valid_lengths.append(len(action_seq))
                Li = len(action_seq)
                
                mi_i = {
                "obs_sequence":   obs_seq.unsqueeze(0).to(self.device),        # [1, Li, obs_dim]
                "action_sequence":action_seq.unsqueeze(0).to(self.device),     # [1, Li]
                "agent_types":    agent_seq.unsqueeze(0).to(self.device),      # [1, Li]
                "positions":      pos_seq.unsqueeze(0).to(self.device),        # [1, Li]
                "action_masks":   action_masks.unsqueeze(0).to(self.device),   # [1, Li, A]
                "padding_mask":   torch.zeros(1, Li, dtype=torch.bool, device=self.device),  # [1, Li]
                "valid_lengths":  torch.tensor([Li], dtype=torch.long, device=self.device),  # [1]
                }
                # Store *detached* to avoid autograd references; CPU is fine (trainer moves to device)
                self._last_model_input[(env_idx, my_seat)] = {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in mi_i.items()
                }

            # 2) Pad to max L in batch (masks pad with False)
            device = self.device
            obs_padded     = torch.nn.utils.rnn.pad_sequence(all_obs,     batch_first=True, padding_value=0.0 ).to(device)
            actions_padded = torch.nn.utils.rnn.pad_sequence(all_actions, batch_first=True, padding_value=0  ).to(device)
            agents_padded  = torch.nn.utils.rnn.pad_sequence(all_agents,  batch_first=True, padding_value=0   ).to(device)
            pos_padded     = torch.nn.utils.rnn.pad_sequence(all_pos,     batch_first=True, padding_value=0   ).to(device)

            # (NEW) if you collected per-timestep masks:
            if all_masks and all_masks[0] is not None:
                masks_padded = torch.nn.utils.rnn.pad_sequence(all_masks, batch_first=True,
                                                            padding_value=0).to(self.device)  # [B, Lmax, A]
            else:
                masks_padded = None

            # 3) Create padding mask from ACTION length (critical: match model’s action_sequence length)
            valid_lengths = torch.tensor(valid_lengths, dtype=torch.long, device=self.device)  # [B]
            Lmax = actions_padded.size(1)
            arangeL = torch.arange(Lmax, device=self.device).unsqueeze(0)                      # [1, Lmax]
            padding_mask = arangeL >= valid_lengths.unsqueeze(1)                               # [B, Lmax]  True = PAD
            
            # 4) Single model forward pass (include valid_lengths and action_masks in the input)
            model_input = {
                'obs_sequence':   obs_padded,
                'action_sequence':actions_padded,
                'agent_types':    agents_padded,
                'positions':      pos_padded,
                'action_masks':   masks_padded,      # NEW
                'padding_mask':   padding_mask,
                'valid_lengths':  valid_lengths,   # mirrors pre-batch path
            }
            

            action_logits, _, state_values, b0, b1, b2 = self.model(**model_input)

            # 5) Per-item sampling at the last valid timestep, masked
            actions_out, log_probs_out, values_out, beliefs_out = [], [], [], []
            for i in range(batch_size):
                last_step_idx = valid_lengths[i] - 1
                logits_i = action_logits[i, last_step_idx]          # [A]
                value_i  = state_values[i, last_step_idx].item()

                # Prefer our per-sequence mask at that exact step; fall back to provided mask_batch[i]
                step_mask = masks_padded[i, last_step_idx]          # [A] bool
                if not step_mask.any() and mask_batch is not None and len(mask_batch) > i:
                    # Fallback (e.g., if older sequences didn’t populate masks)
                    m = torch.as_tensor(mask_batch[i], dtype=torch.bool, device=device)
                    if m.numel() == logits_i.numel():
                        step_mask = m

                masked_logits = logits_i.masked_fill(~step_mask, float("-inf"))
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample()

                actions_out.append(int(action.item()))
                log_probs_out.append(float(dist.log_prob(action).item()))
                values_out.append(value_i)

                belief_preds = []
                if b0 is not None: belief_preds.append(int(torch.argmax(b0[i, last_step_idx]).item()))
                if b1 is not None: belief_preds.append(int(torch.argmax(b1[i, last_step_idx]).item()))
                if b2 is not None: belief_preds.append(int(torch.argmax(b2[i, last_step_idx]).item()))
                beliefs_out.append(belief_preds)

        return (
            np.array(actions_out,   dtype=np.uint8),
            np.array(log_probs_out, dtype=np.float32),
            np.array(values_out,    dtype=np.float32),
            beliefs_out,
        )

    # This agent is not for the Python env, so these methods are not used.
    def get_action(self, *args, **kwargs):
        raise NotImplementedError("Use get_actions_batch for batched environments.")

    def get_last_expert_info(self):
        return None