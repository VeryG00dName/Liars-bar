# src/agents/batch_autoregressive_ppo_agent.py
import torch
import numpy as np
import logging
from typing import Optional, Dict, Any

from src.agents.base_agent import BaseAgent
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.model_factory import ModelFactory as MFactoryUtil
from src.misc import lb
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
        mask_batch: np.ndarray,   # Fallback: env-provided current-step masks
    ):
        B = len(env_indices)
        if B == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        device = self.device

        all_obs, all_actions, all_agents, all_pos, all_masks = [], [], [], [], []
        valid_lengths_py = []

        # -------- 1) per-env sequence prep (CPU) --------
        for i in range(B):
            env_idx = env_indices[i]
            my_seat = seat_indices[i]
            obs_seq, action_seq, agent_seq, pos_seq, action_masks = self._prepare_single_sequence(
                arena.get_env(env_idx), my_seat
            )

            all_obs.append(obs_seq)
            all_actions.append(action_seq)
            all_agents.append(agent_seq)
            all_pos.append(pos_seq)
            all_masks.append(action_masks)
            valid_lengths_py.append(len(action_seq))

            # Always snapshot last model_input for finalize(); keep on CPU
            Li = len(action_seq)
            self._last_model_input[(env_idx, my_seat)] = {
                "obs_sequence":    obs_seq.unsqueeze(0).cpu(),
                "action_sequence": action_seq.unsqueeze(0).cpu(),
                "agent_types":     agent_seq.unsqueeze(0).cpu(),
                "positions":       pos_seq.unsqueeze(0).cpu(),
                "action_masks":    (action_masks.unsqueeze(0).cpu() if action_masks is not None else None),
                # These can be recomputed, but keeping them is cheap:
                "padding_mask":    torch.zeros(1, Li, dtype=torch.bool),
                "valid_lengths":   torch.tensor([Li], dtype=torch.long),
            }

        # -------- 2) pad and move to device --------
        obs_padded     = torch.nn.utils.rnn.pad_sequence(all_obs,     batch_first=True, padding_value=0.0).to(device)
        actions_padded = torch.nn.utils.rnn.pad_sequence(all_actions, batch_first=True, padding_value=0  ).to(device)
        agents_padded  = torch.nn.utils.rnn.pad_sequence(all_agents,  batch_first=True, padding_value=0  ).to(device)
        pos_padded     = torch.nn.utils.rnn.pad_sequence(all_pos,     batch_first=True, padding_value=0  ).to(device)
        Lmax = actions_padded.size(1)

        # -------- 3) build masks / valid lengths --------
        if all_masks and all_masks[0] is not None:
            masks_padded = torch.nn.utils.rnn.pad_sequence(all_masks, batch_first=True, padding_value=0).to(device)
        else:
            masks_padded = None

        valid_lengths = torch.tensor(valid_lengths_py, dtype=torch.long, device=device)   # [B]
        arangeL = torch.arange(Lmax, device=device).unsqueeze(0)                          # [1, Lmax]
        padding_mask = arangeL >= valid_lengths.unsqueeze(1)                               # [B, Lmax] True=PAD

        # -------- 4) single model forward --------
        model_input = {
            'obs_sequence':    obs_padded,
            'action_sequence': actions_padded,
            'agent_types':     agents_padded,
            'positions':       pos_padded,
            'action_masks':    masks_padded,
            'padding_mask':    padding_mask,
            'valid_lengths':   valid_lengths,
        }
        action_logits, _, state_values, b0, b1, b2 = self.model(**model_input)            # [B, L, A], [B, L, 1], ...

        # -------- 5) batched sampling & packing (vectorized) --------
        rows = torch.arange(B, device=device)
        last_idx = (valid_lengths - 1).clamp_min(0)                                       # [B]
        logits_last = action_logits[rows, last_idx, :]                                    # [B, A]
        values_last = state_values[rows, last_idx].squeeze(-1)                            # [B]

        # prefer per-seq masks; fallback to current-step mask_batch row-wise
        if masks_padded is not None:
            step_mask = masks_padded[rows, last_idx, :]                                   # [B, A] bool
        else:
            step_mask = None

        if step_mask is None and mask_batch is not None:
            step_mask = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)      # [B, A]
        elif step_mask is not None and mask_batch is not None:
            fallback = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)
            no_valid = ~step_mask.any(dim=1, keepdim=True)                                # [B,1]
            step_mask = torch.where(no_valid, fallback, step_mask)

        if step_mask is not None:
            logits_last = logits_last.masked_fill(~step_mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits_last)                        # batch-wise
        actions_t   = dist.sample()                                                       # [B]
        log_probs_t = dist.log_prob(actions_t).to(torch.float32)                          # [B]

        # batched belief argmaxes (if present)
        beliefs_out = []
        if b0 is not None:
            b0_last = torch.argmax(b0[rows, last_idx, :], dim=-1)
        if b1 is not None:
            b1_last = torch.argmax(b1[rows, last_idx, :], dim=-1)
        if b2 is not None:
            b2_last = torch.argmax(b2[rows, last_idx, :], dim=-1)

        for i in range(B):
            bi = []
            if b0 is not None: bi.append(int(b0_last[i].item()))
            if b1 is not None: bi.append(int(b1_last[i].item()))
            if b2 is not None: bi.append(int(b2_last[i].item()))
            beliefs_out.append(bi)

        # numpy outputs
        return (
            actions_t.detach().cpu().numpy().astype(np.uint8),
            log_probs_t.detach().cpu().numpy().astype(np.float32),
            values_last.detach().cpu().numpy().astype(np.float32),
            beliefs_out,
        )

    # This agent is not for the Python env, so these methods are not used.
    def get_action(self, *args, **kwargs):
        raise NotImplementedError("Use get_actions_batch for batched environments.")

    def get_last_expert_info(self):
        return None