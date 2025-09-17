# src/agents/batch_autoregressive_ppo_agent.py
import torch
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from src.agents.base_agent import BaseAgent
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.ppo_fused_model import PPOFusedModel
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
        self.label: int = -1
    def reset(self):
        self._last_model_input = {}
        pass

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """
        Load model state dict, automatically detect the architecture (legacy, fused),
        and re-instantiate the correct model class using inferred dimensions.
        """
        if "policy_nets" not in checkpoint or agent_key not in checkpoint["policy_nets"]:
            raise ValueError(f"Checkpoint missing model state for agent '{agent_key}' in 'policy_nets'.")

        model_state_dict = checkpoint["policy_nets"][agent_key]

        # --- ARCHITECTURE DETECTION ---
        ModelClass = None
        if MFactoryUtil.is_fused_model(model_state_dict):
            logger.debug(f"[{self.player_id}] Detected PPOFusedModel architecture.")
            ModelClass = PPOFusedModel
        elif MFactoryUtil.is_ppo_autoregressive_model(model_state_dict):
            logger.debug(f"[{self.player_id}] Detected legacy PPOAutoregressiveModel architecture.")
            ModelClass = PPOAutoregressiveModel
        else:
            raise ValueError(f"The model state for '{agent_key}' is not a valid PPO model.")
        
        try:
            # Infer dimensions that are common to both architectures
            inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
            inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'action_head')
            inferred_hidden_dim = MFactoryUtil.get_hidden_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
            #inferred_belief_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'belief_head_shared')
            inferred_max_seq = model_state_dict.get('position_embedding.weight').shape[0]

            num_heads = inferred_hidden_dim//64
        except Exception as e:
            logger.error(f"Failed to infer dimensions for {self.player_id}: {e}", exc_info=True)
            raise

        self.max_seq_length = inferred_max_seq - 1 
        
        # Instantiate the CORRECT ModelClass with all inferred parameters
        self.model = ModelClass(
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            belief_dim=64,
            hidden_dim=inferred_hidden_dim,
            num_heads=num_heads,
            max_seq_length=inferred_max_seq,
        ).to(self.device)

        try:
            self.model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            # Provide helpful error message for shape mismatches
            logger.error(f"Failed to load state dict for {self.player_id}. This often means the inferred "
                         f"architecture params (layers, heads, etc.) don't match the checkpoint. Error: {e}", exc_info=True)
            raise

        self.model.eval()
        self.reset()
        #logger.info(f"Successfully loaded PPOAutoregressiveModel for agent {self.player_id}.")

    def pop_last_model_input(self, env_idx: int, my_seat: int):
        return self._last_model_input.pop((env_idx, my_seat), None)

    def _get_relative_agent_map(self, num_players: int, my_seat: int) -> Dict[int, int]:
        """Creates a mapping from absolute seat index to relative position (0=me)."""
        return {(my_seat + i) % num_players: i for i in range(num_players)}
    
    @torch.inference_mode()
    def get_actions_batch(self, requests: List[lb.PolicyRequest]):
        B = len(requests)
        if B == 0:
            return np.array([]), np.array([]), np.array([])

        device = self.device
        all_obs, all_actions, all_agents, all_pos = [], [], [], []
        all_masks = []
        valid_lengths = []

        for req in requests:
            L = int(req.valid_len)
            obs = torch.from_numpy(np.array(req.obs_sequence[:L], dtype=np.float32))
            act = torch.from_numpy(np.array(req.action_sequence[:L], dtype=np.int64))
            ag  = torch.from_numpy(np.array(req.agent_type_sequence[:L], dtype=np.int64))
            pos = torch.from_numpy(np.array(req.position_sequence[:L], dtype=np.int64))
            all_obs.append(obs)
            all_actions.append(act)
            all_agents.append(ag)
            all_pos.append(pos)
            valid_lengths.append(L)
            # full sequence action masks [L,7]
            mseq = torch.from_numpy(np.array(req.action_mask_sequence[:L], dtype=np.uint8)).to(torch.bool)
            all_masks.append(mseq)

            self._last_model_input[(req.env, req.seat)] = {
                "obs_sequence":    obs.unsqueeze(0).cpu(),
                "action_sequence": act.unsqueeze(0).cpu(),
                "agent_types":     ag.unsqueeze(0).cpu(),
                "positions":       pos.unsqueeze(0).cpu(),
                "action_masks":    mseq.unsqueeze(0).cpu(),
                "padding_mask":    torch.zeros(1, L, dtype=torch.bool),
                "valid_lengths":   torch.tensor([L], dtype=torch.long),
            }

        obs_padded     = torch.nn.utils.rnn.pad_sequence(all_obs,     batch_first=True, padding_value=0.0).to(device)
        actions_padded = torch.nn.utils.rnn.pad_sequence(all_actions, batch_first=True, padding_value=0  ).to(device)
        agents_padded  = torch.nn.utils.rnn.pad_sequence(all_agents,  batch_first=True, padding_value=0  ).to(device)
        pos_padded     = torch.nn.utils.rnn.pad_sequence(all_pos,     batch_first=True, padding_value=0  ).to(device)
        masks_padded   = torch.nn.utils.rnn.pad_sequence(all_masks,   batch_first=True, padding_value=0  ).to(device)
        Lmax = actions_padded.size(1)

        valid_lengths_t = torch.tensor(valid_lengths, dtype=torch.long, device=device)
        arangeL = torch.arange(Lmax, device=device).unsqueeze(0)
        padding_mask = arangeL >= valid_lengths_t.unsqueeze(1)

        model_input = {
            'obs_sequence':    obs_padded,
            'action_sequence': actions_padded,
            'agent_types':     agents_padded,
            'positions':       pos_padded,
            'action_masks':    masks_padded,
            'padding_mask':    padding_mask,
            'valid_lengths':   valid_lengths_t,
        }
        action_logits, _, state_values, _ = self.model(**model_input)

        rows = torch.arange(B, device=device)
        last_idx = (valid_lengths_t - 1).clamp_min(0)
        logits_last = action_logits[rows, last_idx, :]
        values_last = state_values[rows, last_idx].squeeze(-1)

        # Use the current-step legality mask from sequence masks
        step_mask = masks_padded[rows, last_idx, :]
        logits_last = logits_last.masked_fill(~step_mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits_last)
        actions_t   = dist.sample()
        log_probs_t = dist.log_prob(actions_t).to(torch.float32)

        return (
            actions_t.detach().cpu().numpy().astype(np.uint8),
            log_probs_t.detach().cpu().numpy().astype(np.float32),
            values_last.detach().cpu().numpy().astype(np.float32)
        )

    # This agent is not for the Python env, so these methods are not used.
    def get_action(self, *args, **kwargs):
        raise NotImplementedError("Use get_actions_batch for batched environments.")

    def get_last_expert_info(self):
        return None
