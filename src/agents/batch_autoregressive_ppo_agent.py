# src/agents/batch_autoregressive_ppo_agent.py
import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Sequence

from src.agents.base_agent import BaseAgent
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.ppo_fused_model import PPOFusedModel
from src.model.model_factory import ModelFactory as MFactoryUtil
from src import config


def _req_get(req: Any, key: str, default: Any = None) -> Any:
    if isinstance(req, dict):
        return req.get(key, default)
    return getattr(req, key, default)
logger = logging.getLogger(__name__)


@dataclass
class PreparedPolicyBatch:
    """Lightweight container for a pre-built batch of policy requests."""

    requests: List[Any]
    model_input_cpu: Dict[str, torch.Tensor]
    last_inputs: List[Dict[str, torch.Tensor]]

    def subset(self, indices: Sequence[int], device: torch.device) -> Dict[str, torch.Tensor]:
        """Slice pre-built tensors for ``indices`` and transfer to ``device``."""

        if not indices:
            return {k: torch.empty((0,), device=device) for k in self.model_input_cpu}

        rows = torch.as_tensor(indices, dtype=torch.long)
        sliced: Dict[str, torch.Tensor] = {}
        for key, tensor in self.model_input_cpu.items():
            gather = torch.index_select(tensor, 0, rows)
            if device.type != "cpu":
                gather = gather.to(device=device, non_blocking=True)
            sliced[key] = gather
        return sliced


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
        is_fused = MFactoryUtil.is_fused_model(model_state_dict)
        if is_fused:
            logger.debug(f"[{self.player_id}] Detected PPOFusedModel architecture.")
            ModelClass = PPOFusedModel
        elif MFactoryUtil.is_ppo_autoregressive_model(model_state_dict):
            logger.debug(f"[{self.player_id}] Detected legacy PPOAutoregressiveModel architecture.")
            ModelClass = PPOAutoregressiveModel
        else:
            ModelClass = PPOFusedModel
        
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

        extra_kwargs: Dict[str, Any] = {}
        if is_fused:
            bricks_tensor = None
            for key, tensor in model_state_dict.items():
                if key.endswith("strategy_dictionary.bricks"):
                    bricks_tensor = tensor
                    break
            if bricks_tensor is not None:
                num_bricks, brick_dim = bricks_tensor.shape
                extra_kwargs["num_bricks"] = num_bricks
                extra_kwargs["brick_dim"] = brick_dim
            else:
                extra_kwargs["num_bricks"] = getattr(config, "NUM_BRICKS", 32)
                extra_kwargs["brick_dim"] = getattr(config, "BRICK_DIM", 32)

        # Instantiate the CORRECT ModelClass with all inferred parameters
        self.model = ModelClass(
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            hidden_dim=inferred_hidden_dim,
            num_heads=num_heads,
            max_seq_length=inferred_max_seq,
            **extra_kwargs,
        ).to(self.device)

        try:
            self.model.load_state_dict(model_state_dict, strict=False)
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
    
    @staticmethod
    def build_prepared_batch(requests: List[Any]) -> PreparedPolicyBatch:
        if not requests:
            empty = torch.empty((0, 0))
            model_input_cpu = {
                'obs_sequence': empty,
                'action_sequence': empty.clone(),
                'agent_types': empty.clone(),
                'positions': empty.clone(),
                'action_masks': empty.clone().to(torch.bool),
                'padding_mask': empty.clone().to(torch.bool),
                'valid_lengths': torch.empty((0,), dtype=torch.long),
            }
            return PreparedPolicyBatch([], model_input_cpu, [])

        obs_list: List[torch.Tensor] = []
        act_list: List[torch.Tensor] = []
        agent_list: List[torch.Tensor] = []
        pos_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        valid_lengths: List[int] = []
        last_inputs: List[Dict[str, torch.Tensor]] = []

        for req in requests:
            L = int(_req_get(req, "valid_len", 0))
            obs_seq = _req_get(req, "obs_sequence", ())
            act_seq = _req_get(req, "action_sequence", ())
            agent_seq = _req_get(req, "agent_type_sequence", ())
            pos_seq = _req_get(req, "position_sequence", ())
            mask_seq = _req_get(req, "action_mask_sequence", ())

            obs = torch.from_numpy(np.asarray(obs_seq[:L], dtype=np.float32)).clone()
            act = torch.from_numpy(np.asarray(act_seq[:L], dtype=np.int64)).clone()
            ag = torch.from_numpy(np.asarray(agent_seq[:L], dtype=np.int64)).clone()
            pos = torch.from_numpy(np.asarray(pos_seq[:L], dtype=np.int64)).clone()
            mask = torch.from_numpy(np.asarray(mask_seq[:L], dtype=np.uint8)).to(torch.bool).clone()

            obs_list.append(obs)
            act_list.append(act)
            agent_list.append(ag)
            pos_list.append(pos)
            mask_list.append(mask)
            valid_lengths.append(L)

            last_inputs.append({
                "obs_sequence": obs.unsqueeze(0).cpu(),
                "action_sequence": act.unsqueeze(0).cpu(),
                "agent_types": ag.unsqueeze(0).cpu(),
                "positions": pos.unsqueeze(0).cpu(),
                "action_masks": mask.unsqueeze(0).cpu(),
                "padding_mask": torch.zeros(1, L, dtype=torch.bool),
                "valid_lengths": torch.tensor([L], dtype=torch.long),
            })

        obs_padded = torch.nn.utils.rnn.pad_sequence(obs_list, batch_first=True, padding_value=0.0)
        act_padded = torch.nn.utils.rnn.pad_sequence(act_list, batch_first=True, padding_value=0)
        agent_padded = torch.nn.utils.rnn.pad_sequence(agent_list, batch_first=True, padding_value=0)
        pos_padded = torch.nn.utils.rnn.pad_sequence(pos_list, batch_first=True, padding_value=0)
        mask_padded = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=False)

        valid_lengths_t = torch.tensor(valid_lengths, dtype=torch.long)
        arangeL = torch.arange(act_padded.size(1)).unsqueeze(0)
        padding_mask = arangeL >= valid_lengths_t.unsqueeze(1)

        model_input_cpu = {
            'obs_sequence': obs_padded,
            'action_sequence': act_padded,
            'agent_types': agent_padded,
            'positions': pos_padded,
            'action_masks': mask_padded,
            'padding_mask': padding_mask,
            'valid_lengths': valid_lengths_t,
        }

        return PreparedPolicyBatch(requests, model_input_cpu, last_inputs)

    @torch.inference_mode()
    def get_actions_from_prepared(self,
                                  prepared: PreparedPolicyBatch,
                                  indices: Sequence[int]):
        if not indices:
            return (
                np.array([], dtype=np.uint8),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        model_input = prepared.subset(indices, self.device)
        action_logits, _, state_values = self.model(**model_input)

        valid_lengths = model_input['valid_lengths']
        rows = torch.arange(valid_lengths.shape[0], device=self.device)
        last_idx = (valid_lengths - 1).clamp_min(0)

        logits_last = action_logits[rows, last_idx, :]
        values_last = state_values[rows, last_idx].squeeze(-1)

        step_mask = model_input['action_masks'][rows, last_idx, :]
        logits_last = logits_last.masked_fill(~step_mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits_last)
        actions_t = dist.sample()
        log_probs_t = dist.log_prob(actions_t).to(torch.float32)

        for out_idx, global_idx in enumerate(indices):
            req = prepared.requests[global_idx]
            env_idx = int(_req_get(req, "env", -1))
            seat_idx = int(_req_get(req, "seat", -1))
            self._last_model_input[(env_idx, seat_idx)] = prepared.last_inputs[global_idx]

        return (
            actions_t.detach().cpu().numpy().astype(np.uint8),
            log_probs_t.detach().cpu().numpy().astype(np.float32),
            values_last.detach().cpu().numpy().astype(np.float32),
        )

    @torch.inference_mode()
    def get_actions_batch(self, requests: List[Any]):
        prepared = self.build_prepared_batch(requests)
        return self.get_actions_from_prepared(prepared, list(range(len(prepared.requests))))

    # This agent is not for the Python env, so these methods are not used.
    def get_action(self, *args, **kwargs):
        raise NotImplementedError("Use get_actions_batch for batched environments.")

    def get_last_expert_info(self):
        return None
