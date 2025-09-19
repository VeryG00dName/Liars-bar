# src/agents/autoregressive_ppo_agent.py
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from src.agents.base_agent import BaseAgent
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.ppo_fused_model import PPOFusedModel
from src.model.model_factory import ModelFactory as MFactoryUtil
from src import config
logger = logging.getLogger(__name__)


class PPOAutoregressiveAgent(BaseAgent):
    """
    Agent using the PPOAutoregressiveAgent for action prediction.
    This model internally predicts opponent beliefs and uses a unified
    architecture for all agents in the sequence.
    """
    CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}  # count -> extended action idx

    def __init__(self, device: torch.device, player_id: str):
        super().__init__(device, player_id)
        self.model: Optional[PPOAutoregressiveModel] = None

        # --- Model dimensions (inferred during loading) ---
        self.obs_dim: Optional[int] = 9
        self.action_dim: int = 7 # Standard actions
        self.extended_action_dim: Optional[int] = 9
        self.hidden_dim: Optional[int] = 256
        self.max_seq_length: Optional[int] = None
        self.belief_dim: Optional[int] = None
        self._mask_by_step = {}
        # --- Runtime state ---
        self.sequence_history: List[Dict[str, Any]] = []
        self.env_agent_id_map: Optional[Dict[str, int]] = None # Maps env_id to 0 (self), 1 (opp0), 2 (opp1), 3 (opp2)
        self._last_expert_info: Optional[Dict[str, Any]] = None

    def reset(self):
        """Resets sequence history and internal state for a new game."""
        self.sequence_history = []
        self.env_agent_id_map = None
        self._last_expert_info = None
        self._mask_by_step.clear()

    def set_model(self, model):
        """A simple method to assign the model to the agent."""
        self.model = model
        self.model.eval()

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """
        Load model state dict, automatically detect the architecture (legacy, fused),
        and re-instantiate the correct model class using inferred dimensions.
        This version correctly handles the `use_shared_belief_head` flag for the
        legacy PPOAutoregressiveModel.
        """
        if "policy_nets" not in checkpoint or agent_key not in checkpoint["policy_nets"]:
            raise ValueError(f"Checkpoint missing model state for agent '{agent_key}' in 'policy_nets'.")

        model_state_dict = checkpoint["policy_nets"][agent_key]

        # --- ARCHITECTURE AND FLAG DETECTION ---
        ModelClass = None
        model_kwargs = {} # Arguments for the model constructor

        is_fused = MFactoryUtil.is_fused_model(model_state_dict)
        if is_fused:
            logger.debug(f"[{self.player_id}] Detected PPOFusedModel architecture.")
            ModelClass = PPOFusedModel
            # The fused model doesn't use the use_shared_belief_head flag

        elif MFactoryUtil.is_ppo_autoregressive_model(model_state_dict):
            logger.debug(f"[{self.player_id}] Detected legacy PPOAutoregressiveModel architecture.")
            ModelClass = PPOAutoregressiveModel
            
            # CRITICAL: Detect which belief head style the legacy model uses
            use_shared = 'belief_head_shared.weight' in model_state_dict
            model_kwargs['use_shared_belief_head'] = use_shared
            logger.debug(f"[{self.player_id}] Inferred use_shared_belief_head={use_shared}")
        else:
            raise ValueError(f"The model state for '{agent_key}' is not a valid PPO model.")
        inferred_belief_dim = None
        # --- INFERENCE LOGIC (Common to both) ---
        try:
            inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
            inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'action_head')
            inferred_hidden_dim = MFactoryUtil.get_hidden_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
            
            # Infer belief_dim from whichever head exists
            if 'belief_head_shared.weight' in model_state_dict:
                inferred_belief_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'belief_head_shared')
            elif 'belief_head_op0.weight' in model_state_dict:
                 inferred_belief_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'belief_head_op0')

            inferred_max_seq = model_state_dict.get('position_embedding.weight').shape[0]
        except Exception as e:
            logger.error(f"Failed to infer dimensions for {self.player_id}: {e}", exc_info=True)
            raise

        self.max_seq_length = inferred_max_seq - 1
        
        # Instantiate the correct ModelClass with the correct kwargs
        if is_fused:
            bricks_tensor = None
            for key, tensor in model_state_dict.items():
                if key.endswith("strategy_dictionary.bricks"):
                    bricks_tensor = tensor
                    break

            inferred_num_bricks = None
            inferred_brick_dim = None

            if bricks_tensor is not None:
                inferred_num_bricks, inferred_brick_dim = bricks_tensor.shape
            else:
                activation_w = model_state_dict.get("strategy_dictionary.activation_encoder.2.weight")
                activation_b = model_state_dict.get("strategy_dictionary.activation_encoder.2.bias")
                opp_head_w = model_state_dict.get("opp_action_head.weight")

                if activation_w is not None:
                    inferred_num_bricks = activation_w.shape[0]
                    inferred_brick_dim = activation_w.shape[1]
                elif activation_b is not None:
                    inferred_num_bricks = activation_b.shape[0]

                if opp_head_w is not None:
                    inferred_brick_dim = opp_head_w.shape[1]

            if inferred_num_bricks is None:
                inferred_num_bricks = getattr(config, "NUM_BRICKS", 32) or 32
            if inferred_brick_dim is None:
                inferred_brick_dim = getattr(config, "BRICK_DIM", 32) or 32

            model_kwargs["num_bricks"] = int(inferred_num_bricks)
            model_kwargs["brick_dim"] = int(inferred_brick_dim)

        self.model = ModelClass(
            obs_dim=9,
            action_dim=7,
            hidden_dim=256,
            max_seq_length=256,
            **model_kwargs  # Pass specific args like use_shared_belief_head here
        ).to(self.device)

        try:
            self.model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            logger.error(f"Failed to load state dict for {self.player_id}: {e}", exc_info=True)
            raise

        self.model.eval()
        self.reset()
        logger.info(f"Successfully loaded PPOAutoregressiveModel for agent {self.player_id}.")

    def _revealed_token_from_play(self, e):
        """
        Map a revealed Play to 0..5 (table/non-table × count).
        - table:   1..3 -> 0..2
        - non-table: 1..3 -> 3..5
        """
        cnt = int(e.get("count") or 1)
        cat = e.get("card_category", "table")
        base = 0 if cat == "table" else 3
        return base + (cnt - 1)

    def _rebuild_history_from_gh(self, env, me):
        """
        Build a history where:
        - Our plays are 0..5
        - Opponent plays are hidden 7/8/9
        - Challenge is 6
        - No retro rewrite of the previous token.
        """
        gh = list(getattr(env, "game_history", []))
        seq = []

        HIDDEN_MAP = {1: 7, 2: 8, 3: 9}

        for i, e in enumerate(gh):
            a_type = e.get("action_type")
            actor  = e.get("player")
            step   = int(e.get("step"))
            obs = e.get('observations', {}).get(me, [0.0]*9)
            # Pull cached mask for OUR rows; zeros otherwise
            if actor == me and step in self._mask_by_step:
                mask = self._mask_by_step[step]
            else:
                mask = [0]   * int(self.action_dim)

            if a_type == "Play":
                cnt = int(e.get("count") or 1)

                if actor == me:
                    # Our own play is always revealed (0..5)
                    action = self._revealed_token_from_play(e)
                else:
                    # Opponent play stays hidden in the token stream (7/8/9)
                    action = HIDDEN_MAP.get(cnt, 7)

                seq.append({
                    "agent_id_env": actor,
                    "action": action,
                    "observation": obs,
                    "action_mask": mask if actor == me else [0] * int(self.action_dim)
                })

            elif a_type == "Challenge":
                seq.append({
                    "agent_id_env": actor,
                    "action": 6,
                    "observation": obs,
                    "action_mask": mask if actor == me else [0] * int(self.action_dim)
                })

        return seq

    def _prepare_model_input(self, history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepares tensors for the autoregressive model, matching the training format."""
        PAD = 10 # action pad

        filtered = list(history)

        # 1) Actions (left-shifted inputs)
        raw_actions  = [step.get("action", PAD) for step in filtered]
        input_actions = [PAD] + raw_actions[:-1]

        # 2) Length handling
        current_seq_len = len(filtered)
        max_len = self.max_seq_length
        valid_len = min(current_seq_len, max_len)
        if current_seq_len > max_len:
            filtered      = filtered[-max_len:]
            input_actions = input_actions[-max_len:]

        # 3) Allocate tensors
        obs_seq         = torch.zeros((1, valid_len, self.obs_dim), dtype=torch.float32, device=self.device)
        action_seq      = torch.zeros((1, valid_len), dtype=torch.long, device=self.device)
        agent_type_seq  = torch.ones ((1, valid_len), dtype=torch.long, device=self.device)  # default to opponent
        pos_seq         = torch.arange(valid_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.zeros((1, valid_len, self.action_dim), dtype=torch.bool, device=self.device)
        padding_mask    = torch.zeros(1, valid_len, dtype=torch.bool, device=self.device)     # no padding here

        # 4) Populate
        for i, step_data in enumerate(filtered):
            agent_type = self.env_agent_id_map[step_data["agent_id_env"]]
            agent_type_seq[0, i] = agent_type
            action_seq[0, i]     = input_actions[i]
            obs_np = np.array(step_data["observation"], dtype=np.float32)
            obs_seq[0, i] = torch.from_numpy(obs_np)
            
            if agent_type == 0:
                # Our turn: real mask
                mask_np = np.array(step_data.get("action_mask", [0]*self.action_dim), dtype=bool)
                action_mask_seq[0, i] = torch.from_numpy(mask_np)

        return {
            'obs_sequence':   obs_seq,
            'action_sequence':action_seq,
            'agent_types':    agent_type_seq,
            'positions':      pos_seq,
            'action_masks':   action_mask_seq,
            'padding_mask':   padding_mask,
            'valid_lengths':  torch.tensor([valid_len], device=self.device)
        }

    def get_action(
        self,
        env,
        agent_id_env: str,
        observation: Dict[str, Any] = None,
        info: Dict[str, Any] = None,
        cheat_expert_index: Optional[Any] = None,
        training: bool = False
    ):
        if self.model is None:
            raise RuntimeError(f"AR-PPO model not loaded for player {self.player_id}")

        if len(env.players_hands[agent_id_env]) == 5 and all(p == 0 for p in env.penalties.values()):
            self.reset()
            logger.debug(f"Agent {self.player_id}: New game detected, history cleared.")

        if self.env_agent_id_map is None:
            turn_order_list = list(env.agents)
            my_index = turn_order_list.index(agent_id_env)
            relative_order = turn_order_list[my_index:] + turn_order_list[:my_index]
            self.env_agent_id_map = {player_id: relative_pos for relative_pos, player_id in enumerate(relative_order)}
        gh = list(getattr(env, "game_history", []))
        next_step = (gh[-1]["step"] + 1) if gh else 1
        PAD = 0
        # Cache OUR obs/mask for the step that will be written to GH after env.step()
        obs_curr = env.observe(agent_id_env, newerest=True)[agent_id_env]

        if len(obs_curr) == 7:
            arr = np.asarray(obs_curr)

            out = np.full(9, PAD, dtype=arr.dtype)
            out[:4] = arr[:4]
            out[5:8] = arr[4:]
            obs_curr = out
        _, _, _, _, info = env.last()
        self._mask_by_step[next_step] = list(info.get("action_mask", [0]*self.action_dim))

        # 1) rebuild everything up-to-now
        self.sequence_history = self._rebuild_history_from_gh(env, agent_id_env)

        # 3) append the current (not-yet-in-GH) row for the model to act on
        self.sequence_history.append({
            "agent_id_env": agent_id_env,
            "observation": obs_curr,
            "action_mask": list(info.get("action_mask", [0]*self.action_dim)),
        })

        # 4) run model, write chosen action to the last row
        model_input = self._prepare_model_input(self.sequence_history)
        
        action_logits, _, state_values, _ = self.model(**model_input)
         # --- Process Outputs for the Current Timestep ---
        last_step_idx = model_input['valid_lengths'][0].item() - 1
        logits = action_logits[0, last_step_idx]
        value = state_values[0, last_step_idx]

        # Get belief predictions for each opponent
        #b0 = belief0[0, last_step_idx]
        #b1 = belief1[0, last_step_idx]
        #b2 = belief2[0, last_step_idx]
        
        mask_t = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
        masked_logits = logits.masked_fill(~mask_t, float("-inf"))

         # --- Select Action and Return All Data ---
        if training:
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            self.sequence_history[-1]["action"] = int(action.item())
            # Return all 7 values
            return action.item(), log_prob.item(), value.item()
        else: # Evaluation mode
            action = torch.argmax(masked_logits).item()
            self.sequence_history[-1]["action"] = int(action)
            return action
        
    def get_last_expert_info(self):
        """Returns the most recent internal belief predictions."""
        return self._last_expert_info