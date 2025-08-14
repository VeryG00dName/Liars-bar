# src/agents/autoregressive_agent_full.py
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from src.agents.base_agent import BaseAgent
from src.model.autoregressive_model_full import AutoregressiveGameModelFull
from src.env.liars_deck_env_utils_2 import decode_action
from src.model.model_factory import ModelFactory as MFactoryUtil
logger = logging.getLogger(__name__)


class AutoregressiveAgentFull(BaseAgent):
    """
    Agent using the AutoregressiveGameModelFull for action prediction.
    This model internally predicts opponent beliefs and uses a unified
    architecture for all agents in the sequence.
    """
    CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}  # count -> extended action idx

    def __init__(self, device: torch.device, player_id: str):
        super().__init__(device, player_id)
        self.model: Optional[AutoregressiveGameModelFull] = None

        # --- Model dimensions (inferred during loading) ---
        self.obs_dim: Optional[int] = None
        self.action_dim: int = 7 # Standard actions
        self.extended_action_dim: Optional[int] = None
        self.hidden_dim: Optional[int] = None
        self.max_seq_length: Optional[int] = None
        self.belief_dim: Optional[int] = None # Inferred from the model's belief head
        self._obs_by_step = {}
        # --- Runtime state ---
        self.sequence_history: List[Dict[str, Any]] = []
        self.env_agent_id_map: Optional[Dict[str, int]] = None # Maps env_id to 0 (self), 1 (opp0), 2 (opp1)
        self._last_expert_info: Optional[Dict[str, Any]] = None
        self._last_seen_gh_step: int = -1  # highest env.game_history 'step' we’ve copied in
        self._gh_step_to_seq_idx = {}

    def reset(self):
        """Resets sequence history and internal state for a new game."""
        self.sequence_history = []
        self.env_agent_id_map = None
        self._last_expert_info = None
        self._last_seen_gh_step = -1
        self._gh_step_to_seq_idx.clear()
        self._obs_by_step.clear()

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
        - A per-step 'reveal' feature (0..5 or 7=NO_REVEAL) turns on *after* a challenge.
        """
        gh = list(getattr(env, "game_history", []))
        seq = []

        NO_REVEAL = 7
        HIDDEN_MAP = {1: 7, 2: 8, 3: 9}

        current_reveal = NO_REVEAL         # what the model may see at this step
        last_opp_revealed = None           # 0..5 for the most recent opponent Play (true class), if any
        last_play_was_opp = False

        for i, e in enumerate(gh):
            a_type = e.get("action_type")
            actor  = e.get("player")
            step   = int(e.get("step"))

            # Pull cached obs/mask for OUR rows; zeros otherwise
            if actor == me and step in self._obs_by_step:
                obs, mask = self._obs_by_step[step]
            else:
                obs  = [0.0] * int(self.obs_dim)
                mask = [0]   * int(self.action_dim)

            if a_type == "Play":
                cnt = int(e.get("count") or 1)

                if actor == me:
                    # Our own play is always revealed (0..5)
                    action = self._revealed_token_from_play(e)
                    last_play_was_opp = False
                else:
                    # Opponent play stays hidden in the token stream (7/8/9)
                    action = HIDDEN_MAP.get(cnt, 7)
                    # But remember the true revealed class for *after* a challenge
                    last_opp_revealed = self._revealed_token_from_play(e)
                    last_play_was_opp = True

                seq.append({
                    "agent_id_env": actor,
                    "action": action,
                    "observation": obs,
                    "action_mask": mask if actor == me else [0] * int(self.action_dim),
                    "reveal": current_reveal,   # side feature visible at this step
                })

            elif a_type == "Challenge":
                # Append the challenge row with the *current* reveal value (no leak on this step)
                seq.append({
                    "agent_id_env": actor,
                    "action": 6,
                    "observation": obs,
                    "action_mask": mask if actor == me else [0] * int(self.action_dim),
                    "reveal": current_reveal,   # still NO_REVEAL on the challenge step itself
                })

                # If the last Play was by an opponent, flip reveal ON for subsequent steps
                if last_play_was_opp and last_opp_revealed is not None:
                    current_reveal = last_opp_revealed
                # If it challenged our own play or no valid opp play, keep current_reveal as-is
                last_play_was_opp = False  # consume the reveal condition

            # ignore other action types if any

        return seq

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """
        Load model state dict and re-instantiate AutoregressiveGameModelFull
        using inferred dimensions.
        """
        # --- Extract model state dict from the unified format ---
        if "policy_nets" not in checkpoint:
            raise ValueError("Checkpoint is missing the 'policy_nets' section.")
        if agent_key not in checkpoint["policy_nets"]:
            raise ValueError(f"Checkpoint missing model state for agent '{agent_key}' in 'policy_nets'.")

        model_state_dict = checkpoint["policy_nets"][agent_key]

        if not MFactoryUtil.is_autoregressive_model(model_state_dict):
            raise ValueError(f"The model state for '{agent_key}' is not a valid autoregressive model.")
        logger.debug(f"[{agent_key}] Model state dict successfully extracted.")

        # --- Inference of model dimensions from state_dict ---
        inferred_hidden_dim = None
        inferred_action_dim = None
        inferred_ext_action_dim = None
        inferred_obs_dim = None
        inferred_max_seq = None
        inferred_belief_dim = None
        default_num_heads = 4 # Matches the training script

        try:
            # Infer hidden_dim from action embedding or a transformer layer
            if 'action_embedding.weight' in model_state_dict:
                inferred_hidden_dim = model_state_dict['action_embedding.weight'].shape[-1]
            elif 'transformer.layers.0.linear1.weight' in model_state_dict:
                inferred_hidden_dim = model_state_dict['transformer.layers.0.linear1.weight'].shape[1]

            # Use factory helpers for standard dimensions
            inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'action_head')
            inferred_ext_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'extended_action_head')
            inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')

            # Infer belief_dim from the belief_head output size
            if 'belief_head.weight' in model_state_dict:
                inferred_belief_dim = model_state_dict['belief_head.weight'].shape[0]

            # Infer max_seq_length from position embeddings
            if 'position_embedding.weight' in model_state_dict:
                inferred_max_seq = model_state_dict['position_embedding.weight'].shape[0]

        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Dimension inference failed for AR-Full {self.player_id}: {e}. Will rely on defaults.", exc_info=True)

        # --- Apply defaults and validate ---
        temp_defaults = AutoregressiveGameModelFull(obs_dim=4, action_dim=7, belief_dim=10)
        self.action_dim = inferred_action_dim or temp_defaults.action_dim
        self.extended_action_dim = inferred_ext_action_dim or temp_defaults.extended_action_dim
        self.obs_dim = inferred_obs_dim or temp_defaults.obs_dim
        self.max_seq_length = inferred_max_seq or temp_defaults.max_seq_length
        self.belief_dim = inferred_belief_dim or temp_defaults.belief_dim

        # --- Special handling and validation for hidden_dim ---
        temp_hidden_dim = inferred_hidden_dim or temp_defaults.hidden_dim
        if temp_hidden_dim % default_num_heads != 0:
            logger.warning(f"Inferred/Default hidden_dim ({temp_hidden_dim}) is not divisible by num_heads ({default_num_heads}). Using default hidden_dim from model definition ({temp_defaults.hidden_dim}) instead.")
            self.hidden_dim = temp_defaults.hidden_dim
            if self.hidden_dim % default_num_heads != 0:
                raise ValueError(f"FATAL: Model's default hidden_dim ({self.hidden_dim}) is also not divisible by num_heads ({default_num_heads}).")
        else:
            self.hidden_dim = temp_hidden_dim

        # --- Instantiate the model with inferred dimensions ---
        logger.info(f"Instantiating AR-Full model for {self.player_id} with dims: obs={self.obs_dim}, action={self.action_dim}, belief={self.belief_dim}, hidden={self.hidden_dim}, max_seq={self.max_seq_length}")
        self.model = AutoregressiveGameModelFull(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            belief_dim=self.belief_dim,
            hidden_dim=self.hidden_dim,
            num_heads=default_num_heads,
            num_layers=2,
            max_seq_length=self.max_seq_length,
            num_agent_types=3 # 0: Self, 1: Opponent 0, 2: Opponent 1
        ).to(self.device)

        # --- Load the model weights ---
        try:
            missing, unexpected = self.model.load_state_dict(model_state_dict, strict=True)
            if missing or unexpected:
                logger.warning(f"[{agent_key}] load_state_dict results: missing={missing}, unexpected={unexpected}")
        except RuntimeError as e:
            logger.error(f"Failed to load model state dict for {agent_key}: {e}", exc_info=True)
            raise

        self.model.eval()
        self.reset()
        logger.info(f"Successfully loaded AR-Full model for agent {self.player_id}.")


    def _prepare_model_input(self, history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepares tensors for the autoregressive model, matching the training format."""
        PAD = 0                     # action pad
        NO_REVEAL = 7               # 0..6 are revealed classes, 7 means "no reveal yet"

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
        reveal_seq      = torch.full ((1, valid_len), NO_REVEAL, dtype=torch.long, device=self.device)
        padding_mask    = torch.zeros(1, valid_len, dtype=torch.bool, device=self.device)     # no padding here

        # Compute a carry-forward reveal for rows that don't include it (like the freshly appended current row)
        carry_reveal = NO_REVEAL
        for j in range(len(filtered) - 1, -1, -1):
            if 'reveal' in filtered[j]:
                carry_reveal = int(filtered[j]['reveal'])
                break

        # 4) Populate
        for i, step_data in enumerate(filtered):
            agent_type = self.env_agent_id_map[step_data["agent_id_env"]]
            agent_type_seq[0, i] = agent_type
            action_seq[0, i]     = input_actions[i]

            # Reveal side-feature: prefer per-step value if present; else carry last known state
            r = step_data.get('reveal', carry_reveal)
            reveal_seq[0, i] = int(r)

            if agent_type == 0:
                # Our turn: real obs + real mask
                obs_np = np.array(step_data["observation"], dtype=np.float32)
                obs_seq[0, i] = torch.from_numpy(obs_np)
                mask_np = np.array(step_data.get("action_mask", [0]*self.action_dim), dtype=bool)
                action_mask_seq[0, i] = torch.from_numpy(mask_np)
            else:
                # Opponent turn: zeros (keep alignment with training)
                # (model learns from agent rows; opp rows provide context)
                pass

        return {
            'obs_sequence':   obs_seq,
            'action_sequence':action_seq,
            'agent_types':    agent_type_seq,
            'positions':      pos_seq,
            'action_masks':   action_mask_seq,
            'padding_mask':   padding_mask,
            'valid_lengths':  torch.tensor([valid_len], device=self.device),
            'reveal_sequence':reveal_seq,                 # << NEW
        }

    def get_action(
        self,
        env,
        agent_id_env: str,
        observation: Dict[str, Any],
        info: Dict[str, Any],
        cheat_expert_index: Optional[Any] = None
    ) -> int:
        if self.model is None:
            raise RuntimeError(f"AR-Full model not loaded for player {self.player_id}")

        if len(env.players_hands[agent_id_env]) == 5 and all(p == 0 for p in env.penalties.values()):
            self.reset()
            logger.debug(f"Agent {self.player_id}: New game detected, history cleared.")

        if self.env_agent_id_map is None:
            # Keep this consistent with how you trained
            # Prefer actual seat order if your dataset used fixed player_0/1/2
            if agent_id_env == "player_0":
                self.env_agent_id_map = {"player_0": 0, "player_1": 1, "player_2": 2}
            else:
                others = [a for a in env.possible_agents if a != agent_id_env]
                # sorted() if that matches training, otherwise keep env.possible_agents order
                self.env_agent_id_map = {agent_id_env: 0}
                for i, opp in enumerate(others, start=1):
                    if i <= 2:  # we only embed 3 agent types
                        self.env_agent_id_map[opp] = i
        gh = list(getattr(env, "game_history", []))
        next_step = (gh[-1]["step"] + 1) if gh else 1

        # Cache OUR obs/mask for the step that will be written to GH after env.step()
        obs_curr = env.observe(agent_id_env, newest=True)[agent_id_env]
        self._obs_by_step[next_step] = (
            obs_curr,
            list(info.get("action_mask", [0]*self.action_dim))
)
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
        with torch.inference_mode():
            action_logits, _, _, belief0, belief1 = self.model(**model_input)
        last_step_idx = model_input['valid_lengths'][0].item() - 1
        logits = action_logits[0, last_step_idx]
        mask_t = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
        chosen = torch.argmax(logits.masked_fill(~mask_t, float("-inf"))).item()

        if belief0 is not None and belief1 is not None:
            opponents = sorted([p for p in env.possible_agents if p != agent_id_env])
            self._last_expert_info = {}
            if len(opponents) > 0:
                self._last_expert_info[opponents[0]] = {"expert_index": int(torch.argmax(belief0[0, last_step_idx]).item()), "source": "internal"}
            if len(opponents) > 1:
                self._last_expert_info[opponents[1]] = {"expert_index": int(torch.argmax(belief1[0, last_step_idx]).item()), "source": "internal"}
                
        self.sequence_history[-1]["action"] = chosen
        return chosen

    def get_last_expert_info(self):
        """Returns the most recent internal belief predictions."""
        return self._last_expert_info