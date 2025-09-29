# src/model/model_factory.py
import logging
import torch
import torch.nn as nn # Added nn
from src import config
from src.model.common_model_api import BasePolicyNetwork, BaseValueNetwork, BaseOpponentBehaviorPredictor
from src.model.shen_models import BeliefSpacePolicy, OpponentBeliefModel
# Import new implementations for policy and value networks.
from src.model.new_models import PolicyNetwork as NewPolicyNetwork, ValueNetwork as PPOValueNetwork
# Import MoE models when needed (dynamic import in create_policy_network)

class ModelFactory:

    # --- ADD THESE STATIC METHODS ---
    @staticmethod
    def get_hidden_dim_from_state_dict(state_dict, layer_prefix='fc1'):
        """Determines the hidden dimension from a state dictionary."""
        # Try several candidate prefixes.
        candidate_prefixes = [
            layer_prefix,
            "base_encoder.0",
            "policy_net.fc1",
            "model.fc1",
            "network.0", # For BeliefSpacePolicy
            "experts.0.fc1", # For MoE
            "obs_encoder.0" # For PPOAutoregressiveModel
        ]
        for prefix in candidate_prefixes:
            key = f"{prefix}.weight"
            if key in state_dict:
                return state_dict[key].shape[0] # Output dimension of the layer
        # Fallback: return the first dimension of the first 2D tensor found.
        for key, tensor in state_dict.items():
            if hasattr(tensor, "ndim") and tensor.ndim == 2:
                return tensor.shape[0]
        # If still not found, include available keys in the error message.
        available_keys = list(state_dict.keys())
        raise ValueError(f"Cannot determine hidden_dim from state_dict. Tried prefixes: {candidate_prefixes}. Available keys: {available_keys}")

    @staticmethod
    def get_input_dim_from_state_dict(state_dict, layer_prefix='fc1'):
        """Determines the input dimension from a state dictionary."""
        candidate_prefixes = [
            layer_prefix,
            "base_encoder.0",
            "policy_net.fc1",
            "model.fc1",
            "network.0", # For BeliefSpacePolicy
            "experts.0.fc1", # For MoE
            "obs_encoder.0" # For PPOAutoregressiveModel
        ]
        for prefix in candidate_prefixes:
            key = f"{prefix}.weight"
            if key in state_dict:
                return state_dict[key].shape[1] # Input dimension of the layer
        # Fallback: iterate over all keys and return the input dimension from the first 2D tensor found.
        for key, tensor in state_dict.items():
            if hasattr(tensor, "ndim") and tensor.ndim == 2:
                return tensor.shape[1]
        available_keys = list(state_dict.keys())
        raise ValueError(f"Cannot determine input_dim from state_dict. Tried prefixes: {candidate_prefixes}. Available keys: {available_keys}")
    # --- END OF ADDED METHODS ---

    # (Keep other existing static methods: is_belief_space_policy, create_belief_space_policy, etc.)
    @staticmethod
    def is_belief_space_policy(state_dict):
        network_key = 'network.0.weight'
        policy_key = 'policy_head.weight'
        value_key = 'value_head.0.weight' # Check first layer of value head
        return network_key in state_dict and policy_key in state_dict and value_key in state_dict

    @staticmethod
    def create_belief_space_policy(belief_dim, obs_dim, hidden_dim, output_dim):
        model = BeliefSpacePolicy(
            belief_dim=belief_dim, obs_dim=obs_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )
        return model

    @staticmethod
    def create_opponent_belief_model(event_feature_dim, hidden_dim, num_opponent_types, max_seq_length=400): # Match args
        model = OpponentBeliefModel(
            event_feature_dim=event_feature_dim, hidden_dim=hidden_dim,
            num_opponent_types=num_opponent_types, max_seq_length=max_seq_length
        )
        return model

    @staticmethod
    def is_autoregressive_model(state_dict: dict) -> bool:
        """
        Detect AutoregressiveGameModel (old atomic + new factorized).
        Heuristic: must have a policy action head, and either
        (a) look transformer-ish, or
        (b) have our embedding stack (agent+position plus action/factors).
        """

        def has(name: str) -> bool:
            # exact or common prefixed key
            if name in state_dict:
                return True
            for p in ("module.", "model.", "policy_net.", "policy.", "net."):
                k = p + name
                if k in state_dict or ("module." + k) in state_dict:
                    return True
            # last-resort suffix match (handles extra nesting)
            return any(k.endswith(name) for k in state_dict.keys())

        # 1) AR policies have an action head (and often an opp head)
        has_action_head = any(has(n) for n in (
            "action_head.weight", "action_head.bias",
            "opp_action_head.weight", "opp_action_head.bias",
            # tolerate alt names used in some repos:
            "actor_head.weight", "actor_head.bias",
            "policy_head.weight", "policy_head.bias",
        ))

        # 2) “Transformer-ish” backbone or our causal mask buffer
        has_transformerish = any(has(n) for n in (
            "transformer.layers.0.self_attn.in_proj_weight",
            "transformer.layers.0.linear1.weight",
            "transformer.layers.0.self_attn.out_proj.weight",
            "transformer.layers.0.norm1.weight",
            "causal_bool_mask_full",  # registered buffer in your AR model
        ))

        # 3) Either old atomic action embedding or the new factorized ones
        has_any_action_emb = any(has(n) for n in (
            "action_embedding.weight",              # old
            "act_kind_embedding.weight",            # new
            "count_embedding.weight",
            "table_flag_embedding.weight",
        ))
        has_agent_pos = has("agent_embedding.weight") and has("position_embedding.weight")

        return bool(has_action_head and (has_transformerish or (has_any_action_emb and has_agent_pos)))
    
    @staticmethod
    def is_ppo_autoregressive_model(state_dict: dict) -> bool:
        """
        Detects the PPOAutoregressiveModel, supporting both legacy and shared-head versions.
        Heuristic: Checks for factorized action embeddings, the causal mask, and specific
        PPO heads. It then checks for EITHER the legacy belief heads OR the new shared head.
        """
        has_factor_embeddings = all(k in state_dict for k in [
            'act_kind_embedding.weight', 'count_embedding.weight', 'table_flag_embedding.weight'
        ])
        has_causal_mask = 'causal_bool_mask_full' in state_dict
        has_specific_heads = 'action_head.weight' in state_dict and 'opp_action_head.weight' in state_dict

        # Check for either of the two belief head architectures
        has_legacy_belief_heads = all(k in state_dict for k in [
            'belief_head_op0.weight', 'belief_head_op1.weight', 'belief_head_op2.weight'
        ])
        has_shared_belief_head = 'belief_head_shared.weight' in state_dict

        # A valid model must have the base components and AT LEAST ONE of the belief architectures
        return (has_factor_embeddings and has_causal_mask and has_specific_heads and
                (has_legacy_belief_heads or has_shared_belief_head))

    @staticmethod
    def is_fused_model(state_dict: dict) -> bool:
        """
        Detects the new PPOFusedModel by checking for the unique fusion layer.
        """
        
        if ModelFactory.is_reactive_model(state_dict):
            return False
        
        def _has(key: str) -> bool:
            if key in state_dict:
                return True
            for prefix in ("module.", "model.", "policy_nets.", "_orig_mod."):
                pref_key = f"{prefix}{key}"
                if pref_key in state_dict:
                    return True
            return any(k.endswith(key) for k in state_dict.keys())

        has_strategy_dictionary = any(
            k.endswith("strategy_dictionary.bricks") for k in state_dict.keys()
        )
        if has_strategy_dictionary and _has("action_head.weight"):
            return True

        # Legacy fused models relied on a policy/value fusion stack.
        has_fusion_layer = _has('policy_value_feature_extractor.0.weight')
        has_belief_fc = _has('belief_fc.0.weight')
        has_action_head = _has('action_head.weight')

        return has_fusion_layer and has_belief_fc and has_action_head

    @staticmethod
    def is_reactive_model(state_dict: dict) -> bool:
        """
        Detects the PPOReactiveModel. The key feature is the *absence* of the
        StrategyDictionary, while still having a transformer-based architecture.
        """
        # A reactive model will NOT have any parameters prefixed with "strategy_dictionary."
        has_strategy_dictionary = any(
            k.startswith("strategy_dictionary.") for k in state_dict.keys()
        )
        if has_strategy_dictionary:
            return False

        # To confirm it's our reactive model, we check for other key components
        # that it shares with the Fused model.
        has_transformer = any(k.startswith("transformer.") for k in state_dict.keys())
        has_action_head = any(k.startswith("action_head.") for k in state_dict.keys())

        return has_transformer and has_action_head

    @staticmethod
    def get_belief_dimensions(state_dict):
        """
        Extracts dimensions from a PPOAutoregressiveModel state dictionary.
        This function is now largely superseded by direct inference in the agent's
        load_models_from_checkpoint method, but is kept for compatibility or other uses.
        It has been simplified to reflect the more direct inference.
        """
        logger = logging.getLogger(__name__)
        try:
            # For PPOAutoregressiveModel, obs_dim is inferred from the obs_encoder
            obs_dim = state_dict['obs_encoder.0.weight'].shape[1]
            logger.debug(f"Inferred obs_dim={obs_dim} from obs_encoder.")

            # Infer belief_dim from whichever belief head exists
            if 'belief_head_shared.weight' in state_dict:
                belief_dim = state_dict['belief_head_shared.weight'].shape[0]
                logger.debug(f"Inferred belief_dim={belief_dim} from shared belief head.")
            elif 'belief_head_op0.weight' in state_dict:
                belief_dim = state_dict['belief_head_op0.weight'].shape[0]
                logger.debug(f"Inferred belief_dim={belief_dim} from legacy belief head op0.")
            else:
                raise ValueError("Could not find a valid belief head in the state_dict.")

            # total_input_dim is not a concept for this model as inputs are separate
            # We return obs_dim and belief_dim per opponent
            return (None, obs_dim, belief_dim)

        except (KeyError, ValueError) as e:
            logger.error(f"Failed to get belief dimensions: {e}. Returning defaults.")
            return (None, 7, 10) # Fallback defaults

    @staticmethod
    def is_stacked_observation_model(state_dict):
        """Detects if a state dict is from the older StackedObservationConvModel."""
        is_newer = ModelFactory.is_stacked_newer_observation_model(state_dict)
        has_conv = any(k.startswith('conv_layers.') for k in state_dict.keys())
        has_old_policy_head = 'policy_head.weight' in state_dict
        has_old_value_head = 'value_head.weight' in state_dict

        # It's an older stacked model if it has conv layers and the old heads, BUT NOT the newer heads
        return has_conv and has_old_policy_head and has_old_value_head and not is_newer

    @staticmethod
    def get_belief_input_dim(state_dict):
        _, suggested_obs_dim, _ = ModelFactory.get_belief_dimensions(state_dict)
        return suggested_obs_dim

    @staticmethod
    def get_num_opponent_types(belief_model_state_dict):
        """ Extracts the number of opponent types from an OpponentBeliefModel state dictionary."""
        # --- MODIFIED: Check index 4 based on error log ---
        output_layer_key = 'belief_update.4.weight'
        if output_layer_key not in belief_model_state_dict:
             # Try older keys if structure changed
             output_layer_key = 'belief_update.3.weight'
             if output_layer_key not in belief_model_state_dict:
                  output_layer_key = 'belief_update.2.weight' # Fallback

        if output_layer_key in belief_model_state_dict:
            return belief_model_state_dict[output_layer_key].shape[0] # Output size is num_types
        else:
             keys_found = list(belief_model_state_dict.keys())
             print(f"Warning: Could not find belief update output layer (tried indices 4, 3, 2). Available keys: {keys_found}")
             return 10 # Default

    @staticmethod
    def get_output_dim_from_state_dict(state_dict, layer_prefix='fc4'):
        """Determines the output dimension (action space size) from a policy state dictionary."""
        # Check standard policy output layers first
        candidate_prefixes = [ layer_prefix, 'policy_head', 'experts.0.fc_out', 'strategy_query','action_head', 'belief_head_op0']

        for prefix in candidate_prefixes:
            key_w = f"{prefix}.weight"; key_b = f"{prefix}.bias"
            if key_w in state_dict: return state_dict[key_w].shape[0]
            if key_b in state_dict: return state_dict[key_b].shape[0]

        return 7

    @staticmethod
    def create_policy_network(use_aux_classifier: bool = False, num_opponent_classes: int = None,
                              input_dim: int = 26, hidden_dim: int = config.HIDDEN_DIM, output_dim: int = config.OUTPUT_DIM,
                              use_lstm: bool = True, use_dropout: bool = True, use_layer_norm: bool = True,
                              use_new_model: bool = True, strategy_dim: int = 5, num_opponents: int = 2,
                              use_moe_model: bool = False, num_experts: int = 10) -> BasePolicyNetwork:
        # (Keep implementation as before)
        if use_moe_model:
            from src.model.other_models import PolicyNetwork as MoEPolicyNetwork
            model = MoEPolicyNetwork(
                input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_experts=num_experts,
                use_lstm=use_lstm, use_dropout=use_dropout, use_layer_norm=use_layer_norm
            )
        elif use_new_model:
            model = NewPolicyNetwork(
                input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, use_lstm=use_lstm,
                use_dropout=use_dropout, use_layer_norm=use_layer_norm,
                use_aux_classifier=use_aux_classifier, num_opponent_classes=num_opponent_classes
            )
        else: # Older model from other_models
             from src.model.other_models import PolicyNetwork as OtherPolicyNetwork
             try:
                  model = OtherPolicyNetwork(
                       input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       strategy_dim=strategy_dim, num_opponents=num_opponents,
                       use_lstm=use_lstm, use_dropout=use_dropout
                  )
             except TypeError: # Handle potential changes in constructor signature
                  print("Warning: Old PolicyNetwork constructor might have changed. Trying without strategy/opponent args.")
                  model = OtherPolicyNetwork(
                       input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       use_lstm=use_lstm, use_dropout=use_dropout
                  )
        return model


    @staticmethod
    def create_stacked_observation_model(obs_dim: int, num_actions: int, hidden_dim: int = config.HIDDEN_DIM, num_obs_stack: int = 10) -> nn.Module:
        from src.model.models import StackedObservationConvModel
        model = StackedObservationConvModel(
            obs_dim=obs_dim, num_actions=num_actions, hidden_dim=hidden_dim,
            num_obs_stack=num_obs_stack
            # Note: Older version might not take num_players
        )
        return model

    @staticmethod
    def create_stacked_newer_observation_model(obs_dim: int, num_actions: int, hidden_dim: int = config.HIDDEN_DIM, num_obs_stack: int = 10, num_players: int = 3) -> nn.Module:
        from src.model.models import StackedObservationConvModel
        model = StackedObservationConvModel(
            obs_dim=obs_dim, num_actions=num_actions, hidden_dim=hidden_dim,
            num_obs_stack=num_obs_stack, num_players=num_players # Newer version includes num_players
        )
        return model

    @staticmethod
    def is_stacked_newer_observation_model(state_dict):
        return (any(k.startswith('conv_layers.') for k in state_dict.keys()) and
                'policy_head1.weight' in state_dict and 'value_head1.weight' in state_dict and
                'policy_head2.weight' in state_dict and 'value_head2.weight' in state_dict and
                'game_state_head.weight' in state_dict)

    @staticmethod
    def is_moe_policy(state_dict):
        return any(k.startswith('experts.') for k in state_dict.keys())

    @staticmethod
    def create_value_network(input_dim: int = 26, hidden_dim: int = 64, use_dropout: bool = True, use_layer_norm: bool = True) -> BaseValueNetwork:
        model = PPOValueNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim, use_dropout=use_dropout, use_layer_norm=use_layer_norm
        )
        return model

    @staticmethod
    def create_obp(use_transformer_memory: bool = True, input_dim: int = None, hidden_dim: int = 64, output_dim: int = 2) -> BaseOpponentBehaviorPredictor:
        if input_dim is None: input_dim = config.OPPONENT_INPUT_DIM
        if use_transformer_memory:
            from src.model.new_models import OpponentBehaviorPredictor as NewOpponentBehaviorPredictor
            model = NewOpponentBehaviorPredictor(
                input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, memory_dim=config.STRATEGY_DIM
            )
        else:
            # Ensure the old OBP model is correctly imported and instantiated
            try:
                from src.model.models import OpponentBehaviorPredictor as OldOpponentBehaviorPredictor
                # Check if the old constructor expects memory_dim=0 or just omits it
                try:
                     model = OldOpponentBehaviorPredictor(
                          input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
                     )
                except TypeError: # If constructor changed (e.g., added memory_dim=0)
                     model = OldOpponentBehaviorPredictor(
                          input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, memory_dim=0
                     )
            except ImportError:
                 print("Error: Could not import old OpponentBehaviorPredictor from src.model.models")
                 raise # Re-raise if import fails

        return model

    @staticmethod
    def load_obp_state_dict(model: BaseOpponentBehaviorPredictor, checkpoint_state: dict):
        # (Keep implementation as before)
        model_state = model.state_dict()
        new_state = {}
        for key in model_state:
            if key in checkpoint_state:
                ckpt_param = checkpoint_state[key]; model_param = model_state[key]
                if ckpt_param.shape == model_param.shape: new_state[key] = ckpt_param
                elif key == "fc1.weight" and ckpt_param.shape[1] < model_param.shape[1]:
                    new_weight = model_param.clone(); new_weight[:, :ckpt_param.shape[1]] = ckpt_param; new_state[key] = new_weight
                else: print(f"Warning: skipping OBP parameter {key} due to shape mismatch"); new_state[key] = model_param
            else: new_state[key] = model_state[key]
        model.load_state_dict(new_state); return model