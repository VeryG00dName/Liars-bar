# src/model/model_factory.py
import torch
from src import config
from src.model.common_model_api import BasePolicyNetwork, BaseValueNetwork, BaseOpponentBehaviorPredictor
from src.model.shen_models import BeliefSpacePolicy, OpponentBeliefModel
# Import new implementations for policy and value networks.
from src.model.new_models import PolicyNetwork as NewPolicyNetwork, ValueNetwork as PPOValueNetwork
# Import MoE models when needed (dynamic import in create_policy_network)

class ModelFactory:
    # Add these new methods
    
    @staticmethod
    def is_belief_space_policy(state_dict):
        """
        Detects if a state dictionary comes from a BeliefSpacePolicy model.
        
        Args:
            state_dict: The state dictionary of a model.
            
        Returns:
            bool: True if the state dictionary is from a BeliefSpacePolicy model, False otherwise.
        """
        # BeliefSpacePolicy has network, policy_head and value_head components
        network_key = 'network.0.weight'
        policy_key = 'policy_head.weight'
        value_key = 'value_head.0.weight'
        
        return network_key in state_dict and policy_key in state_dict and value_key in state_dict
    
    @staticmethod
    def create_belief_space_policy(belief_dim, obs_dim, hidden_dim, output_dim):
        """
        Creates a BeliefSpacePolicy model.
        
        Args:
            belief_dim (int): Dimension of the belief vector.
            obs_dim (int): Dimension of the observation vector.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of action space.
            
        Returns:
            BeliefSpacePolicy: A belief space policy instance.
        """
        model = BeliefSpacePolicy(
            belief_dim=belief_dim,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        return model
    
    @staticmethod
    def create_opponent_belief_model(obs_dim, num_opponent_types, hidden_dim):
        """
        Creates an OpponentBeliefModel for updating beliefs about opponent types.
        
        Args:
            obs_dim (int): Dimension of the observation vector.
            num_opponent_types (int): Number of opponent types to model.
            hidden_dim (int): Dimension of hidden layers.
            
        Returns:
            OpponentBeliefModel: An opponent belief model instance.
        """
        model = OpponentBeliefModel(
            obs_dim=obs_dim,
            num_opponent_types=num_opponent_types,
            hidden_dim=hidden_dim
        )
        return model
    
    @staticmethod
    def get_belief_input_dim(state_dict):
        """
        Extracts the observation dimension from a BeliefSpacePolicy state dictionary.
        
        Args:
            state_dict: The state dictionary of a BeliefSpacePolicy.
            
        Returns:
            int: The observation dimension.
        """
        total_input_dim = state_dict['network.0.weight'].shape[1]
        hidden_dim = state_dict['network.0.weight'].shape[0]
        
        # In BeliefSpacePolicy, the input is [observation, belief]
        # So the observation dimension is the difference between total input and hidden dim
        return total_input_dim - hidden_dim
    
    @staticmethod
    def get_num_opponent_types(belief_model_state_dict):
        """
        Extracts the number of opponent types from an OpponentBeliefModel state dictionary.
        
        Args:
            belief_model_state_dict: The state dictionary of an OpponentBeliefModel.
            
        Returns:
            int: The number of opponent types.
        """
        if 'belief_update.2.weight' in belief_model_state_dict:
            return belief_model_state_dict['belief_update.2.weight'].shape[0]
        return 10  # Default value if we can't determine it
    
    @staticmethod
    def create_policy_network(use_aux_classifier: bool = False, num_opponent_classes: int = None,
                              input_dim: int = 26, hidden_dim: int = config.HIDDEN_DIM, output_dim: int = config.OUTPUT_DIM,
                              use_lstm: bool = True, use_dropout: bool = True, use_layer_norm: bool = True,
                              use_new_model: bool = True, strategy_dim: int = 5, num_opponents: int = 2,
                              use_moe_model: bool = False, num_experts: int = 10) -> BasePolicyNetwork:
        if use_moe_model:
            # Import and instantiate the MoE model from other_models
            from src.model.other_models import PolicyNetwork as MoEPolicyNetwork
            model = MoEPolicyNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                use_lstm=use_lstm,
                use_dropout=use_dropout,
                use_layer_norm=use_layer_norm,
                num_experts=num_experts
            )
        elif use_new_model:
            # Instantiate the new model version.
            model = NewPolicyNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                use_lstm=use_lstm,
                use_dropout=use_dropout,
                use_layer_norm=use_layer_norm,
                use_aux_classifier=use_aux_classifier,
                num_opponent_classes=num_opponent_classes
            )
        else:
            # Import and instantiate the older version from other_models.
            from src.model.other_models import PolicyNetwork as OtherPolicyNetwork
            model = OtherPolicyNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                strategy_dim=strategy_dim,
                num_opponents=num_opponents,
                use_lstm=use_lstm,
                use_dropout=use_dropout
            )
        return model
    
    @staticmethod
    def create_stacked_observation_model(
        obs_dim: int,
        num_actions: int, 
        hidden_dim: int = config.HIDDEN_DIM,
        num_obs_stack: int = 10
    ) -> torch.nn.Module:
        """
        Creates a StackedObservationConvModel that handles both policy and value functions.
        
        Args:
            obs_dim: Dimension of each observation
            num_actions: Number of possible actions
            hidden_dim: Dimension of hidden layers
            num_obs_stack: Number of historical observations to include in the stack
            
        Returns:
            A StackedObservationConvModel instance
        """
        from src.model.models import StackedObservationConvModel
        model = StackedObservationConvModel(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_obs_stack=num_obs_stack
        )
        return model
    
    @staticmethod
    def create_stacked_newer_observation_model(
        obs_dim: int,
        num_actions: int, 
        hidden_dim: int = config.HIDDEN_DIM,
        num_obs_stack: int = 10,
        num_players: int = 3
    ) -> torch.nn.Module:
        """
        Creates a StackedObservationConvModel that handles policy, value, and game state prediction
        with the newer observation format.
        
        Args:
            obs_dim: Dimension of each observation
            num_actions: Number of possible actions
            hidden_dim: Dimension of hidden layers
            num_obs_stack: Number of historical observations to include in the stack
            num_players: Number of players in the game (for game state dimension calculation)
            
        Returns:
            A StackedObservationConvModel instance with the updated prediction head
        """
        from src.model.models import StackedObservationConvModel
        model = StackedObservationConvModel(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_obs_stack=num_obs_stack,
            num_players=num_players
        )
        return model

    @staticmethod
    def is_stacked_newer_observation_model(state_dict):
        """
        Detects if a state dictionary comes from an updated StackedObservationConvModel
        that uses the newer observation format.
        
        Args:
            state_dict: The state dictionary of a model.
            
        Returns:
            bool: True if the state dictionary is from an updated StackedObservationConvModel, False otherwise.
        """
        # The updated model uses dual heads named policy_head1/2 and value_head1/2 instead of policy_head and value_head
        # And it has a game_state_head instead of next_obs_head
        return (any(k.startswith('conv_layers.') for k in state_dict.keys()) and 
                'policy_head1.weight' in state_dict and 
                'value_head1.weight' in state_dict and
                'policy_head2.weight' in state_dict and
                'value_head2.weight' in state_dict and
                'game_state_head.weight' in state_dict)
    
    @staticmethod
    def is_moe_policy(state_dict):
        """
        Detects if a policy state dictionary comes from a Mixture of Experts model.
        
        Args:
            state_dict: The state dictionary of a policy network.
            
        Returns:
            bool: True if the state dictionary is from an MoE model, False otherwise.
        """
        # MoE models have expert-specific layers like 'experts.0.fc1.weight'
        return any(k.startswith('experts.') for k in state_dict.keys())

    @staticmethod
    def create_value_network(input_dim: int = 26, hidden_dim: int = 64,
                             use_dropout: bool = True, use_layer_norm: bool = True) -> BaseValueNetwork:
        model = PPOValueNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_dropout=use_dropout,
            use_layer_norm=use_layer_norm
        )
        return model

    @staticmethod
    def create_obp(use_transformer_memory: bool = True, 
                   input_dim: int = None, hidden_dim: int = 64, output_dim: int = 2) -> BaseOpponentBehaviorPredictor:
        if input_dim is None:
            input_dim = config.OPPONENT_INPUT_DIM
        if use_transformer_memory:
            # Use the new OBP which requires memory integration.
            from src.model.new_models import OpponentBehaviorPredictor as NewOpponentBehaviorPredictor
            model = NewOpponentBehaviorPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                memory_dim=config.STRATEGY_DIM
            )
        else:
            # Use the old OBP from src.model.models that doesn't require memory.
            from src.model.models import OpponentBehaviorPredictor as OldOpponentBehaviorPredictor
            model = OldOpponentBehaviorPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
        return model

    @staticmethod
    def load_obp_state_dict(model: BaseOpponentBehaviorPredictor, checkpoint_state: dict):
        """
        Loads the checkpoint state into the OBP model.
        If the checkpoint's fc1.weight has a smaller second dimension than the model,
        then copy the overlapping columns and leave the rest as initialized.
        """
        model_state = model.state_dict()
        new_state = {}
        for key in model_state:
            if key in checkpoint_state:
                ckpt_param = checkpoint_state[key]
                model_param = model_state[key]
                if ckpt_param.shape == model_param.shape:
                    new_state[key] = ckpt_param
                elif key == "fc1.weight" and ckpt_param.shape[1] < model_param.shape[1]:
                    new_weight = model_param.clone()
                    new_weight[:, :ckpt_param.shape[1]] = ckpt_param
                    new_state[key] = new_weight
                else:
                    print(f"Warning: skipping parameter {key} due to shape mismatch: "
                          f"checkpoint {ckpt_param.shape} vs model {model_param.shape}")
                    new_state[key] = model_param
            else:
                new_state[key] = model_state[key]
        model.load_state_dict(new_state)
        return model
