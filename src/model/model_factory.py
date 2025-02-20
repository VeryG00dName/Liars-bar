import torch
from src import config
from src.model.common_model_api import BasePolicyNetwork, BaseValueNetwork, BaseOpponentBehaviorPredictor

# Import new implementations for policy and value networks.
from src.model.new_models import PolicyNetwork as PPONetwork, ValueNetwork as PPOValueNetwork

class ModelFactory:
    """
    Factory class to create unified model instances for PPO agents and OBP.
    """
    
    @staticmethod
    def create_policy_network(use_aux_classifier: bool = False, num_opponent_classes: int = None,
                              input_dim: int = 26, hidden_dim: int = 64, output_dim: int = config.OUTPUT_DIM,
                              use_lstm: bool = True, use_dropout: bool = True, use_layer_norm: bool = True) -> BasePolicyNetwork:
        model = PPONetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_lstm=use_lstm,
            use_dropout=use_dropout,
            use_layer_norm=use_layer_norm,
            use_aux_classifier=use_aux_classifier,
            num_opponent_classes=num_opponent_classes
        )
        return model

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
