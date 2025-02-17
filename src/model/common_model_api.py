# src/model/common_model_api.py
import abc
import torch.nn as nn

class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for all models.
    Ensures that each model implements a common API.
    """

    @abc.abstractmethod
    def load_checkpoint(self, checkpoint_path: str, device: str):
        """
        Load the model parameters from a checkpoint file.
        """
        pass

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        """
        Perform a forward pass.
        """
        pass

    @abc.abstractmethod
    def get_input_dim(self) -> int:
        """
        Return the expected input dimension for the model.
        """
        pass

    @abc.abstractmethod
    def get_output_dim(self) -> int:
        """
        Return the output dimension for the model.
        """
        pass

# Define common abstract classes for specific model types.

class BasePolicyNetwork(BaseModel):
    """
    Abstract class for policy networks.
    """
    @abc.abstractmethod
    def act(self, observation):
        """
        Given an observation, return an action probability distribution or a sampled action.
        """
        pass

class BaseValueNetwork(BaseModel):
    """
    Abstract class for value networks.
    """
    @abc.abstractmethod
    def evaluate(self, observation):
        """
        Given an observation, return a state value estimate.
        """
        pass

class BaseOpponentBehaviorPredictor(BaseModel):
    """
    Abstract class for opponent behavior predictors.
    """
    @abc.abstractmethod
    def predict(self, opponent_features, memory_embedding=None):
        """
        Given opponent features (and optionally a memory embedding),
        return a prediction (e.g., probabilities of behaviors).
        """
        pass
