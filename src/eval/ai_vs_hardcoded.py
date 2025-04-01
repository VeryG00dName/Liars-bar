# src/evaluation/ai_vs_hardcoded.py
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import re
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from src.model.shen_models import BeliefSpacePolicy, OpponentBeliefModel
# Import your environment, agents, evaluation helpers, and model factory.
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)
from src.eval.evaluate_utils import (
    get_hidden_dim_from_state_dict,
    evaluate_agents
)
from src.model.model_factory import ModelFactory
from src.training.train_vs_everyone import load_specific_historical_models

from src.misc.cheat.ai_vs_hardcoded_cheat import LABELS
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("AgentBattleground")
logger.propagate = True
# --- Helper function (unchanged) ---

def get_input_dim_from_state_dict(state_dict, candidate_prefix='fc1'):
    """
    Attempts to determine the input dimension from a policy state dictionary.
    It searches for candidate keys (e.g. "fc1.weight", "base_encoder.0.weight", etc.)
    and returns the second dimension of the first matching weight tensor.
    """
    candidate_prefixes = [
        candidate_prefix,
        "base_encoder.0",
        "policy_net.fc1",
        "model.fc1"
    ]
    for prefix in candidate_prefixes:
        key = f"{prefix}.weight"
        if key in state_dict:
            return state_dict[key].shape[1]
    # Fallback: iterate over all keys and return the input dimension from the first 2D tensor found.
    for key, tensor in state_dict.items():
        if hasattr(tensor, "ndim") and tensor.ndim == 2:
            return tensor.shape[1]
    available_keys = list(state_dict.keys())
    raise ValueError(f"Cannot determine input_dim from state_dict. Tried prefixes: {candidate_prefixes}. "
                     f"Available keys: {available_keys}")

# --- New helper: Detect if a state dict is from StackedObservationConvModel ---
def is_stacked_observation_model(state_dict):
    """
    Detects if a state dictionary comes from a StackedObservationConvModel.
    
    Args:
        state_dict: The state dictionary of a model.
        
    Returns:
        bool: True if the state dictionary is from a StackedObservationConvModel, False otherwise.
    """
    # StackedObservationConvModel will have conv_layers and both policy_head and value_head
    return (any(k.startswith('conv_layers.') for k in state_dict.keys()) and 
            'policy_head.weight' in state_dict and 
            'value_head.weight' in state_dict)

def is_stacked_newer_observation_model(state_dict):
    """
    Detects if a state dictionary comes from a StackedObservationConvModel.
    
    Args:
        state_dict: The state dictionary of a model.
        
    Returns:
        bool: True if the state dictionary is from a StackedObservationConvModel, False otherwise.
    """
    # StackedObservationConvModel will have conv_layers and both policy_head and value_head
    return (any(k.startswith('conv_layers.') for k in state_dict.keys()) and 
            'policy_head1.weight' in state_dict and 
            'value_head1.weight' in state_dict)

# --- New helper: Get observation dimension from StackedObservationConvModel state dict ---
def get_obs_dim_from_stacked_model(state_dict):
    """
    Extracts the observation dimension from a StackedObservationConvModel state dictionary.
    
    Args:
        state_dict: The state dictionary of a StackedObservationConvModel.
        
    Returns:
        int: The observation dimension.
    """
    # Try to find the first convolutional layer
    for key in state_dict.keys():
        if key.endswith('weight') and 'conv_layers' in key:
            # Conv1d weight shape is (out_channels, in_channels, kernel_size)
            # For the first conv layer, in_channels is the observation dimension
            shape = state_dict[key].shape
            if len(shape) == 3:  # Conv1d weight
                return shape[2]  # kernel_size dimension gives the observation dimension
    
    # Fallback
    return None

# --- Custom QListWidget for drag-and-drop ---
class DropListWidget(QtWidgets.QListWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.main_window.on_file_drop(file_path)
            event.acceptProposedAction()
        else:
            event.ignore()

# --- Helper: Detect if a policy state dict comes from new_models ---
def is_new_policy(state_dict):
    if "fc_classifier.weight" in state_dict:
        return True
    elif "strategy_query.weight" in state_dict:
        return False
    else:
        # Default to new model if unclear.
        return True

# --- Worker thread to run the battleground matches ---
class BattlegroundWorker(QThread):
    progress_signal = pyqtSignal(int)
    results_signal = pyqtSignal(dict)
    expert_signal = pyqtSignal(dict)  # New signal for expert activations
    error_signal = pyqtSignal(str)
    
    # Now include extra parameters "onev2" and "duo"
    def __init__(self, ai_agents, historical_models, hardcoded_agents, rounds, two_player=None, parent=None, cheat=False, onev2=False, duo=False):
        super().__init__(parent)
        self.ai_agents = ai_agents
        self.historical_models = historical_models
        self.hardcoded_agents = hardcoded_agents
        self.rounds = rounds
        self.two_player = two_player  # if not None, pass the player id to eliminate
        self.cheat = cheat
        self.onev2 = onev2      # if True, run 1v2 mode (one AI against the same opponent in both slots)
        self.duo = duo          # if True, run duo mode (one AI against a pair of different opponents)
        self.expert_activations = {}
    
    def run(self):
        combined_opponents = {}
        # Add hardcoded agents.
        for name, cls in self.hardcoded_agents.items():
            combined_opponents[name] = ("hardcoded", cls)
        # Then add historical models.
        for identifier, hist_model in self.historical_models.items():
            combined_opponents[identifier] = ("historical", hist_model)
        
        progress_counter = 0
        results = {}
        # Initialize expert activation tracking
        self.expert_activations = {}
        
        # Duo mode: iterate over every pair of opponents.
        if self.duo:
            duo_pairs = list(itertools.combinations(combined_opponents.items(), 2))
            for ((opp_name1, (opp_type1, opp_obj1)), (opp_name2, (opp_type2, opp_obj2))) in duo_pairs:
                opponent_name = f"{opp_name1}+{opp_name2}"
                if self.cheat:
                    cheat_index1 = LABELS.get(opp_name1, None)
                    cheat_index2 = LABELS.get(opp_name2, None)
                    cheat_expert_index = (cheat_index1, cheat_index2)
                else:
                    cheat_expert_index = None
                # Pass a tuple of opponent types and a tuple of opponent objects.
                cumulative_wins, expert_acts = self.run_match(
                    (opp_type1, opp_type2), 
                    (opp_obj1, opp_obj2), 
                    opponent_name, 
                    episodes=self.rounds, 
                    progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep), 
                    cheat_expert_index=cheat_expert_index
                )
                wins = [
                    cumulative_wins.get("player_0", 0),
                    cumulative_wins.get("player_1", 0),
                    cumulative_wins.get("player_2", 0)
                ]
                self.expert_activations[opponent_name] = expert_acts
                progress_counter += self.rounds
                self.progress_signal.emit(progress_counter)
                results[opponent_name] = wins
        else:
            # Normal and onev2 modes.
            for opp_name, (opp_type, opp_obj) in combined_opponents.items():
                if self.onev2:
                    if self.cheat:
                        cheat_index1 = LABELS.get(opp_name, None)
                        cheat_expert_index = (cheat_index1, cheat_index1)
                    else:
                        cheat_expert_index = None
                    cumulative_wins, expert_acts = self.run_match(opp_type, opp_obj, opp_name, episodes=self.rounds, progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep), cheat_expert_index=cheat_expert_index)
                else:
                    if self.cheat:
                        cheat_expert_index = LABELS.get(opp_name, None)
                    else:
                        cheat_expert_index = None
                    cumulative_wins, expert_acts = self.run_match(opp_type, opp_obj, opp_name, episodes=self.rounds, progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep), cheat_expert_index=cheat_expert_index)
                wins = [
                    cumulative_wins.get("player_0", 0),
                    cumulative_wins.get("player_1", 0),
                    cumulative_wins.get("player_2", 0)
                ]
                self.expert_activations[opp_name] = expert_acts
                progress_counter += self.rounds
                self.progress_signal.emit(progress_counter)
                results[opp_name] = wins
        
        # Emit the expert activations and the results
        self.expert_signal.emit(self.expert_activations)
        self.results_signal.emit(results)
    
    def run_match(self, opponent_type, opponent_obj, opponent_name, episodes, progress_callback=None, cheat_expert_index=None): 
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        players_in_this_game = {}
        logger = logging.getLogger("BattlegroundWorker")
        
        # --- Set up AI agent(s) and opponent(s) based on mode ---
        if isinstance(opponent_type, (tuple, list)):
            # Duo mode: Use a single AI agent in player_0, and two different opponents for player_1 and player_2.
            # Load AI agent from self.ai_agents["player_0"]
            agent_data = self.ai_agents["player_0"]
            policy_state_dict = agent_data["policy_net"]
            if agent_data.get("is_belief_space_policy", False):
                try:
                    # Get dimensions from the model's state dict
                    if 'network.0.weight' in policy_state_dict:
                        hidden_dim = policy_state_dict['network.0.weight'].shape[0]
                        total_input_dim = policy_state_dict['network.0.weight'].shape[1]
                    else:
                        # Try fallback methods
                        hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, "network.0")
                        if 'network.0.bias' in policy_state_dict:
                            hidden_dim = policy_state_dict['network.0.bias'].shape[0]

                        # Try to find any linear layer to infer dimensions
                        for net_key, tensor in policy_state_dict.items():
                            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and '.weight' in net_key and 'network' in net_key:
                                total_input_dim = tensor.shape[1]
                                break
                        else:
                            # If still not found, use ModelFactory to estimate dimensions
                            dimensions = ModelFactory.get_belief_dimensions(policy_state_dict)
                            if dimensions[0] is not None:
                                total_input_dim, obs_dim, belief_dim = dimensions
                            else:
                                # Last resort fallback
                                total_input_dim = 29  # Default if we can't determine
                    
                    if total_input_dim is not None:
                        # Compute a better estimate of observation and belief dimensions
                        
                        # First check if the model has configuration attributes saved
                        if hasattr(policy_state_dict, 'get') and policy_state_dict.get('total_input_dim') is not None:
                            total_input_dim = policy_state_dict['total_input_dim'].item()
                        if hasattr(policy_state_dict, 'get') and policy_state_dict.get('obs_dim') is not None:
                            obs_dim = policy_state_dict['obs_dim'].item()
                            belief_dim = total_input_dim - obs_dim
                        else:
                            key = "player_0"
                            # Make a better estimate by analyzing the observation
                            sample_obs = env.observe(key, new=True)[key]
                            estimated_obs_dim = min(sample_obs.shape[0], total_input_dim - 10)  # Ensure at least 10 for belief
                            
                            # For display and initialization
                            obs_dim = estimated_obs_dim
                            belief_dim = total_input_dim - obs_dim
                        
                        # Create BeliefSpacePolicy with the exact dimensions from the checkpoint
                        policy_net = BeliefSpacePolicy(
                            belief_dim=belief_dim,
                            obs_dim=obs_dim,
                            hidden_dim=hidden_dim,
                            output_dim=env.action_spaces[key].n
                        )
                        
                        # Load state dict with better error handling
                        try:
                            policy_net.load_state_dict(policy_state_dict, strict=False)
                            
                            # Check for NaN/Inf in loaded weights and fix them
                            has_invalid_params = False
                            for name, param in policy_net.named_parameters():
                                if torch.isnan(param).any() or torch.isinf(param).any():
                                    logger.warning(f"NaN/Inf found in parameter {name}, fixing...")
                                    has_invalid_params = True
                                    # Zero out the problematic values
                                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            if has_invalid_params:
                                logger.info("Fixed NaN/Inf values in model parameters")
                                
                        except Exception as e:
                            logger.warning(f"Error loading state dict: {str(e)}. Attempting to adjust...")
                            # Try to match parameter shapes
                            mismatched_keys = []
                            for name, param in policy_net.named_parameters():
                                if name in policy_state_dict:
                                    checkpoint_param = policy_state_dict[name]
                                    if param.shape != checkpoint_param.shape:
                                        mismatched_keys.append(name)
                                        logger.warning(f"Shape mismatch for {name}: model {param.shape} vs checkpoint {checkpoint_param.shape}")
                            
                            # If there are mismatches, try to adjust the state dict
                            if mismatched_keys:
                                adjusted_state_dict = {}
                                for name, param in policy_state_dict.items():
                                    if name in mismatched_keys:
                                        # Skip mismatched keys
                                        continue
                                    adjusted_state_dict[name] = param
                                
                                # Load the adjusted state dict
                                policy_net.load_state_dict(adjusted_state_dict, strict=False)
                                logger.info("Loaded adjusted state dict with skipped mismatched keys")
                        
                        policy_net.to(device).eval()
                        
                        # Load OpponentBeliefModel if available with similar error handling
                        belief_model_state = agent_data["belief_model"]
                        belief_model = None
                        
                        if belief_model_state is not None:
                            try:
                                # Get belief model dimensions with fallbacks
                                if 'encoder.0.weight' in belief_model_state:
                                    belief_hidden_dim = belief_model_state['encoder.0.weight'].shape[0]
                                    belief_obs_dim = belief_model_state['encoder.0.weight'].shape[1]
                                else:
                                    belief_hidden_dim = get_hidden_dim_from_state_dict(belief_model_state, "encoder.0")
                                    belief_obs_dim = obs_dim  # Use the same as policy network
                                
                                # Get num_opponent_types from final layer
                                if 'belief_update.2.weight' in belief_model_state:
                                    num_opponent_types = belief_model_state['belief_update.2.weight'].shape[0]
                                else:
                                    num_opponent_types = agent_data.get('num_opponent_types', 10)
                                
                                # Create and load belief model
                                belief_model = OpponentBeliefModel(
                                    event_feature_dim=5,
                                    max_seq_length=config.MAX_SQUENCE_LENGTH,
                                    hidden_dim=config.HIDDEN_DIM // 4,
                                    num_opponent_types=10
                                ).to(device)
                                # Load state dict with error handling
                                belief_model.load_state_dict(belief_model_state, strict=True)
                                
                                # Check for NaN/Inf values
                                for name, param in belief_model.named_parameters():
                                    if torch.isnan(param).any() or torch.isinf(param).any():
                                        logger.warning(f"NaN/Inf found in belief model parameter {name}, fixing...")
                                        param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
                                    
                                belief_model.to(device).eval()
                            except Exception as e:
                                logger.warning(f"Failed to create belief model: {str(e)}")
                                belief_model = None
                        
                        players_in_this_game[key] = {
                            "policy_net": policy_net,
                            "belief_model": belief_model,
                            "obs_version": 2,  # New observation format
                            "rating": None,
                            "uses_memory": False,
                            "track_experts": False,
                            "is_stacked_model": False,
                            "is_newer_obs_model": False,
                            "is_belief_space_policy": True,
                            "num_opponent_types": num_opponent_types if 'num_opponent_types' in locals() else 10
                        }
                    else:
                        raise ValueError("Could not determine total input dimension for BeliefSpacePolicy")
                except Exception as e:
                    logger.error(f"Failed to create BeliefSpacePolicy: {str(e)}")
                    raise
            else:
                # Otherwise, use the traditional model initialization.
                hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, "fc1")
                obs_dim = agent_data["input_dim"]
                policy_net = ModelFactory.create_policy_network(
                    input_dim=obs_dim,
                    hidden_dim=hidden_dim,
                    output_dim=env.action_spaces["player_0"].n,
                    use_new_model=True
                )
                policy_net.load_state_dict(policy_state_dict, strict=False)
                policy_net.to(device).eval()
                belief_model = None
            players_in_this_game["player_0"] = {
                "policy_net": policy_net,
                "belief_model": belief_model,
                "obs_version": 2,
                "rating": None,
                "uses_memory": False,
                "track_experts": False,
                "is_belief_space_policy": agent_data.get("is_belief_space_policy", False)
            }
            # Now create opponent players for player_1 and player_2 from the duo pair.
            for idx, (opp_type, opp_obj) in enumerate(zip(opponent_type, opponent_obj), start=1):
                key = f"player_{idx}"
                if opp_type == "hardcoded":
                    opponent_instance = opp_obj(opponent_name)
                    players_in_this_game[key] = {
                        "hardcoded_bot": True,
                        "agent": opponent_instance,
                        "obs_version": 2,
                        "rating": None,
                        "uses_memory": False
                    }
                elif opp_type == "historical":
                    hist_state_dict = opp_obj.state_dict()
                    hidden_dim = get_hidden_dim_from_state_dict(hist_state_dict, "fc1")
                    obs_dim = hist_state_dict["fc1.weight"].shape[1]
                    policy_net = ModelFactory.create_policy_network(
                        input_dim=obs_dim,
                        hidden_dim=hidden_dim,
                        output_dim=env.action_spaces[key].n,
                        use_new_model=True
                    )
                    policy_net.load_state_dict(hist_state_dict, strict=False)
                    policy_net.to(device).eval()
                    obp_model = ModelFactory.create_obp(
                        use_transformer_memory=True,
                        input_dim=config.OPPONENT_INPUT_DIM,
                        hidden_dim=config.OPPONENT_HIDDEN_DIM,
                        output_dim=2
                    )
                    obp_model.to(device).eval()
                    players_in_this_game[key] = {
                        "policy_net": policy_net,
                        "obp_model": obp_model,
                        "obs_version": 2,
                        "rating": None,
                        "uses_memory": True
                    }
                else:
                    raise ValueError(f"Unsupported opponent type in duo mode: {opp_type}")
        elif self.onev2:
            # onev2 mode: Use a single AI agent for player_0 and assign the same opponent to both player_1 and player_2.
            agent_data = self.ai_agents["player_0"]
            policy_state_dict = agent_data["policy_net"]
            if agent_data.get("is_belief_space_policy", False):
                hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, "network.0")
                total_input_dim = policy_state_dict['network.0.weight'].shape[1]
                sample_obs = env.observe("player_0", new=True)["player_0"]
                obs_dim = min(sample_obs.shape[0], total_input_dim - 10)
                belief_dim = total_input_dim - obs_dim
                policy_net = BeliefSpacePolicy(
                    belief_dim=belief_dim,
                    obs_dim=obs_dim,
                    hidden_dim=hidden_dim,
                    output_dim=env.action_spaces["player_0"].n
                )
                policy_net.load_state_dict(policy_state_dict, strict=False)
                policy_net.to(device).eval()
                belief_model_state = agent_data["belief_model"]
                if belief_model_state is not None:
                    # Determine dimensions (example values; adjust based on your requirements)
                    belief_hidden_dim = get_hidden_dim_from_state_dict(belief_model_state, "encoder.0")
                    belief_obs_dim = belief_model_state["encoder.0.weight"].shape[1] if "encoder.0.weight" in belief_model_state else obs_dim
                    num_opponent_types = 10  # or retrieve from agent_data if available

                    # Instantiate the belief model.
                    belief_model = OpponentBeliefModel(
                        event_feature_dim=5,
                        max_seq_length=config.MAX_SQUENCE_LENGTH,
                        hidden_dim=belief_hidden_dim,
                        num_opponent_types=num_opponent_types
                    ).to(device)
                    # Load the parameters.
                    belief_model.load_state_dict(belief_model_state, strict=True)
                else:
                    belief_model = None
            else:
                hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, "fc1")
                obs_dim = agent_data["input_dim"]
                policy_net = ModelFactory.create_policy_network(
                    input_dim=obs_dim,
                    hidden_dim=hidden_dim,
                    output_dim=env.action_spaces["player_0"].n,
                    use_new_model=True
                )
                policy_net.load_state_dict(policy_state_dict, strict=False)
                policy_net.to(device).eval()
                belief_model = None
            players_in_this_game["player_0"] = {
                "policy_net": policy_net,
                "belief_model": belief_model,
                "obs_version": 2,
                "rating": None,
                "uses_memory": False,
                "track_experts": False,
                "is_belief_space_policy": agent_data.get("is_belief_space_policy", False)
            }
            # For onev2, assign the same opponent to both player_1 and player_2.
            if opponent_type == "hardcoded":
                opponent_instance = opponent_obj(opponent_name)
                for key in ["player_1", "player_2"]:
                    players_in_this_game[key] = {
                        "hardcoded_bot": True,
                        "agent": opponent_instance,
                        "obs_version": 2,
                        "rating": None,
                        "uses_memory": False
                    }
            elif opponent_type == "historical":
                hist_state_dict = opponent_obj.state_dict()
                hidden_dim = get_hidden_dim_from_state_dict(hist_state_dict, "fc1")
                obs_dim = hist_state_dict["fc1.weight"].shape[1]
                policy_net = ModelFactory.create_policy_network(
                    input_dim=obs_dim,
                    hidden_dim=hidden_dim,
                    output_dim=env.action_spaces["player_2"].n,
                    use_new_model=True
                )
                policy_net.load_state_dict(hist_state_dict, strict=False)
                policy_net.to(device).eval()
                obp_model = ModelFactory.create_obp(
                    use_transformer_memory=True,
                    input_dim=config.OPPONENT_INPUT_DIM,
                    hidden_dim=config.OPPONENT_HIDDEN_DIM,
                    output_dim=2
                )
                obp_model.to(device).eval()
                for key in ["player_1", "player_2"]:
                    players_in_this_game[key] = {
                        "policy_net": policy_net,
                        "obp_model": obp_model,
                        "obs_version": 2,
                        "rating": None,
                        "uses_memory": True
                    }
            else:
                raise ValueError(f"Unknown opponent type in onev2 mode: {opponent_type}")
        else:
            # Normal mode: Load AI agents for player_0 and player_1, and assign opponent to player_2.
            for key in ["player_0", "player_1"]:
                agent_data = self.ai_agents[key]
                policy_state_dict = agent_data["policy_net"]
                if agent_data.get("is_belief_space_policy", False):
                    try:
                        if 'network.0.weight' in policy_state_dict:
                            hidden_dim = policy_state_dict['network.0.weight'].shape[0]
                            total_input_dim = policy_state_dict['network.0.weight'].shape[1]
                        else:
                            hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, "network.0")
                            if 'network.0.bias' in policy_state_dict:
                                hidden_dim = policy_state_dict['network.0.bias'].shape[0]
                            for net_key, tensor in policy_state_dict.items():
                                if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and '.weight' in net_key and 'network' in net_key:
                                    total_input_dim = tensor.shape[1]
                                    break
                            else:
                                dimensions = ModelFactory.get_belief_dimensions(policy_state_dict)
                                if dimensions[0] is not None:
                                    total_input_dim, obs_dim, belief_dim = dimensions
                                else:
                                    total_input_dim = 29
                        if total_input_dim is not None:
                            if hasattr(policy_state_dict, 'get') and policy_state_dict.get('total_input_dim') is not None:
                                total_input_dim = policy_state_dict['total_input_dim'].item()
                            if hasattr(policy_state_dict, 'get') and policy_state_dict.get('obs_dim') is not None:
                                obs_dim = policy_state_dict['obs_dim'].item()
                                belief_dim = total_input_dim - obs_dim
                            else:
                                sample_obs = env.observe(key, new=True)[key]
                                estimated_obs_dim = min(sample_obs.shape[0], total_input_dim - 10)
                                obs_dim = estimated_obs_dim
                                belief_dim = total_input_dim - obs_dim
                            policy_net = BeliefSpacePolicy(
                                belief_dim=belief_dim,
                                obs_dim=obs_dim,
                                hidden_dim=hidden_dim,
                                output_dim=env.action_spaces[key].n
                            )
                            try:
                                policy_net.load_state_dict(policy_state_dict, strict=False)
                                has_invalid_params = False
                                for name, param in policy_net.named_parameters():
                                    if torch.isnan(param).any() or torch.isinf(param).any():
                                        logger.warning(f"NaN/Inf found in parameter {name}, fixing...")
                                        has_invalid_params = True
                                        param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
                                if has_invalid_params:
                                    logger.info("Fixed NaN/Inf values in model parameters")
                            except Exception as e:
                                logger.warning(f"Error loading state dict: {str(e)}. Attempting to adjust...")
                                mismatched_keys = []
                                for name, param in policy_net.named_parameters():
                                    if name in policy_state_dict:
                                        checkpoint_param = policy_state_dict[name]
                                        if param.shape != checkpoint_param.shape:
                                            mismatched_keys.append(name)
                                            logger.warning(f"Shape mismatch for {name}: model {param.shape} vs checkpoint {checkpoint_param.shape}")
                                if mismatched_keys:
                                    adjusted_state_dict = {}
                                    for name, param in policy_state_dict.items():
                                        if name in mismatched_keys:
                                            continue
                                        adjusted_state_dict[name] = param
                                    policy_net.load_state_dict(adjusted_state_dict, strict=False)
                                    logger.info("Loaded adjusted state dict with skipped mismatched keys")
                            policy_net.to(device).eval()
                            belief_model_state = agent_data["belief_model"]
                            if belief_model_state is not None:
                                # Determine dimensions (example values; adjust based on your requirements)
                                belief_hidden_dim = get_hidden_dim_from_state_dict(belief_model_state, "encoder.0")
                                belief_obs_dim = belief_model_state["encoder.0.weight"].shape[1] if "encoder.0.weight" in belief_model_state else obs_dim
                                num_opponent_types = 10  # or retrieve from agent_data if available

                                # Instantiate the belief model.
                                belief_model = OpponentBeliefModel(
                                    event_feature_dim=5,
                                    max_seq_length=config.MAX_SQUENCE_LENGTH,
                                    hidden_dim=belief_hidden_dim,
                                    num_opponent_types=num_opponent_types
                                ).to(device)
                                # Load the parameters.
                                belief_model.load_state_dict(belief_model_state, strict=True)
                            else:
                                belief_model = None
                        else:
                            raise ValueError("Could not determine total input dimension for BeliefSpacePolicy")
                    except Exception as e:
                        logger.error(f"Failed to create BeliefSpacePolicy: {str(e)}")
                        raise
                    players_in_this_game[key] = {
                        "policy_net": policy_net,
                        "belief_model": belief_model,
                        "obs_version": 2,
                        "rating": None,
                        "uses_memory": False,
                        "track_experts": False,
                        "is_stacked_model": False,
                        "is_newer_obs_model": False,
                        "is_belief_space_policy": True,
                        "num_opponent_types": agent_data.get('num_opponent_types', 10)
                    }
                elif agent_data.get("is_newer_obs_model", False):
                    obs_dim = agent_data.get("input_dim")
                    if obs_dim is None:
                        obs_dim = get_obs_dim_from_stacked_model(policy_state_dict)
                        if obs_dim is None:
                            obs = env.observe(key, newer=True)[key]
                            obs_dim = obs.shape[0]
                    hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, "fc_layers.0")
                    if hidden_dim is None:
                        hidden_dim = config.HIDDEN_DIM
                    policy_net = ModelFactory.create_stacked_observation_model(
                        obs_dim=obs_dim,
                        num_actions=env.action_spaces[key].n,
                        hidden_dim=hidden_dim,
                        num_obs_stack=config.NUM_OBS_STACK
                    )
                    policy_net.load_state_dict(policy_state_dict, strict=False)
                    policy_net.to(device).eval()
                    players_in_this_game[key] = {
                        "policy_net": policy_net,
                        "obp_model": obp_model,
                        "obs_version": 5,
                        "rating": None,
                        "uses_memory": False,
                        "track_experts": False,
                        "is_stacked_model": True,
                        "is_newer_obs_model": True,
                        "is_conditional_model": False,
                        "is_belief_space_policy": False,
                        "observation_stacks": deque(maxlen=config.NUM_OBS_STACK)
                    }
                    sample_obs = env.observe(key, newer=True)[key]
                    for _ in range(config.NUM_OBS_STACK):
                        players_in_this_game[key]["observation_stacks"].append(np.zeros_like(sample_obs))
                elif agent_data.get("is_stacked_model", False):
                    obs_dim = agent_data.get("input_dim")
                    if obs_dim is None:
                        obs_dim = get_obs_dim_from_stacked_model(policy_state_dict)
                        if obs_dim is None:
                            obs = env.observe(key, new=True)[key]
                            obs_dim = obs.shape[0]
                    hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, "fc_layers.0")
                    if hidden_dim is None:
                        hidden_dim = config.HIDDEN_DIM
                    policy_net = ModelFactory.create_stacked_observation_model(
                        obs_dim=obs_dim,
                        num_actions=env.action_spaces[key].n,
                        hidden_dim=hidden_dim,
                        num_obs_stack=config.NUM_OBS_STACK
                    )
                    policy_net.load_state_dict(policy_state_dict, strict=False)
                    policy_net.to(device).eval()
                    players_in_this_game[key] = {
                        "policy_net": policy_net,
                        "obp_model": None,
                        "obs_version": 3,
                        "rating": None,
                        "uses_memory": False,
                        "track_experts": False,
                        "is_stacked_model": True,
                        "is_newer_obs_model": False,
                        "is_conditional_model": False,
                        "is_belief_space_policy": False,
                        "observation_stacks": deque(maxlen=config.NUM_OBS_STACK)
                    }
                    sample_obs = env.observe(key, new=True)[key]
                    for _ in range(config.NUM_OBS_STACK):
                        players_in_this_game[key]["observation_stacks"].append(np.zeros_like(sample_obs))
                else:
                    hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, "fc1")
                    obs_dim = agent_data["input_dim"]
                    is_moe_model = ModelFactory.is_moe_policy(policy_state_dict)
                    new_model_flag = is_new_policy(policy_state_dict)
                    if is_moe_model:
                        policy_net = ModelFactory.create_policy_network(
                            input_dim=obs_dim,
                            hidden_dim=hidden_dim,
                            output_dim=env.action_spaces[key].n,
                            use_aux_classifier=True,
                            num_opponent_classes=config.NUM_OPPONENT_CLASSES,
                            use_moe_model=True,
                            num_experts=10
                        )
                    elif new_model_flag:
                        policy_net = ModelFactory.create_policy_network(
                            input_dim=obs_dim,
                            hidden_dim=hidden_dim,
                            output_dim=env.action_spaces[key].n,
                            use_aux_classifier=True,
                            num_opponent_classes=config.NUM_OPPONENT_CLASSES,
                            use_new_model=True
                        )
                    else:
                        policy_net = ModelFactory.create_policy_network(
                            input_dim=obs_dim,
                            hidden_dim=hidden_dim,
                            output_dim=env.action_spaces[key].n,
                            use_new_model=False,
                            strategy_dim=config.STRATEGY_DIM,
                            num_opponents=env.num_players - 1
                        )
                    policy_net.load_state_dict(policy_state_dict, strict=False)
                    policy_net.to(device).eval()
                    obp_state = agent_data["obp_model"]
                    if obp_state is not None:
                        obp_hidden_dim = get_hidden_dim_from_state_dict(obp_state, "fc1")
                        obp_input_dim = obp_state["fc1.weight"].shape[1]
                        if obp_input_dim == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
                            obp_model = ModelFactory.create_obp(
                                use_transformer_memory=True,
                                input_dim=config.OPPONENT_INPUT_DIM,
                                hidden_dim=obp_hidden_dim,
                                output_dim=2
                            )
                        elif obp_input_dim == config.OPPONENT_INPUT_DIM:
                            obp_model = ModelFactory.create_obp(
                                use_transformer_memory=False,
                                input_dim=config.OPPONENT_INPUT_DIM,
                                hidden_dim=obp_hidden_dim,
                                output_dim=2
                            )
                        else:
                            raise ValueError(f"Unexpected OBP input dimension: {obp_input_dim}")
                        obp_model = ModelFactory.load_obp_state_dict(obp_model, obp_state)
                        obp_model.to(device).eval()
                        example_observation = torch.randn(1, config.OPPONENT_INPUT_DIM).to(device)
                        example_memory_embedding = torch.randn(1, config.STRATEGY_DIM).to(device)
                        obp_model = torch.jit.trace(obp_model, (example_observation, example_memory_embedding))
                    else:
                        obp_model = None
                    players_in_this_game[key] = {
                        "policy_net": policy_net,
                        "obp_model": obp_model,
                        "obs_version": agent_data["obs_version"],
                        "rating": None,
                        "uses_memory": agent_data["uses_memory"],
                        "track_experts": True,
                        "is_stacked_model": False,
                        "is_newer_obs_model": False,
                        "is_belief_space_policy": False,
                        "is_conditional_model": False
                    }
            # --- Opponent as player_2 ---
            if opponent_type == "hardcoded":
                opponent_instance = opponent_obj(opponent_name)
                players_in_this_game["player_2"] = {
                    "hardcoded_bot": True,
                    "agent": opponent_instance,
                    "obs_version": 2,
                    "rating": None,
                    "uses_memory": False
                }
            elif opponent_type == "historical":
                hist_state_dict = opponent_obj.state_dict()
                hidden_dim = get_hidden_dim_from_state_dict(hist_state_dict, "fc1")
                obs_dim = hist_state_dict["fc1.weight"].shape[1]
                is_moe_model = ModelFactory.is_moe_policy(hist_state_dict)
                new_model_flag = is_new_policy(hist_state_dict)
                if is_moe_model:
                    policy_net = ModelFactory.create_policy_network(
                        input_dim=obs_dim,
                        hidden_dim=hidden_dim,
                        output_dim=env.action_spaces["player_2"].n,
                        use_aux_classifier=True,
                        num_opponent_classes=config.NUM_OPPONENT_CLASSES,
                        use_moe_model=True,
                        num_experts=10
                    )
                elif new_model_flag:
                    policy_net = ModelFactory.create_policy_network(
                        input_dim=obs_dim,
                        hidden_dim=hidden_dim,
                        output_dim=env.action_spaces["player_2"].n,
                        use_aux_classifier=True,
                        num_opponent_classes=config.NUM_OPPONENT_CLASSES,
                        use_new_model=True
                    )
                else:
                    policy_net = ModelFactory.create_policy_network(
                        input_dim=obs_dim,
                        hidden_dim=hidden_dim,
                        output_dim=env.action_spaces["player_2"].n,
                        use_new_model=False,
                        strategy_dim=config.STRATEGY_DIM,
                        num_opponents=env.num_players - 1
                    )
                policy_net.load_state_dict(hist_state_dict, strict=False)
                policy_net.to(device).eval()
                obp_model = ModelFactory.create_obp(
                    use_transformer_memory=True,
                    input_dim=config.OPPONENT_INPUT_DIM,
                    hidden_dim=config.OPPONENT_HIDDEN_DIM,
                    output_dim=2
                )
                obp_model.to(device).eval()
                players_in_this_game["player_2"] = {
                    "policy_net": policy_net,
                    "obp_model": obp_model,
                    "obs_version": 2,
                    "rating": None,
                    "uses_memory": True
                }
            else:
                raise ValueError(f"Unknown opponent type: {opponent_type}")
        
        # When calling evaluate_agents, pass the cheat_expert_index (which may be scalar or a tuple).
        cumulative_wins, _, _, _, _, expert_activations = evaluate_agents(
            env, device, players_in_this_game, episodes=episodes, 
            two_player=self.two_player, track_experts=True,
            progress_callback=progress_callback,
            cheat_expert_index=cheat_expert_index
        )
        return cumulative_wins, expert_activations

# --- Main GUI class using PyQt with a Discord-like style ---
class AgentBattlegroundGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent Battleground")
        self.resize(1000, 700)
        self.loaded_models = {}
        self.hardcoded_agents = {
            "Classic": Classic,
            "GreedyCardSpammer": GreedyCardSpammer,
            "RandomAgent": RandomAgent,
            "SelectiveTableConservativeChallenger": lambda name: SelectiveTableConservativeChallenger(name),
            "StrategicChallenger": lambda name: StrategicChallenger(name, 3, 2),
            "TableFirstConservativeChallenger": TableFirstConservativeChallenger,
            "TableNonTableAgent": TableNonTableAgent
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.historical_models = {}
        hist_models_list = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
        for model, identifier in hist_models_list:
            self.historical_models[identifier] = model

        # Store previous results for comparison.
        self.previous_results = None
        self.current_results = None
        # Store expert activations
        self.expert_activations = None

        self.initUI()

    def initUI(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # --- Model Files Group ---
        model_files_group = QtWidgets.QGroupBox("Model Files")
        model_files_layout = QtWidgets.QVBoxLayout(model_files_group)
        self.file_list = DropListWidget(self)
        self.file_list.setMaximumHeight(60)
        model_files_layout.addWidget(self.file_list)
        drop_label = QtWidgets.QLabel("Drag and drop .pth files here")
        model_files_layout.addWidget(drop_label)
        main_layout.addWidget(model_files_group, 0)

        # --- Model Info Group ---
        model_info_group = QtWidgets.QGroupBox("Model Info")
        model_info_layout = QtWidgets.QVBoxLayout(model_info_group)
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(80)
        model_info_layout.addWidget(self.info_text)
        main_layout.addWidget(model_info_group)

        # --- AI Agents Selection Group ---
        ai_selection_group = QtWidgets.QGroupBox("AI Agents Selection")
        ai_selection_layout = QtWidgets.QGridLayout(ai_selection_group)
        self.agent_selectors = {}
        for i in range(2):
            label = QtWidgets.QLabel(f"AI Agent {i+1}:")
            ai_selection_layout.addWidget(label, i, 0)
            combo = QtWidgets.QComboBox()
            combo.setEditable(False)
            ai_selection_layout.addWidget(combo, i, 1)
            self.agent_selectors[i] = combo
        main_layout.addWidget(ai_selection_group)

        # --- Control Buttons and Options Layout ---
        control_layout = QtWidgets.QHBoxLayout()
        refresh_button = QtWidgets.QPushButton("Refresh Agents")
        refresh_button.clicked.connect(self.update_agent_selectors)
        control_layout.addWidget(refresh_button)
        start_button = QtWidgets.QPushButton("Start Battleground")
        start_button.clicked.connect(self.start_battleground)
        control_layout.addWidget(start_button)
        rounds_label = QtWidgets.QLabel("Rounds:")
        control_layout.addWidget(rounds_label)
        self.rounds_spinbox = QtWidgets.QSpinBox()
        self.rounds_spinbox.setMinimum(1)
        self.rounds_spinbox.setMaximum(1000)
        self.rounds_spinbox.setValue(20)
        control_layout.addWidget(self.rounds_spinbox)

        # --- New Checkboxes ---
        self.two_player_checkbox = QtWidgets.QCheckBox("2 Player Mode")
        control_layout.addWidget(self.two_player_checkbox)
        self.combine_ai_checkbox = QtWidgets.QCheckBox("Combine Columns")
        control_layout.addWidget(self.combine_ai_checkbox)
        self.combine_ai_checkbox.stateChanged.connect(self.update_results_display)
        # Disable combine checkbox when 2 Player is active.
        self.two_player_checkbox.stateChanged.connect(
            lambda state: self.combine_ai_checkbox.setEnabled(state == Qt.Unchecked)
        )
        self.cheat_checkbox = QtWidgets.QCheckBox("Cheat")  # <-- New cheat checkbox
        control_layout.addWidget(self.cheat_checkbox)

        # --- New Checkboxes for 1v2 and Duo Modes ---
        self.onev2_checkbox = QtWidgets.QCheckBox("1v2 Mode")
        control_layout.addWidget(self.onev2_checkbox)

        self.duo_checkbox = QtWidgets.QCheckBox("Duo Mode")
        control_layout.addWidget(self.duo_checkbox)

        # --- Compare Results Button ---
        self.compare_button = QtWidgets.QPushButton("Compare Results")
        self.compare_button.clicked.connect(self.compare_results)
        control_layout.addWidget(self.compare_button)
        
        # --- Show Expert Usage Button ---
        self.expert_button = QtWidgets.QPushButton("Show Expert Usage")
        self.expert_button.clicked.connect(self.show_expert_usage)
        control_layout.addWidget(self.expert_button)

        main_layout.addLayout(control_layout)

        # --- Progress Bar ---
        progress_layout = QtWidgets.QHBoxLayout()
        progress_label = QtWidgets.QLabel("Progress:")
        progress_layout.addWidget(progress_label)
        self.progress_bar = QtWidgets.QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        # --- Results Group ---
        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(320)
        self.results_text.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        results_layout.addWidget(self.results_text)
        main_layout.addWidget(results_group, 1)

    def on_file_drop(self, file_path):
        file_path = file_path.strip()
        if not file_path.endswith(".pth"):
            self.show_info("Only .pth files are supported")
            return
        if file_path in self.loaded_models:
            self.show_info("Model already loaded")
            return
        try:
            self.load_model(file_path)
            self.file_list.addItem(os.path.basename(file_path))
            self.update_agent_selectors()
            self.show_info(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def load_model(self, file_path):
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format")
        
        # Check for different model formats
        if "policy_nets" in checkpoint:
            # Traditional format with separate policy_nets and value_nets
            required_keys = ["policy_nets"]
            if any(k not in checkpoint for k in required_keys):
                raise ValueError("Missing required keys in checkpoint")
            
            any_policy = next(iter(checkpoint["policy_nets"].values()))
            
            # Check if this is a BeliefSpacePolicy model
            if ModelFactory.is_belief_space_policy(any_policy):
                # For BeliefSpacePolicy, determine dimensions and opponent types
                hidden_dim = get_hidden_dim_from_state_dict(any_policy, "network.0")
                total_input_dim = any_policy['network.0.weight'].shape[1]
                # Observation dimension is total input minus belief dimension
                obs_dim = ModelFactory.get_belief_input_dim(any_policy)
                
                # Get the belief model if available
                belief_model = checkpoint.get("belief_model", None)
                
                # Determine number of opponent types from belief model
                if belief_model:
                    num_opponent_types = ModelFactory.get_num_opponent_types(belief_model)
                else:
                    num_opponent_types = 10  # Default value if we can't determine it
                
                self.loaded_models[file_path] = {
                    "policy_nets": checkpoint["policy_nets"],
                    "belief_model": belief_model,
                    "obs_version": 2,  # Assume version 2 for BeliefSpacePolicy (new observation format)
                    "input_dim": obs_dim,
                    "uses_memory": False,
                    "is_stacked_model": False,
                    "is_newer_obs_model": False,
                    "is_belief_space_policy": True,
                    "num_opponent_types": num_opponent_types
                }
                return
            # First check if the loaded model is a StackedObservationConvModel with newer obs format
            if is_stacked_newer_observation_model(any_policy):
                # For StackedNewerObservationModel, determine obs_dim from the model
                obs_dim = get_obs_dim_from_stacked_model(any_policy)  # Reuse this function as it works the same way
                if obs_dim is None:
                    # Fallback: estimate from input tensors
                    try:
                        # Try an alternative approach
                        for key in any_policy.keys():
                            if key.startswith('conv_layers') and key.endswith('weight'):
                                if len(any_policy[key].shape) == 3:  # Conv1d weight tensor
                                    obs_dim = any_policy[key].shape[2]
                                    break
                    except:
                        # Fallback to default
                        obs_dim = 8  # Default for newer observation format
                
                # Set observation version to 5 for StackedNewerObservationModel
                obs_version = 5
                uses_memory = False
                
                self.loaded_models[file_path] = {
                    "policy_nets": checkpoint["policy_nets"],
                    "obp_model": checkpoint.get("obp_model", None),
                    "obs_version": obs_version,
                    "input_dim": obs_dim,
                    "uses_memory": uses_memory,
                    "is_stacked_model": True,
                    "is_newer_obs_model": True,
                    "is_conditional_model": False
                }
            # Then check if the loaded model is a standard StackedObservationConvModel
            elif is_stacked_observation_model(any_policy):
                # For StackedObservationConvModel, determine obs_dim from the model
                obs_dim = get_obs_dim_from_stacked_model(any_policy)
                if obs_dim is None:
                    # Fallback: estimate from input tensors
                    try:
                        # Try an alternative approach to determine the dimension
                        for key in any_policy.keys():
                            if key.startswith('conv_layers') and key.endswith('weight'):
                                if len(any_policy[key].shape) == 3:  # Conv1d weight tensor
                                    obs_dim = any_policy[key].shape[2]
                                    break
                    except:
                        # If all else fails, default to a reasonable value
                        obs_dim = 14  # Default for new observation format
                
                # Set observation version to 3 for StackedObservationConvModel
                obs_version = 3
                uses_memory = False
                
                self.loaded_models[file_path] = {
                    "policy_nets": checkpoint["policy_nets"],
                    "obp_model": checkpoint.get("obp_model", None),
                    "obs_version": obs_version,
                    "input_dim": obs_dim,
                    "uses_memory": uses_memory,
                    "is_stacked_model": True,
                    "is_newer_obs_model": False,
                    "is_conditional_model": False
                }
            else:
                # For traditional models, determine input_dim as before
                if "base_encoder.0.weight" in any_policy:
                    base_dim = any_policy["base_encoder.0.weight"].shape[1]
                    num_opponents = 2  # Default value
                    input_dim = base_dim + (config.STRATEGY_DIM * num_opponents)
                else:
                    try:
                        input_dim = any_policy['fc1.weight'].shape[1]
                    except KeyError:
                        input_dim = get_input_dim_from_state_dict(any_policy, candidate_prefix='fc1')
                
                obs_dim = input_dim
                
                # Set observation version based on input_dim
                if input_dim == 18:
                    obs_version = 1
                elif input_dim in (16, 24, 26):
                    obs_version = 2
                else:
                    obs_version = 2  # Default to newer format if unsure
                
                uses_memory = True
                
                self.loaded_models[file_path] = {
                    "policy_nets": checkpoint["policy_nets"],
                    "obp_model": checkpoint.get("obp_model", None),
                    "obs_version": obs_version,
                    "input_dim": obs_dim,
                    "uses_memory": uses_memory,
                    "is_stacked_model": False,
                    "is_newer_obs_model": False,
                    "is_conditional_model": False
                }
            
        elif "model" in checkpoint:
            # New format with a single model
            any_policy = checkpoint["model"]
            
            # First check for stacked newer observation model
            if is_stacked_newer_observation_model(any_policy):
                # For StackedNewerObservationModel
                obs_dim = get_obs_dim_from_stacked_model(any_policy)
                if obs_dim is None:
                    obs_dim = 8  # Default for newer observation format
                
                # Create two entries for player_0 and player_1 with the same model
                self.loaded_models[file_path] = {
                    "policy_nets": {
                        "player_0": checkpoint["model"],
                        "player_1": checkpoint["model"]
                    },
                    "obp_model": None,
                    "obs_version": 5,  # Use version 5 for StackedNewerObservationModel
                    "input_dim": obs_dim,
                    "uses_memory": False,
                    "is_stacked_model": True,
                    "is_newer_obs_model": True,
                    "is_conditional_model": False
                }
            # Then check for standard stacked observation model
            elif is_stacked_observation_model(any_policy):
                # For StackedObservationConvModel
                obs_dim = get_obs_dim_from_stacked_model(any_policy)
                if obs_dim is None:
                    obs_dim = 14  # Default for new observation format
                
                # Create two entries for player_0 and player_1 with the same model
                self.loaded_models[file_path] = {
                    "policy_nets": {
                        "player_0": checkpoint["model"],
                        "player_1": checkpoint["model"]
                    },
                    "obp_model": None,
                    "obs_version": 3,  # Use version 3 for StackedObservationConvModel
                    "input_dim": obs_dim,
                    "uses_memory": False,
                    "is_stacked_model": True,
                    "is_newer_obs_model": False,
                    "is_conditional_model": False
                }
            else:
                raise ValueError("Unrecognized model format in checkpoint")
        else:
            raise ValueError("Unrecognized checkpoint format")

    def show_info(self, message):
        self.info_text.setPlainText(message)

    def update_agent_selectors(self):
        agent_options = []
        for file_path, data in self.loaded_models.items():
            folder_name = os.path.basename(os.path.dirname(file_path))
            if data.get("is_belief_space_policy", False):
                model_type = "Belief"
            elif data.get("is_stacked_model", False):
                model_type = "StackedObs"
            elif ModelFactory.is_moe_policy(next(iter(data["policy_nets"].values()))):
                model_type = "MoE"
            else:
                model_type = "Standard"
            for agent_name in data["policy_nets"].keys():
                display_text = f"{folder_name} - {os.path.basename(file_path)} - {agent_name} ({model_type})"
                agent_options.append(display_text)
        for i in range(2):
            self.agent_selectors[i].clear()
            self.agent_selectors[i].addItems(agent_options)
            if agent_options:
                self.agent_selectors[i].setCurrentIndex(0)

    def load_selected_agents(self):
        """Loads the selected AI agents from the selectors."""
        ai_agents = {}
        try:
            for i in range(2):
                selection = self.agent_selectors[i].currentText()
                if not selection:
                    raise ValueError(f"Select AI Agent {i+1}")
                parts = selection.split(" - ")
                if len(parts) < 3:
                    raise ValueError("Invalid agent format")
                
                # Handle the model type in parentheses
                agent_part = parts[2]
                agent_name = agent_part.split(" (")[0]
                
                folder_name, file_name = parts[0], parts[1]
                file_path_candidates = [p for p in self.loaded_models.keys() if os.path.basename(p) == file_name]
                if not file_path_candidates:
                    raise ValueError(f"File for {file_name} not found among loaded models.")
                file_path = file_path_candidates[0]
                model_data = self.loaded_models[file_path]
                key = f"player_{i}"
                
                # Check if this is a BeliefSpacePolicy model (has belief_model instead of obp_model)
                if model_data.get("is_belief_space_policy", False):
                    ai_agents[key] = {
                        "policy_net": model_data["policy_nets"][agent_name],
                        "belief_model": model_data.get("belief_model"),
                        "obs_version": model_data["obs_version"],
                        "input_dim": model_data["input_dim"],
                        "uses_memory": model_data["uses_memory"],
                        "is_stacked_model": model_data.get("is_stacked_model", False),
                        "is_belief_space_policy": True,
                        "num_opponent_types": model_data.get("num_opponent_types", 10)
                    }
                else:
                    # Handle standard models
                    ai_agents[key] = {
                        "policy_net": model_data["policy_nets"][agent_name],
                        "obp_model": model_data.get("obp_model"),
                        "obs_version": model_data["obs_version"],
                        "input_dim": model_data["input_dim"],
                        "uses_memory": model_data["uses_memory"],
                        "is_stacked_model": model_data.get("is_stacked_model", False),
                        "is_newer_obs_model": model_data.get("is_newer_obs_model", False)
                    }
            return ai_agents
        except Exception as e:
            self.show_info(f"Error loading selected agents: {str(e)}")
            return None

    def start_battleground(self):
        ai_agents = self.load_selected_agents()
        if not ai_agents:
            return

        # Determine two_player parameter from checkbox:
        two_player_param = "player_1" if self.two_player_checkbox.isChecked() else None
        onev2_enabled = self.onev2_checkbox.isChecked()
        duo_enabled = self.duo_checkbox.isChecked()
        rounds = self.rounds_spinbox.value()
        if duo_enabled:
            total_opponents = len(self.hardcoded_agents) + len(self.historical_models)
            total_matches = rounds * (total_opponents * (total_opponents - 1) // 2)
        else:
            total_matches = rounds * (len(self.hardcoded_agents) + len(self.historical_models))
        self.progress_bar.setMaximum(total_matches)
        self.progress_bar.setValue(0)
        # If there are already results, store them for later comparison.
        if self.results_text.toPlainText().strip():
            try:
                # For simplicity, assume previous results have been stored in self.current_results.
                self.previous_results = self.current_results
            except Exception:
                self.previous_results = None
        self.results_text.clear()
        
        # Read cheat checkbox state
        cheat_flag = self.cheat_checkbox.isChecked()

        self.worker = BattlegroundWorker(
            ai_agents, self.historical_models, self.hardcoded_agents, rounds,
            two_player=two_player_param,
            cheat=cheat_flag,
            onev2=onev2_enabled,
            duo=duo_enabled
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.results_signal.connect(self.display_results)
        self.worker.expert_signal.connect(self.store_expert_activations)
        self.worker.error_signal.connect(lambda msg: self.show_info(f"Error: {msg}"))
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def store_expert_activations(self, expert_activations):
        """Store expert activations data"""
        self.expert_activations = expert_activations
        logger.info(f"Received expert activations for {len(expert_activations)} opponents")

    # --- Display results with optional combining ---
    def display_results(self, results):
        self.current_results = results  # Save the current results
        combine = (not self.two_player_checkbox.isChecked()) and self.combine_ai_checkbox.isChecked()
        is_onev2 = self.onev2_checkbox.isChecked()
        is_duo = self.duo_checkbox.isChecked()
        
        if combine:
            if is_onev2 or is_duo:
                # In 1v2 mode, combine opponent columns.
                html = """
                <table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;">
                <thead>
                    <tr style="background-color: #4f545c;">
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Name</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">AI Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Combined Opponent Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">AI Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Result</th>
                    </tr>
                </thead>
                <tbody>
                """
                total_ai = total_opp = 0
                for opp_name, wins in results.items():
                    # wins[0] is AI wins; wins[1] and wins[2] are opponent wins
                    combined_opp_wins = wins[1] + wins[2]
                    total = wins[0] + combined_opp_wins
                    ai_rate = wins[0] / total if total > 0 else 0.0
                    opp_rate = combined_opp_wins / total if total > 0 else 0.0
                    result_str = "Win" if ai_rate > 0.5 else "Loss"
                    total_ai += wins[0]
                    total_opp += combined_opp_wins
                    row = f"""
                    <tr>
                        <td style="border: 1px solid #7289da; padding: 6px;">{opp_name}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{wins[0]}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{combined_opp_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{ai_rate:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp_rate:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{result_str}</td>
                    </tr>
                    """
                    html += row
                # Add overall row
                overall_total = total_ai + total_opp
                overall_ai_rate = total_ai / overall_total if overall_total > 0 else 0.0
                overall_opp_rate = total_opp / overall_total if overall_total > 0 else 0.0
                overall_result = "Win" if overall_ai_rate > 0.5 else "Loss"
                overall_row = f"""
                <tr>
                    <td style="border: 1px solid #7289da; padding: 6px;">Overall</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_ai}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_opp}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_ai_rate:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_opp_rate:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_result}</td>
                </tr>
                """
                html += overall_row
                html += """
                </tbody>
                </table>
                """
            else:
                # Normal mode: combine AI columns.
                html = """
                <table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;">
                <thead>
                    <tr style="background-color: #4f545c;">
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Name</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Combined AI Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Combined AI Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Result</th>
                    </tr>
                </thead>
                <tbody>
                """
                total_ai = total_opp = 0
                for opp_name, wins in results.items():
                    combined_ai_wins = wins[0] + wins[1]
                    opp_wins = wins[2]
                    total = combined_ai_wins + opp_wins
                    ai_rate = combined_ai_wins / total if total > 0 else 0.0
                    opp_rate = opp_wins / total if total > 0 else 0.0
                    result_str = "Win" if ai_rate > 0.5 else "Loss"
                    total_ai += combined_ai_wins
                    total_opp += opp_wins
                    row = f"""
                    <tr>
                        <td style="border: 1px solid #7289da; padding: 6px;">{opp_name}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{combined_ai_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{ai_rate:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp_rate:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{result_str}</td>
                    </tr>
                    """
                    html += row
                # Add overall row
                overall_total = total_ai + total_opp
                overall_ai_rate = total_ai / overall_total if overall_total > 0 else 0.0
                overall_opp_rate = total_opp / overall_total if overall_total > 0 else 0.0
                overall_result = "Win" if overall_ai_rate > 0.5 else "Loss"
                overall_row = f"""
                <tr>
                    <td style="border: 1px solid #7289da; padding: 6px;">Overall</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_ai}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_opp}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_ai_rate:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_opp_rate:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_result}</td>
                </tr>
                """
                html += overall_row
                html += """
                </tbody>
                </table>
                """
        else:
            if is_onev2 or is_duo:
                # In 1v2 mode without combining, show separate opponent win columns.
                html = """
                <table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;">
                <thead>
                    <tr style="background-color: #4f545c;">
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Name</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">AI Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent1 Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent2 Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">AI Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent1 Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent2 Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Result</th>
                    </tr>
                </thead>
                <tbody>
                """
                total_ai = total_opp1 = total_opp2 = 0
                for opp_name, wins in results.items():
                    ai_wins, opp1_wins, opp2_wins = wins
                    total = ai_wins + opp1_wins + opp2_wins
                    ai_rate = ai_wins / total if total > 0 else 0.0
                    opp1_rate = opp1_wins / total if total > 0 else 0.0
                    opp2_rate = opp2_wins / total if total > 0 else 0.0
                    total_ai += ai_wins
                    total_opp1 += opp1_wins
                    total_opp2 += opp2_wins
                    row = f"""
                    <tr>
                        <td style="border: 1px solid #7289da; padding: 6px;">{opp_name}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{ai_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp1_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp2_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{ai_rate:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp1_rate:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp2_rate:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{"Win" if ai_rate > 0.5 else "Loss"}</td>
                    </tr>
                    """
                    html += row
                # Add overall row
                overall_total = total_ai + total_opp1 + total_opp2
                overall_ai_rate = total_ai / overall_total if overall_total > 0 else 0.0
                overall_opp1_rate = total_opp1 / overall_total if overall_total > 0 else 0.0
                overall_opp2_rate = total_opp2 / overall_total if overall_total > 0 else 0.0
                overall_result = "Win" if overall_ai_rate > 0.5 else "Loss"
                overall_row = f"""
                <tr>
                    <td style="border: 1px solid #7289da; padding: 6px;">Overall</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_ai}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_opp1}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_opp2}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_ai_rate:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_opp1_rate:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_opp2_rate:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_result}</td>
                </tr>
                """
                html += overall_row
                html += """
                </tbody>
                </table>
                """
            else:
                # Normal mode without combining: show separate AI win columns.
                html = """
                <table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;">
                <thead>
                    <tr style="background-color: #4f545c;">
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Name</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">AI1 Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">AI2 Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">AI1 Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">AI2 Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>
                        <th style="border: 1px solid #7289da; padding: 8px;">Result</th>
                    </tr>
                </thead>
                <tbody>
                """
                total_ai1 = total_ai2 = total_opp = 0
                for opp_name, wins in results.items():
                    ai1_wins, ai2_wins, opp_wins = wins
                    total = ai1_wins + ai2_wins + opp_wins
                    rate1 = ai1_wins / total if total > 0 else 0.0
                    rate2 = ai2_wins / total if total > 0 else 0.0
                    opp_rate = opp_wins / total if total > 0 else 0.0
                    total_ai1 += ai1_wins
                    total_ai2 += ai2_wins
                    total_opp += opp_wins
                    combined_rate = (ai1_wins + ai2_wins) / total if total > 0 else 0.0
                    row = f"""
                    <tr>
                        <td style="border: 1px solid #7289da; padding: 6px;">{opp_name}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{ai1_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{ai2_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp_wins}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{rate1:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{rate2:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp_rate:.2%}</td>
                        <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{"Win" if combined_rate > 0.5 else "Loss"}</td>
                    </tr>
                    """
                    html += row
                # Add overall row
                overall_total = total_ai1 + total_ai2 + total_opp
                overall_rate1 = total_ai1 / overall_total if overall_total > 0 else 0.0
                overall_rate2 = total_ai2 / overall_total if overall_total > 0 else 0.0
                overall_opp_rate = total_opp / overall_total if overall_total > 0 else 0.0
                overall_combined_rate = (total_ai1 + total_ai2) / overall_total if overall_total > 0 else 0.0
                overall_result = "Win" if overall_combined_rate > 0.5 else "Loss"
                overall_row = f"""
                <tr>
                    <td style="border: 1px solid #7289da; padding: 6px;">Overall</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_ai1}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_ai2}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{total_opp}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_rate1:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_rate2:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_opp_rate:.2%}</td>
                    <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{overall_result}</td>
                </tr>
                """
                html += overall_row
                html += """
                </tbody>
                </table>
                """
        self.results_text.setHtml(html)
        
    def update_results_display(self):
        """ Updates the displayed results when Combine AI Columns is toggled. """
        if self.current_results:
            self.display_results(self.current_results)  # Reformat and display results
        
    def compare_results(self):
        # Compare previous_results and current_results and plot two bar charts.
        if self.previous_results is None or self.current_results is None:
            QtWidgets.QMessageBox.information(self, "Comparison", "No previous results to compare. Run at least two battles.")
            self.previous_results = self.current_results
            return

        def make_acronym(name):
            # If the name length is 7 or less, return as is.
            if len(name) <= 7:
                return name

            # Attempt to split on common delimiters.
            parts = re.split(r'[+\_]', name)
            if len(parts) > 1:
                # Filter out any empty parts and take first letters.
                acronym = ''.join(part[0].upper() for part in parts if part)
                if len(acronym) > 0:
                    return acronym

            # If no delimiter was found or it didn't work, try to split by camel case.
            camel_parts = re.findall(r'[A-Z][^A-Z]*', name)
            if len(camel_parts) > 1:
                acronym = ''.join(part[0].upper() for part in camel_parts)
                if len(acronym) > 0:
                    return acronym

            # Fallback: simply return the first 7 characters.
            return name[:7]

        opp_names = list(self.current_results.keys())
        # Create shortened labels for plotting.
        display_names = [make_acronym(name) for name in opp_names]

        ai_prev_rates = []
        opp_prev_rates = []
        ai_curr_rates = []
        opp_curr_rates = []

        # Determine mode: if either onev2 or duo is checked.
        is_special_mode = self.onev2_checkbox.isChecked() or self.duo_checkbox.isChecked()

        for opp in opp_names:
            prev = self.previous_results.get(opp, [0, 0, 0])
            curr = self.current_results.get(opp, [0, 0, 0])
            if is_special_mode:
                # In 1v2/duo mode, AI wins are only from column 0;
                # opponent wins are the sum of columns 1 and 2.
                prev_ai_wins = prev[0]
                curr_ai_wins = curr[0]
                prev_opp_wins = prev[1] + prev[2]
                curr_opp_wins = curr[1] + curr[2]
            else:
                # Normal mode: combine AI wins from columns 0 and 1.
                prev_ai_wins = prev[0] + prev[1]
                curr_ai_wins = curr[0] + curr[1]
                prev_opp_wins = prev[2]
                curr_opp_wins = curr[2]
            
            prev_total = prev_ai_wins + prev_opp_wins
            curr_total = curr_ai_wins + curr_opp_wins

            ai_prev = prev_ai_wins / prev_total if prev_total > 0 else 0
            opp_prev = prev_opp_wins / prev_total if prev_total > 0 else 0
            ai_curr = curr_ai_wins / curr_total if curr_total > 0 else 0
            opp_curr = curr_opp_wins / curr_total if curr_total > 0 else 0

            ai_prev_rates.append(ai_prev)
            opp_prev_rates.append(opp_prev)
            ai_curr_rates.append(ai_curr)
            opp_curr_rates.append(opp_curr)

        x = np.arange(len(opp_names))
        width = 0.35

        # Create the figure for AI win rates.
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.bar(x - width/2, ai_prev_rates, width, label='Previous AI Win Rate')
        ax1.bar(x + width/2, ai_curr_rates, width, label='Current AI Win Rate')
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_names, rotation=45)
        ax1.set_ylabel("Win Rate")
        ax1.set_title("AI Win Rate Comparison")
        # Place legend outside the plot area
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

        # Create the figure for Opponent win rates.
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.bar(x - width/2, opp_prev_rates, width, label='Previous Opponent Win Rate')
        ax2.bar(x + width/2, opp_curr_rates, width, label='Current Opponent Win Rate')
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_names, rotation=45)
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Opponent Win Rate Comparison")
        # Place legend outside the plot area
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

        # Update previous_results with current_results for future comparisons.
        self.previous_results = self.current_results

    def show_expert_usage(self):
        """Display expert activation information"""
        if not self.expert_activations:
            QtWidgets.QMessageBox.information(self, "Expert Usage", 
                                            "No expert activation data available. Run a battle first.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Expert Activation Analysis")
        dialog.setMinimumSize(800, 600)
        layout = QtWidgets.QVBoxLayout(dialog)

        # Create a tab widget to show data for each AI agent separately.
        tab_widget = QtWidgets.QTabWidget()

        for player_idx, player in enumerate(["player_0", "player_1"]):
            player_tab = QtWidgets.QWidget()
            player_layout = QtWidgets.QVBoxLayout(player_tab)

            # Build an HTML table showing per-opponent most used expert info.
            html = f"""<h2>Expert Activations for AI Agent {player_idx+1}</h2>
            <table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;">
            <thead>
                <tr style="background-color: #4f545c;">
                <th style="border: 1px solid #7289da; padding: 8px;">Opponent</th>
                <th style="border: 1px solid #7289da; padding: 8px;">Set 1 - Most Used Expert</th>
                <th style="border: 1px solid #7289da; padding: 8px;">Set 2 - Most Used Expert</th>
                </tr>
            </thead>
            <tbody>
            """

            # Lists for plotting
            opponent_names = []
            set1_rates = []
            set1_experts = []
            set2_rates = []
            set2_experts = []

            for opp_name, activations in self.expert_activations.items():
                player_activations = activations.get(player, {})
                if not player_activations:
                    continue

                # Partition activations based on key value.
                set1 = {}
                set2 = {}
                for k, v in player_activations.items():
                    try:
                        key_int = int(k)
                    except ValueError:
                        continue
                    if key_int < 10:
                        set1[k] = v
                    else:
                        # Reindex second set so that keys become 0-9.
                        set2[str(key_int - 10)] = v

                # Process Set 1.
                total1 = sum(set1.values())
                if total1 > 0:
                    expert1, count1 = max(set1.items(), key=lambda x: x[1])
                    rate1 = count1 / total1
                else:
                    expert1, rate1 = "N/A", 0

                # Process Set 2.
                if set2:
                    total2 = sum(set2.values())
                    if total2 > 0:
                        expert2, count2 = max(set2.items(), key=lambda x: x[1])
                        rate2 = count2 / total2
                    else:
                        expert2, rate2 = "N/A", 0
                else:
                    expert2, rate2 = "N/A", 0

                # Add row to HTML table.
                html += f"""
                <tr>
                <td style="border: 1px solid #7289da; padding: 6px;">{opp_name}</td>
                <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">Expert {expert1} ({rate1:.1%})</td>
                <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">Expert {expert2} ({rate2:.1%})</td>
                </tr>
                """

                opponent_names.append(opp_name)
                set1_rates.append(rate1)
                set1_experts.append(expert1)
                set2_rates.append(rate2)
                set2_experts.append(expert2)

            html += """
            </tbody>
            </table>
            <p><b>Note:</b> For each opponent the graph shows two bars (if available) representing the activation rate of the most used expert for each of the two activation sets.</p>
            """

            text = QtWidgets.QTextEdit()
            text.setReadOnly(True)
            text.setHtml(html)
            player_layout.addWidget(text)

            if opponent_names:
                num_opponents = len(opponent_names)
                x = np.arange(num_opponents)
                bar_width = 0.35

                figure = plt.figure(figsize=(10, 6))
                ax = figure.add_subplot(111)

                bars1 = ax.bar(x - bar_width/2, set1_rates, bar_width, label='Set 1')
                bars2 = ax.bar(x + bar_width/2, set2_rates, bar_width, label='Set 2')

                ax.set_xticks(x)
                ax.set_xticklabels(opponent_names, rotation=45, ha='right')
                ax.set_ylabel('Activation Rate')
                ax.set_title(f'Most Used Expert Activation for AI Agent {player_idx+1}')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

                # Annotate bars with expert id.
                for bar, expert in zip(bars1, set1_experts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height, f"E{expert}", ha='center', va='bottom', fontsize=9)
                for bar, expert in zip(bars2, set2_experts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height, f"E{expert}", ha='center', va='bottom', fontsize=9)

                canvas = FigureCanvasQTAgg(figure)
                player_layout.addWidget(canvas)

            tab_widget.addTab(player_tab, f"AI Agent {player_idx+1}")

        layout.addWidget(tab_widget)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec_()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # Apply a dark, Discord-like style
    dark_stylesheet = """
    QWidget {
        background-color: #2f3136;
        color: #dcddde;
        font-family: "Helvetica", "Arial", sans-serif;
    }
    QGroupBox {
        border: 1px solid #202225;
        border-radius: 4px;
        margin-top: 1ex;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px;
        color: #fff;
    }
    QPushButton {
        background-color: #7289da;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        color: #fff;
    }
    QPushButton:hover {
        background-color: #5b6eae;
    }
    QLineEdit, QComboBox, QSpinBox, QTextEdit, QListWidget {
        background-color: #36393f;
        border: 1px solid #202225;
        border-radius: 4px;
        padding: 4px;
    }
    QProgressBar {
        background-color: #36393f;
        border: 1px solid #202225;
        border-radius: 4px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #7289da;
        border-radius: 4px;
    }
    """
    app.setStyleSheet(dark_stylesheet)

    window = AgentBattlegroundGUI()
    window.show()
    sys.exit(app.exec_())