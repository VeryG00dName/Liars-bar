# src/evaluation/evaluate_utils.py
import itertools
import time
import os
import json
import logging
import random
from typing import Any, Dict, Optional
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, deque
from pettingzoo.utils import agent_selector
import torch.nn.functional as F
# Rich library imports for progress and scoreboard display
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

# OpenSkill for rating updates
from openskill.models import PlackettLuce

from src.agents.base_agent import BaseAgent
from src.model.model_factory import ModelFactory
from src.model.other_models import PolicyNetwork, ValueNetwork, StrategyTransformer ,OpponentBehaviorPredictor
from src.model.memory import delete_opponent_memory
# Model and config imports
from src import config

# Additional imports for memory and environment utilities
from src.env.liars_deck_env_utils import query_opponent_memory_full
from src.training.train_transformer import EventEncoder
from src.training.train_extras import convert_memory_to_features, convert_memory_to_features2, set_seed
# Constants for observation versions
OBS_VERSION_1 = 1
OBS_VERSION_2 = 2

# Initialize OpenSkill model (adjust parameters as needed)
model = PlackettLuce(mu=25.0, sigma=25.0 / 3, beta=25.0 / 6)

# Global variables for transformer-based memory integration
global_response2idx = None
global_action2idx = None
global_event_encoder = None
global_strategy_transformer = None
global_response2idx2 = None
global_action2idx2 = None
global_event_encoder2 = None
global_strategy_transformer2 = None
# ----------------------------
# (Other utility functions remain unchanged)
# ----------------------------

def load_combined_checkpoint(checkpoint_path, device):
    """Load a combined checkpoint from the given path."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def get_hidden_dim_from_state_dict(state_dict, layer_prefix='fc1'):
    # Try several candidate prefixes.
    candidate_prefixes = [
        layer_prefix,
        "base_encoder.0",
        "policy_net.fc1",
        "model.fc1"
    ]
    for prefix in candidate_prefixes:
        key = f"{prefix}.weight"
        if key in state_dict:
            return state_dict[key].shape[0]
    # Fallback: return the first 2D tensor's first dimension.
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return tensor.shape[0]
    # If still not found, include available keys in the error message.
    available_keys = list(state_dict.keys())
    raise ValueError(f"Cannot determine hidden_dim from state_dict. Tried prefixes: {candidate_prefixes}. Available keys: {available_keys}")

def get_input_dim_from_state_dict(state_dict, candidate_prefix='fc1'):
    """
    Determines the input dimension of a policy network from its state dictionary.
    
    It checks several candidate prefixes (e.g. "fc1", "base_encoder.0", etc.) and returns
    the second dimension (i.e. the input dimension) of the first matching weight tensor.
    
    Args:
        state_dict (dict): The state dictionary of a policy network.
        candidate_prefix (str): The first candidate prefix to try.
    
    Returns:
        int: The input dimension for the policy network.
    
    Raises:
        ValueError: If no appropriate tensor is found in the state_dict.
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

def assign_final_ranks(triple, cumulative_wins):
    """
    Assign ranks to players based on cumulative wins.
    Ties are handled by assigning the same rank.
    """
    sorted_by_wins = sorted(triple, key=lambda pid: cumulative_wins[pid], reverse=True)
    ranks_dict = {}
    current_rank = 0
    prev_wins = None
    for i, pid in enumerate(sorted_by_wins):
        wins = cumulative_wins[pid]
        if i == 0:
            ranks_dict[pid] = current_rank
            prev_wins = wins
        else:
            if wins == prev_wins:
                ranks_dict[pid] = current_rank
            else:
                current_rank = i
                ranks_dict[pid] = current_rank
            prev_wins = wins
    return [ranks_dict[pid] for pid in triple]

def update_openskill_batch(players, triple, ranks):
    """
    Update OpenSkill ratings for a batch of players based on their ranks.
    Each player is considered as a separate team.
    """
    match = []
    for pid in triple:
        match.append([players[pid]['rating']])
    new_ratings = model.rate(match, ranks=ranks)
    for i, pid in enumerate(triple):
        players[pid]['rating'] = new_ratings[i][0]

def save_scoreboard(players, filename="scoreboard.json"):
    """
    Save the current scoreboard to a JSON file.
    """
    data = {}
    for pid, pdata in players.items():
        data[pid] = {
            "score": pdata["rating"].ordinal()
        }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def load_scoreboard(filename="scoreboard.json"):
    """
    Load the scoreboard from a JSON file.
    """
    if not os.path.exists(filename):
        return {}
    with open(filename, "r") as f:
        return json.load(f)

def compute_ranks(scoreboard):
    """
    Compute ranks based on the scoreboard.
    """
    sorted_players = sorted(scoreboard.items(), key=lambda x: x[1]['score'], reverse=True)
    ranks = {}
    current_rank = 1
    for player_id, pdata in sorted_players:
        ranks[player_id] = current_rank
        current_rank += 1
    return ranks

def compare_scoreboards(old_scoreboard, current_players):
    """
    Compare the old scoreboard with the current players to determine score and rank changes.
    """
    differences = {}
    old_ranks = compute_ranks(old_scoreboard)
    new_scoreboard = {pid: {"score": current_players[pid]["rating"].ordinal()} for pid in current_players}
    new_ranks = compute_ranks(new_scoreboard)
    for pid in current_players:
        current_score = current_players[pid]['rating'].ordinal()
        old_score = old_scoreboard.get(pid, {}).get('score', None)
        current_rank = new_ranks[pid]
        old_rank = old_ranks.get(pid, None)
        if old_score is not None:
            score_diff = round(current_score - old_score, 2)
        else:
            score_diff = None
        if old_rank is not None:
            rank_change = old_rank - current_rank
        else:
            rank_change = None
        differences[pid] = {"score_change": score_diff, "rank_change": rank_change}
    return differences

def format_rank_change(rank_change):
    """
    Format the rank change for display.
    """
    if rank_change is None:
        return "New"
    elif rank_change > 0:
        return f"+{rank_change}"
    elif rank_change < 0:
        return f"{rank_change}"
    else:
        return "0"

def plot_agent_heatmap(agent_h2h, title):
    """
    Plots a heatmap for agent vs. agent win counts.
    """
    agents = sorted(agent_h2h.keys())
    heatmap_data = pd.DataFrame(index=agents, columns=agents, data=0)
    for agent, opponents in agent_h2h.items():
        for opponent, wins in opponents.items():
            heatmap_data.loc[agent, opponent] = wins
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='Blues')
    plt.title(title)
    plt.ylabel('Agent')
    plt.xlabel('Opponent')
    plt.tight_layout()
    plt.savefig("agent_head_to_head_heatmap.png")
    plt.close()

def _convert_to_v1_observation(raw_obs, num_players):
    """
    Convert new observation format to legacy v1 format.
    """
    logger = logging.getLogger("Evaluate")
    logger.debug("Starting conversion to v1 observation")
    card_counts = raw_obs[2:5]
    binary_active = [1.0 if c > 0 else 0.0 for c in card_counts]
    new_opp_features = []
    for i in range(num_players - 1):
        feat_start = 5 + i * 4
        feat = raw_obs[feat_start:feat_start + 4]
        if len(feat) < 4:
            logger.warning(f"Insufficient opponent features for player {i}. Padding with zeros.")
            feat = list(feat) + [0.0] * (4 - len(feat))
        new_opp_features.extend([feat[0], feat[1], feat[2], 0.0, feat[3]])
    converted = np.concatenate([raw_obs[:2], [raw_obs[2]], binary_active, new_opp_features])
    logger.debug(f"Conversion complete. New observation shape: {converted.shape}")
    return converted

def adapt_observation_for_version(obs, num_players, version):
    """
    Convert observation to match the expected format for the agent's version.
    """
    logger = logging.getLogger("Evaluate")
    if version == OBS_VERSION_1:
        logger.debug(f"Converting observation to v1 for version {version}")
        converted_obs = _convert_to_v1_observation(obs, num_players)
        logger.debug(f"Converted observation shape: {converted_obs.shape}")
        return converted_obs
    logger.debug(f"No conversion needed for version {version}")
    return obs

def get_opponent_memory_embedding(current_agent, opponent, device):
    """
    Given the current agent and an opponent identifier, query the opponent's memory,
    convert it into features, and compute a transformer-based memory embedding.
    Returns a tensor of shape (1, config.STRATEGY_DIM).
    """
    global global_response2idx, global_action2idx, global_event_encoder, global_strategy_transformer
    transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
    
    logger = logging.getLogger("Evaluate")
    logger.debug("Entering get_opponent_memory_embedding")
    
    # Load categorical mappings if not already loaded.
    if global_response2idx is None or global_action2idx is None:
        logger.debug("Global response/action mappings not set; loading from checkpoint if available.")
        if os.path.exists(transformer_checkpoint_path):
            ckpt = torch.load(transformer_checkpoint_path, map_location=device, weights_only=False)
            global_response2idx = ckpt.get("response2idx", {})
            global_action2idx = ckpt.get("action2idx", {})
            logger.debug(f"Loaded response2idx with {len(global_response2idx)} entries and action2idx with {len(global_action2idx)} entries.")
        else:
            global_response2idx = {}
            global_action2idx = {}
            logger.debug("Transformer checkpoint not found; using empty mappings.")
    
    # Query memory events.
    logger.debug(f"Querying memory for current_agent: {current_agent}, opponent: {opponent}")
    mem_summary = query_opponent_memory_full(current_agent, opponent)
    logger.debug(f"Memory summary: {mem_summary}")
    if len(mem_summary) < 200:
        pad_event = {"response": "", "triggering_action": "", "penalties": 0, "card_count": 0}
        mem_summary = mem_summary + [pad_event] * (200 - len(mem_summary))
    # Convert memory events to features.
    features_list = convert_memory_to_features(mem_summary, global_response2idx, global_action2idx)
    logger.debug(f"Converted features list: {features_list}")
    
    if features_list:
        feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
        logger.debug(f"Feature tensor shape: {feature_tensor.shape}")
        
        # Ensure global_event_encoder is loaded.
        if global_event_encoder is None:
            logger.debug("Global event encoder not set; initializing.")
            global_event_encoder = EventEncoder(
                response_vocab_size=len(global_response2idx),
                action_vocab_size=len(global_action2idx),
                token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
            ).to(device)
            if os.path.exists(transformer_checkpoint_path):
                ckpt = torch.load(transformer_checkpoint_path, map_location=device, weights_only=False)
                global_event_encoder.load_state_dict(ckpt["event_encoder_state_dict"])
                global_event_encoder.eval()
                logger.debug("Loaded event encoder state_dict from checkpoint.")
            else:
                logger.debug("Transformer checkpoint not found; event encoder initialized with random weights.")
        
        # Ensure global_strategy_transformer is loaded.
        if global_strategy_transformer is None:
            logger.debug("Global strategy transformer not set; initializing.")
            global_strategy_transformer = StrategyTransformer(
                num_tokens=config.STRATEGY_NUM_TOKENS,
                token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
                nhead=config.STRATEGY_NHEAD,
                num_layers=config.STRATEGY_NUM_LAYERS,
                strategy_dim=config.STRATEGY_DIM,
                num_classes=config.STRATEGY_NUM_CLASSES,
                dropout=config.STRATEGY_DROPOUT,
                use_cls_token=True
            ).to(device)
            if os.path.exists(transformer_checkpoint_path):
                ckpt = torch.load(transformer_checkpoint_path, map_location=device, weights_only=False)
                global_strategy_transformer.load_state_dict(ckpt["transformer_state_dict"], strict=False)
                global_strategy_transformer.eval()
                logger.debug("Loaded strategy transformer state_dict from checkpoint.")
            else:
                logger.debug("Transformer checkpoint not found; strategy transformer initialized with random weights.")
            global_strategy_transformer.token_embedding = torch.nn.Identity()
            global_strategy_transformer.classification_head.eval()
        
        with torch.no_grad():
            logger.debug("Passing feature tensor through event encoder.")
            projected = global_event_encoder(feature_tensor)
            logger.debug(f"Projected features shape: {projected.shape}")
            strategy_embedding, _ = global_strategy_transformer(projected)
            logger.debug(f"Strategy embedding shape: {strategy_embedding.shape}")
        
        return strategy_embedding
    else:
        logger.debug("No features extracted from memory; returning zeros.")
        return torch.zeros((1, config.STRATEGY_DIM), dtype=torch.float32, device=device)

# ----------------------------
# New Unified Rich Progress and Scoreboard
# ----------------------------

class RichProgressScoreboard:
    """
    This class combines a progress bar and a live-updating scoreboard using Rich.
    It can be used by both regular evaluations and tournaments.
    """
    def __init__(self, total_steps, players):
        self.console = Console()
        self.total = total_steps
        self.current = 0
        self.players = players
        self.steps_per_sec = 0.0  # track steps per second

        # Create a custom column to show steps/sec after the percentage.
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            # Use dictionary indexing to access the steps_per_sec field.
            TextColumn("[bold]{task.fields[steps_per_sec]}[/bold]", justify="left"),
        )
        # Provide an initial value for the steps_per_sec field.
        self.task_id = self.progress.add_task(
            "Evaluating...",
            total=self.total,
            steps_per_sec="0.00 steps/sec"
        )

        self.live = Live(self._generate_layout(), console=self.console, refresh_per_second=4)
        self.live.__enter__()

    def _generate_scoreboard_table(self, differences=None):
        table = Table(title="Live Scoreboard", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim")
        table.add_column("Player ID", min_width=20)
        table.add_column("Skill", justify="right")
        table.add_column("Match Win Rate", justify="right")
        table.add_column("Round Win Rate", justify="right")
        table.add_column("Δ Rank", justify="right")
        
        sorted_players = sorted(self.players.items(), key=lambda x: x[1]['rating'].ordinal(), reverse=True)
        for rank, (pid, data) in enumerate(sorted_players, 1):
            skill = data['rating'].ordinal()
            # Retrieve the two win rates.
            match_win_rate = data.get('win_rate_match', 0.0)
            round_win_rate = data.get('win_rate_total', 0.0)
            
            # Determine rank change using differences (if provided)
            if differences and pid in differences:
                rank_change = differences[pid].get("rank_change")
                if rank_change is None:
                    rank_change_str = "New"
                elif rank_change > 0:
                    rank_change_str = f"[green]+{rank_change}[/green]"
                elif rank_change < 0:
                    rank_change_str = f"[red]{rank_change}[/red]"
                else:
                    rank_change_str = "0"
            else:
                rank_change_str = ""
            
            # Color code the rank number based on position.
            if rank == 1:
                rank_str = f"[bold gold1]{rank}[/bold gold1]"
            elif rank == 2:
                rank_str = f"[bold silver]{rank}[/bold silver]"
            elif rank == 3:
                rank_str = f"[bold dark_orange]{rank}[/bold dark_orange]"
            else:
                rank_str = str(rank)
            
            table.add_row(
                rank_str,
                pid,
                f"{skill:.2f}",
                f"{match_win_rate:.2%}",
                f"{round_win_rate:.2%}",
                rank_change_str
            )
        return table

    def _generate_layout(self, differences=None):
        progress_panel = Panel(
            self.progress,
            title="Progress",
            height=3
        )
        scoreboard = self._generate_scoreboard_table(differences)
        layout = Layout()
        layout.split_column(
            Layout(progress_panel, size=3),
            Layout(scoreboard, ratio=1)
        )
        return layout

    def update(self, increment=1, differences=None, description=None, steps_per_sec=None):
        if steps_per_sec is not None:
            self.steps_per_sec = steps_per_sec
        self.current += increment

        self.progress.update(
            self.task_id,
            advance=increment,
            description=description or "Evaluating...",
            steps_per_sec=f"{self.steps_per_sec:.2f} steps/sec"
        )

        # Refresh the live layout.
        self.live.update(self._generate_layout(differences))

    def advance_progress(self, increment=1):
        """Advance the progress bar without re-rendering scoreboard differences."""
        self.progress.update(self.task_id, advance=increment)

    def update_scoreboard(self, differences=None, steps_per_sec=None, description=None):
        """Update the scoreboard (rank differences, steps/sec, etc.) on the live layout."""
        if steps_per_sec is not None:
            self.steps_per_sec = steps_per_sec
        self.progress.update(
            self.task_id,
            description=description or "Evaluating...",
            steps_per_sec=f"{self.steps_per_sec:.2f} steps/sec"
        )
        # Rebuild and refresh the layout including scoreboard differences.
        self.live.update(self._generate_layout(differences))

    def close(self):
        self.live.__exit__(None, None, None)

def rich_print_expert_activations(expert_activations, agent_map):
    """
    Displays MoE expert activations in a table with columns:
      - Agent ID
      - Selected Expert
      - Opponent ID
      - Activation Details

    :param expert_activations: A dict keyed by actual agent IDs (MoE agents) with values that are
           dicts mapping opponent actual IDs to a details dict containing:
             "selected_expert": (the selected expert index),
             "activation_details": (a string or list with details).
    :param agent_map: A dict mapping environment IDs to actual agent IDs (for display, if needed).
                      (Here, keys are already actual agent IDs so an identity mapping is fine.)
    """
    console = Console()
    table = Table(title="MoE Expert Activations")
    table.add_column("Agent ID", style="bold")
    table.add_column("Selected Expert", style="cyan")
    table.add_column("Opponent ID", style="magenta")
    table.add_column("Activation Details", style="green")

    for agent_id, opp_dict in expert_activations.items():
        # Use the agent_map to get the display name (if desired).
        display_agent = agent_map.get(agent_id, agent_id)
        if not opp_dict:
            table.add_row(display_agent, "-", "-", "No activations")
            continue
        for opp_id, details in opp_dict.items():
            display_opp = agent_map.get(opp_id, opp_id)
            if isinstance(details, dict):
                selected_expert = details.get("selected_expert", "-")
                activation_str = str(details.get("activation_details", ""))
            else:
                selected_expert = details
                activation_str = str(details)
            table.add_row(display_agent, str(selected_expert), display_opp, activation_str)
    console.print(table)

# ----------------------------
# New Unified initialize players function
# ----------------------------
def initialize_players(base_dir, device):
    """
    Unified function to load checkpoint-based players.

    If base_dir contains any subdirectories, each subdirectory is treated as a version
    (with its name used as a prefix for player IDs). Otherwise, base_dir is assumed to directly
    contain checkpoint files (and the filename is used as the prefix).

    For each checkpoint file (with a ".pth" extension, ignoring transformer_classifier.pth),
    the checkpoint is loaded and the policy, value, and OBP models (if available) are created.
    The observation version is set based on the input dimension of the policy network.
    """
    logger = logging.getLogger("Evaluate")
    players = {}
    items = os.listdir(base_dir)
    has_subdirs = any(os.path.isdir(os.path.join(base_dir, item)) for item in items)

    def process_checkpoint(checkpoint_path, identifier_prefix):
        local_players = {}
        try:
            checkpoint = load_combined_checkpoint(checkpoint_path, device)
            policy_nets = checkpoint['policy_nets']
            value_nets = checkpoint['value_nets']
            obp_model_state = checkpoint.get('obp_model', None)
            obp_model = None

            # Process OBP if available.
            if obp_model_state is not None:
                example_observation = torch.randn(1, config.OPPONENT_INPUT_DIM).to(device)
                example_memory_embedding = torch.randn(1, config.STRATEGY_DIM).to(device)
                fc1_weight = obp_model_state.get("fc1.weight", None)
                if fc1_weight is None:
                    raise ValueError("OBP state dict missing fc1.weight")
                obp_hidden_dim = get_hidden_dim_from_state_dict(obp_model_state, layer_prefix='fc1')
                if fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
                    # New OBP with memory
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=config.OPPONENT_INPUT_DIM,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2,
                        memory_dim=config.STRATEGY_DIM
                    ).to(device)
                    obp_model.eval()
                    obp_model = torch.jit.trace(obp_model, (example_observation, example_memory_embedding))
                elif fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM:
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=config.OPPONENT_INPUT_DIM,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    ).to(device)
                    obp_model.eval()
                    obp_model = torch.jit.trace(obp_model, example_observation)
                else:
                    raise ValueError(f"Unexpected OBP input dimension: {fc1_weight.shape[1]}")
                obp_model.load_state_dict(obp_model_state)
            # Determine observation version using one of the policy networks.
            any_policy = next(iter(policy_nets.values()))
            try:
                actual_input_dim = any_policy['fc1.weight'].shape[1]
            except KeyError:
                actual_input_dim = get_input_dim_from_state_dict(any_policy, candidate_prefix='fc1')
            if actual_input_dim == 18:
                obs_version = 1
            elif actual_input_dim in (16, 24, 26):
                obs_version = 2
            else:
                raise ValueError(f"Unknown input dimension: {actual_input_dim}")

            # Process each agent in the checkpoint.
            for agent_name, policy_state_dict in policy_nets.items():
                uses_memory = ("fc4.weight" in policy_state_dict)
                # Determine if the checkpoint uses an auxiliary classifier.
                use_aux_classifier = "fc_classifier.weight" in policy_state_dict
                # For new models, infer the number of opponent classes.
                if use_aux_classifier:
                    num_opponent_classes = policy_state_dict["fc_classifier.weight"].shape[0]
                else:
                    num_opponent_classes = None
                policy_hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, layer_prefix='fc1')

                # Check for MoE models first.
                if ModelFactory.is_moe_policy(policy_state_dict):
                    policy_net = ModelFactory.create_policy_network(
                        input_dim=actual_input_dim,
                        hidden_dim=policy_hidden_dim,
                        output_dim=config.OUTPUT_DIM,
                        use_aux_classifier=use_aux_classifier,
                        num_opponent_classes=num_opponent_classes,
                        use_moe_model=True,
                        num_experts=10
                    ).to(device)
                else:
                    # For non-MoE models, use new model parameters if the checkpoint uses an auxiliary classifier,
                    # otherwise use the old model configuration.
                    if use_aux_classifier:
                        policy_net = ModelFactory.create_policy_network(
                            input_dim=actual_input_dim,
                            hidden_dim=policy_hidden_dim,
                            output_dim=config.OUTPUT_DIM,
                            use_aux_classifier=use_aux_classifier,
                            num_opponent_classes=num_opponent_classes,
                            use_new_model=True
                        ).to(device)
                    else:
                        policy_net = ModelFactory.create_policy_network(
                            input_dim=actual_input_dim,
                            hidden_dim=policy_hidden_dim,
                            output_dim=config.OUTPUT_DIM,
                            use_new_model=False,
                            strategy_dim=config.STRATEGY_DIM,
                            num_opponents=2  # Adjust based on your environment if needed
                        ).to(device)
                policy_net.load_state_dict(policy_state_dict, strict=False)
                policy_net.eval()

                value_state_dict = value_nets[agent_name]
                value_hidden_dim = get_hidden_dim_from_state_dict(value_state_dict, layer_prefix='fc1')
                value_net = ValueNetwork(
                    input_dim=actual_input_dim,
                    hidden_dim=value_hidden_dim,
                    use_dropout=True,
                    use_layer_norm=True
                ).to(device)
                value_net.load_state_dict(value_state_dict, strict=False)
                value_net.eval()

                # Construct player_id using the identifier_prefix.
                # (In recursive mode, identifier_prefix is the subdirectory name; in flat mode it is the file name.)
                player_id = f"{identifier_prefix}_player_{agent_name.replace('player_', '')}"
                rating = model.rating(name=player_id)
                local_players[player_id] = {
                    'policy_net': policy_net,
                    'value_net': value_net,
                    'obp_model': obp_model,
                    'obs_version': obs_version,
                    'rating': rating,
                    'uses_memory': uses_memory,
                    # Tournament-specific fields:
                    'score': rating.ordinal(),
                    'wins_match': 0,           # Counts one win per match (i.e. match-level win)
                    'total_round_wins': 0,     # Cumulative total of rounds won across matches
                    'games_played': 0,         # Number of matches played
                    'win_rate_match': 0.0,     # wins_match / games_played (match win rate)
                    'win_rate_total': 0.0      # total_round_wins / (num_games_per_match * games_played) (round win rate)
                }
            return local_players
        except Exception as e:
            logger.error(f"Failed to process checkpoint {checkpoint_path}: {e}")
            return {}

    if has_subdirs:
        # Recursive mode: iterate each subdirectory.
        for sub in items:
            sub_path = os.path.join(base_dir, sub)
            if os.path.isdir(sub_path):
                for file in os.listdir(sub_path):
                    if file.endswith(".pth"):
                        checkpoint_path = os.path.join(sub_path, file)
                        players.update(process_checkpoint(checkpoint_path, identifier_prefix=sub))
    else:
        # Flat mode: checkpoint files are directly under base_dir.
        for file in items:
            if file.endswith(".pth") and file != "transformer_classifier.pth":
                checkpoint_path = os.path.join(base_dir, file)
                players.update(process_checkpoint(checkpoint_path, identifier_prefix=file))
    return players

# ----------------------------
# Unified Evaluation Function
# ----------------------------

def evaluate_agents(
    env,
    device: torch.device,
    players_in_this_game: Dict[str, BaseAgent],
    episodes: int = 11,
    two_player: Optional[str] = None,
    track_experts: bool = False,
    progress_callback=None,
    cheat_expert_index: Optional[Any] = None # Keep Any type hint
    ) -> tuple:
    """
    Evaluate multiple agents over a number of episodes.

    Returns:
      cumulative_wins, action_counts, game_wins_list,
      avg_steps, steps_per_sec,
      expert_activations_by_player_id (or None),
      player_id_map
    """
    logger = logging.getLogger("Evaluate")
    logger.info(f"Starting evaluation with {len(players_in_this_game)} agents for {episodes} episodes.")

    # Map from your BaseAgent.get_player_id() → environment agent_id
    player_id_map = {agent.get_player_id(): env_id
                     for env_id, agent in players_in_this_game.items()}
    all_player_ids = list(player_id_map.keys())

    # Clear any persistent opponent memory before evaluation
    delete_opponent_memory()

    # Initialize statistics
    action_counts = {pid: defaultdict(int) for pid in all_player_ids}
    cumulative_wins = {pid: 0 for pid in all_player_ids}
    game_wins_list = []
    total_steps = 0

    # Expert tracking container
    expert_activations_by_player_id = None
    if track_experts:
        expert_activations_by_player_id = {
            pid: {'steps': []}
            for pid in all_player_ids
        }

    # Fix randomness
    set_seed(config.SEED)

    start_time = time.time()
    for game_idx in range(1, episodes + 1):
        # Reset environment and agents
        env.reset(seed=game_idx)
        for agent in players_in_this_game.values():
            agent.reset()

        # Optionally pre-eliminate one player
        if two_player is not None and two_player in env.penalties:
            env.penalties[two_player] = env.penalty_thresholds[two_player]
            env.terminations[two_player] = True
            logger.debug(f"Pre-eliminated {two_player}")
            alive = [a for a in env.possible_agents if not env.terminations.get(a, False)]
            env.agents = alive
            env._agent_selector = agent_selector(alive)
            env.agent_selection = env._agent_selector.next()
        else:
            env.agents = env.possible_agents[:]
            env._agent_selector = agent_selector(env.agents)
            env.agent_selection = env._agent_selector.next()

        # Play one game
        steps_in_game = 0
        game_active = True
        while game_active and env.agent_selection is not None:
            steps_in_game += 1
            agent_id_env = env.agent_selection
            observation = env.observe(agent_id_env)
            _, _, terminated, truncated, info = env.last()
            if terminated or truncated:
                env.step(None)
                continue

            current_agent = players_in_this_game[agent_id_env]
            player_id = current_agent.get_player_id()

            try:
                action = current_agent.get_action(
                    env, agent_id_env, observation, info, cheat_expert_index
                )
                action_counts[player_id][action] += 1

                # Step-level expert info capture
                if track_experts and hasattr(current_agent, 'get_last_expert_info'):
                    expert_info = current_agent.get_last_expert_info()
                    if expert_info:
                        expert_activations_by_player_id[player_id]['steps'].append(expert_info)
                        logger.debug(
                            f"[Game {game_idx} Step {steps_in_game}] "
                            f"Agent {player_id} expert info: {expert_info}"
                        )

            except Exception as e:
                logger.error(
                    f"Error during get_action for {player_id} ({agent_id_env}): {e}",
                    exc_info=True
                )
                mask = info.get('action_mask', [1] * env.action_spaces[agent_id_env].n)
                valid_actions = [i for i, m in enumerate(mask) if m == 1]
                action = random.choice(valid_actions) if valid_actions else 0

            env.step(action)
            if not env.agents:
                game_active = False

        # End of one game: record win
        total_steps += steps_in_game
        wins = {pid: 0 for pid in all_player_ids}
        winner_env = env.winner
        if winner_env and winner_env in players_in_this_game:
            pid = players_in_this_game[winner_env].get_player_id()
            cumulative_wins[pid] += 1
            wins[pid] = 1
        game_wins_list.append(wins)

        if progress_callback is not None:
            progress_callback(game_idx)

    # Final stats
    elapsed = time.time() - start_time
    steps_per_sec = total_steps / elapsed if elapsed > 0 else 0.0
    avg_steps = total_steps / episodes if episodes > 0 else 0.0

    logger.info(
        f"Evaluation finished in {elapsed:.2f}s "
        f"({steps_per_sec:.2f} steps/s, {avg_steps:.2f} steps/game)."
    )
    logger.info(f"Cumulative wins: {cumulative_wins}")

    if track_experts:
        return cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec, expert_activations_by_player_id, player_id_map
    else:
        return cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec, None, player_id_map