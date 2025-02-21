# src/evaluation/evaluate_utils.py
import itertools
import time
import os
import json
import logging
import random
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from pettingzoo.utils import agent_selector

# Rich library imports for progress and scoreboard display
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

# OpenSkill for rating updates
from openskill.models import PlackettLuce

from src.model.new_models import PolicyNetwork, ValueNetwork, StrategyTransformer ,OpponentBehaviorPredictor

# Model and config imports
from src import config

# Additional imports for memory and environment utilities
from src.model.memory import get_opponent_memory
from src.env.liars_deck_env_utils import query_opponent_memory_full
from src.training.train_transformer import EventEncoder

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

# ----------------------------
# (Other utility functions remain unchanged)
# ----------------------------

def load_combined_checkpoint(checkpoint_path, device):
    """Load a combined checkpoint from the given path."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def get_hidden_dim_from_state_dict(state_dict, layer_prefix='fc1'):
    """Determine hidden dimension from a state dictionary based on layer prefix."""
    weight_key = f"{layer_prefix}.weight"
    if weight_key in state_dict:
        return state_dict[weight_key].shape[0]
    else:
        for key in state_dict.keys():
            if key.endswith('.weight') and ('fc' in key or 'layer' in key):
                return state_dict[key].shape[0]
    raise ValueError(f"Cannot determine hidden_dim from state_dict for layer prefix '{layer_prefix}'.")

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
            ckpt = torch.load(transformer_checkpoint_path, map_location=device)
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
                ckpt = torch.load(transformer_checkpoint_path, map_location=device)
                global_event_encoder.load_state_dict(ckpt["event_encoder_state_dict"])
                global_event_encoder.eval()
                logger.debug("Loaded event encoder state_dict from checkpoint.")
            else:
                logger.debug("Transformer checkpoint not found; event encoder initialized with random weights.")
        
        # Ensure global_strategy_transformer is loaded.
        if global_strategy_transformer is None:
            logger.debug("Global strategy transformer not set; initializing.")
            from src.model.new_models import StrategyTransformer
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
                ckpt = torch.load(transformer_checkpoint_path, map_location=device)
                global_strategy_transformer.load_state_dict(ckpt["transformer_state_dict"], strict=False)
                global_strategy_transformer.eval()
                logger.debug("Loaded strategy transformer state_dict from checkpoint.")
            else:
                logger.debug("Transformer checkpoint not found; strategy transformer initialized with random weights.")
            # Remove classification head and override token embedding.
            global_strategy_transformer.token_embedding = torch.nn.Identity()
            global_strategy_transformer.classification_head = None
        
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
# OBP Inference Functions (unchanged except normalization)
# ----------------------------

def run_obp_inference(obp_model, obs, device, num_players, agent_version, current_agent, env, memory_embeddings=None):
    """
    Runs OBP inference for an observation.
    If memory_embeddings is provided (and the OBP model was trained with transformer memory),
    it will be passed to the OBP model.
    """
    logger = logging.getLogger("Evaluate")
    if obp_model is None:
        num_opponents = num_players - 1
        logger.debug(f"No OBP model available for agent version {agent_version}. Returning default 0.0s.")
        return [0.0] * num_opponents

    if agent_version == 1:
        opp_feature_dim = 5
    elif agent_version == 2:
        opp_feature_dim = 4
    else:
        raise ValueError(f"Unknown agent_version: {agent_version}")

    # Determine whether the OBP model requires memory.
    fc1_weight = obp_model.state_dict().get("fc1.weight", None)
    is_new_obp = False
    if fc1_weight is not None:
        input_dim = fc1_weight.shape[1]
        if input_dim == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
            is_new_obp = True

    num_opponents = num_players - 1
    opp_features_start = len(obs) - (num_opponents * opp_feature_dim)
    opponents = [opp for opp in env.possible_agents if opp != current_agent]
    obp_probs = []
    for i, opp in enumerate(opponents):
        start_idx = opp_features_start + i * opp_feature_dim
        end_idx = start_idx + opp_feature_dim
        opp_vec = obs[start_idx:end_idx]
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if is_new_obp and memory_embeddings is not None:
                mem_emb = memory_embeddings[i]
                logits = obp_model(opp_vec_tensor, mem_emb)
            else:
                logits = obp_model(opp_vec_tensor)
            probs = torch.softmax(logits, dim=-1)
            obp_probs.append(probs[0, 1].item())
    return obp_probs

def run_obp_inference_tournament(obp_model, obs, device, num_players, obs_version, current_agent, opponents, memory_embeddings=None):
    """
    Specialized OBP inference for tournament mode.
    If memory_embeddings is provided and the OBP model requires memory,
    each opponent's memory embedding is passed along with its observation vector.
    """
    if obp_model is None:
        return []
    if obs_version == 1 or obs_version == "OBS_VERSION_1":
        opp_feature_dim = 5
    elif obs_version == 2 or obs_version == "OBS_VERSION_2":
        opp_feature_dim = 4
    else:
        raise ValueError(f"Unknown observation version: {obs_version}")
    
    num_opponents = len(opponents)
    opp_features_start = len(obs) - (num_opponents * opp_feature_dim)
    obp_probs = []
    
    fc1_weight = obp_model.state_dict().get("fc1.weight", None)
    is_new_obp = False
    if fc1_weight is not None:
        input_dim = fc1_weight.shape[1]
        if input_dim == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
            is_new_obp = True

    for i, opp in enumerate(opponents):
        start_idx = opp_features_start + i * opp_feature_dim
        end_idx = start_idx + opp_feature_dim
        opp_vec = obs[start_idx:end_idx]
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if is_new_obp and memory_embeddings is not None:
                mem_emb = memory_embeddings[i]
                logits = obp_model(opp_vec_tensor, mem_emb)
            else:
                logits = obp_model(opp_vec_tensor)
            probs = torch.softmax(logits, dim=-1)
            obp_probs.append(probs[0, 1].item())
    return obp_probs

def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

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

    def close(self):
        self.live.__exit__(None, None, None)


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
                elif fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM:
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=config.OPPONENT_INPUT_DIM,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    ).to(device)
                else:
                    raise ValueError(f"Unexpected OBP input dimension: {fc1_weight.shape[1]}")
                obp_model.load_state_dict(obp_model_state)
                obp_model.eval()

            # Determine observation version using one of the policy networks.
            any_policy = next(iter(policy_nets.values()))
            actual_input_dim = any_policy['fc1.weight'].shape[1]
            if actual_input_dim == 18:
                obs_version = 1
            elif actual_input_dim in (16, 24, 26):
                obs_version = 2
            else:
                raise ValueError(f"Unknown input dimension: {actual_input_dim}")

            # Process each agent in the checkpoint.
            for agent_name, policy_state_dict in policy_nets.items():
                uses_memory = ("fc4.weight" in policy_state_dict)
                use_aux_classifier = "fc_classifier.weight" in policy_state_dict
                if use_aux_classifier:
                    # Infer the number of opponent classes from the first dimension of the classifier weight.
                    num_opponent_classes = policy_state_dict["fc_classifier.weight"].shape[0]
                else:
                    num_opponent_classes = None
                policy_hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, layer_prefix='fc1')

                policy_net = PolicyNetwork(
                    input_dim=actual_input_dim,
                    hidden_dim=policy_hidden_dim,
                    output_dim=config.OUTPUT_DIM,
                    use_lstm=True,
                    use_dropout=True,
                    use_layer_norm=True,
                    use_aux_classifier=use_aux_classifier,
                    num_opponent_classes=num_opponent_classes
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

def evaluate_agents(env, device, players_in_this_game, episodes=11, is_tournament=False):
    """
    Unified evaluation function used by both regular evaluation and tournaments.
    This version computes memory embeddings per opponent using get_opponent_memory_embedding,
    applies min–max normalization, and then passes the normalized embeddings to OBP inference,
    exactly as during training.
    """
    logger = logging.getLogger("Evaluate")
    player_ids = list(players_in_this_game.keys())
    
    # Map environment agent names to player IDs.
    agent_to_player = {f'player_{i}': player_ids[i] for i in range(env.num_players)}
    
    # Clear persistent memory for agents that use memory.
    for env_agent in agent_to_player:
        pid = agent_to_player[env_agent]
        if players_in_this_game[pid].get('uses_memory', False) and players_in_this_game[pid]['obs_version'] == 2:
            get_opponent_memory(env_agent).memory.clear()

    action_counts = {pid: defaultdict(int) for pid in player_ids}
    cumulative_wins = {pid: 0 for pid in player_ids}
    total_steps = 0
    game_wins_list = []
    start_time = time.time()

    for game_idx in range(1, episodes + 1):
        env.reset()
        env.agents = list(agent_to_player.keys())
        env._agent_selector = agent_selector(env.agents)
        env.agent_selection = env._agent_selector.next() if env.agents else None

        steps_in_game = 0
        game_wins = {pid: 0 for pid in player_ids}

        while env.agent_selection is not None:
            steps_in_game += 1
            agent = env.agent_selection
            obs, reward, termination, truncation, info = env.last()
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                env.step(None)
                continue

            observation = env.observe(agent)
            if isinstance(observation, dict):
                observation = observation.get(agent, None)
            if not isinstance(observation, np.ndarray):
                logger.error(f"Expected np.ndarray, got {type(observation)}.")
                env.step(None)
                continue

            player_id = agent_to_player[agent]
            player_data = players_in_this_game[player_id]
            
            # Hardcoded agent handling.
            if player_data.get('hardcoded_bot', False):
                mask = info.get('action_mask', [1] * config.OUTPUT_DIM)
                table_card = getattr(env, 'table_card', None)
                action = player_data['agent'].play_turn(observation, mask, table_card)
                action_counts[player_id][action] += 1
                env.step(action)
                continue

            obp_model = player_data.get('obp_model', None)
            version = player_data['obs_version']
            converted_obs = adapt_observation_for_version(observation, env.num_players, version)
            
            # Decide whether OBP model uses transformer memory.
            fc1_weight = None
            use_tournament = is_tournament
            memory_embeddings = None
            if obp_model is not None:
                fc1_weight = obp_model.state_dict().get("fc1.weight", None)
                if fc1_weight is not None and fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
                    use_tournament = True
                    # Compute memory embeddings for each opponent using get_opponent_memory_embedding.
                    opponents = [opp for opp in env.possible_agents if opp != agent]
                    mem_emb_list = []
                    for opp in opponents:
                        emb = get_opponent_memory_embedding(agent, opp, device)
                        # emb has shape (1, config.STRATEGY_DIM)
                        mem_emb_list.append(emb.cpu().numpy().flatten())
                    # Concatenate embeddings into one 1D array.
                    mem_concat = np.concatenate(mem_emb_list, axis=0)
                    # Min-max normalization.
                    mem_min = mem_concat.min()
                    mem_max = mem_concat.max()
                    logger.debug(f"Memory min: {mem_min}, max: {mem_max}")
                    if mem_max - mem_min == 0:
                        norm_mem = mem_concat
                    else:
                        norm_mem = (mem_concat - mem_min) / (mem_max - mem_min)
                    # Split back into per–opponent segments.
                    segment_size = config.STRATEGY_DIM
                    memory_embeddings = []
                    for i in range(len(opponents)):
                        seg = norm_mem[i * segment_size:(i + 1) * segment_size]
                        memory_embeddings.append(torch.tensor(seg, dtype=torch.float32, device=device).unsqueeze(0))
                else:
                    use_tournament = is_tournament

            if use_tournament:
                opponents = [opp for opp in env.possible_agents if opp != agent]
                obp_probs = run_obp_inference_tournament(
                    obp_model, converted_obs, device, env.num_players, version, agent, opponents,
                    memory_embeddings=memory_embeddings
                )
            else:
                obp_probs = run_obp_inference(
                    obp_model, converted_obs, device, env.num_players, version, agent, env,
                    memory_embeddings=memory_embeddings
                )
            
            # Build default observation.
            default_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32)], axis=0)
            
            # If persistent memory is used (obs_version==2), concatenate transformer features.
            if player_data.get('uses_memory', False) and version == 2 and memory_embeddings is not None:
                # For consistency, flatten the concatenated normalized memory embeddings.
                transformer_features = np.concatenate([emb.cpu().numpy().flatten() for emb in memory_embeddings], axis=0)
                final_obs = np.concatenate([default_obs, transformer_features], axis=0)
            else:
                final_obs = default_obs

            # Convert final_obs to tensor.
            final_obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
            policy_net = player_data['policy_net']
            # Initialize LSTM hidden state.
            num_layers = policy_net.lstm.num_layers
            batch_size = final_obs_tensor.size(0)
            hidden_size = policy_net.lstm.hidden_size
            hidden_state = (
                torch.zeros(num_layers, batch_size, hidden_size, device=device),
                torch.zeros(num_layers, batch_size, hidden_size, device=device)
            )
            with torch.no_grad():
                probs, _, _ = policy_net(final_obs_tensor, hidden_state)
            probs = torch.clamp(probs, 1e-8, 1.0)
            mask = info.get('action_mask', [1] * config.OUTPUT_DIM)
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
            masked_probs = probs * mask_tensor
            if masked_probs.sum() <= 0:
                logger.warning(f"All actions masked for {agent}; using uniform.")
                masked_probs = mask_tensor + 1e-8
            masked_probs /= masked_probs.sum()
            m = torch.distributions.Categorical(masked_probs)
            action = m.sample().item()
            action_counts[player_id][action] += 1
            env.step(action)

        winner_agent = env.winner
        if winner_agent:
            winner_player = agent_to_player.get(winner_agent, None)
            if winner_player:
                game_wins[winner_player] += 1
            else:
                logger.error(f"Winner agent {winner_agent} not found.")
        else:
            logger.warning("No winner detected.")
        for pid in player_ids:
            cumulative_wins[pid] += game_wins[pid]
        total_steps += steps_in_game
        game_wins_list.append(game_wins)

    elapsed_time = time.time() - start_time
    steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
    avg_steps = total_steps / episodes if episodes > 0 else 0
    return cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec