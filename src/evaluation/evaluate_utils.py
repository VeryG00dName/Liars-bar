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

# Model and config imports
from src.model.models import PolicyNetwork, OpponentBehaviorPredictor, ValueNetwork
from src import config

# Additional imports for memory and environment utilities
from src.model.memory import get_opponent_memory
from src.env.liars_deck_env_utils import query_opponent_memory_full

from sklearn.preprocessing import StandardScaler

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
    
    if global_response2idx is None or global_action2idx is None:
        if os.path.exists(transformer_checkpoint_path):
            ckpt = torch.load(transformer_checkpoint_path, map_location=device)
            global_response2idx = ckpt.get("response2idx", {})
            global_action2idx = ckpt.get("action2idx", {})
        else:
            global_response2idx = {}
            global_action2idx = {}
    
    mem_summary = query_opponent_memory_full(current_agent, opponent)
    features_list = convert_memory_to_features(mem_summary, global_response2idx, global_action2idx)
    
    if features_list:
        feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
        if global_event_encoder is None:
            from src.training.train_transformer import EventEncoder
            global_event_encoder = EventEncoder(
                response_vocab_size=len(global_response2idx),
                action_vocab_size=len(global_action2idx),
                token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
            ).to(device)
            if os.path.exists(transformer_checkpoint_path):
                ckpt = torch.load(transformer_checkpoint_path, map_location=device)
                global_event_encoder.load_state_dict(ckpt["event_encoder_state_dict"])
                global_event_encoder.eval()
        if global_strategy_transformer is None:
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
            global_strategy_transformer.token_embedding = torch.nn.Identity()
            global_strategy_transformer.classification_head = None
        with torch.no_grad():
            projected = global_event_encoder(feature_tensor)
            strategy_embedding, _ = global_strategy_transformer(projected)
        return strategy_embedding
    else:
        return torch.zeros((1, config.STRATEGY_DIM), dtype=torch.float32, device=device)

# ----------------------------
# OBP Inference Functions (unchanged)
# ----------------------------

def run_obp_inference(obp_model, obs, device, num_players, agent_version, current_agent, env):
    """
    Runs OBP inference for an observation using the regular method.
    (This function is used if the OBP model does NOT require a memory embedding.)
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
    fc1_weight = obp_model.state_dict().get("fc1.weight", None)
    is_new_obp = False
    if fc1_weight is not None:
        input_dim = fc1_weight.shape[1]
        # If the input dimension requires memory integration, we must supply it.
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
            if is_new_obp:
                # Fall back to tournament inference if memory is required.
                obp_probs.append(run_obp_inference_tournament(obp_model, obs, device, num_players, agent_version, current_agent, opponents)[i])
            else:
                logits = obp_model(opp_vec_tensor)
                probs = torch.softmax(logits, dim=-1)
                obp_probs.append(probs[0, 1].item())
    return obp_probs

def run_obp_inference_tournament(obp_model, obs, device, num_players, obs_version, current_agent, opponents):
    """
    Specialized OBP inference for tournament mode.
    This version supplies the required memory_embedding to the OBP model.
    """
    if obp_model is None:
        return []
    if obs_version == OBS_VERSION_1 or obs_version == 1:
        opp_feature_dim = 5
    elif obs_version == OBS_VERSION_2 or obs_version == 2:
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
            if is_new_obp:
                memory_emb = get_opponent_memory_embedding(current_agent, opp, device)
                logits = obp_model(opp_vec_tensor, memory_emb)
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
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
        )
        self.task_id = self.progress.add_task("Evaluating...", total=self.total)
        self.live = Live(self._generate_layout(), console=self.console, refresh_per_second=4)
        self.live.__enter__()

    def _generate_scoreboard_table(self, differences=None):
        table = Table(title="Live Scoreboard", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim")
        table.add_column("Player ID", min_width=20)
        table.add_column("Skill", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Δ Rank", justify="right")
        sorted_players = sorted(self.players.items(), key=lambda x: x[1]['rating'].ordinal(), reverse=True)
        for rank, (pid, data) in enumerate(sorted_players, 1):
            skill = data['rating'].ordinal()
            win_rate = data.get('win_rate', 0.0)
            diff = differences.get(pid, {}) if differences else {}
            rank_change = format_rank_change(diff.get('rank_change', None))
            table.add_row(
                str(rank),
                pid,
                f"{skill:.2f}",
                f"{win_rate:.2%}",
                rank_change
            )
        return table

    def _generate_layout(self, differences=None):
        progress_panel = Panel(self.progress, title="Progress", height=3)
        scoreboard = self._generate_scoreboard_table(differences)
        layout = Layout()
        layout.split_column(
            Layout(progress_panel, size=3),
            Layout(scoreboard, ratio=1)
        )
        return layout

    def update(self, increment=1, differences=None, description=None):
        self.current += increment
        self.progress.update(self.task_id, advance=increment, description=description or "Evaluating...")
        self.live.update(self._generate_layout(differences))

    def close(self):
        self.live.__exit__(None, None, None)

# ----------------------------
# Unified Evaluation Function (unchanged)
# ----------------------------

def evaluate_agents(env, device, players_in_this_game, episodes=11, is_tournament=False):
    """Unified evaluation function used by both regular evaluation and tournaments"""
    logger = logging.getLogger("Evaluate")
    player_ids = list(players_in_this_game.keys())
    
    # Environment setup: Map environment agent names to player IDs.
    agent_to_player = {f'player_{i}': player_ids[i] for i in range(env.num_players)}
    
    # Memory initialization: Clear persistent memory for agents using memory (obs_version == 2).
    for env_agent in agent_to_player:
        pid = agent_to_player[env_agent]
        if players_in_this_game[pid].get('uses_memory', False) and players_in_this_game[pid]['obs_version'] == OBS_VERSION_2:
            get_opponent_memory(env_agent).memory.clear()

    # Tracking structures for action counts, wins, and steps.
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

            # Process observation.
            observation = env.observe(agent)
            if isinstance(observation, dict):
                observation = observation.get(agent, None)
            if not isinstance(observation, np.ndarray):
                logger.error(f"Expected np.ndarray, got {type(observation)}.")
                env.step(None)
                continue

            player_id = agent_to_player[agent]
            player_data = players_in_this_game[player_id]
            
            # ===== Handle Hardcoded Bot =====
            if player_data.get('hardcoded_bot', False):
                mask = info.get('action_mask', [1] * config.OUTPUT_DIM)
                # Optionally, pass table card if available in the environment
                table_card = getattr(env, 'table_card', None)
                action = player_data['agent'].play_turn(observation, mask, table_card)
                if action in action_counts[player_id]:
                    action_counts[player_id][action] += 1
                env.step(action)
                continue
            # ===================================

            obp_model = player_data.get('obp_model', None)
            version = player_data['obs_version']

            # Adapt the observation.
            converted_obs = adapt_observation_for_version(observation, env.num_players, version)
            
            # Decide which OBP inference to use.
            if obp_model is not None:
                fc1_weight = obp_model.state_dict().get("fc1.weight", None)
                if fc1_weight is not None and fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
                    # The OBP model requires memory embedding.
                    use_tournament = True
                else:
                    use_tournament = is_tournament
            else:
                use_tournament = False

            if use_tournament:
                opponents = [opp for opp in env.possible_agents if opp != agent]
                obp_probs = run_obp_inference_tournament(
                    obp_model, converted_obs, device, env.num_players, version, agent, opponents
                )
            else:
                obp_probs = run_obp_inference(
                    obp_model, converted_obs, device, env.num_players, version, agent, env
                )
            
            default_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32)], axis=0)
            expected_dim = player_data['policy_net'].fc1.in_features

            # Memory integration for agents using persistent memory (obs_version == 2).
            if player_data.get('uses_memory', False) and version == OBS_VERSION_2:
                required_mem_dim = expected_dim - (len(converted_obs) + len(obp_probs))
                transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
                global global_response2idx, global_action2idx, global_event_encoder, global_strategy_transformer
                if global_response2idx is None or global_action2idx is None:
                    if os.path.exists(transformer_checkpoint_path):
                        ckpt = torch.load(transformer_checkpoint_path, map_location=device)
                        global_response2idx = ckpt.get("response2idx", {})
                        global_action2idx = ckpt.get("action2idx", {})
                    else:
                        global_response2idx = {}
                        global_action2idx = {}
                from src.env.liars_deck_env_utils import query_opponent_memory_full
                mem_features_list = []
                for opp in env.possible_agents:
                    if opp != agent:
                        mem_summary = query_opponent_memory_full(agent, opp)
                        features_list = convert_memory_to_features(mem_summary, global_response2idx, global_action2idx)
                        if features_list:
                            feature_tensor = torch.tensor(features_list, dtype=torch.float, device=device).unsqueeze(0)
                            if global_event_encoder is None:
                                from src.training.train_transformer import EventEncoder
                                global_event_encoder = EventEncoder(
                                    response_vocab_size=len(global_response2idx),
                                    action_vocab_size=len(global_action2idx),
                                    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
                                ).to(device)
                                if os.path.exists(transformer_checkpoint_path):
                                    ckpt = torch.load(transformer_checkpoint_path, map_location=device)
                                    global_event_encoder.load_state_dict(ckpt["event_encoder_state_dict"])
                                    global_event_encoder.eval()
                            if global_strategy_transformer is None:
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
                                    global_strategy_transformer.load_state_dict(ckpt.get("transformer_state_dict", ckpt), strict=False)
                                    global_strategy_transformer.eval()
                                global_strategy_transformer.token_embedding = torch.nn.Identity()
                                global_strategy_transformer.classification_head = None
                            with torch.no_grad():
                                projected = global_event_encoder(feature_tensor)
                                strategy_embedding, _ = global_strategy_transformer(projected)
                            mem_features_list.append(strategy_embedding.cpu().numpy().flatten())
                        else:
                            mem_features_list.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))
                if mem_features_list:
                    mem_features = np.concatenate(mem_features_list, axis=0)
                else:
                    mem_features = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                current_mem_dim = mem_features.shape[0]
                if current_mem_dim < required_mem_dim:
                    pad = np.zeros(required_mem_dim - current_mem_dim, dtype=np.float32)
                    mem_features = np.concatenate([mem_features, pad], axis=0)
                elif current_mem_dim > required_mem_dim:
                    mem_features = mem_features[:required_mem_dim]
                    scaler = StandardScaler()
                    if mem_features.size > 0:
                        normalized_transformer_features = scaler.fit_transform(np.array(mem_features).reshape(1, -1)).flatten()
                        mem_features = normalized_transformer_features
                final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32), mem_features], axis=0)
            else:
                final_obs = default_obs

            if len(final_obs) != expected_dim:
                raise ValueError(
                    f"Observation dimension mismatch for {player_id}. Expected {expected_dim}, got {len(final_obs)}. Version: {version}"
                )

            obs_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
            policy_net = player_data['policy_net']
            with torch.no_grad():
                probs, _, _ = policy_net(obs_tensor, None)
            probs = torch.clamp(probs, 1e-8, 1.0)
            mask = info.get('action_mask', [1] * config.OUTPUT_DIM)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
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
                if is_tournament:
                    players_in_this_game[winner_player]['wins'] += 1
            else:
                logger.error(f"Winner agent {winner_agent} not found.")
        else:
            logger.warning("No winner detected.")
        for pid in player_ids:
            cumulative_wins[pid] += game_wins[pid]
            if is_tournament:
                players_in_this_game[pid]['games_played'] += 1
        total_steps += steps_in_game
        game_wins_list.append(game_wins)

    end_time = time.time()
    elapsed_time = end_time - start_time
    steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
    avg_steps = total_steps / episodes if episodes > 0 else 0
    return cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec
