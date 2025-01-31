# src/training/train_tournament.py

import logging
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.evaluate import evaluate_agents
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.training.train_multi_utils import save_multi_checkpoint, load_multi_checkpoint
from src.training.train_utils import compute_gae
from src.training.train_extras import set_seed
from src.training.train import train
from src import config

def create_player_pool(total_players, device):
    """Create initial pool of players with unique IDs"""
    player_pool = {}
    for p in range(total_players):
        key = f"player_{p}"
        policy_net = PolicyNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        value_net = ValueNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
        optimizer_value = torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE)
        player_pool[key] = {
            'policy_net': policy_net,
            'value_net': value_net,
            'optimizer_policy': optimizer_policy,
            'optimizer_value': optimizer_value,
            'entropy_coef': config.INIT_ENTROPY_COEF,
            'wins': 0,
            'games_played': 0
        }
    return player_pool

def evaluate_tournament(players, device, num_games_per_player=20):
    """Evaluate all players through random matchups"""
    logger = logging.getLogger('TournamentEval')
    logger.info("Starting tournament evaluation...")
    
    player_ids = list(players.keys())
    win_counts = {pid: 0 for pid in player_ids}
    
    # Run random matchups
    for _ in range(num_games_per_player):
        group = random.sample(player_ids, 3)
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        
        # Create temporary agent mapping
        players_in_game = {pid: players[pid] for pid in group}
        
        # Run evaluation
        cumulative_wins, _, _, _ = evaluate_agents(
            env=env,
            device=device,
            players_in_this_game=players_in_game,
            episodes=5
        )
        
        # Update stats
        for pid in group:
            win_counts[pid] += cumulative_wins.get(pid, 0)
            players[pid]['games_played'] += 5
            players[pid]['wins'] += cumulative_wins.get(pid, 0)

    # Calculate win rates
    rankings = sorted(player_ids, 
                     key=lambda x: win_counts[x]/players[x]['games_played'] if players[x]['games_played'] > 0 else 0, 
                     reverse=True)
    
    logger.info("Tournament evaluation completed.")
    return rankings

def cull_and_replace(players, rankings, cull_percent=0.2, device=torch.device("cpu")):
    """Replace bottom performers with new agents and top clones"""
    logger = logging.getLogger('Evolution')
    total_players = len(rankings)
    num_to_cull = int(total_players * cull_percent)
    
    # Identify bottom performers
    bottom_cutoff = total_players - num_to_cull
    survivors = rankings[:bottom_cutoff]
    culled = rankings[bottom_cutoff:]
    
    # Remove culled players
    for pid in culled:
        del players[pid]
    
    # Create replacements: 50% new, 50% clones from top 25%
    num_new = num_to_cull // 2
    num_clones = num_to_cull - num_new
    top_performers = rankings[:max(1, len(rankings)//4)]
    
    # Add new agents
    for i in range(num_new):
        new_id = f"player_{len(players)}"
        policy_net = PolicyNetwork(
            config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM,
            use_lstm=True, use_dropout=True, use_layer_norm=True
        ).to(device)
        value_net = ValueNetwork(
            config.INPUT_DIM, config.HIDDEN_DIM,
            use_dropout=True, use_layer_norm=True
        ).to(device)
        players[new_id] = {
            'policy_net': policy_net,
            'value_net': value_net,
            'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
            'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
            'entropy_coef': config.INIT_ENTROPY_COEF,
            'wins': 0,
            'games_played': 0
        }
    
    # Clone top performers
    for _ in range(num_clones):
        if not top_performers:
            break
        original_id = random.choice(top_performers)
        clone_id = f"clone_{original_id}_{len(players)}"
        
        # Clone networks
        original = players[original_id]
        policy_net = PolicyNetwork(
            config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM,
            use_lstm=True, use_dropout=True, use_layer_norm=True
        ).to(device)
        policy_net.load_state_dict(original['policy_net'].state_dict())
        
        value_net = ValueNetwork(
            config.INPUT_DIM, config.HIDDEN_DIM,
            use_dropout=True, use_layer_norm=True
        ).to(device)
        value_net.load_state_dict(original['value_net'].state_dict())
        
        players[clone_id] = {
            'policy_net': policy_net,
            'value_net': value_net,
            'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
            'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
            'entropy_coef': original['entropy_coef'],
            'wins': 0,
            'games_played': 0
        }
    
    logger.info(f"Culled {len(culled)} agents, added {num_new} new and {num_clones} cloned agents")
    return players

def main():
    set_seed()
    logger = logging.getLogger('TrainTournament')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    
    device = torch.device(config.DEVICE)
    writer = SummaryWriter(log_dir=os.path.join(config.TENSORBOARD_RUNS_DIR, "tournament"))
    
    # Training parameters
    TOTAL_PLAYERS = 12
    GROUP_SIZE = 3
    BATCHES_BETWEEN_EVALS = 2
    CULL_PERCENT = 0.25
    TOTAL_BATCHES = 10
    
    # Initialize environment
    temp_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
    config.set_derived_config(
        temp_env.observation_spaces[temp_env.agents[0]],
        temp_env.action_spaces[temp_env.agents[0]],
        GROUP_SIZE - 1
    )
    
    # Create player pool
    player_pool = create_player_pool(TOTAL_PLAYERS, device)
    
    # Load existing checkpoints
    start_batch, loaded_entropy, obp_state, obp_optim_state = load_multi_checkpoint(
        player_pool, config.CHECKPOINT_DIR, GROUP_SIZE
    )
    
    # Main training loop
    for batch in range(start_batch, TOTAL_BATCHES + 1):
        logger.info(f"\n=== Processing Batch {batch}/{TOTAL_BATCHES} ===")
        
        # Train all groups
        groups = [list(player_pool.keys())[i:i+GROUP_SIZE] 
                 for i in range(0, len(player_pool), GROUP_SIZE)]
        
        for group in groups:
            env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
            agent_map = {f'player_{i}': group[i] for i in range(GROUP_SIZE)}
            
            train(
                agents_dict={pid: player_pool[pid] for pid in group},
                env=env,
                device=device,
                num_episodes=config.EPISODES_PER_BATCH,
                log_tensorboard=True,
                writer=writer,
                agent_mapping=agent_map
            )
        
        # Evolutionary selection
        if batch % BATCHES_BETWEEN_EVALS == 0:
            # Evaluate and cull
            rankings = evaluate_tournament(player_pool, device)
            player_pool = cull_and_replace(player_pool, rankings, CULL_PERCENT, device)
            
            # Save evolved population
            save_multi_checkpoint(
                player_pool=player_pool,
                obp_model=None,  # Assuming OBP not used in this implementation
                obp_optimizer=None,
                batch=batch,
                checkpoint_dir=config.CHECKPOINT_DIR,
                group_size=GROUP_SIZE
            )
    
    writer.close()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()