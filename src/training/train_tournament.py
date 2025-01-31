# src/training/train_tournament.py

import logging
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.training.train_multi_utils import save_multi_checkpoint, load_multi_checkpoint
from src.training.train import train
from src.training.train_extras import set_seed
from src.misc.tune_eval import run_group_swiss_tournament
from src import config

from src.misc.tune_eval import openskill_model

# Tournament configuration
TOURNAMENT_INTERVAL = 2        # Run tournament every N batches
CULL_PERCENTAGE = 0.5          # Remove bottom 20% each tournament
CLONE_PERCENTAGE = 0.5         # 50% of new agents are clones of top performers

def configure_logger():
    logger = logging.getLogger('TrainTournament')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def initialize_obp(device):
    """Initialize Opponent Behavior Predictor model and optimizer"""
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2
    ).to(device)
    
    obp_optimizer = torch.optim.Adam(obp_model.parameters(), 
                                   lr=config.OPPONENT_LEARNING_RATE)
    return obp_model, obp_optimizer

def create_new_agent(device):
    """Create a brand new untrained agent"""
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM
    ).to(device)
    
    value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM
    ).to(device)
    
    return {
        'policy_net': policy_net,
        'value_net': value_net,
        'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
        'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
        'entropy_coef': config.INIT_ENTROPY_COEF
    }

def clone_agent(source_agent, device):
    """Create a deep copy of an existing agent"""
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM
    ).to(device)
    policy_net.load_state_dict(source_agent['policy_net'].state_dict())
    
    value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM
    ).to(device)
    value_net.load_state_dict(source_agent['value_net'].state_dict())
    
    return {
        'policy_net': policy_net,
        'value_net': value_net,
        'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
        'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
        'entropy_coef': source_agent['entropy_coef']
    }

def run_tournament(env, device, player_pool, obp_model, logger):
    """Evaluate all agents and return sorted rankings"""
    logger.info("Starting tournament evaluation...")
    
    # Convert pool to tournament format
    players = {}
    for pid, agent in player_pool.items():
        players[pid] = {
            'policy_net': agent['policy_net'],
            'value_net': agent['value_net'],
            'obp_model': obp_model,
            'rating': openskill_model.rating(name=pid),  # Initialize OpenSkill rating
            'score': 0.0,
            'wins': 0,
            'games_played': 0
        }
    
    # Run Swiss tournament
    final_rankings = run_group_swiss_tournament(
        env=env,
        device=device,
        players=players,
        num_games_per_match=11,
        NUM_ROUNDS=7
    )
    
    return final_rankings

def cull_and_replace(player_pool, rankings, device, logger):
    """Cull bottom performers and replace with new/cloned agents"""
    num_agents = len(rankings)
    num_cull = int(num_agents * CULL_PERCENTAGE)
    culled_ids = rankings[-num_cull:]
    
    logger.info(f"Culling {num_cull} agents: {culled_ids}")
    
    # Remove culled agents from pool
    for pid in culled_ids:
        del player_pool[pid]
    
    # Create replacement agents
    num_new = int(num_cull * CLONE_PERCENTAGE)
    num_clones = num_cull - num_new
    
    # Get top performers for cloning
    clone_source_ids = rankings[:max(1, int(num_clones*2))]
    
    # Add new agents
    new_players = {}
    for i in range(num_cull):
        if i < num_new:
            new_id = f"new_{len(player_pool)+i}"
            new_players[new_id] = create_new_agent(device)
        else:
            source_id = random.choice(clone_source_ids)
            clone_id = f"clone_{source_id}_{len(player_pool)+i}"
            new_players[clone_id] = clone_agent(player_pool[source_id], device)
    
    # Add new players to pool
    player_pool.update(new_players)
    logger.info(f"Added {len(new_players)} new agents ({num_new} new, {num_clones} clones)")
    
    return list(player_pool.keys())

def main():
    set_seed()
    logger = configure_logger()
    device = torch.device(config.DEVICE)
    
    # Initialize OBP model
    obp_model, obp_optimizer = initialize_obp(device)
    
    # Initialize environment
    base_env = LiarsDeckEnv(num_players=3, render_mode=None)
    config.set_derived_config(
        base_env.observation_spaces[base_env.agents[0]],
        base_env.action_spaces[base_env.agents[0]],
        num_opponents=2
    )
    
    # Initialize player pool with GROUP_SIZE alignment
    GROUP_SIZE = 3
    TOTAL_PLAYERS = 15  # Must be multiple of GROUP_SIZE
    player_pool = {f"player_{i}": create_new_agent(device) for i in range(TOTAL_PLAYERS)}
    
    # Load existing checkpoints
    start_batch, loaded_entropy, obp_state, obp_optim_state = load_multi_checkpoint(
        player_pool, config.CHECKPOINT_DIR, GROUP_SIZE
    )
    
    # Enforce GROUP_SIZE alignment in loaded pool
    current_count = len(player_pool)
    padding = (-current_count) % GROUP_SIZE
    for i in range(padding):
        player_pool[f"pad_{current_count + i}"] = create_new_agent(device)
    
    # Load OBP states
    if obp_state is not None:
        obp_model.load_state_dict(obp_state)
    if obp_optim_state is not None:
        obp_optimizer.load_state_dict(obp_optim_state)
    
    # Training loop
    writer = SummaryWriter(log_dir=config.MULTI_LOG_DIR)
    for batch_id in range(start_batch, 20):
        logger.info(f"\n=== Batch {batch_id} ===")
        
        # Create groups with size validation
        all_ids = list(player_pool.keys())
        groups = [all_ids[i:i+GROUP_SIZE] for i in range(0, len(all_ids), GROUP_SIZE)]
        
        for group in groups:
            if len(group) != GROUP_SIZE:
                logger.warning(f"Skipping invalid group size {len(group)}")
                continue
            
            env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
            try:
                agent_map = {env.agents[i]: group[i] for i in range(GROUP_SIZE)}
            except IndexError as e:
                logger.error(f"Group formation failed: {str(e)}")
                continue
            
            train(
                agents_dict={pid: player_pool[pid] for pid in group},
                env=env,
                device=device,
                num_episodes=2000,
                log_tensorboard=True,
                writer=writer,
                logger=logger,
                agent_mapping=agent_map,
                obp_model=obp_model,
                obp_optimizer=obp_optimizer
            )
        
        # Tournament and evolution
        if batch_id % TOURNAMENT_INTERVAL == 0:
            tournament_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
            rankings = run_tournament(tournament_env, device, player_pool, obp_model, logger)
            updated_ids = cull_and_replace(player_pool, rankings, device, logger)
            
            # Maintain GROUP_SIZE alignment
            padding = (-len(updated_ids)) % GROUP_SIZE
            for i in range(padding):
                updated_ids.append(f"pad_{len(updated_ids)}")
                player_pool[updated_ids[-1]] = create_new_agent(device)
            
            # Remix groups
            random.shuffle(updated_ids)
            groups = [updated_ids[i:i+GROUP_SIZE] 
                     for i in range(0, len(updated_ids), GROUP_SIZE)]
            
            logger.info(f"New group composition: {groups}")
        
        # Save checkpoints
        save_multi_checkpoint(
            player_pool=player_pool,
            obp_model=obp_model,
            obp_optimizer=obp_optimizer,
            batch=batch_id,
            checkpoint_dir=config.CHECKPOINT_DIR,
            group_size=GROUP_SIZE
        )
    
    writer.close()
    logger.info("Training completed with tournament-based evolution")

if __name__ == "__main__":
    main()