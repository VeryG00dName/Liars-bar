# src/training/train_tournament.py

import logging
import os
import random
import time
import numpy as np
import torch
import uuid
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.training.train_multi_utils import save_multi_checkpoint, load_multi_checkpoint
from src.training.train import train
from src.training.train_extras import set_seed
from src.misc.tune_eval import run_group_swiss_tournament, openskill_model
from src import config

# Tournament configuration
TOURNAMENT_INTERVAL = 3        # Run tournament every N batches
CULL_PERCENTAGE = 0.2          # Remove bottom 20% each tournament
CLONE_PERCENTAGE = 0.5         # 50% of new agents are clones of top performers
GROUP_SIZE = 3                  # Players per group
TOTAL_PLAYERS = 12              # Initial population size (must be multiple of GROUP_SIZE)

# Lineage tracking
CLONE_REGISTRY = defaultdict(int)

def configure_logger():
    logger = logging.getLogger('TrainTournament')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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
    obp_optimizer = torch.optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    return obp_model, obp_optimizer

def generate_agent_name(source_id=None):
    """Generate structured agent names with lineage tracking"""
    if source_id is None:
        return f"new_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    
    if source_id.startswith("clone"):
        parts = source_id.split("_")
        try:
            orig = parts[parts.index("orig")+1]
            gen = int(parts[parts.index("v")+1]) + 1
        except (ValueError, IndexError):
            orig = source_id
            gen = 1
    else:
        orig = source_id.split('_')[0]
        gen = 1
    
    return f"clone_v{gen}_orig_{orig}_{uuid.uuid4().hex[:4]}"

def create_new_agent(device):
    """Create a new agent with lineage tracking"""
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
        'entropy_coef': config.INIT_ENTROPY_COEF,
        'lineage': {
            'created_at': time.time(),
            'source': 'initial',
            'generation': 0
        }
    }

def clone_agent(source_agent, source_id, device):
    """Create a tracked clone of an existing agent"""
    clone_id = generate_agent_name(source_id)
    
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
    
    # Update lineage tracking
    lineage = source_agent['lineage'].copy()
    lineage['generation'] += 1
    lineage['source'] = source_id
    
    CLONE_REGISTRY[source_id] += 1
    
    return {
        'policy_net': policy_net,
        'value_net': value_net,
        'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
        'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
        'entropy_coef': source_agent['entropy_coef'],
        'lineage': lineage
    }

def maintain_player_pool_size(player_pool, group_size):
    """Ensure player count is always a multiple of group_size"""
    current_size = len(player_pool)
    remainder = current_size % group_size
    
    if remainder != 0:
        needed = (group_size - remainder) % group_size
        if needed > 0:
            for i in range(needed):
                new_id = generate_agent_name()
                player_pool[new_id] = create_new_agent(torch.device(config.DEVICE))
        else:
            sorted_ids = sorted(player_pool.keys(), 
                              key=lambda x: player_pool[x]['lineage']['generation'])
            for pid in sorted_ids[:abs(needed)]:
                del player_pool[pid]
    
    return player_pool

def run_tournament(env, device, player_pool, obp_model, logger):
    """Evaluate all agents and return sorted rankings"""
    logger.info("Starting tournament evaluation...")
    
    players = {}
    for pid, agent in player_pool.items():
        players[pid] = {
            'policy_net': agent['policy_net'],
            'value_net': agent['value_net'],
            'obp_model': obp_model,
            'rating': openskill_model.rating(name=pid),
            'score': 0.0,
            'wins': 0,
            'games_played': 0
        }
    
    final_rankings = run_group_swiss_tournament(
        env=env,
        device=device,
        players=players,
        num_games_per_match=11,
        NUM_ROUNDS=5
    )
    
    # Update scores in player pool
    for pid, player_data in players.items():
        if pid in player_pool:
            player_pool[pid]['score'] = player_data['score']
    
    return final_rankings

def cull_and_replace(player_pool, rankings, device, logger, group_size):
    """Evolutionary replacement with lineage tracking"""
    num_agents = len(rankings)
    base_cull = max(group_size, int(num_agents * CULL_PERCENTAGE))
    cull_adjusted = base_cull - (base_cull % group_size)
    num_cull = max(group_size, cull_adjusted)
    
    if (num_agents - num_cull) % group_size != 0:
        num_cull += (num_agents - num_cull) % group_size
    
    culled_ids = rankings[-num_cull:]
    logger.info(f"Culling {len(culled_ids)} agents")
    
    # Remove culled agents
    for pid in culled_ids:
        del player_pool[pid]
    
    # Create replacements
    new_players = {}
    for i in range(num_cull):
        if i < num_cull * CLONE_PERCENTAGE:
            new_id = generate_agent_name()
            new_players[new_id] = create_new_agent(device)
        else:
            source_id = random.choice(rankings[:group_size*2])
            try:
                cloned_agent = clone_agent(player_pool[source_id], source_id, device)
                new_id = generate_agent_name(source_id)
                new_players[new_id] = cloned_agent
            except KeyError:
                logger.warning(f"Clone source {source_id} missing, creating new agent")
                new_id = generate_agent_name()
                new_players[new_id] = create_new_agent(device)
    
    player_pool.update(new_players)
    
    # Final size validation
    player_pool = maintain_player_pool_size(player_pool, group_size)
    
    logger.info(f"Added {len(new_players)} new agents")
    return list(player_pool.keys())

def main():
    set_seed()
    logger = configure_logger()
    device = torch.device(config.DEVICE)
    
    # Initialize OBP
    obp_model, obp_optimizer = initialize_obp(device)
    
    # Environment setup
    base_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
    config.set_derived_config(
        base_env.observation_spaces[base_env.agents[0]],
        base_env.action_spaces[base_env.agents[0]],
        num_opponents=GROUP_SIZE-1
    )
    
    # Initialize player pool
    player_pool = {}
    for i in range(TOTAL_PLAYERS):
        agent_id = generate_agent_name()
        player_pool[agent_id] = create_new_agent(device)
    
    # Load checkpoints
    start_batch, loaded_entropy, obp_state, obp_optim_state = load_multi_checkpoint(
        player_pool, config.CHECKPOINT_DIR, GROUP_SIZE
    )
    
    # Load OBP states
    if obp_state is not None:
        obp_model.load_state_dict(obp_state)
    if obp_optim_state is not None:
        obp_optimizer.load_state_dict(obp_optim_state)
    
    # Training loop
    writer = SummaryWriter(log_dir=config.MULTI_LOG_DIR)
    try:
        for batch_id in range(start_batch, 20):
            logger.info(f"\n=== Batch {batch_id} ===")
            
            # Maintain group size
            player_pool = maintain_player_pool_size(player_pool, GROUP_SIZE)
            
            # Form groups
            all_ids = list(player_pool.keys())
            random.shuffle(all_ids)
            groups = [all_ids[i:i+GROUP_SIZE] for i in range(0, len(all_ids), GROUP_SIZE)]
            
            # Train each group
            for group_idx, group in enumerate(groups):
                if len(group) != GROUP_SIZE:
                    logger.warning(f"Skipping invalid group {group_idx} with size {len(group)}")
                    continue
                
                env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
                try:
                    agent_map = {env.agents[i]: group[i] for i in range(GROUP_SIZE)}
                except IndexError as e:
                    logger.error(f"Group formation failed: {str(e)}")
                    continue
                
                logger.info(f"Training group {group_idx}: {[n.split('_')[0] for n in group]}")
                
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
            
            # Evolutionary step
            if batch_id % TOURNAMENT_INTERVAL == 0:
                logger.info("Starting evolutionary step...")
                
                # Run tournament
                tournament_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
                temp_pool = maintain_player_pool_size(player_pool.copy(), GROUP_SIZE)
                rankings = run_tournament(tournament_env, device, temp_pool, obp_model, logger)
                
                # Evolve population
                updated_ids = cull_and_replace(player_pool, rankings, device, logger, GROUP_SIZE)
                
                # Form new groups
                random.shuffle(updated_ids)
                groups = [updated_ids[i:i+GROUP_SIZE] 
                         for i in range(0, len(updated_ids), GROUP_SIZE)]
                
                logger.info(f"New population: {len(player_pool)} agents")
                logger.debug(f"Sample agents: {list(player_pool.keys())[:3]}")
            
            # Save checkpoint
            save_multi_checkpoint(
                player_pool=player_pool,
                obp_model=obp_model,
                obp_optimizer=obp_optimizer,
                batch=batch_id,
                checkpoint_dir=config.CHECKPOINT_DIR,
                group_size=GROUP_SIZE
            )
            
    finally:
        writer.close()
        logger.info("Training completed. Final checkpoint saved.")

if __name__ == "__main__":
    main()