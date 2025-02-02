# src/training/train_tournament.py

import logging
import os
import random
import time
import uuid
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.training.train_multi_utils import save_multi_checkpoint, load_multi_checkpoint
from src.training.train import train
from src.training.train_extras import set_seed
from src.misc.tune_eval import run_group_swiss_tournament, openskill_model
from src import config

# Tournament configuration
TOURNAMENT_INTERVAL = 2
CULL_PERCENTAGE = 0.2
CLONE_PERCENTAGE = 0.5
GROUP_SIZE = 3
TOTAL_PLAYERS = 12

CLONE_REGISTRY = defaultdict(int)

def configure_logger():
    """Set up logging infrastructure"""
    logger = logging.getLogger('TrainTournament')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def initialize_obp(device):
    """Initialize Opponent Behavior Predictor model and optimizer"""
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2
    ).to(device)
    
    obp_optimizer = torch.optim.Adam(
        obp_model.parameters(),
        lr=config.OPPONENT_LEARNING_RATE
    )
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
        orig = source_id
        gen = 1
    
    return f"clone_v{gen}_orig_{orig}_{uuid.uuid4().hex[:4]}"

def create_new_agent(device):
    """Create a new agent with full neural architecture"""
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
    
    return {
        'policy_net': policy_net,
        'value_net': value_net,
        'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
        'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
        'entropy_coef': config.INIT_ENTROPY_COEF,
        'architecture': 'LSTM_v1'
    }

def clone_agent(source_agent, source_id, device):
    """Create a clone with architecture preservation"""
    clone_id = generate_agent_name(source_id)
    
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    policy_net.load_state_dict(source_agent['policy_net'].state_dict())
    
    value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    value_net.load_state_dict(source_agent['value_net'].state_dict())
    
    CLONE_REGISTRY[source_id] += 1
    
    return {
        'policy_net': policy_net,
        'value_net': value_net,
        'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
        'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
        'entropy_coef': source_agent['entropy_coef'],
        'lineage': {
            'source': source_id,
            'clone_gen': CLONE_REGISTRY[source_id],
            'created_at': time.time()
        }
    }

def run_tournament(env, device, player_pool, obp_model, logger):
    """Evaluate all agents in Swiss-style tournament"""
    logger.info("Initializing tournament with %d agents...", len(player_pool))
    
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
    
    return final_rankings

def maintain_player_pool_size(player_pool, group_size):
    """Ensure player count is always a multiple of group_size"""
    current_size = len(player_pool)
    remainder = current_size % group_size
    
    if remainder != 0:
        needed = (group_size - remainder) % group_size
        for i in range(needed):
            new_id = generate_agent_name()
            player_pool[new_id] = create_new_agent(torch.device(config.DEVICE))
    
    return player_pool

def cull_and_replace(player_pool, rankings, device, logger):
    """Evolutionary replacement with size maintenance"""
    num_agents = len(rankings)
    num_cull = int(num_agents * CULL_PERCENTAGE)
    
    num_cull = num_cull - (num_cull % GROUP_SIZE)
    num_cull = max(GROUP_SIZE, num_cull)
    
    culled_ids = rankings[-num_cull:]
    logger.info("Culling %d agents: %s...", num_cull, culled_ids[-3:])
    
    for pid in culled_ids:
        del player_pool[pid]
    
    new_players = {}
    for i in range(num_cull):
        if i < int(num_cull * CLONE_PERCENTAGE):
            new_id = generate_agent_name()
            new_players[new_id] = create_new_agent(device)
        else:
            source_id = random.choice(rankings[:GROUP_SIZE*2])
            new_players[generate_agent_name(source_id)] = clone_agent(
                player_pool[source_id], source_id, device
            )
    
    player_pool.update(new_players)
    player_pool = maintain_player_pool_size(player_pool, GROUP_SIZE)
    
    logger.info("Population updated: %d agents (%d new, %d clones)", 
               len(player_pool), int(num_cull * CLONE_PERCENTAGE), 
               num_cull - int(num_cull * CLONE_PERCENTAGE))
    return player_pool

def save_best_checkpoint(player_pool, obp_model, obp_optimizer, max_rating, checkpoint_dir):
    """Save the best checkpoint with current player pool and OBP models."""
    checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
    torch.save({
        'player_pool': {pid: {
            'policy_net': agent['policy_net'].state_dict(),
            'value_net': agent['value_net'].state_dict(),
            'entropy_coef': agent['entropy_coef'],
            'architecture': agent.get('architecture', 'LSTM_v1')
        } for pid, agent in player_pool.items()},
        'obp_model': obp_model.state_dict(),
        'obp_optimizer': obp_optimizer.state_dict(),
        'max_rating': max_rating
    }, checkpoint_path)

def load_best_checkpoint(device, checkpoint_dir):
    """Load the best checkpoint into player_pool and OBP models."""
    checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        return None, None, None, None
    
    checkpoint = torch.load(checkpoint_path)
    player_pool = {}
    for pid, agent_state in checkpoint['player_pool'].items():
        policy_net = PolicyNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        policy_net.load_state_dict(agent_state['policy_net'])
        
        value_net = ValueNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        value_net.load_state_dict(agent_state['value_net'])
        
        player_pool[pid] = {
            'policy_net': policy_net,
            'value_net': value_net,
            'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
            'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
            'entropy_coef': agent_state['entropy_coef'],
            'architecture': agent_state['architecture']
        }
    
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2
    ).to(device)
    obp_model.load_state_dict(checkpoint['obp_model'])
    
    obp_optimizer = torch.optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    obp_optimizer.load_state_dict(checkpoint['obp_optimizer'])
    
    return player_pool, obp_model, obp_optimizer, checkpoint['max_rating']

def main():
    set_seed()
    logger = configure_logger()
    device = torch.device(config.DEVICE)
    
    obp_model, obp_optimizer = initialize_obp(device)
    
    base_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
    config.set_derived_config(
        base_env.observation_spaces[base_env.agents[0]],
        base_env.action_spaces[base_env.agents[0]],
        num_opponents=GROUP_SIZE - 1
    )
    
    player_pool = {f"player_{i}": create_new_agent(device) for i in range(TOTAL_PLAYERS)}
    
    start_batch, loaded_entropy_coefs, obp_state, obp_optim_state = load_multi_checkpoint(
        player_pool, config.CHECKPOINT_DIR, GROUP_SIZE
    )
    
    if loaded_entropy_coefs is not None:
        for agent, coef in loaded_entropy_coefs.items():
            if agent in player_pool:
                player_pool[agent]['entropy_coef'] = coef
    
    if obp_state is not None:
        obp_model.load_state_dict(obp_state)
    if obp_optim_state is not None:
        obp_optimizer.load_state_dict(obp_optim_state)
    
    writer = SummaryWriter(log_dir=config.MULTI_LOG_DIR)
    block_episode_offset = (start_batch - 1) * 2000
    best_max_rating = -float('inf')

    try:
        for batch_id in range(start_batch, 10):
            logger.info("\n=== Starting Batch %d ===", batch_id)
            
            player_pool = maintain_player_pool_size(player_pool, GROUP_SIZE)
            
            all_ids = list(player_pool.keys())
            random.shuffle(all_ids)
            groups = [all_ids[i:i + GROUP_SIZE] for i in range(0, len(all_ids), GROUP_SIZE)]
            
            for group_idx, group in enumerate(groups):
                if len(group) != GROUP_SIZE:
                    logger.error("Invalid group size %d, skipping", len(group))
                    continue
                
                logger.info("Training Group %d: %s", group_idx + 1, group)
                env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
                agent_map = {env.agents[i]: group[i] for i in range(GROUP_SIZE)}
                
                train(
                    agents_dict={pid: player_pool[pid] for pid in group},
                    env=env,
                    device=device,
                    num_episodes=2000,
                    episode_offset=block_episode_offset,
                    log_tensorboard=True,
                    writer=writer,
                    logger=logger,
                    agent_mapping=agent_map,
                    obp_model=obp_model,
                    obp_optimizer=obp_optimizer
                )
            
            block_episode_offset += 2000
            
            if batch_id % TOURNAMENT_INTERVAL == 0:
                logger.info("\n--- Running Evolution Tournament ---")
                tournament_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
                
                temp_pool = maintain_player_pool_size(player_pool.copy(), GROUP_SIZE)
                best_agents, best_obp_model, best_obp_optim, loaded_max_rating = load_best_checkpoint(device, config.CHECKPOINT_DIR)
                if best_agents:
                    for pid in list(best_agents.keys()):
                        new_pid = generate_agent_name(source_id=pid)
                        best_agents[new_pid] = best_agents.pop(pid)
                    temp_pool.update(best_agents)
                    temp_pool = maintain_player_pool_size(temp_pool, GROUP_SIZE)
                
                rankings = run_tournament(tournament_env, device, temp_pool, obp_model, logger)
                
                current_max_rating = max([player['rating'].mu for player in temp_pool.values()])
                
                if current_max_rating > best_max_rating:
                    best_max_rating = current_max_rating
                    save_best_checkpoint(temp_pool, obp_model, obp_optimizer, best_max_rating, config.CHECKPOINT_DIR)
                    logger.info("New best checkpoint saved with rating: %.2f", best_max_rating)
                
                player_pool = cull_and_replace(temp_pool, rankings, device, logger)
                logger.info("New population: %s", list(player_pool.keys())[:6] + ["..."])
            
            save_multi_checkpoint(
                player_pool=player_pool,
                obp_model=obp_model,
                obp_optimizer=obp_optimizer,
                batch=batch_id,
                checkpoint_dir=config.CHECKPOINT_DIR,
                group_size=GROUP_SIZE
            )
            logger.info("Checkpoint saved for batch %d", batch_id)
            
    finally:
        writer.close()
        logger.info("Training session completed")

if __name__ == "__main__":
    main()