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
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.training.train_multi_utils import save_multi_checkpoint, load_multi_checkpoint
from src.training.train import train
from src.training.train_extras import set_seed
from tune.tune_eval import run_group_swiss_tournament, openskill_model
from src import config

# For visualization
import networkx as nx
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
torch.backends.cudnn.benchmark = True

# Tournament configuration
TOURNAMENT_INTERVAL = config.TOURNAMENT_INTERVAL
CULL_PERCENTAGE = config.CULL_PERCENTAGE
CLONE_PERCENTAGE = config.CLONE_PERCENTAGE
GROUP_SIZE = config.GROUP_SIZE
TOTAL_PLAYERS = config.TOTAL_PLAYERS

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

def move_agent_to_device(agent, device):
    """Helper to move an agent's networks to the specified device."""
    agent['policy_net'].to(device)
    agent['value_net'].to(device)
    move_optimizer_state(agent['optimizer_policy'], device)
    move_optimizer_state(agent['optimizer_value'], device)

def move_pool_to_device(player_pool, device):
    """Move an entire player pool's agents to the specified device."""
    for pid, agent in player_pool.items():
        move_agent_to_device(agent, device)

def move_optimizer_state(optimizer, device):
    """Move all state tensors in an optimizer to a given device."""
    for param_key, param_state in optimizer.state.items():
        for state_key, state_value in param_state.items():
            if torch.is_tensor(state_value):
                param_state[state_key] = state_value.to(device)

def initialize_obp(device):
    """Initialize Opponent Behavior Predictor model and optimizer on the given device."""
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM, 
        hidden_dim=config.OPPONENT_HIDDEN_DIM, 
        output_dim=2,
        memory_dim=config.STRATEGY_DIM  # Transformer memory embedding dimension.
    ).to(device)
    obp_optimizer = torch.optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    return obp_model, obp_optimizer

def generate_agent_name(source_id=None):
    """Generate structured agent names with lineage tracking."""
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

def create_new_agent(on_device='cpu'):
    """Create a new agent on CPU by default."""
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=True,
        use_layer_norm=True
    ).to(on_device)
    value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        use_dropout=True,
        use_layer_norm=True
    ).to(on_device)
    return {
        'policy_net': policy_net,
        'value_net': value_net,
        'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
        'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
        'entropy_coef': config.INIT_ENTROPY_COEF,
        'architecture': 'LSTM_v1'
    }

def clone_agent(source_agent, source_id, on_device='cpu'):
    """Create a clone with architecture preservation on CPU by default."""
    clone_id = generate_agent_name(source_id)
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=True,
        use_layer_norm=True
    ).to(on_device)
    policy_net.load_state_dict(source_agent['policy_net'].state_dict())
    value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        use_dropout=True,
        use_layer_norm=True
    ).to(on_device)
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
    """Evaluate all agents in a Swiss-style tournament."""
    # Move agents to device for evaluation
    for pid, agent in player_pool.items():
        move_agent_to_device(agent, device)
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
        'games_played': 0,
        'obs_version': agent.get('obs_version', 2),
        'uses_memory': True
    }
    final_rankings = run_group_swiss_tournament(
        env=env,
        device=device,
        players=players,
        num_games_per_match=11,
        NUM_ROUNDS=5
    )
    # Move agents back to CPU after evaluation
    for pid, agent in player_pool.items():
        move_agent_to_device(agent, 'cpu')
    return final_rankings

def maintain_player_pool_size(player_pool, group_size):
    """Ensure player count is always a multiple of group_size."""
    current_size = len(player_pool)
    remainder = current_size % group_size
    if remainder != 0:
        needed = (group_size - remainder) % group_size
        for i in range(needed):
            new_id = generate_agent_name()
            player_pool[new_id] = create_new_agent(on_device='cpu')
    return player_pool

def cull_and_replace(player_pool, rankings, device, logger):
    """Evolutionary replacement with size maintenance."""
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
            new_players[new_id] = create_new_agent(on_device='cpu')
        else:
            source_id = random.choice(rankings[:GROUP_SIZE*2])
            new_players[generate_agent_name(source_id)] = clone_agent(
                player_pool[source_id], source_id, on_device='cpu'
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
            'architecture': agent.get('architecture', 'LSTM_v1'),
            'rating': agent.get('rating')
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
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    player_pool = {}
    for pid, agent_state in checkpoint['player_pool'].items():
        policy_net = PolicyNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True
        ).to('cpu')
        policy_net.load_state_dict(agent_state['policy_net'])
        
        value_net = ValueNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            use_dropout=True,
            use_layer_norm=True
        ).to('cpu')
        value_net.load_state_dict(agent_state['value_net'])
        
        player_pool[pid] = {
            'policy_net': policy_net,
            'value_net': value_net,
            'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
            'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
            'entropy_coef': agent_state['entropy_coef'],
            'architecture': agent_state['architecture'],
            'rating': agent_state.get('rating'),
            'obs_version': agent_state.get('obs_version', getattr(config, 'OBS_VERSION', 2))
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

def trim_pool_by_rankings(player_pool, rankings, total_players, logger):
    """
    Trim the pool to the top total_players, ensuring the pool size is a multiple of GROUP_SIZE.
    """
    filtered_rankings = [pid for pid in rankings if pid in player_pool]
    if len(filtered_rankings) >= total_players:
        top_ids = filtered_rankings[:total_players]
        trimmed_pool = {pid: player_pool[pid] for pid in top_ids}
        logger.info("Trimmed pool from %d to %d players.", len(player_pool), len(trimmed_pool))
        return trimmed_pool
    else:
        while len(player_pool) < total_players:
            new_pid = generate_agent_name()
            player_pool[new_pid] = create_new_agent(on_device='cpu')
        logger.info("Expanded pool to %d players.", total_players)
        return player_pool

#######################
# Lineage Visualization
#######################
def build_lineage_graph(player_pool, output_file='lineage_graph.png'):
    """
    Build and save a lineage graph of the agents using NetworkX.
    Each node represents an agent, and an edge from A to B indicates that B was cloned from A.
    """
    G = nx.DiGraph()
    # Add nodes for all agents.
    for pid in player_pool.keys():
        G.add_node(pid)
    # Add edges for cloned agents.
    for pid, agent in player_pool.items():
        if 'lineage' in agent:
            source = agent['lineage'].get('source')
            if source is not None:
                G.add_edge(source, pid)
    # Draw the graph.
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True, font_size=8)
    plt.title('Agent Lineage Graph')
    plt.savefig(output_file)
    plt.close()
    print(f"Lineage graph saved to {output_file}")

#######################
# Main Training Loop
#######################
def main():
    set_seed()
    logger = configure_logger()
    device = torch.device(config.DEVICE)  # e.g., 'cuda' or 'cpu'
    obp_model, obp_optimizer = initialize_obp(device)
    
    base_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
    config.set_derived_config(
        base_env.observation_spaces[base_env.agents[0]],
        base_env.action_spaces[base_env.agents[0]],
        num_opponents=GROUP_SIZE - 1
    )
    
    player_pool = {f"player_{i}": create_new_agent(on_device='cpu') for i in range(TOTAL_PLAYERS)}
    
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
        for batch_id in range(start_batch, 20):
            logger.info("\n=== Starting Batch %d ===", batch_id)
            player_pool = maintain_player_pool_size(player_pool, GROUP_SIZE)
            all_ids = list(player_pool.keys())
            random.shuffle(all_ids)
            groups = [all_ids[i:i + GROUP_SIZE] for i in range(0, len(all_ids), GROUP_SIZE)]
            
            # Sequentially train each group on the designated device
            for group_idx, group in enumerate(groups):
                if len(group) != GROUP_SIZE:
                    logger.error("Invalid group size %d, skipping", len(group))
                    continue
                logger.info("Training Group %d: %s", group_idx + 1, group)
                env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
                agent_map = {env.agents[i]: group[i] for i in range(GROUP_SIZE)}
                # Move agents in the group to the training device
                for pid in group:
                    move_agent_to_device(player_pool[pid], device)
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
                # Move trained agents back to CPU
                for pid in group:
                    move_agent_to_device(player_pool[pid], 'cpu')
            
            block_episode_offset += 2000
            
            if batch_id % TOURNAMENT_INTERVAL == 0:
                logger.info("\n--- Running Evolution Tournament ---")
                tournament_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
                temp_pool = player_pool.copy()
                temp_pool = maintain_player_pool_size(temp_pool, GROUP_SIZE)
                best_agents, best_obp_model, best_obp_optim, loaded_max_rating = load_best_checkpoint(device, config.CHECKPOINT_DIR)
                if best_agents:
                    for pid in list(best_agents.keys()):
                        new_pid = generate_agent_name(source_id=pid)
                        best_agents[new_pid] = best_agents.pop(pid)
                    temp_pool.update(best_agents)
                    temp_pool = maintain_player_pool_size(temp_pool, GROUP_SIZE)
                rankings = run_tournament(tournament_env, device, temp_pool, obp_model, logger)
                for pid, agent in temp_pool.items():
                    if 'rating' not in agent:
                        agent['rating'] = openskill_model.rating(name=pid)
                current_max_rating = max([agent['rating'].mu for agent in temp_pool.values()])
                if current_max_rating > best_max_rating:
                    best_max_rating = current_max_rating
                    save_best_checkpoint(temp_pool, obp_model, obp_optimizer, best_max_rating, config.CHECKPOINT_DIR)
                    logger.info("New best checkpoint saved with rating: %.2f", best_max_rating)
                trimmed_pool = trim_pool_by_rankings(temp_pool, rankings, TOTAL_PLAYERS, logger)
                filtered_rankings = [pid for pid in rankings if pid in trimmed_pool]
                player_pool = cull_and_replace(trimmed_pool, filtered_rankings, device, logger)
                logger.info("New population after evolutionary replacement: %s", list(player_pool.keys())[:6] + ["..."])
            
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
        # Build and save the lineage graph for visualization.
        build_lineage_graph(player_pool, output_file='lineage_graph.png')
        logger.info("Lineage graph has been built and saved.")

if __name__ == "__main__":
    main()
