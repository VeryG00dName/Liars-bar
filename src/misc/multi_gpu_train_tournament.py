# src/training/multi_gpu_train_tournament.py

import logging
import os
import random
import time
import uuid
import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.training.train_multi_utils import save_multi_checkpoint, load_multi_checkpoint
from src.training.train import train
from src.training.train_extras import set_seed
from src.misc.tune_eval import run_group_swiss_tournament, openskill_model
from src import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

torch.backends.cudnn.benchmark = True

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
    """Initialize Opponent Behavior Predictor model and optimizer on GPU (or CPU if desired)."""
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
    """Generate structured agent names with lineage tracking.
    
    If cloning an existing agent, this function extracts the ultimate original
    name (i.e. the part after the last "_orig_") and counts how many times
    the string "clone_v" appears in the source_id to determine the generation.
    This prevents nested clone prefixes.
    """
    if source_id is None:
        # Use only the last 2 digits of the current epoch time.
        timestamp = str(int(time.time()))[-2:]
        return f"new_{timestamp}_{uuid.uuid4().hex[:6]}"
    
    # If the source_id already includes an "_orig_" marker, extract the ultimate original.
    if "_orig_" in source_id:
        # Split on "_orig_" and take the last piece.
        parts = source_id.split("_orig_")
        ultimate_original_with_suffix = parts[-1]  # e.g. "player_4_95ee"
        # Remove the trailing random suffix (assumed to be 4 hex characters after an underscore)
        original_candidate, sep, suffix = ultimate_original_with_suffix.rpartition('_')
        original = original_candidate if sep else ultimate_original_with_suffix
        # Count the number of times "clone_v" appears in source_id to determine generation.
        num_clones = source_id.count("clone_v")
        new_gen = num_clones + 1
    else:
        original = source_id
        new_gen = 1

    return f"clone_v{new_gen}_orig_{original}_{uuid.uuid4().hex[:4]}"


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
    
    # Create the agent dictionary
    agent = {
        'policy_net': policy_net,
        'value_net': value_net,
        'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
        'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
        'entropy_coef': config.INIT_ENTROPY_COEF,
        'architecture': 'LSTM_v1'
    }
    
    # Initialize a default rating using openskill_model.
    # You can use a unique id or some default value as the agent name.
    agent['rating'] = openskill_model.rating(name=str(uuid.uuid4()))
    return agent

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

    agent = {
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
    # Initialize a default rating for the clone as well.
    agent['rating'] = openskill_model.rating(name=clone_id)
    return agent

def run_tournament(env, device, player_pool, obp_model, logger):
    """Evaluate all agents in Swiss-style tournament on GPU or CPU as desired."""
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
            'games_played': 0
        }
    final_rankings = run_group_swiss_tournament(
        env=env,
        device=device,
        players=players,
        num_games_per_match=11,
        NUM_ROUNDS=5
    )
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
    # Filter rankings to only include agents that are in the current pool.
    filtered_rankings = [pid for pid in rankings if pid in player_pool]
    num_agents = len(filtered_rankings)
    num_cull = int(num_agents * CULL_PERCENTAGE)
    num_cull = num_cull - (num_cull % GROUP_SIZE)
    num_cull = max(GROUP_SIZE, num_cull)
    culled_ids = filtered_rankings[-num_cull:]
    logger.info("Culling %d agents: %s...", num_cull, culled_ids[-3:])
    for pid in culled_ids:
        del player_pool[pid]
    new_players = {}
    for i in range(num_cull):
        if i < int(num_cull * CLONE_PERCENTAGE):
            new_id = generate_agent_name()
            new_players[new_id] = create_new_agent(on_device='cpu')
        else:
            # Choose a source from the top GROUP_SIZE*2 agents in the filtered rankings.
            source_id = random.choice(filtered_rankings[:GROUP_SIZE*2])
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
    """Save the best checkpoint, ensuring the number of agents is a multiple of GROUP_SIZE."""
    num_agents = len(player_pool)
    if num_agents == 0:
        return
    group_size = config.GROUP_SIZE
    top_n_initial = max(1, int(num_agents * 0.1))
    top_n = ((top_n_initial + group_size - 1) // group_size) * group_size
    top_n = min(top_n, num_agents)
    top_n = (top_n // group_size) * group_size
    if top_n == 0:
        return
    sorted_agents = sorted(player_pool.items(), key=lambda x: x[1]['rating'].mu, reverse=True)
    top_agents = dict(sorted_agents[:top_n])
    checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
    torch.save({
        'player_pool': {pid: {
            'policy_net': agent['policy_net'].state_dict(),
            'value_net': agent['value_net'].state_dict(),
            'entropy_coef': agent['entropy_coef'],
            'architecture': agent.get('architecture', 'LSTM_v1'),
            'rating': agent['rating']
        } for pid, agent in top_agents.items()},
        'obp_model': obp_model.state_dict(),
        'obp_optimizer': obp_optimizer.state_dict(),
        'max_rating': max_rating
    }, checkpoint_path)
    logging.info(f"Saved best checkpoint with {top_n} agents (top {top_n/num_agents*100:.1f}%).")

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
            'rating': agent_state['rating']
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
    """Trim the pool to the top total_players, ensuring it's a multiple of GROUP_SIZE."""
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

def average_state_dicts(state_dicts):
    """Averages a list of state_dicts by summing and dividing by the number of dicts."""
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_tensor = state_dicts[0][key].clone().float()
        for sd in state_dicts[1:]:
            avg_tensor.add_(sd[key].clone().float())
        avg_tensor.div_(len(state_dicts))
        avg_state[key] = avg_tensor
    return avg_state

def train_group_process(group, agent_group_state, block_episode_offset, gpu_id, obp_state):
    """Runs in a subprocess. Trains agents in `group` on GPU `gpu_id` and logs progress."""
    logger = logging.getLogger(f"TrainGroup-{gpu_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logger.info(f"Starting training for Group on GPU {gpu_id}: {group}")
    try:
        device = torch.device(f"cuda:{gpu_id}")
        local_obp_model, local_obp_optimizer = initialize_obp(device)
        local_obp_model.load_state_dict(obp_state)
        for pid, agent in agent_group_state.items():
            move_agent_to_device(agent, device)
        env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
        agent_map = {env.agents[i]: group[i] for i in range(GROUP_SIZE)}
        logger.info("Training started.")
        train(
            agents_dict=agent_group_state,
            env=env,
            device=device,
            num_episodes=2000,
            episode_offset=block_episode_offset,
            log_tensorboard=False,
            writer=None,
            logger=logger,
            agent_mapping=agent_map,
            obp_model=local_obp_model,
            obp_optimizer=local_obp_optimizer
        )
        logger.info("Training finished.")
        for pid, agent in agent_group_state.items():
            move_agent_to_device(agent, 'cpu')
        local_obp_model.to('cpu')
        updated_obp_state = local_obp_model.state_dict()
        logger.info(f"Finished training for Group on GPU {gpu_id}: {group}")
        return {'agents': agent_group_state, 'obp_state': updated_obp_state}
    except Exception as e:
        logger.error(f"Error in training group on GPU {gpu_id}: {str(e)}", exc_info=True)
        raise

def main():
    set_seed()
    logger = configure_logger()
    device = torch.device(config.DEVICE)
    obp_model, obp_optimizer = initialize_obp(device)
    
    # Initialize environment and config
    base_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
    config.set_derived_config(
        base_env.observation_spaces[base_env.agents[0]],
        base_env.action_spaces[base_env.agents[0]],
        num_opponents=GROUP_SIZE - 1
    )
    
    # Initialize player pool
    player_pool = {f"player_{i}": create_new_agent(on_device='cpu') for i in range(TOTAL_PLAYERS)}
    
    # Load checkpoints if available
    start_batch = 1
    best_max_rating = -float('inf')
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
    
    try:
        for batch_id in range(start_batch, 20):
            logger.info("\n=== Starting Batch %d ===", batch_id)
            
            # Ensure valid group sizes
            player_pool = maintain_player_pool_size(player_pool, GROUP_SIZE)
            
            # Split into groups and train
            all_ids = list(player_pool.keys())
            random.shuffle(all_ids)
            groups = [all_ids[i:i + GROUP_SIZE] for i in range(0, len(all_ids), GROUP_SIZE)]
            
            # Train groups on available GPUs
            available_gpus = list(range(torch.cuda.device_count()))
            if not available_gpus:
                available_gpus = [0]  # Fallback to CPU if needed
                
            futures = []
            with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
                for group_idx, group in enumerate(groups):
                    if len(group) != GROUP_SIZE:
                        logger.error("Invalid group size %d, skipping", len(group))
                        continue
                        
                    gpu_id = available_gpus[group_idx % len(available_gpus)]
                    logger.info("Submitting training for Group %d on GPU %d: %s", 
                               group_idx + 1, gpu_id, group)
                    
                    # Deep copy agents for isolated training
                    agent_group_state = {pid: copy.deepcopy(player_pool[pid]) for pid in group}
                    
                    future = executor.submit(
                        train_group_process,
                        group,
                        agent_group_state,
                        block_episode_offset,
                        gpu_id,
                        obp_model.state_dict()
                    )
                    futures.append(future)
                
                # Collect results and update OBP
                obp_states = []
                for future in as_completed(futures):
                    result = future.result()
                    updated_agents = result['agents']
                    
                    # Merge trained agents back into pool
                    for pid, agent in updated_agents.items():
                        player_pool[pid] = agent
                    
                    # Collect OBP states for averaging
                    obp_states.append(result['obp_state'])
                
                # Average OBP models from all groups
                if obp_states:
                    new_obp_state = average_state_dicts(obp_states)
                    obp_model.load_state_dict(new_obp_state)
                    obp_optimizer = torch.optim.Adam(obp_model.parameters(), 
                                                   lr=config.OPPONENT_LEARNING_RATE)
            
            block_episode_offset += 2000
            
            # Run tournament every TOURNAMENT_INTERVAL batches
            if batch_id % TOURNAMENT_INTERVAL == 0:
                logger.info("\n--- Running Evolution Tournament ---")
                tournament_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
                
                # Create temporary pool with current agents
                temp_pool = player_pool.copy()
                temp_pool = maintain_player_pool_size(temp_pool, GROUP_SIZE)
                
                # Track merged historical agents
                merged_old_best_ids = set()
                
                # Merge best historical agents into tournament
                best_agents, best_obp_model, best_obp_optim, loaded_max_rating = load_best_checkpoint(device, config.CHECKPOINT_DIR)
                if best_agents:
                    # Rename and merge historical best agents
                    for pid in list(best_agents.keys()):
                        new_pid = generate_agent_name(source_id=pid)
                        temp_pool[new_pid] = best_agents[pid]
                        merged_old_best_ids.add(new_pid)
                    temp_pool = maintain_player_pool_size(temp_pool, GROUP_SIZE)
                
                # Run tournament
                rankings = run_tournament(tournament_env, device, temp_pool, obp_model, logger)
                
                # Update ratings for all agents
                for pid, agent in temp_pool.items():
                    if 'rating' not in agent:
                        agent['rating'] = openskill_model.rating(name=pid)
                
                # Determine if top agent is new
                current_max_rating = max([agent['rating'].mu for agent in temp_pool.values()])
                top_agent_id = rankings[0]
                is_top_agent_old = top_agent_id in merged_old_best_ids
                
                # Evolutionary update
                trimmed_pool = trim_pool_by_rankings(temp_pool, rankings, TOTAL_PLAYERS, logger)
                player_pool = cull_and_replace(trimmed_pool, rankings, device, logger)
                
                # New checkpoint condition
                if (current_max_rating > best_max_rating) or (not is_top_agent_old):
                    best_max_rating = max(best_max_rating, current_max_rating)
                    save_best_checkpoint(player_pool, obp_model, obp_optimizer, best_max_rating, config.CHECKPOINT_DIR)
                    logger.info(f"New best checkpoint saved. Reason: {'new top agent' if not is_top_agent_old else 'higher rating (%.2f)' % current_max_rating}")
            
            # Save regular checkpoint
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
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()