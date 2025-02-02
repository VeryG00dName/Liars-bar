# src/training/train_league.py
import logging
import os
import random
import time
import uuid
from collections import deque
import torch

# Import your existing modules and functions
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.training.train_multi_utils import save_multi_checkpoint, load_multi_checkpoint
from src.training.train_extras import set_seed
from src.misc.tune_eval import run_group_swiss_tournament, openskill_model
from src.training.train import train
from src import config

# Use a smaller set of agents to stay under 32GB.
# For example, if each model takes ~333MB, 80 agents is ~26.6GB.
MAX_TOTAL_AGENTS = 80

# League configuration
MAIN_LEAGUE_SIZE = 10
TRAINING_LEAGUE_SIZE = 20
HISTORICAL_POOL_SIZE = 15  # max number of historical snapshots to keep
TOURNAMENT_INTERVAL = 2   # batches between tournaments
PROMOTION_INTERVAL = 5    # batches between league promotions/demotions
GROUP_SIZE = 3            # number of agents per training match

# Global agent registry (agent_id -> agent state)
agent_registry = {}

# --- Helper functions for device management ---

def move_agent_to_device(agent, device):
    """Move agent networks, optimizer states, and OBP (if present) to the specified device."""
    agent['policy_net'] = agent['policy_net'].to(device)
    agent['value_net'] = agent['value_net'].to(device)
    move_optimizer_state(agent['optimizer_policy'], device)
    move_optimizer_state(agent['optimizer_value'], device)
    # Move OBP state if present in the agent dict (rarely stored per agent)
    if 'obp_model' in agent:
        agent['obp_model'] = agent['obp_model'].to(device)
    if 'obp_optimizer' in agent:
        move_optimizer_state(agent['obp_optimizer'], device)

def move_optimizer_state(optimizer, device):
    """Move all optimizer state tensors to the specified device."""
    for param_key, param_state in optimizer.state.items():
        for state_key, state_value in param_state.items():
            if torch.is_tensor(state_value):
                optimizer.state[param_key][state_key] = state_value.to(device)

def move_obp_to_device(obp_model, obp_optimizer, device):
    """Move OBP model and its optimizer state to the specified device."""
    obp_model.to(device)
    move_optimizer_state(obp_optimizer, device)

# --- OBP Initialization ---
def initialize_obp(device):
    """Initialize OBP model and optimizer on the given device."""
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2
    ).to(device)
    obp_optimizer = torch.optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    return obp_model, obp_optimizer

# --- Agent Creation and Cloning ---

def generate_agent_name(source_id=None):
    """Generate a structured agent name with lineage tracking."""
    if source_id is None:
        return f"agent_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    else:
        return f"clone_{source_id}_{uuid.uuid4().hex[:4]}"

def create_new_agent(device='cpu'):
    """Create a new agent (on device) using your model definitions."""
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
        'role': None  # will be assigned later
    }

def clone_agent(source_agent, source_id, device='cpu'):
    """Clone an agent's state (deep copy of networks and optimizers)."""
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
    return {
        'policy_net': policy_net,
        'value_net': value_net,
        'optimizer_policy': torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE),
        'optimizer_value': torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE),
        'entropy_coef': source_agent['entropy_coef'],
        'role': None
    }

# --- League Management ---
def initialize_leagues():
    """Create league structure. The leagues hold only agent IDs."""
    leagues = {
        'main_league': set(),         # best agents
        'training_league': set(),     # agents training to become main
        'historical_pool': deque(maxlen=HISTORICAL_POOL_SIZE)  # past snapshots
    }
    return leagues

def add_agent_to_registry(agent, league_set, agent_id=None):
    """Add an agent to the global registry and a league."""
    if agent_id is None:
        agent_id = generate_agent_name()
    agent_registry[agent_id] = agent
    league_set.add(agent_id)
    return agent_id

def promote_demote_agents(leagues, tournament_results, logger):
    """
    Based on tournament rankings (a list of agent IDs sorted best-first),
    promote top training league agents to the main league and demote worst main league agents.
    """
    main_agents = [pid for pid in tournament_results if pid in leagues['main_league']]
    training_agents = [pid for pid in tournament_results if pid in leagues['training_league']]
    
    demote_count = max(1, int(len(main_agents) * 0.2))
    for pid in main_agents[-demote_count:]:
        leagues['main_league'].remove(pid)
        leagues['training_league'].add(pid)
        logger.info(f"Demoted {pid} from main to training league.")
    
    promote_candidates = training_agents[:demote_count]
    for pid in promote_candidates:
        if len(leagues['main_league']) < MAIN_LEAGUE_SIZE:
            leagues['training_league'].remove(pid)
            leagues['main_league'].add(pid)
            logger.info(f"Promoted {pid} from training to main league.")

def league_matchmaking(leagues, group_size):
    """
    Select a mixed set of agents for a match.
    We form a candidate pool from main, training, and historical agents.
    """
    candidate_ids = list(leagues['main_league'] | leagues['training_league'] | set(leagues['historical_pool']))
    if len(candidate_ids) >= group_size:
        selected = random.sample(candidate_ids, group_size)
    else:
        selected = candidate_ids
    return selected

def update_historical_pool(leagues, top_agent_ids):
    """
    Add a few of the top performers (by ID) to the historical pool.
    """
    for pid in top_agent_ids:
        if pid not in leagues['historical_pool']:
            leagues['historical_pool'].append(pid)

# --- Training Functions ---
def train_mixed_group(leagues, group_ids, device, writer, logger, obp_model, obp_optimizer):
    """
    Train a group of agents against a mixed set of opponents.
    Each agent's temporary clone and its opponents are moved to GPU for training and then returned to CPU.
    """
    temp_agents = {}
    for pid in group_ids:
        source_agent = agent_registry[pid]
        temp_agents[pid] = clone_agent(source_agent, pid, device=device)
        move_agent_to_device(temp_agents[pid], device)
    
    # For each agent, sample opponents and train.
    for pid in group_ids:
        opponent_ids = league_matchmaking(leagues, group_size=4)
        opponent_ids = [opp for opp in opponent_ids if opp != pid]
        opponents = {opp: agent_registry[opp] for opp in opponent_ids}
        for opp in opponents.values():
            move_agent_to_device(opp, device)
        
        env = LiarsDeckEnv(num_players=1 + len(opponents), render_mode=None)
        seats = env.agents
        agent_map = {seats[0]: pid}
        for seat, opp in zip(seats[1:], opponents.keys()):
            agent_map[seat] = opp

        # Pass the OBP model and optimizer along with agents.
        train(
            agents_dict={pid: temp_agents[pid] for pid in [pid] + list(opponents.keys())},
            env=env,
            device=device,
            num_episodes=2000,
            episode_offset=0,
            log_tensorboard=False,
            writer=writer,
            logger=logger,
            agent_mapping=agent_map,
            obp_model=obp_model,       
            obp_optimizer=obp_optimizer
        )
        
        for opp in opponents.values():
            move_agent_to_device(opp, 'cpu')
    
    for pid in group_ids:
        move_agent_to_device(temp_agents[pid], 'cpu')
        agent_registry[pid] = temp_agents[pid]

def train_training_league(leagues, device, writer, logger, obp_model, obp_optimizer):
    """
    Train all agents in the training league in a focused curriculum.
    Each group is moved to GPU for training then returned to CPU.
    """
    training_ids = list(leagues['training_league'])
    random.shuffle(training_ids)
    groups = [training_ids[i:i+GROUP_SIZE] for i in range(0, len(training_ids), GROUP_SIZE)]
    for group in groups:
        if len(group) != GROUP_SIZE:
            continue
        env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
        agent_map = {env.agents[i]: group[i] for i in range(GROUP_SIZE)}
        for pid in group:
            move_agent_to_device(agent_registry[pid], device)
        train(
            agents_dict={pid: agent_registry[pid] for pid in group},
            env=env,
            device=device,
            num_episodes=2000,
            episode_offset=0,
            log_tensorboard=False,
            writer=writer,
            logger=logger,
            agent_mapping=agent_map,
            obp_model=obp_model,
            obp_optimizer=obp_optimizer
        )
        for pid in group:
            move_agent_to_device(agent_registry[pid], 'cpu')

def run_cross_league_tournament(leagues, device, logger, obp_model):
    """
    Run a tournament across all leagues using the Swiss-style tournament function.
    OBP is provided (its optimizer is not used during tournament evaluation).
    Returns a ranking (list of agent IDs sorted best-first).
    """
    combined_ids = list(leagues['main_league'] | leagues['training_league'] | set(leagues['historical_pool']))
    temp_pool = {pid: agent_registry[pid] for pid in combined_ids}
    temp_pool = maintain_player_pool_size(temp_pool, GROUP_SIZE)
    tournament_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=None)
    for pid in temp_pool:
        move_agent_to_device(temp_pool[pid], device)
    rankings = run_group_swiss_tournament(
        env=tournament_env,
        device=device,
        players={pid: {
            'policy_net': temp_pool[pid]['policy_net'],
            'value_net': temp_pool[pid]['value_net'],
            'obp_model': obp_model,  # OBP model is shared
            'rating': openskill_model.rating(name=pid),
            'score': 0.0,
            'wins': 0,
            'games_played': 0
        } for pid in temp_pool},
        num_games_per_match=11,
        NUM_ROUNDS=5
    )
    logger.info("Tournament top performers: %s", rankings[:10])
    for pid in temp_pool:
        move_agent_to_device(temp_pool[pid], 'cpu')
    return rankings

# --- Main Training Loop ---
def main():
    set_seed()
    logger = logging.getLogger("TrainLeague")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())

    device = torch.device(config.DEVICE)  # e.g., 'cuda'
    cpu_device = torch.device('cpu')
    writer = None  # Replace with SummaryWriter(log_dir=...) if using tensorboard logging

    # Initialize the OBP model and optimizer.
    obp_model, obp_optimizer = initialize_obp(device)
    # Initially move OBP to CPU.
    move_obp_to_device(obp_model, obp_optimizer, cpu_device)

    leagues = initialize_leagues()

    # Initialize training league with new agents.
    for i in range(TRAINING_LEAGUE_SIZE):
        new_agent = create_new_agent(device='cpu')
        add_agent_to_registry(new_agent, leagues['training_league'])
    
    start_batch = 1  # or load from checkpoint if resuming

    try:
        for batch_id in range(start_batch, 1000):
            logger.info(f"\n=== Batch {batch_id} ===")
            
            # 1. Train a mixed group from the full league.
            mixed_group_ids = league_matchmaking(leagues, GROUP_SIZE)
            logger.info(f"Mixed group training for agents: {mixed_group_ids}")
            # Move OBP to GPU before training.
            move_obp_to_device(obp_model, obp_optimizer, device)
            train_mixed_group(leagues, mixed_group_ids, device, writer, logger, obp_model, obp_optimizer)
            # Move OBP back to CPU after training.
            move_obp_to_device(obp_model, obp_optimizer, cpu_device)
            
            # 2. Train agents in the training league separately.
            move_obp_to_device(obp_model, obp_optimizer, device)
            train_training_league(leagues, device, writer, logger, obp_model, obp_optimizer)
            move_obp_to_device(obp_model, obp_optimizer, cpu_device)
            
            # 3. Every few batches, run a cross-league tournament.
            if batch_id % TOURNAMENT_INTERVAL == 0:
                move_obp_to_device(obp_model, obp_optimizer, device)
                rankings = run_cross_league_tournament(leagues, device, logger, obp_model)
                move_obp_to_device(obp_model, obp_optimizer, cpu_device)
                update_historical_pool(leagues, rankings[:5])
                
                if batch_id % PROMOTION_INTERVAL == 0:
                    promote_demote_agents(leagues, rankings, logger)
                    save_multi_checkpoint(
                        player_pool={pid: agent_registry[pid] for pid in leagues['main_league']},
                        obp_model=obp_model, 
                        obp_optimizer=obp_optimizer,
                        batch=batch_id,
                        checkpoint_dir=config.CHECKPOINT_DIR,
                        group_size=GROUP_SIZE
                    )
                    logger.info("League checkpoint saved.")
                    
            total_agents = len(agent_registry)
            if total_agents < MAX_TOTAL_AGENTS:
                new_agent = create_new_agent(device='cpu')
                add_agent_to_registry(new_agent, leagues['training_league'])
                logger.info("Added a new training league agent.")
    
    except Exception as e:
        logger.exception("Training interrupted due to error: %s", e)
    
    logger.info("Training session completed.")

# Helper: ensure pool size is a multiple of group_size.
def maintain_player_pool_size(player_pool, group_size):
    pool = player_pool.copy()
    remainder = len(pool) % group_size
    if remainder != 0:
        needed = group_size - remainder
        for i in range(needed):
            dummy_agent = create_new_agent(device='cpu')
            dummy_id = generate_agent_name()
            pool[dummy_id] = dummy_agent
        # Note: Dummy agents may be removed after tournament evaluation.
    return pool

if __name__ == "__main__":
    main()
