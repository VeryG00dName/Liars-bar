# src/tests/gpu_test.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

# Suppress PyTorch UserWarnings (e.g., batch_first issue)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# Suppress PyTorch FutureWarnings (e.g., torch.load weights_only=True warning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random

# Imports from your project
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.reward_restriction_wrapper import RewardRestrictionWrapper
from src.training.train_utils import compute_gae, train_obp
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor, StrategyTransformer
from src.model.memory import RolloutMemory
from src.training.train_extras import set_seed, extract_obp_training_data, run_obp_inference
from src.env.liars_deck_env_utils import query_opponent_memory_full
from src import config

from torch.amp import autocast, GradScaler  # Updated import for GradScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
torch.backends.cudnn.benchmark = True

# Set up logger for this script.
logger = logging.getLogger("gpu_test")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

########################################
# Helper: Convert memory events into 4D feature vectors
########################################
def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    Each event is expected to be a dictionary with keys: "response", "triggering_action", "penalties", and "card_count".
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        # Map categorical features to indices (or floats)
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

########################################
# Benchmark configuration overrides
########################################
HIDDEN_DIM_LIST = [128, 512, 1024, 1504, 2000]
NUM_EPISODES = 100         # Total episodes to run during the benchmark
UPDATE_STEPS = 5           # Frequency (in episodes) of the PPO update phase
K_EPOCHS = 4               # Number of PPO epochs per update
OBP_UPDATE_STEPS = 5       # Frequency (in episodes) to train OBP
OBP_BATCH_SIZE = 200       # Batch size for OBP training
OBP_TRAINING_THRESHOLD = 100  # Minimum OBP data points before training

########################################
# Benchmark function using your real environment with OBP and Transformer integration with profiling
########################################
def measure_training_speed(hidden_dim, device, use_cudnn_benchmark=True, use_mixed_precision=False):
    torch.backends.cudnn.benchmark = use_cudnn_benchmark
    scaler = GradScaler(enabled=use_mixed_precision)
    
    # Initialize environment and measure reset time
    t0 = time.perf_counter()
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, _ = env.reset()
    reset_time = time.perf_counter() - t0
    logger.info(f"Initial environment reset time: {reset_time:.4f}s")
    
    agents = env.agents
    
    # Initialize networks
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    
    value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    
    # Updated OBP model initialization with memory_dim and hidden_dim
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM, 
        hidden_dim=config.OPPONENT_HIDDEN_DIM, 
        output_dim=2,
        memory_dim=config.STRATEGY_DIM
    ).to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.001)
    optimizer_value = optim.Adam(value_net.parameters(), lr=0.001)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=0.001)
    
    # Initialize rollout memory and OBP memory
    memory = RolloutMemory(agents)
    obp_memory = []
    total_steps = 0
    overall_start = time.perf_counter()
    
    # --- Instantiate Transformer and Event Encoder (for transformer memory integration) ---
    from src.training.train_transformer import EventEncoder
    strategy_transformer = StrategyTransformer(
        num_tokens=config.STRATEGY_NUM_TOKENS,
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
        nhead=config.STRATEGY_NHEAD,
        num_layers=config.STRATEGY_NUM_LAYERS,
        strategy_dim=config.STRATEGY_DIM,
        num_classes=config.STRATEGY_NUM_CLASSES,
        dropout=config.STRATEGY_DROPOUT,
        use_cls_token=True
    ).to(device)
    
    transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
    if os.path.exists(transformer_checkpoint_path):
        t_chk_start = time.perf_counter()
        checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
        strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=False)
        if "response2idx" in checkpoint and "action2idx" in checkpoint:
            response2idx = checkpoint["response2idx"]
            action2idx = checkpoint["action2idx"]
        else:
            raise ValueError("Transformer checkpoint is missing categorical mappings.")
        event_encoder = EventEncoder(
            response_vocab_size=len(response2idx),
            action_vocab_size=len(action2idx),
            token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
        ).to(device)
        event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
        event_encoder.eval()
        t_chk = time.perf_counter() - t_chk_start
        logger.info(f"Transformer checkpoint load time: {t_chk:.4f}s")
    else:
        # Fallback to defaults if checkpoint is not found.
        response2idx = {}
        action2idx = {}
        event_encoder = EventEncoder(
            response_vocab_size=config.STRATEGY_NUM_TOKENS,
            action_vocab_size=config.STRATEGY_NUM_TOKENS,
            token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
        ).to(device)
    # Override transformer's token embedding and remove classification head
    strategy_transformer.token_embedding = torch.nn.Identity()
    strategy_transformer.classification_head = None
    strategy_transformer.eval()
    
    # Profiling accumulators
    total_env_reset_time = 0.0
    total_obp_transformer_time = 0.0
    total_action_time = 0.0
    total_ppo_update_time = 0.0
    total_obp_training_time = 0.0

    for episode in range(NUM_EPISODES):
        t_episode_start = time.perf_counter()
        t_reset = time.perf_counter()
        obs, _ = env.reset()
        total_env_reset_time += time.perf_counter() - t_reset
        
        env.agents = agents  # Maintain original agent order
        
        while env.agent_selection is not None:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            
            # Simplified observation processing
            t_obs = time.perf_counter()
            observation = env.observe(agent)[agent]
            action_mask = env.infos[agent]['action_mask']
            t_obs_done = time.perf_counter()
            
            # --- Compute OBP and Transformer Integration ---
            t_transformer_start = time.perf_counter()
            obp_memory_embeddings = []
            transformer_embeddings = []
            for opp in env.possible_agents:
                if opp == agent:
                    continue
                mem_summary = query_opponent_memory_full(agent, opp)
                features_list = convert_memory_to_features(mem_summary, response2idx, action2idx)
                if features_list:
                    feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        projected = event_encoder(feature_tensor)
                        strategy_embedding, _ = strategy_transformer(projected)
                else:
                    strategy_embedding = None
                if strategy_embedding is not None:
                    obp_memory_embeddings.append(strategy_embedding)
                    transformer_embeddings.append(strategy_embedding.cpu().detach().numpy().flatten())
                else:
                    zero_emb = torch.zeros(1, config.STRATEGY_DIM, device=device)
                    obp_memory_embeddings.append(zero_emb)
                    transformer_embeddings.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))
            with torch.no_grad():
                obp_probs = run_obp_inference(
                    obp_model, observation, device, env.num_players, 
                    memory_embeddings=obp_memory_embeddings
                )
            total_obp_transformer_time += time.perf_counter() - t_transformer_start
            
            # Normalize transformer embeddings using min–max normalization.
            if transformer_embeddings:
                embeddings_arr = np.concatenate(transformer_embeddings, axis=0)
                min_val = embeddings_arr.min()
                max_val = embeddings_arr.max()
                if max_val - min_val == 0:
                    normalized_transformer_features = embeddings_arr
                else:
                    normalized_transformer_features = (embeddings_arr - min_val) / (max_val - min_val)
            else:
                normalized_transformer_features = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
            
            # --- Form the final observation ---
            final_obs = np.concatenate([
                observation,
                np.array(obp_probs, dtype=np.float32),
                normalized_transformer_features
            ], axis=0)
            assert final_obs.shape[0] == config.INPUT_DIM, \
                f"Expected observation dimension {config.INPUT_DIM}, got {final_obs.shape[0]}"
            obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # --- Action Selection ---
            t_action_start = time.perf_counter()
            with autocast(device_type='cuda', enabled=use_mixed_precision):
                probs, _, _ = policy_net(obs_tensor, None)
            probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
            action_mask_tensor = torch.tensor(action_mask, device=device, dtype=torch.float32)
            masked_probs = probs * action_mask_tensor
            if masked_probs.sum() == 0:
                masked_probs = action_mask_tensor.float()
            masked_probs /= masked_probs.sum()
            action = Categorical(masked_probs).sample().item()
            env.step(action)
            total_steps += 1
            total_action_time += time.perf_counter() - t_action_start
            
            # Store transition (including action_mask now)
            memory.store_transition(
                agent=agent,
                state=final_obs,
                action=action,
                log_prob=0.1,    # Dummy value (could be replaced by actual log_prob if needed)
                reward=0.0,      # Dummy value for benchmarking
                is_terminal=False,
                state_value=0.0,
                action_mask=action_mask
            )
        # End of episode loop
        t_episode = time.perf_counter() - t_episode_start
        #logger.info(f"Episode {episode+1} completed in {t_episode:.4f}s")
        
        # At the end of the episode, collect OBP training data.
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)
        
        # --- Periodic PPO Update using stored transitions ---
        if episode % UPDATE_STEPS == 0:
            t_ppo_start = time.perf_counter()
            # Aggregate transitions from all agents in memory.
            all_states = []
            all_actions = []
            all_log_probs = []
            all_rewards = []
            all_is_terminals = []
            all_state_values = []
            for ag in agents:
                all_states.extend(memory.states[ag])
                all_actions.extend(memory.actions[ag])
                all_log_probs.extend(memory.log_probs[ag])
                all_rewards.extend(memory.rewards[ag])
                all_is_terminals.extend(memory.is_terminals[ag])
                all_state_values.extend(memory.state_values[ag])
            if len(all_states) > 0:
                states = torch.tensor(np.array(all_states, dtype=np.float32), device=device)
                actions = torch.tensor(np.array(all_actions, dtype=np.int64), device=device)
                old_log_probs = torch.tensor(np.array(all_log_probs, dtype=np.float32), device=device)
                values = np.array(all_state_values, dtype=np.float32)
                next_values = np.concatenate([values[1:], np.array([0])])
                advantages, returns = compute_gae(
                    rewards=all_rewards,
                    dones=all_is_terminals,
                    values=values.tolist(),
                    next_values=next_values.tolist(),
                    gamma=0.99,
                    lam=0.95
                )
                returns = torch.tensor(np.array(returns, dtype=np.float32), device=device)
                advantages = torch.tensor(np.array(advantages, dtype=np.float32), device=device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                for _ in range(K_EPOCHS):
                    with autocast(device_type='cuda', enabled=use_mixed_precision):
                        probs, _, _ = policy_net(states, None)
                        probs = torch.clamp(probs, 1e-8, 1.0)
                        m = Categorical(probs)
                        new_log_probs = m.log_prob(actions)
                        loss_policy = - (torch.exp(new_log_probs - old_log_probs) * advantages).mean()
                        values_pred = value_net(states)
                        loss_value = nn.MSELoss()(values_pred.view(-1), returns)
                        loss = loss_policy + 0.5 * loss_value
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_policy)
                    scaler.step(optimizer_value)
                    scaler.update()
                    optimizer_policy.zero_grad()
                    optimizer_value.zero_grad()
                memory.reset()
            total_ppo_update_time += time.perf_counter() - t_ppo_start
        
        # --- Periodic OBP Training ---
        if episode % OBP_UPDATE_STEPS == 0 and len(obp_memory) >= OBP_TRAINING_THRESHOLD:
            t_obp_train_start = time.perf_counter()
            train_obp(obp_model, obp_optimizer, obp_memory, device, logger=logger)
            obp_memory = []
            total_obp_training_time += time.perf_counter() - t_obp_train_start

    overall_time = time.perf_counter() - overall_start
    logger.info(f"Total steps: {total_steps}, Overall time: {overall_time:.4f}s")
    logger.info(f"Avg env reset time: {total_env_reset_time/NUM_EPISODES:.4f}s")
    logger.info(f"Avg OBP & Transformer inference time: {total_obp_transformer_time/NUM_EPISODES:.4f}s")
    logger.info(f"Avg action selection time: {total_action_time/total_steps:.6f}s")
    logger.info(f"Avg PPO update time (per update): {total_ppo_update_time/(NUM_EPISODES/UPDATE_STEPS):.4f}s")
    logger.info(f"Avg OBP training time (per update): {total_obp_training_time/(NUM_EPISODES/OBP_UPDATE_STEPS):.4f}s")
    
    return total_steps / overall_time if overall_time > 0 else 0.0

########################################
# Main Benchmark Loop
########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn_options = [False, True]
    mixed_precision_options = [False, True]
    results = []

    print("Running benchmark with your LiarsDeck environment, OBP and Transformer integration with profiling...")

    for hidden_dim in HIDDEN_DIM_LIST:
        for cudnn_bm in cudnn_options:
            for mp in mixed_precision_options:
                sps = measure_training_speed(
                    hidden_dim=hidden_dim,
                    device=device,
                    use_cudnn_benchmark=cudnn_bm,
                    use_mixed_precision=mp
                )
                results.append({
                    'hidden_dim': hidden_dim,
                    'cudnn_benchmark': cudnn_bm,
                    'mixed_precision': mp,
                    'steps_per_second': sps
                })
                print(f"[hidden_dim={hidden_dim}, cudnn_benchmark={cudnn_bm}, mixed_precision={mp}] => {sps:.2f} steps/s")

    print("\n=== Final Results ===")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
