# src/misc/gpu_test.py

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
from collections import defaultdict, Counter
import random

# Imports from your project
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.reward_restriction_wrapper import RewardRestrictionWrapper
from src.training.train_utils import compute_gae, train_obp
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.model.memory import RolloutMemory
from src.training.train_extras import set_seed, extract_obp_training_data, run_obp_inference
from src import config

from torch.amp import autocast, GradScaler  # Updated import for GradScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

torch.backends.cudnn.benchmark = True

########################################
# Benchmark configuration overrides
########################################
HIDDEN_DIM_LIST = [128, 512, 1024, 1504, 2000]
NUM_EPISODES = 100         # Total episodes to run during the benchmark
UPDATE_STEPS = 5           # Frequency (in episodes) of the PPO update phase
K_EPOCHS = 4               # Number of PPO epochs per update
OBP_UPDATE_STEPS = 5      # Frequency (in episodes) to train OBP
OBP_BATCH_SIZE = 200        # Batch size for OBP training
OBP_TRAINING_THRESHOLD = 100  # Minimum OBP data points before training

########################################
# Benchmark function using your real environment with OBP
########################################

def measure_training_speed(hidden_dim, device, use_cudnn_benchmark=True, use_mixed_precision=False):
    torch.backends.cudnn.benchmark = use_cudnn_benchmark
    scaler = GradScaler(enabled=use_mixed_precision)
    
    # Initialize environment
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, _ = env.reset()
    agents = env.agents
    
    # Initialize networks
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=False,
        use_layer_norm=False
    ).to(device)
    
    value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim,
        use_dropout=False,
        use_layer_norm=False
    ).to(device)
    
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=hidden_dim,
        output_dim=2
    ).to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.001)
    optimizer_value = optim.Adam(value_net.parameters(), lr=0.001)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=0.001)
    
    memory = RolloutMemory(agents)
    obp_memory = []
    total_steps = 0
    start_time = time.time()
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        env.agents = agents  # Maintain original agent order
        
        while env.agent_selection is not None:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            
            # Simplified observation processing
            observation = env.observe(agent)[agent]
            action_mask = env.infos[agent]['action_mask']
            
            # Run OBP inference
            with torch.no_grad():
                obp_probs = run_obp_inference(obp_model, observation, device, env.num_players)
                final_obs = np.concatenate([observation, obp_probs], axis=0)
            
            # Action selection
            obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with autocast(device_type='cuda', enabled=use_mixed_precision):
                probs, _ = policy_net(obs_tensor, None)
            
            probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
            action_mask_tensor = torch.tensor(action_mask, device=device)
            masked_probs = probs * action_mask_tensor
            if masked_probs.sum() == 0:
                masked_probs = action_mask_tensor.float()
            masked_probs /= masked_probs.sum()
            
            action = Categorical(masked_probs).sample().item()
            env.step(action)
            total_steps += 1
            
            # Store simplified transition (no rewards/values for benchmarking)
            memory.store_transition(
                agent=agent,
                state=final_obs,
                action=action,
                log_prob=0.1,  # Dummy value
                reward=0.0,    # Dummy value
                is_terminal=False,
                state_value=0.0
            )
        
        # Periodic model updates (mimic real training flow)
        if episode % UPDATE_STEPS == 0:
            # Dummy PPO update with mixed precision
            for _ in range(K_EPOCHS):
                with autocast(device_type='cuda', enabled=use_mixed_precision):
                    states = torch.randn(64, config.INPUT_DIM, device=device)  # Dummy data
                    actions = torch.randint(0, config.OUTPUT_DIM, (64,), device=device)
                    
                    # Policy update
                    probs, _ = policy_net(states, None)
                    loss_policy = -Categorical(probs).log_prob(actions).mean()
                    
                    # Value update
                    values = value_net(states)
                    loss_value = nn.MSELoss()(values, torch.randn_like(values))
                    
                    # Combined loss
                    loss = loss_policy + 0.5 * loss_value
                
                scaler.scale(loss).backward()
                scaler.step(optimizer_policy)
                scaler.step(optimizer_value)
                scaler.update()
                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
        
        # Periodic OBP training
        if episode % OBP_UPDATE_STEPS == 0 and len(obp_memory) >= OBP_TRAINING_THRESHOLD:
            obp_batch = random.sample(obp_memory, OBP_BATCH_SIZE)
            with autocast(device_type='cuda', enabled=use_mixed_precision):
                features = torch.randn(OBP_BATCH_SIZE, config.OPPONENT_INPUT_DIM, device=device)
                labels = torch.randint(0, 2, (OBP_BATCH_SIZE,), device=device)
                logits = obp_model(features)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(obp_optimizer)
            scaler.update()
            obp_optimizer.zero_grad()
    
    elapsed_time = time.time() - start_time
    return total_steps / elapsed_time if elapsed_time > 0 else 0.0

########################################
# Main Benchmark Loop
########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn_options = [False, True]
    mixed_precision_options = [False, True]
    results = []

    print("Running benchmark with your LiarsDeck environment and OBP...")

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