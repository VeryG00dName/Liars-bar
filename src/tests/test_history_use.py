import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv

def test_observation_dependency(model, env, num_tests=50):
    """
    Simple test to determine if the model uses observation history.
    
    This function compares model outputs between:
    1. Normal observation history
    2. History filled with the most recent observation repeated
    
    Args:
        model: Your trained model
        env: Environment instance
        num_tests: Number of test cases to run
    
    Returns:
        Dictionary with test results
    """
    device = next(model.parameters()).device
    
    # Metrics to track
    results = {
        "action_match_rate": 0,
        "value_differences": [],
        "prob_distances": [],
        "actions_with_history": [],
        "actions_without_history": [],
        "real_gate_weights": [],
        "fake_gate_weights": []
    }
    
    print(f"Running observation history dependency test ({num_tests} iterations)...")
    env.reset()
    
    observation_stack = deque(maxlen=50)  # Assume max stack size of 50
    
    # Pre-fill with zeros based on the first observation
    sample_obs = env.observe('player_0', newer=True)['player_0']
    for _ in range(50):
        observation_stack.append(np.zeros_like(sample_obs))
    
    tests_completed = 0
    
    while tests_completed < num_tests:
        # Check if the game has ended (agent_selection is None) and reset if needed.
        if env.agent_selection is None:
            env.reset()
            continue

        agent = env.agent_selection
        
        # Only test when it's player_0's turn
        if agent != 'player_0':
            # Take random valid action for other agents
            mask = env.infos.get(agent, {}).get('action_mask', [1, 1, 1, 1, 1, 1, 1])
            valid_actions = [i for i, m in enumerate(mask) if m == 1]
            action = np.random.choice(valid_actions) if valid_actions else 0
            env.step(action)
            continue
        
        # Get current observation and add to history
        observation_dict = env.observe('player_0', newer=True)
        current_obs = observation_dict['player_0']
        
        # Ensure observation shape consistency in the stack
        if len(observation_stack) > 0:
            # Check if the current observation has the same shape as those in the stack
            if current_obs.shape != np.array(observation_stack[0]).shape:
                print(f"Warning: Observation shape mismatch. Current: {current_obs.shape}, Stack: {np.array(observation_stack[0]).shape}")
                # Resize older observations if needed
                new_stack = deque(maxlen=50)
                for old_obs in observation_stack:
                    if np.array(old_obs).shape != current_obs.shape:
                        # Pad or truncate to match the shape
                        new_obs = np.zeros_like(current_obs)
                        min_shape = min(len(old_obs), len(current_obs))
                        new_obs[:min_shape] = old_obs[:min_shape]
                        new_stack.append(new_obs)
                    else:
                        new_stack.append(old_obs)
                observation_stack = new_stack
                
        # Add current observation to stack
        observation_stack.append(current_obs.copy())
        
        # Get action mask
        mask = env.infos.get('player_0', {}).get('action_mask', [1, 1, 1, 1, 1, 1, 1])
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        
        # Ensure all observations in the stack have the same shape
        if not all(np.array(obs).shape == np.array(current_obs).shape for obs in observation_stack):
            print("Warning: Not all observations in stack have same shape. Fixing...")
            uniform_stack = []
            for obs in observation_stack:
                if np.array(obs).shape != np.array(current_obs).shape:
                    # Create a zero array of the right shape and copy what we can
                    new_obs = np.zeros_like(current_obs)
                    min_shape = min(len(obs), len(current_obs))
                    new_obs[:min_shape] = obs[:min_shape]
                    uniform_stack.append(new_obs)
                else:
                    uniform_stack.append(obs)
            
            # Verify all shapes are now the same
            shapes = [np.array(obs).shape for obs in uniform_stack]
            if len(set(shapes)) > 1:
                print(f"Error: Still have inconsistent shapes in stack: {shapes}")
                continue
                
            real_history = uniform_stack
        else:
            real_history = list(observation_stack)
        
        # Create two different observation stacks:
        # 1. Real history
        real_tensor = torch.tensor(
            np.array(real_history),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        
        # 2. Current observation repeated
        fake_history = [
            current_obs.copy() if i == len(observation_stack) - 1 else np.zeros_like(current_obs)
            for i in range(len(observation_stack))
        ]
        
        fake_tensor = torch.tensor(
            np.array(fake_history),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        
        with torch.no_grad():
            # With real history - adapt to newer model with game state prediction head
            try:
                # First try with the updated model architecture (policy, value, game_state_pred, gate)
                real_policy, real_value, _, real_gate = model(real_tensor)
            except ValueError:
                # If that fails, try with the original architecture (policy, value, next_obs_pred, gate)
                real_policy, real_value, _, real_gate = model(real_tensor)
                
            real_probs = F.softmax(real_policy, dim=-1).squeeze(0)
            real_masked_probs = real_probs * mask_tensor
            if real_masked_probs.sum() > 0:
                real_masked_probs = real_masked_probs / real_masked_probs.sum()
            
            # With fake history
            try:
                # First try with the updated model architecture
                fake_policy, fake_value, _, fake_gate = model(fake_tensor)
            except ValueError:
                # If that fails, try with the original architecture
                fake_policy, fake_value, _, fake_gate = model(fake_tensor)
                
            fake_probs = F.softmax(fake_policy, dim=-1).squeeze(0)
            fake_masked_probs = fake_probs * mask_tensor
            if fake_masked_probs.sum() > 0:
                fake_masked_probs = fake_masked_probs / fake_masked_probs.sum()
            
            # Get highest probability actions
            real_action = torch.argmax(real_masked_probs).item()
            fake_action = torch.argmax(fake_masked_probs).item()
            
            # Calculate KL divergence between probability distributions
            kl_div = F.kl_div(
                real_masked_probs.log(),
                fake_masked_probs,
                reduction='sum'
            ).item()
            
            # Store the gate weights
            results["real_gate_weights"].append(real_gate.squeeze(0).cpu().numpy())
            results["fake_gate_weights"].append(fake_gate.squeeze(0).cpu().numpy())
            
            results["action_match_rate"] += int(real_action == fake_action)
            results["value_differences"].append(abs(real_value.item() - fake_value.item()))
            results["prob_distances"].append(kl_div)
            results["actions_with_history"].append(real_action)
            results["actions_without_history"].append(fake_action)
        
        # Take action in environment using real history
        env.step(real_action)
        tests_completed += 1
        if tests_completed % 10 == 0:
            print(f"Completed {tests_completed}/{num_tests} tests")
    
    # Calculate final metrics
    match_rate = results["action_match_rate"] / num_tests
    avg_value_diff = np.mean(results["value_differences"])
    avg_prob_distance = np.mean(results["prob_distances"])
    
    # Calculate average gate weights
    real_gate_weights = np.array(results["real_gate_weights"])
    fake_gate_weights = np.array(results["fake_gate_weights"])
    
    avg_real_gate = np.mean(real_gate_weights, axis=0)
    avg_fake_gate = np.mean(fake_gate_weights, axis=0)
    
    print("\n=== Observation History Test Results ===")
    print(f"Action match rate: {match_rate:.2f}")
    print(f"Average value difference: {avg_value_diff:.4f}")
    print(f"Average probability distance: {avg_prob_distance:.4f}")
    print(f"Average gate weights with history: Head 1 = {avg_real_gate[0]:.4f}, Head 2 = {avg_real_gate[1]:.4f}")
    print(f"Average gate weights without history: Head 1 = {avg_fake_gate[0]:.4f}, Head 2 = {avg_fake_gate[1]:.4f}")
    
    # Plot action distributions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(results["actions_with_history"], bins=7, alpha=0.7, label="With History")
    plt.title("Actions with History")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 2, 2)
    plt.hist(results["actions_without_history"], bins=7, alpha=0.7, label="Without History")
    plt.title("Actions without History")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    
    # Plot gate weights distribution
    plt.subplot(2, 2, 3)
    plt.hist(real_gate_weights[:, 0], bins=20, alpha=0.7, label="Head 1")
    plt.hist(real_gate_weights[:, 1], bins=20, alpha=0.7, label="Head 2")
    plt.title("Gate Weights Distribution with History")
    plt.xlabel("Gate Weight")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.hist(fake_gate_weights[:, 0], bins=20, alpha=0.7, label="Head 1")
    plt.hist(fake_gate_weights[:, 1], bins=20, alpha=0.7, label="Head 2")
    plt.title("Gate Weights Distribution without History")
    plt.xlabel("Gate Weight")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot gate weights over time
    plt.figure(figsize=(12, 6))
    x = range(len(real_gate_weights))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, real_gate_weights[:, 0], label="Head 1")
    plt.plot(x, real_gate_weights[:, 1], label="Head 2")
    plt.title("Gate Weights Over Time with History")
    plt.xlabel("Test Number")
    plt.ylabel("Gate Weight")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, fake_gate_weights[:, 0], label="Head 1")
    plt.plot(x, fake_gate_weights[:, 1], label="Head 2")
    plt.title("Gate Weights Over Time without History")
    plt.xlabel("Test Number")
    plt.ylabel("Gate Weight")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create pie charts for average gate usage
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.pie([avg_real_gate[0], avg_real_gate[1]], 
            labels=["Head 1", "Head 2"],
            autopct='%1.1f%%',
            startangle=90)
    plt.title("Average Gate Usage with History")
    
    plt.subplot(1, 2, 2)
    plt.pie([avg_fake_gate[0], avg_fake_gate[1]], 
            labels=["Head 1", "Head 2"],
            autopct='%1.1f%%',
            startangle=90)
    plt.title("Average Gate Usage without History")
    
    plt.tight_layout()
    plt.show()
    
    if match_rate > 0.9:
        print("\nDIAGNOSIS: The model does not use observation history effectively.")
        print("It produces nearly identical actions with or without historical context.")
    elif match_rate > 0.7:
        print("\nDIAGNOSIS: The model makes limited use of observation history.")
        print("History affects decisions in some cases but not most.")
    else:
        print("\nDIAGNOSIS: The model effectively uses observation history.")
        print("Actions differ significantly with vs. without historical context.")
    
    print("\nGATE ACTIVATION ANALYSIS:")
    if np.abs(avg_real_gate[0] - avg_fake_gate[0]) < 0.1:
        print("The model uses similar gate activation patterns with or without history.")
    else:
        print("Gate activation patterns differ significantly based on history presence.")
        head_with_history = 1 if avg_real_gate[0] < 0.5 else 2
        head_without_history = 1 if avg_fake_gate[0] < 0.5 else 2
        print(f"The model prefers Head {head_with_history} with history and Head {head_without_history} without history.")
    
    return results

def main():
    device = torch.device(config.DEVICE)
    
    checkpoint_dir = os.path.dirname(config.DEFAULT_CHECKPOINT_PATH)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_episode_3000.pth")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine if this is a newer observation model (with game state prediction)
    is_newer_model = False
    if isinstance(checkpoint, dict):
        # Check if this is the newer model format
        if "policy_nets" in checkpoint:
            test_state_dict = next(iter(checkpoint["policy_nets"].values()))
            is_newer_model = 'policy_head1.weight' in test_state_dict and 'game_state_head.weight' in test_state_dict
        elif "model" in checkpoint:
            test_state_dict = checkpoint["model"]
            is_newer_model = 'policy_head1.weight' in test_state_dict and 'game_state_head.weight' in test_state_dict
    
    # Create the appropriate model instance based on the checkpoint
    from src.model.models import StackedObservationConvModel
    
    if is_newer_model:
        print("Detected newer model with game state prediction head")
        
        # Sample observation to determine dimension
        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
        sample_obs = env.observe('player_0', newer=True)['player_0']
        obs_dim = sample_obs.shape[0]
        
        # Create model with the appropriate parameters for newer observation format
        model = StackedObservationConvModel(
            obs_dim=obs_dim,
            num_actions=config.OUTPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_obs_stack=50,  # Assuming 50 observations in history
            num_players=config.NUM_PLAYERS  # Pass number of players for game state dimension
        )
    else:
        print("Using standard model with next observation prediction head")
        model = StackedObservationConvModel(
            obs_dim=9,  # Default for standard observation
            num_actions=config.OUTPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_obs_stack=50,  # Assuming 50 observations in history
        )
    
    # Load the model state from either "model_state_dict" or "policy_nets"
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "policy_nets" in checkpoint:
            # Assume we want the state for "player_0"
            state_dict = checkpoint["policy_nets"]["player_0"]
            model.load_state_dict(state_dict)
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("Checkpoint does not contain a recognized key for model state")
    else:
        raise ValueError("Checkpoint format is not recognized as a dictionary")
    
    model.to(device)
    model.eval()
    
    # Create the environment instance.
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    
    # Run the observation dependency test.
    test_observation_dependency(model, env)

if __name__ == "__main__":
    main()