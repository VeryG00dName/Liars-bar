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
        "actions_without_history": []
    }
    
    # Run the test
    print(f"Running observation history dependency test ({num_tests} iterations)...")
    env.reset()
    
    observation_stack = deque(maxlen=50)  # Assume max stack size of 50
    
    # Pre-fill with zeros
    sample_obs = env.observe('player_0', new=True)['player_0']
    for _ in range(50):
        observation_stack.append(np.zeros_like(sample_obs))
    
    tests_completed = 0
    
    while tests_completed < num_tests:
        # Only test when it's player_0's turn
        if env.agent_selection != 'player_0':
            # Take random valid action for other agents
            agent = env.agent_selection
            mask = env.infos[agent].get('action_mask', [1, 1, 1, 1, 1, 1, 1])
            valid_actions = [i for i, m in enumerate(mask) if m == 1]
            action = np.random.choice(valid_actions) if valid_actions else 0
            env.step(action)
            continue
        
        # Get current observation and add to history
        observation_dict = env.observe('player_0', new=True)
        current_obs = observation_dict['player_0']
        observation_stack.append(current_obs.copy())
        
        # Get action mask
        mask = env.infos['player_0'].get('action_mask', [1, 1, 1, 1, 1, 1, 1])
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        
        # Create two different observation stacks:
        # 1. Real history
        real_history = list(observation_stack)
        real_tensor = torch.tensor(
            np.array(real_history), 
            dtype=torch.float32, 
            device=device
        ).unsqueeze(0)
        
        # 2. Current observation repeated
        fake_history = [current_obs.copy() for _ in range(len(observation_stack))]
        fake_tensor = torch.tensor(
            np.array(fake_history), 
            dtype=torch.float32, 
            device=device
        ).unsqueeze(0)
        
        # Run both through the model
        with torch.no_grad():
            # With real history
            real_policy, real_value, _ = model(real_tensor)
            real_probs = F.softmax(real_policy, dim=-1).squeeze(0)
            real_masked_probs = real_probs * mask_tensor
            if real_masked_probs.sum() > 0:
                real_masked_probs = real_masked_probs / real_masked_probs.sum()
            
            # With fake history
            fake_policy, fake_value, _ = model(fake_tensor)
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
            
            # Track metrics
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
    
    print("\n=== Observation History Test Results ===")
    print(f"Action match rate: {match_rate:.2f}")
    print(f"Average value difference: {avg_value_diff:.4f}")
    print(f"Average probability distance: {avg_prob_distance:.4f}")
    
    # Plot action distributions
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(results["actions_with_history"], bins=7, alpha=0.7, label="With History")
    plt.title("Actions with History")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(results["actions_without_history"], bins=7, alpha=0.7, label="Without History")
    plt.title("Actions without History")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    if match_rate > 0.9:
        print("\nDIAGNOSIS: The model does not use observation history effectively.")
        print("It produces nearly identical actions with or without historical context.")
    elif match_rate > 0.7:
        print("\nDIAGNOSIS: The model makes limited use of observation history.")
        print("History affects decisions in some cases but not most.")
    else:
        print("\nDIAGNOSIS: The model effectively uses observation history.")
        print("Actions differ significantly with vs. without historical context.")
    
    return results

def main():
    # Set device
    device = torch.device(config.DEVICE)
    
    # Build the checkpoint path.
    # Use the directory from config.DEFAULT_CHECKPOINT_PATH but with the filename "checkpoint_episode_3000.pth"
    checkpoint_dir = os.path.dirname(config.DEFAULT_CHECKPOINT_PATH)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_episode_3000.pth")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create an instance of OpponentConditionalModel
    from src.model.models import OpponentConditionalModel
    model = OpponentConditionalModel(
        obs_dim=9,
        num_actions=config.OUTPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_obs_stack=50,         # Assuming 50 observations in history
        num_opponent_classes=10    # Default number of opponent classes
    )
    
    # Load the state dict from the checkpoint.
    # Check if the checkpoint contains "model_state_dict", otherwise check for "policy_nets".
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "policy_nets" in checkpoint:
            # Assume we want the state for "player_0"
            state_dict = checkpoint["policy_nets"]["player_0"]
            model.load_state_dict(state_dict)
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
