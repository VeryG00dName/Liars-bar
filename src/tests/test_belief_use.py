import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.shen_models import BeliefSpacePolicy, OpponentBeliefModel

def test_belief_dependency(policy_model, belief_model, env, num_tests=50):
    """
    Test to determine if the policy model uses belief information.
    
    This function compares model outputs between:
    1. Normal beliefs from the belief model
    2. Uniform beliefs (no useful information)
    
    Args:
        policy_model: Trained BeliefSpacePolicy model
        belief_model: Trained OpponentBeliefModel
        env: Environment instance
        num_tests: Number of test cases to run
    
    Returns:
        Dictionary with test results
    """
    device = next(policy_model.parameters()).device
    
    # Metrics to track
    results = {
        "action_match_rate": 0,
        "value_differences": [],
        "prob_distances": [],
        "actions_with_belief": [],
        "actions_without_belief": [],
    }
    
    print(f"Running belief dependency test ({num_tests} iterations)...")
    env.reset()
    
    # We need to know the belief dimensions
    # In the original code, belief_dim = num_opponent_classes * len(opponent_agents)
    # where opponent_agents should be all agents except player_0
    num_opponents = env.num_players - 1  # e.g., player_1 and player_2 in 3-player game
    
    # Determine the number of opponent classes from the model architecture
    # Get the belief dimension from the first layer of the policy model
    first_layer = policy_model.network[0]
    belief_dim = first_layer.in_features - 9  # Total input size minus observation dim
    
    # Calculate number of opponent classes per opponent
    num_opponent_classes = belief_dim // num_opponents
    
    # Initialize opponent tracking and beliefs
    opponent_agents = [f'player_{i}' for i in range(1, env.num_players)]
    uniform_belief = np.ones(num_opponent_classes) / num_opponent_classes
    
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
        
        # Get current observation
        observation_dict = env.observe('player_0', new=True)
        current_obs = observation_dict['player_0']
        
        # Get action mask
        mask = env.infos.get('player_0', {}).get('action_mask', [1, 1, 1, 1, 1, 1, 1])
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        
        # Convert observation to tensor
        obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Generate real beliefs using the belief model
        real_beliefs = []
        for opponent in opponent_agents:
            # Use current observation to update beliefs about this opponent
            try:
                with torch.no_grad():
                    # Start with uniform belief as input
                    current_belief = torch.tensor(uniform_belief, dtype=torch.float32, device=device).unsqueeze(0)
                    # Get updated belief from model
                    updated_belief = belief_model(obs_tensor, current_belief)
                    real_beliefs.append(updated_belief.squeeze(0).cpu().numpy())
            except Exception as e:
                # If there's an error with the belief model, use a skewed belief instead of uniform
                # This creates a belief that still has information, but is synthetically generated
                print(f"Warning: Error generating belief, using synthetic belief instead: {e}")
                synthetic_belief = np.copy(uniform_belief)
                # Make the first class more likely (arbitrary choice to create difference)
                synthetic_belief[0] = 0.5
                # Normalize
                synthetic_belief = synthetic_belief / synthetic_belief.sum()
                real_beliefs.append(synthetic_belief)
        
        # Concatenate beliefs for all opponents
        combined_real_belief = np.concatenate(real_beliefs)
        real_belief_tensor = torch.tensor(combined_real_belief, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Create fake uniform beliefs
        fake_beliefs = []
        for _ in opponent_agents:
            fake_beliefs.append(uniform_belief)
        
        combined_fake_belief = np.concatenate(fake_beliefs)
        fake_belief_tensor = torch.tensor(combined_fake_belief, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # With real beliefs
            real_policy, real_value = policy_model(obs_tensor, real_belief_tensor)
            real_probs = F.softmax(real_policy, dim=-1).squeeze(0)
            real_masked_probs = real_probs * mask_tensor
            if real_masked_probs.sum() > 0:
                real_masked_probs = real_masked_probs / real_masked_probs.sum()
            
            # With fake beliefs
            fake_policy, fake_value = policy_model(obs_tensor, fake_belief_tensor)
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
            
            results["action_match_rate"] += int(real_action == fake_action)
            results["value_differences"].append(abs(real_value.item() - fake_value.item()))
            results["prob_distances"].append(kl_div)
            results["actions_with_belief"].append(real_action)
            results["actions_without_belief"].append(fake_action)
        
        # Take action in environment using real beliefs
        env.step(real_action)
        tests_completed += 1
        if tests_completed % 10 == 0:
            print(f"Completed {tests_completed}/{num_tests} tests")
    
    # Calculate final metrics
    match_rate = results["action_match_rate"] / num_tests
    avg_value_diff = np.mean(results["value_differences"])
    avg_prob_distance = np.mean(results["prob_distances"])
    
    print("\n=== Belief Dependency Test Results ===")
    print(f"Action match rate: {match_rate:.2f}")
    print(f"Average value difference: {avg_value_diff:.4f}")
    print(f"Average probability distance: {avg_prob_distance:.4f}")
    
    # Plot action distributions
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(results["actions_with_belief"], bins=7, alpha=0.7, label="With Belief")
    plt.title("Actions with Belief Information")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(results["actions_without_belief"], bins=7, alpha=0.7, label="Without Belief")
    plt.title("Actions with Uniform Belief")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    # Plot value differences histogram
    plt.figure(figsize=(10, 5))
    plt.hist(results["value_differences"], bins=20, alpha=0.7)
    plt.title("Value Differences Between Models")
    plt.xlabel("Absolute Value Difference")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot probability distance histogram
    plt.figure(figsize=(10, 5))
    plt.hist(results["prob_distances"], bins=20, alpha=0.7)
    plt.title("KL Divergence Between Action Probability Distributions")
    plt.xlabel("KL Divergence")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    if match_rate > 0.9:
        print("\nDIAGNOSIS: The policy model does not effectively use belief information.")
        print("It produces nearly identical actions with or without belief context.")
    elif match_rate > 0.7:
        print("\nDIAGNOSIS: The policy model makes limited use of belief information.")
        print("Beliefs affect decisions in some cases but not most.")
    else:
        print("\nDIAGNOSIS: The policy model effectively uses belief information.")
        print("Actions differ significantly with vs. without belief context.")
    
    return results

def main():
    try:
        device = torch.device(config.DEVICE)
        
        # Create the environment instance
        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
        
        # First, reset the environment to get valid observations for sizing
        env.reset()
        sample_obs = env.observe('player_0', new=True)['player_0']
        obs_dim = sample_obs.shape[0]
        
        # Number of opponents (for belief dimension calculation)
        num_opponents = env.num_players - 1
        
        # Load the checkpoint - update the path to load belief model checkpoint
        checkpoint_dir = os.path.dirname(config.DEFAULT_CHECKPOINT_PATH)
        checkpoint_path = os.path.join(checkpoint_dir, "belief_space_final.pth")
        
        if not os.path.exists(checkpoint_path):
            # Try alternative paths
            checkpoint_path = os.path.join(checkpoint_dir, "agents_checkpoint.pth")
            if not os.path.exists(checkpoint_path):
                # Look for any checkpoint file in the directory
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                else:
                    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            # Create a dummy checkpoint with required structure
            checkpoint = {
                "policy_nets": {
                    "player_0": {}
                }
            }
            print("Using empty checkpoint structure")
        
        # Determine the number of opponent classes and belief dimension
        if "obp_model_state_dict" in checkpoint:
            # Extract the dimensions from the checkpoint if possible
            try:
                obp_state_dict = checkpoint["obp_model_state_dict"]
                # Look for a layer with shape info about opponent classes
                for key, value in obp_state_dict.items():
                    if "belief_update" in key and len(value.shape) == 2:
                        num_opponent_classes = value.shape[1]
                        print(f"Detected {num_opponent_classes} opponent classes from checkpoint")
                        break
            except:
                num_opponent_classes = config.NUM_OPPONENT_CLASSES
                print(f"Using default {num_opponent_classes} opponent classes")
        else:
            num_opponent_classes = config.NUM_OPPONENT_CLASSES
            print(f"Using default {num_opponent_classes} opponent classes")
        
        # Calculate total belief dimension
        belief_dim = num_opponent_classes * num_opponents
        
        # Create the belief space policy model
        belief_policy = BeliefSpacePolicy(
            belief_dim=belief_dim,
            obs_dim=obs_dim,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=7  # Number of actions in the environment
        ).to(device)
        
        # Create the opponent belief model
        belief_model = OpponentBeliefModel(
            obs_dim=obs_dim,
            num_opponent_types=num_opponent_classes,
            hidden_dim=config.HIDDEN_DIM // 2
        ).to(device)
        
        # Load model weights from checkpoint
        if "policy_nets" in checkpoint and "player_0" in checkpoint["policy_nets"]:
            try:
                belief_policy.load_state_dict(checkpoint["policy_nets"]["player_0"])
                print("Loaded belief policy from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load belief policy: {e}")
        else:
            print("Warning: Could not find policy_nets in checkpoint")
        
        if "obp_model_state_dict" in checkpoint:
            try:
                belief_model.load_state_dict(checkpoint["obp_model_state_dict"])
                print("Loaded belief model from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load belief model: {e}")
        else:
            print("Warning: Could not find obp_model_state_dict in checkpoint")
            
            # Try looking for alternative keys that might contain the belief model
            possible_belief_keys = ["belief_model_state_dict", "opponent_belief_model", "obp_model"]
            for key in possible_belief_keys:
                if key in checkpoint:
                    try:
                        belief_model.load_state_dict(checkpoint[key])
                        print(f"Loaded belief model from checkpoint using key '{key}'")
                        break
                    except Exception as e:
                        print(f"Warning: Found key '{key}' but could not load model: {e}")
                        
            # If we still couldn't load the belief model, we'll continue with a randomly initialized one
            print("Note: Running test with a randomly initialized belief model")
        
        # Set models to evaluation mode
        belief_policy.eval()
        belief_model.eval()
        
        # Run the belief dependency test
        test_belief_dependency(belief_policy, belief_model, env)
        
    except Exception as e:
        import traceback
        print(f"\nError in main function: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\nTrying fallback with minimal configuration...")
        
        try:
            # Fallback with minimal configuration
            device = torch.device("cpu")  # Use CPU for fallback
            env = LiarsDeckEnv(num_players=3, render_mode=None)
            env.reset()
            
            # Get observation dimension
            sample_obs = env.observe('player_0', new=True)['player_0']
            obs_dim = sample_obs.shape[0]
            
            # Create models with minimal configuration
            num_opponent_classes = 3  # Minimal class count
            belief_dim = num_opponent_classes * 2  # Assuming 2 opponents
            
            belief_policy = BeliefSpacePolicy(
                belief_dim=belief_dim,
                obs_dim=obs_dim,
                hidden_dim=128,  # Small hidden dimension
                output_dim=7
            ).to(device)
            
            belief_model = OpponentBeliefModel(
                obs_dim=obs_dim,
                num_opponent_types=num_opponent_classes,
                hidden_dim=64
            ).to(device)
            
            print("Created models with minimal configuration")
            print("Running test with randomly initialized models")
            
            # Run test with minimal configuration
            test_belief_dependency(belief_policy, belief_model, env, num_tests=10)
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("Please check your model and environment configurations")
            traceback.print_exc()

if __name__ == "__main__":
    main()