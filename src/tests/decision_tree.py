#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from tqdm import tqdm
# Import your configuration, environment, and model classes.
from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.shen_models import BeliefSpacePolicy, OpponentBeliefModel

# =============================================================================
# Helper Functions (Real Implementations)
# =============================================================================
def generate_diverse_states(env, n=1000):
    """
    Generate diverse game states by simulating random gameplay.
    This function resets the environment repeatedly and collects states 
    observed by 'player_0'. Adjust the simulation policy as needed.
    """
    states = []
    while len(states) < n:
        env.reset()
        done = False
        while not done and len(states) < n:
            current_agent = env.agent_selection
            if current_agent == 'player_0':
                obs = env.observe('player_0', new=True)['player_0']
                states.append(obs)
                # Take a random valid action for player_0.
                mask = env.infos.get('player_0', {}).get('action_mask', [1]*7)
                valid_actions = [i for i, m in enumerate(mask) if m == 1]
                action = np.random.choice(valid_actions) if valid_actions else 0
                env.step(action)
            else:
                # For other agents, simply take a random valid action.
                mask = env.infos.get(current_agent, {}).get('action_mask', [1]*7)
                valid_actions = [i for i, m in enumerate(mask) if m == 1]
                action = np.random.choice(valid_actions) if valid_actions else 0
                env.step(action)
            if env.agent_selection is None:
                done = True
    return states

def create_belief_vector(perm):
    """
    Given an opponent permutation (e.g. ('0', '1')), create a belief vector.
    Assumes there are 10 opponent types and 2 slots (vector length 20).
    """
    num_opponent_types = 10
    opponent_slots = 2
    belief_array = np.zeros(num_opponent_types * opponent_slots, dtype=np.float32)
    for i, opp in enumerate(perm):
        belief_array[i * num_opponent_types + int(opp)] = 1.0
    return belief_array

def extract_state_features(state):
    """
    Extract state features from the observation.
    Here, we assume the raw state is already a meaningful feature vector.
    """
    return state

def model_predict(model, state, belief):
    """
    Given a model, a state (numpy array), and a belief (numpy array),
    convert them to tensors, run the model, and return the predicted action.
    """
    device = next(model.parameters()).device
    # Convert state to tensor and add batch dimension.
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # Convert belief to tensor and add batch dimension.
    belief_tensor = torch.tensor(belief, dtype=torch.float32, device=device).unsqueeze(0)
    # Run the model in evaluation mode.
    model.eval()
    with torch.no_grad():
        logits, _ = model(state_tensor, belief_tensor)
        # Optionally, apply masking if your model uses an action mask.
        action = torch.argmax(logits, dim=1).item()
    return action

def extract_decision_boundaries(model, env, permutations, n_states=1000):
    """
    For each opponent permutation, use a set of diverse states to record the model's decision.
    We form a dataset with state features as X and labels that combine the opponent permutation
    with the predicted action. A decision tree is then trained to approximate the decision boundaries.
    """
    print("Generating diverse states ...")
    states = generate_diverse_states(env, n=n_states)
    
    X = []  # State features
    y = []  # Labels as string "perm:action"
    print("Collecting model decisions ...")
    for perm in tqdm(permutations, desc="Processing permutations"):
        belief = create_belief_vector(perm)
        for state in states:
            features = extract_state_features(state)
            action = model_predict(model, features, belief)
            X.append(features)
            y.append(f"{perm}:{action}")
    
    # Train a decision tree classifier.
    print("Training decision tree classifier ...")
    tree = DecisionTreeClassifier(max_depth=50, random_state=42)
    tree.fit(X, y)
    
    # Extract and return decision rules.
    decision_rules = export_text(tree)
    return decision_rules

def analyze_decision_rules(decision_rules):
    """
    Analyze decision tree rules to identify distinct strategy patterns.
    Here, we simply print and return the text. You may extend this to perform more advanced analysis.
    """
    print("Extracted Decision Rules:")
    print(decision_rules)
    return decision_rules

# =============================================================================
# Main Script
# =============================================================================
def main():
    try:
        # Set up device.
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Create the environment instance.
        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
        env.reset()
        sample_obs = env.observe('player_0', new=True)['player_0']
        obs_dim = sample_obs.shape[0]
        
        # Number of opponents (for belief dimension calculation).
        num_opponents = env.num_players - 1
        
        # Load checkpoint for the belief policy model.
        checkpoint_dir = os.path.dirname(config.DEFAULT_CHECKPOINT_PATH)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_episode_100.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "agents_checkpoint.pth")
            if not os.path.exists(checkpoint_path):
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                else:
                    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            checkpoint = {"policy_nets": {"player_0": {}}}
            print("Using empty checkpoint structure")
        
        # Determine opponent classes.
        if "obp_model_state_dict" in checkpoint:
            try:
                obp_state_dict = checkpoint["obp_model_state_dict"]
                for key, value in obp_state_dict.items():
                    if "belief_update" in key and len(value.shape) == 2:
                        num_opponent_classes = value.shape[1]
                        print(f"Detected {num_opponent_classes} opponent classes from checkpoint")
                        break
            except Exception as e:
                num_opponent_classes = config.NUM_OPPONENT_CLASSES
                print(f"Using default {num_opponent_classes} opponent classes")
        else:
            num_opponent_classes = config.NUM_OPPONENT_CLASSES
            print(f"Using default {num_opponent_classes} opponent classes")
        
        # Calculate belief dimension.
        belief_dim = num_opponent_classes * num_opponents
        
        # Create the belief space policy model.
        belief_policy = BeliefSpacePolicy(
            belief_dim=belief_dim,
            obs_dim=obs_dim,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=7  # Number of actions.
        ).to(device)
        
        # Load model weights.
        if "policy_nets" in checkpoint and "player_0" in checkpoint["policy_nets"]:
            try:
                belief_policy.load_state_dict(checkpoint["policy_nets"]["player_0"])
                print("Loaded belief policy from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load belief policy: {e}")
        else:
            print("Warning: Could not find policy_nets in checkpoint")
        belief_policy.eval()
        
        # Generate opponent permutations.
        # We use the same logic as in previous analyses; here we only need the permutation part.
        def get_permutations():
            num_opponent_types = 10
            opponent_slots = 2
            opponent_types = [str(i) for i in range(num_opponent_types)]
            perms = list(itertools.product(opponent_types, repeat=opponent_slots))
            return perms
        
        permutations = get_permutations()
        print(f"Generated {len(permutations)} opponent permutations.")
        
        # Extract decision boundaries from the model.
        decision_rules = extract_decision_boundaries(belief_policy, env, permutations, n_states=1000)
        
        # Analyze (and print) the decision rules.
        analyze_decision_rules(decision_rules)
        
    except Exception as e:
        import traceback
        print(f"\nError in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
