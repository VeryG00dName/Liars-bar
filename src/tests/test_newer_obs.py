#!/usr/bin/env python
# test_newer_obs.py - Tests the newer observation format in Liar's Deck Environment

import numpy as np
import random
import sys
import os
import time

# Add the parent directory to the path so we can import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils import get_newer_observations

def print_observation_details(obs, agent, env):
    """Pretty print the observation for debugging"""
    print(f"\n{'=' * 40}")
    print(f"OBSERVATION FOR {agent}:")
    print(f"{'=' * 40}")
    
    # Get the hand vector size (should be 2)
    hand_vector_size = 2
    hand_vector = obs[:hand_vector_size]
    
    # Get the number of players for determining observation parts
    num_players = len(env.possible_agents)
    
    # Extract parts from the observation
    last_actions_start = hand_vector_size
    last_actions_end = last_actions_start + (num_players - 1)
    challenge_outcome_idx = last_actions_end
    elimination_status_start = challenge_outcome_idx + 1
    
    # Print hand vector (only populated on first turn)
    print(f"Hand vector: {hand_vector}")
    if np.any(hand_vector):
        print("  ↳ First turn - hand information provided")
        table_cards_percent = hand_vector[0]
        non_table_cards_percent = hand_vector[1]
        print(f"  ↳ Table cards: {table_cards_percent * 5:.0f}/5 ({table_cards_percent:.2f})")
        print(f"  ↳ Non-table cards: {non_table_cards_percent * 5:.0f}/5 ({non_table_cards_percent:.2f})")
    else:
        print("  ↳ Not first turn - no hand information provided")
    
    # Print last actions
    last_actions = obs[last_actions_start:last_actions_end]
    print(f"Last actions of opponents:")
    other_agents = [ag for ag in env.possible_agents if ag != agent]
    for i, (ag, act) in enumerate(zip(other_agents, last_actions)):
        if act == 0:
            action_desc = "No action / Eliminated"
        elif act == 4:
            action_desc = "Challenge"
        else:
            action_desc = f"Play {int(act)} card(s)"
        print(f"  ↳ {ag}: {action_desc} (code: {act})")
    
    # Print challenge outcome
    challenge_outcome = obs[challenge_outcome_idx]
    if challenge_outcome == 0:
        outcome_desc = "No recent challenge"
    elif challenge_outcome > 0:
        outcome_desc = "Last challenge SUCCEEDED (play was invalid)"
    else:
        outcome_desc = "Last challenge FAILED (play was valid)"
    print(f"Challenge outcome: {outcome_desc} (value: {challenge_outcome})")
    
    # Print elimination status
    elimination_status = obs[elimination_status_start:]
    print(f"Player elimination status:")
    for i, (ag, status) in enumerate(zip(env.possible_agents, elimination_status)):
        status_desc = "ELIMINATED" if status > 0 else "Active"
        print(f"  ↳ {ag}: {status_desc}")
    
    print(f"{'=' * 40}")
    print(f"Raw observation shape: {obs.shape}")
    print(f"Raw observation values: {obs}")
    print(f"{'=' * 40}\n")

def run_test(num_episodes=3):
    """Run random games and print observations"""
    env = LiarsDeckEnv(render_mode="human")
    
    for episode in range(num_episodes):
        print(f"\n\n{'#' * 60}")
        print(f"STARTING EPISODE {episode + 1}/{num_episodes}")
        print(f"{'#' * 60}\n")
        
        obs, infos = env.reset(seed=episode)
        done = False
        step_count = 0
        
        while not done:
            agent = env.agent_selection
            if agent is None:
                print("Episode ended - no active agent")
                break
                
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            
            # Render the environment in human-readable format
            print(f"\nSTEP {step_count} - {agent}'s turn")
            env.render()
            
            # Get and print the newer observation
            # First explicitly call observe to ensure action_mask is populated
            env.observe(agent)  # This should populate the action_mask in infos
            newer_obs = get_newer_observations(env, agent_specific=agent)
            print_observation_details(newer_obs[agent], agent, env)
            
            # Choose a random valid action (with fallback if mask not available)
            if 'action_mask' not in env.infos[agent]:
                print(f"Warning: action_mask not found in infos for {agent}")
                print(f"Available info keys: {list(env.infos[agent].keys())}")
                # Fallback - use _compute_action_mask directly
                action_mask = env._compute_action_mask(agent)
            else:
                action_mask = env.infos[agent]['action_mask']
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            if not valid_actions:
                print("No valid actions available!")
                done = True
                break
                
            action = random.choice(valid_actions)
            
            # Display the action being taken
            action_types = ["Play 1 table card", "Play 2 table cards", "Play 3 table cards",
                           "Play 1 non-table card", "Play 2 non-table cards", "Play 3 non-table cards",
                           "Challenge"]
            print(f"{agent} takes action: {action_types[action]} (code: {action})")
            
            # Take the action
            env.step(action)
            step_count += 1
            
            # Check if episode is done (all agents terminated)
            if all(env.terminations.values()):
                print("All agents terminated - episode complete")
                done = True
                
            # Limit steps to prevent infinite loops
            if step_count > 100:
                print("Maximum steps reached - ending episode")
                done = True
        
        print(f"\nEpisode {episode + 1} complete after {step_count} steps")
        if env.winner:
            print(f"Winner: {env.winner}")
        else:
            print("No winner determined")
    
    env.close()

if __name__ == "__main__":
    print("Testing Liar's Deck Environment with Newer Observations")
    run_test(num_episodes=2)