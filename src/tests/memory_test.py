#!/usr/bin/env python
import random
from collections import Counter

import numpy as np

# Import your LiarsDeck environment and opponent memory helper.
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.memory import get_opponent_memory

# Provided decode_action function.
def decode_action(action):
    """
    Decodes the discrete action into its components.
    
    Args:
        action (int or None): Discrete action index or None.
    
    Returns:
        tuple: (action_type, card_category, count)
    """
    if action is None:
        return 'Invalid', None, None

    if 0 <= action <= 5:
        return 'Play', 'table' if action < 3 else 'non-table', (action % 3) + 1
    elif action == 6:
        return 'Challenge', None, None
    return 'Invalid', None, None

def format_decoded(decoded_tuple):
    """
    Formats the tuple returned by decode_action into a string for comparison.
    
    For Play actions, it now returns "play_{count}" (ignoring the card category),
    and for Challenge actions, it returns "challenge".
    
    Args:
        decoded_tuple (tuple): (action_type, card_category, count)
    
    Returns:
        str: "play_{count}" for Play actions, or "challenge" for Challenge actions.
    """
    action_type, card_category, count = decoded_tuple
    if action_type == 'Play':
        return f"play_{count}"
    elif action_type == 'Challenge':
        return "challenge"
    else:
        return "invalid"

def run_random_game():
    """
    Runs a single game of LiarsDeck with random valid moves.
    Each move is chosen based on the action mask.
    
    Returns:
        moves (dict): Keys are player IDs (e.g., "player_0") and values are lists of the action integers taken.
    """
    env = LiarsDeckEnv(num_players=3, render_mode=None)
    moves = {agent: [] for agent in env.agents}
    env.reset()
    
    while env.agent_selection is not None:
        current_agent = env.agent_selection
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            env.step(None)
            continue

        action_mask = info.get('action_mask')
        if action_mask is None:
            raise ValueError("No action_mask provided in info.")

        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if not valid_actions:
            raise ValueError("No valid actions available!")
        chosen_action = random.choice(valid_actions)
        moves[current_agent].append(chosen_action)
        env.step(chosen_action)
    return moves

def aggregate_recorded_moves(all_game_moves):
    """
    Aggregates moves across multiple games per player.
    
    Args:
        all_game_moves (list): List of moves dictionaries from each game.
    
    Returns:
        dict: Combined moves per player.
    """
    combined_moves = {}
    for game_moves in all_game_moves:
        for agent, actions in game_moves.items():
            if agent not in combined_moves:
                combined_moves[agent] = []
            combined_moves[agent].extend(actions)
    return combined_moves

def collect_memory_events_for_player(player, all_agents):
    """
    For a given player, collects all memory events recorded by opponents about that player.
    
    Since each opponent tracks moves made by specific players,
    we iterate over all other agents (observers) and collect events where the key equals the player.
    
    Args:
        player (str): The player ID whose moves we want to compare.
        all_agents (list): List of all player IDs.
    
    Returns:
        list: All memory event dictionaries for the given player.
    """
    memory_events = []
    for observer in all_agents:
        if observer == player:
            continue  # Skip self
        mem_obj = get_opponent_memory(observer)
        if player in mem_obj.memory:
            memory_events.extend(list(mem_obj.memory[player]))
    return memory_events

def compare_moves(recorded_moves, memory_events):
    """
    Decodes the recorded moves using decode_action and formats them.
    Also extracts the "response" from each memory event (and lowercases it).
    Then compares the frequency counts between the two lists.
    
    Args:
        recorded_moves (list of int): The list of action integers recorded for a player.
        memory_events (list of dict): List of memory events (each expected to have a "response" key).
    
    Returns:
        tuple: (decoded_recorded, decoded_memory, match_count)
               where decoded_recorded is a list of decoded move strings,
               decoded_memory is a list of memory responses (lowercase),
               and match_count is the total count of frequency matches.
    """
    decoded_recorded = [format_decoded(decode_action(action)) for action in recorded_moves]
    decoded_memory = [event.get("response", "").lower() for event in memory_events]
    
    counter_recorded = Counter(decoded_recorded)
    counter_memory = Counter(decoded_memory)
    
    match_count = sum(min(counter_recorded[m], counter_memory.get(m, 0)) for m in counter_recorded)
    return decoded_recorded, decoded_memory, match_count

def main():
    num_games = 10
    all_game_moves = []
    
    # Run a series of games.
    for i in range(num_games):
        print(f"Running game {i+1}...")
        moves = run_random_game()
        all_game_moves.append(moves)
    
    # Combine moves from all games per player.
    combined_moves = aggregate_recorded_moves(all_game_moves)
    
    print("\n--- Comparing Decoded Recorded Moves to Opponent Memory ---\n")
    agents = list(combined_moves.keys())
    
    for player in agents:
        rec_moves = combined_moves[player]
        mem_events = collect_memory_events_for_player(player, agents)
        decoded_recorded, decoded_memory, match_count = compare_moves(rec_moves, mem_events)
        total_rec = len(decoded_recorded)
        total_mem = len(decoded_memory)
        diff_count = total_rec - match_count
        
        print(f"Player {player}:")
        print("  Recorded Moves (decoded):")
        print(f"    {decoded_recorded}")
        print("  Memory Responses (decoded) for this player (from opponents):")
        print(f"    {decoded_memory}")
        print(f"  Total Recorded Moves: {total_rec}")
        print(f"  Total Memory Events: {total_mem}")
        print(f"  Matches (by frequency): {match_count}")
        print(f"  Differences: {diff_count}")
        print("-" * 60)

if __name__ == "__main__":
    main()
