# src/model/hard_coded_agents.py

import numpy as np
import random

class GreedyCardSpammer:
    def __init__(self, agent_name):
        self.name = agent_name
        
    def play_turn(self, observation, action_mask, table_card):
        """
        Strategy:
        1. Play maximum allowed non-table cards first
        2. Then play maximum allowed table cards
        3. Only challenge as last resort
        """
        hand_vector = observation[:2]  # From encode_hand()
        table_count_norm, non_table_count_norm = hand_vector
        total_cards = int(round(table_count_norm * 5 + non_table_count_norm * 5))
        
        # Calculate actual card counts (approximated from normalized values)
        table_cards = int(round(table_count_norm * 5))
        non_table_cards = int(round(non_table_count_norm * 5))
        
        # Action priorities: 5 -> 3 (max non-table first), then 2 -> 0 (table)
        for action in [5, 4, 3]:  # Non-table actions (count 3, 2, 1)
            if action_mask[action] == 1 and non_table_cards >= (action - 2):
                return action
                
        for action in [2, 1, 0]:  # Table actions (count 3, 2, 1)
            if action_mask[action] == 1 and table_cards >= (action + 1):
                return action
                
        return 6  # Challenge as last resort
    
class TableFirstConservativeChallenger:
    def __init__(self, agent_name):
        self.name = agent_name
    
    def play_turn(self, observation, action_mask, table_card):
        """
        Strategy:
        1. If holding 1 card total - always challenge
        2. Play maximum allowed table cards first (actions 0-2)
        3. Play non-table cards one at a time (action 3 only)
        4. Challenge as last resort
        """
        # Extract card counts from observation (normalized values)
        table_count = int(round(observation[0] * 5))  # Table cards + Jokers
        non_table_count = int(round(observation[1] * 5))
        total_cards = table_count + non_table_count

        # Immediate challenge if down to last card
        if total_cards == 1:
            return 6  # Challenge action

        # Try to play table cards (maximum possible first)
        for action in [2, 1, 0]:  # Check actions for 3, 2, 1 table cards
            required = (action % 3) + 1
            if action_mask[action] and table_count >= required:
                return action

        # Play single non-table card if possible
        if action_mask[3] and non_table_count >= 1:
            return 3

        # Final fallback to challenge
        return 6

class StrategicChallenger:
    def __init__(self, agent_name, num_players, agent_index):
        """
        Initialize the StrategicChallenger.

        Args:
            agent_name (str): Name of the agent.
            num_players (int): Total number of players in the game.
            agent_index (int): The index of this agent (e.g., 2 for 'player_2').
        """
        self.name = agent_name
        self.num_players = num_players
        self.agent_index = agent_index  # Assign directly without parsing from name

    def play_turn(self, observation, action_mask, table_card):
        """
        Strategy:
        1. Challenge if last action played 2+ cards
        2. Challenge in 2-player endgame with 2v1 cards
        3. Play 1 non-table card (action 3) if possible
        4. Play 1 table card (action 0) if possible
        5. Final fallback to challenge
        """
        # Extract observation components
        hand_vector = observation[:2]
        last_action_count = int(round(observation[2]))  # Normalized to actual count
        active_players_vector = observation[3:3+self.num_players]
        
        # Calculate actual card counts
        table_cards = int(round(hand_vector[0] * 5))
        non_table_cards = int(round(hand_vector[1] * 5))
        total_cards = table_cards + non_table_cards
        
        # 1. Challenge if last play was 2+ cards
        if last_action_count >= 2:
            return 6

        # 2. Endgame challenge condition
        active_counts = [int(round(c * 5)) for c in active_players_vector]
        active_agents = [c for c in active_counts if c > 0]
        
        if len(active_agents) == 2:
            my_cards = active_counts[self.agent_index]
            opponent_cards = sum(active_counts) - my_cards
            if my_cards == 2 and opponent_cards == 1:
                return 6

        # 3. Play 1 non-table card (action 3)
        if action_mask[3] and non_table_cards >= 1:
            return 3
            
        # 4. Play 1 table card (action 0)
        if action_mask[0] and table_cards >= 1:
            return 0
            
        # 5. Final challenge
        return 6

class RandomAgent:
    def __init__(self, agent_name):
        self.name = agent_name        
    
    def play_turn(self, observation, action_mask, table_card):
        return random.choice([i for i in range(7) if action_mask[i] == 1])
