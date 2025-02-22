# src/model/hard_coded_agents.py

import numpy as np
import random
from src import config
if config.SEED:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
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

class SelectiveTableConservativeChallenger:
    def __init__(self, agent_name):
        self.name = agent_name

    def play_turn(self, observation, action_mask, table_card):
        """
        Strategy:
        1. If holding 1 card total, challenge.
        2. If exactly 1 table card is held:
           a. If more than one non-table card is held, play 1 non-table card (action 3).
           b. If exactly 1 non-table card is held (i.e. hand is [1 table, 1 non-table]),
              play the table card (action 0) to force a one-card situation next turn.
        3. Otherwise, follow the TableFirstConservativeChallenger logic:
           a. Play maximum allowed table cards (actions 2, 1, 0).
           b. Then play a non-table card (action 3).
           c. Challenge as last resort.
        """
        # Extract card counts from observation (normalized values)
        table_count = int(round(observation[0] * 5))  # Table cards + Jokers
        non_table_count = int(round(observation[1] * 5))
        total_cards = table_count + non_table_count

        # Immediate challenge if only one card remains.
        if total_cards == 1:
            return 6  # Challenge action

        # Special handling when there is exactly one table card.
        if table_count == 1:
            # If there are more non-table cards than one, play a non-table card.
            if non_table_count > 1:
                if action_mask[3]:
                    return 3  # Play one non-table card
            # If there is exactly one non-table card, play the table card.
            elif non_table_count == 1:
                if action_mask[0]:
                    return 0  # Play the table card to force one remaining card next turn

        # Default strategy: try to play table cards (maximum allowed first).
        # Here, actions 2, 1, 0 correspond to playing 3, 2, and 1 table cards respectively.
        for action in [2, 1, 0]:
            required = (action % 3) + 1  # 3, 2, or 1 table cards respectively
            if action_mask[action] and table_count >= required:
                return action

        # Next, try to play one non-table card.
        if action_mask[3] and non_table_count >= 1:
            return 3

        # If no other move is available, challenge.
        return 6

class TableNonTableAgent:
    def __init__(self, agent_name):
        self.name = agent_name
        # This flag is used to commit to playing table cards consecutively when starting from 2 table cards.
        self.commit_to_table = False

    def play_turn(self, observation, action_mask, table_card):
        """
        Strategy:
        1. If 3 or more table cards are available, play 3 table cards (action 2).
        2. If in the middle of a 2-table-card series (commit mode), play a table card (action 0).
        3. If exactly 2 table cards are available (and not already committed), initiate the series by playing one table card (action 0).
        4. If non-table cards are 3 or more, play 2 non-table cards (action 4).
        5. If exactly 1 table card is available (and not in commit mode):
           - If there are any non-table cards, play one non-table card (action 3).
           - Otherwise, play the table card (action 0).
        6. If no table cards remain, try playing one non-table card (action 3).
        7. If none of the above applies, challenge (action 6).
        
        Notes:
          - We assume that the observation’s first two values are normalized counts for table and non-table cards.
          - The counts are rescaled by 5 (as in the other bots).
          - Action indices:
              - Action 2: play 3 table cards.
              - Action 0: play 1 table card.
              - Action 4: play 2 non-table cards.
              - Action 3: play 1 non-table card.
              - Action 6: challenge.
        """
        # Decode card counts (using the same scaling as the other agents).
        table_cards = int(round(observation[0] * 5))
        non_table_cards = int(round(observation[1] * 5))

        # 1. If 3 or more table cards: play 3 table cards.
        if table_cards >= 3 and action_mask[2] == 1:
            self.commit_to_table = False  # clear any previous commit mode
            return 2

        # 2. If we are in commit mode, force playing a table card.
        if self.commit_to_table:
            if table_cards > 0 and action_mask[0] == 1:
                return 0
            else:
                # If for some reason we can no longer play a table card, cancel commit mode.
                self.commit_to_table = False

        # 3. If exactly 2 table cards, start the commit mode and play one table card.
        if table_cards == 2 and action_mask[0] == 1:
            self.commit_to_table = True
            return 0

        # 4. If non-table cards are 3 or more, play 2 non-table cards.
        if non_table_cards >= 3 and action_mask[4] == 1:
            return 4

        # 5. If exactly 1 table card (and not in commit mode):
        if table_cards == 1:
            # Prefer playing non-table cards one at a time if available.
            if non_table_cards > 0 and action_mask[3] == 1:
                return 3
            # Otherwise, play the table card.
            if action_mask[0] == 1:
                return 0

        # 6. If no table cards remain, try playing one non-table card.
        if table_cards == 0:
            if non_table_cards >= 1 and action_mask[3] == 1:
                return 3

        # 7. Fallback: challenge.
        return 6

class Classic:
    def __init__(self, agent_name):
        self.name = agent_name

    def play_turn(self, observation, action_mask, table_card):
        """
        Classic Bot Strategy:
        1. Challenge if the opponent has exactly 1 card left.
        2. Challenge if the opponent's last played move involved more than 1 card.
        3. Otherwise, play 1 table card at a time (action 0) as long as table cards remain.
        4. If no table cards remain, play 1 non-table card at a time (action 3).
        5. Fallback: challenge (action 6).

        Note: This implementation assumes a 2-player game.
        """
        # Decode card counts (using the same scaling as the other bots).
        table_cards = int(round(observation[0] * 5))
        non_table_cards = int(round(observation[1] * 5))
        total_cards = table_cards + non_table_cards

        # Determine how many cards the opponent last played.
        last_action_count = int(round(observation[2]))

        # Extract active players' card counts.
        # Assuming a 2-player game, observation[3:5] contains normalized counts for both players.
        active_counts = [int(round(c * 5)) for c in observation[3:5]]
        # Our card count is total_cards; thus the opponent's card count is:
        opponent_cards = sum(active_counts) - total_cards

        # 1. Challenge if the opponent has exactly 1 card left.
        if opponent_cards == 1:
            return 6

        # 2. Challenge if the opponent's last play involved more than 1 card.
        if last_action_count > 1:
            return 6

        # 3. If we have any table cards, play 1 table card (action 0).
        if table_cards > 0 and action_mask[0] == 1:
            return 0

        # 4. If no table cards remain but we have non-table cards, play 1 non-table card (action 3).
        if non_table_cards > 0 and action_mask[3] == 1:
            return 3

        # 5. Fallback: challenge.
        return 6

class RandomAgent:
    def __init__(self, agent_name):
        self.name = agent_name        
    
    def play_turn(self, observation, action_mask, table_card):
        return random.choice([i for i in range(7) if action_mask[i] == 1])
