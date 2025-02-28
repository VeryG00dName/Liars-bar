# src/model/opponent_memory.py
import numpy as np
from collections import defaultdict

class OpponentMemory:
    """
    In-memory module that tracks opponent tendencies across games in a single training run.
    """
    def __init__(self, owner_id):
        self.owner_id = owner_id  # The agent who owns this memory
        self.memory = {}
    
    def update(self, opponent, response, triggering_action=None, penalties=0, card_count=0):
        """
        Update memory with a new observation of an opponent's action.
        
        Args:
            opponent: Opponent ID
            response: Action taken by opponent (e.g., "Play_2", "Challenge")
            triggering_action: What preceded this action (optional)
            penalties: Current penalty count for this opponent
            card_count: Current card count for this opponent
        """
        if opponent not in self.memory:
            self.memory[opponent] = {
                'play_counts': {},
                'bluff_rates': {},  # {play_count: [bluff_count, total_count]}
                'challenge_rates': {},  # {target: [success_count, total_count]}
                'action_history': [],
                'penalties_history': [],
                'games_played': 0
            }
        
        # Update opponent record
        record = self.memory[opponent]
        
        # Record this action
        action_entry = {
            'response': response,
            'trigger': triggering_action,
            'penalties': penalties,
            'card_count': card_count
        }
        record['action_history'].append(action_entry)
        
        # Trim history if it gets too long
        if len(record['action_history']) > 1000:
            record['action_history'] = record['action_history'][-1000:]
        
        # Update specific statistics based on action type
        if response.startswith('Play_'):
            count = int(response.split('_')[1])
            record['play_counts'][count] = record['play_counts'].get(count, 0) + 1
    
    def record_bluff(self, opponent, was_bluff, play_count):
        """
        Record whether an opponent's play was a bluff.
        
        Args:
            opponent: Opponent ID
            was_bluff: Boolean indicating if the play was a bluff
            play_count: Number of cards played
        """
        if opponent not in self.memory:
            return
            
        # Update bluff statistics
        record = self.memory[opponent]
        if play_count not in record['bluff_rates']:
            record['bluff_rates'][play_count] = [0, 0]  # [bluff_count, total_count]
            
        bluff_stats = record['bluff_rates'][play_count]
        if was_bluff:
            bluff_stats[0] += 1
        bluff_stats[1] += 1
    
    def record_challenge_result(self, opponent, success, target=None):
        """
        Record the result of a challenge.
        
        Args:
            opponent: Opponent who made the challenge
            success: Whether the challenge was successful
            target: Who was challenged (optional)
        """
        if opponent not in self.memory:
            return
            
        # Update challenge statistics
        record = self.memory[opponent]
        target_key = target if target else 'overall'
        
        if target_key not in record['challenge_rates']:
            record['challenge_rates'][target_key] = [0, 0]  # [success_count, total_count]
            
        challenge_stats = record['challenge_rates'][target_key]
        if success:
            challenge_stats[0] += 1
        challenge_stats[1] += 1
    
    def new_game(self, opponent):
        """Record that a new game has started with this opponent."""
        if opponent not in self.memory:
            self.memory[opponent] = {
                'play_counts': {},
                'bluff_rates': {},
                'challenge_rates': {},
                'action_history': [],
                'penalties_history': [],
                'games_played': 0
            }
        
        self.memory[opponent]['games_played'] += 1
        self.memory[opponent]['penalties_history'].append(0)
    
    def get_bluff_tendency(self, opponent, play_count=None):
        """
        Get the opponent's tendency to bluff with a specific play count.
        
        Args:
            opponent: Opponent ID
            play_count: Number of cards played (if None, returns overall tendency)
            
        Returns:
            Float between 0-1 representing bluff probability, 0.5 if unknown
        """
        if opponent not in self.memory:
            return 0.5  # Default to 50% for unknown opponents
        
        record = self.memory[opponent]
        
        if play_count is not None:
            # Get specific play count bluff rate
            if play_count not in record['bluff_rates']:
                return 0.5  # Default if no data
                
            bluff_stats = record['bluff_rates'][play_count]
            if bluff_stats[1] > 0:
                return bluff_stats[0] / bluff_stats[1]
            else:
                return 0.5  # Default if no data
        else:
            # Calculate overall bluff rate
            total_bluffs = sum(stats[0] for stats in record['bluff_rates'].values())
            total_plays = sum(stats[1] for stats in record['bluff_rates'].values())
            if total_plays > 0:
                return total_bluffs / total_plays
            else:
                return 0.5  # Default if no data
    
    def get_challenge_tendency(self, opponent, target=None):
        """
        Get the opponent's tendency to challenge successfully.
        
        Args:
            opponent: Opponent ID
            target: Specific target (if None, returns overall tendency)
            
        Returns:
            Float between 0-1 representing challenge success rate, 0.5 if unknown
        """
        if opponent not in self.memory:
            return 0.5  # Default for unknown opponents
        
        record = self.memory[opponent]
        
        if target is not None:
            # Get specific target challenge rate
            target_key = target if target else 'overall'
            if target_key not in record['challenge_rates']:
                return 0.5  # Default if no data
                
            challenge_stats = record['challenge_rates'][target_key]
            if challenge_stats[1] > 0:
                return challenge_stats[0] / challenge_stats[1]
            else:
                return 0.5  # Default if no data
        else:
            # Calculate overall challenge success rate
            total_successes = sum(stats[0] for stats in record['challenge_rates'].values())
            total_challenges = sum(stats[1] for stats in record['challenge_rates'].values())
            if total_challenges > 0:
                return total_successes / total_challenges
            else:
                return 0.5  # Default if no data
    
    def get_summary(self, opponent):
        """
        Get a summary vector representing the opponent's playstyle.
        
        Args:
            opponent: Opponent ID
            
        Returns:
            numpy array of features describing the opponent
        """
        if opponent not in self.memory:
            return np.array([0.5, 0.5, 0.5, 0.5, 0, 0], dtype=np.float32)
        
        record = self.memory[opponent]
        
        # Create feature vector
        features = np.array([
            self.get_bluff_tendency(opponent, play_count=1),  # Bluff rate with 1 card
            self.get_bluff_tendency(opponent, play_count=2),  # Bluff rate with 2 cards
            self.get_bluff_tendency(opponent, play_count=3),  # Bluff rate with 3 cards
            self.get_challenge_tendency(opponent),  # Challenge success rate
            min(record['games_played'] / 50, 1.0),  # Normalized games played (cap at 50)
            min(len(record['action_history']) / 500, 1.0)  # Normalized action history (cap at 500)
        ], dtype=np.float32)
        
        return features
    
    def get_full_memory(self, opponent):
        """
        Get the full memory record for an opponent.
        
        Args:
            opponent: Opponent ID
            
        Returns:
            Full memory record or empty dict if opponent unknown
        """
        return self.memory.get(opponent, {})

# Global registry to access opponent memory from anywhere
_opponent_memories = {}

def get_opponent_memory(agent_id):
    """Get the opponent memory for a specific agent."""
    if agent_id not in _opponent_memories:
        _opponent_memories[agent_id] = OpponentMemory(agent_id)
    return _opponent_memories[agent_id]