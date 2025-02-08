# src/model/memory.py

import numpy as np
from collections import deque

class RolloutMemory:
    def __init__(self, agents):
        """
        Initializes the Rollout Memory for storing experiences.

        Args:
            agents (list): List of agent identifiers.
        """
        self.agents = agents
        self.reset()

    def reset(self):
        """
        Clears all stored experiences.
        """
        self.states = {agent: [] for agent in self.agents}
        self.actions = {agent: [] for agent in self.agents}
        self.log_probs = {agent: [] for agent in self.agents}
        self.rewards = {agent: [] for agent in self.agents}
        self.is_terminals = {agent: [] for agent in self.agents}
        self.state_values = {agent: [] for agent in self.agents}
        self.advantages = {agent: [] for agent in self.agents}
        self.returns = {agent: [] for agent in self.agents}
        self.action_masks = {agent: [] for agent in self.agents}

    def store_transition(self, agent, state, action, log_prob, reward, is_terminal, state_value, action_mask):
        """
        Stores a single transition for a specific agent.

        Args:
            agent (str): Agent identifier.
            state (np.ndarray): Observation/state.
            action (int): Action taken.
            log_prob (float): Log probability of the action.
            reward (float): Reward received.
            is_terminal (bool): Flag indicating if the episode ended.
            state_value (float): Estimated value of the state.
        """
        self.states[agent].append(state)
        self.actions[agent].append(action)
        self.log_probs[agent].append(log_prob)
        self.rewards[agent].append(reward)
        self.is_terminals[agent].append(is_terminal)
        self.state_values[agent].append(state_value)
        self.action_masks[agent].append(action_mask)


class OpponentMemory:
    def __init__(self, max_events=100):
        """
        Initialize a persistent per-agent opponent memory with aggregates.
        
        Args:
            max_events (int): Maximum number of events to store per opponent.
        """
        # Store events as a deque with a fixed maximum length.
        self.memory = {}         # {opponent_id: deque([event, event, ...], maxlen=max_events)}
        # Aggregated statistics for each opponent.
        self.aggregates = {}     # {opponent_id: {'total': int, 'challenge_count': int, 
                                 #                'card_count_sum': int, 'penalties_sum': int,
                                 #                'three_card_trigger_count': int}}
        self.max_events = max_events

    def update(self, opponent, response, triggering_action, penalties, card_count):
        """
        Record an event for a given opponent and update the running aggregates.
        
        Args:
            opponent (str): Opponent's identifier.
            response (str): The action taken by the opponent (e.g., "Challenge", "Play").
            triggering_action (str): The action that prompted the opponent’s response.
            penalties (int): The opponent’s penalty count at that moment.
            card_count (int): The opponent’s number of cards at that moment.
        """
        event = {
            'response': response,
            'triggering_action': triggering_action,
            'penalties': penalties,
            'card_count': card_count,
        }
        # Initialize data structures if needed.
        if opponent not in self.memory:
            self.memory[opponent] = deque(maxlen=self.max_events)
            self.aggregates[opponent] = {
                'total': 0,
                'challenge_count': 0,
                'card_count_sum': 0,
                'penalties_sum': 0,
                'three_card_trigger_count': 0
            }

        # If the deque is full, remove the oldest event and subtract its contributions.
        if len(self.memory[opponent]) == self.max_events:
            old_event = self.memory[opponent][0]
            self.aggregates[opponent]['total'] -= 1
            if old_event['response'] == "Challenge":
                self.aggregates[opponent]['challenge_count'] -= 1
            self.aggregates[opponent]['card_count_sum'] -= old_event['card_count']
            self.aggregates[opponent]['penalties_sum'] -= old_event['penalties']
            if old_event['triggering_action'] == "Play_3":
                self.aggregates[opponent]['three_card_trigger_count'] -= 1

        # Append the new event.
        self.memory[opponent].append(event)
        # Update aggregates.
        self.aggregates[opponent]['total'] += 1
        if response == "Challenge":
            self.aggregates[opponent]['challenge_count'] += 1
        self.aggregates[opponent]['card_count_sum'] += card_count
        self.aggregates[opponent]['penalties_sum'] += penalties
        if triggering_action == "Play_3":
            self.aggregates[opponent]['three_card_trigger_count'] += 1

    def get_summary(self, opponent):
        """
        Produce a fixed-length summary vector for an opponent based on stored events.
        Returns a vector (e.g., [challenge_rate, normalized_avg_card_count, normalized_avg_penalties, three_card_trigger_rate]).
        
        If no events exist for the opponent, returns a zero vector.
        This method now runs in O(1) time.
        """
        agg = self.aggregates.get(opponent)
        if not agg or agg['total'] == 0:
            return np.zeros(4, dtype=np.float32)
        total = agg['total']
        challenge_rate = agg['challenge_count'] / total
        avg_card_count = agg['card_count_sum'] / total
        avg_penalties = agg['penalties_sum'] / total
        three_card_rate = agg['three_card_trigger_count'] / total
        summary = np.array([
            challenge_rate,       # Challenge rate
            avg_card_count / 5.0, # Normalized average card count (assuming max hand size 5)
            avg_penalties / 3.0,  # Normalized average penalties (if threshold is 3)
            three_card_rate       # Fraction of 3-card triggers
        ], dtype=np.float32)
        return summary


# Global dictionary to hold persistent opponent memories per agent.
PERSISTENT_OPPONENT_MEMORIES = {}

def get_opponent_memory(agent):
    """
    Returns the persistent OpponentMemory instance for the given agent.
    If one does not exist, it creates it.
    """
    if agent not in PERSISTENT_OPPONENT_MEMORIES:
        PERSISTENT_OPPONENT_MEMORIES[agent] = OpponentMemory(max_events=50)
    return PERSISTENT_OPPONENT_MEMORIES[agent]