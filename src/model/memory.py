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
    def __init__(self, max_events=200):
        """
        Initialize per-agent opponent memory with separate early/late aggregates.
        
        Args:
            max_events (int): Maximum number of events to store per opponent.
        """
        self.memory = {}     # {opponent_id: deque(..., maxlen=max_events)}
        self.aggregates = {} # {opponent_id: {early_total, late_total, early_challenge_count, late_challenge_count,
                             #                early_three_card_trigger_count, late_three_card_trigger_count}}
        self.max_events = max_events

    def update(self, opponent, response, triggering_action, penalties, card_count):
        """
        Record an event and update early/late aggregates based on the card count.
        
        Args:
            opponent (str): Opponent's identifier.
            response (str): E.g., "Challenge" or another response type.
            triggering_action (str): E.g., "Play_3" if it's a three-card play.
            penalties (int): Current penalty count (not used in this summary, but you might log it).
            card_count (int): Current card count of the opponent.
                             (If card_count < 3, consider the event as occurring in the late phase.)
        """
        event = {
            'response': response,
            'triggering_action': triggering_action,
            'penalties': penalties,
            'card_count': card_count
        }
        
        # Initialize storage for opponent if necessary.
        if opponent not in self.memory:
            self.memory[opponent] = deque(maxlen=self.max_events)
            self.aggregates[opponent] = {
                'early_total': 0,
                'late_total': 0,
                'early_challenge_count': 0,
                'late_challenge_count': 0,
                'early_three_card_trigger_count': 0,
                'late_three_card_trigger_count': 0
            }
        
        # (Optionally, if the deque is full, you might subtract the oldest event's contribution here.)

        self.memory[opponent].append(event)
        agg = self.aggregates[opponent]
        
        # Use the opponent's card count to determine if the event is early or late.
        if card_count < 3:
            # Late event
            agg['late_total'] += 1
            if response == "Challenge":
                agg['late_challenge_count'] += 1
            if triggering_action == "Play_3":
                agg['late_three_card_trigger_count'] += 1
        else:
            # Early event
            agg['early_total'] += 1
            if response == "Challenge":
                agg['early_challenge_count'] += 1
            if triggering_action == "Play_3":
                agg['early_three_card_trigger_count'] += 1

    def get_summary(self, opponent):
        """
        Produce a summary vector with early/late challenge rates and three-card challenge rates.
        Returns a vector of shape (4,).
        """
        agg = self.aggregates.get(opponent, None)
        if not agg:
            return np.zeros(4, dtype=np.float32)
        
        early_total = agg['early_total']
        late_total = agg['late_total']
        
        early_challenge_rate = (agg['early_challenge_count'] / early_total) if early_total > 0 else 0.0
        late_challenge_rate = (agg['late_challenge_count'] / late_total) if late_total > 0 else 0.0
        early_three_rate = (agg['early_three_card_trigger_count'] / early_total) if early_total > 0 else 0.0
        late_three_rate = (agg['late_three_card_trigger_count'] / late_total) if late_total > 0 else 0.0
        
        summary = np.array([
            early_challenge_rate,
            late_challenge_rate,
            early_three_rate,
            late_three_rate
        ], dtype=np.float32)
        return summary

    def get_full_memory(self, opponent):
        """
        Return the full memory (all recorded events) for the given opponent as a list.
        If no events are recorded, returns an empty list.
        """
        if opponent in self.memory:
            return list(self.memory[opponent])
        else:
            return []
        
# Global dictionary to hold persistent opponent memories per agent.
PERSISTENT_OPPONENT_MEMORIES = {}

def get_opponent_memory(agent):
    if agent not in PERSISTENT_OPPONENT_MEMORIES:
        PERSISTENT_OPPONENT_MEMORIES[agent] = OpponentMemory(max_events=200)
    return PERSISTENT_OPPONENT_MEMORIES[agent]