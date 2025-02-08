# src/model/memory.py

import numpy as np

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
    def __init__(self, decay=0.9):
        """
        Initialize a persistent per-agent opponent memory.
        The memory maps an opponent’s identifier to a list of events.
        
        Args:
            decay (float): A decay factor for older observations (if used later).
        """
        self.memory = {}  # {opponent_id: [event, event, ...]}
        self.decay = decay

    def update(self, opponent, response, triggering_action, penalties, card_count):
        """
        Record an event for a given opponent.
        
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
        if opponent not in self.memory:
            self.memory[opponent] = []
        self.memory[opponent].append(event)

    def get_summary(self, opponent):
        """
        Produce a fixed-length summary vector for an opponent based on stored events.
        Returns a vector (e.g., [challenge_rate, normalized_avg_card_count, normalized_avg_penalties, three_card_trigger_rate]).
        
        If no events exist for the opponent, returns a zero vector.
        """
        events = self.memory.get(opponent, [])
        if not events:
            return np.zeros(4, dtype=np.float32)
        
        total = len(events)
        challenge_count = sum(1 for e in events if e['response'] == "Challenge")
        avg_card_count = np.mean([e['card_count'] for e in events])
        avg_penalties = np.mean([e['penalties'] for e in events])
        three_card_trigger_count = sum(1 for e in events if e['triggering_action'] == "Play_3")
        
        summary = np.array([
            challenge_count / total,         # Challenge rate
            avg_card_count / 5.0,              # Normalized average card count (assuming max hand size of 5)
            avg_penalties / 3.0,               # Normalized average penalty count (if threshold is 3)
            three_card_trigger_count / total   # Fraction for 3-card triggers
        ], dtype=np.float32)
        return summary

    def decay_memory(self):
        """
        Optionally, prune the memory so that only recent events are kept.
        """
        max_events = 50  # For example, keep only the last 50 events per opponent
        for opponent, events in self.memory.items():
            if len(events) > max_events:
                self.memory[opponent] = events[-max_events:]

# Global dictionary to hold persistent opponent memories per agent.
PERSISTENT_OPPONENT_MEMORIES = {}

def get_opponent_memory(agent):
    """
    Returns the persistent OpponentMemory instance for the given agent.
    If one does not exist, it creates it.
    """
    if agent not in PERSISTENT_OPPONENT_MEMORIES:
        PERSISTENT_OPPONENT_MEMORIES[agent] = OpponentMemory()
    return PERSISTENT_OPPONENT_MEMORIES[agent]