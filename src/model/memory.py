# src/model/memory.py

import numpy as np
from collections import deque, namedtuple

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
        self.expert_inputs = {agent: [] for agent in self.agents}

    def store_transition(self, agent, state, action, log_prob, reward, is_terminal, state_value, action_mask, expert_input=None):
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
        self.expert_inputs[agent].append(expert_input)

class SumTree:
    """
    A binary sum tree data structure used for efficient priority-based sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (transitions)
        self.tree = np.zeros(2 * capacity - 1)  # Tree array: [internal nodes | leaf nodes]
        self.data_pointer = 0  # Current position to write new data
        self.size = 0  # Current number of elements
        
    def add(self, priority, data_idx):
        """Add new data with its priority to the tree."""
        tree_idx = self.data_pointer + self.capacity - 1  # Index in the tree array
        
        # Update the leaf node
        self.tree[tree_idx] = priority
        
        # Propagate the change up through the tree
        self.update(tree_idx)
        
        # Update data pointer and size
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        return tree_idx
        
    def update(self, tree_idx, priority=None):
        """Update the priority of a node and propagate the change through the tree."""
        if priority is not None:
            self.tree[tree_idx] = priority
            
        # Get the parent node and update it
        parent = (tree_idx - 1) // 2
        
        while parent >= 0:
            # Parent's value is the sum of its children
            left = 2 * parent + 1
            right = left + 1
            self.tree[parent] = self.tree[left] + self.tree[right]
            parent = (parent - 1) // 2
    
    def get_leaf(self, v):
        """
        Get a leaf node based on a value.
        
        Args:
            v (float): Value to search for
            
        Returns:
            tuple: (tree_idx, priority, data_idx)
        """
        parent = 0
        
        while True:
            left = 2 * parent + 1
            right = left + 1
            
            # If we reach a leaf node, break
            if left >= len(self.tree):
                break
                
            # Otherwise, go left or right
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left]
                parent = right
                
        tree_idx = parent
        data_idx = tree_idx - self.capacity + 1
        
        return tree_idx, self.tree[tree_idx], data_idx
    
    def total_priority(self):
        """Return the sum of all priorities."""
        return self.tree[0]

Transition = namedtuple('Transition', 
                        ['state', 'action', 'log_prob', 'reward', 'is_terminal', 
                         'state_value', 'action_mask', 'expert_input'])

class PrioritizedReplayBuffer:
    """
    A prioritized replay buffer that stores experiences and samples them based on priority.
    """
    def __init__(self, agents, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            agents (list): List of agent identifiers.
            capacity (int): Maximum size of the buffer.
            alpha (float): Controls how much prioritization is used (0 = uniform, 1 = full prioritization).
            beta (float): Controls importance sampling weights (0 = no correction, 1 = full correction).
            beta_increment (float): Amount to increase beta per sampling.
            epsilon (float): Small value to add to priorities to ensure non-zero probability.
        """
        self.agents = agents
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initial max priority
        
        # Create a separate buffer for each agent
        self.buffers = {}
        for agent in agents:
            self.buffers[agent] = {
                'tree': SumTree(capacity),
                'transitions': deque(maxlen=capacity)
            }
            
    def store_transition(self, agent, state, action, log_prob, reward, is_terminal, state_value, action_mask, expert_input=None):
        """
        Store a transition with max priority.
        
        Args:
            agent (str): Agent identifier.
            state (np.ndarray): Observation/state.
            action (int): Action taken.
            log_prob (float): Log probability of the action.
            reward (float): Reward received.
            is_terminal (bool): Flag indicating if the episode ended.
            state_value (float): Estimated value of the state.
            action_mask (list): Mask of valid actions.
            expert_input (tuple, optional): Additional information for auxiliary tasks.
        """
        # Create a transition object
        transition = Transition(
            state=state,
            action=action,
            log_prob=log_prob,
            reward=reward,
            is_terminal=is_terminal,
            state_value=state_value,
            action_mask=action_mask,
            expert_input=expert_input
        )
        
        # Add transition to the buffer
        self.buffers[agent]['transitions'].append(transition)
        
        # Calculate priority based on max priority
        priority = (self.max_priority + self.epsilon) ** self.alpha
        
        # Add to sum tree
        idx = len(self.buffers[agent]['transitions']) - 1
        self.buffers[agent]['tree'].add(priority, idx)
    
    def sample(self, agent, batch_size):
        """
        Sample a batch of transitions based on priority.
        
        Args:
            agent (str): Agent identifier.
            batch_size (int): Number of transitions to sample.
            
        Returns:
            tuple: (batch, indices, importance_weights)
        """
        buffer = self.buffers[agent]
        tree = buffer['tree']
        transitions = buffer['transitions']
        
        # Get actual size (may be less than capacity if not filled yet)
        n = min(len(transitions), tree.size)
        if n == 0:
            return None, None, None
        
        batch_size = min(batch_size, n)
        
        # Prepare batch arrays
        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate segment size
        segment = tree.total_priority() / batch_size
        
        # Increment beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get batch of experiences
        for i in range(batch_size):
            # Get a random value within the segment
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            
            # Get experience from the tree
            tree_idx, priority, data_idx = tree.get_leaf(v)
            
            # Ensure data_idx is valid
            if data_idx < 0 or data_idx >= len(transitions):
                continue
                
            # Store data
            batch.append(transitions[data_idx])
            indices[i] = tree_idx
            priorities[i] = priority
        
        # Calculate importance sampling weights
        sampling_probabilities = priorities / tree.total_priority()
        importance_weights = np.power(n * sampling_probabilities, -self.beta)
        importance_weights /= importance_weights.max()  # Normalize
        
        return batch, indices, importance_weights
    
    def update_priorities(self, agent, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Args:
            agent (str): Agent identifier.
            indices (list): List of tree indices.
            td_errors (list): List of TD errors.
        """
        for idx, td_error in zip(indices, td_errors):
            # Calculate priority from TD error
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update tree
            self.buffers[agent]['tree'].update(idx, priority)
    
    def is_ready(self, agent, min_size=1000):
        """Check if the buffer has enough data for sampling."""
        return len(self.buffers[agent]['transitions']) >= min_size
    
    def size(self, agent):
        """Return the current size of the buffer for an agent."""
        return len(self.buffers[agent]['transitions'])
    
    def reset(self):
        """Clear all buffers."""
        for agent in self.agents:
            self.buffers[agent] = {
                'tree': SumTree(self.capacity),
                'transitions': deque(maxlen=self.capacity)
            }

class OpponentMemory:
    def __init__(self, max_events=400):
        """
        Initialize per-agent opponent memory with separate early/late aggregates.
        
        Args:
            max_events (int): Maximum number of events to store per opponent.
        """
        self.memory = {}     # {opponent_id: deque(..., maxlen=max_events)}
        self.aggregates = {} # {opponent_id: {early_total, late_total, early_challenge_count, late_challenge_count,
                             #                early_three_card_trigger_count, late_three_card_trigger_count}}
        self.max_events = max_events

    def update(self, opponent, response, triggering_action, penalties, card_count, challenge_success=None):
        """
        Record an event with challenge outcome information.
        
        Args:
            opponent (str): Opponent's identifier.
            response (str): E.g., "Challenge" or another response type.
            triggering_action (str): E.g., "Play_3" if it's a three-card play.
            penalties (int): Current penalty count.
            card_count (int): Current card count of the opponent.
            challenge_success (bool, optional): Whether a challenge was successful.
                                            (True means the play was a bluff)
        """
        event = {
            'response': response,
            'triggering_action': triggering_action,
            'penalties': penalties,
            'card_count': card_count,
            'challenge_success': challenge_success
        }
        
        # Initialize storage for opponent if necessary
        if opponent not in self.memory:
            self.memory[opponent] = deque(maxlen=self.max_events)
            self.aggregates[opponent] = {
                'early_total': 0,
                'late_total': 0,
                'early_challenge_count': 0,
                'late_challenge_count': 0,
                'early_three_card_trigger_count': 0,
                'late_three_card_trigger_count': 0,
                'early_successful_challenge_count': 0,
                'late_successful_challenge_count': 0
            }

        self.memory[opponent].append(event)
        agg = self.aggregates[opponent]
        
        # Use the opponent's card count to determine if the event is early or late
        is_late = card_count < 3
        phase_prefix = 'late_' if is_late else 'early_'
        
        # Update base counts
        agg[f'{phase_prefix}total'] += 1
        if response == "Challenge":
            agg[f'{phase_prefix}challenge_count'] += 1
        if triggering_action == "Play_3":
            agg[f'{phase_prefix}three_card_trigger_count'] += 1
        
        # Track successful challenges (implies the play was a bluff)
        if challenge_success is not None and challenge_success:
            agg[f'{phase_prefix}successful_challenge_count'] += 1

    def update_last_play(self, opponent, challenge_success):
        """
        Updates the most recent 'Play' action for the given opponent with challenge result.
        
        Args:
            opponent (str): The opponent whose memory we're updating
            challenge_success (bool): Whether the challenge against their play was successful
                                    (True means the play was a bluff)
        
        Returns:
            bool: Whether the update was successful
        """
        if opponent not in self.memory or not self.memory[opponent]:
            return False
            
        # Iterate through memory in reverse to find the last Play action
        for i in range(len(self.memory[opponent])-1, -1, -1):
            event = self.memory[opponent][i]
            if event['response'].startswith('Play_'):
                # Update this event with the challenge result
                self.memory[opponent][i]['challenge_success'] = challenge_success
                
                # Update aggregates if needed
                agg = self.aggregates[opponent]
                is_late = event['card_count'] < 3
                phase_prefix = 'late_' if is_late else 'early_'
                
                if challenge_success:
                    agg[f'{phase_prefix}successful_challenge_count'] += 1
                    
                return True
        
        return False

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
        PERSISTENT_OPPONENT_MEMORIES[agent] = OpponentMemory(max_events=400)
    return PERSISTENT_OPPONENT_MEMORIES[agent]

def clear_opponent_memory(agent, opponent):
    """
    Clear the memory of a specific opponent for a given agent.
    
    Args:
        agent (str): The agent identifier.
        opponent (str): The opponent identifier.
    """
    if agent in PERSISTENT_OPPONENT_MEMORIES and opponent in PERSISTENT_OPPONENT_MEMORIES[agent].memory:
        del PERSISTENT_OPPONENT_MEMORIES[agent].memory[opponent]
        del PERSISTENT_OPPONENT_MEMORIES[agent].aggregates[opponent]
        
def delete_opponent_memory():
    """
    Delete all persistent opponent memories.
    """
    global PERSISTENT_OPPONENT_MEMORIES
    PERSISTENT_OPPONENT_MEMORIES = {}