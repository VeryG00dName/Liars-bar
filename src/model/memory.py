# src/model/memory.py

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

    def store_transition(self, agent, state, action, log_prob, reward, is_terminal, state_value):
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
