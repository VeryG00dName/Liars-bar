import torch
import numpy as np

def compute_bayes_update(prior, likelihood, evidence=None):
    """
    Apply Bayes' rule to update beliefs.
    
    Args:
        prior: Prior belief distribution [batch_size, num_states]
        likelihood: P(evidence | state) [batch_size, num_states]
        evidence: Optional evidence tensor
        
    Returns:
        Posterior belief distribution [batch_size, num_states]
    """
    # Bayes' rule: P(state | evidence) ∝ P(evidence | state) × P(state)
    posterior = prior * likelihood
    
    # Normalize
    posterior_sum = posterior.sum(dim=-1, keepdim=True)
    mask = (posterior_sum > 0)
    posterior[mask] = posterior[mask] / posterior_sum[mask]
    
    # If all zero (no valid posterior), use prior
    posterior[~mask] = prior[~mask]
    
    return posterior

def traverse_decision_points(env_history, agent_id=None):
    """
    Extract decision points from environment history.
    
    Args:
        env_history: History from the environment
        agent_id: Optional agent ID to filter history
        
    Returns:
        List of decision points
    """
    decision_points = []
    
    if isinstance(env_history, dict):
        # If it's already organized by agent
        if agent_id and agent_id in env_history:
            return env_history[agent_id]
        
        # Otherwise, flatten all agents' histories
        for agent, history in env_history.items():
            if agent_id and agent != agent_id:
                continue
            for point in history:
                point['agent'] = agent
                decision_points.append(point)
    elif isinstance(env_history, list):
        # If it's a flat list of history entries
        for entry in env_history:
            entry_agent = entry.get('agent')
            if agent_id and entry_agent != agent_id:
                continue
            decision_points.append(entry)
    
    return decision_points

def compute_counterfactual_reach(actions, beliefs, policy):
    """
    Compute counterfactual reach probabilities.
    
    Args:
        actions: List of actions taken
        beliefs: Prior belief distribution
        policy: Policy function that returns action probabilities
        
    Returns:
        Updated belief distribution using counterfactual reasoning
    """
    updated_beliefs = beliefs.clone()
    
    for action in actions:
        action_type = action.get('action_type')
        action_id = action.get('action')
        if action_type and action_id is not None:
            # Get action probabilities under different states
            action_probs = policy(action_type, action_id)
            
            # Update beliefs using Bayes' rule
            updated_beliefs = compute_bayes_update(updated_beliefs, action_probs)
    
    return updated_beliefs

def normalize_belief_distribution(beliefs, epsilon=1e-8):
    """
    Normalize a belief distribution to ensure it sums to 1.
    
    Args:
        beliefs: Belief distribution tensor
        epsilon: Small value to avoid division by zero
        
    Returns:
        Normalized belief distribution
    """
    sum_vals = beliefs.sum(dim=-1, keepdim=True)
    normalized = beliefs / (sum_vals + epsilon)
    return normalized
