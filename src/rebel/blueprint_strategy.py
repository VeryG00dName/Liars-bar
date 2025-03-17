# src/model/blueprint_strategy.py
import gzip
import hashlib
import json
import time
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
from collections import defaultdict
import os
import pickle

# Update the BlueprintStrategy class to incorporate opponent identity

class BlueprintStrategy:
    """
    Stores and provides access to a pre-computed strategy blueprint for the game.
    Now incorporates opponent identity, efficient caching, and similarity-based lookups.
    """
    def __init__(self, policy_net=None, belief_model=None):
        """Initialize the blueprint strategy either empty or with networks."""
        self.policy_net = policy_net
        self.belief_model = belief_model
        
        # Storage for state-action mappings with opponent identity
        self.strategy_map = defaultdict(lambda: np.zeros(7))  # For 7 actions
        self.value_map = {}
        self.visit_counts = defaultdict(int)
        self.last_update_time = defaultdict(float)  # Track when each state was last updated
        
        # State metadata for similarity search
        self.state_features = {}  # Maps state_key to feature vector
        self.state_embeddings = {}  # For similarity search
        
        # CFR-related data
        self.average_strategy = defaultdict(lambda: np.zeros(7))
        self.cumulative_regrets = defaultdict(lambda: np.zeros(7))
        
        # Cache for faster query results
        self._query_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_max_size = 10000  # Maximum cache size
        
        # Nearest neighbor model for similar state lookup
        self._nn_model = None
        self._nn_keys = []
        self._nn_features = []
        self._nn_rebuild_threshold = 100  # Rebuild after this many new states
        self._nn_new_states = 0
        
        # Track changes for incremental saving
        self._dirty_keys = set()  # Keys that have been modified since last save
        self._last_save_time = 0
        
        # Compression level for saving
        self.compression_level = 5  # 1-9, higher is more compression but slower
    
    def state_to_key(self, public_obs, beliefs=None, opponent_id=None):
        """
        Convert public observation and belief state to a unique key for storage.
        Optimized for computational efficiency with tensor support.
        
        Args:
            public_obs: Public observation vector (numpy array or tensor)
            beliefs: Current belief state (numpy array, tensor, or None)
            opponent_id: The opponent's identity (optional)
        
        Returns:
            String key that uniquely identifies this public state
        """
        # Handle tensor inputs efficiently
        if isinstance(public_obs, torch.Tensor):
            # Convert tensor to numpy for consistent hashing
            public_obs_np = public_obs.detach().cpu().numpy()
        else:
            public_obs_np = public_obs
            
        # Round values to reduce precision issues and hash collisions
        # Use 5 decimal places for sufficient precision without excessive uniqueness
        public_obs_rounded = np.round(public_obs_np, 5)
        
        # Normalize beliefs for consistent hashing
        belief_str = ""
        if beliefs is not None:
            if isinstance(beliefs, torch.Tensor):
                beliefs_np = beliefs.detach().cpu().numpy()
            else:
                beliefs_np = beliefs
                
            # Round beliefs for consistency
            beliefs_rounded = np.round(beliefs_np, 5)
            
            # Create compact string representation
            belief_str = hashlib.md5(beliefs_rounded.tobytes()).hexdigest()
        
        # Create combined key
        key_parts = [str(public_obs_rounded.tobytes()), belief_str]
        
        # Add opponent identity if provided
        if opponent_id is not None:
            key_parts.append(str(opponent_id))
        
        # Use MD5 hash for consistent, compact representation
        combined_str = "_".join(key_parts)
        return hashlib.md5(combined_str.encode('utf-8')).hexdigest()
    
    def extract_state_features(self, public_obs, beliefs):
        """
        Extract a feature vector from state for similarity search.
        
        Args:
            public_obs: Public observation
            beliefs: Belief state
            
        Returns:
            Feature vector representing this state
        """
        # Convert inputs to numpy arrays if they're tensors
        if isinstance(public_obs, torch.Tensor):
            public_obs_np = public_obs.detach().cpu().numpy()
        else:
            public_obs_np = public_obs
            
        if isinstance(beliefs, torch.Tensor):
            beliefs_np = beliefs.detach().cpu().numpy()
        else:
            beliefs_np = beliefs
        
        # Extract key features for similarity comparison
        # Flatten and concatenate public observation and beliefs
        if beliefs_np is not None:
            # Flatten multi-dimensional beliefs
            if beliefs_np.ndim > 2:
                beliefs_flat = beliefs_np.reshape(-1)
            else:
                beliefs_flat = beliefs_np.flatten()
                
            # Concatenate features with appropriate weighting
            # Public obs is more important for matching
            features = np.concatenate([
                public_obs_np.flatten() * 2.0,  # Weight public observation higher
                beliefs_flat
            ])
        else:
            features = public_obs_np.flatten()
            
        return features
    
    def find_similar_states(self, features, max_states=3, similarity_threshold=0.9):
        """
        Find similar states using nearest neighbor search.
        
        Args:
            features: Feature vector for the query state
            max_states: Maximum number of similar states to return
            similarity_threshold: Minimum similarity score to consider
            
        Returns:
            List of (state_key, similarity) tuples for similar states
        """
        # Check if we have enough states for meaningful search
        if len(self._nn_keys) < 5:
            return []
            
        # Rebuild model if needed
        if self._nn_model is None:
            self._rebuild_nn_model()
            
        # Ensure features is the right shape
        features = features.reshape(1, -1)
        
        # Find nearest neighbors
        distances, indices = self._nn_model.kneighbors(features, n_neighbors=min(max_states, len(self._nn_keys)))
        
        # Convert distances to similarities (1 - normalized distance)
        max_dist = np.max(distances) if distances.size > 0 else 1.0
        similarities = 1.0 - (distances / max(max_dist, 1e-8))
        
        # Filter by similarity threshold
        similar_states = []
        for i, idx in enumerate(indices[0]):
            if similarities[0][i] >= similarity_threshold:
                similar_states.append((self._nn_keys[idx], similarities[0][i]))
                
        return similar_states
    
    def _rebuild_nn_model(self):
        """Rebuild the nearest neighbor model with current state data."""
        if not self._nn_features:
            return
            
        # Build model using scikit-learn
        self._nn_model = NearestNeighbors(n_neighbors=min(5, len(self._nn_features)), 
                                         algorithm='ball_tree')
        self._nn_model.fit(np.array(self._nn_features))
        self._nn_new_states = 0
    
    def update_from_search(self, public_obs, beliefs, cfr_strategy, value, regrets, visits=1, opponent_id=None):
        """
        Update blueprint from search results, with selective updating based on visits
        and decay for old entries.
        
        Args:
            public_obs: Public observation
            beliefs: Belief state
            cfr_strategy: CFR strategy (average)
            value: Value estimate
            regrets: Counterfactual regrets
            visits: Visit count for weighting (higher = more important state)
            opponent_id: The opponent's identity (optional)
        """
        # Create key for this state
        key = self.state_to_key(public_obs, beliefs, opponent_id)
        
        # Track this key as modified for incremental saving
        self._dirty_keys.add(key)
        
        # Compute feature vector for similarity search
        if key not in self.state_features:
            features = self.extract_state_features(public_obs, beliefs)
            self.state_features[key] = features
            
            # Add to nearest neighbor data
            self._nn_keys.append(key)
            self._nn_features.append(features)
            self._nn_new_states += 1
            
            # Rebuild model if we've added enough new states
            if self._nn_new_states >= self._nn_rebuild_threshold:
                self._rebuild_nn_model()
        
        # Apply selective updating based on visit counts
        current_visits = self.visit_counts[key]
        total_visits = current_visits + visits
        
        # Track update time for decay
        current_time = time.time()
        self.last_update_time[key] = current_time
        
        # Calculate decay factor for old data
        # Newer updates have more weight than older ones
        if current_visits > 0:
            # Compute time-based decay factor
            time_since_last_update = current_time - self.last_update_time.get(key, 0)
            time_decay = max(0.5, np.exp(-0.1 * time_since_last_update))
            
            # Blend factor based on visits and time decay
            # More visits = slower update (more stable)
            alpha = min(0.3, visits / (total_visits + 10))
            alpha = max(alpha, 0.01)  # Ensure some minimum update
            
            # Apply decay to the update
            self.strategy_map[key] = (1 - alpha) * self.strategy_map[key] + alpha * cfr_strategy
            self.value_map[key] = (1 - alpha) * self.value_map.get(key, 0) + alpha * value
            
            # Update CFR data
            self.average_strategy[key] = (1 - alpha) * self.average_strategy[key] + alpha * cfr_strategy
            self.cumulative_regrets[key] = (1 - alpha) * self.cumulative_regrets[key] + alpha * regrets
        else:
            # First visit, use values directly
            self.strategy_map[key] = cfr_strategy
            self.value_map[key] = value
            self.average_strategy[key] = cfr_strategy
            self.cumulative_regrets[key] = regrets
        
        # Update visit count
        self.visit_counts[key] = total_visits
        
        # Clear any cached queries that might depend on this state
        if key in self._query_cache:
            del self._query_cache[key]
    
    def merge_similar_states(self, state_key, public_obs, beliefs, action_mask=None):
        """
        Merge similar states for more robust lookups.
        
        Args:
            state_key: Key for the query state
            public_obs: Public observation for the query state
            beliefs: Beliefs for the query state
            action_mask: Optional action mask
            
        Returns:
            Merged strategy and value
        """
        # Extract features for similarity search
        if state_key not in self.state_features and public_obs is not None:
            features = self.extract_state_features(public_obs, beliefs)
        else:
            features = self.state_features.get(state_key)
            
        if features is None:
            return None, None
            
        # Find similar states
        similar_states = self.find_similar_states(features, max_states=3, similarity_threshold=0.85)
        
        if not similar_states:
            return None, None
            
        # Merge strategies and values based on similarity
        merged_strategy = np.zeros(7)
        merged_value = 0.0
        total_weight = 0.0
        
        for similar_key, similarity in similar_states:
            if similar_key in self.strategy_map:
                # Weight by similarity and visit count
                weight = similarity * np.sqrt(self.visit_counts[similar_key])
                
                # Apply action mask if provided
                if action_mask is not None:
                    masked_strategy = self.strategy_map[similar_key] * action_mask
                    if np.sum(masked_strategy) > 0:
                        masked_strategy = masked_strategy / np.sum(masked_strategy)
                    else:
                        continue  # Skip if no valid actions after masking
                else:
                    masked_strategy = self.strategy_map[similar_key]
                
                # Accumulated weighted average
                merged_strategy += weight * masked_strategy
                merged_value += weight * self.value_map.get(similar_key, 0.0)
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            merged_strategy /= total_weight
            merged_value /= total_weight
            return merged_strategy, merged_value
            
        return None, None
    
    def update_strategy(self, public_obs, beliefs, strategy, value, visits=1, opponent_id=None):
        """
        Update the blueprint with a new strategy for a given state.
        Now incorporates opponent identity.
        
        Args:
            public_obs: Public observation vector
            beliefs: Current belief state
            strategy: Strategy (probability distribution over actions)
            value: Value estimate for this state
            visits: Number of visits to this state
            opponent_id: The opponent's identity (optional)
        """
        key = self.state_to_key(public_obs, beliefs, opponent_id)
        
        # Incremental update weighted by visits
        current_visits = self.visit_counts[key]
        total_visits = current_visits + visits
        
        if current_visits > 0:
            # Update with weighted average
            self.strategy_map[key] = (
                (current_visits / total_visits) * self.strategy_map[key] +
                (visits / total_visits) * strategy
            )
            self.value_map[key] = (
                (current_visits / total_visits) * self.value_map.get(key, 0) +
                (visits / total_visits) * value
            )
        else:
            # First visit
            self.strategy_map[key] = strategy
            self.value_map[key] = value
        
        self.visit_counts[key] = total_visits
    
    def query(self, public_obs, beliefs=None, action_mask=None, opponent_id=None):
        """
        Query the blueprint for a strategy in the given state.
        Optimized with caching and fallback strategies for cache misses.
        
        Args:
            public_obs: Public observation vector
            beliefs: Current belief state (optional)
            action_mask: Mask of valid actions (optional)
            opponent_id: The opponent's identity (optional)
        
        Returns:
            Tuple of (strategy, value)
        """
        # Create cache key for fast lookup
        cache_key = None
        if public_obs is not None:
            try:
                # Generate cache key from inputs
                if isinstance(public_obs, torch.Tensor):
                    obs_hash = hash(public_obs.detach().cpu().numpy().tobytes())
                else:
                    obs_hash = hash(str(public_obs))
                    
                belief_hash = 0
                if beliefs is not None:
                    if isinstance(beliefs, torch.Tensor):
                        belief_hash = hash(beliefs.detach().cpu().numpy().tobytes())
                    else:
                        belief_hash = hash(str(beliefs))
                
                mask_hash = 0
                if action_mask is not None:
                    if isinstance(action_mask, torch.Tensor):
                        mask_hash = hash(action_mask.detach().cpu().numpy().tobytes())
                    else:
                        mask_hash = hash(str(action_mask))
                
                cache_key = hash((obs_hash, belief_hash, mask_hash, str(opponent_id)))
                
                # Check cache for hit
                if cache_key in self._query_cache:
                    self._cache_hits += 1
                    return self._query_cache[cache_key]
                    
                self._cache_misses += 1
            except:
                # If there's any error in cache key generation, continue without caching
                pass
        
        # Try opponent-specific key first
        state_key = None
        if opponent_id is not None:
            opponent_key = self.state_to_key(public_obs, beliefs, opponent_id)
            if opponent_key in self.strategy_map:
                strategy = self.strategy_map[opponent_key]
                value = self.value_map.get(opponent_key, 0.0)
                
                # Apply action mask if provided
                if action_mask is not None:
                    masked_strategy = strategy * action_mask
                    if np.sum(masked_strategy) > 0:
                        masked_strategy = masked_strategy / np.sum(masked_strategy)
                    else:
                        # Fallback to uniform over valid actions
                        valid_actions = np.where(action_mask)[0]
                        masked_strategy = np.zeros_like(strategy)
                        if len(valid_actions) > 0:  # Safety check
                            masked_strategy[valid_actions] = 1.0 / len(valid_actions)
                    result = (masked_strategy, value)
                else:
                    result = (strategy, value)
                
                # Cache result
                if cache_key is not None:
                    self._update_cache(cache_key, result)
                return result
        
        # Fall back to general state key without opponent identity
        generic_key = self.state_to_key(public_obs, beliefs)
        
        # If state exists in our map
        if generic_key in self.strategy_map:
            strategy = self.strategy_map[generic_key]
            value = self.value_map.get(generic_key, 0.0)
            
            # Apply action mask if provided
            if action_mask is not None:
                masked_strategy = strategy * action_mask
                if np.sum(masked_strategy) > 0:
                    masked_strategy = masked_strategy / np.sum(masked_strategy)
                else:
                    # Fallback to uniform over valid actions
                    valid_actions = np.where(action_mask)[0]
                    masked_strategy = np.zeros_like(strategy)
                    if len(valid_actions) > 0:  # Safety check
                        masked_strategy[valid_actions] = 1.0 / len(valid_actions)
                result = (masked_strategy, value)
            else:
                result = (strategy, value)
            
            # Cache result
            if cache_key is not None:
                self._update_cache(cache_key, result)
            return result
        
        # Try similarity-based lookup for missing states
        merged_strategy, merged_value = self.merge_similar_states(
            generic_key, public_obs, beliefs, action_mask)
            
        if merged_strategy is not None:
            result = (merged_strategy, merged_value)
            # Cache result
            if cache_key is not None:
                self._update_cache(cache_key, result)
            return result
        
        # If we have neural networks but state not in map, use them
        if self.policy_net is not None and beliefs is not None:
            # Convert to tensors
            device = next(self.policy_net.parameters()).device
            if not isinstance(public_obs, torch.Tensor):
                public_obs_tensor = torch.FloatTensor(public_obs).unsqueeze(0).to(device)
            else:
                public_obs_tensor = public_obs.to(device)
                
            if not isinstance(beliefs, torch.Tensor):
                beliefs_tensor = torch.FloatTensor(beliefs).unsqueeze(0).to(device)
            else:
                beliefs_tensor = beliefs.to(device)
            
            # Get public policy
            with torch.no_grad():
                probs, value, _ = self.policy_net.public_policy(public_obs_tensor, beliefs_tensor)
                strategy = probs.squeeze(0).cpu().numpy()
                value = value.item()
            
            # Apply action mask if provided
            if action_mask is not None:
                masked_strategy = strategy * action_mask
                if np.sum(masked_strategy) > 0:
                    masked_strategy = masked_strategy / np.sum(masked_strategy)
                else:
                    # Fallback to uniform over valid actions
                    valid_actions = np.where(action_mask)[0]
                    masked_strategy = np.zeros_like(strategy)
                    if len(valid_actions) > 0:  # Safety check
                        masked_strategy[valid_actions] = 1.0 / len(valid_actions)
                result = (masked_strategy, value)
            else:
                result = (strategy, value)
            
            # Cache result
            if cache_key is not None:
                self._update_cache(cache_key, result)
            return result
        
        # Fallback to uniform random if no data available
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            strategy = np.zeros(7)
            if len(valid_actions) > 0:  # Safety check
                strategy[valid_actions] = 1.0 / len(valid_actions)
            result = (strategy, 0.0)
        else:
            result = (np.ones(7) / 7, 0.0)
            
        # Cache result
        if cache_key is not None:
            self._update_cache(cache_key, result)
        return result
    
    def _update_cache(self, cache_key, result):
        """Update query cache with LRU policy."""
        # Add to cache
        self._query_cache[cache_key] = result
        
        # Implement LRU cache by clearing oldest entries when cache gets too large
        if len(self._query_cache) > self._cache_max_size:
            # Remove 20% of the cache when full
            num_to_remove = self._cache_max_size // 5
            keys_to_remove = list(self._query_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self._query_cache[key]

    def query_with_fallback(self, public_obs, beliefs, action_mask, opponent_id=None):
        """
        Extended query method with multiple fallback strategies for robustness.
        
        Args:
            public_obs: Public observation vector
            beliefs: Current belief state
            action_mask: Mask of valid actions
            opponent_id: The opponent's identity (optional)
            
        Returns:
            Tuple of (strategy, value, source)
        """
        # Try direct key lookup first
        state_key = self.state_to_key(public_obs, beliefs, opponent_id)
        if state_key in self.strategy_map:
            strategy = self.strategy_map[state_key]
            value = self.value_map.get(state_key, 0.0)
            
            # Apply action mask
            masked_strategy = self._apply_mask(strategy, action_mask)
            return masked_strategy, value, "direct"
        
        # Try similar state lookup (relaxed matching)
        merged_strategy, merged_value = self.merge_similar_states(
            state_key, public_obs, beliefs, action_mask)
            
        if merged_strategy is not None:
            return merged_strategy, merged_value, "similar"
        
        # Try policy network fallback
        if self.policy_net is not None:
            strategy, value = self._query_network(public_obs, beliefs, action_mask)
            return strategy, value, "network"
        
        # Ultimate fallback: uniform random over valid actions
        valid_actions = np.where(action_mask)[0]
        strategy = np.zeros(7)
        if len(valid_actions) > 0:
            strategy[valid_actions] = 1.0 / len(valid_actions)
        
        return strategy, 0.0, "uniform"
    
    def _apply_mask(self, strategy, action_mask):
        """Helper method to apply action mask to strategy."""
        masked_strategy = strategy * action_mask
        if np.sum(masked_strategy) > 0:
            return masked_strategy / np.sum(masked_strategy)
        
        # Fallback to uniform if mask zeros out the strategy
        valid_actions = np.where(action_mask)[0]
        uniform_strategy = np.zeros_like(strategy)
        if len(valid_actions) > 0:
            uniform_strategy[valid_actions] = 1.0 / len(valid_actions)
        return uniform_strategy
    
    def _query_network(self, public_obs, beliefs, action_mask):
        """Helper method to query policy network."""
        device = next(self.policy_net.parameters()).device
        
        # Convert to tensors
        if not isinstance(public_obs, torch.Tensor):
            public_obs_tensor = torch.FloatTensor(public_obs).unsqueeze(0).to(device)
        else:
            public_obs_tensor = public_obs.to(device)
            
        if not isinstance(beliefs, torch.Tensor):
            beliefs_tensor = torch.FloatTensor(beliefs).unsqueeze(0).to(device)
        else:
            beliefs_tensor = beliefs.to(device)
        
        # Get policy from network
        with torch.no_grad():
            probs, value, _ = self.policy_net.public_policy(public_obs_tensor, beliefs_tensor)
            strategy = probs.squeeze(0).cpu().numpy()
            value = value.item()
        
        # Apply action mask
        return self._apply_mask(strategy, action_mask), value
    
    def update_from_search(self, public_obs, beliefs, cfr_strategy, value, regrets, visits=1, opponent_id=None):
        """
        Update blueprint from search results, including CFR information.
        Now incorporates opponent identity.
        
        Args:
            public_obs: Public observation
            beliefs: Belief state
            cfr_strategy: CFR strategy (average)
            value: Value estimate
            regrets: Counterfactual regrets
            visits: Visit count for weighting
            opponent_id: The opponent's identity (optional)
        """
        key = self.state_to_key(public_obs, beliefs, opponent_id)
        
        # Update strategy and value
        self.update_strategy(public_obs, beliefs, cfr_strategy, value, visits, opponent_id)
        
        # Update CFR data
        self.average_strategy[key] = cfr_strategy
        
        # Update cumulative regrets (weighted by visits)
        current_regrets = self.cumulative_regrets[key]
        self.cumulative_regrets[key] = current_regrets + regrets * visits
    
    def save(self, filepath, incremental=True):
        """
        Save the blueprint strategy to disk with compression and incremental saving.

        Args:
            filepath: Path to save the blueprint file
            incremental: Whether to use incremental saving (only modified states)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Check if we can perform an incremental save:
        can_incremental = incremental and os.path.exists(filepath) and self._last_save_time > 0

        if can_incremental:
            # Try to load the existing data and update only modified keys
            try:
                with gzip.open(filepath, 'rb') as f:
                    existing_data = pickle.load(f)

                # Get existing maps, defaulting to empty dictionaries if not found
                strategy_map = existing_data.get('strategy_map', {})
                value_map = existing_data.get('value_map', {})
                visit_counts = existing_data.get('visit_counts', {})
                average_strategy = existing_data.get('average_strategy', {})
                cumulative_regrets = existing_data.get('cumulative_regrets', {})
                state_features = existing_data.get('state_features', {})

                # Update only the modified keys
                for key in self._dirty_keys:
                    if key in self.strategy_map:
                        strategy_map[key] = self.strategy_map[key]
                        value_map[key] = self.value_map.get(key, 0.0)
                        visit_counts[key] = self.visit_counts[key]
                        average_strategy[key] = self.average_strategy[key]
                        cumulative_regrets[key] = self.cumulative_regrets[key]
                        if key in self.state_features:
                            state_features[key] = self.state_features[key]

                # Prepare the data to save
                data = {
                    'strategy_map': strategy_map,
                    'value_map': value_map,
                    'visit_counts': visit_counts,
                    'average_strategy': average_strategy,
                    'cumulative_regrets': cumulative_regrets,
                    'state_features': state_features,
                    'timestamp': time.time(),
                    'num_states': len(strategy_map),
                    'modified_keys': list(self._dirty_keys)
                }

                # Save metadata separately for quick loading
                metadata = {
                    'timestamp': time.time(),
                    'num_states': len(strategy_map),
                    'num_modified': len(self._dirty_keys),
                    'cache_hits': self._cache_hits,
                    'cache_misses': self._cache_misses
                }
                metadata_path = filepath.replace('.pkl', '_meta.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
            except Exception as e:
                print(f"Error during incremental save: {e}")
                # If incremental saving fails, fall back to full save
                can_incremental = False

        if not can_incremental:
            # Perform a full save
            data = {
                'strategy_map': dict(self.strategy_map),
                'value_map': self.value_map,
                'visit_counts': dict(self.visit_counts),
                'average_strategy': dict(self.average_strategy),
                'cumulative_regrets': dict(self.cumulative_regrets),
                'state_features': self.state_features,
                'timestamp': time.time(),
                'num_states': len(self.strategy_map)
            }

            # Save metadata separately for quick loading
            metadata = {
                'timestamp': time.time(),
                'num_states': len(self.strategy_map),
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses
            }
            metadata_path = filepath.replace('.pkl', '_meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

        # Save using compression
        with gzip.open(filepath, 'wb', compresslevel=self.compression_level) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update tracking
        self._last_save_time = time.time()
        self._dirty_keys = set()  # Clear dirty keys after saving
    
    @classmethod
    def load(cls, filepath, policy_net=None, belief_model=None):
        """
        Load a blueprint strategy from disk with compression support.
        
        Args:
            filepath: Path to the blueprint file
            policy_net: Optional policy network
            belief_model: Optional belief model
            
        Returns:
            Loaded BlueprintStrategy instance
        """
        blueprint = cls(policy_net=policy_net, belief_model=belief_model)
        
        # First check if metadata exists for quick inspection
        metadata_path = filepath.replace('.pkl', '_meta.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loading blueprint with {metadata.get('num_states', 'unknown')} states, " 
                     f"last modified: {time.ctime(metadata.get('timestamp', 0))}")
            except:
                pass
        
        # Load the full blueprint data
        try:
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
        except:
            # Try without compression as fallback
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load blueprint: {e}")
        
        # Initialize from loaded data
        blueprint.strategy_map = defaultdict(lambda: np.zeros(7), data['strategy_map'])
        blueprint.value_map = data['value_map']
        blueprint.visit_counts = defaultdict(int, data['visit_counts'])
        blueprint.average_strategy = defaultdict(lambda: np.zeros(7), data['average_strategy'])
        blueprint.cumulative_regrets = defaultdict(lambda: np.zeros(7), data['cumulative_regrets'])
        
        # Load state features for similarity search if available
        if 'state_features' in data:
            blueprint.state_features = data['state_features']
            
            # Set up nearest neighbor data structures
            blueprint._nn_keys = list(blueprint.state_features.keys())
            blueprint._nn_features = [blueprint.state_features[k] for k in blueprint._nn_keys]
            
            # Build nearest neighbor model if enough states
            if len(blueprint._nn_keys) >= 10:
                blueprint._rebuild_nn_model()
        
        # Initialize tracking for incremental saves
        blueprint._last_save_time = time.time()
        blueprint._dirty_keys = set()
        
        return blueprint
    
    def prune_low_visit_states(self, min_visits=5, max_states=None):
        """
        Prune states with low visit counts to reduce memory usage.
        
        Args:
            min_visits: Minimum visit count to keep a state
            max_states: Maximum number of states to keep (prioritize by visits)
            
        Returns:
            Number of states removed
        """
        # Get states sorted by visit count
        sorted_states = sorted(self.visit_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Determine threshold for pruning
        if max_states is not None and len(sorted_states) > max_states:
            keep_count = max_states
        else:
            keep_count = sum(1 for _, visits in sorted_states if visits >= min_visits)
            
        # No pruning needed
        if keep_count >= len(sorted_states):
            return 0
            
        # Get keys to keep
        keys_to_keep = set(key for key, _ in sorted_states[:keep_count])
        
        # Find keys to remove
        keys_to_remove = set(self.strategy_map.keys()) - keys_to_keep
        
        # Remove data for pruned states
        for key in keys_to_remove:
            if key in self.strategy_map:
                del self.strategy_map[key]
            if key in self.value_map:
                del self.value_map[key]
            if key in self.visit_counts:
                del self.visit_counts[key]
            if key in self.average_strategy:
                del self.average_strategy[key]
            if key in self.cumulative_regrets:
                del self.cumulative_regrets[key]
            if key in self.state_features:
                del self.state_features[key]
        
        # Reset nearest neighbor model
        if keys_to_remove:
            self._nn_keys = list(self.state_features.keys())
            self._nn_features = [self.state_features[k] for k in self._nn_keys]
            self._nn_model = None
            self._nn_new_states = 0
        
        # Mark all remaining states as dirty for next save
        self._dirty_keys = set(self.strategy_map.keys())
        
        return len(keys_to_remove)