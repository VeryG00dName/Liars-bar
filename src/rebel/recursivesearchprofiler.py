import cProfile
import pstats
import time
import functools
import io
from collections import defaultdict
import numpy as np

class RecursiveSearchProfiler:
    """
    Profiling wrapper for the RecursiveSearchAgent.
    Provides both function-level and detailed timing metrics.
    """
    def __init__(self, agent, enabled=True):
        self.agent = agent
        self.enabled = enabled
        self.function_timers = defaultdict(list)
        self.section_timers = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.profiler = None
        
        # Stats collection
        self.simulation_times = []
        self.belief_update_times = []
        self.mcts_search_times = []
        self.nn_inference_times = []
        
        # Apply decorator to key methods if enabled
        if enabled:
            self._patch_agent_methods()
    
    def _patch_agent_methods(self):
        """Apply timing decorators to key agent methods."""
        methods_to_profile = [
            ('mcts_search', 'MCTS Search'),
            ('_simulate', 'Simulation'),
            ('update_beliefs', 'Belief Update'),
            ('compute_cfr_strategy', 'CFR Strategy'),
            ('get_transformer_memory_embeddings', 'Transformer Memory'),
            ('_compute_counterfactual_beliefs', 'Counterfactual Beliefs'),
            ('play_turn', 'Play Turn'),
            ('_apply_progressive_pruning', 'Progressive Pruning'),
            ('check_early_termination', 'Early Termination'),
        ]
        
        for method_name, display_name in methods_to_profile:
            if hasattr(self.agent, method_name):
                original_method = getattr(self.agent, method_name)
                setattr(self.agent, method_name, 
                        self._time_function(original_method, display_name))
    
    def _time_function(self, func, name):
        """Decorator to time function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            self.call_counts[name] += 1
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            self.function_timers[name].append(elapsed)
            
            # Special handling for key methods to track stats
            if name == 'MCTS Search':
                self.mcts_search_times.append(elapsed)
            elif name == 'Simulation':
                self.simulation_times.append(elapsed)
            elif name == 'Belief Update':
                self.belief_update_times.append(elapsed)
            
            return result
        return wrapper
    
    def start_section(self, name):
        """Start timing a code section manually."""
        if not self.enabled:
            return None
        
        section_id = f"{name}_{id(name)}"
        self.call_counts[name] += 1
        return (name, time.time())
    
    def end_section(self, section_data):
        """End timing a code section manually."""
        if not self.enabled or section_data is None:
            return
        
        name, start_time = section_data
        elapsed = time.time() - start_time
        self.section_timers[name].append(elapsed)
        
        # Special case for neural network inference
        if 'neural network' in name.lower():
            self.nn_inference_times.append(elapsed)
    
    def start_profiler(self):
        """Start the cProfile profiler."""
        if self.enabled:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
    
    def stop_profiler(self, output_file='agent_profile.prof'):
        """Stop the profiler and save results."""
        if self.enabled and self.profiler:
            self.profiler.disable()
            self.profiler.dump_stats(output_file)
            # Also print summary to console
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            print(s.getvalue())
    
    def get_timer_stats(self):
        """Get statistics for all timed functions."""
        if not self.enabled:
            return {}
        
        stats = {}
        # Process function timers
        for name, times in self.function_timers.items():
            if not times:
                continue
            stats[name] = {
                'total_time': sum(times),
                'mean_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'count': self.call_counts[name],
                'percent_of_total': 0.0  # Will fill in later
            }
        
        # Process section timers
        for name, times in self.section_timers.items():
            if not times:
                continue
            stats[f"SECTION_{name}"] = {
                'total_time': sum(times),
                'mean_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'count': self.call_counts[name],
                'percent_of_total': 0.0  # Will fill in later
            }
        
        # Calculate total time and percentages
        if 'Play Turn' in stats:
            total_time = stats['Play Turn']['total_time']
            for name in stats:
                if total_time > 0:
                    stats[name]['percent_of_total'] = (stats[name]['total_time'] / total_time) * 100
        
        return stats
    
    def print_summary(self):
        """Print a summary of profiling results."""
        if not self.enabled:
            print("Profiling disabled.")
            return
        
        stats = self.get_timer_stats()
        
        print("\n===== PROFILING SUMMARY =====")
        print(f"Total methods profiled: {len(stats)}")
        
        # Sort by total time
        sorted_stats = sorted(
            [(name, data) for name, data in stats.items()],
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        print("\nTop time consumers:")
        for i, (name, data) in enumerate(sorted_stats[:10]):
            print(f"{i+1}. {name}")
            print(f"   - Total: {data['total_time']:.4f}s ({data['percent_of_total']:.1f}%)")
            print(f"   - Mean: {data['mean_time']*1000:.2f}ms over {data['count']} calls")
        
        print("\nPerformance hotspots:")
        if self.simulation_times:
            print(f"Simulations: Avg {np.mean(self.simulation_times)*1000:.2f}ms per call")
            print(f"  - Total simulations: {len(self.simulation_times)}")
        
        if self.belief_update_times:
            print(f"Belief updates: Avg {np.mean(self.belief_update_times)*1000:.2f}ms per update")
            print(f"  - Total updates: {len(self.belief_update_times)}")
        
        if self.nn_inference_times:
            print(f"Neural network inference: Avg {np.mean(self.nn_inference_times)*1000:.2f}ms per call")
            print(f"  - Total inferences: {len(self.nn_inference_times)}")
        
        if self.mcts_search_times:
            print(f"MCTS searches: Avg {np.mean(self.mcts_search_times)*1000:.2f}ms per search")
            print(f"  - Total searches: {len(self.mcts_search_times)}")
            if 'num_simulations' in dir(self.agent):
                print(f"  - Simulations per search: {self.agent.num_simulations}")
        
        print("\n============================")


# Usage example for the RecursiveSearchAgent class
def add_profiling_to_agent(agent, enabled=True):
    """
    Add profiling capabilities to a RecursiveSearchAgent instance.
    
    Args:
        agent: The RecursiveSearchAgent instance to profile
        enabled: Whether profiling should be enabled
        
    Returns:
        The profiler instance
    """
    profiler = RecursiveSearchProfiler(agent, enabled=enabled)
    
    # Add timing to the policy network forward pass
    if hasattr(agent, 'policy_net') and hasattr(agent.policy_net, 'forward'):
        original_forward = agent.policy_net.forward
        
        @functools.wraps(original_forward)
        def timed_forward(*args, **kwargs):
            section = profiler.start_section("Policy Network Inference")
            result = original_forward(*args, **kwargs)
            profiler.end_section(section)
            return result
            
        agent.policy_net.forward = timed_forward
    
    # Add timing to the belief model
    if hasattr(agent, 'belief_model') and hasattr(agent.belief_model, 'forward'):
        original_belief_forward = agent.belief_model.forward
        
        @functools.wraps(original_belief_forward)
        def timed_belief_forward(*args, **kwargs):
            section = profiler.start_section("Belief Model Inference")
            result = original_belief_forward(*args, **kwargs)
            profiler.end_section(section)
            return result
            
        agent.belief_model.forward = timed_belief_forward
    
    # Add timing to the value network
    if hasattr(agent, 'value_net') and hasattr(agent.value_net, 'forward'):
        original_value_forward = agent.value_net.forward
        
        @functools.wraps(original_value_forward)
        def timed_value_forward(*args, **kwargs):
            section = profiler.start_section("Value Network Inference")
            result = original_value_forward(*args, **kwargs)
            profiler.end_section(section)
            return result
            
        agent.value_net.forward = timed_value_forward
    
    # Enhance mcts_search with detailed sections
    if hasattr(agent, 'mcts_search'):
        original_mcts = agent.mcts_search
        
        @functools.wraps(original_mcts)
        def enhanced_mcts_search(observation, action_mask):
            # Overall timing is already handled by our decorator
            
            # Start profiler if this is the top-level call
            is_root = not hasattr(agent, '_in_mcts_search') or not agent._in_mcts_search
            if is_root:
                agent._in_mcts_search = True
                profiler.start_profiler()
            
            # Track key sections of MCTS
            belief_section = profiler.start_section("MCTS Belief Update")
            agent.update_beliefs(observation, action_mask)
            profiler.end_section(belief_section)
            
            policy_section = profiler.start_section("MCTS Policy Priors")
            # The original function will execute the policy network call
            profiler.end_section(policy_section)
            
            sim_section = profiler.start_section("MCTS Simulations")
            # The original function will execute the simulations
            profiler.end_section(sim_section)
            
            result = original_mcts(observation, action_mask)
            
            if is_root:
                agent._in_mcts_search = False
                profiler.stop_profiler()
            
            return result
            
        agent.mcts_search = enhanced_mcts_search
    
    # Add the profiler to the agent for easy access
    agent.profiler = profiler
    
    return profiler