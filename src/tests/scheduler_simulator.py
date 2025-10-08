import argparse
import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# --- Constants & Bucket Definitions ---
# These can be modified to test different bucketing strategies
PAD_BUCKET_BOUNDARIES = [64, 128, 256, 384, 480]

def get_bucket(seq_len: int) -> int:
    """Assigns a sequence length to a bucket boundary."""
    for boundary in PAD_BUCKET_BOUNDARIES:
        if seq_len <= boundary:
            return boundary
    return seq_len # Should ideally align with a max bucket

class CostModel:
    """Loads performance data and provides a cost function via interpolation."""
    def __init__(self, performance_csv: Path):
        print(f"Loading performance data from: {performance_csv}")
        df = pd.read_csv(performance_csv)
        
        # Ensure correct data types
        df['batch_size'] = pd.to_numeric(df['batch_size'], errors='coerce')
        df['seq_len'] = pd.to_numeric(df['seq_len'], errors='coerce')
        df['mean_ms'] = pd.to_numeric(df['mean_ms'], errors='coerce')
        df.dropna(inplace=True)

        self.points = df[['batch_size', 'seq_len']].values
        self.values = df['mean_ms'].values
        print(f"Cost model initialized with {len(self.values)} data points.")

    def get_cost(self, batch_size: int, seq_len: int) -> float:
        """
        Estimates the execution time for a batch of a given size and sequence length.
        Uses 2D interpolation to estimate costs for configurations not in the data.
        """
        if batch_size <= 0:
            return 0.0
        
        # Clamp to the known boundaries of our data to avoid extrapolation errors
        min_bs, max_bs = self.points[:, 0].min(), self.points[:, 0].max()
        min_sl, max_sl = self.points[:, 1].min(), self.points[:, 1].max()
        
        query_bs = np.clip(batch_size, min_bs, max_bs)
        query_sl = np.clip(seq_len, min_sl, max_sl)
        
        # griddata is perfect for interpolating on a scattered 2D grid like ours
        estimated_cost = griddata(self.points, self.values, (query_bs, query_sl), method='linear')
        
        # If interpolation fails (e.g., query is outside convex hull), fallback to nearest
        if np.isnan(estimated_cost):
            estimated_cost = griddata(self.points, self.values, (query_bs, query_sl), method='nearest')
            
        return float(estimated_cost)

# --- Scenario Generation ---
def generate_scenario(
    num_requests: int, num_policies: int, avg_seq_len: int, seq_len_std: int
) -> List[Tuple[int, int]]:
    """Generates a realistic list of pending requests."""
    policies = list(range(num_policies))
    # Skew the distribution so some policies are more common than others
    policy_weights = [(1 / (i + 1)) for i in range(num_policies)]
    
    requests = []
    for _ in range(num_requests):
        policy_id = random.choices(policies, weights=policy_weights, k=1)[0]
        seq_len = max(1, int(random.gauss(avg_seq_len, seq_len_std)))
        requests.append((policy_id, seq_len))
        
    return requests

# --- Scheduling Strategies ---
def schedule_naive_greedy(
    scenario: List[Tuple[int, int]], cost_model: CostModel
) -> float:
    """Strategy 1: Batch all requests for a policy together, no matter the seq_len."""
    groups = defaultdict(list)
    for policy_id, seq_len in scenario:
        groups[policy_id].append(seq_len)
        
    total_cost = 0.0
    for policy_id, seq_lens in groups.items():
        batch_size = len(seq_lens)
        padded_seq_len = max(seq_lens)
        total_cost += cost_model.get_cost(batch_size, padded_seq_len)
        
    return total_cost

def schedule_strict_bucketing(
    scenario: List[Tuple[int, int]], cost_model: CostModel
) -> float:
    """Strategy 2: Group by (policy, bucket) and run each as a tiny batch."""
    groups = defaultdict(list)
    for policy_id, seq_len in scenario:
        bucket = get_bucket(seq_len)
        groups[(policy_id, bucket)].append(seq_len)
        
    total_cost = 0.0
    for (policy_id, bucket), seq_lens in groups.items():
        batch_size = len(seq_lens)
        # By definition, all seq_lens in this group are <= bucket
        padded_seq_len = bucket
        total_cost += cost_model.get_cost(batch_size, padded_seq_len)
        
    return total_cost

def schedule_smart_hybrid(
    scenario: List[Tuple[int, int]],
    cost_model: CostModel,
    merge_threshold: int,
) -> float:
    """
    Strategy 3: Start with strict bucketing, then merge small adjacent buckets
    if it's cheaper to do so.
    """
    # 1. Start with strict buckets
    groups = defaultdict(list)
    for policy_id, seq_len in scenario:
        bucket = get_bucket(seq_len)
        groups[(policy_id, bucket)].append(seq_len)
    
    # 2. For each policy, try to merge small buckets
    total_cost = 0.0
    policy_groups = defaultdict(list)
    for (policy_id, bucket), seq_lens in groups.items():
        policy_groups[policy_id].append({'bucket': bucket, 'seq_lens': seq_lens})

    for policy_id, buckets in policy_groups.items():
        # Sort buckets by size to process them in order
        buckets.sort(key=lambda x: x['bucket'])
        
        merged_batches: List[Dict] = []
        
        current_batch = None
        for bucket_info in buckets:
            if not current_batch:
                current_batch = bucket_info
                continue

            # Should we merge the current bucket into our running batch?
            # Only consider merging if the running batch is small
            if len(current_batch['seq_lens']) < merge_threshold:
                # Calculate cost of running separately
                cost_separate = cost_model.get_cost(len(current_batch['seq_lens']), current_batch['bucket']) + \
                                cost_model.get_cost(len(bucket_info['seq_lens']), bucket_info['bucket'])

                # Calculate cost of running combined
                combined_lens = current_batch['seq_lens'] + bucket_info['seq_lens']
                cost_combined = cost_model.get_cost(len(combined_lens), bucket_info['bucket']) # Pad to the larger bucket

                if cost_combined < cost_separate:
                    # It's cheaper to merge! Update the running batch.
                    current_batch['seq_lens'] = combined_lens
                    current_batch['bucket'] = bucket_info['bucket']
                else:
                    # Not worth merging. Finalize the running batch and start a new one.
                    merged_batches.append(current_batch)
                    current_batch = bucket_info
            else:
                 # The running batch is already big enough. Finalize it.
                merged_batches.append(current_batch)
                current_batch = bucket_info

        if current_batch:
            merged_batches.append(current_batch)
            
        # 3. Calculate final cost from the merged batches
        for batch in merged_batches:
            total_cost += cost_model.get_cost(len(batch['seq_lens']), batch['bucket'])

    return total_cost

def main():
    parser = argparse.ArgumentParser(description="Simulate and evaluate scheduler performance.")
    parser.add_argument("performance_csv", type=Path, help="Path to the inference cost CSV file.")
    parser.add_argument("--num-trials", type=int, default=1000, help="Number of random scenarios to simulate.")
    parser.add_argument("--num-requests", type=int, default=200, help="Total pending requests per scenario.")
    parser.add_argument("--num-policies", type=int, default=20, help="Number of distinct historical policies.")
    parser.add_argument("--avg-seq-len", type=int, default=100, help="Average sequence length in scenarios.")
    parser.add_argument("--seq-len-std", type=int, default=50, help="Standard deviation of sequence lengths.")
    
    args = parser.parse_args()

    cost_model = CostModel(args.performance_csv)
    
    # Define strategies to test
    strategies = {
        "Naive Greedy": lambda s, c: schedule_naive_greedy(s, c),
        "Strict Bucketing": lambda s, c: schedule_strict_bucketing(s, c),
        # Test hybrid strategy with different thresholds for what's considered a "small" batch
        "Hybrid (merge<16)": lambda s, c: schedule_smart_hybrid(s, c, merge_threshold=16),
        "Hybrid (merge<32)": lambda s, c: schedule_smart_hybrid(s, c, merge_threshold=32),
        "Hybrid (merge<64)": lambda s, c: schedule_smart_hybrid(s, c, merge_threshold=64),
    }

    results = defaultdict(list)

    print(f"\nRunning {args.num_trials} simulation trials...")
    for i in range(args.num_trials):
        if (i + 1) % 100 == 0:
            print(f"  ... trial {i+1}/{args.num_trials}")
        
        scenario = generate_scenario(
            args.num_requests, args.num_policies, args.avg_seq_len, args.seq_len_std
        )
        
        for name, func in strategies.items():
            results[name].append(func(scenario, cost_model))

    print("\n--- Simulation Results ---")
    print(f"Averaged over {args.num_trials} trials. Lower is better (total ms).")
    
    avg_results = {name: np.mean(times) for name, times in results.items()}
    
    # Sort by performance
    sorted_results = sorted(avg_results.items(), key=lambda item: item[1])
    
    baseline_time = sorted_results[0][1]

    print("\nRank | Strategy          | Avg. Time (ms) | Relative Performance")
    print("---- | ----------------- | -------------- | --------------------")
    for i, (name, avg_time) in enumerate(sorted_results):
        perf = (avg_time / baseline_time - 1) * 100
        perf_str = f"+{perf:.2f}%" if perf > 0 else "Baseline"
        print(f"#{i+1:<3} | {name:<17} | {avg_time:>14.2f} | {perf_str}")

if __name__ == "__main__":
    main()