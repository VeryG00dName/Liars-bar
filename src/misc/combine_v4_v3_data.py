import os
import argparse
import pickle
import pandas as pd
from collections import defaultdict


def normalize_name(name: str) -> str:
    if isinstance(name, str) and name.startswith("Historical_"):
        return name[len("Historical_"):]
    return name

def load_multiple_pickles(file_path: str) -> list:
    """Load all pickled objects from a file with multiple pickle dumps."""
    all_objects = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                obj = pickle.load(f)
                all_objects.append(obj)
            except EOFError:
                break
            except Exception as e:
                print(f"Error reading object from {file_path}: {e}")
                break
    return all_objects


def load_round_data(data_dir: str) -> list:
    all_sequences = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.pkl'):
            continue
        file_path = os.path.join(data_dir, fname)
        try:
            data = load_multiple_pickles(file_path)
            for item in data:
                if isinstance(item, list):
                    all_sequences.extend(item)
                elif isinstance(item, dict):
                    all_sequences.append(item)
                else:
                    print(f"Warning: Unexpected item type in {fname}: {type(item)}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return all_sequences


def extract_matchup_key_for_lookup(round_data: dict) -> str:
    seq = round_data.get('sequence', [])
    for step in seq[:3]:
        belief = step.get('belief')
        if isinstance(belief, list) and len(belief) >= 2:
            short1 = normalize_name(belief[0])
            short2 = normalize_name(belief[1])
            return f"{short1}+{short2}"
    return None


def filter_and_combine(v3_data: list, v4_data: list, table_df: pd.DataFrame) -> list:
    better_map = dict(zip(table_df['Matchup'], table_df['Better Version']))
    combined = []
    matchup_counts = defaultdict(int)
    unmatched = 0

    v3_lookup = defaultdict(list)
    v4_lookup = defaultdict(list)

    for rd in v3_data:
        key = extract_matchup_key_for_lookup(rd)
        v3_lookup[key].append(rd)

    for rd in v4_data:
        key = extract_matchup_key_for_lookup(rd)
        v4_lookup[key].append(rd)

    all_keys = set(v3_lookup.keys()).union(set(v4_lookup.keys()))
    for key in all_keys:
        version = better_map.get(key, 'V3')  # default to V3
        selected = v3_lookup.get(key, []) if version == 'V3' else v4_lookup.get(key, [])
        combined.extend(selected)
        matchup_counts[key] += len(selected)

    # Fallback for unmatched V3 sequences
    for rd in v3_lookup.get(None, []):
        combined.append(rd)
        unmatched += 1

    return combined, matchup_counts, unmatched


def main():
    parser = argparse.ArgumentParser(description="Filter and combine PS data from two versions based on matchup win rates")
    parser.add_argument('--table-file', required=True)
    parser.add_argument('--data-dir-v3', required=True)
    parser.add_argument('--data-dir-v4', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    table_df = pd.read_csv(args.table_file)

    print("Loading v3 data...")
    v3_data = load_round_data(args.data_dir_v3)
    print(f"Loaded {len(v3_data)} sequences from v3.")

    print("Loading v4 data...")
    v4_data = load_round_data(args.data_dir_v4)
    print(f"Loaded {len(v4_data)} sequences from v4.")

    combined, matchup_counts, unmatched = filter_and_combine(v3_data, v4_data, table_df)
    print(f"Selected {len(combined)} total sequences after filtering.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'combined_data.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(combined, f)
    print(f"Saved combined data to {out_path}\n")

    print("Matchup breakdown:")
    for k, v in sorted(matchup_counts.items(), key=lambda x: -x[1]):
        print(f"{k}: {v} sequences")
    print(f"\nUnmatched fallback sequences from V3: {unmatched}")


if __name__ == '__main__':
    main()
