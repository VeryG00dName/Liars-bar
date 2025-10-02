#!/usr/bin/env python3
"""
Detects non-transitive "counter loops" and generational upsets from a
head-to-head win-rate matrix, using a correct, fast, and memory-efficient
implementation of Johnson's algorithm for simple cycle enumeration.
"""
import argparse
import heapq
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set

import numpy as np
import pandas as pd

BOT_RE = re.compile(r"^cpp_bot_")  # e.g. cpp_bot_1_GreedyCardSpammer
def is_bot(name: str) -> bool:
    return bool(BOT_RE.match(name))

@dataclass(frozen=True)
class Edge:
    """Represents a directed edge in the dominance graph."""
    src: str
    dst: str
    margin: float  # WinRate(src vs dst) - WinRate(dst vs src)
    p: float       # WinRate(src vs dst)

Graph = Dict[str, Dict[str, Edge]]

# ---------------------------- Core Logic ----------------------------

def load_matrix_and_build_graph(csv_path: str, threshold: float) -> Tuple[Graph, pd.DataFrame]:
    """
    Loads a CSV win-rate matrix and builds a dominance graph (vectorized).
    Edge a->b exists when (p_ab - p_ba) > threshold and both entries are finite.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)

    # Make sure it's square and aligned over the same ordered labels.
    all_players = sorted(set(df.index) | set(df.columns))
    df = df.reindex(index=all_players, columns=all_players, fill_value=np.nan).astype(np.float32)

    names = df.index.to_list()
    M = df.to_numpy(copy=False)  # float32

    # Valid pairs (both directions present)
    valid = np.isfinite(M) & np.isfinite(M.T)
    # Margin mask
    margin_mat = M - M.T  # float32, nan where either was nan
    mask = valid & (margin_mat > threshold)
    # Remove diagonal just in case
    np.fill_diagonal(mask, False)

    # Build adjacency only for true edges
    adj: Graph = defaultdict(dict)
    src_idx, dst_idx = np.where(mask)
    for i, j in zip(src_idx.tolist(), dst_idx.tolist()):
        a, b = names[i], names[j]
        # margin_mat[i, j] is already p_ab - p_ba; M[i, j] is p_ab
        adj[a][b] = Edge(a, b, margin=float(margin_mat[i, j]), p=float(M[i, j]))
    return adj, df

def find_simple_cycles_johnson(adj: Graph, max_len: int | None = None) -> Iterable[List[str]]:
    """
    Enumerate all simple cycles using Johnson's algorithm.
    - Excludes bots (names starting with 'cpp_bot_') from cycle search.
    - Prunes DFS by `max_len` so recursion depth stays bounded.
    - Uses iterative unblock to avoid recursion blowups on dense SCCs.
    Returns cycles as lists of node names.
    """
    # ---- Build node list = sources ∪ destinations, excluding bots ----
    all_nodes: Set[str] = set(adj.keys())
    for nbrs in adj.values():
        all_nodes.update(nbrs.keys())
    nodes = sorted(n for n in all_nodes if not is_bot(n))
    if not nodes:
        return
        yield  # keep as generator

    # Map to ints for speed
    n = len(nodes)
    idx = {name: i for i, name in enumerate(nodes)}

    # Int adjacency (only among non-bot nodes)
    A: List[Set[int]] = [set() for _ in range(n)]
    for u_name in nodes:
        u = idx[u_name]
        if u_name in adj:
            for v_name in adj[u_name].keys():
                if v_name in idx:
                    A[u].add(idx[v_name])

    # ---- Tarjan SCC on subgraph induced by vertices >= start ----
    def sccs_from(start: int) -> List[List[int]]:
        index = 0
        stack: List[int] = []
        onstack = [False] * n
        indices = [-1] * n
        low = [0] * n
        comps: List[List[int]] = []

        def strongconnect(v: int):
            nonlocal index
            indices[v] = index
            low[v] = index
            index += 1
            stack.append(v)
            onstack[v] = True
            for w in A[v]:
                if w < start:
                    continue
                if indices[w] == -1:
                    strongconnect(w)
                    if low[w] < low[v]:
                        low[v] = low[w]
                elif onstack[w]:
                    if indices[w] < low[v]:
                        low[v] = indices[w]
            if low[v] == indices[v]:
                comp: List[int] = []
                while True:
                    w = stack.pop()
                    onstack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(comp)

        for v in range(start, n):
            if indices[v] == -1:
                strongconnect(v)
        return comps

    blocked: Set[int] = set()
    B: List[Set[int]] = [set() for _ in range(n)]
    stack_path: List[int] = []

    # Iterative unblock to avoid recursion depth issues
    def unblock(u: int):
        todo = [u]
        seen_local = set()
        while todo:
            x = todo.pop()
            if x in seen_local:
                continue
            seen_local.add(x)
            if x in blocked:
                blocked.remove(x)
                # Copy because we'll mutate
                for w in list(B[x]):
                    B[x].remove(w)
                    todo.append(w)

    # Johnson's circuit subroutine with depth cap
    def circuit(v: int, s: int, H: List[Set[int]]):
        closed = False
        stack_path.append(v)
        blocked.add(v)
        for w in H[v]:
            if w == s:
                # Found a cycle
                yield [nodes[i] for i in stack_path]
                closed = True
            elif w not in blocked and (max_len is None or len(stack_path) < max_len):
                for cyc in circuit(w, s, H):
                    closed = True
                    yield cyc
        if closed:
            unblock(v)
        else:
            for w in H[v]:
                B[w].add(v)
        stack_path.pop()

    # ---- Main Johnson loop ----
    start = 0
    while start < n:
        # SCCs of subgraph induced by vertices >= start
        comps = sccs_from(start)

        # Find the SCC with the least vertex that actually contains a cycle
        best_comp = None
        best_min = None
        for comp in comps:
            # SCC contains a cycle if |comp|>1 or self-loop
            has_cycle = (len(comp) > 1) or any((v in A[v]) for v in comp)
            if not has_cycle:
                continue
            cmin = min(comp)
            if cmin < start:
                continue
            if best_min is None or cmin < best_min:
                best_min = cmin
                best_comp = comp

        if best_comp is None:
            break

        s = best_min
        C = set(best_comp)

        # Build H = subgraph induced by this SCC
        H: List[Set[int]] = [set() for _ in range(n)]
        for u in C:
            H[u] = {w for w in A[u] if w in C}

        # Reset Johnson bookkeeping for this SCC
        blocked.clear()
        for u in C:
            B[u].clear()
        stack_path.clear()

        # Run circuit search from s
        for cyc in circuit(s, s, H):
            yield cyc

        # Remove s from A
        for u in range(n):
            if s in A[u]:
                A[u].remove(s)
        A[s].clear()

        start = s + 1

def _find_sccs_tarjan(nodes: List[str], adj: Graph) -> List[List[str]]:
    """Helper to find strongly connected components using Tarjan's algorithm."""
    index_counter = 0
    stack: List[str] = []
    on_stack: Dict[str, bool] = defaultdict(bool)
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    result: List[List[str]] = []

    def strong_connect(v: str):
        nonlocal index_counter
        indices[v] = index_counter
        lowlinks[v] = index_counter
        index_counter += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj.get(v, {}):
            if w not in indices:
                strong_connect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if v == w:
                    break
            result.append(scc)

    for v in nodes:
        if v not in indices:
            strong_connect(v)
    return result

def _canon_cycle_key(cyc: List[str]) -> Tuple[str, ...]:
    # canonicalize up to rotation and reversal to avoid duplicates
    n = len(cyc)
    rots_fwd = [tuple(cyc[i:] + cyc[:i]) for i in range(n)]
    rev = list(reversed(cyc))
    rots_rev = [tuple(rev[i:] + rev[:i]) for i in range(n)]
    return min(min(ots) for ots in (rots_fwd, rots_rev))

def find_and_report_cycles(adj: Graph, min_len: int, max_len: int, max_report: int):
    print(f"Searching for simple cycles of length {min_len}-{max_len} (bots excluded)...")
    top_cycles: List[Tuple[float, List[str]]] = []
    seen: Set[Tuple[str, ...]] = set()
    count = 0

    # >>> key change: pass max_len into Johnson so DFS depth is bounded
    for cycle in find_simple_cycles_johnson(adj, max_len=max_len):
        L = len(cycle)
        if L < min_len or L > max_len:
            continue
        key = _canon_cycle_key(cycle)
        if key in seen:
            continue
        seen.add(key)
        count += 1

        margin = min(adj[u][v].margin for u, v in zip(cycle, cycle[1:] + [cycle[0]]))
        if len(top_cycles) < max_report:
            heapq.heappush(top_cycles, (margin, cycle))
        elif margin > top_cycles[0][0]:
            heapq.heapreplace(top_cycles, (margin, cycle))

    if not top_cycles:
        print("No counter cycles found with the current threshold.")
        return

    print(f"\nFound {count} unique cycles. Reporting top {len(top_cycles)} by bottleneck margin:")
    for i, (margin, cycle) in enumerate(sorted(top_cycles, key=lambda x: x[0], reverse=True), 1):
        print(f"#{i:02d} | Bottleneck: {margin:.3f} | Len: {len(cycle)} | {' >> '.join(cycle)} → {cycle[0]}")

def find_and_report_upsets(df: pd.DataFrame, threshold: float):
    """
    Reports:
      1) Lower-gen beats higher-gen within the same test (as before)
      2) BOT upsets: any cpp_bot_* defeating a non-bot with margin > threshold
    """
    print("\nLower-gen beats higher-gen (same test, margin > threshold):")

    pattern = re.compile(r"^(test\d+)_gen_(\d+)")
    def parse_name(name: str) -> Tuple[str | None, int]:
        m = pattern.match(name)
        return (m.group(1), int(m.group(2))) if m else (None, -1)

    players = df.index.to_list()
    values = df.to_numpy(copy=False)

    # ---------- Part 1: generational upsets (unchanged logic) ----------
    gen_upsets = []
    for i, a in enumerate(players):
        run_a, gen_a = parse_name(a)
        if gen_a == -1:
            continue
        for j in range(i + 1, len(players)):
            b = players[j]
            run_b, gen_b = parse_name(b)
            if gen_b == -1 or run_a != run_b:
                continue

            p_ab = values[i, j]
            p_ba = values[j, i]
            if not (np.isfinite(p_ab) and np.isfinite(p_ba)):
                continue

            if gen_a < gen_b and (p_ab - p_ba) > threshold:
                gen_upsets.append((a, b, p_ab, p_ba, p_ab - p_ba))
            elif gen_b < gen_a and (p_ba - p_ab) > threshold:
                gen_upsets.append((b, a, p_ba, p_ab, p_ba - p_ab))

    if not gen_upsets:
        print("(none)")
    else:
        gen_upsets.sort(key=lambda x: x[4], reverse=True)
        for lower_gen, higher_gen, wr_lower, wr_higher, margin in gen_upsets:
            print(f"{lower_gen} ({wr_lower:.3f}) DEFEATS {higher_gen} ({wr_higher:.3f}) | Margin: {margin:.3f}")

    # ---------- Part 2: bot upsets ----------
    print("\nBOT upsets (cpp_bot_* defeats non-bot, margin > threshold):")

    bot_upsets = []
    name_to_idx = {name: k for k, name in enumerate(players)}
    for a in players:
        ia = name_to_idx[a]
        for b in players:
            if a == b:
                continue
            ib = name_to_idx[b]
            pa_b = values[ia, ib]
            pb_a = values[ib, ia]
            if not (np.isfinite(pa_b) and np.isfinite(pb_a)):
                continue

            # If a is bot and b is non-bot, check if bot beats non-bot
            if is_bot(a) and not is_bot(b):
                margin = pa_b - pb_a
                if margin > threshold:
                    bot_upsets.append((a, b, pa_b, pb_a, margin))

    if not bot_upsets:
        print("(none)")
    else:
        bot_upsets.sort(key=lambda x: x[4], reverse=True)
        for bot, nonbot, wr_bot, wr_nonbot, margin in bot_upsets:
            print(f"{bot} ({wr_bot:.3f}) DEFEATS {nonbot} ({wr_nonbot:.3f}) | Margin: {margin:.3f}")

# ------------------------------- CLI --------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect non-transitive counter loops and generational upsets from a H2H win-rate matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("csv", help="Path to the square H2H win-rate matrix CSV file.")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Minimum win-rate margin (WR_A - WR_B) to be considered a 'beat'.")
    parser.add_argument("--min-len", type=int, default=3, help="Minimum cycle length to report.")
    parser.add_argument("--max-len", type=int, default=5, help="Maximum cycle length to search for.")
    parser.add_argument("--max-report", type=int, default=32, help="Maximum number of top cycles to report.")
    args = parser.parse_args()

    try:
        print(f"Loading matrix from: {args.csv}")
        adj, df = load_matrix_and_build_graph(args.csv, threshold=args.threshold)
        print(f"Graph built with {len(adj)} nodes and {sum(len(v) for v in adj.values())} edges.")
        
        find_and_report_cycles(adj, args.min_len, args.max_len, args.max_report)
        find_and_report_upsets(df, args.threshold)

    except FileNotFoundError:
        print(f"Error: The file '{args.csv}' was not found.")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()