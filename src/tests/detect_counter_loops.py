#!/usr/bin/env python3
"""
Detect non-transitive "counter loops" (e.g., Rock→Paper→Scissors→Rock) from a
CSV heatmap of head‑to‑head win rates for a multi‑strategy game.

Adds:
- Default threshold = 0.05
- Tracker for when a *lower generation* beats a *higher generation* within the
  same run/test, using the same threshold.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set
import math
import pandas as pd
import re

@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    margin: float  # M[src,dst] - M[dst,src]
    p: float       # M[src,dst]

Graph = Dict[str, Dict[str, Edge]]  # adjacency: u -> {v: Edge}

# ---------------------------- CSV loading ----------------------------

def load_matrix(csv_path: str, percent: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    first_col = df.columns[0]
    if (first_col not in df.columns[1:]) or first_col.lower() in {"run", "strategy", "name", "id"}:
        df = df.set_index(first_col)
    df.index = [str(i).strip() for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce")
    if percent:
        df = df / 100.0
    for s in df.index:
        if s not in df.columns:
            df[s] = pd.NA
    df = df[[c for c in df.columns if c in set(df.index)]]
    df = df.reindex(columns=df.index)
    return df

# ----------------------- Build dominance graph ----------------------

def build_graph(M: pd.DataFrame, threshold: float) -> Graph:
    names = list(M.index)
    adj: Graph = defaultdict(dict)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            p_ab = float(M.loc[a, b])
            p_ba = float(M.loc[b, a])
            if math.isnan(p_ab) or math.isnan(p_ba):
                continue
            margin = p_ab - p_ba
            if p_ab >= 0.5 and margin >= threshold:
                adj[a][b] = Edge(a, b, margin=margin, p=p_ab)
    return adj

# -------------------- Johnson's simple cycles algorithm -------------

def johnson_simple_cycles(adj: Graph,
                          min_len: int = 3,
                          max_len: int = 8) -> Iterable[List[str]]:
    node_list = sorted(adj.keys())
    A: Dict[str, Set[str]] = {u: set(vs.keys()) for u, vs in adj.items()}
    blocked: Dict[str, bool] = {}
    B: Dict[str, Set[str]] = {u: set() for u in A}
    stack: List[str] = []

    def unblock(u: str):
        blocked[u] = False
        while B[u]:
            w = B[u].pop()
            if blocked.get(w, False):
                unblock(w)

    def circuit(start: str, v: str):
        f = False
        stack.append(v)
        blocked[v] = True
        for w in A.get(v, ()):  # neighbors
            if w < start:
                continue
            if w == start:
                if min_len <= len(stack) <= max_len:
                    yield list(stack)
                f = True
            elif len(stack) < max_len and not blocked.get(w, False):
                for cyc in circuit(start, w):
                    yield cyc
                    f = True
        if f:
            unblock(v)
        else:
            for w in A.get(v, ()):  # neighbors
                if v not in B[w]:
                    B[w].add(v)
        stack.pop()

    for i, s in enumerate(node_list):
        eligible = [u for u in node_list if u >= s]
        A = {u: {v for v in A.get(u, set()) if v in eligible and v >= s}
             for u in eligible}
        blocked = {u: False for u in eligible}
        B = {u: set() for u in eligible}
        for cyc in circuit(s, s) or []:
            yield cyc

def is_chordless(cyc, adj):
    k = len(cyc)
    nodes = set(cyc)
    for i, u in enumerate(cyc):
        allowed = {cyc[(i+1)%k], cyc[(i-1)%k]}  # only next/prev on the ring
        for v in adj.get(u, {}):
            if v in nodes and v not in allowed:
                return False
    return True

# --------------------------- Cycle scoring --------------------------

def cycle_bottleneck_margin(cycle: List[str], adj: Graph) -> float:
    m = float("inf")
    for u, v in zip(cycle, cycle[1:] + cycle[:1]):
        m = min(m, adj[u][v].margin)
    return m

# ----------------------- Gen parsing helpers ------------------------

def parse_run_and_gen(name: str) -> Tuple[str, int]:
    """Extract run/test id and generation number from a label like 'test40_gen_26'."""
    m = re.match(r"^(.*)_gen_(\d+)$", name)
    if not m:
        return name, -1
    return m.group(1), int(m.group(2))

# ------------------------------- CLI --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Detect non-transitive counter loops from a H2H win-rate heatmap.")
    ap.add_argument("csv", help="Path to square heatmap CSV")
    ap.add_argument("--threshold", type=float, default=0.05, help="Min margin (P(a>b)-P(b>a)) to create edge [default: 0.05]")
    ap.add_argument("--percent", action="store_true", help="Interpret cells as percentages (0-100) instead of 0-1")
    ap.add_argument("--min-len", type=int, default=3, help="Minimum cycle length [default: 3]")
    ap.add_argument("--max-len", type=int, default=8, help="Maximum cycle length [default: 8]")
    ap.add_argument("--dot", type=str, default=None, help="Optional path to write GraphViz .dot")
    ap.add_argument("--prefer-run", type=str, default=None,
                    help="Run/test ID to prioritize (e.g. 'test40'). Cycles fully within this run are shown first, then mixed, then none.")
    args = ap.parse_args()

    M = load_matrix(args.csv, percent=args.percent)
    G = build_graph(M, threshold=args.threshold)

    # Find and score cycles
    cycles = list(johnson_simple_cycles(G, min_len=args.min_len, max_len=args.max_len))
    cycles = [c for c in cycles if is_chordless(c, G)]
    scored: List[Tuple[float, List[str]]] = [
        (cycle_bottleneck_margin(c, G), c) for c in cycles
    ]

    if args.dot:
        # collect edges that appear in the kept cycles
        edges_in_cycles = {
            (u, v)
            for cyc in cycles
            for (u, v) in zip(cyc, cyc[1:] + cyc[:1])
        }
        nodes_in_cycles = {u for (u, v) in edges_in_cycles} | {v for (u, v) in edges_in_cycles}

        with open(args.dot, "w", encoding="utf-8") as f:
            f.write("digraph G {\n")
            # optional: explicitly declare nodes (helps if you later style nodes)
            for u in sorted(nodes_in_cycles):
                f.write(f'  "{u}";\n')
            # only edges that belong to chordless cycles
            for (u, v) in sorted(edges_in_cycles):
                e = G[u][v]
                f.write(f'  "{u}" -> "{v}" [label="{e.margin:.3f} / p={e.p:.3f}"];\n')
            f.write("}\n")
        print(f'Wrote {args.dot} (from {len(cycles)} chordless cycles, {len(edges_in_cycles)} edges)')
    
    # Prioritized sorting: cycles within preferred run, then mixed (include it), then none
    def cycle_runs(cyc: List[str]) -> set[str]:
        return {parse_run_and_gen(n)[0] for n in cyc}

    def group_key(cyc: List[str]) -> int:
        if not args.prefer_run:
            return 1  # neutral grouping when no preference
        runs = cycle_runs(cyc)
        if runs == {args.prefer_run}:
            return 0  # only preferred run
        if args.prefer_run in runs:
            return 1  # mixed
        return 2      # none from preferred

    scored.sort(key=lambda x: (group_key(x[1]), -x[0], len(x[1]), x[1]))

    if not scored:
        print("No counter cycles found with current threshold.")
    else:
        hdr = f"Found {len(scored)} cycles (showing top 32)"
        if args.prefer_run:
            hdr += f" [prioritized for '{args.prefer_run}']"
        print(hdr + ":\n")

        last_group = None
        for i, (marg, cyc) in enumerate(scored[:32], 1):
            if args.prefer_run:
                g = group_key(cyc)
                if g != last_group:
                    titles = {
                        0: f"== Cycles within {args.prefer_run} ==",
                        1: f"== Mixed cycles (include {args.prefer_run}) ==",
                        2: f"== Cycles with no strategies from {args.prefer_run} ==",
                    }
                    print(titles[g])
                    last_group = g
            edges = [f"{u}→{v} ({G[u][v].margin:+.3f})" for u, v in zip(cyc, cyc[1:] + cyc[:1])]
            print(f"#{i:02d}  length={len(cyc)}  bottleneck_margin={marg:.3f}  cycle: {'  '.join(edges)}")

    # ----------------- Tracker: lower-gen beats higher-gen ------------
    print("\nLower-gen beats higher-gen (same test, threshold applied):")
    count = 0
    for u, nbrs in G.items():
        run_u, gen_u = parse_run_and_gen(u)
        for v, e in nbrs.items():
            run_v, gen_v = parse_run_and_gen(v)
            if run_u == run_v and gen_u >= 0 and gen_v >= 0:
                if gen_u < gen_v:  # lower gen beats higher gen
                    count += 1
                    print(f"{u} → {v} margin={e.margin:.3f} p={e.p:.3f}")
    if count == 0:
        print("(none)")

if __name__ == "__main__":
    main()