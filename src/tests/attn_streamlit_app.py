"""
attn_streamlit_app.py
Minimal viewer for attention logs produced by run_attn_collect.py.

Run:
  streamlit run attn_streamlit_app.py -- --logdir attn_logs
"""

import argparse, os, json
import numpy as np
import streamlit as st
import plotly.graph_objects as go

def load_episode(path):
    with open(path, "r") as f:
        return json.load(f)

def render_episode(ep):
    steps = ep["steps"]
    st.sidebar.write(f"Steps: {len(steps)}")

    # Step selector
    t = st.slider("Step (hover draws lines from this step back to previous steps)", 0, len(steps)-1, len(steps)-1)
    step = steps[t]

    # Layer names
    layer_names = sorted(list(step["attn_last"].keys()))
    if not layer_names:
        st.warning("No attention recorded on this step. Select a step where actor==player_0 (our turn).")
        return

    layer = st.selectbox("Layer", layer_names, index=0)
    A = np.array(step["attn_last"][layer])  # [H, T]
    H, T = A.shape
    head = st.slider("Head", 0, H-1, 0)

    # Token axis labels = indices (raw expert mode)
    token_labels = [str(i) for i in range(T)]

    # Row list (top-down)
    st.subheader("Timeline (expert mode)")
    st.caption("Each line is a step: actor (0=self, 1/2=opponents), action token, and minimal stats.")
    rows = []
    for s in steps:
        rows.append(f"{s['t']:>4} | a={s['actor']} | act={s['action_token']}")
    st.text("\n".join(rows))

    st.subheader("Last-token → all previous keys (bar)")
    w = A[head]
    fig_bar = go.Figure()
    fig_bar.add_bar(x=list(range(T)), y=w)
    fig_bar.update_layout(xaxis_title="Key index (k)", yaxis_title="Attention weight")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("BertViz-style arcs (top-k per head)")
    topk = st.slider("Top-k per head", 1, min(15, T), min(5, T))
    thresh = float(st.number_input("Min weight threshold", 0.0, 1.0, 0.0, 0.01))

    # Draw arcs from query t to keys k < t
    x = list(range(T))
    y = [0]*T
    fig_arc = go.Figure()
    fig_arc.add_scatter(x=x, y=y, mode="markers+text",
                        text=[f"{i}" for i in range(T)], textposition="top center")

    # compute top-k indices by weight
    idxs = np.argsort(w)[::-1]
    shown = 0
    for k in idxs:
        if k >= T-1:  # skip last self
            continue
        wt = float(w[k])
        if wt < thresh: break
        xm = (t + k) / 2.0
        h  = max(0.2, (t - k) / 2.0)  # arc height
        path = f"M {k},0 Q {xm},{h} {t},0"
        alpha = 0.2 + 0.8 * (wt / (w.max() + 1e-9))
        fig_arc.add_shape(type="path", path=path, line=dict(width=2, color=f"rgba(0,0,0,{alpha:.3f})"))
        shown += 1
        if shown >= topk: break

    fig_arc.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig_arc, use_container_width=True)

    # Logits + mask (if present)
    if step["logits"]:
        st.subheader("Masked policy logits at this step (our turn)")
        st.write(np.array(step["logits"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="attn_logs")
    args, _ = ap.parse_known_args()

    st.sidebar.title("Episodes")
    logs = sorted([f for f in os.listdir(args.logdir) if f.endswith(".json")])
    if not logs:
        st.warning(f"No logs found in {args.logdir}/")
        return
    fname = st.sidebar.selectbox("Pick an episode", logs, index=len(logs)-1)
    ep = load_episode(os.path.join(args.logdir, fname))
    render_episode(ep)

if __name__ == "__main__":
    main()