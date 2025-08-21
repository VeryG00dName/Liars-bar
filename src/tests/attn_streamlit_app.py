#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attn_streamlit_app.py (v3.2)
- Fixes the JavaScript data injection bug that broke all hover events.
- The two-column step list is a persistent component.
- View-specific content renders correctly in the panel below.
"""
import argparse, os, json
import streamlit as st
import streamlit.components.v1 as components

def load_episode(path):
    with open(path, "r") as f:
        return json.load(f)

def precompute_layer_meta(steps):
    layer_names, num_heads = [], 0
    for s in steps:
        if s.get("attn_last"):
            for L_name, attn_data in s["attn_last"].items():
                if L_name not in layer_names:
                    layer_names.append(L_name)
                if isinstance(attn_data, list) and len(attn_data) > num_heads:
                    num_heads = len(attn_data)
    return sorted(layer_names), num_heads

PALETTE = ["#e6194B","#3cb44b","#4363d8","#f58231","#911eb4","#46f0f0",
           "#f032e6","#bcf60c","#fabebe","#008080","#e6beff","#9A6324"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="attn_logs")
    args, _ = ap.parse_known_args()

    st.set_page_config(page_title="Game Attention Viz", layout="wide")
    st.sidebar.title("Episodes")
    logs = sorted([f for f in os.listdir(args.logdir) if f.endswith(".json")])
    if not logs:
        st.warning(f"No logs found in {args.logdir}/"); return
    
    fname = st.sidebar.selectbox("Pick an episode", logs, index=len(logs)-1)
    ep = load_episode(os.path.join(args.logdir, fname))
    steps = ep["steps"]
    layers, num_heads = precompute_layer_meta(steps)
    if not layers:
        st.warning("This episode has no recorded attention on our turns."); return

    view_mode = st.radio("Visualization Mode", ["Head View", "Model View", "Neuron View"], horizontal=True)
    
    st.sidebar.title("Controls")
    L_idx = st.sidebar.selectbox("Layer", list(range(len(layers))), format_func=lambda i: layers[i], index=0)
    
    head_idx = 0
    if view_mode == "Neuron View" and num_heads > 0:
        head_idx = st.sidebar.slider("Head for Neuron View", 0, num_heads - 1, 0)
    
    topk = st.sidebar.slider("Top-k per head", 1, 20, 7)
    thresh = st.sidebar.slider("Min weight", 0.0, 1.0, 0.0, 0.01)

    data_json = json.dumps({"steps": steps, "layers": layers})
    palette_json = json.dumps(PALETTE)
    total_height = 820 

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
/* --- SHARED STEP VIEW --- */
#wrap {{ position: relative; width: 100%; height: 400px; border: 1px solid #ddd; margin-bottom: 10px; }}
.col {{ position: absolute; top: 0; bottom: 0; overflow-y: scroll; }}
#colL {{ left: 0; right: 50.5%; border-right: 1px solid #eee; }}
#colR {{ left: 50.5%; right: 0; }}
#listL, #listR {{ padding: 6px 8px; }}
#overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
.line {{ white-space: pre; padding: 1px 4px; border-radius: 3px; font-size: 11px; line-height: 1.15; cursor: default; }}
.line:hover {{ background: rgba(0,0,0,0.05); }}
.hilite {{ outline: 2px solid rgba(0,0,0,0.2); }}
.muted {{ color: #666; font-size: 12px; margin: 4px 0 6px 0; }}

/* --- VIEW-SPECIFIC CONTENT AREA --- */
#view-content-area {{ height: 380px; border: 1px solid #ddd; padding: 10px; overflow-y: auto; }}
.view-container {{ display: none; }}
.view-container.active {{ display: block; }}

/* Head View Content */
.legend {{ margin: 6px 0; }}
.legend span {{ display: inline-block; width: 16px; height: 10px; margin-right: 4px; }}
.badge {{ font-size: 10px; padding: 1px 4px; border-radius: 3px; background: #eee; margin-left: 4px; }}

/* Model View Content */
#model-view-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }}
.mv-cell {{ border: 1px solid #ccc; border-radius: 4px; }}
.mv-cell-title {{ font-size: 10px; padding: 2px 4px; background: #f0f0f0; text-align: center; }}
.mv-cell-canvas {{ width: 100%; height: 100px; }}

/* Neuron View Content */
#neuron-view {{ display: flex; justify-content: center; align-items: flex-start; gap: 20px; margin-top: 10px; }}
.nv-col {{ display: flex; flex-direction: column; align-items: center; }}
.nv-label {{ font-size: 12px; font-weight: bold; margin-bottom: 5px; }}
.nv-vector {{ display: flex; flex-direction: column; gap: 1px; border: 1px solid #ccc; padding: 2px; }}
.nv-neuron {{ width: 20px; height: 10px; }}
#nv-info {{ text-align: center; font-size: 12px; }}
</style>
</head>
<body>

<!-- PERSISTENT SHARED COMPONENT: TWO-COLUMN STEP LIST -->
<div id="wrap">
  <div id="colL" class="col"><div class="muted" style="padding: 6px 8px;">SOURCE — hover here</div><div id="listL"></div></div>
  <div id="colR" class="col"><div class="muted" style="padding: 6px 8px;">TARGET — highlights appear here</div><div id="listR"></div></div>
  <canvas id="overlay"></canvas>
</div>

<!-- DYNAMIC CONTENT AREA: Switches based on selected view -->
<div id="view-content-area">
    <div id="head-view" class="view-container">
        <div class="muted">Attention from hovered source to targets, colored by head.</div>
        <div class="legend" id="legend"></div>
    </div>

    <div id="model-view" class="view-container">
        <div class="muted">Model View: Bird's-eye view of attention from hovered source across all layers/heads.</div>
        <div id="model-view-grid"></div>
    </div>

    <div id="neuron-view" class="view-container">
        <div class="muted">Neuron View: Hover a source, then CLICK a target to see Q*K details for the selected Layer/Head.</div>
        <div id="nv-info">Select a source (hover) and target (click) to begin.</div>
        <div id="neuron-view">
            <div class="nv-col"><div class="nv-label">Query (Q)</div><div id="nv-q-vector" class="nv-vector"></div></div>
            <div class="nv-col"><div class="nv-label">Key (K)</div><div id="nv-k-vector" class="nv-vector"></div></div>
            <div class="nv-col"><div class="nv-label">Q * K</div><div id="nv-prod-vector" class="nv-vector"></div></div>
        </div>
    </div>
</div>

<script>
// --- CORRECTED DATA INJECTION ---
const DATA = {data_json};
const LAYERS = DATA.layers;
const PALETTE = {palette_json};
const VIEW_MODE = '{view_mode}';
const L_IDX = {L_idx};
const HEAD_IDX = {head_idx};
const TOPK = {topk};
const THRESH = {thresh};

// --- View Switching ---
const viewContainers = document.querySelectorAll('.view-container');
viewContainers.forEach(vc => vc.classList.remove('active'));
const activeView = document.getElementById(VIEW_MODE.toLowerCase().replace(' ', '-'));
if (activeView) activeView.classList.add('active');

// --- Element References ---
const wrap = document.getElementById('wrap');
const colL = document.getElementById('colL');
const colR = document.getElementById('colR');
const listL = document.getElementById('listL');
const listR = document.getElementById('listR');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const legend = document.getElementById('legend');
const modelViewGrid = document.getElementById('model-view-grid');

// --- Build text lines on both sides ---
const linesL = [], linesR = [];
DATA.steps.forEach((s, idx) => {{
  const mkLine = (container) => {{
    const l = document.createElement('div');
    l.className = 'line';
    l.dataset.index = idx;
    const mi = s.mi_row || {{t: idx, agent_type: s.actor, prev_action_token: 0, obs: s.obs || []}};
    const obs = (mi.obs || []).map(x => Number.parseFloat(x).toFixed(2)).join(',');
    l.textContent = `${{String(idx).padStart(4)}}|act=${{s.action_token}}|actor=${{s.actor}}|obs=[${{obs.substring(0,30)}}...]`;
    container.appendChild(l);
    return l;
  }};
  linesL.push(mkLine(listL));
  linesR.push(mkLine(listR));
}});

// --- Sizing and Scrolling ---
let syncing = false;
function syncScroll(from, to) {{ if (!syncing) {{ syncing = true; to.scrollTop = from.scrollTop; requestAnimationFrame(() => syncing = false); }} }}
colL.addEventListener('scroll', () => syncScroll(colL, colR));
colR.addEventListener('scroll', () => syncScroll(colR, colL));

function resizeCanvas() {{
  if (!wrap.offsetParent) return;
  const rw = wrap.getBoundingClientRect();
  canvas.width = rw.width; canvas.height = rw.height;
  canvas.style.width = rw.width + 'px'; canvas.style.height = rw.height + 'px';
}}
resizeCanvas(); window.addEventListener('resize', resizeCanvas);

function lineCenterY(el) {{
  const r = el.getBoundingClientRect(); const rw = wrap.getBoundingClientRect();
  return (r.top - rw.top) + r.height/2;
}}
function anchorXs() {{
  const rL = colL.getBoundingClientRect(); const rR = colR.getBoundingClientRect(); const rw = wrap.getBoundingClientRect();
  return [(rL.right - rw.left) - 8, (rR.left - rw.left) + 8];
}}

// --- Core Drawing Logic ---
let lastHover = -1, lastClick = -1;

function clearAllVisuals() {{
    ctx.clearRect(0,0,canvas.width, canvas.height);
    modelViewGrid.innerHTML = '';
    linesR.forEach(l => l.style.background = '');
    linesL.forEach(l => l.classList.remove('hilite'));
    // Reset Neuron view
    document.getElementById('nv-info').textContent = 'Select a source (hover) and target (click) to begin.';
    ['q','k','prod'].forEach(p => document.getElementById(`nv-${{p}}-vector`).innerHTML = '');
}}

function renderLegend(H) {{
  legend.innerHTML = '';
  for (let h=0; h<H; h++) {{
    const span = document.createElement('span');
    span.style.backgroundColor = PALETTE[h % PALETTE.length]; span.title = `Head ${{h+1}}`;
    legend.appendChild(span);
  }}
}}

// --- HEAD VIEW ---
function drawHeadView(qIdx) {{
  clearAllVisuals();
  const step = DATA.steps[qIdx]; if (!step) return;
  const Lname = LAYERS[L_IDX];
  const A = step.attn_last && step.attn_last[Lname]; if (!A || !A.length) return;
  const H = A.length;
  renderLegend(H);

  const [xL, xR] = anchorXs();
  const yQ = lineCenterY(linesL[qIdx]);

  for (let h=0; h<H; h++) {{
    const headWeights = A[h];
    const pairs = headWeights.map((w,k) => [k,w]).sort((a,b) => b[1]-a[1]);
    const maxw = Math.max(...headWeights, 1e-9);
    let drawn = 0;
    for (const [k, wt] of pairs) {{
      if (k >= qIdx || wt < THRESH) continue;
      const yK = lineCenterY(linesR[k]);
      const color = PALETTE[h % PALETTE.length];
      const xm = (xL + xR) / 2;
      const ym = (yK + yQ) / 2 - Math.max(20, Math.abs(yQ - yK)/3);
      ctx.beginPath(); ctx.moveTo(xL, yQ); ctx.quadraticCurveTo(xm, ym, xR, yK);
      ctx.strokeStyle = color; ctx.globalAlpha = 0.25 + 0.75 * (wt / maxw);
      ctx.lineWidth = 2; ctx.stroke();
      linesR[k].style.background = color + '22';
      if (++drawn >= TOPK) break;
    }}
  }}
  ctx.globalAlpha = 1.0;
  linesL[qIdx].classList.add('hilite');
}}

// --- MODEL VIEW ---
function drawModelView(qIdx) {{
    clearAllVisuals();
    linesL[qIdx].classList.add('hilite');
    const step = DATA.steps[qIdx]; if (!step || !step.attn_last) return;

    LAYERS.forEach((Lname, l_idx) => {{
        const A = step.attn_last[Lname]; if (!A || !A.length) return;
        const H = A.length;
        for (let h=0; h<H; h++) {{
            const cell = document.createElement('div'); cell.className = 'mv-cell';
            const title = document.createElement('div'); title.className = 'mv-cell-title';
            title.textContent = `${{Lname}} / Head ${{h}}`;
            const canvas = document.createElement('canvas'); canvas.className = 'mv-cell-canvas';
            cell.appendChild(title); cell.appendChild(canvas);
            modelViewGrid.appendChild(cell);

            const ctx = canvas.getContext('2d');
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width; canvas.height = rect.height;
            const yQ = rect.height * (qIdx / DATA.steps.length);
            
            const headWeights = A[h];
            const pairs = headWeights.map((w,k) => [k,w]).sort((a,b) => b[1]-a[1]);
            const maxw = Math.max(...headWeights.filter(w => !isNaN(w)), 1e-9);
            let drawn = 0;
            for (const [k, wt] of pairs) {{
                if (k >= qIdx || wt < THRESH) continue;
                const yK = rect.height * (k / DATA.steps.length);
                ctx.beginPath(); ctx.moveTo(0, yQ); ctx.lineTo(rect.width, yK);
                ctx.strokeStyle = PALETTE[h % PALETTE.length];
                ctx.globalAlpha = 0.1 + 0.9 * (wt / maxw);
                ctx.lineWidth = 1.5; ctx.stroke();
                if (++drawn >= TOPK) break;
            }}
        }}
    }});
}}

// --- NEURON VIEW ---
function drawNeuronView(qIdx, kIdx) {{
    clearAllVisuals();
    const info = document.getElementById('nv-info');
    const qVecDiv = document.getElementById('nv-q-vector');
    const kVecDiv = document.getElementById('nv-k-vector');
    const prodVecDiv = document.getElementById('nv-prod-vector');
    
    const step = DATA.steps[qIdx]; if (!step) return;
    const Lname = LAYERS[L_IDX];
    const N = step.neuron_data && step.neuron_data[Lname];
    if (!N) {{ info.textContent = `Neuron data not found for Layer ${{L_IDX}}.`; return; }}

    info.textContent = `Q*K for Source:${{qIdx}} -> Target:${{kIdx}} (Layer:${{L_IDX}}, Head:${{HEAD_IDX}})`;
    linesL[qIdx].classList.add('hilite');
    linesR[kIdx].style.background = '#e8e8e8';

    const q_vec = N.q_last[HEAD_IDX];
    const k_vec = N.k_all[HEAD_IDX][kIdx];
    if (!q_vec || !k_vec) {{ info.textContent = 'Q or K vector not found for this head/index.'; return; }}

    const products = q_vec.map((q, i) => q * k_vec[i]);
    const dot_product = products.reduce((a, b) => a + b, 0);
    const max_abs_val = Math.max(...[...q_vec, ...k_vec].map(v => Math.abs(v)));
    const max_abs_prod = Math.max(...products.map(v => Math.abs(v)));

    const valToColor = (v, max_abs) => {{
        const alpha = Math.min(1, Math.abs(v) / (max_abs || 1));
        return v > 0 ? `rgba(50, 50, 200, ${{alpha}})` : `rgba(200, 100, 50, ${{alpha}})`;
    }};

    q_vec.forEach((v, i) => {{
        const neuron = document.createElement('div'); neuron.className = 'nv-neuron';
        neuron.style.background = valToColor(v, max_abs_val); neuron.title = `q[${{i}}] = ${{v.toFixed(4)}}`;
        qVecDiv.appendChild(neuron);
    }});
    k_vec.forEach((v, i) => {{
        const neuron = document.createElement('div'); neuron.className = 'nv-neuron';
        neuron.style.background = valToColor(v, max_abs_val); neuron.title = `k[${{i}}] = ${{v.toFixed(4)}}`;
        kVecDiv.appendChild(neuron);
    }});
    products.forEach((v, i) => {{
        const neuron = document.createElement('div'); neuron.className = 'nv-neuron';
        neuron.style.background = valToColor(v, max_abs_prod); neuron.title = `prod[${{i}}] = ${{v.toFixed(4)}}`;
        prodVecDiv.appendChild(neuron);
    }});
    info.innerHTML += `<br><b>Dot Product (pre-softmax score): ${{dot_product.toFixed(4)}}</b>`;
}}


// --- Event Handling ---
function handleHover(e) {{
  let el = e.target;
  while (el && !el.classList.contains('line')) el = el.parentElement;
  if (!el) return;
  const idx = parseInt(el.dataset.index);
  if (idx === lastHover) return;
  lastHover = idx;
  
  if (VIEW_MODE === 'Head View') {{
      drawHeadView(idx);
  }} else if (VIEW_MODE === 'Model View') {{
      drawModelView(idx);
  }} else {{ // Neuron View
      clearAllVisuals();
      linesL[idx].classList.add('hilite');
      if (lastClick !== -1 && lastClick < idx) drawNeuronView(idx, lastClick);
  }}
}}

function handleClick(e) {{
    if (VIEW_MODE !== 'Neuron View') return;
    let el = e.target;
    while (el && !el.classList.contains('line')) el = el.parentElement;
    if (!el) return;
    const idx = parseInt(el.dataset.index);
    if (idx >= lastHover) return; // Can only attend to the past
    lastClick = idx;
    drawNeuronView(lastHover, lastClick);
}}

listL.addEventListener('mousemove', handleHover);
listR.addEventListener('click', handleClick);

wrap.addEventListener('mouseleave', () => {{
  lastHover = -1;
  lastClick = -1;
  clearAllVisuals();
}});

// Redraw Head View on scroll to keep arcs aligned
colL.addEventListener('scroll', () => {{ 
    if (lastHover >= 0 && VIEW_MODE === 'Head View') drawHeadView(lastHover); 
}});
colR.addEventListener('scroll', () => {{ 
    if (lastHover >= 0 && VIEW_MODE === 'Head View') drawHeadView(lastHover); 
}});


// --- Initial State ---
function initialize() {{
    let startIdx = -1;
    for (let i=DATA.steps.length-1; i>=0; --i) {{ 
      const s = DATA.steps[i];
      const Lname = LAYERS[L_IDX];
      if (s.attn_last && s.attn_last[Lname] && s.attn_last[Lname].length) {{
          startIdx = i;
          break;
      }}
    }}
    if (startIdx !== -1) {{
        lastHover = startIdx;
        if (VIEW_MODE === 'Head View') drawHeadView(startIdx);
        if (VIEW_MODE === 'Model View') drawModelView(startIdx);
    }}
}}
initialize();

</script>
</body>
</html>
"""

    components.html(html, height=total_height, scrolling=False)

if __name__ == "__main__":
    main()