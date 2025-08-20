
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attn_streamlit_app.py (v2.2)
- Two lists (left & right), both top-down; hover on LEFT draws head-colored arcs to RIGHT.
- Full model-input row printed on both sides (expert mode).
- Synchronized scrolling + responsive canvas; compact font for density.
"""
import argparse, os, json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

def load_episode(path):
    with open(path, "r") as f:
        return json.load(f)

def precompute_layer_meta(steps):
    layer_names = []
    for s in steps:
        for L in s.get("attn_last", {}).keys():
            if L not in layer_names:
                layer_names.append(L)
    return sorted(layer_names)

PALETTE = [
    "#e6194B","#3cb44b","#4363d8","#f58231","#911eb4","#46f0f0",
    "#f032e6","#bcf60c","#fabebe","#008080","#e6beff","#9A6324"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="attn_logs")
    args, _ = ap.parse_known_args()

    st.set_page_config(page_title="Game Attention Viz", layout="wide")
    st.sidebar.title("Episodes")
    logs = sorted([f for f in os.listdir(args.logdir) if f.endswith(".json")])
    if not logs:
        st.warning(f"No logs found in {args.logdir}/")
        return
    fname = st.sidebar.selectbox("Pick an episode", logs, index=len(logs)-1)
    ep = load_episode(os.path.join(args.logdir, fname))
    steps = ep["steps"]
    layers = precompute_layer_meta(steps)
    if not layers:
        st.warning("This episode has no recorded attention on our turns.")
        return

    L_idx = st.sidebar.selectbox("Layer", list(range(len(layers))), format_func=lambda i: layers[i], index=0)
    topk = st.sidebar.slider("Top-k per head", 1, 20, 7)
    thresh = st.sidebar.slider("Min weight", 0.0, 1.0, 0.0, 0.01)

    data_json = json.dumps({"steps": steps, "layers": layers})
    palette_json = json.dumps(PALETTE)

    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
#wrap {{ position: relative; width: 100%; height: 80vh; border: 1px solid #ddd; }}
.col {{ position: absolute; top: 0; bottom: 0; overflow: auto; }}
#colL {{ left: 0; right: 50.5%; border-right: 1px solid #eee; }}
#colR {{ left: 50.5%; right: 0; }}
#listL, #listR {{ padding: 6px 8px; }}
#overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
.line {{ white-space: pre; padding: 1px 4px; border-radius: 3px; font-size: 11px; line-height: 1.15; }}
.line:hover {{ background: rgba(0,0,0,0.05); }}
.badge {{ font-size: 10px; padding: 1px 4px; border-radius: 3px; background: #eee; margin-left: 4px; }}
.legend {{ margin: 6px 0; }}
.legend span {{ display: inline-block; width: 16px; height: 10px; margin-right: 4px; }}
.controls {{ margin: 8px 0; }}
.hilite {{ outline: 2px solid rgba(0,0,0,0.2); }}
.muted {{ color: #666; }}
.header {{ font-size: 12px; margin: 4px 0 6px 0; }}
</style>
</head>
<body>
<div class="controls">
  <label>Layer:
    <select id="layerSel"></select>
  </label>
  <label style="margin-left:16px;">Top-k:
    <input id="topk" type="number" min="1" max="50" value="{topk}" style="width:60px;">
  </label>
  <label style="margin-left:16px;">Min weight:
    <input id="thresh" type="number" min="0" max="1" step="0.01" value="{thresh}" style="width:80px;">
  </label>
</div>
<div class="legend" id="legend"></div>

<div id="wrap">
  <div id="colL" class="col">
    <div class="header muted">SOURCE — hover here</div>
    <div id="listL"></div>
  </div>
  <div id="colR" class="col">
    <div class="header muted">TARGET — attention highlights appear here</div>
    <div id="listR"></div>
  </div>
  <canvas id="overlay"></canvas>
</div>

<script>
const DATA = {data_json};
const LAYERS = DATA.layers;
const PALETTE = {palette_json};

// Controls
const layerSel = document.getElementById('layerSel');
LAYERS.forEach((L, i) => {{ const o = document.createElement('option'); o.value = i; o.textContent = L; layerSel.appendChild(o); }});
layerSel.value = {L_idx};
const topkInput = document.getElementById('topk');
const threshInput = document.getElementById('thresh');

// Legend
const legend = document.getElementById('legend');
function renderLegend(H) {{
  legend.innerHTML = '';
  for (let h=0; h<H; h++) {{
    const span = document.createElement('span');
    span.style.backgroundColor = PALETTE[h % PALETTE.length];
    span.title = `Head ${{h+1}}`;
    legend.appendChild(span);
  }}
  legend.innerHTML += '<span class="badge">Heads</span>';
}}

// Columns & overlay
const wrap = document.getElementById('wrap');
const colL = document.getElementById('colL');
const colR = document.getElementById('colR');
const listL = document.getElementById('listL');
const listR = document.getElementById('listR');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

// Build text lines on both sides
const linesL = [], linesR = [];
DATA.steps.forEach((s, idx) => {{
  const mkLine = (container, side) => {{
    const l = document.createElement('div');
    l.className = 'line';
    l.dataset.index = idx;
    const mi = s.mi_row || {{t: idx, agent_type: s.actor, prev_action_token: (idx>0?DATA.steps[idx-1].action_token:0), obs: s.obs || [], mask: s.mask || []}};
    const obs = (mi.obs || []).map(x => Number.parseFloat(x).toFixed(3)).join(', ');
    const mask = (mi.mask || []).join('');
    l.textContent = `${{String(idx).padStart(4)}} | actor=${{s.actor}} | act=${{s.action_token}} | mi.t=${{mi.t}} | agent_type=${{mi.agent_type}} | prev=${{mi.prev_action_token}} | obs=[${{obs}}] | mask=[${{mask}}]`;
    container.appendChild(l);
    return l;
  }};
  linesL.push(mkLine(listL, 'L'));
  linesR.push(mkLine(listR, 'R'));
}});

// Sync scroll between columns
let syncing = false;
function syncScroll(from, to) {{
  if (syncing) return;
  syncing = true;
  to.scrollTop = from.scrollTop;
  requestAnimationFrame(() => syncing = false);
}}
colL.addEventListener('scroll', () => syncScroll(colL, colR));
colR.addEventListener('scroll', () => syncScroll(colR, colL));

// Sizing
function resizeCanvas() {{
  const rw = wrap.getBoundingClientRect();
  canvas.width = rw.width;
  canvas.height = rw.height;
  canvas.style.width = rw.width + 'px';
  canvas.style.height = rw.height + 'px';
}}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Utility: Y center of a line relative to wrap
function lineCenterY(el) {{
  const r = el.getBoundingClientRect();
  const rw = wrap.getBoundingClientRect();
  return (r.top - rw.top) + r.height/2;
}}

// X anchors (right edge of left column, left edge of right column)
function anchorXs() {{
  const rL = colL.getBoundingClientRect();
  const rR = colR.getBoundingClientRect();
  const rw = wrap.getBoundingClientRect();
  const xL = (rL.right - rw.left) - 8; // a little inset
  const xR = (rR.left - rw.left) + 8;
  return [xL, xR];
}}

// Draw arcs from left row q to right rows using attn weights
let lastHover = -1;
function drawForIndex(qIdx) {{
  ctx.clearRect(0,0,canvas.width, canvas.height);
  linesR.forEach(l => l.style.background = '');
  linesL.forEach(l => l.classList.remove('hilite'));

  const step = DATA.steps[qIdx];
  if (!step) return;

  const Lname = LAYERS[parseInt(layerSel.value)];
  const A = step.attn_last && step.attn_last[Lname];
  if (!A || !A.length) return;

  const H = A.length;
  renderLegend(H);

  // anchors
  const [xL, xR] = anchorXs();
  const yQ = lineCenterY(linesL[qIdx]);

  const topk = parseInt(topkInput.value || '5');
  const thr = parseFloat(threshInput.value || '0');

  for (let h=0; h<H; h++) {{
    const headWeights = A[h];
    const T = headWeights.length;
    const pairs = [];
    for (let k=0; k<T; k++) {{ pairs.push([k, headWeights[k]]); }}
    pairs.sort((a,b) => b[1]-a[1]);
    let drawn = 0;
    const maxw = Math.max(...headWeights, 1e-9);
    for (const [k, wt] of pairs) {{
      if (k >= qIdx) continue;
      if (wt < thr) continue;
      const yK = lineCenterY(linesR[k]);
      const color = PALETTE[h % PALETTE.length];
      const xm = (xL + xR) / 2;
      const ym = (yK + yQ) / 2 - Math.max(20, Math.abs(yQ - yK)/3);
      ctx.beginPath();
      ctx.moveTo(xL, yQ);
      ctx.quadraticCurveTo(xm, ym, xR, yK);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.25 + 0.75 * (wt / maxw);
      ctx.lineWidth = 2;
      ctx.stroke();
      linesR[k].style.background = color + '22';
      drawn++;
      if (drawn >= topk) break;
    }}
  }}
  ctx.globalAlpha = 1.0;
  linesL[qIdx].classList.add('hilite');
}}

// Hover handling on LEFT column
listL.addEventListener('mousemove', (e) => {{
  let el = e.target;
  while (el && !el.classList.contains('line')) el = el.parentElement;
  if (!el) return;
  const idx = parseInt(el.dataset.index);
  if (idx === lastHover) return;
  lastHover = idx;
  drawForIndex(idx);
}});

wrap.addEventListener('mouseleave', () => {{
  lastHover = -1;
  ctx.clearRect(0,0,canvas.width, canvas.height);
  linesR.forEach(l => l.style.background='');
  linesL.forEach(l => l.classList.remove('hilite'));
}});

// Redraw on scroll to keep alignment
[colL, colR].forEach(sc => sc.addEventListener('scroll', () => {{ if (lastHover>=0) drawForIndex(lastHover); }}));

// Initial state: draw for last step that has attention
let startIdx = DATA.steps.length-1;
for (let i=DATA.steps.length-1; i>=0; --i) {{ 
  const s = DATA.steps[i];
  const Lname = LAYERS[{L_idx}];
  if (s.attn_last && s.attn_last[Lname] && s.attn_last[Lname].length) {{ startIdx = i; break; }}
}}
drawForIndex(startIdx);

// React to control changes
[layerSel, topkInput, threshInput].forEach(ctrl => ctrl.addEventListener('change', () => {{ if (lastHover>=0) drawForIndex(lastHover); }}));

</script>
</body>
</html>
""".format(
        data_json=data_json,
        palette_json=palette_json,
        L_idx=L_idx,
        topk=topk,
        thresh=thresh,
    )

    components.html(html, height=820, scrolling=True)

if __name__ == "__main__":
    main()
