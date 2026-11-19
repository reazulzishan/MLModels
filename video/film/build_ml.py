"""CS Funda — "Machine Learning Models, explained" (YouTube 16:9 tutorial + one 9:16 Reel under 3 minutes).

Reuses the CS Funda engine (captions, real-cell camera, glass, charts) through the ScamGuard build module.
python build_ml.py        -> this folder (YouTube 1920x1080)
python build_ml.py reel   -> ../reel (1080x1920)
Every number and every visual comes from notebooks/ml_models_live.ipynb (-> reports/viz.json).
"""
import importlib.util, json, math, re, shutil, sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
VID = HERE.parent
ROOT = VID.parent

spec = importlib.util.spec_from_file_location("sg", Path.home() / "Projects/ScamGuard/video/film/build_sg.py")
sg = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg)
b, e = sg.b, sg.e

# ---- point the engine at this project
b.HERE = HERE; e.HERE = HERE; sg.HERE = HERE
b.REC = VID
b.EP1_SFX = False
b.BOXES = json.loads((ROOT / "recording/boxes.json").read_text())
meta = json.loads((VID / "vo/scenes.json").read_text())
b.SCENES = dict(meta)
b.OUTROS = {}
V = json.loads((ROOT / "reports/viz.json").read_text())
LV = json.loads((ROOT / "reports/live.json").read_text())   # one real example through each model (video/realviz_live.py, verified)
b.HOLD = {"Q03": 0.4, "Q04": 0.4, "Q05": 0.4, "Q06": 0.6, "Q07": 0.5, "Q08": 0.5, "Q09": 0.4, "Q10": 1.2, "Y01": 0.4, "Y03": 0.8, "Y06": 0.6, "Y09": 0.5, "Y11": 0.8, "Y14": 1.0, "Y17": 0.6, "Y19": 1.0, "Y22": 0.8, "Y26": 0.6, "Y28": 1.4,
          "Y31": 0.8, "Y34": 0.6, "Y37": 1.0, "Y40": 1.2, "Y41": 0.6, "Y43": 0.8, "Y45": 0.6, "Y46": 1.6, "Y47": 1.6,
          "R03": 0.4, "R04": 0.4, "R05": 0.4, "R06": 0.6, "R07": 0.5, "R08": 0.5, "R09": 0.4, "R10": 1.2}
b.SFX = {k: ("whoosh-short", 0.0) for k in ["Y04", "Y07", "Y10", "Y13", "Y16", "Y18", "Y21", "Y24", "Y27", "Y30", "Y33", "Y36", "Y39", "Y42", "Y44"]}
b.SFX.update({"Y46": ("whoosh-cinematic", 0.0), "Q03": ("whoosh-short", 0.0), "Q05": ("whoosh-short", 0.0), "Q07": ("whoosh-short", 0.0), "Q09": ("whoosh-short", 0.0)})
INK, AMB, RED, GRN, MUTE = b.INK, b.AMB, b.RED, b.GRN, sg.MUTE
BLU, PUR = "#6E9BFF", "#B48CFF"
CLS = [RED, AMB, GRN]
DIG = ["#FF6B6B", "#FFB224", "#3DDC97", "#6E9BFF", "#B48CFF", "#FF8FD1", "#5FD7E8", "#C9E265", "#F3F0E8", "#FF9F6B"]
at, label, kinetic, chapter, glass, aura = b.at, b.label, b.kinetic, b.chapter, b.glass, b.aura
chart, curve, dot, hbars, layer, card, meters, stamp, countup, capture, seg = e.chart, e.curve, e.dot, e.hbars, e.layer, sg.card, sg.meters, sg.stamp, sg.countup, sg.capture, sg.seg
sfx, EXTRA = e.sfx, e.EXTRA
FONT = sg.FONT

MODELS = [("Linear Regression", "TABLES · PREDICT A NUMBER"), ("Logistic Regression", "TABLES · YES / NO"), ("k-Nearest Neighbours", "TABLES · SMALL DATA"),
          ("Decision Tree", "TABLES · RULES"), ("Random Forest", "TABLES · ENSEMBLE"), ("Gradient Boosting", "TABLES · ENSEMBLE"),
          ("Support Vector Machine", "TABLES · MANY FEATURES"), ("Naive Bayes", "TEXT"), ("K-Means", "NO LABELS · CLUSTERING"),
          ("PCA", "NO LABELS · COMPRESSION"), ("Neural Network (MLP)", "DEEP LEARNING"), ("CNN", "IMAGES"), ("LSTM", "SEQUENCES · TIME SERIES"),
          ("Transformer", "TEXT · LANGUAGE"), ("Large Language Model", "TEXT · LANGUAGE")]
# which model each YouTube beat belongs to (1-based)
MODEL_OF = {**{f"Y{i:02d}": 1 for i in (4, 5, 6)}, **{f"Y{i:02d}": 2 for i in (7, 8, 9)}, **{f"Y{i:02d}": 3 for i in (10, 11, 12)},
            **{f"Y{i:02d}": 4 for i in (13, 14, 15)}, **{f"Y{i:02d}": 5 for i in (16, 17)}, **{f"Y{i:02d}": 6 for i in (18, 19, 20)},
            **{f"Y{i:02d}": 7 for i in (21, 22, 23)}, **{f"Y{i:02d}": 8 for i in (24, 25, 26)}, **{f"Y{i:02d}": 9 for i in (27, 28, 29)},
            **{f"Y{i:02d}": 10 for i in (30, 31, 32)}, **{f"Y{i:02d}": 11 for i in (33, 34, 35)}, **{f"Y{i:02d}": 12 for i in (36, 37, 38)},
            **{f"Y{i:02d}": 13 for i in (39, 40, 41)}, **{f"Y{i:02d}": 14 for i in (42, 43)}, **{f"Y{i:02d}": 15 for i in (44, 45)}}
STAGE = {}
for _m in range(1, 16):
    _beats = sorted(k for k, v in MODEL_OF.items() if v == _m)
    _names = ["WHAT IT DOES", "HOW IT WORKS", "WHY · WHEN TO USE"] if len(_beats) == 3 else ["WHAT · HOW IT WORKS", "WHY · WHEN TO USE"]
    STAGE.update(dict(zip(_beats, _names)))
FIRST_BEAT = {"Y04", "Y07", "Y10", "Y13", "Y16", "Y18", "Y21", "Y24", "Y27", "Y30", "Y33", "Y36", "Y39", "Y42", "Y44"}


# ---------------------------------------------------------------- frame pieces
def model_header(sc, n, big=False):
    """Model n/15: name, data-type tag and a 15-step progress rail (consistent on every model scene)."""
    u, W, H = sc.u, sc.W, sc.H
    name, tag = MODELS[n - 1]
    x0, y0 = W * 0.05, H * 0.05
    rail = sc.id("rail"); dots = ""
    for i in range(15):
        dots += f'<i style="background:{AMB if i < n - 1 else (INK if i == n - 1 else "rgba(243,240,232,.18)")};{"width:" + str(30 * u) + "px;" if i == n - 1 else ""}"></i>'
    sc.add(f'<div id="{rail}" class="rail">{dots}</div>', f"#{rail}{{left:{x0}px;top:{y0}px;}}")
    sc.css.append(f"#root .rail{{position:absolute;display:flex;gap:{8 * u}px;align-items:center;z-index:9;}}#root .rail i{{display:block;width:{12 * u}px;height:{12 * u}px;border-radius:{6 * u}px;}}")
    plate = sc.id("hplate")
    sc.add(f'<div id="{plate}"></div>', f"#{plate}{{position:absolute;left:0;top:0;width:{W}px;height:{y0 + 110 * u}px;background:linear-gradient(#141517 0,#141517 75%,rgba(20,21,23,0));z-index:8;}}")
    hd = sc.id("mh")
    sc.add(f'<div id="{hd}" class="mh"><span>MODEL {n:02d}</span><b>{escape(name)}</b><em>{tag}</em></div>', f"#{hd}{{left:{x0}px;top:{y0 + 30 * u}px;}}")
    sc.css.append(f"#root .mh{{position:absolute;display:flex;align-items:baseline;gap:{18 * u}px;white-space:nowrap;z-index:9;}}#root .mh span{{font-family:'JetBrains Mono',monospace;color:{AMB};font-size:{26 * u}px;font-weight:700;}}"
                  f"#root .mh b{{font-size:{(52 if big else 40) * u}px;font-weight:900;}}#root .mh em{{font-style:normal;font-size:{22 * u}px;font-weight:800;color:#141517;background:{AMB};padding:{4 * u}px {12 * u}px;border-radius:{8 * u}px;}}")
    if sc.sid in FIRST_BEAT:
        sc.js.append(f'tl.fromTo("#{hd} b",{{opacity:0,x:{-30 * u}}},{{opacity:1,x:0,duration:.5,ease:"expo.out"}},0.05);')
    if sc.sid in STAGE:
        st = sc.id("stage")
        sc.add(f'<div id="{st}" class="stage">{STAGE[sc.sid]}</div>', f"#{st}{{right:{W * 0.05}px;top:{y0 + 30 * u}px;}}")
        sc.css.append(f"#root .stage{{position:absolute;z-index:9;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:{24 * u}px;letter-spacing:.12em;color:{GRN};"
                      f"border:{2 * u}px solid {GRN};padding:{6 * u}px {14 * u}px;border-radius:{8 * u}px;}}")
        sc.js.append(f'tl.fromTo("#{st}",{{opacity:0,x:{20 * u}}},{{opacity:1,x:0,duration:.4,ease:"expo.out"}},0.15);')


def tip(sc, name, good, bad, x, y, w, t):
    """'Use it when / avoid when' card."""
    u = sc.u; el = sc.id(name)
    sc.add(f'<div id="{el}" class="tipc"><p><b style="color:{GRN}">✓ USE WHEN</b>{good}</p><p><b style="color:{RED}">✗ AVOID WHEN</b>{bad}</p></div>', f"#{el}{{left:{x}px;top:{y}px;width:{w}px;}}")
    sc.css.append(f"#root .tipc{{position:absolute;padding:{22 * u}px {26 * u}px;{glass(sc)}}}#root .tipc p{{margin:0 0 {12 * u}px;font-size:{30 * u}px;line-height:1.35;color:#E8E5DD;}}"
                  f"#root .tipc b{{display:block;font-size:{22 * u}px;letter-spacing:.08em;margin-bottom:{4 * u}px;}}")
    sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,y:{30 * u}}},{{opacity:1,y:0,duration:.5,ease:"expo.out"}},{t:.2f});')


def example(sc, name, emoji, text, x, y, w, t, sub="everyday example"):
    u = sc.u; el = sc.id(name)
    sc.add(f'<div id="{el}" class="exa"><i>{emoji}</i><div><small>{sub}</small>{text}</div></div>', f"#{el}{{left:{x}px;top:{y}px;width:{w}px;}}")
    sc.css.append(f"#root .exa{{position:absolute;display:flex;gap:{22 * u}px;align-items:center;padding:{22 * u}px {28 * u}px;{glass(sc)}}}"
                  f"#root .exa i{{font-style:normal;font-size:{72 * u}px;}}#root .exa div{{font-size:{38 * u}px;font-weight:800;line-height:1.25;}}"
                  f"#root .exa small{{display:block;font-size:{22 * u}px;color:{AMB};letter-spacing:.1em;text-transform:uppercase;margin-bottom:{6 * u}px;}}")
    sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,x:{-40 * u}}},{{opacity:1,x:0,duration:.5,ease:"expo.out"}},{t:.2f});')


def formula(sc, name, html, x, y, w, t, size=40):
    u = sc.u; el = sc.id(name)
    sc.add(f'<div id="{el}" class="fml">{html}</div>', f"#{el}{{left:{x}px;top:{y}px;width:{w}px;}}")
    sc.css.append(f"#root .fml{{position:absolute;padding:{18 * u}px {26 * u}px;font-family:'JetBrains Mono',monospace;font-size:{size * u}px;color:{INK};line-height:1.45;{glass(sc)}}}"
                  f"#root .fml small{{display:block;font-family:Inter,sans-serif;font-size:{24 * u}px;color:{MUTE};}}")
    sc.appear(el, t)


def scatter(sc, bx, X, Y, xs, ys, cols, t, r=5, stagger=0.0, op=0.8):
    groups = {}
    for a_, b_, c_ in zip(xs, ys, cols):
        groups.setdefault(c_, []).append(f'<circle cx="{X(a_):.1f}" cy="{Y(b_):.1f}" r="{r * sc.u:.1f}"/>')
    for j, (c_, pts) in enumerate(groups.items()):
        g = sc.id(f"sg{len(sc.html)}_{j}")
        layer(sc, bx, f'<g id="{g}" fill="{c_}" fill-opacity="{op}">{"".join(pts)}</g>')
        sc.js.append(f'tl.fromTo("#{g}",{{opacity:0}},{{opacity:1,duration:.5}},{t + j * stagger:.2f});')


def grid_img(sc, name, M, x, y, cell, t, cmap="gray", vmin=None, vmax=None, gap=0):
    """A small real matrix (image, filter, feature map) as coloured cells."""
    u = sc.u; el = sc.id(name)
    flat = [v for row in M for v in row]; lo = min(flat) if vmin is None else vmin; hi = max(flat) if vmax is None else vmax
    rows = ""
    for row in M:
        for v in row:
            q = 0 if hi == lo else (v - lo) / (hi - lo)
            if cmap == "gray":
                c = int(20 + 230 * q); col = f"rgb({c},{c},{c})"
            else:   # diverging: red negative, amber positive
                z = (v / max(abs(lo), abs(hi), 1e-9)); col = f"rgba(255,107,107,{-z:.2f})" if z < 0 else f"rgba(255,178,36,{z:.2f})"
            rows += f'<i style="background:{col}"></i>'
    n = len(M[0])
    sc.add(f'<div id="{el}" class="gim">{rows}</div>', f"#{el}{{left:{x}px;top:{y}px;width:{n * (cell + gap)}px;grid-template-columns:repeat({n},{cell}px);gap:{gap}px;}}")
    sc.css.append("#root .gim{position:absolute;display:grid;background:#141517;}#root .gim i{display:block;aspect-ratio:1;}")
    sc.appear(el, t, y=10)
    return el


def tree_diagram(sc, x, y, w, h, t0, step=0.35):
    """The real depth-3 tree from the notebook, drawn node by node."""
    u = sc.u; T = V["tree"]
    pos, order = {}, []
    def walk(i, d, lo, hi):
        pos[i] = (x + (lo + hi) / 2 * w, y + d * h / 3.4); order.append(i)
        if T["left"][i] != -1:
            mid = (lo + hi) / 2; walk(T["left"][i], d + 1, lo, mid); walk(T["right"][i], d + 1, mid, hi)
    walk(0, 0, 0, 1)
    lines = "".join(f'<line x1="{pos[i][0]:.1f}" y1="{pos[i][1]:.1f}" x2="{pos[c][0]:.1f}" y2="{pos[c][1]:.1f}" stroke="rgba(243,240,232,.35)" stroke-width="{3 * u}"/>'
                    for i in order for c in (T["left"][i], T["right"][i]) if c != -1)
    layer(sc, (0, 0, sc.W, sc.H), f'<g id="{sc.id("tl")}">{lines}</g>')
    sc.js.append(f'tl.fromTo("#{sc.id("tl")}",{{opacity:0}},{{opacity:1,duration:.6}},{t0:.2f});')
    for k, i in enumerate(order):
        el = sc.id(f"tn{i}")
        leaf = T["left"][i] == -1
        mal, ben = T["value"][i]
        if leaf:
            win = "benign" if ben >= mal else "malignant"
            txt = f'<b style="color:{GRN if win == "benign" else RED}">{win}</b><small>{ben + mal} tumours</small>'
        else:
            txt = f'<b>{escape(T["feature"][i])}<br>≤ {T["threshold"][i]:.2f}?</b><small>{T["samples"][i]} tumours</small>'
        sc.add(f'<div id="{el}" class="tnode{" leaf" if leaf else ""}">{txt}</div>', f"#{el}{{left:{pos[i][0]}px;top:{pos[i][1]}px;}}")
        sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,scale:.7,xPercent:-50,yPercent:-50}},{{opacity:1,scale:1,xPercent:-50,yPercent:-50,duration:.35,ease:"back.out(1.8)"}},{t0 + k * step:.2f});')
    sc.css.append(f"#root .tnode{{position:absolute;z-index:7;text-align:center;padding:{10 * u}px {14 * u}px;border-radius:{12 * u}px;background:#25262A;border:{2 * u}px solid rgba(243,240,232,.25);"
                  f"font-size:{22 * u}px;white-space:nowrap;}}#root .tnode b{{display:block;font-size:{22 * u}px;}}#root .tnode small{{display:block;color:#B8B5AE;font-size:{18 * u}px;}}"
                  f"#root .tnode.leaf{{border-color:rgba(243,240,232,.12);}}")
    return pos


def knn_map(sc, x, y, s, t, stride=2, regions=True, t_pts=None):
    """Real k-NN decision regions (best k) with the real wines on top."""
    u = sc.u; K = V["knn"]; G = K["grid"][::stride]; G = [row[::stride] for row in G]
    n = len(G); cell = s / n; rects = []
    if not regions: G = []
    for i, row in enumerate(G):
        for j, c in enumerate(row):
            rects.append(f'<rect x="{x + j * cell:.1f}" y="{y + s - (i + 1) * cell:.1f}" width="{cell + .6:.1f}" height="{cell + .6:.1f}" fill="{CLS[c]}" fill-opacity=".22"/>')
    if rects:
        g = sc.id("kg"); layer(sc, (0, 0, sc.W, sc.H), f'<g id="{g}">{"".join(rects)}</g>')
        sc.js.append(f'tl.fromTo("#{g}",{{opacity:0}},{{opacity:1,duration:.8}},{t:.2f});')
    L_ = K["lim"]; X = lambda v: x + (v + L_) / (2 * L_) * s; Y = lambda v: y + s - (v + L_) / (2 * L_) * s
    scatter(sc, (0, 0, sc.W, sc.H), X, Y, K["x"], K["y"], [CLS[c] for c in K["c"]], t - 0.6 if t_pts is None else t_pts, r=6, op=.95)
    return X, Y


def network(sc, x, y, w, h, t, sizes=(8, 8, 10), labels=("64 inputs", "64 hidden", "10 outputs")):
    u = sc.u; cols = []
    for li, n in enumerate(sizes):
        cx = x + li * w / (len(sizes) - 1); cols.append([(cx, y + (k + .5) * h / n) for k in range(n)])
    lines = "".join(f'<line x1="{a[0]:.0f}" y1="{a[1]:.0f}" x2="{c[0]:.0f}" y2="{c[1]:.0f}" stroke="rgba(110,155,255,.18)" stroke-width="{1.2 * u}"/>'
                    for L1, L2 in zip(cols, cols[1:]) for a in L1 for c in L2)
    nodes = "".join(f'<circle cx="{p[0]:.0f}" cy="{p[1]:.0f}" r="{9 * u}" fill="{[BLU, AMB, GRN][li]}"/>' for li, L1 in enumerate(cols) for p in L1)
    g = sc.id("net"); layer(sc, (0, 0, sc.W, sc.H), f'<g id="{g}">{lines}{nodes}</g>')
    sc.js.append(f'tl.fromTo("#{g}",{{opacity:0,scale:.95,transformOrigin:"50% 50%"}},{{opacity:1,scale:1,duration:.6}},{t:.2f});')
    for li, lab in enumerate(labels):
        label(sc, f"nl{li}", lab, x + li * w / (len(sizes) - 1), y + h + 16 * u, size=24, color=MUTE, anchor="center", t=t + .2)
    return cols


def pulse(sc, name, x0, y0, x1, y1, t, d=1.0, color=AMB, n=3, gap=0.35):
    u = sc.u
    for k in range(n):
        el = sc.id(f"{name}{k}")
        sc.add(f'<div id="{el}" class="pls" style="background:{color};box-shadow:0 0 {18 * u}px {color}"></div>', f"#{el}{{left:{x0}px;top:{y0}px;}}")
        sc.js.append(f'tl.fromTo("#{el}",{{x:0,y:0,opacity:1}},{{x:{x1 - x0},y:{y1 - y0},duration:{d},ease:"none"}},{t + k * gap:.2f});tl.to("#{el}",{{opacity:0,duration:.1}},{t + k * gap + d:.2f});')
    sc.css.append(f"#root .pls{{position:absolute;width:{16 * u}px;height:{16 * u}px;margin:{-8 * u}px 0 0 {-8 * u}px;border-radius:50%;opacity:0;z-index:7;}}")



# ---------------------------------------------------------------- LIVE: one real example moving through each model
def frames(sc, name, texts, x, y, t0, dt, size=30, color=INK, w=None, mono=True):
    """A live read-out: one text per frame, each shown only during its slot (deterministic, no callbacks)."""
    u = sc.u
    for i, tx in enumerate(texts):
        el = sc.id(f"{name}{i}")
        sc.add(f'<div id="{el}" class="frm">{tx}</div>', f"#{el}{{left:{x}px;top:{y}px;{f'width:{w}px;' if w else ''}}}")
        sc.js.append(f'tl.set("#{el}",{{opacity:1}},{t0 + i * dt:.2f});')
        if i < len(texts) - 1:
            sc.js.append(f'tl.set("#{el}",{{opacity:0}},{t0 + (i + 1) * dt:.2f});')
    sc.css.append(f"#root .frm{{position:absolute;opacity:0;font-size:{size * u}px;font-weight:800;color:{color};{'font-family:JetBrains Mono,monospace;' if mono else ''}white-space:nowrap;}}")


def live_gd(sc, bx, X, Y, t0, dur, readout=None):
    """The line moving through the real gradient-descent steps (w, b from the notebook's loop)."""
    u = sc.u; G = LV["gd"]; xa, xb = -2.2, 4.2
    ln = sc.id("gdl")
    layer(sc, bx, f'<line id="{ln}" x1="{X(xa):.1f}" y1="{Y(G[0][0] * xa + G[0][1]):.1f}" x2="{X(xb):.1f}" y2="{Y(G[0][0] * xb + G[0][1]):.1f}" stroke="{AMB}" stroke-width="{6 * u}" stroke-linecap="round"/>')
    sc.js.append(f'tl.fromTo("#{ln}",{{opacity:0}},{{opacity:1,duration:.3}},{t0 - .3:.2f});')
    pick = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 59]; dt = dur / len(pick)
    for i, k in enumerate(pick[1:], 1):
        w_, b_ = G[k][0], G[k][1]
        sc.js.append(f'tl.to("#{ln}",{{attr:{{y1:{Y(w_ * xa + b_):.1f},y2:{Y(w_ * xb + b_):.1f}}},duration:{dt:.2f},ease:"none"}},{t0 + (i - 1) * dt:.2f});')
    if readout:
        frames(sc, "gdr", [f"step {k:>2} · w = {G[k][0]:.3f} · b = {G[k][1]:.3f} · error {G[k][2]:.3f}" for k in pick], readout[0], readout[1], t0, dt, size=28, color=AMB)
    return dt * len(pick)


def live_sigmoid(sc, bx, X, Y, t0, at_xy=None):
    """Two real test tumours: their weighted sum z slides along the axis, then rises to the curve."""
    u = sc.u
    for j, ex in enumerate(LV["logit"]):
        col = GRN if ex["true"] == 1 else RED; d_ = sc.id(f"ld{j}"); tt = t0 + j * 2.2
        layer(sc, bx, f'<circle id="{d_}" cx="{X(0):.1f}" cy="{Y(0):.1f}" r="{14 * u}" fill="{col}" stroke="#F3F0E8" stroke-width="{3 * u}"/>')
        sc.js.append(f'tl.fromTo("#{d_}",{{opacity:0}},{{opacity:1,duration:.2}},{tt:.2f});'
                     f'tl.to("#{d_}",{{attr:{{cx:{X(ex["z"]):.1f}}},duration:.8,ease:"power2.inOut"}},{tt + .2:.2f});'
                     f'tl.to("#{d_}",{{attr:{{cy:{Y(ex["p"]):.1f}}},duration:.7,ease:"power2.out"}},{tt + 1.0:.2f});')
        lb = sc.id(f"lt{j}")
        sc.add(f'<div id="{lb}" class="lgt" style="color:{col}">z = {ex["z"]:+.2f} → P(benign) = {ex["p"]:.3f} → {"BENIGN" if ex["p"] > .5 else "MALIGNANT"}</div>',
               f"#{lb}{{left:{(at_xy or bx)[0] + 20 * u}px;top:{(at_xy or bx)[1] + 20 * u + j * 48 * u}px;}}")
        sc.appear(lb, tt + 1.6, y=0, d=.3)
    sc.css.append(f"#root .lgt{{position:absolute;z-index:6;font-family:'JetBrains Mono',monospace;font-weight:800;font-size:{26 * u}px;background:rgba(20,21,23,.85);padding:{4 * u}px {10 * u}px;border-radius:{6 * u}px;}}")


def live_knn(sc, X, Y, t0, tally_xy):
    """A real test wine: lines to its 5 real nearest neighbours, then the vote."""
    u = sc.u; K = LV["knn"]; qx, qy = X(K["q"][0]), Y(K["q"][1])
    lines = "".join(f'<line x1="{qx:.1f}" y1="{qy:.1f}" x2="{X(p_[0]):.1f}" y2="{Y(p_[1]):.1f}" stroke="{CLS[c_]}" stroke-width="{4 * u}" stroke-dasharray="1 1" pathLength="1" stroke-dashoffset="1"/>'
                    for p_, c_ in zip(K["nb"], K["nb_c"]))
    g = sc.id("kl"); layer(sc, (0, 0, sc.W, sc.H), f'<g id="{g}">{lines}<circle cx="{qx:.1f}" cy="{qy:.1f}" r="{16 * u}" fill="#F3F0E8"/></g>')
    sc.js.append(f'tl.fromTo("#{g}",{{opacity:0}},{{opacity:1,duration:.2}},{t0:.2f});tl.to("#{g} line",{{strokeDashoffset:0,duration:.5,stagger:.25}},{t0 + .2:.2f});')
    votes = {}
    for c_ in K["nb_c"]: votes[c_] = votes.get(c_, 0) + 1
    win = max(votes, key=votes.get)
    txt = " · ".join(f'<span style="color:{CLS[c_]}">{n_}× cultivar {c_ + 1}</span>' for c_, n_ in sorted(votes.items(), key=lambda z_: -z_[1]))
    el = sc.id("tal"); sc.add(f'<div id="{el}" class="tal">5 nearest vote: {txt} → <b style="color:{CLS[win]}">cultivar {win + 1}</b></div>', f"#{el}{{left:{tally_xy[0]}px;top:{tally_xy[1]}px;}}")
    sc.css.append(f"#root .tal{{position:absolute;font-size:{30 * u}px;font-weight:800;max-width:{sc.W * .42}px;line-height:1.4;}}")
    sc.appear(el, t0 + 1.6, y=10)


def live_tree_path(sc, pos, t0, dt=1.1):
    """One real tumour travelling down the real tree, answering each question."""
    u = sc.u; P_ = LV["tree_path"]; nodes = [p_["node"] for p_ in P_["path"]] + [P_["leaf"]]
    tok = sc.id("tok")
    sc.add(f'<div id="{tok}" class="tok">🔬</div>', f"#{tok}{{left:{pos[nodes[0]][0]}px;top:{pos[nodes[0]][1] - 70 * u}px;}}")
    sc.css.append(f"#root .tok{{position:absolute;z-index:8;font-size:{44 * u}px;width:{60 * u}px;height:{60 * u}px;margin:{-30 * u}px 0 0 {-30 * u}px;text-align:center;line-height:{60 * u}px;}}")
    sc.appear(tok, t0, y=0, d=.2)
    for i, n_ in enumerate(nodes[1:], 1):
        sc.js.append(f'tl.to("#{tok}",{{x:{pos[n_][0] - pos[nodes[0]][0]:.1f},y:{pos[n_][1] - pos[nodes[0]][1]:.1f},duration:{dt * .6:.2f},ease:"power2.inOut"}},{t0 + i * dt - dt * .6:.2f});')
    for i, n_ in enumerate(nodes):
        sc.js.append(f'tl.to("#{sc.id(f"tn{n_}")}",{{borderColor:"{AMB}",backgroundColor:"#4C3F29",duration:.2}},{t0 + i * dt:.2f});')
    steps = [f'{p_["feature"]} = {p_["value"]} {"≤" if p_["left"] else ">"} {p_["thr"]}  →  {"left" if p_["left"] else "right"}' for p_ in P_["path"]]
    steps.append(f'answer: {"MALIGNANT" if P_["pred"] == 0 else "BENIGN"}  (true: {"malignant" if P_["true"] == 0 else "benign"}) ✓')
    return steps


def live_votes(sc, x, y, w, t0, dur):
    """The real 300 votes for one tumour, filling in one by one."""
    u = sc.u; R_ = LV["rf_votes"]; cell = (w - 29 * 3 * u) / 30
    sq = "".join(f'<i style="background:{GRN if v_ == 1 else RED}"></i>' for v_ in R_["order"])
    g = sc.id("vg"); sc.add(f'<div id="{g}" class="vgrid">{sq}</div>', f"#{g}{{left:{x}px;top:{y}px;width:{w}px;grid-template-columns:repeat(30,{cell:.1f}px);}}")
    sc.css.append(f"#root .vgrid{{position:absolute;display:grid;gap:{3 * u}px;}}#root .vgrid i{{display:block;height:{cell:.1f}px;border-radius:{2 * u}px;opacity:0;}}")
    sc.js.append(f'tl.to("#{g} i",{{opacity:1,duration:.05,stagger:{dur / 300:.4f}}},{t0:.2f});')
    c1 = label(sc, "vb", "0 benign", x, y + 10 * (cell + 3 * u) + 20 * u, size=44, color=GRN, weight=900, t=t0)
    c2 = label(sc, "vm", "0 malignant", x + w * 0.5, y + 10 * (cell + 3 * u) + 20 * u, size=44, color=RED, weight=900, t=t0)
    countup(sc, c1, R_["benign"], t0, d=dur, sel="div", suffix=" benign"); countup(sc, c2, R_["malignant"], t0, d=dur, sel="div", suffix=" malignant")
    return R_


def live_house(sc, box, t0, dur):
    """One real house: LightGBM's prediction after 1, 5, 10 … 600 trees, against the true value."""
    u = sc.u; Hh = LV["gbm_house"]
    X, Y, bx = chart(sc, "hs", box, (0.5, len(Hh["stages"]) + .5), (0, 5.6), "trees used", "predicted value (×$100k)",
                     ([], [0, 1, 2, 3, 4, 5], lambda v: f"{v:g}"), t0=max(.1, t0 - .5))
    curve(sc, bx, "tru", X, Y, [.5, len(Hh["stages"]) + .5], [Hh["true"]] * 2, GRN, t0, dash=f"{12 * u} {8 * u}", width=4)
    dt = dur / len(Hh["stages"])
    bars = ""
    for i, (s_, p_) in enumerate(zip(Hh["stages"], Hh["pred"]), 1):
        bars += f'<rect id="{sc.id(f"hb{i}")}" x="{X(i) - 26 * u:.1f}" y="{Y(p_):.1f}" width="{52 * u}" height="{Y(0) - Y(p_):.1f}" fill="{AMB}" opacity="0"/>'
        bars += f'<text id="{sc.id(f"ht{i}")}" x="{X(i):.1f}" y="{Y(0) + 34 * u:.1f}" text-anchor="middle" font-size="{22 * u}" fill="#B8B5AE" font-family="JetBrains Mono" opacity="0">{s_}</text>'
    g = sc.id("hbars"); layer(sc, bx, f'<g id="{g}">{bars}</g>')
    for i in range(1, len(Hh["stages"]) + 1):
        sc.js.append(f'tl.to(["#{sc.id(f"hb{i}")}","#{sc.id(f"ht{i}")}"],{{opacity:1,duration:.25}},{t0 + (i - 1) * dt:.2f});')
    frames(sc, "hr", [f"after {s_:>3} trees: ${p_ * 100:,.0f}k   (true ${Hh['true'] * 100:,.0f}k)" for s_, p_ in zip(Hh["stages"], Hh["pred"])],
           box[0], box[1] + box[3] + 16 * u, t0, dt, size=30, color=AMB)


def live_nb(sc, x, y, w, t0):
    """A real test post: the words that pushed it towards rec.autos, then the probabilities."""
    u = sc.u; N_ = LV["nb_post"]
    rows = [(f"“{w_}”", c_, GRN if c_ > 0 else RED, t0 + i * .35, False) for i, (w_, c_) in enumerate(N_["words"])]
    label(sc, "nbh", f"one real test post ({N_['n_words']} words) · how much each word favours rec.autos (log-probability)", x, y - 34 * u, size=22, color=MUTE, t=t0 - .3, w=w)
    hbars(sc, "nbw", [(a_, abs(c_), col, t_, s_) for a_, c_, col, t_, s_ in rows], (x, y, w, 0), 7.0, fmt="{:.2f}", rowh=60, lab_w=.32)
    top3 = sorted(N_["proba"].items(), key=lambda z_: -z_[1])[:3]
    meters(sc, "nbp", [(k_, v_, t0 + 2.6 + i * .25) for i, (k_, v_) in enumerate(top3)], x + w + 70 * u, y + 40 * u, sc.W * 0.26, fmt="{:.1%}", rowh=90)


def live_hidden(sc, x, y, cell, t0):
    u = sc.u; A = LV["mlp_hidden"]
    sq = "".join(f'<i style="background:rgba(255,178,36,{a_:.2f})"></i>' for a_ in A)
    g = sc.id("hid"); sc.add(f'<div id="{g}" class="hgrid">{sq}</div>', f"#{g}{{left:{x}px;top:{y}px;grid-template-columns:repeat(8,{cell}px);}}")
    sc.css.append(f"#root .hgrid{{position:absolute;display:grid;gap:{4 * u}px;}}#root .hgrid i{{display:block;height:{cell}px;border-radius:50%;border:{1.5 * u}px solid rgba(243,240,232,.25);opacity:0;}}")
    sc.js.append(f'tl.to("#{g} i",{{opacity:1,duration:.15,stagger:.02}},{t0:.2f});')


# ---------------------------------------------------------------- visuals shared by YouTube and the Reel
def v_linreg(sc, box, t_pts, t_line, loss_box=None, t_loss=0, live=None):
    L_ = V["linreg"]
    X, Y, bx = chart(sc, "lr", box, (-2.2, 4.2), (0, 5.2), "median income (standardised)", "house value (×$100k)",
                     ([-2, 0, 2, 4], [0, 1, 2, 3, 4, 5], lambda v: f"{v:g}"), t0=max(.1, t_pts - .5))
    scatter(sc, bx, X, Y, L_["x"], L_["y"], [BLU] * len(L_["x"]), t_pts, r=4, op=.5)
    if live:   # (t0, duration, readout position)
        live_gd(sc, bx, X, Y, live[0], live[1], live[2])
    else:
        curve(sc, bx, "fit", X, Y, [-2.2, 4.2], [L_["w"] * -2.2 + L_["b"], L_["w"] * 4.2 + L_["b"]], AMB, t_line, d=1.0, width=6)
    if loss_box:
        X2, Y2, bx2 = chart(sc, "ls", loss_box, (0, 60), (0, 6), "gradient-descent step", "error (MSE)", ([0, 20, 40, 60], [0, 2, 4, 6], lambda v: f"{v:g}"), t0=t_loss - .4)
        curve(sc, bx2, "lc", X2, Y2, list(range(60)), L_["loss"], RED, t_loss, d=(live[1] if live else 1.6), width=5)


def v_sigmoid(sc, box, t):
    L_ = V["logreg"]
    X, Y, bx = chart(sc, "sg", box, (-12, 12), (-0.05, 1.05), "weighted sum z = w·x + b", "P(benign)", ([-10, -5, 0, 5, 10], [0, .5, 1], lambda v: f"{v:g}"), t0=max(.1, t - .5))
    zs = [z / 10 for z in range(-120, 121)]
    curve(sc, bx, "sig", X, Y, zs, [1 / (1 + math.exp(-z)) for z in zs], AMB, t, d=1.2, width=6)
    curve(sc, bx, "half", X, Y, [-12, 12], [.5, .5], MUTE, t + .4, dash=f"{10 * sc.u} {8 * sc.u}", width=3)
    zc = [max(-12, min(12, z)) for z in L_["z"]]
    scatter(sc, bx, X, Y, zc, L_["p"], [GRN if y_ == 1 else RED for y_ in L_["y"]], t + 1.0, r=6, op=.85)


def v_gbm(sc, box, t):
    G = V["gbm"]
    X, Y, bx = chart(sc, "gb", box, (0, 600), (0.3, 1.2), "number of trees", "test error (RMSE)", ([0, 200, 400, 600], [.4, .8, 1.2], lambda v: f"{v:g}"), t0=max(.1, t - .5))
    curve(sc, bx, "gc", X, Y, [i * 5 for i in range(len(G["curve"]))], G["curve"], GRN, t, d=2.0, width=6)


def cluster_map():
    """Give each cluster the colour of its majority real type (does not change the score)."""
    K = V["kmeans"]; last = K["steps"][-1]["labels"]; m = {}
    for c in set(last):
        votes = [t for l_, t in zip(last, K["true"]) if l_ == c]; m[c] = max(set(votes), key=votes.count)
    return m


def v_kmeans(sc, box, t0, dt=1.0, show_true_at=None):
    K = V["kmeans"]; u = sc.u; cm = cluster_map()
    xs, ys = K["x"], K["y"]
    X, Y, bx = chart(sc, "km", box, (min(xs) - .5, max(xs) + .5), (min(ys) - .5, max(ys) + .5), "PCA 1", "PCA 2", ([], [], lambda v: ""), t0=max(.1, t0 - .6))
    grey = sc.id("kgrey"); layer(sc, bx, f'<g id="{grey}" fill="#B8B5AE" fill-opacity=".6">' + "".join(f'<circle cx="{X(a_):.1f}" cy="{Y(b_):.1f}" r="{6 * u:.1f}"/>' for a_, b_ in zip(xs, ys)) + "</g>")
    sc.js.append(f'tl.fromTo("#{grey}",{{opacity:0}},{{opacity:1,duration:.4}},{max(.1, t0 - .4):.2f});')
    for si, st in enumerate(K["steps"]):
        g = sc.id(f"ks{si}")
        pts = "".join(f'<circle cx="{X(a_):.1f}" cy="{Y(b_):.1f}" r="{6 * u:.1f}" fill="{CLS[cm.get(l_, l_)]}"/>' for a_, b_, l_ in zip(xs, ys, st["labels"]))
        cen = "".join(f'<g transform="translate({X(c[0]):.1f},{Y(c[1]):.1f})"><circle r="{20 * u}" fill="none" stroke="#F3F0E8" stroke-width="{4 * u}"/>'
                      f'<path d="M{-12 * u} 0H{12 * u}M0 {-12 * u}V{12 * u}" stroke="{CLS[cm.get(i, i)]}" stroke-width="{6 * u}"/></g>' for i, c in enumerate(st["centers"]))
        layer(sc, bx, f'<g id="{g}" fill-opacity=".85">{pts}{cen}</g>')
        t = t0 + si * dt
        sc.js.append(f'tl.fromTo("#{g}",{{opacity:0}},{{opacity:1,duration:.3}},{t:.2f});')
        if si < len(K["steps"]) - 1:
            sc.js.append(f'tl.to("#{g}",{{opacity:0,duration:.3}},{t + dt:.2f});')
        if si == 0:
            sc.js.append(f'tl.to("#{grey}",{{opacity:0,duration:.3}},{t:.2f});')
    return bx


def v_cnn_filters(sc, x, y, cell, t):
    u = sc.u
    for i, F in enumerate(V["cnn"]["filters"]):
        grid_img(sc, f"f{i}", F, x + (i % 4) * (5 * cell + 18 * u), y + (i // 4) * (5 * cell + 18 * u), cell, t + i * 0.12, cmap="div", gap=1)


def v_lstm(sc, box, t):
    Ls = V["lstm"]; n = len(Ls["actual"])
    lo, hi = min(Ls["actual"] + Ls["fc"]) - 2, max(Ls["actual"] + Ls["fc"]) + 2
    ticks_y = [v for v in range(int(lo) // 5 * 5 + 5, int(hi) + 1, 5)]
    X, Y, bx = chart(sc, "ls", box, (0, n - 1), (lo, hi), f"month ({Ls['dates'][0][:4]}–{Ls['dates'][-1][:4]})", "CO₂ (ppm)",
                     ([0, 30, 60, 90, 119], ticks_y, lambda v: f"{v:g}"), t0=max(.1, t - .5))
    curve(sc, bx, "act", X, Y, list(range(n)), Ls["actual"], "#F3F0E8", t, d=1.6, width=4)
    xs = list(range(n - 36, n))
    curve(sc, bx, "nv", X, Y, xs, Ls["naive"], GRN, t + 1.8, d=1.0, width=5)
    curve(sc, bx, "fc", X, Y, xs, Ls["fc"], RED, t + 2.6, d=1.0, width=5)
    curve(sc, bx, "cut", X, Y, [n - 36, n - 36], [lo, hi], MUTE, t + 1.4, dash=f"{8 * sc.u} {8 * sc.u}", width=3)


def v_tokens_attention(sc, x, y, w, t, small=False):
    u = sc.u; sc.css.append(FONT)
    S = V["sg"]; pcs, att = S["attention"]["pieces"], S["attention"]["cls_to_token"]; mx = max(att)
    colw = w / len(pcs); bh = (160 if not small else 200) * u
    for i, (pc, a_) in enumerate(zip(pcs, att)):
        bar, lb = sc.id(f"ab{i}"), sc.id(f"al{i}")
        hh = max(3, bh * a_ / mx); col = RED if a_ / mx > .8 else (AMB if a_ / mx > .55 else "rgba(243,240,232,.35)")
        sc.add(f'<div id="{bar}" class="ab"></div><div id="{lb}" class="al">{escape(pc.replace("▁", ""))}</div>',
               f"#{bar}{{left:{x + i * colw + colw * .15}px;top:{y + bh - hh}px;width:{colw * .7}px;height:{hh}px;background:{col};}}#{lb}{{left:{x + i * colw}px;top:{y + bh + 8 * u}px;width:{colw}px;}}")
        sc.js.append(f'tl.fromTo("#{bar}",{{scaleY:0}},{{scaleY:1,duration:.5,ease:"expo.out"}},{t + i * .03:.2f});')
        sc.appear(lb, t, y=0, d=.2)
    sc.css.append(f"#root .ab{{position:absolute;border-radius:{4 * u}px {4 * u}px 0 0;transform-origin:50% 100%;}}#root .al{{position:absolute;text-align:center;"
                  f"font-family:'NotoBN','JetBrains Mono',monospace;font-size:{(18 if not small else 17) * u}px;color:#D9D6CE;{'writing-mode:vertical-rl;' if small else ''}}}")


# ---------------------------------------------------------------- scenes
def S_ML(sid, W, H):
    sc = b.Scene(sid, W, H); u, P = sc.u, sc.P
    L = W * (0.05 if not P else 0.05); d = b.dur(sid)
    if sid in MODEL_OF:
        model_header(sc, MODEL_OF[sid])
    top = H * 0.17
    if sid == "Y01":      # hook: a wall of model names, the 15 light up
        aura(sc, (AMB, BLU))
        names = [m[0] for m in MODELS] + ["Lasso", "Ridge", "ElasticNet", "AdaBoost", "CatBoost", "XGBoost", "DBSCAN", "GMM", "t-SNE", "UMAP", "GRU", "ViT", "BERT", "GAN", "VAE", "Diffusion", "Q-learning", "HMM", "ARIMA", "Prophet", "Isolation Forest"]
        import random; rnd = random.Random(3); rnd.shuffle(names)
        for i, n in enumerate(names):
            el = sc.id(f"w{i}"); core = n in [m[0] for m in MODELS]
            sc.add(f'<div id="{el}" class="wn{" core" if core else ""}">{escape(n)}</div>', f"#{el}{{left:{W * 0.03 + (i % 6) * W * 0.16}px;top:{H * 0.12 + (i // 6) * H * 0.105}px;}}")
            sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,y:{10 * u}}},{{opacity:1,y:0,duration:.3}},{.05 + i * .03:.2f});')
        sc.js.append(f'tl.to("#root .wn:not(.core)",{{opacity:.12,duration:.5}},{at(sid, "fifteen", default=4):.2f});tl.to("#root .wn.core",{{color:"{AMB}",borderColor:"{AMB}",scale:1.06,duration:.4}},{at(sid, "fifteen", default=4):.2f});')
        sc.css.append(f"#root .wn{{position:absolute;padding:{8 * u}px {14 * u}px;border:{2 * u}px solid rgba(243,240,232,.2);border-radius:{12 * u}px;font-size:{22 * u}px;font-weight:800;white-space:nowrap;}}")
    elif sid == "Y02":    # every model is a function
        chapter(sc, "THE BIG IDEA")
        bw = W * 0.24
        for i, (t_, sub, tok) in enumerate([("data in", "x: numbers, words, pixels", "data"), ("model f(x; θ)", "θ = parameters (the knobs)", "inside"), ("prediction ŷ", "a number, a class, a probability", "prediction")]):
            card(sc, f"c{i}", t_, sub, L + i * (bw + W * 0.07), H * 0.24, bw, at(sid, tok, default=1 + i), AMB if i == 1 else INK, big=46)
        pulse(sc, "p", L + bw * 0.5, H * 0.24 + 70 * u, L + 2 * (bw + W * 0.07) + bw * .5, H * 0.24 + 70 * u, at(sid, "comes", default=3), d=1.6)
        formula(sc, "loss", "error = mean( (ŷ − y)² )<small>training = change θ a little, again and again, so the error goes down</small>", L, H * 0.52, W * 0.62, at(sid, "training", default=7))
        el = sc.id("loop"); sc.add(f'<div id="{el}" class="loop">↻</div>', f"#{el}{{left:{L + W * 0.70}px;top:{H * 0.48}px;}}")
        sc.css.append(f"#root .loop{{position:absolute;font-size:{180 * u}px;color:{AMB};}}")
        sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,rotation:-180}},{{opacity:1,rotation:0,duration:.8,ease:"expo.out"}},{at(sid, "again", default=9):.2f});')
    elif sid == "Y03":    # the data map
        chapter(sc, "WHICH DATA DO YOU HAVE?")
        types = [("🧮", "Numbers in a table", "price, age, income", "table"), ("📝", "Text", "SMS, reviews, posts", "text"), ("🖼️", "Images", "pixels", "images"),
                 ("📈", "Sequences", "values over time", "change"), ("🏷️", "No labels", "no right answers given", "labels")]
        cw = (W - 2 * L - 4 * 24 * u) / 5
        for i, (em, t_, sub, tok) in enumerate(types):
            el = sc.id(f"t{i}")
            sc.add(f'<div id="{el}" class="dt"><i>{em}</i><b>{t_}</b><span>{sub}</span></div>', f"#{el}{{left:{L + i * (cw + 24 * u)}px;top:{H * 0.30}px;width:{cw}px;}}")
            sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,y:{40 * u}}},{{opacity:1,y:0,duration:.5,ease:"back.out(1.6)"}},{at(sid, tok, default=2 + i):.2f});')
        sc.css.append(f"#root .dt{{position:absolute;text-align:center;padding:{30 * u}px {16 * u}px;{glass(sc)}}}#root .dt i{{display:block;font-style:normal;font-size:{84 * u}px;}}"
                      f"#root .dt b{{display:block;font-size:{34 * u}px;margin-top:{12 * u}px;}}#root .dt span{{display:block;font-size:{24 * u}px;color:{MUTE};margin-top:{6 * u}px;}}")
        label(sc, "k", "15 models · each one on this map · each one on real data", W / 2, H * 0.66, size=40, color=AMB, weight=900, anchor="center", t=at(sid, "map", default=10))
    # ---------- 1 linear regression
    elif sid == "Y04":
        example(sc, "ex", "🏠", "Guess a house price<br>from the owner's income", L, top + 40 * u, W * 0.36, 0.4)
        v_linreg(sc, (W * 0.44, top, W * 0.52, H * 0.62), at(sid, "income", default=4), at(sid, "straight", default=8))
        formula(sc, "f", "ŷ = w · x + b", L, top + 260 * u, W * 0.30, at(sid, "line", default=9), size=48)
    elif sid == "Y05":
        tg = at(sid, "random", default=2)
        v_linreg(sc, (L, top, W * 0.44, H * 0.58), 0.2, 0.6, loss_box=(W * 0.52, top, W * 0.43, H * 0.58), t_loss=tg,
                 live=(tg, max(6.0, at(sid, "sixty", default=10) - tg + 2.5), (L, H * 0.79)))
        label(sc, "gd", "w ← w − learning rate × ∂error/∂w", W * 0.52, H * 0.79, size=30, mono=True, color=AMB, t=at(sid, "gradient", default=7))
    elif sid == "Y06":
        capture(sc, [seg(3, 0.2, d * 0.55, [("R² 0.576", at(sid, "58", default=4)), ("MAE $53,320", at(sid, "53", default=6))], zoom=2.0)])
        tip(sc, "tip", "you predict a number and want a fast, explainable first baseline", "the relationship is curved or full of interactions", W * 0.18, H * 0.30, W * 0.64, d * 0.58)
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.55:.2f});')
    # ---------- 2 logistic regression
    elif sid == "Y07":
        example(sc, "e1", "🩺", "Is this tumour benign?", L, top + 30 * u, W * 0.42, 0.5)
        example(sc, "e2", "📧", "Is this email spam?", L, top + 200 * u, W * 0.42, at(sid, "email", default=4))
        label(sc, "a", "Answer: YES or NO", W * 0.56, top + 120 * u, size=64, weight=900, color=AMB, t=at(sid, "yes", default=2))
    elif sid == "Y08":
        v_sigmoid(sc, (L, top, W * 0.58, H * 0.64), at(sid, "sigmoid", default=3))
        Xs, Ys = (lambda v: L + 110 * u + (v + 12) / 24 * (W * 0.58 - 150 * u)), (lambda v: top + 30 * u + (1.05 - v) / 1.1 * (H * 0.64 - 130 * u))
        live_sigmoid(sc, (0, 0, W, H), Xs, Ys, at(sid, "probability", default=6), at_xy=(L, top + H * 0.64 + 10 * u))
        formula(sc, "f", "P(yes) = 1 / (1 + e<sup>−z</sup>)<small>z = w·x + b · above 0.5 → yes</small>", W * 0.66, top + 40 * u, W * 0.30, at(sid, "curve", default=2), size=36)
        label(sc, "lg", "● benign   ● malignant  (real test tumours)", W * 0.66, top + 260 * u, size=26, color="#D9D6CE", t=at(sid, "sigmoid", default=3) + 1, w=W * .3)
    elif sid == "Y09":
        capture(sc, [seg(6, 0.2, d * 0.55, [("accuracy 0.986", at(sid, "98", default=5))], zoom=2.0)])
        tip(sc, "tip", "any yes/no question on table data, and you need trustworthy probabilities", "the boundary between classes is very curved", W * 0.18, H * 0.30, W * 0.64, d * 0.58)
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.55:.2f});')
    # ---------- 3 kNN
    elif sid == "Y10":
        example(sc, "ex", "🏘️", "“Tell me who your neighbours are,<br>and I'll tell you who you are”", L, top + 30 * u, W * 0.44, 0.4)
        s = H * 0.62; X, Y = knn_map(sc, W * 0.56, top, s, 0, regions=False, t_pts=.6)   # points only (regions come next beat)
        live_knn(sc, X, Y, at(sid, "new", default=6), (L, top + 470 * u))
        label(sc, "k", "k closest points vote", L, top + 260 * u, size=44, weight=900, color=AMB, t=at(sid, "vote", default=9))
        formula(sc, "f", "distance = √( (x₁−x₂)² + (y₁−y₂)² )", L, top + 350 * u, W * 0.44, at(sid, "closest", default=8), size=32)
    elif sid == "Y11":
        s = H * 0.66; knn_map(sc, W * 0.06, top, s, 0.4)
        K = V["knn"]; x2 = W * 0.06 + s + 70 * u
        label(sc, "a", "real wines · alcohol × flavanoids", x2, top, size=30, color=MUTE, t=.4, w=W * .4)
        label(sc, "b1", "● cultivar 1   ● cultivar 2   ● cultivar 3", x2, top + 50 * u, size=28, color="#D9D6CE", t=.6, w=W * .4)
        meters(sc, "m", [("k = 1", K["acc_k1"], at(sid, "one", default=8)), (f"k = {K['best_k']} (best)", K["acc_best"], at(sid, "best", default=10))], x2, top + 140 * u, W * 0.36, fmt="{:.1%}")
        label(sc, "c", "coloured regions = where the neighbours' vote lands", x2, top + 360 * u, size=26, color=MUTE, t=at(sid, "region", default=5), w=W * .38)
    elif sid == "Y12":
        capture(sc, [seg(8, 0.2, d * 0.45, [("best k=2 accuracy 0.907", 1.4)], zoom=2.0)])
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.45:.2f});')
        tip(sc, "tip", "the dataset is small and similar things really behave alike", "data is big (every prediction compares with every example) or features are on different scales", W * 0.18, H * 0.30, W * 0.64, d * 0.47)
    # ---------- 4 decision tree
    elif sid == "Y13":
        example(sc, "ex", "❓", "A game of yes / no questions", L, top + 30 * u, W * 0.42, 0.4)
        el = sc.id("q1"); sc.add(f'<div id="{el}" class="bigq">Is the tumour radius<br>above 16.80?</div>', f"#{el}{{left:{W * 0.55}px;top:{top + 40 * u}px;}}")
        sc.css.append(f"#root .bigq{{position:absolute;padding:{30 * u}px {40 * u}px;font-size:{52 * u}px;font-weight:900;line-height:1.2;{glass(sc)}}}")
        sc.appear(el, at(sid, "radius", default=5))
        label(sc, "y", "YES →", W * 0.58, top + 300 * u, size=48, color=RED, weight=900, t=at(sid, "yes", default=8))
        label(sc, "n", "NO →", W * 0.76, top + 300 * u, size=48, color=GRN, weight=900, t=at(sid, "yes", default=8) + .3)
        label(sc, "nx", "…then the next question, and the next", W * 0.55, top + 400 * u, size=34, color=MUTE, t=at(sid, "next", default=10))
    elif sid == "Y14":
        pos = tree_diagram(sc, W * 0.04, H * 0.29, W * 0.92, H * 0.47, 0.4, step=0.25)
        t_p = 4.6; steps = live_tree_path(sc, pos, t_p, dt=1.4)
        frames(sc, "tp", steps, W / 2 - W * 0.30, H * 0.775, t_p, 1.4, size=28, color=AMB)
        label(sc, "g", "each split = the question that makes the groups purest (lowest Gini impurity)", W / 2, H * 0.205, size=26, color=MUTE, anchor="center", t=at(sid, "purest", default=4))
        label(sc, "a", f"the real tree · depth 3 · test accuracy {V['tree']['acc']:.1%}", W / 2, H * 0.155, size=30, color=AMB, weight=800, anchor="center", t=1.0)
    elif sid == "Y15":
        capture(sc, [seg(12, 0.2, d, [("one deep tree accuracy 0.923", at(sid, "92", default=4))], zoom=2.0)])
        label(sc, "w", "a deep tree memorises its training data → overfitting", W / 2, H * 0.78, size=40, color=RED, weight=900, anchor="center", t=at(sid, "memorises", default=2))
    # ---------- 5 random forest
    elif sid == "Y16":
        example(sc, "ex", "🌲", "Ask 300 experts, take the vote", L, top + 20 * u, W * 0.42, 0.4)
        label(sc, "vh", "one real tumour · the vote of all 300 trees", W * 0.52, top, size=28, color=MUTE, t=at(sid, "vote", default=3), w=W * .44)
        R_ = live_votes(sc, W * 0.52, top + 50 * u, W * 0.43, at(sid, "vote", default=3), 4.5)
        label(sc, "vr", f"→ BENIGN wins (true: {'benign' if R_['true'] == 1 else 'malignant'}) ✓", W * 0.52, top + 470 * u, size=36, color=GRN, weight=900, t=at(sid, "vote", default=3) + 4.8, w=W * .44)
        formula(sc, "f", "each tree: random rows (bootstrap)<br>+ random features at each split<small>different mistakes → they cancel out in the vote</small>", L, top + 230 * u, W * 0.42, at(sid, "random", default=7), size=30)
    elif sid == "Y17":
        R_ = V["rf"]
        meters(sc, "m", [("one deep tree", R_["tree_acc"], .4), ("random forest · 300 trees", R_["rf_acc"], at(sid, "95", default=4))], L, top + 20 * u, W * 0.42, fmt="{:.1%}")
        label(sc, "it", "what the forest looks at most (real importances)", W * 0.52, top, size=28, color=MUTE, t=at(sid, "default", default=6) - 1, w=W * .44)
        hbars(sc, "im", [(n_, v, AMB, at(sid, "default", default=6) - 1 + i * .2, i == 0) for i, (n_, v) in enumerate(R_["imp"])], (W * 0.52, top + 60 * u, W * 0.44, 0), .2, fmt="{:.3f}", rowh=74, lab_w=.55)
        tip(sc, "tip", "you want a strong, safe default for table data", "you need the smallest, fastest model", L, H * 0.52, W * 0.42, at(sid, "default", default=6))
    # ---------- 6 gradient boosting
    elif sid == "Y18":
        example(sc, "ex", "🧑‍🎓", "A student who studies only<br>yesterday's mistakes", L, top + 20 * u, W * 0.44, 0.4)
        live_house(sc, (W * 0.52, top, W * 0.44, H * 0.56), at(sid, "first", default=4), max(6.0, d - at(sid, "first", default=4) - 1.5))
        label(sc, "hl", "one real house · dashed = its true value", W * 0.52, top - 40 * u, size=24, color=MUTE, t=at(sid, "first", default=4), w=W * .44)
        formula(sc, "f", "F<sub>m</sub>(x) = F<sub>m−1</sub>(x) + η · tree<sub>m</sub>(x)<small>each new tree is trained on the residuals y − F<sub>m−1</sub>(x)</small>", L, top + 260 * u, W * 0.44, at(sid, "second", default=6), size=32)
    elif sid == "Y19":
        v_gbm(sc, (L, top, W * 0.50, H * 0.62), at(sid, "error", default=2))
        G = V["gbm"]["r2"]
        label(sc, "h", "R² on house prices (higher is better)", W * 0.60, top, size=30, color=MUTE, t=at(sid, "linear", default=6), w=W * .36)
        hbars(sc, "r2", [("Linear Regression", G["Linear Regression"], MUTE, at(sid, "linear", default=6), False), ("Random Forest", G["Random Forest"], AMB, at(sid, "forest", default=8), False),
                         ("LightGBM", G["LightGBM"], GRN, at(sid, "lightgbm", default=10), True)], (W * 0.60, top + 60 * u, W * 0.36, 0), 1.0, fmt="{:.3f}", rowh=100, lab_w=.45)
    elif sid == "Y20":
        capture(sc, [seg(14, 0.2, d * 0.5, [("R² 0.854", 1.4)], zoom=3.4)])
        tip(sc, "tip", "you have table data (numbers + categories): LightGBM, XGBoost, CatBoost usually win", "data is tiny, or you need a model a person can read line by line", W * 0.18, H * 0.30, W * 0.64, d * 0.52)
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.5:.2f});')
    # ---------- 7 SVM
    elif sid in ("Y21", "Y22"):
        S_ = V["svm"]
        if sid == "Y21":
            example(sc, "ex", "🛣️", "Build the widest road<br>between two villages", L, top + 20 * u, W * 0.40, 0.4)
        box = (W * (0.48 if sid == "Y21" else 0.05), top, W * (0.47 if sid == "Y21" else 0.55), H * 0.64)
        X, Y, bx = chart(sc, "sv", box, (-2.6, 2.6), (-2.6, 2.6), "alcohol (std.)", "flavanoids (std.)", ([-2, 0, 2], [-2, 0, 2], lambda v: f"{v:g}"), t0=.2)
        scatter(sc, bx, X, Y, S_["x"], S_["y"], [CLS[c] for c in S_["c"]], .5, r=6, op=.9)
        w1, w2 = S_["w"]; bb = S_["b"]
        def line(off):   # w1·x + w2·y + b = off, clipped to the plotted square
            pts = []
            for x_ in (-2.6, 2.6):
                y_ = (off - bb - w1 * x_) / w2
                if -2.6 <= y_ <= 2.6: pts.append((x_, y_))
            for y_ in (-2.6, 2.6):
                x_ = (off - bb - w2 * y_) / w1
                if -2.6 <= x_ <= 2.6: pts.append((x_, y_))
            pts = sorted(set((round(a_, 4), round(c_, 4)) for a_, c_ in pts))[:2]
            return [p_[0] for p_ in pts], [p_[1] for p_ in pts]
        tl_ = at(sid, "road", default=3) if sid == "Y21" else at(sid, "margin", default=3)
        xs_, ys_ = line(0); curve(sc, bx, "bd", X, Y, xs_, ys_, AMB, tl_, d=.8, width=6)
        for k, off in enumerate((1, -1)):
            xs_, ys_ = line(off); curve(sc, bx, f"mg{k}", X, Y, xs_, ys_, MUTE, tl_ + .6, dash=f"{10 * u} {8 * u}", width=3)
        if sid == "Y22":
            ring = "".join(f'<circle cx="{X(S_["x"][i]):.1f}" cy="{Y(S_["y"][i]):.1f}" r="{13 * u}" fill="none" stroke="#F3F0E8" stroke-width="{3 * u}"/>' for i in S_["sv"])
            g = sc.id("svr"); layer(sc, bx, f'<g id="{g}">{ring}</g>')
            sc.js.append(f'tl.fromTo("#{g}",{{opacity:0}},{{opacity:1,duration:.5}},{at(sid, "28", default=8):.2f});')
            label(sc, "n", f"{S_['n_sv']} support vectors", W * 0.64, top + 40 * u, size=72, color=AMB, weight=900, t=at(sid, "28", default=8))
            label(sc, "n2", "only the points touching the margin decide the boundary", W * 0.64, top + 140 * u, size=30, color="#D9D6CE", t=at(sid, "closest", default=6), w=W * .32)
            formula(sc, "f", "maximise margin = 2 / ‖w‖<small>subject to every point being on its side</small>", W * 0.64, top + 300 * u, W * 0.32, at(sid, "widest", default=4), size=30)
    elif sid == "Y23":
        capture(sc, [seg(16, 0.2, d * 0.5, [("accuracy 0.981", at(sid, "98", default=4))], zoom=2.0)])
        tip(sc, "tip", "small or medium data with many features (also text, with a linear kernel)", "you have millions of rows (training gets slow)", W * 0.18, H * 0.30, W * 0.64, d * 0.52)
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.5:.2f});')
        label(sc, "k", "kernel trick → curved boundaries", W / 2, H * 0.16, size=40, color=AMB, weight=900, anchor="center", t=at(sid, "kernel", default=1))
    # ---------- 8 naive Bayes
    elif sid == "Y24":
        example(sc, "ex", "💬", "“tires … brake … dealer …”<br>→ probably about cars", L, top + 20 * u, W * 0.46, 0.4, sub="text example")
        for i, w_ in enumerate(["tires", "brake", "dealer"]):
            el = sc.id(f"wd{i}"); sc.add(f'<div id="{el}" class="chipw">{w_}</div>', f"#{el}{{left:{W * 0.56 + i * 230 * u}px;top:{top + 60 * u}px;}}")
            sc.appear(el, at(sid, w_, default=3 + i * .5), y=10)
        sc.css.append(f"#root .chipw{{position:absolute;padding:{12 * u}px {24 * u}px;border-radius:{30 * u}px;background:rgba(255,178,36,.18);border:{2 * u}px solid {AMB};font-size:{36 * u}px;font-weight:800;}}")
        label(sc, "r", "→ rec.autos 🚗", W * 0.56, top + 200 * u, size=64, weight=900, color=GRN, t=at(sid, "cars", default=6))
    elif sid == "Y25":
        formula(sc, "f", "P(topic | words) ∝ P(topic) × Π P(word | topic)<small>Bayes' rule, with the “naive” assumption that words are independent</small>", L, top + 20 * u, W * 0.90, 0.4, size=44)
        live_nb(sc, L, H * 0.40, W * 0.58, at(sid, "multiplies", default=4))
    elif sid == "Y26":
        capture(sc, [seg(18, 0.2, d * 0.5, [("Naive Bayes accuracy 0.881", at(sid, "88", default=4))], zoom=2.0)])
        tip(sc, "tip", "you need a fast, simple first baseline for text classification", "word order or meaning matters (use a Transformer)", W * 0.18, H * 0.30, W * 0.64, d * 0.52)
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.5:.2f});')
    # ---------- 9 k-means
    elif sid == "Y27":
        example(sc, "ex", "👥", "Sort a crowd into groups<br>without knowing any names", L, top + 20 * u, W * 0.42, 0.4, sub="no labels")
        K = V["kmeans"]; xs, ys = K["x"], K["y"]
        X, Y, bx = chart(sc, "km", (W * 0.50, top, W * 0.45, H * 0.62), (min(xs) - .5, max(xs) + .5), (min(ys) - .5, max(ys) + .5), "PCA 1", "PCA 2", ([], [], lambda v: ""), t0=.3)
        scatter(sc, bx, X, Y, xs, ys, ["#B8B5AE"] * len(xs), .6, r=6, op=.7)
        label(sc, "q", "178 real wines · 13 features · no labels", W * 0.50, H * 0.81, size=28, color=MUTE, t=1.0, w=W * .45)
    elif sid == "Y28":
        steps = [("1 · pick k centres", "pick"), ("2 · assign each point to its nearest centre", "assign"), ("3 · move each centre to its group's middle", "move"), ("4 · repeat until nothing changes", "repeat")]
        for i, (t_, tok) in enumerate(steps):
            label(sc, f"s{i}", t_, L, top + 20 * u + i * 80 * u, size=34, weight=800, color=AMB if i == 3 else INK, t=at(sid, tok, default=1 + i * 2), w=W * .38)
        v_kmeans(sc, (W * 0.46, top, W * 0.50, H * 0.64), at(sid, "assign", default=3), dt=max(0.8, (d - at(sid, "assign", default=3) - 1.5) / 6))
        label(sc, "c", f"converged in {len(V['kmeans']['steps'])} real steps", L, top + 400 * u, size=40, color=GRN, weight=900, t=at(sid, "six", default=d - 2))
    elif sid == "Y29":
        K = V["kmeans"]; xs, ys = K["x"], K["y"]
        for j, (tt, labs) in enumerate([("K-Means groups", K["steps"][-1]["labels"]), ("real cultivars (hidden from it)", K["true"])]):
            X, Y, bx = chart(sc, f"c{j}", (L + j * W * 0.46, top + 30 * u, W * 0.44, H * 0.52), (min(xs) - .5, max(xs) + .5), (min(ys) - .5, max(ys) + .5), "", "", ([], [], lambda v: ""), t0=.3 + j * .6)
            cm = cluster_map() if j == 0 else {}
            scatter(sc, bx, X, Y, xs, ys, [CLS[cm.get(l_, l_)] for l_ in labs], .6 + j * .6, r=6, op=.9)
            label(sc, f"l{j}", tt, L + j * W * 0.46, top - 10 * u, size=30, weight=800, color=AMB if j == 0 else MUTE, t=.3 + j * .6)
        label(sc, "a", f"agreement (adjusted Rand index): {K['ari']:.3f}", W / 2, H * 0.78, size=44, color=GRN, weight=900, anchor="center", t=at(sid, "index", default=4))
        label(sc, "u", "each group coloured by its closest real type", W / 2, H * 0.835, size=24, color=MUTE, anchor="center", t=at(sid, "index", default=4) + .5)
    # ---------- 10 PCA
    elif sid == "Y30":
        example(sc, "ex", "🔦", "The shadow of a 3-D object:<br>the right angle keeps the shape", L, top + 20 * u, W * 0.44, 0.4)
        el = sc.id("cube"); sc.add(f'<div id="{el}" class="cube"><i></i><i></i><i></i><i></i><i></i><i></i></div>', f"#{el}{{left:{W * 0.66}px;top:{top + 60 * u}px;}}")
        cs = 220 * u
        faces = [f"translateZ({cs / 2}px)", f"rotateY(180deg) translateZ({cs / 2}px)", f"rotateY(90deg) translateZ({cs / 2}px)", f"rotateY(-90deg) translateZ({cs / 2}px)",
                 f"rotateX(90deg) translateZ({cs / 2}px)", f"rotateX(-90deg) translateZ({cs / 2}px)"]
        sc.css.append(f"#root .cube{{position:absolute;width:{cs}px;height:{cs}px;transform-style:preserve-3d;}}#root .cube i{{position:absolute;inset:0;border:{3 * u}px solid {AMB};background:rgba(255,178,36,.08);}}"
                      + "".join(f"#root .cube i:nth-child({k + 1}){{transform:{f};}}" for k, f in enumerate(faces)))
        sc.js.append(f'tl.fromTo("#{el}",{{rotationX:-25,rotationY:20}},{{rotationX:-25,rotationY:200,duration:{d:.2f},ease:"none"}},0);')
        sh = sc.id("sh"); sc.add(f'<div id="{sh}" class="shd"></div>', f"#{sh}{{left:{W * 0.66}px;top:{top + 400 * u}px;}}")
        sc.css.append(f"#root .shd{{position:absolute;width:{cs}px;height:{40 * u}px;border-radius:50%;background:radial-gradient(rgba(0,0,0,.65),transparent 70%);}}")
        label(sc, "k", "many features → a few new axes", L, top + 260 * u, size=40, weight=900, color=AMB, t=at(sid, "dimensions", default=6), w=W * .44)
    elif sid == "Y31":
        P_ = V["pca"]
        X, Y, bx = chart(sc, "ev", (L, top, W * 0.40, H * 0.60), (0.5, 20.5), (0, .16), "component", "variance kept", ([1, 5, 10, 15, 20], [0, .05, .1, .15], lambda v: f"{v:g}"), t0=.3)
        bars = "".join(f'<rect x="{X(i + 1) - 10 * u:.1f}" y="{Y(v):.1f}" width="{20 * u}" height="{Y(0) - Y(v):.1f}" fill="{AMB if i < 2 else BLU}"/>' for i, v in enumerate(P_["ev"]))
        g = sc.id("evb"); layer(sc, bx, f'<g id="{g}">{bars}</g>'); sc.js.append(f'tl.fromTo("#{g}",{{opacity:0}},{{opacity:1,duration:.6}},{at(sid, "varies", default=2):.2f});')
        X2, Y2, bx2 = chart(sc, "pc", (W * 0.48, top, W * 0.48, H * 0.60), (min(P_["x"]) - 2, max(P_["x"]) + 2), (min(P_["y"]) - 2, max(P_["y"]) + 2), "PC 1", "PC 2", ([], [], lambda v: ""), t0=at(sid, "two", default=6) - .5)
        scatter(sc, bx2, X2, Y2, P_["x"], P_["y"], [DIG[c] for c in P_["c"]], at(sid, "two", default=6), r=4, op=.85)
        label(sc, "a", f"2 components keep {P_['cum2']:.1%} · {P_['n90']} keep 90%", L, H * 0.775, size=36, color=AMB, weight=900, t=at(sid, "ninety", default=10), w=W * .44)
        label(sc, "b", "1,200 real digit images, 64 pixels squeezed to 2 numbers · each colour = one digit 0–9", W * 0.48, H * 0.775, size=24, color=MUTE, t=at(sid, "two", default=6), w=W * .48)
    elif sid == "Y32":
        capture(sc, [seg(22, 0.2, d * 0.5, [("21 components keep 90%", 1.4)], zoom=2.0)])
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.5:.2f});')
        tip(sc, "tip", "you want to visualise data, compress features, or remove noise before another model", "you need to explain results in the original features", W * 0.18, H * 0.30, W * 0.64, d * 0.52)
    # ---------- 11 MLP
    elif sid == "Y33":
        network(sc, W * 0.50, top + 20 * u, W * 0.40, H * 0.56, .4)
        formula(sc, "f", "neuron: a = ReLU(w·x + b)<small>ReLU(z) = max(0, z) · stacked layers learn their own features</small>", L, top + 40 * u, W * 0.40, at(sid, "weighted", default=3), size=34)
        example(sc, "ex", "🧠", "Tiny decision makers,<br>stacked in layers", L, top + 260 * u, W * 0.40, at(sid, "stack", default=8))
    elif sid == "Y34":
        cols = network(sc, W * 0.05, top + 20 * u, W * 0.36, H * 0.50, .3)
        pulse(sc, "fw", cols[0][3][0], cols[0][3][1], cols[2][5][0], cols[2][5][1], .6, d=1.2, color=AMB)
        pulse(sc, "bk", cols[2][5][0], cols[2][5][1], cols[0][3][0], cols[0][3][1], at(sid, "backwards", default=4), d=1.2, color=RED)
        label(sc, "fwl", "→ forward: predict", W * 0.05, H * 0.76, size=28, color=AMB, weight=800, t=.6)
        label(sc, "bkl", "← backpropagation: error flows back, every weight adjusts", W * 0.05, H * 0.80, size=28, color=RED, weight=800, t=at(sid, "backwards", default=4), w=W * .4)
        M_ = V["mlp"]
        X, Y, bx = chart(sc, "lc", (W * 0.50, top, W * 0.46, H * 0.56), (0, len(M_["loss"])), (0, 2.5), "epoch", "loss", ([0, 100, 200, 300], [0, 1, 2], lambda v: f"{v:g}"), t0=at(sid, "loss", default=8) - .5)
        curve(sc, bx, "lcv", X, Y, list(range(len(M_["loss"]))), M_["loss"], GRN, at(sid, "loss", default=8), d=1.4, width=5)
        label(sc, "w", f"{M_['weights']:,} weights", W * 0.50, H * 0.78, size=48, color=AMB, weight=900, t=at(sid, "4", default=7))
    elif sid == "Y35":
        M_ = V["mlp"]
        grid_img(sc, "dg", [M_["digit"][i * 8:(i + 1) * 8] for i in range(8)] if isinstance(M_["digit"][0], int) else M_["digit"], L, top + 20 * u, 42 * u, .3, gap=2)
        label(sc, "dl", f"one real test digit (true: {M_['true']})", L, top + 400 * u, size=28, color=MUTE, t=.3, w=W * .3)
        order = sorted(range(10), key=lambda k: -M_["probs"][k])[:4]
        meters(sc, "pm", [(f"digit {k}", M_["probs"][k], .6 + i * .2) for i, k in enumerate(order)], W * 0.34, top + 20 * u, W * 0.30, fmt="{:.1%}")
        label(sc, "un", "right, but not sure: 1 vs 8", W * 0.34, top + 420 * u, size=28, color=AMB, weight=800, t=1.6, w=W * .3)
        live_hidden(sc, W * 0.70, top + 40 * u, 34 * u, .5)
        label(sc, "hn", "64 hidden neurons firing (real)", W * 0.70, top, size=24, color=MUTE, t=.5, w=W * .26)
        capture(sc, [seg(24, d * 0.45, d, [("MLP accuracy 0.982", d * 0.45 + 1.4)], zoom=3.4)])
        for nm in ("dg", "dl", "un", "hid", "hn"):
            sc.js.append(f'tl.to("#{sc.id(nm)}",{{opacity:0,duration:.3}},{d * 0.45 - .3:.2f});')
        sc.js.append(f'tl.to("#root [id^={sc.id("pm")}]",{{opacity:0,duration:.3}},{d * 0.45 - .3:.2f});')
    # ---------- 12 CNN
    elif sid == "Y36":
        img = V["cnn"]["image"]; cell = 15 * u
        grid_img(sc, "img", img, W * 0.58, top + 20 * u, cell, .3)
        mg = sc.id("mag"); sc.add(f'<div id="{mg}" class="mag"></div>', f"#{mg}{{left:{W * 0.58}px;top:{top + 20 * u}px;}}")
        sc.css.append(f"#root .mag{{position:absolute;width:{5 * cell}px;height:{5 * cell}px;border:{5 * u}px solid {AMB};box-shadow:0 0 {30 * u}px rgba(255,178,36,.6);border-radius:{6 * u}px;z-index:6;}}")
        kf = ",".join(f'{{x:{(k % 6) * 4 * cell:.0f},y:{(k // 6) * 4 * cell:.0f}}}' for k in range(36))
        sc.js.append(f'tl.to("#{mg}",{{keyframes:[{kf}],duration:{max(2, d - 1.5):.2f},ease:"none"}},.6);')
        example(sc, "ex", "🔍", "A magnifying glass sliding over a photo:<br>edges → curves → shapes", L, top + 20 * u, W * 0.48, .4)
        formula(sc, "f", "feature(i,j) = Σ filter · patch(i,j)<small>the same small filter slides over every position (convolution)</small>", L, top + 230 * u, W * 0.48, at(sid, "edges", default=5), size=32)
        label(sc, "il", "real 28×28 test image  →  feature map of filter 1 (real)", W * 0.58, top + 460 * u, size=24, color=MUTE, t=.3, w=W * .38)
        fm = grid_img(sc, "fm0", V["cnn"]["fmap"][0], W * 0.58 + 28 * cell + 40 * u, top + 20 * u + 7 * cell, 15 * u, .5)
        D_ = max(2, d - 1.5)
        sc.js.append(f'tl.fromTo("#{fm}",{{clipPath:"inset(0 0 100% 0)"}},{{clipPath:"inset(0 0 0% 0)",duration:{D_:.2f},ease:"steps(6)"}},.6);')
    elif sid == "Y37":
        C_ = V["cnn"]; cell = 22 * u
        label(sc, "a", "8 real learned filters · 5×5 weights each (amber +, red −)", L, top, size=28, color=MUTE, t=.2, w=W * .44)
        v_cnn_filters(sc, L, top + 50 * u, cell, at(sid, "filter", default=1))
        grid_img(sc, "img", C_["image"], W * 0.50, top + 50 * u, 9 * u, at(sid, "seven", default=6))
        label(sc, "b", "input: a 7", W * 0.50, top + 330 * u, size=26, color=MUTE, t=at(sid, "seven", default=6))
        for i, F in enumerate(C_["fmap"]):
            grid_img(sc, f"fm{i}", F, W * 0.69 + (i % 2) * 15 * 14 * u, top + 50 * u + (i // 2) * 15 * 14 * u, 13 * u, at(sid, "seven", default=6) + .4 + i * .2)
        label(sc, "c", "what 4 filters see in it (real feature maps)", W * 0.69, top + 50 * u + 2 * 15 * 14 * u + 10 * u, size=24, color=MUTE, t=at(sid, "seven", default=6) + .5, w=W * .28)
    elif sid == "Y38":
        capture(sc, [seg(26, 0.2, d * 0.5, [("54,314 weights", at(sid, "54", default=4)), ("CNN test accuracy 0.9754", at(sid, "97", default=6))], zoom=2.0)])
        tip(sc, "tip", "you work with images (or any grid-like data)", "your data is a table — boosting will be simpler and stronger", W * 0.18, H * 0.30, W * 0.64, d * 0.52)
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.5:.2f});')
    # ---------- 13 LSTM
    elif sid == "Y39":
        example(sc, "ex", "📖", "Reading a story and<br>remembering what came before", L, top + 20 * u, W * 0.44, .4)
        gates = [("forget gate", "what to drop", RED, "forget"), ("input gate", "what to store", GRN, "remember"), ("output gate", "what to pass on", AMB, "pass")]
        for i, (g_, sub, col, tok) in enumerate(gates):
            card(sc, f"g{i}", g_, sub, W * 0.54, top + i * 140 * u, W * 0.40, at(sid, tok, default=4 + i), col, big=42)
        formula(sc, "f", "c<sub>t</sub> = f<sub>t</sub> ⊙ c<sub>t−1</sub> + i<sub>t</sub> ⊙ g<sub>t</sub><small>the cell state c carries the memory along the sequence</small>", L, top + 260 * u, W * 0.44, at(sid, "gates", default=5), size=34)
    elif sid == "Y40":
        v_lstm(sc, (L, top, W * 0.60, H * 0.64), at(sid, "forecast", default=3))
        Ls = V["lstm"]
        label(sc, "a", "● actual   ● simple rule   ● LSTM", W * 0.68, top, size=28, color="#D9D6CE", t=at(sid, "forecast", default=3) + 1, w=W * .3)
        meters(sc, "m", [("simple rule · error", Ls["mae_naive"] / 4, at(sid, "one", default=10)), ("LSTM · error", Ls["mae"] / 4, at(sid, "three", nth=1, default=13))], W * 0.68, top + 80 * u, W * 0.28, fmt="")
        label(sc, "e1", f"{Ls['mae_naive']:.2f} ppm", W * 0.68 + W * 0.22, top + 100 * u, size=36, color=GRN, weight=900, mono=True, t=at(sid, "one", default=10))
        label(sc, "e2", f"{Ls['mae']:.2f} ppm", W * 0.68 + W * 0.22, top + 196 * u, size=36, color=RED, weight=900, mono=True, t=at(sid, "three", nth=1, default=13))
        label(sc, "n", "real Mauna Loa CO₂ · the last 36 months were hidden and forecast", W * 0.68, top + 300 * u, size=24, color=MUTE, t=at(sid, "forecast", default=3), w=W * .28)
    elif sid == "Y41":
        capture(sc, [seg(28, 0.2, d * 0.5, [("LSTM MAE 2.95 ppm", 1.4), ("seasonal-naive MAE 0.99 ppm", 2.8)], zoom=2.0)])
        el = sc.id("les"); sc.add(f'<div id="{el}" class="les">Always beat a simple baseline<br>before you trust a complex model.</div>', f"#{el}{{left:{W / 2}px;top:{H * 0.40}px;}}")
        sc.css.append(f"#root .les{{position:absolute;transform:translateX(-50%);text-align:center;padding:{30 * u}px {44 * u}px;font-size:{56 * u}px;font-weight:900;color:{AMB};line-height:1.25;{glass(sc)}}}")
        sc.appear(el, d * 0.52)
        sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{d * 0.5:.2f});')
    # ---------- 14 Transformer
    elif sid == "Y42":
        words = ["Your", "bKash", "account", "will", "be", "blocked", "—", "verify", "your", "PIN"]
        sc.css.append(FONT); xs_ = [L + i * (W * 0.9 / len(words)) for i in range(len(words))]; yw = H * 0.48
        for i, w_ in enumerate(words):
            el = sc.id(f"tw{i}"); sc.add(f'<div id="{el}" class="tw">{escape(w_)}</div>', f"#{el}{{left:{xs_[i]}px;top:{yw}px;}}")
            sc.appear(el, .3 + i * .06, y=10, d=.3)
        sc.css.append(f"#root .tw{{position:absolute;padding:{10 * u}px {16 * u}px;border-radius:{10 * u}px;background:#25262A;font-size:{34 * u}px;font-weight:800;}}")
        arcs = ""
        src = 9
        for j in range(len(words)):
            if j == src: continue
            x1, x2 = xs_[src] + 40 * u, xs_[j] + 40 * u; hgt = 60 * u + abs(src - j) * 22 * u
            wgt = 6 if words[j] in ("bKash", "blocked", "verify") else 2
            arcs += f'<path d="M{x1:.0f} {yw - 6 * u:.0f} Q{(x1 + x2) / 2:.0f} {yw - hgt:.0f} {x2:.0f} {yw - 6 * u:.0f}" fill="none" stroke="{AMB}" stroke-opacity="{.9 if wgt > 2 else .35}" stroke-width="{wgt * u}"/>'
        g = sc.id("arcs"); layer(sc, (0, 0, W, H), f'<g id="{g}">{arcs}</g>')
        sc.js.append(f'tl.fromTo("#{g}",{{opacity:0}},{{opacity:1,duration:.6}},{at(sid, "every", default=5):.2f});')
        label(sc, "a", "attention: every word looks at every other word, all at once", W / 2, H * 0.24, size=40, weight=900, color=AMB, anchor="center", t=at(sid, "attention", default=7))
        label(sc, "il", "illustration of attention (the real weights are in the next scene)", W / 2, H * 0.62, size=24, color=MUTE, anchor="center", t=at(sid, "every", default=5))
        formula(sc, "f", "Attention(Q,K,V) = softmax(QKᵀ / √d) · V", W * 0.22, H * 0.68, W * 0.56, at(sid, "attention", default=7) + .5, size=36)
    elif sid == "Y43":
        S = V["sg"]; sc.css.append(FONT)
        label(sc, "a", f"real Transformer (XLM-RoBERTa, fine-tuned) · {len(S['tokens']['pieces'])} tokens of a Bangla scam SMS", L, top, size=28, color=MUTE, t=.2, w=W * .9)
        v_tokens_attention(sc, L, top + 70 * u, W * 0.9, at(sid, "attention", default=4))
        meters(sc, "m", [("P(scam) after fine-tuning", S["before_after"]["after"][2], at(sid, "sure", default=9))], L, H * 0.66, W * 0.5, fmt="{:.1%}")
        label(sc, "s", "from my ScamGuard project", W * 0.62, H * 0.68, size=28, color=AMB, weight=800, t=at(sid, "project", default=1))
    # ---------- 15 LLM
    elif sid == "Y44":
        S = V["sg"]; q = {}
        for w_, v in S["qwen_next"]["top"]: q.setdefault(w_.strip() or "·", v)
        label(sc, "p", "“…Is this SMS a scam? Answer Yes or No.”  →  next word?", L, top + 10 * u, size=36, weight=800, t=.4, w=W * .9)
        meters(sc, "m", [(f"“{k}”", v, at(sid, "probability", default=7) + i * .25) for i, (k, v) in enumerate(list(q.items())[:4])], L, top + 110 * u, W * 0.55, fmt="{:.4f}")
        formula(sc, "f", "P(next word | all previous words)<small>trained on huge text: predicting the next word, again and again</small>", W * 0.64, top + 110 * u, W * 0.32, at(sid, "predicts", default=3), size=30)
        label(sc, "n", "real next-word probabilities · Qwen2.5-3B · from ScamGuard", L, H * 0.78, size=24, color=MUTE, t=at(sid, "probability", default=7))
    elif sid == "Y45":
        S = V["sg"]; rows = [(r_["model"].split(" (")[0], r_["PR-AUC"], GRN if i == 0 else (AMB if i < 3 else MUTE), .4 + i * .25, i == 0) for i, r_ in enumerate(S["locked"])]
        label(sc, "h", "locked real-world test (PR-AUC)", L, top, size=32, color=MUTE, t=.2)
        hbars(sc, "lb", rows, (L, top + 60 * u, W * 0.56, 0), 1.0, fmt="{:.3f}", rowh=86, lab_w=.42)
        tip(sc, "tip", "language: meaning, context, many languages — and you have the compute", "a small table or a tiny budget (start simple)", W * 0.64, top + 40 * u, W * 0.32, at(sid, "use", default=8))
    # ---------- decision guide + outro
    elif sid == "Y46":
        chapter(sc, "WHICH MODEL SHOULD YOU USE?")
        rows = [("🧮 numbers in a table", "linear / logistic regression → gradient boosting", "table"), ("📝 text", "naive Bayes → Transformer", "text"),
                ("🖼️ images", "CNN", "images"), ("📈 sequences", "simple baseline → LSTM / Transformer", "sequences"), ("🏷️ no labels", "k-means · PCA", "labels")]
        for i, (a_, b_, tok) in enumerate(rows):
            y_ = top + 30 * u + i * 118 * u
            card(sc, f"a{i}", a_, "", L, y_, W * 0.32, at(sid, tok, default=1 + i * 3), INK, big=40)
            el = sc.id(f"ar{i}"); sc.add(f'<div id="{el}" class="arw">→</div>', f"#{el}{{left:{L + W * 0.34}px;top:{y_ + 4 * u}px;}}")
            sc.appear(el, at(sid, tok, default=1 + i * 3) + .2, y=0)
            card(sc, f"b{i}", b_, "", L + W * 0.40, y_, W * 0.52, at(sid, tok, default=1 + i * 3) + .35, AMB, big=40)
        sc.css.append(f"#root .arw{{position:absolute;font-size:{60 * u}px;color:{AMB};font-weight:900;}}")
    elif sid == "Y47":
        aura(sc, (AMB, GRN))
        for i, (t_, tok) in enumerate([("① divide your data honestly", "divide"), ("② compare against a baseline", "compare"), ("③ trust only the untouched test set", "trust")]):
            label(sc, f"r{i}", t_, L, H * 0.20 + i * 100 * u, size=60, weight=900, t=at(sid, tok, default=1 + i * 2), w=W * .9)
        label(sc, "gh", "Code + every result: github.com/reazulzishan", L, H * 0.62, size=44, color=AMB, weight=900, t=at(sid, "github", default=9), w=W * .9)
        label(sc, "yt", "CS Funda · one model at a time", L, H * 0.70, size=36, color="#D9D6CE", weight=800, t=at(sid, "machine", default=7), w=W * .9)
    # ---------------------------------------------------------------- the Reel (9:16)
    else:
        S_reel(sc, sid, W, H, L, d)
    return sc


def S_reel(sc, sid, W, H, L, d):
    u = sc.u; top = H * 0.16; rid = sid.replace("Q", "R")
    if rid == "R01":
        aura(sc, (AMB, BLU))
        for i, (n, tag) in enumerate(MODELS):
            el = sc.id(f"m{i}")
            sc.add(f'<div id="{el}" class="mc"><b>{i + 1:02d}</b>{escape(n)}<em>{tag.split(" · ")[0]}</em></div>', f"#{el}{{left:{L + (i % 2) * W * 0.46}px;top:{H * 0.10 + (i // 2) * 82 * u}px;width:{W * 0.44}px;}}")
            sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,x:{-20 * u}}},{{opacity:1,x:0,duration:.25}},{i * .04:.2f});')
        sc.css.append(f"#root .mc{{position:absolute;padding:{10 * u}px {14 * u}px;border-radius:{12 * u}px;background:#25262A;font-size:{26 * u}px;font-weight:800;display:flex;gap:{10 * u}px;align-items:center;}}"
                      f"#root .mc b{{color:{AMB};font-family:'JetBrains Mono',monospace;}}#root .mc em{{margin-left:auto;font-style:normal;font-size:{16 * u}px;color:#B8B5AE;}}")
        label(sc, "q", "Which one should you use?", W / 2, H * 0.52, size=72, weight=900, color=AMB, anchor="center", t=0.3, w=W * .9)
    elif rid == "R02":
        label(sc, "h", "TRAINING = GRADIENT DESCENT", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        tn = at(sid, "nudging", default=3)
        v_linreg(sc, (L, top + 30 * u, W * 0.90, H * 0.26), .2, .5, loss_box=(L, top + 30 * u + H * 0.29, W * 0.90, H * 0.22), t_loss=tn, live=(tn, max(4.5, d - tn - .6), None))
        label(sc, "f", "w ← w − η · ∂error/∂w", L, H * 0.70, size=40, mono=True, color=GRN, weight=700, t=at(sid, "gradient", default=6))
    elif rid == "R03":
        label(sc, "h", "LINEAR → LOGISTIC (SIGMOID)", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        v_sigmoid(sc, (L, top + 30 * u, W * 0.90, H * 0.36), at(sid, "sigmoid", default=3))
        bx_ = (L, top + 30 * u, W * 0.90, H * 0.36)
        Xs, Ys = (lambda v: bx_[0] + 110 * u + (v + 12) / 24 * (bx_[2] - 150 * u)), (lambda v: bx_[1] + 30 * u + (1.05 - v) / 1.1 * (bx_[3] - 130 * u))
        live_sigmoid(sc, (0, 0, W, H), Xs, Ys, at(sid, "sigmoid", default=3) + 1.2, at_xy=(L, bx_[1] + bx_[3] + 10 * u))
        label(sc, "f", "P(yes) = 1 / (1 + e^−(w·x+b))", L, H * 0.63, size=40, mono=True, color=INK, t=at(sid, "sigmoid", default=3) + .5)
        label(sc, "a", f"real tumours: {V['logreg']['acc']:.1%} correct", L, H * 0.68, size=48, color=GRN, weight=900, t=at(sid, "98", default=8))
    elif rid == "R04":
        label(sc, "h", "DECISION TREE → RANDOM FOREST", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        pos = tree_diagram(sc, W * 0.02, top + 90 * u, W * 0.96, H * 0.31, .2, step=.15)
        live_tree_path(sc, pos, 2.4, dt=.9)
        R_ = V["rf"]
        meters(sc, "m", [("one tree", R_["tree_acc"], at(sid, "92", default=6)), ("300 trees voting", R_["rf_acc"], at(sid, "95", default=8))], L, H * 0.56, W * 0.88, fmt="{:.1%}")
    elif rid == "R05":
        label(sc, "h", "GRADIENT BOOSTING", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        label(sc, "f", "F_m = F_m−1 + η · tree_m (fit to the residuals)", L, top + 10 * u, size=30, mono=True, color=INK, t=.4, w=W * .9)
        live_house(sc, (L, top + 90 * u, W * 0.90, H * 0.28), at(sid, "mistakes", default=3) - 1.0, 4.5)
        G = V["gbm"]["r2"]
        hbars(sc, "r2", [("Linear", G["Linear Regression"], MUTE, at(sid, "linear", default=6), False), ("LightGBM", G["LightGBM"], GRN, at(sid, "85", default=8), True)],
              (L, H * 0.56, W * 0.90, 0), 1.0, fmt="R² {:.3f}", rowh=110, lab_w=.32)
    elif rid == "R06":
        label(sc, "h", "K-MEANS · NO LABELS", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        v_kmeans(sc, (L, top + 30 * u, W * 0.90, H * 0.40), .6, dt=max(.7, (d - 2) / 6))
        label(sc, "a", f"agreement with the hidden real types: {V['kmeans']['ari']:.2f}", L, H * 0.64, size=36, color=GRN, weight=900, t=at(sid, "matched", default=7), w=W * .9)
    elif rid == "R07":
        label(sc, "h", "CNN · LEARNED FILTERS", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        cell = 30 * u
        v_cnn_filters(sc, L + 40 * u, top + 30 * u, cell, at(sid, "filters", default=2))
        iy = top + 30 * u + 2 * (5 * cell + 18 * u) + 30 * u
        grid_img(sc, "img", V["cnn"]["image"], L + 40 * u, iy, 13 * u, at(sid, "picture", default=3))
        fm = grid_img(sc, "fm0", V["cnn"]["fmap"][0], L + 40 * u + 28 * 13 * u + 80 * u, iy + 2 * 13 * u, 13 * u, at(sid, "picture", default=3))
        sc.js.append(f'tl.fromTo("#{fm}",{{clipPath:"inset(0 0 100% 0)"}},{{clipPath:"inset(0 0 0% 0)",duration:3,ease:"steps(6)"}},{at(sid, "picture", default=3):.2f});')
        label(sc, "fml", "image  →  what filter 1 sees (real)", L + 40 * u, iy + 28 * 13 * u + 20 * u, size=26, color=MUTE, t=at(sid, "picture", default=3), w=W * .9)
        label(sc, "a", f"{V['cnn']['acc']:.1%} on 10,000 new digits", L, H * 0.68, size=48, color=GRN, weight=900, t=at(sid, "97", default=8))
    elif rid == "R08":
        label(sc, "h", "LSTM · MEMORY FOR SEQUENCES", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        v_lstm(sc, (L, top + 30 * u, W * 0.90, H * 0.36), at(sid, "carbon", default=3))
        Ls = V["lstm"]
        label(sc, "a", f"simple rule {Ls['mae_naive']:.2f} ppm  vs  LSTM {Ls['mae']:.2f} ppm", L, H * 0.60, size=38, color=AMB, weight=900, t=at(sid, "baseline", default=7), w=W * .9)
    elif rid == "R09":
        label(sc, "h", "TRANSFORMER → LLM", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        v_tokens_attention(sc, L, top + 30 * u, W * 0.90, at(sid, "attend", default=3), small=True)
        S = V["sg"]; q = {}
        for w_, v in S["qwen_next"]["top"]: q.setdefault(w_.strip() or "·", v)
        meters(sc, "m", [(f"next word “{k}”", v, at(sid, "next", default=7) + i * .2) for i, (k, v) in enumerate(list(q.items())[:2])], L, H * 0.50, W * 0.88, fmt="{:.4f}")
    elif rid == "R10":
        rows = [("🧮 tables", "gradient boosting"), ("📝 text", "Transformers"), ("🖼️ images", "CNN"), ("🏷️ no labels", "k-means")]
        for i, (a_, b_) in enumerate(rows):
            card(sc, f"r{i}", a_, b_, L, top + i * 150 * u, W * 0.90, .3 + i * .6, AMB, big=48)
        label(sc, "yt", "Full tutorial: YouTube @csfunda.official", L, H * 0.62, size=44, weight=900, color=AMB, t=at(sid, "youtube", default=8), w=W * .9)
        label(sc, "c", "Which one do you use? Comment ↓", L, H * 0.68, size=44, weight=900, t=at(sid, "comment", default=10), w=W * .9)


ORDER = [f"Y{i:02d}" for i in range(1, 48)]
REEL = [f"Q{i:02d}" for i in range(1, 11)]
for i in range(1, 11):   # the Reel's voice lines are R01..R10 in the script
    b.SCENES[f"Q{i:02d}"] = {**meta[f"R{i:02d}"], "id": f"Q{i:02d}"}
    for src_, dst_ in [(VID / f"words/R{i:02d}.words.json", VID / f"words/Q{i:02d}.words.json"), (HERE / f"assets/vo/R{i:02d}.wav", HERE / f"assets/vo/Q{i:02d}.wav")]:
        if src_.exists(): dst_.write_bytes(src_.read_bytes())
b.S, b.ORDER = S_ML, ORDER


def badge(out, W, H, reel, total):
    u = min(W, H) / 1080
    (out / "compositions/badge.html").write_text(f"""<template>
<style>#root{{position:absolute;inset:0;}}
#root #bd{{position:absolute;right:{W * 0.04}px;top:{H * 0.025}px;font-family:Inter,sans-serif;font-weight:800;color:#F3F0E8;font-size:{26 * u}px;
 padding:{8 * u}px {16 * u}px;border-radius:{12 * u}px;background:rgba(20,21,23,.8);opacity:0;}}#root #bd b{{color:{AMB}}}</style>
<div id="root" data-composition-id="badge" data-width="{W}" data-height="{H}"><div id="bd">CS Funda · <b>ML Models</b></div></div>
<script>const tl=gsap.timeline({{paused:true}});tl.fromTo("#bd",{{opacity:0}},{{opacity:1,duration:.5}},3.0);window.__timelines["badge"]=tl;</script></template>""")
    return (f'<div id="badge" data-composition-id="badge" data-composition-src="compositions/badge.html" data-start="0" data-duration="{total}" '
            f'data-track-index="6" data-track-kind="graphics" data-width="{W}" data-height="{H}" style="z-index:6"></div>')


b.part_badge = badge


def add_extra_audio(out):
    idx = out / "index.html"; h = idx.read_text()
    starts = {s.upper(): float(a) for s, a in re.findall(r'compositions/([yq]\d+)\.html" data-start="([\d.]+)"', h)}
    lines = []
    for sid, evs in EXTRA.items():
        if sid not in starts: continue
        for k, (t, name, vol) in enumerate(evs):
            lines.append(f'<audio id="x-{sid.lower()}-{k}" class="clip" src="assets/sfx/{name}.mp3" data-start="{starts[sid] + t:.2f}" '
                         f'data-duration="{b.SFXLEN[name]}" data-track-index="{30 + k % 6}" data-volume="{vol}"></audio>')
    idx.write_text(h.replace("</div>\n<script>", "\n".join(lines) + "\n</div>\n<script>", 1))


if __name__ == "__main__":
    b.SFXLEN = {f.stem: round(float(__import__("subprocess").run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                capture_output=True, text=True).stdout), 2) for f in (HERE / "assets/sfx").glob("*.mp3")}
    EXTRA.clear()
    if len(sys.argv) > 1 and sys.argv[1] == "reel":
        b.LEAD, b.TAIL = 0.1, 0.3
        out = VID / "reel"; shutil.rmtree(out, ignore_errors=True)
        b.project(out, 1080, 1920, REEL, reel=1)
        add_extra_audio(out)
    else:
        b.project(HERE, 1920, 1080, ORDER)
        add_extra_audio(HERE)
