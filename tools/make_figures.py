"""Generate the README figures as dependency-free SVG from the project's real numbers.

Sources: swimming/logs/train.log (per-epoch validation accuracy), karate_vlm/results/*.json
(label counts per VLM), docs/PROJECT_STATE.md (window-length experiment, timings),
app/backend/app/services/pose_engine_service.py (pose-engine comparison).
usage: python tools/make_figures.py
"""
import json
import re
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "font-family='Segoe UI,Helvetica,Arial,sans-serif'"
C = ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]


def svg(w, h, body, name):
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' {FONT}>"
         f"<rect width='{w}' height='{h}' rx='12' fill='#ffffff' stroke='#d0d7de'/>{body}</svg>")
    (OUT / name).write_text(s, encoding="utf-8")


def text(x, y, t, size=13, fill="#1f2328", anchor="start", weight="normal"):
    return (f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}'>{escape(str(t))}</text>")


def box(x, y, w, h, label, sub="", fill="#eff6ff", stroke="#2563eb"):
    b = f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>"
    b += text(x + w / 2, y + (h / 2 - (4 if sub else -5)), label, 13, "#1f2328", "middle", "bold")
    if sub:
        b += text(x + w / 2, y + h / 2 + 14, sub, 11, "#57606a", "middle")
    return b


def arrow(x1, y1, x2, y2):
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#57606a' stroke-width='1.6'/>"
            f"<polygon points='{x2},{y2} {x2 - 8 * (1 if x2 >= x1 else -1) if y1 == y2 else x2 - 4},"
            f"{y2 - 4 if y1 == y2 else y2 - 8 * (1 if y2 >= y1 else -1)} "
            f"{x2 - 8 * (1 if x2 >= x1 else -1) if y1 == y2 else x2 + 4},"
            f"{y2 + 4 if y1 == y2 else y2 - 8 * (1 if y2 >= y1 else -1)}' fill='#57606a'/>")


# 1 ---------------------------------------------------------------- pipeline
b = text(20, 28, "Shared analysis pipeline (karate + swimming)", 16, weight="bold")
xs = [20, 190, 370, 550, 730]
labels = [("Video", "upload / webcam"), ("Pose engine", "OpenPose · RTMPose · YOLO"),
          ("Tracking + players", "ByteTrack · re-ID · Viterbi"), ("BODY_25 → COCO-17", "pure gather, 8 joints dropped"),
          ("Windows (1-2 s)", "30 / 60 frames")]
for x, (l, s) in zip(xs, labels):
    b += box(x, 60, 150, 56, l, s)
for a, c in zip(xs, xs[1:]):
    b += arrow(a + 150, 88, c, 88)
b += arrow(805, 116, 805, 160)
b += box(660, 160, 290, 56, "Skeleton classifier", "ST-GCN / PoseC3D (MMAction2)", "#f0fdf4", "#16a34a")
b += box(330, 160, 290, 56, "Vision-Language Model", "Qwen3-VL · Qwen3.5 · Gemma 3 (llama.cpp)", "#fff7ed", "#f97316")
b += arrow(660, 188, 620, 188)
b += box(20, 160, 270, 56, "Synthetic / MoCap data", "SwimXYZ (done) · Kyokushin (planned)", "#faf5ff", "#9333ea")
b += arrow(290, 188, 330, 188)
b += arrow(805, 216, 805, 262) + arrow(475, 216, 475, 262)
b += box(330, 262, 620, 50, "Timeline per player + HTML report + live view in the browser", "", "#f6f8fa", "#57606a")
svg(980, 330, b, "pipeline.svg")

# 2 ----------------------------------------------------- swimming learning curve
acc = {}
for l in (ROOT / "swimming/logs/train.log").read_text(encoding="utf-8", errors="ignore").splitlines():
    m = re.search(r"Epoch\(val\) \[(\d+)\].*acc/top1: ([\d.]+)", l)
    if m:
        acc[int(m.group(1))] = float(m.group(2))
W, H, L, T, R, Bm = 640, 340, 70, 50, 30, 50
b = text(20, 28, "Swimming ST-GCN — validation top-1 per epoch (synthetic held-out motions)", 14, weight="bold")
lo, hi = 0.95, 1.0
def px(e): return L + (e - 1) * (W - L - R) / 9
def py(a): return T + (hi - a) * (H - T - Bm) / (hi - lo)
for v in (0.95, 0.96, 0.97, 0.98, 0.99, 1.0):
    b += f"<line x1='{L}' y1='{py(v)}' x2='{W - R}' y2='{py(v)}' stroke='#eaeef2'/>" + text(L - 8, py(v) + 4, f"{v * 100:.0f}%", 11, "#57606a", "end")
pts = " ".join(f"{px(e):.1f},{py(a):.1f}" for e, a in sorted(acc.items()))
b += f"<polyline points='{pts}' fill='none' stroke='{C[0]}' stroke-width='2.5'/>"
for e, a in acc.items():
    b += f"<circle cx='{px(e):.1f}' cy='{py(a):.1f}' r='4' fill='{C[0]}'/>" + text(px(e), H - Bm + 20, e, 11, "#57606a", "middle")
b += text(px(9), py(acc[9]) - 10, f"best {acc[9] * 100:.2f}% (epoch 9)", 12, C[0], "middle", "bold")
b += text((L + W - R) / 2, H - 10, "epoch", 12, "#57606a", "middle")
b += text(20, H - 10, "synthetic data only", 11, "#dc2626")
svg(W, H, b, "swim_val_accuracy.svg")

# 3 ---------------------------------------------------------------- VLM labels
names = {"qwen3vl": "Qwen3-VL-4B", "qwen35": "Qwen3.5-4B", "gemma3": "Gemma 3 4B"}
cnt = {}
for k in names:
    r = json.loads((ROOT / f"karate_vlm/results/results_{k}.json").read_text(encoding="utf-8"))["results"]
    c = Counter(x[p]["label"] for x in r for p in ("player1", "player2"))
    cnt[k] = c
labs = [l for l, _ in sum(cnt.values(), Counter()).most_common() if l != "MOVEMENT"]
b = text(20, 28, "VLM label counts on the same match (276 player-windows each, MOVEMENT excluded)", 14, weight="bold")
x0, y0, rowh = 270, 60, 30
mx = max(cnt[k][l] for k in cnt for l in labs)
for i, l in enumerate(labs):
    y = y0 + i * rowh
    b += text(x0 - 10, y + 17, l, 11, "#1f2328", "end")
    for j, k in enumerate(names):
        w = 330 * cnt[k][l] / mx
        b += f"<rect x='{x0}' y='{y + j * 8}' width='{w:.1f}' height='7' fill='{C[j]}'/>"
        if cnt[k][l]:
            b += text(x0 + w + 4, y + j * 8 + 7, cnt[k][l], 9, "#57606a")
for j, k in enumerate(names):
    b += f"<rect x='{x0 + 380}' y='{50 + j * 18}' width='12' height='12' fill='{C[j]}'/>" + text(x0 + 398, 61 + j * 18, f"{names[k]} (MOVEMENT: {cnt[k]['MOVEMENT']})", 11)
svg(960, y0 + len(labs) * rowh + 20, b, "vlm_label_counts.svg")

# 4 ---------------------------------------------------------------- pose engines
def bars(title, rows, name, unit="%", maxv=100, color=None):
    b = text(20, 28, title, 14, weight="bold")
    for i, (l, v) in enumerate(rows):
        y = 55 + i * 34
        b += text(190, y + 16, l, 12, "#1f2328", "end")
        b += f"<rect x='200' y='{y}' width='{320 * v / maxv:.1f}' height='22' rx='4' fill='{(color or C)[i % len(color or C)]}'/>"
        b += text(205 + 320 * v / maxv, y + 16, f"{v:g}{unit}", 12, "#57606a")
    svg(600, 70 + len(rows) * 34, b, name)

bars("Swimmer found in frame, front-view clip (higher is better)",
     [("RTMDet + RTMPose", 69), ("YOLO11-pose", 34), ("OpenPose BODY_25", 25)], "pose_engines_swim.svg")
bars("Karate: strikes found vs window length (1 s is the optimum)",
     [("12 frames (0.4 s)", 0), ("30 frames (1.0 s)", 6), ("100 frames (3.3 s)", 1)], "window_length.svg", "", 6)
bars("OpenPose speed: seconds per frame (lower is better)",
     [("CPU build", 2.19), ("GPU build", 0.071)], "openpose_speed.svg", " s", 2.19)

# 5 ---------------------------------------------------------------- approaches map
b = text(20, 28, "Karate: every approach, and where it stands", 16, weight="bold")
rows = [
    ("A  Pose pipeline (OpenPose / RTMPose / YOLO)", "pose → track → players → windows", "BUILT", "#16a34a"),
    ("B  Object detection / person detection", "YOLO11 · RTMDet boxes, ByteTrack, referee filtering", "BUILT", "#16a34a"),
    ("C  Pretrained skeleton classifier (NTU60)", "ST-GCN + PoseC3D ensemble, out-of-domain", "BUILT, weak", "#ca8a04"),
    ("D  Vision-Language Model, zero-shot", "12 WKF labels, 3 VLMs compared", "PROTOTYPE", "#ca8a04"),
    ("E  Simulated / synthetic data", "SwimXYZ done · Kyokushin MoCap → 2D planned", "SWIM done, karate PLANNED", "#2563eb"),
    ("F  Data acquisition (public / gated sets)", "research done, requests not sent", "RESEARCHED", "#2563eb"),
    ("G  Own annotation + fine-tune", "VLM proposals → human review → ST-GCN", "NOT STARTED", "#dc2626"),
    ("H  Scoring-event model (yuko / waza-ari / ippon)", "needs separate labels and model", "NOT STARTED", "#dc2626"),
]
for i, (t, s, st, col) in enumerate(rows):
    y = 50 + i * 46
    b += f"<rect x='20' y='{y}' width='820' height='38' rx='8' fill='#f6f8fa' stroke='#d0d7de'/>"
    b += text(34, y + 17, t, 13, weight="bold") + text(34, y + 32, s, 11, "#57606a")
    b += f"<rect x='660' y='{y + 7}' width='170' height='24' rx='12' fill='{col}'/>" + text(745, y + 24, st, 11, "#ffffff", "middle", "bold")
svg(860, 50 + len(rows) * 46 + 10, b, "karate_approaches.svg")

# 6 ---------------------------------------------------------------- VLM flow
b = text(20, 28, "VLM labelling pipeline (karate_vlm/)", 16, weight="bold")
steps = [("windows.py", "Player 1/2 + 1.2 s windows,\n0.8 s stride"), ("make_grids.py", "2×4 numbered frame grid,\nblue / orange player boxes"),
         ("run_vlm.py", "llama.cpp server,\nJSON-schema answer"), ("report.py / compare.py", "HTML timelines +\nmodel comparison")]
for i, (t, s) in enumerate(steps):
    x = 20 + i * 235
    b += box(x, 60, 205, 70, t, "", "#fff7ed", "#f97316")
    for k, line in enumerate(s.split("\n")):
        b += text(x + 102, 108 + k * 13, line, 11, "#57606a", "middle")
    if i:
        b += arrow(x - 30, 95, x, 95)
b += text(20, 165, "Answer fields: what_happened → attack → side → kick_type → height → label (12 WKF labels) → confidence", 12, "#57606a")
svg(960, 185, b, "vlm_flow.svg")
print("written:", sorted(p.name for p in OUT.iterdir()))
