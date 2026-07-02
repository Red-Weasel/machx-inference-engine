#!/usr/bin/env python3
"""
IE Native Engine vs llama.cpp (best backend) — per-model one-pager.

Reproducible generator for Visual-Diagrams/ie_vs_llamacpp_2026-06-12.{pdf,png}.
Every number traces to docs/benchmark_matrix_2026-06-09.md (recorded; CPU-only,
no GPU run). Run inside the venv that has matplotlib:

    ~/venv/bin/python "Visual-Diagrams/ie_vs_llamacpp_2026-06-12.py"

Status colours: WIN (green) / COMPETITIVE (yellow) / BEHIND (red), judged on the
worse of prefill/decode unless noted.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
INK = "#10223a"
SUBINK = "#4a5b72"
IE_BLUE = "#1565c0"
LL_PURPLE = "#6a1b9a"
GREEN = "#1f8a4c"
GREEN_BG = "#e4f4ea"
YELLOW = "#c98a00"
YELLOW_BG = "#fdf3da"
RED = "#c62828"
RED_BG = "#fbe6e6"
GREY_BG = "#f4f6fa"
LINE = "#d4dae6"

STATUS = {
    "WIN": (GREEN, GREEN_BG),
    "COMPETITIVE": (YELLOW, YELLOW_BG),
    "BEHIND": (RED, RED_BG),
}

# ----------------------------------------------------------------------------
# Recorded data — all from docs/benchmark_matrix_2026-06-09.md
# cols: model, sub, llama-backend, pp_ie, pp_ll, pp_ratio, tg_ie, tg_ll, tg_ratio,
#       ppl, status, lever
# ----------------------------------------------------------------------------
ROWS = [
    dict(
        model="Qwen3.6-35B-A3B  —  CROWN (MoE+DeltaNet)",
        sub="the model the engine was built for — beats llama on BOTH",
        backend="SYCL master",
        pp_ie="1144", pp_ll="1064", pp_r="+7.6%", pp_state="WIN",
        tg_ie="84.1", tg_ll="81.31", tg_r="+3.5%", tg_state="WIN",
        ppl="6.45", status="WIN",
        lever="At the front. Bespoke int-dot Q8 MoE kernel (E_ffn=512, top-2).",
    ),
    dict(
        model="Qwen3.6-27B  —  DeltaNet hybrid",
        sub="hybrid DeltaNet + dense MLP",
        backend="Vulkan",
        pp_ie="577", pp_ll="303", pp_r="1.9×", pp_state="WIN",
        tg_ie="10.0", tg_ll="9.72", tg_r="parity", tg_state="COMPETITIVE",
        ppl="5.34", status="WIN",
        lever="Prefill wins big (oneDNN GEMM); decode at parity.",
    ),
    dict(
        model="Qwen3 / Qwen2 dense  (8B)",
        sub="dense family baseline (Mistral / Llama / Phi / distills inherit this)",
        backend="SYCL",
        pp_ie="1190", pp_ll="1036", pp_r="+14.9%", pp_state="WIN",
        tg_ie="43.7", tg_ll="77.7", tg_r="0.56×", tg_state="BEHIND",
        ppl="2.94", status="COMPETITIVE",
        lever="Prefill wins; DECODE behind — the Q6_K GEMV cliff (Lever 2).",
    ),
    dict(
        model="Qwen3-Coder-30B-A3B  —  qwen3moe",
        sub="standard top-8 MoE (E_ffn=768)",
        backend="Vulkan",
        pp_ie="651", pp_ll="984", pp_r="0.66×", pp_state="COMPETITIVE",
        tg_ie="37.4", tg_ll="58.6", tg_r="0.64×", tg_state="COMPETITIVE",
        ppl="11.98", status="COMPETITIVE",
        lever="PREFILL 8.1× (GPU router + generalized int-dot down) — was 0.10×.",
    ),
]

# Validated-correctness (ride the dense profile) + multi-GPU note
NOTE_ROWS = [
    dict(
        model="Wave-1 validated (correctness; ride dense path)",
        body="Mistral-Small-3.2-24B (PPL 7.42) · DeepSeek-R1-Distill-Qwen/Llama "
             "· Phi-4 (PPL 8.25). Perf inherits the dense profile: prefill "
             "competitive, decode behind (Lever 2).",
    ),
    dict(
        model="Multi-GPU 72B (tensor-parallel)",
        body="decode 10.4 tok/s = 1.44× over our own layer-split (NOT a llama "
             "A/B) · PPL 8.97.",
    ),
]

# ----------------------------------------------------------------------------
# Figure  (one page, portrait, fits A4/Letter)
# ----------------------------------------------------------------------------
W, H = 13.2, 17.0
fig = plt.figure(figsize=(W, H), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def box(x, y, w, h, fc, ec="none", lw=0, r=0.6, z=1, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha,
                       mutation_aspect=H / W)
    ax.add_patch(p)
    return p

def txt(x, y, s, size, color=INK, weight="normal", ha="left", va="center",
        family="DejaVu Sans", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va,
            family=family, style=style, zorder=10)

# ---- Title band -------------------------------------------------------------
box(0, 95.4, 100, 4.6, INK, z=1)
txt(3.2, 98.4, "Our Engine  vs  llama.cpp", 25, "white", "bold", va="center")
txt(3.2, 96.3, "best-backend, per-model head-to-head  —  where we win, "
               "and where to focus next", 12.5, "#aebfd6", va="center")
txt(97, 98.4, "2026-06-12", 12, "#aebfd6", ha="right", va="center")
txt(97, 96.5, "Intel Arc Pro B70 (BMG-G31) 32 GB", 10.5, "#aebfd6",
    ha="right", va="center")

# ---- Legend strip -----------------------------------------------------------
ly = 93.0
def chip(x, label, fg, bg):
    box(x, ly - 0.95, 2.2, 1.9, bg, ec=fg, lw=1.2, r=0.4)
    txt(x + 3.0, ly, label, 10.5, fg, "bold", va="center")
chip(3.2, "WIN", GREEN, GREEN_BG)
chip(16.5, "COMPETITIVE", YELLOW, YELLOW_BG)
chip(36.5, "BEHIND", RED, RED_BG)
txt(97, ly, "tok/s  ·  ours / llama / ratio  ·  PPL = quality "
            "(lower better)", 10, SUBINK, ha="right", va="center")

# ---- Column headers ---------------------------------------------------------
hy = 89.6
box(2.5, hy - 1.2, 95, 2.4, "#27405f", z=1)
COL_PP = 49.0
COL_TG = 70.5
COL_PPL = 90.5
txt(3.6, hy, "MODEL", 11, "white", "bold", va="center")
txt(COL_PP, hy, "PREFILL  pp512", 10.5, "white", "bold", ha="center", va="center")
txt(COL_TG, hy, "DECODE  tg128", 10.5, "white", "bold", ha="center", va="center")
txt(COL_PPL, hy, "PPL", 10.5, "white", "bold", ha="center", va="center")

# ---- Model rows -------------------------------------------------------------
row_h = 8.7
gap = 0.7
top = hy - 2.6
y = top

def metric_cell(cx, cy, ie_v, ll_v, ratio, state):
    fg, bg = STATUS[state]
    # ratio pill
    box(cx - 6.6, cy + 0.55, 13.2, 2.0, bg, ec=fg, lw=1.1, r=0.45)
    txt(cx, cy + 1.55, ratio, 13.5, fg, "bold", ha="center", va="center")
    # ours / llama
    txt(cx, cy - 0.95, f"{ie_v}", 12.5, IE_BLUE, "bold", ha="right", va="center")
    txt(cx + 0.4, cy - 0.95, "  /  ", 10.5, SUBINK, ha="center", va="center")
    txt(cx + 0.9, cy - 0.95, f"{ll_v}", 12.5, LL_PURPLE, "bold", ha="left",
        va="center")

for i, r in enumerate(ROWS):
    yc = y - row_h / 2
    fg, bg = STATUS[r["status"]]
    # row card
    box(2.5, y - row_h, 95, row_h, "white", ec=LINE, lw=1.0, r=0.6)
    # status accent bar
    box(2.5, y - row_h, 1.1, row_h, fg, r=0.5)
    # model name + sub + lever
    txt(5.2, yc + 2.7, r["model"], 13.5, INK, "bold", va="center")
    txt(5.2, yc + 0.75, r["sub"], 9.8, SUBINK, style="italic", va="center")
    # lever line
    txt(5.2, yc - 1.5, "▸ " + r["lever"], 9.6, fg, "bold", va="center")
    txt(5.2, yc - 3.35, f"llama backend: {r['backend']}", 9.0, SUBINK,
        va="center")
    # metric cells
    metric_cell(COL_PP, yc, r["pp_ie"], r["pp_ll"], r["pp_r"], r["pp_state"])
    metric_cell(COL_TG, yc, r["tg_ie"], r["tg_ll"], r["tg_r"], r["tg_state"])
    # ppl
    txt(COL_PPL, yc, r["ppl"], 14, INK, "bold", ha="center", va="center")
    y -= row_h + gap

# ---- ours/llama colour key under rows --------------------------------------
ky = y - 0.4
txt(COL_PP, ky, "ours", 9, IE_BLUE, "bold", ha="right", va="center")
txt(COL_PP + 0.9, ky, "llama", 9, LL_PURPLE, "bold", ha="left", va="center")
txt(COL_PP + 0.45, ky, "/", 9, SUBINK, ha="center", va="center")
y = ky - 1.4

# ---- Validated / multi-GPU note band ---------------------------------------
nb_h = 5.6
box(2.5, y - nb_h, 95, nb_h, GREY_BG, ec=LINE, lw=1.0, r=0.6)
ty = y - 1.4
for r in NOTE_ROWS:
    txt(4.4, ty, r["model"] + ":", 9.6, INK, "bold", va="top")
    txt(4.4, ty - 1.35, r["body"], 9.0, SUBINK, va="top")
    ty -= 2.7
y -= nb_h + 1.2

# ---- FOCUS NEXT box ---------------------------------------------------------
fb_h = 14.6
box(2.5, y - fb_h, 95, fb_h, "#0e2440", r=0.7)
box(2.5, y - 2.3, 95, 2.3, "#16365c", r=0.7)
txt(4.4, y - 1.15, "FOCUS NEXT  —  two transferable kernel investments that "
                   "turn broad coverage into broad competitive perf",
    13, "white", "bold", va="center")

def lever(lx, ly0, lw, num, title, body, foot, accent):
    box(lx, ly0 - 9.6, lw, 9.6, "#13294a", ec=accent, lw=1.4, r=0.6)
    box(lx + 0.9, ly0 - 2.4, 2.5, 1.9, accent, r=0.4)
    txt(lx + 2.15, ly0 - 1.45, num, 13, "#0e2440", "bold", ha="center",
        va="center")
    txt(lx + 4.1, ly0 - 1.45, title, 11.5, "white", "bold", va="center")
    # wrap body manually
    yy = ly0 - 3.55
    for line in body:
        txt(lx + 1.0, yy, line, 9.3, "#cdd8e8", va="center")
        yy -= 1.4
    txt(lx + 1.0, ly0 - 9.0, foot, 9.1, accent, "bold", va="center")

lever(
    4.4, y - 2.9, 45.0, "✓",
    "MoE-expert GEMM kernel  —  DONE (8.1×)",
    [
        "GPU-gemm router (host loop was 66% of",
        "prefill wall) + generalized int-dot W4A8",
        "down for E_ffn≠512 (new moe_qwen3.cpp;",
        "crown moe_fused.cpp untouched).",
    ],
    "→ qwen3moe prefill 80.6→651 (12×→1.5× behind); MoE shapes now Next-80B-ready",
    "#7CFFB2",
)
lever(
    52.0, y - 2.9, 45.0, "2",
    "Dense-decode GEMV",
    [
        "The Q6_K cliff / repacked / split-K (the",
        "P3b known-hard). Closes the dense-family",
        "decode loss (43.7 vs 77.7) shared by",
        "Mistral / Llama / Phi / distills.",
    ],
    "→ turns the whole validated dense family competitive on decode",
    "#7fd1ff",
)
y -= fb_h + 1.0

# ---- Backend-fairness caveat footnote --------------------------------------
cf_h = 4.2
box(2.5, y - cf_h, 95, cf_h, YELLOW_BG, ec=YELLOW, lw=1.1, r=0.5)
txt(4.4, y - 1.25, "⚠  Backend-fairness caveat", 10, YELLOW, "bold",
    va="center")
txt(4.4, y - 2.55,
    "Crown + dense use llama SYCL (llama's strongest Intel backend); the 27B + "
    "qwen3moe rows use llama Vulkan (it has the GPU DeltaNet",
    8.8, "#7a5a00", va="center")
txt(4.4, y - 3.6,
    "shaders / was on hand). A llama SYCL re-run is pending for the fairest "
    "qwen3moe gap — treat that ratio as approximate, not exact.",
    8.8, "#7a5a00", va="center")
y -= cf_h + 0.5

txt(50, y - 0.8,
    "Same physical B70, model VRAM-resident for both engines.  Every figure "
    "traces to docs/benchmark_matrix_2026-06-09.md  ·  no GPU run for this chart.",
    8.3, SUBINK, ha="center", va="center")

# ----------------------------------------------------------------------------
out_base = __file__.rsplit(".", 1)[0]
fig.savefig(out_base + ".pdf", dpi=150)
fig.savefig(out_base + ".png", dpi=150)
print("wrote", out_base + ".pdf", "and", out_base + ".png")
