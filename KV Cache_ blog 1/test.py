

"""
kv_experiments.py: every experiment and figure for the KV-cache post.
 
Put this next to your model.py, then:
 
    python kv_experiments.py               # run all experiments on the GPU, then draw figures
    python kv_experiments.py --runs 7      # more timing runs (median is taken per position)
    python kv_experiments.py --plots-only  # redraw figures from results/results.json (no GPU needed)
 
Needs from model.py:
    GPT, GPTconfig, load_hf_weights
    GPT.forward(idx, past_kvs=None) -> (logits, new_kvs)
    Block.forward(x, past_kv=None)  -> (x, present)
 
Outputs (in ./results):
    results.json   raw numbers for every experiment
    summary.md     the key numbers, formatted for the post
    figs/*.png     figures (200 dpi) + figs/*.svg (crisp for the web)
 
Hand-drawn look: drop a handwriting .ttf into ./fonts (recommended: xkcd-script.ttf
from github.com/ipython/xkcd-font, or "Patrick Hand" from Google Fonts). Without one,
matplotlib falls back to its default font with the sketchy xkcd lines still on.
"""
 
import argparse
import glob
import json
import logging
import os
import time
 
import numpy as np
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
 
OUT = "results"
FIGS = os.path.join(OUT, "figs")
PROMPT = "Hello, I'm a language model,"
 
# Known fp32 (non-tensor-core) peaks, for the utilization figure.
PEAK_FP32 = {"A40": 37.4e12, "A100": 19.5e12, "RTX 4090": 82.6e12, "RTX 4050": 8.9e12}
 
 
# =============================================================================
# Experiments (need torch + a GPU)
# =============================================================================
 
def run_experiments(n_runs):
    import torch
    import torch.nn as nn
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from model import GPT, GPTconfig, load_hf_weights
 
    assert torch.cuda.is_available(), "CUDA GPU required for the timing experiments"
    torch.backends.cuda.matmul.allow_tf32 = False  # honest fp32
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda"
    sync = torch.cuda.synchronize
 
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    cfg = GPTconfig()
    model = load_hf_weights(GPT(cfg)).to(dev).eval()
    idx0 = torch.tensor([tok.encode(PROMPT)], device=dev)
    max_new = cfg.block_size - idx0.size(1)
    gpu = torch.cuda.get_device_name(0)
    R = {"gpu": gpu, "prompt": PROMPT, "n_runs": n_runs, "config": cfg.__dict__}
    print(f"[setup] {gpu}, prompt={idx0.size(1)} tokens, generating up to {cfg.block_size}")
 
    # ---- E1: correctness vs Hugging Face -----------------------------------
    hf = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    with torch.no_grad():
        mine, _ = model(idx0)
        ref = hf(idx0).logits
    R["hf_max_abs_logit_diff"] = (mine - ref).abs().max().item()
    del hf
    torch.cuda.empty_cache()
    print(f"[E1] max |logit diff| vs HF: {R['hf_max_abs_logit_diff']:.3e}")
 
    # ---- generation helpers -------------------------------------------------
    @torch.no_grad()
    def gen_nocache(idx, n):
        for _ in range(n):
            logits, _ = model(idx)
            idx = torch.cat((idx, logits[:, -1, :].argmax(-1, keepdim=True)), dim=1)
        return idx
 
    @torch.no_grad()
    def gen_cached(idx, n, fwd=model):
        logits, kvs = fwd(idx)
        for _ in range(n):
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            idx = torch.cat((idx, nxt), dim=1)
            logits, kvs = fwd(nxt, kvs)
        return idx
 
    # ---- E2: cached == no-cache ---------------------------------------------
    N_EQ = 200
    a, b = gen_nocache(idx0, N_EQ), gen_cached(idx0, N_EQ)
    eq = torch.equal(a, b)
    R["cache_equal"] = {"tokens": N_EQ, "equal": eq}
    if not eq:
        first = int((a[0] != b[0]).nonzero()[0])
        with torch.no_grad():
            lg, _ = model(a[:, :first])
        top2 = lg[0, -1].topk(2).values
        R["cache_equal"].update(first_divergence=first,
                                top2_logit_gap=(top2[0] - top2[1]).item())
    print(f"[E2] cached == no-cache over {N_EQ} tokens: {R['cache_equal']}")
 
    # ---- E3: per-step timing ------------------------------------------------
    @torch.no_grad()
    def time_run(use_cache):
        idx = idx0.clone()
        lens, ms = [], []
        if use_cache:
            logits, kvs = model(idx)                       # prefill, not timed
        for _ in range(max_new):
            sync(); t0 = time.perf_counter()
            if use_cache:
                nxt = logits[:, -1, :].argmax(-1, keepdim=True)
                logits, kvs = model(nxt, kvs)
                L = kvs[0][0].size(2)                      # context this token attended over
            else:
                logits, _ = model(idx)
                nxt = logits[:, -1, :].argmax(-1, keepdim=True)
                L = idx.size(1)
            sync()
            ms.append((time.perf_counter() - t0) * 1e3)
            lens.append(L)
            idx = torch.cat((idx, nxt), dim=1)
        return lens, ms
 
    for use_cache in (False, True):                       # warmup both paths
        with torch.no_grad():
            if use_cache:
                gen_cached(idx0, 20)
            else:
                gen_nocache(idx0, 20)
    sync()
 
    timing = {}
    for name, use_cache in (("nocache", False), ("cache", True)):
        runs = []
        for r in range(n_runs):
            lens, ms = time_run(use_cache)
            runs.append(ms)
            print(f"[E3] {name} run {r + 1}/{n_runs}: total {sum(ms) / 1e3:.2f} s")
        timing[name] = {"lens": lens, "runs": runs}
    R["timing"] = timing
 
    # ---- E4: FLOP model (counted from the model itself) ---------------------
    n_lin = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    R["flops"] = {"linear_weights": n_lin, "n_layer": cfg.n_layer, "n_embd": cfg.n_embd}
    peak = next((v for k, v in PEAK_FP32.items() if k in gpu), None)
    R["peak_fp32_flops"] = peak
 
    # ---- E5: memory ---------------------------------------------------------
    elt = next(model.parameters()).element_size()
    per_tok = 2 * cfg.n_layer * cfg.n_embd * elt
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    masks = sum(b.numel() * b.element_size() for n, b in model.named_buffers() if n.endswith("mask"))
    checkpoints = set(range(64, cfg.block_size + 1, 64))
    measured = []
    with torch.no_grad():
        torch.cuda.empty_cache(); sync()
        base = torch.cuda.memory_allocated()
        logits, kvs = model(idx0)
        idx = idx0.clone()
        while idx.size(1) < cfg.block_size:
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            idx = torch.cat((idx, nxt), dim=1)
            logits, kvs = model(nxt, kvs)
            L = kvs[0][0].size(2)
            if L in checkpoints:
                sync()
                extra = logits.numel() * logits.element_size() + idx.numel() * idx.element_size()
                measured.append((L, torch.cuda.memory_allocated() - base - extra))
    R["memory"] = {"bytes_per_token_formula": per_tok, "weights_bytes": weights,
                   "mask_buffers_bytes": masks, "measured": measured}
    print(f"[E5] KV cache: {per_tok / 1024:.0f} KiB/token formula; "
          f"measured at {measured[-1][0]}: {measured[-1][1] / 2**20:.1f} MiB")
 
    # ---- E6: wrong positions on purpose -------------------------------------
    def forward_wrong_pos(idx, past_kvs=None):
        T = idx.size(1)
        pos = torch.arange(0, T, device=idx.device)       # BUG on purpose: ignores T_past
        x = model.transformer.wte(idx) + model.transformer.wpe(pos)
        new = []
        for i, blk in enumerate(model.transformer.h):
            x, p = blk(x, None if past_kvs is None else past_kvs[i])
            new.append(p)
        return model.lm_head(model.transformer.ln_f(x)), new
 
    good = gen_cached(idx0, 60)
    bad = gen_cached(idx0, 60, fwd=forward_wrong_pos)
    R["positions_demo"] = {"correct": tok.decode(good[0]), "wrong_positions": tok.decode(bad[0])}
    print("[E6] wrong-positions text:\n   ", R["positions_demo"]["wrong_positions"][:200])
 
    return R
 
 
# =============================================================================
# Derived numbers
# =============================================================================
 
def derive(R):
    t = np.array(R["timing"]["nocache"]["lens"], dtype=float)
    nc = np.median(np.array(R["timing"]["nocache"]["runs"]), axis=0)
    ca = np.median(np.array(R["timing"]["cache"]["runs"]), axis=0)
    f = R["flops"]
    att = 4 * f["n_layer"] * f["n_embd"]                       # attention FLOPs coefficient
    fl_nc = 2 * f["linear_weights"] * t + att * t ** 2          # = t * fl_ca
    fl_ca = 2 * f["linear_weights"] + att * t
    return dict(t=t, nc=nc, ca=ca, fl_nc=fl_nc, fl_ca=fl_ca,
                speedup=nc / ca, flop_ratio=fl_nc / fl_ca,
                cum_nc=np.cumsum(nc) / 1e3, cum_ca=np.cumsum(ca) / 1e3)
 
 
def at(D, key, L):
    i = int(np.argmin(np.abs(D["t"] - L)))
    return float(D[key][i])
 
 
def write_summary(R, D):
    m = R["memory"]
    peak = R.get("peak_fp32_flops")
    rows = []
    for L in (10, 100, 500, 1000):
        util_nc = at(D, "fl_nc", L) / (at(D, "nc", L) / 1e3)
        util_ca = at(D, "fl_ca", L) / (at(D, "ca", L) / 1e3)
        pct = (lambda u: f"{100 * u / peak:.2f}%") if peak else (lambda u: "n/a")
        rows.append(f"| {L} | {at(D, 'nc', L):.2f} | {at(D, 'ca', L):.2f} | "
                    f"{at(D, 'speedup', L):.1f}× | {at(D, 'flop_ratio', L):.0f}× | "
                    f"{util_nc / 1e12:.2f} ({pct(util_nc)}) | {util_ca / 1e12:.3f} ({pct(util_ca)}) |")
    L_last, b_last = m["measured"][-1]
    s = f"""# KV cache: key numbers
 
Setup: GPT-2 small (124M, from scratch), {R['gpu']}, fp32, TF32 off, batch 1,
greedy decoding, prompt "{R['prompt']}", median of {R['n_runs']} runs per position.
 
## Correctness
- Max |logit diff| vs Hugging Face GPT-2: **{R['hf_max_abs_logit_diff']:.2e}**
- Cached generation == no-cache generation ({R['cache_equal']['tokens']} tokens): **{R['cache_equal']['equal']}**
 
## Speed (ms per generated token)
 
| context t | no cache | KV cache | measured speedup | FLOP reduction | no-cache TFLOPS (% peak) | cached TFLOPS (% peak) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}
 
Total time to generate to {int(D['t'][-1]) + 1} tokens: **{D['cum_nc'][-1]:.2f} s** without cache vs **{D['cum_ca'][-1]:.2f} s** with it.
 
## Memory
- KV cache per token (formula 2 × layers × d_model × bytes): **{m['bytes_per_token_formula'] / 1024:.0f} KiB**
- Formula at {L_last} tokens: **{m['bytes_per_token_formula'] * L_last / 2**20:.1f} MiB**; measured: **{b_last / 2**20:.1f} MiB**
- Weights: **{m['weights_bytes'] / 2**20:.0f} MiB**, so the cache outgrows the weights at
  batch × context ≈ **{m['weights_bytes'] / m['bytes_per_token_formula']:,.0f} tokens** (≈ batch {m['weights_bytes'] / (m['bytes_per_token_formula'] * 1024):.1f} at full context)
- Bonus: the 12 causal-mask buffers cost **{m['mask_buffers_bytes'] / 2**20:.0f} MiB** (one 1024×1024 fp32 mask per layer)
 
## Wrong positions (arange(0, T) during decode): no crash, silently wrong
Correct:  {R['positions_demo']['correct']!r}
 
Wrong:    {R['positions_demo']['wrong_positions']!r}
"""
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(s)
    print(s)
 
 
# =============================================================================
# Hand-drawn figure style
# =============================================================================
 
PAPER = "#faf6ee"     # warm paper
INK = "#2b2a27"       # near-black ink for text, axes
MUTED = "#7a776f"     # secondary ink
GRID = "#e6dfd1"
NOCACHE = "#eb6834"   # categorical slot 2 (orange)
CACHE = "#2a78d6"     # categorical slot 1 (blue)
MEASURED = "#1baf7a"  # categorical slot 3 (aqua)
# Validated together on PAPER (all checks pass; contrast < 3:1 for orange/aqua,
# so every series is direct-labeled, never identified by color alone).
 
 
def find_hand_font():
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    for path in sorted(glob.glob("fonts/*.ttf") + glob.glob("fonts/*.otf")):
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path).get_name()
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("xkcd Script", "xkcd", "Humor Sans", "Patrick Hand", "Comic Neue"):
        if name in installed:
            return name
    return None
 
 
HAND_FONT = None
 
 
def sketch(fig_w=9, fig_h=5.2):
    """Open a hand-drawn figure. Returns (fig, ax)."""
    plt.rcParams.update({
        "font.family": [HAND_FONT, "DejaVu Sans"] if HAND_FONT else ["DejaVu Sans"],
        "font.size": 14,
        "path.effects": [patheffects.withStroke(linewidth=3, foreground=PAPER)],
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 1.4,
        "xtick.major.size": 6, "ytick.major.size": 6,
    })
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    return fig, ax
 
 
def titled(fig, ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=19, pad=30, color=INK)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=11.5, color=MUTED, va="bottom")
 
 
def label_end(ax, x, y, text, color, dy=0, ha="left"):
    """Direct label at a line's end, in ink (a color dot carries identity)."""
    ax.plot([x], [y], "o", ms=7, color=color, zorder=5)
    ax.annotate(text, (x, y), xytext=(10, dy), textcoords="offset points",
                fontsize=13, color=INK, va="center", ha=ha)
 
 
def note(ax, text, xy, xytext, rad=0.25):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=12, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3,
                                connectionstyle=f"arc3,rad={rad}"))
 
 
def plain_log_ticks(ax, values, fmt):
    """Log axis with plain labels (1x, 10x ...) instead of 10^n."""
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.yaxis.set_major_locator(FixedLocator(values))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([fmt(v) for v in values])
    ax.set_ylim(values[0] * 0.7, values[-1] * 1.6)
 
 
def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIGS}/{name}.png")
 
 
# =============================================================================
# Figures
# =============================================================================
 
def setup_line(R):
    return f"GPT-2 small · {R['gpu']} · fp32 · batch 1 · median of {R['n_runs']} runs"
 
 
def fig1_nocache(R, D):
    t, nc = D["t"], D["nc"]
    raw = np.array(R["timing"]["nocache"]["runs"])
    fig, ax = sketch()
    ax.scatter(np.tile(t, raw.shape[0]), raw.ravel(), s=3, color=NOCACHE, alpha=0.18,
               linewidths=0, zorder=2)
    ax.plot(t, nc, color=NOCACHE, lw=2.4, zorder=3)
    floor = float(np.median(nc[t < 100]))
    titled(fig, ax, "Without a cache, every token costs more than the last", setup_line(R))
    ax.set_xlabel("tokens in context")
    ax.set_ylabel("ms per generated token")
    ax.set_ylim(0, nc.max() * 1.15)
    ax.set_xlim(0, t.max() * 1.12)
    note(ax, f"flat ≈ {floor:.0f} ms: the GPU is mostly\nwaiting on Python + kernel launches",
         xy=(60, floor), xytext=(120, floor + 0.45 * (nc.max() - floor)), rad=0.3)
    x2 = 0.72 * t.max()
    note(ax, "each step re-runs every layer\non every old token",
         xy=(x2, at(D, "nc", x2)), xytext=(0.3 * t.max(), 0.82 * nc.max()), rad=-0.25)
    label_end(ax, t[-1], nc[-1], f"{nc[-1]:.0f} ms", NOCACHE)
    save(fig, "fig1_no_cache")
 
 
def fig2_cache_vs_nocache(R, D):
    t, nc, ca = D["t"], D["nc"], D["ca"]
    fig, ax = sketch()
    ax.plot(t, nc, color=NOCACHE, lw=2.4)
    ax.plot(t, ca, color=CACHE, lw=2.4)
    titled(fig, ax, "The cache flattens the curve, down to a floor", setup_line(R))
    ax.set_xlabel("tokens in context")
    ax.set_ylabel("ms per generated token")
    ax.set_ylim(0, nc.max() * 1.15)
    ax.set_xlim(0, t.max() * 1.2)
    label_end(ax, t[-1], nc[-1], "no cache", NOCACHE)
    label_end(ax, t[-1], ca[-1], "KV cache", CACHE)
    xm = 0.97 * t.max()
    y_nc, y_ca = at(D, "nc", xm), at(D, "ca", xm)
    ax.annotate("", xy=(xm, y_ca), xytext=(xm, y_nc),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(xm - 15, (y_nc + y_ca) / 2, f"{D['speedup'][-1]:.1f}× faster", ha="right",
            va="center", fontsize=14, color=INK)
    save(fig, "fig2_cache_vs_nocache")
 
 
def fig3_total_time(R, D):
    n = np.arange(1, len(D["t"]) + 1)
    fig, ax = sketch()
    ax.plot(n, D["cum_nc"], color=NOCACHE, lw=2.4)
    ax.plot(n, D["cum_ca"], color=CACHE, lw=2.4)
    titled(fig, ax, "Per-step cost is linear, so total time is quadratic",
           setup_line(R) + " · cumulative")
    ax.set_xlabel("tokens generated")
    ax.set_ylabel("total seconds")
    ax.set_xlim(0, n.max() * 1.2)
    ax.set_ylim(0, D["cum_nc"][-1] * 1.12)
    label_end(ax, n[-1], D["cum_nc"][-1], f"no cache: {D['cum_nc'][-1]:.1f} s", NOCACHE)
    label_end(ax, n[-1], D["cum_ca"][-1], f"KV cache: {D['cum_ca'][-1]:.1f} s", CACHE)
    save(fig, "fig3_total_time")
 
 
def fig4_flops_vs_speedup(R, D):
    t = D["t"]
    fig, ax = sketch()
    ax.plot(t, D["flop_ratio"], color=MUTED, lw=2, linestyle="--")
    ax.plot(t, D["speedup"], color=CACHE, lw=2.4)
    ax.set_yscale("log")
    plain_log_ticks(ax, [1, 10, 100, 1000], lambda v: f"{v:g}×")
    titled(fig, ax, "1000× less math is not 1000× less time",
           "how many times cheaper a cached step is · " + setup_line(R))
    ax.set_xlabel("tokens in context")
    ax.set_ylabel("× (no cache ÷ KV cache)")
    ax.set_xlim(0, t.max() * 1.25)
    label_end(ax, t[-1], D["flop_ratio"][-1], f"FLOPs: {D['flop_ratio'][-1]:.0f}× fewer", MUTED)
    label_end(ax, t[-1], D["speedup"][-1], f"time: {D['speedup'][-1]:.1f}× less", CACHE)
    x = 0.55 * t.max()
    note(ax, "the gap = fixed per-step overhead\nthe cache can't remove",
         xy=(x, np.sqrt(at(D, "flop_ratio", x) * at(D, "speedup", x))),
         xytext=(0.08 * t.max(), 0.6 * D["flop_ratio"].max()), rad=0.2)
    save(fig, "fig4_flops_vs_speedup")
 
 
def fig5_utilization(R, D):
    peak = R.get("peak_fp32_flops")
    if not peak:
        print("[fig] skipping utilization (unknown GPU peak)")
        return
    t = D["t"]
    u_nc = 100 * D["fl_nc"] / (D["nc"] / 1e3) / peak
    u_ca = 100 * D["fl_ca"] / (D["ca"] / 1e3) / peak
    fig, ax = sketch()
    ax.plot(t, u_nc, color=NOCACHE, lw=2.4)
    ax.plot(t, u_ca, color=CACHE, lw=2.4)
    ax.set_yscale("log")
    plain_log_ticks(ax, [0.1, 1, 10, 100], lambda v: f"{v:g}%")
    titled(fig, ax, "How busy is the GPU, really?",
           f"achieved FLOPS as % of {peak / 1e12:.1f} TFLOPS fp32 peak · " + setup_line(R))
    ax.set_xlabel("tokens in context")
    ax.set_ylabel("% of peak (log scale)")
    ax.set_xlim(0, t.max() * 1.3)
    label_end(ax, t[-1], u_nc[-1], f"no cache: {u_nc[-1]:.0f}%", NOCACHE)
    label_end(ax, t[-1], u_ca[-1], f"KV cache: {u_ca[-1]:.2f}%", CACHE)
    save(fig, "fig5_utilization")
 
 
def fig6_memory(R, D):
    m = R["memory"]
    per_tok_mib = m["bytes_per_token_formula"] / 2**20
    L = np.array([p[0] for p in m["measured"]])
    meas = np.array([p[1] for p in m["measured"]]) / 2**20
    xs = np.linspace(0, R["config"]["block_size"], 100)
    fig, ax = sketch()
    ax.plot(xs, per_tok_mib * xs, color=MUTED, lw=2, linestyle="--")
    ax.plot(L, meas, "o", color=MEASURED, ms=8)
    titled(fig, ax, f"{m['bytes_per_token_formula'] / 1024:.0f} KiB per token, every token",
           "KV cache size for one sequence · formula vs measured (torch.cuda.memory_allocated)")
    ax.set_xlabel("tokens in context")
    ax.set_ylabel("KV cache (MiB)")
    ax.set_xlim(0, xs.max() * 1.3)
    ax.set_ylim(0, max(meas.max(), per_tok_mib * xs.max()) * 1.2)
    label_end(ax, xs[-1], per_tok_mib * xs[-1], "formula: 2 · layers · d · bytes", MUTED, dy=14)
    label_end(ax, L[-1], meas[-1], f"measured: {meas[-1]:.0f} MiB", MEASURED, dy=-14)
    w = m["weights_bytes"] / (m["bytes_per_token_formula"] * R["config"]["block_size"])
    ax.text(0.03, 0.9, f"weights: {m['weights_bytes'] / 2**20:.0f} MiB, so at full context the cache\n"
                       f"outgrows the weights from batch ≈ {w:.0f}",
            transform=ax.transAxes, fontsize=12, color=INK, va="top")
    save(fig, "fig6_kv_memory")
 
 
def make_figures(R):
    global HAND_FONT
    HAND_FONT = find_hand_font()
    print(f"[fig] font: {HAND_FONT or 'default (add a .ttf to ./fonts for the hand-drawn font)'}")
    D = derive(R)
    write_summary(R, D)
    with plt.xkcd(scale=0.7, length=160, randomness=1.6):
        fig1_nocache(R, D)
        fig2_cache_vs_nocache(R, D)
        fig3_total_time(R, D)
        fig4_flops_vs_speedup(R, D)
        fig5_utilization(R, D)
        fig6_memory(R, D)
 
 
# =============================================================================
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.json")
    if args.plots_only:
        R = json.load(open(path, encoding="utf-8"))
    else:
        R = run_experiments(args.runs)
        json.dump(R, open(path, "w", encoding="utf-8"))
    make_figures(R)
 

