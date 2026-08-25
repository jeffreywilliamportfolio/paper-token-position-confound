#!/usr/bin/env python3
"""
Regenerate paper figures from verified archive data.

Every value plotted here traces to a source file in:
  moe-routing-organized/legacy-learning-runs/token-confound-archive/

Per-level routing-entropy means (all-token and last-token) are transcribed from
CROSS_MODEL_POSITION_CONFOUND.md (identical to the per-level tables in
journals/JOURNAL-ARCHIVE.md, "March 5: The Confound Discovery").

Per-level mean prompt-token counts for DeepSeek V3.1 are computed from
data/ds31-168q-1_prefill.json.

Summary Spearman correlations are transcribed from CROSS_MODEL_POSITION_CONFOUND.md.

Figure 3 (v1.1) plots the per-token position diagnostic (data/diagnostic_results.json,
Qwen 397B, five prompts) in place of the v1.0 Spearman bar chart, whose values
are carried by Table 1 of the paper.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCHIVE = "/Volumes/ExternalSSD/moe-routing-organized/legacy-learning-runs/token-confound-archive"
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

LEVELS = list(range(1, 13))

# --- Verified per-level means (CROSS_MODEL_POSITION_CONFOUND.md) -------------
# DeepSeek V3.1 (256 experts, 8 active, 58 MoE layers), n=14 per level
DS_ALL = [0.8370, 0.8300, 0.8337, 0.8431, 0.8653, 0.8552,
          0.8508, 0.8799, 0.8761, 0.8666, 0.8829, 0.8857]
DS_LAST = [0.8696, 0.8446, 0.8369, 0.8539, 0.8465, 0.8576,
           0.8421, 0.8605, 0.8472, 0.8548, 0.8548, 0.8476]
# Qwen 397B (512 experts, 10 active, 60 MoE layers), n=14 per level
QW_ALL = [0.8809, 0.8740, 0.8755, 0.8790, 0.8815, 0.8807,
          0.8782, 0.8937, 0.8830, 0.8848, 0.8876, 0.8856]
QW_LAST = [0.8841, 0.8787, 0.8769, 0.8734, 0.8754, 0.8713,
           0.8794, 0.8699, 0.8803, 0.8784, 0.8761, 0.8785]

# --- Verified summary Spearman rho (CROSS_MODEL_POSITION_CONFOUND.md) --------
RHO = {
    "DeepSeek V3.1": {
        "all_level": 0.8019, "all_tokens": 0.8797,
        "last_level": 0.0177, "last_tokens": 0.1608,
    },
    "Qwen 397B": {
        "all_level": 0.6166, "all_tokens": 0.7813,
        "last_level": -0.0622, "last_tokens": -0.2197,
    },
}

# --- Computed per-level mean tokens, DeepSeek V3.1 (ds31-168q-1_prefill.json) -
def ds31_level_tokens():
    d = json.load(open(os.path.join(ARCHIVE, "data/ds31-168q-1_prefill.json")))
    by = defaultdict(list)
    for p in d["per_prompt"]:
        by[p["level"]].append(p["n_tokens"])
    return [sum(by["L%d" % i]) / len(by["L%d" % i]) for i in LEVELS]

DS_TOKENS = ds31_level_tokens()

C_ALL = "#1f4e79"   # all-token
C_LAST = "#c55a11"  # last-token


# ---------------------------------------------------------------------------
# Figure 1: per-level RE means, all-token vs last-token, both models
# ---------------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    for ax, name, a, l in [
        (axes[0], "DeepSeek V3.1", DS_ALL, DS_LAST),
        (axes[1], "Qwen 397B", QW_ALL, QW_LAST),
    ]:
        ax.plot(LEVELS, a, "-o", color=C_ALL, ms=4, label="all-token mean RE")
        ax.plot(LEVELS, l, "-s", color=C_LAST, ms=4, label="final-token RE")
        ax.set_title(name)
        ax.set_xlabel("prompt complexity level (L1–L12)")
        ax.set_xticks(LEVELS)
    axes[0].set_ylabel("routing entropy (normalized)")
    axes[0].legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_level_means.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_level_means.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: all-token RE vs prompt token count, per level, DeepSeek V3.1
# ---------------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    sc = ax.scatter(DS_TOKENS, DS_ALL, c=LEVELS, cmap="viridis", s=55,
                    edgecolor="k", linewidth=0.4, zorder=3)
    for x, y, L in zip(DS_TOKENS, DS_ALL, LEVELS):
        ax.annotate("L%d" % L, (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7.5, color="#333333")
    ax.set_xlabel("mean prompt token count (per level)")
    ax.set_ylabel("all-token mean routing entropy")
    ax.set_title("DeepSeek V3.1: all-token routing entropy vs prompt length")
    cb = fig.colorbar(sc, ax=ax, ticks=[1, 6, 12])
    cb.set_label("complexity level", fontsize=8.5)
    ax.text(0.04, 0.94,
            r"per-prompt Spearman $\rho$ = +0.88 (n = 168)",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_token_confound.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_token_confound.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: per-token routing entropy vs prefill position (Qwen 397B diagnostic)
# Source: token-confound-archive/data/diagnostic_results.json (5 prompts,
# 60 MoE layers => Qwen3.5-397B). Raw per-position values plus the OLS fit
# stored in the archive record (slope/intercept). No model is run.
# ---------------------------------------------------------------------------
def fig3():
    d = json.load(open(os.path.join(ARCHIVE, "data/diagnostic_results.json")))
    order = sorted(d.keys(), key=lambda k: d[k]["n_tokens"])
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    for i, k in enumerate(order):
        v = d[k]
        y = v["position_entropy"]
        x = list(range(len(y)))
        c = cmap(0.1 + 0.8 * i / max(1, len(order) - 1))
        ax.plot(x, y, color=c, lw=0.6, alpha=0.4)
        fit = [v["intercept"] + v["slope"] * t for t in x]
        ax.plot(x, fit, color=c, lw=2.2, ls="--",
                label="%s, %d tok, slope %.1e" % (v["level"], v["n_tokens"], v["slope"]))
        ax.plot(x[-1], y[-1], "o", color=c, ms=6, mec="k", mew=0.5, zorder=4)
    ax.set_xlabel("prefill token position")
    ax.set_ylabel("routing entropy (normalized)")
    ax.set_title("Qwen 397B: routing entropy vs. position", fontsize=10)
    ax.set_ylim(0.745, 0.93)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92, title="prompt (level, length, fit slope/token)", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_position_curves.pdf"))
    fig.savefig(os.path.join(OUT, "fig3_position_curves.png"))
    plt.close(fig)
    # print the per-prompt Spearman(position, entropy) for the text
    try:
        from scipy.stats import spearmanr
        for k in order:
            v = d[k]
            r, pv = spearmanr(range(v["n_tokens"]), v["position_entropy"])
            print("fig3 %s L=%s n=%d spearman(pos,RE)=%+.2f p=%.2g slope=%.2e p=%.2g first5=%.3f last5=%.3f"
                  % (k, v["level"], v["n_tokens"], r, pv, v["slope"], v["p_value"],
                     sum(v["position_entropy"][:5]) / 5, sum(v["position_entropy"][-5:]) / 5))
    except ImportError:
        pass


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote figures to", OUT)
    print("DS per-level mean tokens:", [round(t, 1) for t in DS_TOKENS])
