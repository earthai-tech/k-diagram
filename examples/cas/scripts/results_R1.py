"""
R1: Table 1 + Figure 1 (Reliability & PIT)

Single-figure Reliability (Fig 1a) and PIT (Fig 1b), 3 columns
= models. Within each panel, three domains are overlaid with
fixed, color-blind-safe colors.

Inputs (from results_config):
  - metrics_all_domains.csv
  - predictions_wind.csv
  - predictions_hydro.csv
  - predictions_subsidence.csv

Outputs (under data/cas/outputs/):
  - figure1a_reliability.png / .pdf
  - figure1b_pit.png / .pdf
  - table1_metrics_by_domain_model.csv / .tex
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from results_config import (
    DOMAIN_COLOR,
    DOMAIN_LABEL,
    DOMAIN_MARKER,
    DOMAIN_ORDER,
    METRICS_PATH,
    MODEL_LABEL,
    MODEL_ORDER,
    N_BOOT,
    NOMINAL_COVERAGE,
    OUTDIR,
    PIT_BINS,
    PRED_HYDRO,
    PRED_SUBS,
    PRED_WIND,
    load_preds,
    reliability_ci,
    reliability_points,
    rng,
)

# -------------------------
# Table 1 (for §4.1)
# -------------------------
metrics = pd.read_csv(METRICS_PATH)

table1 = (
    metrics.groupby(["domain", "model"], as_index=False)
    .agg(
        mean_crps=("crps", "mean"),
        mean_winkler=("winkler", "mean"),
        mean_cov=("coverage", "mean"),
    )
    .sort_values(["domain", "model"])
)

table1["delta_cov"] = table1["mean_cov"] - NOMINAL_COVERAGE

path_tbl_csv = OUTDIR / "table1_metrics_by_domain_model.csv"
path_tbl_tex = OUTDIR / "table1_metrics_by_domain_model.tex"

table1.to_csv(path_tbl_csv, index=False)
with open(path_tbl_tex, "w", encoding="utf-8") as f:
    f.write(table1.to_latex(index=False, float_format="%.4g"))

print(f"[Saved] {path_tbl_csv}")
print(f"[Saved] {path_tbl_tex}")

# -------------------------
# Load predictions
# -------------------------
wind = load_preds(PRED_WIND, "wind")
hydro = load_preds(PRED_HYDRO, "hydro")
subs = load_preds(PRED_SUBS, "subsidence")

preds = pd.concat([wind, hydro, subs], ignore_index=True)

# speed: split once by model
by_model = {m: preds[preds["model"] == m].copy() for m in MODEL_ORDER}

# -------------------------
# Figure 1a: Reliability
# -------------------------
fig_rel, axes_rel = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)

for j, model in enumerate(MODEL_ORDER):
    ax = axes_rel[j]
    subm = by_model[model]

    # ideal 45°
    ax.plot(
        [0, 1],
        [0, 1],
        ls="--",
        lw=1.0,
        color="0.5",
        zorder=1,
    )

    # per-domain overlay
    for d in DOMAIN_ORDER:
        submd = subm[subm["domain"] == d]
        y = submd["y"].to_numpy()
        q10 = submd["q10"].to_numpy()
        q50 = submd["q50"].to_numpy()
        q90 = submd["q90"].to_numpy()

        taus, emp = reliability_points(y, q10, q50, q90)
        lo, hi = reliability_ci(y, q10, q50, q90, n_boot=N_BOOT, rng=rng)

        ax.errorbar(
            taus,
            emp,
            yerr=[emp - lo, hi - emp],
            fmt=f"{DOMAIN_MARKER[d]}-",
            lw=2.0,
            ms=6,
            mew=0.8,
            mfc="white",
            color=DOMAIN_COLOR[d],
            zorder=3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal probability $\\tau$")
    if j == 0:
        ax.set_ylabel("Empirical $\\Pr(Y \\le q_\\tau)$")
    ax.set_title(MODEL_LABEL.get(model, model.upper()))
    ax.grid(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# shared legend
handles_rel = [
    Line2D(
        [0],
        [0],
        color=DOMAIN_COLOR[d],
        marker=DOMAIN_MARKER[d],
        lw=2,
        ms=6,
        mfc="white",
        label=DOMAIN_LABEL[d],
    )
    for d in DOMAIN_ORDER
]
fig_rel.legend(
    handles=handles_rel,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

rel_png = OUTDIR / "figure1a_reliability.png"
rel_pdf = OUTDIR / "figure1a_reliability.pdf"

fig_rel.savefig(rel_png, bbox_inches="tight")
fig_rel.savefig(rel_pdf, bbox_inches="tight")
plt.close(fig_rel)

print(f"[Saved] {rel_png}")
print(f"[Saved] {rel_pdf}")

# -------------------------
# Figure 1b: PIT
# -------------------------
fig_pit, axes_pit = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)

edges = np.linspace(0, 1, PIT_BINS + 1)

# find a shared y-limit across panels
global_ymax = 0.0
for model in MODEL_ORDER:
    subm = by_model[model]
    for d in DOMAIN_ORDER:
        pits = subm[subm["domain"] == d]["pit"].to_numpy()
        if pits.size == 0:
            continue
        counts, _ = np.histogram(pits, bins=edges, density=True)
        global_ymax = max(global_ymax, counts.max())
global_ymax *= 1.12

for j, model in enumerate(MODEL_ORDER):
    ax = axes_pit[j]
    subm = by_model[model]

    for d in DOMAIN_ORDER:
        pits = subm[subm["domain"] == d]["pit"].to_numpy()
        if pits.size == 0:
            continue
        counts, _ = np.histogram(pits, bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(
            centers,
            counts,
            where="mid",
            color=DOMAIN_COLOR[d],
            lw=1.8,
            zorder=3,
            label=DOMAIN_LABEL[d],
        )
        ax.fill_between(
            centers,
            counts,
            step="mid",
            color=DOMAIN_COLOR[d],
            alpha=0.15,
            zorder=2,
        )

    # uniform reference
    ax.plot(
        [0, 1],
        [1, 1],
        ls="--",
        lw=1.0,
        color="0.5",
        zorder=1,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, global_ymax)
    ax.set_xlabel("PIT")
    if j == 0:
        ax.set_ylabel("Density")
    ax.set_title(MODEL_LABEL.get(model, model.upper()))
    ax.grid(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# shared legend
handles_pit = [
    Patch(
        facecolor=DOMAIN_COLOR[d],
        edgecolor=DOMAIN_COLOR[d],
        alpha=0.25,
        label=DOMAIN_LABEL[d],
    )
    for d in DOMAIN_ORDER
]
fig_pit.legend(
    handles=handles_pit,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

pit_png = OUTDIR / "figure1b_pit.png"
pit_pdf = OUTDIR / "figure1b_pit.pdf"

fig_pit.savefig(pit_png, bbox_inches="tight")
fig_pit.savefig(pit_pdf, bbox_inches="tight")
plt.close(fig_pit)

print(f"[Saved] {pit_png}")
print(f"[Saved] {pit_pdf}")
print("[Done] R1 assets written to outputs/")
