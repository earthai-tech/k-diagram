"""
Figure 3 – Rank stability of CAS under parameter variation
===========================================================

Computes Kendall's rank correlation (τ) between the default CAS model
ranking and rankings obtained under an 18-point parameter grid, for each
domain.  Visualises as:

  Panel A (heatmap): τ values on a grid (λ × γ) for each h and domain
  Panel B (summary bar): mean, median and min τ per domain (matches Table 4)

Parameter grid (18 settings = 3h × 3λ × 2γ):
  h ∈ {3, 5, 7}    (window_size; default = 5)
  λ ∈ {0.5, 1.0, 2.0}  (lambda_; default = 1.0)
  γ ∈ {1.0, 2.0}   (gamma; default = 1.0)
Default: h=5, λ=1.0, γ=1.0, kernel=triangular

CAS is averaged across horizons for each (domain, model) cell, then
models are ranked by mean CAS (lower is better = higher rank).

Outputs (data/cas/outputs/):
  figure3_rank_stability.png / .pdf
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import os as _os
_REPO_ROOT = _HERE.parents[2]
_REAL_DATA = _REPO_ROOT / "data" / "cas"
if _REAL_DATA.exists():
    _os.environ.setdefault("KDIAGRAM_DATA_DIR", str(_REAL_DATA))

try:
    from results_config import (
        DOMAIN_COLOR,
        DOMAIN_LABEL,
        DOMAIN_ORDER,
        MODEL_LABEL,
        MODEL_ORDER,
        OUTDIR,
        PRED_HYDRO,
        PRED_SUBS,
        PRED_WIND,
        enforce_non_crossing,
    )
except ModuleNotFoundError:
    _REPO = _HERE.parents[2]
    DATA_ROOT = _REPO / "data" / "cas"
    OUTDIR = DATA_ROOT / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PRED_WIND = DATA_ROOT / "modeling_results_ok" / "predictions_wind.csv"
    PRED_HYDRO = DATA_ROOT / "modeling_results_ok" / "predictions_hydro.csv"
    PRED_SUBS = DATA_ROOT / "modeling_results_ok" / "predictions_subsidence.csv"
    DOMAIN_COLOR = {"hydro": "#0072B2", "wind": "#E69F00", "subsidence": "#009E73"}
    DOMAIN_LABEL = {"hydro": "Hydro", "wind": "Wind", "subsidence": "Subsidence"}
    DOMAIN_ORDER = ["hydro", "wind", "subsidence"]
    MODEL_ORDER = ["qar", "qgbm", "xtft"]
    MODEL_LABEL = {"qar": "QAR", "qgbm": "QGBM", "xtft": "XTFT"}

    def enforce_non_crossing(q10, q50, q90):
        q50c = np.maximum(q50, q10)
        q90c = np.maximum(q90, q50c)
        return q10, q50c, q90c

_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kdiagram.metrics import cluster_aware_severity_score

# ---------------------------------------------------------------------------
# Parameter grid (18 settings)
# ---------------------------------------------------------------------------
H_VALUES = [3, 5, 7]           # window_size
LAMBDA_VALUES = [0.5, 1.0, 2.0]
GAMMA_VALUES = [1.0, 2.0]
KERNEL = "triangular"

# Default setting
H_DEFAULT, LAM_DEFAULT, GAM_DEFAULT = 5, 1.0, 1.0

PRED_FILES = {
    "hydro":       PRED_HYDRO,
    "wind":        PRED_WIND,
    "subsidence":  PRED_SUBS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all(pred_file, domain):
    df = pd.read_csv(pred_file)
    q10, q50, q90 = enforce_non_crossing(
        df["q10"].to_numpy(), df["q50"].to_numpy(), df["q90"].to_numpy()
    )
    df["q10"], df["q50"], df["q90"] = q10, q50, q90
    return df


def mean_cas(df, model, h, lam, gam):
    """Mean CAS across all horizons for one (model, param) combo."""
    sub = df[df["model"] == model].dropna(subset=["q10", "q90", "y"])
    if sub.empty:
        return np.nan
    scores = []
    for _, grp in sub.groupby("horizon"):
        y = grp["y"].to_numpy()
        q10 = grp["q10"].to_numpy()
        q90 = grp["q90"].to_numpy()
        if len(y) < 3:
            continue
        y_pred = np.column_stack([q10, q90])
        try:
            s = cluster_aware_severity_score(
                y, y_pred,
                window_size=h, kernel=KERNEL,
                lambda_=lam, gamma=gam,
            )
            if np.isfinite(s):
                scores.append(s)
        except Exception:
            pass
    return np.mean(scores) if scores else np.nan


def model_ranking(cas_dict: dict) -> list[str]:
    """Rank models by CAS ascending; NaN models placed last."""
    valid = {m: v for m, v in cas_dict.items() if np.isfinite(v)}
    nan_m = [m for m, v in cas_dict.items() if not np.isfinite(v)]
    ranked = sorted(valid, key=lambda m: valid[m]) + nan_m
    return ranked


def kendall_tau_corr(rank_a: list[str], rank_b: list[str]) -> float:
    """Kendall τ between two model orderings (both must include same models)."""
    common = [m for m in rank_a if m in rank_b]
    if len(common) < 2:
        return np.nan
    pos_a = {m: i for i, m in enumerate(rank_a)}
    pos_b = {m: i for i, m in enumerate(rank_b)}
    a_vals = [pos_a[m] for m in common]
    b_vals = [pos_b[m] for m in common]
    tau, _ = kendalltau(a_vals, b_vals)
    return float(tau)


# ---------------------------------------------------------------------------
# Compute CAS for all (domain, model, setting)
# ---------------------------------------------------------------------------
print("Computing CAS across parameter grid …")

# Pre-load data
data = {}
for domain in DOMAIN_ORDER:
    data[domain] = load_all(PRED_FILES[domain], domain)

# Build full result table
rows = []
all_settings = list(itertools.product(H_VALUES, LAMBDA_VALUES, GAMMA_VALUES))
is_default = lambda h, lam, gam: (h == H_DEFAULT and lam == LAM_DEFAULT and gam == GAM_DEFAULT)

for domain in DOMAIN_ORDER:
    df = data[domain]
    print(f"  {domain} …")
    for h, lam, gam in all_settings:
        cas_dict = {}
        for model in MODEL_ORDER:
            cas_dict[model] = mean_cas(df, model, h, lam, gam)
        rows.append(
            dict(
                domain=domain, h=h, lam=lam, gam=gam,
                default=is_default(h, lam, gam),
                **{f"cas_{m}": cas_dict[m] for m in MODEL_ORDER},
                ranking=model_ranking(cas_dict),
            )
        )

results = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Compute Kendall τ relative to default ranking
# ---------------------------------------------------------------------------
tau_rows = []
for domain in DOMAIN_ORDER:
    default_row = results[
        (results.domain == domain) & results.default
    ].iloc[0]
    default_ranking = default_row["ranking"]

    for _, row in results[results.domain == domain].iterrows():
        tau = kendall_tau_corr(default_ranking, row["ranking"])
        tau_rows.append(
            dict(
                domain=domain,
                h=row["h"], lam=row["lam"], gam=row["gam"],
                default=row["default"],
                tau=tau,
            )
        )

tau_df = pd.DataFrame(tau_rows)

# Summary table matching Table 4 of the paper
summary = (
    tau_df[~tau_df.default]
    .groupby("domain")["tau"]
    .agg(mean_tau="mean", median_tau="median", min_tau="min", n_settings="count")
    .reset_index()
)
print("\n  Rank-stability summary (Table 4):")
print(summary.to_string(index=False))

# Save summary
tau_df.to_csv(OUTDIR / "table3_rank_stability_full.csv", index=False)
summary.to_csv(OUTDIR / "table3_rank_stability_summary.csv", index=False)

# ---------------------------------------------------------------------------
# Figure layout: 2-row × 3-col (heatmaps) + 1 summary panel
# ---------------------------------------------------------------------------
# Heatmap: for each domain, show τ on λ-axis (cols) × γ-axis (rows), one sub-panel per h
# Summary: mean/median/min τ per domain as grouped bars

N_DOMAINS = len(DOMAIN_ORDER)
N_H = len(H_VALUES)

# τ colour scale
TAU_VMIN, TAU_VMAX = -1.0, 1.0
CMAP = plt.cm.RdYlGn  # red (low τ) → yellow → green (τ=1)

# Disable constrained_layout for this figure (we position colorbar manually)
with plt.rc_context({"figure.constrained_layout.use": False}):
    fig = plt.figure(figsize=(14, 11))
fig.set_constrained_layout(False)

# Reserve right margin for the colorbar
fig.subplots_adjust(left=0.07, right=0.88, top=0.91, bottom=0.07,
                    hspace=0.52, wspace=0.38)

# Two-zone layout: heatmap block (rows 0..N_DOMAINS-1) + bar chart (last row)
outer_gs = fig.add_gridspec(
    2, 1,
    height_ratios=[3.0, 1.3],
    hspace=0.45,
    left=0.07, right=0.88, top=0.91, bottom=0.07,
)
heat_gs = outer_gs[0].subgridspec(N_DOMAINS, N_H, hspace=0.52, wspace=0.38)
bar_ax  = fig.add_subplot(outer_gs[1])

all_heat_axes = []
last_im = None

for di, domain in enumerate(DOMAIN_ORDER):
    dom_tau = tau_df[tau_df.domain == domain]
    col = DOMAIN_COLOR[domain]

    for hi, h_val in enumerate(H_VALUES):
        ax = fig.add_subplot(heat_gs[di, hi])
        all_heat_axes.append(ax)
        sub = dom_tau[dom_tau.h == h_val]

        # Build γ (rows) × λ (cols) matrix
        mat = np.full((len(GAMMA_VALUES), len(LAMBDA_VALUES)), np.nan)
        for gi, gam in enumerate(GAMMA_VALUES):
            for li, lam in enumerate(LAMBDA_VALUES):
                row = sub[(sub.lam == lam) & (sub.gam == gam)]
                if not row.empty:
                    mat[gi, li] = row["tau"].values[0]

        im = ax.imshow(
            mat, aspect="auto",
            vmin=TAU_VMIN, vmax=TAU_VMAX,
            cmap=CMAP, origin="upper",
        )
        last_im = im

        # Cell value annotations
        for gi in range(len(GAMMA_VALUES)):
            for li in range(len(LAMBDA_VALUES)):
                val = mat[gi, li]
                txt = f"{val:.2f}" if np.isfinite(val) else "-"
                cell_norm = (val - TAU_VMIN) / (TAU_VMAX - TAU_VMIN)
                txt_color = "black" if 0.25 < cell_norm < 0.75 else "white"
                ax.text(
                    li, gi, txt,
                    ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=txt_color,
                )

        # Black border on the default cell (lam=1.0, gam=1.0)
        def_li = LAMBDA_VALUES.index(LAM_DEFAULT)
        def_gi = GAMMA_VALUES.index(GAM_DEFAULT)
        ax.add_patch(
            plt.Rectangle(
                (def_li - 0.5, def_gi - 0.5), 1, 1,
                fill=False, edgecolor="black", lw=2.5, zorder=5
            )
        )

        ax.set_xticks(range(len(LAMBDA_VALUES)))
        ax.set_yticks(range(len(GAMMA_VALUES)))
        ax.set_xticklabels([f"$\\lambda$={v}" for v in LAMBDA_VALUES], fontsize=8.5)
        ax.set_yticklabels([f"$\\gamma$={v}" for v in GAMMA_VALUES], fontsize=8.5)

        if hi == 0:
            ax.set_ylabel(
                DOMAIN_LABEL[domain], fontweight="bold",
                color=col, fontsize=11, labelpad=6,
            )
        if di == 0:
            ax.set_title(f"$h = {h_val}$", fontsize=11, fontweight="bold", pad=5)
        if di == N_DOMAINS - 1:
            ax.set_xlabel("Cluster weight", fontsize=8.5)

        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_linewidth(0.6)

# ── Colorbar: placed in the reserved right strip [0.90, top..bottom of heatmap]
# Compute the bounding box of the heatmap block in figure coordinates.
fig.canvas.draw()   # needed so get_position() is accurate
heat_positions = [ax.get_position() for ax in all_heat_axes]
cbar_left   = 0.905
cbar_bottom = min(p.y0 for p in heat_positions)
cbar_height = max(p.y1 for p in heat_positions) - cbar_bottom
cbar_ax = fig.add_axes([cbar_left, cbar_bottom, 0.018, cbar_height])

sm = plt.cm.ScalarMappable(
    cmap=CMAP, norm=plt.Normalize(vmin=TAU_VMIN, vmax=TAU_VMAX)
)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Kendall $\\tau$", fontsize=10, labelpad=8)
cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
cbar.ax.tick_params(labelsize=9)

# ---------------------------------------------------------------------------
# Summary bar chart (bottom panel)
# ---------------------------------------------------------------------------
x = np.arange(N_DOMAINS)
bw = 0.25

stat_labels = [("mean_tau", "Mean $\\tau$", "o-"), ("median_tau", "Median $\\tau$", "s--"), ("min_tau", "Min $\\tau$", "^:")]
colors_stat = ["#2196F3", "#4CAF50", "#FF5722"]

for si, (col_key, stat_label, fmt) in enumerate(stat_labels):
    vals = [
        summary[summary.domain == d][col_key].values[0]
        if d in summary.domain.values else np.nan
        for d in DOMAIN_ORDER
    ]
    bars = bar_ax.bar(
        x + (si - 1) * bw, vals, bw,
        color=colors_stat[si], alpha=0.82,
        label=stat_label, edgecolor="white", lw=0.8
    )
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            bar_ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5
            )

# Domain-color patches at x-tick labels
bar_ax.set_xticks(x)
bar_ax.set_xticklabels(
    [DOMAIN_LABEL[d] for d in DOMAIN_ORDER], fontsize=11
)
for tick, d in zip(bar_ax.get_xticklabels(), DOMAIN_ORDER):
    tick.set_color(DOMAIN_COLOR[d])
    tick.set_fontweight("bold")

bar_ax.axhline(1.0, color="0.4", lw=0.8, ls="--", label="Perfect stability ($\\tau=1$)")
bar_ax.set_ylim(-0.05, 1.30)
bar_ax.set_ylabel("Kendall $\\tau$")
bar_ax.set_title(
    "Rank-stability summary (mean / median / min Kendall $\\tau$ across 17 non-default settings)",
    fontweight="bold"
)
bar_ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
bar_ax.grid(True, axis="y", alpha=0.22, ls="--")
for sp in ("top", "right"):
    bar_ax.spines[sp].set_visible(False)

# ---------------------------------------------------------------------------
# Super-title
# ---------------------------------------------------------------------------
fig.suptitle(
    "Rank stability of CAS under parameter variation\n"
    "(black border = default; each cell = $\\tau$ vs. default ranking)",
    y=0.99, fontsize=12.5, fontweight="bold"
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_png = OUTDIR / "figure3_rank_stability.png"
out_pdf = OUTDIR / "figure3_rank_stability.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"\n[Saved] {out_png}")
print(f"[Saved] {out_pdf}")
print("[Done] figure3_rank_stability complete.")
