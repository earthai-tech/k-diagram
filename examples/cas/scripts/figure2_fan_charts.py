"""
Figure 2 – Episode fan charts with clustered-failure overlays  (fig5.png)
==========================================================================

Displays representative fan charts for wind, hydrology, and subsidence,
each showing:
  - observation y as filled circles
  - forecast median (q50) as a dashed line
  - 90 % prediction band (q10–q90) as a shaded ribbon
  - interval violations marked with triangular glyphs
  - local CAS severity as a stem plot (secondary axis / inset)

One row per domain.  The model with the highest horizon-1 CAS that has
valid (non-NaN) predictions is used per domain:
  wind       → QGBM  (CAS ≈ 0.198)
  hydro      → QGBM  (CAS ≈ 0.839)
  subsidence → QGBM  (CAS ≈ 0.228)  [XTFT excluded: pathological q-crossing]

Outputs (data/cas/outputs/):
  figure2_fan_charts.png / .pdf   (also saved as fig5.png for the paper)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

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

    def enforce_non_crossing(q10, q50, q90):
        q50c = np.maximum(q50, q10)
        q90c = np.maximum(q90, q50c)
        return q10, q50c, q90c

_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kdiagram.metrics import cluster_aware_severity_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# (domain, model, horizon) to display in each row
PANEL_SPECS = [
    ("wind",        "qgbm", 1),
    ("hydro",       "qgbm", 1),
    ("subsidence",  "qgbm", 1),
]
PRED_FILES = {
    "wind":        PRED_WIND,
    "hydro":       PRED_HYDRO,
    "subsidence":  PRED_SUBS,
}
MODEL_LABELS = {"qar": "QAR", "qgbm": "QGBM", "xtft": "XTFT"}

WINDOW_SIZE = 5    # h = 2
LAMBDA = 1.0
GAMMA = 1.0
KERNEL = "triangular"

# Colour roles
ALPHA_BAND = 0.30
ALPHA_VIO_SHADE = 0.22


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_domain_model(pred_file, domain, model, horizon):
    df = pd.read_csv(pred_file)
    sub = df[(df["model"] == model) & (df["horizon"] == horizon)].copy()
    if sub.empty:
        raise ValueError(f"No data for {domain}/{model}/h={horizon}")
    q10, q50, q90 = enforce_non_crossing(
        sub["q10"].to_numpy(), sub["q50"].to_numpy(), sub["q90"].to_numpy()
    )
    sub["q10"], sub["q50"], sub["q90"] = q10, q50, q90
    sub = sub.dropna(subset=["q10", "q50", "q90", "y"])
    return sub.reset_index(drop=True)


def compute_severity(df):
    y = df["y"].to_numpy()
    q10 = df["q10"].to_numpy()
    q90 = df["q90"].to_numpy()
    y_pred = np.column_stack([q10, q90])
    score, details = cluster_aware_severity_score(
        y, y_pred,
        window_size=WINDOW_SIZE,
        kernel=KERNEL,
        lambda_=LAMBDA,
        gamma=GAMMA,
        return_details=True,
    )
    return score, details


# ---------------------------------------------------------------------------
# Draw one fan-chart panel
# ---------------------------------------------------------------------------

def draw_fan_panel(ax, ax_sev, df, details, cas_score, domain, model, horizon):
    """Fill ax with fan chart; fill ax_sev with severity stems."""
    col = DOMAIN_COLOR[domain]
    n = len(df)
    t = np.arange(n)

    y = df["y"].to_numpy()
    q10 = df["q10"].to_numpy()
    q50 = df["q50"].to_numpy()
    q90 = df["q90"].to_numpy()
    sv = details["severity"].to_numpy()
    viol = details["is_anomaly"].to_numpy().astype(bool)

    # ---------- fan (prediction band) ----------
    ax.fill_between(
        t, q10, q90,
        color=col, alpha=ALPHA_BAND, label=f"90% P.I."
    )

    # ---------- forecast median ----------
    ax.plot(t, q50, color=col, lw=2.0, ls="--", alpha=0.85, label="Median (q50)")

    # ---------- observations: covered (small dot) vs violation (red marker) ----------
    mask_ok = ~viol
    ax.scatter(
        t[mask_ok], y[mask_ok],
        s=14, color="0.25", zorder=4, label="Obs (covered)"
    )
    if viol.any():
        ax.scatter(
            t[viol], y[viol],
            s=55, marker="v", color="#C0392B", zorder=5,
            edgecolors="white", lw=0.6, label="Obs (violation)"
        )

    # ---------- shade persistent violation runs ----------
    in_run = False
    run_start = None
    for i in range(n):
        if viol[i] and not in_run:
            run_start = i
            in_run = True
        elif not viol[i] and in_run:
            ax.axvspan(
                run_start - 0.4, i - 0.6,
                ymin=0, ymax=1, color="#C0392B", alpha=0.08, zorder=1
            )
            in_run = False
    if in_run:
        ax.axvspan(run_start - 0.4, n - 0.6, color="#C0392B", alpha=0.08, zorder=1)

    # ---------- axes decoration ----------
    ax.set_xlim(-0.5, n - 0.5)
    _ptp = np.ptp(y) if np.ptp(y) > 0 else 1.0
    margin = 0.12 * _ptp
    ax.set_ylim(y.min() - margin, y.max() + margin * 1.6)
    ax.set_ylabel(DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=11)
    ax.yaxis.label.set_rotation(90)
    ax.grid(True, axis="y", alpha=0.20, ls="--")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # CAS annotation
    ax.annotate(
        f"CAS = {cas_score:.4f}",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=10, fontweight="bold",
        color=col,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=0.9, alpha=0.9)
    )

    # Model/horizon label
    ax.annotate(
        f"{MODEL_LABELS.get(model, model.upper())}  |  h = {horizon}",
        xy=(0.01, 0.95), xycoords="axes fraction",
        ha="left", va="top", fontsize=9, color="0.4", style="italic"
    )

    # ---------- severity stems on lower sub-axis ----------
    if ax_sev is not None:
        markerline, stemlines, baseline = ax_sev.stem(
            t, sv, linefmt="-", markerfmt="o", basefmt="k-"
        )
        plt.setp(stemlines, color=col, lw=1.3, alpha=0.80)
        plt.setp(markerline, color=col, markersize=3.5, zorder=5)
        plt.setp(baseline, lw=0.5, color="0.4")

        ax_sev.set_xlim(-0.5, n - 0.5)
        ax_sev.set_ylabel("$s_t$", fontsize=9)
        ax_sev.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax_sev.grid(True, axis="y", alpha=0.18, ls="--")
        for sp in ("top", "right"):
            ax_sev.spines[sp].set_visible(False)

    return ax


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
print("Building Figure 2 (fan charts) …")

N_ROWS = len(PANEL_SPECS)
# Each domain gets a tall panel (fan) + a short severity strip
height_ratios = []
for _ in range(N_ROWS):
    height_ratios += [3.2, 0.9]

fig = plt.figure(figsize=(15, N_ROWS * 5.0))
gs = gridspec.GridSpec(
    N_ROWS * 2, 1,
    figure=fig,
    height_ratios=height_ratios,
    hspace=0.08,
)

for row_idx, (domain, model, horizon) in enumerate(PANEL_SPECS):
    print(f"  Processing {domain}/{model}/h={horizon} …")
    df = load_domain_model(PRED_FILES[domain], domain, model, horizon)
    cas_score, details = compute_severity(df)
    print(f"    CAS = {cas_score:.5f}  |  n = {len(df)}  |  violations = {details['is_anomaly'].sum()}")

    fan_row = row_idx * 2
    sev_row = row_idx * 2 + 1

    ax_fan = fig.add_subplot(gs[fan_row])
    ax_sev = fig.add_subplot(gs[sev_row], sharex=ax_fan)

    draw_fan_panel(ax_fan, ax_sev, df, details, cas_score, domain, model, horizon)

    # Only show x-labels on severity strip (bottom of each domain)
    plt.setp(ax_fan.get_xticklabels(), visible=False)

    is_last = (row_idx == N_ROWS - 1)
    if is_last:
        ax_sev.set_xlabel("Step index within test window", fontsize=11)
    else:
        plt.setp(ax_sev.get_xticklabels(), visible=False)

    # Shared legend on the first domain only
    if row_idx == 0:
        handles, labels = ax_fan.get_legend_handles_labels()
        ax_fan.legend(
            handles, labels,
            loc="upper left", fontsize=8.5, frameon=True,
            framealpha=0.92, edgecolor="0.7",
            ncol=2
        )

# ---------------------------------------------------------------------------
# Super-title and column header
# ---------------------------------------------------------------------------
fig.suptitle(
    "Representative fan charts with clustered-failure overlays\n"
    r"(triangular kernel, $h=2$, $\lambda=1$, $\gamma=1$;  "
    r"red triangles $\triangledown$ = violations; red shading = violation runs)",
    y=1.02, fontsize=12.5, fontweight="bold"
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
# Disable constrained_layout before subplots_adjust to avoid UserWarning
fig.set_constrained_layout(False)
fig.subplots_adjust(top=0.96, hspace=0.0)

for stem in ("figure2_fan_charts", "fig5"):
    for ext in ("png", "pdf"):
        out = OUTDIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[Saved] {out}")

plt.close(fig)
print("[Done] figure2_fan_charts complete.")
