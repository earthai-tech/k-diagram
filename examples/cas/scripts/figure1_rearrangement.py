"""
Figure 1 – Controlled Rearrangement Experiment
===============================================

Shows that two forecasting sequences with identical coverage, exceedance
magnitudes, and interval-score contributions can have different CAS when
violations are temporally clustered versus isolated.

Setup (§5.1 of the revised CAS paper):
  n = 30, 6 violations, relative exceedance e = 0.1, band width = 1.0
  kernel = triangular, h = 2 (window_size = 5), λ = 1, γ = 1

Outputs (data/cas/outputs/):
  figure1_rearrangement.png / .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ---------------------------------------------------------------------------
# Path setup – allow running from the repo root or from the scripts directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# results_config._find_repo_root stops at examples/cas/ because examples/cas/data/ exists.
# Force the correct data root via env var before importing results_config.
import os  # noqa: E402

_REPO_ROOT = _HERE.parents[2]  # scripts -> cas -> examples -> k-diagram
_REAL_DATA = _REPO_ROOT / "data" / "cas"
if _REAL_DATA.exists():
    os.environ.setdefault("KDIAGRAM_DATA_DIR", str(_REAL_DATA))

try:
    from results_config import OUTDIR
except ModuleNotFoundError:
    OUTDIR = _HERE.parents[2] / "data" / "cas" / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

# Add k-diagram package root if needed
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kdiagram.metrics import cluster_aware_severity_score  # noqa: E402

# ---------------------------------------------------------------------------
# Style constants (consistent with results_config)
# ---------------------------------------------------------------------------
COLOR_BAND = "#AED6F1"  # light blue fill for prediction band
COLOR_MEDIAN = "#1A5276"  # dark blue for median line
COLOR_OBS_OK = "#2ECC71"  # green for covered observations
COLOR_OBS_VIO = "#C0392B"  # red for violation observations
COLOR_STEM = "#E74C3C"  # red for severity stems
COLOR_GRID = "#D5D8DC"  # light grey grid

WINDOW_SIZE = 5  # h = 2  (triangular kernel reaches ±2 steps)
LAMBDA = 1.0
GAMMA = 1.0
KERNEL = "triangular"

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
N = 30
Q10 = np.zeros(N)  # lower bound = 0
Q90 = np.ones(N)  # upper bound = 1   → band width = 1
Y_COVER = 0.5 * np.ones(N)
Y_VIOLATE = 1.1 * np.ones(N)  # relative exceedance e = (1.1-1.0)/1.0 = 0.1

# Q50 (forecast median) at midpoint of band
Q50 = 0.5 * np.ones(N)

# Isolated configuration: violations separated by 5 > h=2 steps
ISOLATED_IDX = [2, 7, 12, 17, 22, 27]

# Clustered configuration: same 6 violations in one contiguous block
CLUSTERED_IDX = [12, 13, 14, 15, 16, 17]

# Both have: coverage=0.80, mean interval score=1.40, mean rel. exceedance=0.02
# CAS_isolated=0.020, CAS_clustered≈0.0367  (verified analytically)


def make_y(violation_idx: list[int]) -> np.ndarray:
    y = Y_COVER.copy()
    y[violation_idx] = Y_VIOLATE[0]
    return y


def compute_cas_details(y, name=""):
    y_pred = np.column_stack([Q10, Q90])
    score, df = cluster_aware_severity_score(
        y,
        y_pred,
        window_size=WINDOW_SIZE,
        kernel=KERNEL,
        lambda_=LAMBDA,
        gamma=GAMMA,
        return_details=True,
    )
    if name:
        print(
            f"  [{name}] CAS = {score:.5f} | "
            f"coverage = {np.mean((y >= Q10) & (y <= Q90)):.3f} | "
            f"mean magnitude = {df['magnitude'].mean():.4f}"
        )
    return score, df


print("Computing CAS for rearrangement experiment …")
y_iso = make_y(ISOLATED_IDX)
y_clu = make_y(CLUSTERED_IDX)

cas_iso, df_iso = compute_cas_details(y_iso, "Isolated")
cas_clu, df_clu = compute_cas_details(y_clu, "Clustered")

# Winkler / interval score (α = 0.10 for 90% interval)


def winkler(y, q10, q90, alpha=0.10):
    bw = q90 - q10
    score = bw.copy()
    above = y > q90
    below = y < q10
    score[above] += 2.0 / alpha * (y[above] - q90[above])
    score[below] += 2.0 / alpha * (q10[below] - y[below])
    return score.mean()


wink_iso = winkler(y_iso, Q10, Q90)
wink_clu = winkler(y_clu, Q10, Q90)
cov_iso = np.mean((y_iso >= Q10) & (y_iso <= Q90))
cov_clu = np.mean((y_clu >= Q10) & (y_clu <= Q90))
# 'magnitude' column = relative exceedance per point (NaN for covered)
exc_iso = df_iso["magnitude"].mean()
exc_clu = df_clu["magnitude"].mean()

print(
    f"\n  Isolated  -> Coverage {cov_iso:.3f}  Winkler {wink_iso:.3f}  CAS {cas_iso:.5f}"
)
print(
    f"  Clustered -> Coverage {cov_clu:.3f}  Winkler {wink_clu:.3f}  CAS {cas_clu:.5f}"
)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
t = np.arange(N)

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(
    3,
    2,
    figure=fig,
    height_ratios=[2.2, 1.2, 1.2],
    hspace=0.55,
    wspace=0.32,
)

# Row 0: time-series panels
ax_ts_l = fig.add_subplot(gs[0, 0])
ax_ts_r = fig.add_subplot(gs[0, 1], sharey=ax_ts_l)

# Row 1: severity stems
ax_sv_l = fig.add_subplot(gs[1, 0])
ax_sv_r = fig.add_subplot(gs[1, 1], sharey=ax_sv_l)

# Row 2: metric comparison (spans both columns)
ax_bar = fig.add_subplot(gs[2, :])


def _draw_fanplot(ax, y, viol_idx, title, show_ylabel=True):
    """Draw prediction band + observations for one configuration."""
    # Prediction band
    ax.fill_between(
        t, Q10, Q90, color=COLOR_BAND, alpha=0.55, label="90% P.I."
    )
    # Median
    ax.plot(t, Q50, color=COLOR_MEDIAN, lw=1.8, ls="--", label="Median")

    # Covered observations
    mask_ok = np.ones(N, dtype=bool)
    mask_ok[viol_idx] = False
    mask_vio = np.zeros(N, dtype=bool)
    mask_vio[viol_idx] = True

    ax.scatter(
        t[mask_ok],
        y[mask_ok],
        color=COLOR_OBS_OK,
        s=30,
        zorder=4,
        label="Covered",
    )
    ax.scatter(
        t[mask_vio],
        y[mask_vio],
        color=COLOR_OBS_VIO,
        s=60,
        marker="v",
        zorder=5,
        label="Violation",
    )

    # Vertical drop lines at violations
    for idx in viol_idx:
        ax.vlines(
            idx,
            Q90[idx],
            y[idx],
            colors=COLOR_OBS_VIO,
            lw=1.2,
            ls="-",
            alpha=0.7,
            zorder=3,
        )

    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.1, 1.35)
    ax.set_xlabel("Time step $t$")
    if show_ylabel:
        ax.set_ylabel("Value")
    ax.grid(True, color=COLOR_GRID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Annotate violation indices
    ax.annotate(
        f"{len(viol_idx)} violations",
        xy=(0.98, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9.5,
        color=COLOR_OBS_VIO,
        style="italic",
    )
    return ax


def _draw_stems(ax, df, title, show_ylabel=True):
    """Draw local severity as a stem plot."""
    t_vals = df.index if "t" not in df.columns else df["t"].values
    sv = df["severity"].values

    markerline, stemlines, baseline = ax.stem(
        t_vals,
        sv,
        linefmt=COLOR_STEM,
        markerfmt="o",
        basefmt="k-",
    )
    plt.setp(markerline, color=COLOR_STEM, markersize=4, zorder=5)
    plt.setp(stemlines, lw=1.4, alpha=0.85)
    plt.setp(baseline, lw=0.6)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_xlabel("Time step $t$")
    if show_ylabel:
        ax.set_ylabel("Local severity $s_t$")
    ax.set_title(title, fontweight="bold", pad=5)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
    ax.grid(True, color=COLOR_GRID, axis="y")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


# Plot time series
_draw_fanplot(
    ax_ts_l, y_iso, ISOLATED_IDX, "Isolated violations", show_ylabel=True
)
_draw_fanplot(
    ax_ts_r, y_clu, CLUSTERED_IDX, "Clustered violations", show_ylabel=False
)

# Add violation-position labels below the isolation panel
for idx in ISOLATED_IDX:
    ax_ts_l.text(
        idx,
        -0.07,
        str(idx),
        ha="center",
        va="top",
        fontsize=7.5,
        color=COLOR_OBS_VIO,
    )
for idx in CLUSTERED_IDX:
    ax_ts_r.text(
        idx,
        -0.07,
        str(idx),
        ha="center",
        va="top",
        fontsize=7.5,
        color=COLOR_OBS_VIO,
    )

# Common legend on top-right panel only
handles, labels = ax_ts_l.get_legend_handles_labels()
ax_ts_r.legend(
    handles,
    labels,
    loc="upper right",
    fontsize=9,
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
)

# Plot severity stems
df_iso_indexed = df_iso.copy()
df_clu_indexed = df_clu.copy()

_draw_stems(ax_sv_l, df_iso, "Local severity (isolated)", show_ylabel=True)
_draw_stems(ax_sv_r, df_clu, "Local severity (clustered)", show_ylabel=False)

# Annotate total CAS on stem panels
for ax, cas in [(ax_sv_l, cas_iso), (ax_sv_r, cas_clu)]:
    ax.annotate(
        f"CAS = {cas:.4f}",
        xy=(0.97, 0.93),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=COLOR_STEM,
        bbox=dict(
            boxstyle="round,pad=0.3", fc="white", ec=COLOR_STEM, lw=0.8
        ),
    )

# ---------------------------------------------------------------------------
# Bottom: metric comparison bar chart
# ---------------------------------------------------------------------------
metrics = {
    "Coverage": (cov_iso, cov_clu),
    "Mean relative\nexceedance": (exc_iso, exc_clu),
    "Mean interval\nscore": (wink_iso, wink_clu),
    "CAS": (cas_iso, cas_clu),
}
metric_names = list(metrics.keys())
iso_vals = [metrics[k][0] for k in metric_names]
clu_vals = [metrics[k][1] for k in metric_names]

x = np.arange(len(metric_names))
bw = 0.32

bars_iso = ax_bar.bar(
    x - bw / 2,
    iso_vals,
    bw,
    color="#2196F3",
    alpha=0.82,
    label="Isolated",
    edgecolor="white",
    lw=0.8,
)
bars_clu = ax_bar.bar(
    x + bw / 2,
    clu_vals,
    bw,
    color="#E53935",
    alpha=0.82,
    label="Clustered",
    edgecolor="white",
    lw=0.8,
)

# Value annotations on bars
for bars in (bars_iso, bars_clu):
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.002,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metric_names, fontsize=10.5)
ax_bar.set_ylabel("Metric value")
ax_bar.set_title(
    "Metric comparison: same violations, different arrangement",
    fontweight="bold",
)
ax_bar.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)
ax_bar.grid(True, axis="y", color=COLOR_GRID)
for sp in ("top", "right"):
    ax_bar.spines[sp].set_visible(False)

# Highlight the CAS bar pair with a bracket
y_top = max(cas_iso, cas_clu) * 1.35
ax_bar.annotate(
    "",
    xy=(x[-1] - bw / 2, cas_iso),
    xytext=(x[-1] + bw / 2, cas_clu),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
)
ax_bar.text(
    x[-1],
    max(cas_iso, cas_clu) * 1.15,
    f"+{100 * (cas_clu - cas_iso) / cas_iso:.0f}%",
    ha="center",
    fontsize=9,
    color="black",
    fontweight="bold",
)

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
fig.suptitle(
    "Controlled rearrangement experiment\n"
    r"$n=30$, 6 violations, $\hat{e}=0.1$, triangular kernel, $h=2$, $\lambda=1$, $\gamma=1$",
    y=1.01,
    fontsize=13,
    fontweight="bold",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_png = OUTDIR / "figure1_rearrangement.png"
out_pdf = OUTDIR / "figure1_rearrangement.pdf"

fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"\n[Saved] {out_png}")
print(f"[Saved] {out_pdf}")
print("[Done] figure1_rearrangement complete.")
