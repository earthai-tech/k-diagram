"""
Appendix Figure A2 – CAS(λ) linear fan and pairwise crossing points
=====================================================================

For fixed h, K, γ, each model's CAS is linear in λ:
  CAS_j(λ) = A_j + λ · B_j

where A_j = mean exceedance (λ=0 baseline) and
      B_j = mean density-weighted exceedance.

This figure:
  Panel row 1 (3 plots, one per domain): CAS(λ) fan lines for all models,
    crossing points marked where two models swap rank.
  Panel row 2 (1 summary plot): A_j vs B_j scatter with model markers,
    showing which models have both low baseline and low density slope.

Outputs (data/cas/outputs/):
  figA2_lambda_crossing.png / .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import os  # noqa: E402

_REPO_ROOT = _HERE.parents[2]
_REAL_DATA = _REPO_ROOT / "data" / "cas"
if _REAL_DATA.exists():
    os.environ.setdefault("KDIAGRAM_DATA_DIR", str(_REAL_DATA))

try:
    from results_config import (
        DOMAIN_COLOR,
        DOMAIN_LABEL,
        DOMAIN_ORDER,
        MODEL_LABEL,
        MODEL_MARK,
        MODEL_ORDER,
        MODEL_STYLE,
        OUTDIR,
        PRED_HYDRO,
        PRED_SUBS,
        PRED_WIND,
        enforce_non_crossing,
    )
except ModuleNotFoundError:
    _DATA = _REAL_DATA
    OUTDIR = _DATA / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PRED_WIND = _DATA / "modeling_results_ok" / "predictions_wind.csv"
    PRED_HYDRO = _DATA / "modeling_results_ok" / "predictions_hydro.csv"
    PRED_SUBS = _DATA / "modeling_results_ok" / "predictions_subsidence.csv"
    DOMAIN_COLOR = {
        "hydro": "#0072B2",
        "wind": "#E69F00",
        "subsidence": "#009E73",
    }
    DOMAIN_LABEL = {
        "hydro": "Hydrology",
        "wind": "Wind",
        "subsidence": "Subsidence",
    }
    DOMAIN_ORDER = ["hydro", "wind", "subsidence"]
    MODEL_ORDER = ["qar", "qgbm", "xtft"]
    MODEL_LABEL = {"qar": "QAR", "qgbm": "QGBM", "xtft": "XTFT"}
    MODEL_MARK = {"qar": "o", "qgbm": "s", "xtft": "^"}
    MODEL_STYLE = {"qar": "-", "qgbm": "--", "xtft": "-."}

    def enforce_non_crossing(q10, q50, q90):
        q50c = np.maximum(q50, q10)
        return q10, q50c, np.maximum(q90, q50c)


if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from kdiagram.metrics import cluster_aware_severity_score  # noqa: E402

PRED_FILES = {"hydro": PRED_HYDRO, "wind": PRED_WIND, "subsidence": PRED_SUBS}

# ── config ────────────────────────────────────────────────────────────────────
H_DEFAULT = 5
KERNEL = "triangular"
GAMMA_ = 1.0
LAM_DEFAULT = 1.0
LAM_RANGE = np.linspace(0.0, 3.0, 200)
LAM_PRACTICAL = (0.5, 2.0)  # shaded region


# ── compute A_j and B_j per model ────────────────────────────────────────────
def load_domain(pred_file):
    df = pd.read_csv(pred_file)
    q10, q50, q90 = enforce_non_crossing(
        df["q10"].to_numpy(), df["q50"].to_numpy(), df["q90"].to_numpy()
    )
    df["q10"], df["q50"], df["q90"] = q10, q50, q90
    df = df[df["q90"] - df["q10"] > 1e-6].reset_index(drop=True)
    return df


def compute_AB(df, model, h=H_DEFAULT, kernel=KERNEL, gamma=GAMMA_):
    """Return (A_j, B_j) averaged over all horizons."""
    sub = df[df["model"] == model].dropna(subset=["q10", "q90", "y"])
    if sub.empty:
        return np.nan, np.nan
    A_list, B_list = [], []
    for _, grp in sub.groupby("horizon"):
        y = grp["y"].to_numpy()
        q10 = grp["q10"].to_numpy()
        q90 = grp["q90"].to_numpy()
        if len(y) < 3:
            continue
        # CAS at λ=0 → A_j
        yp = np.column_stack([q10, q90])
        try:
            A = cluster_aware_severity_score(
                y, yp, window_size=h, kernel=kernel, lambda_=0.0, gamma=gamma
            )
            # CAS at λ=1 → A_j + B_j
            AB = cluster_aware_severity_score(
                y, yp, window_size=h, kernel=kernel, lambda_=1.0, gamma=gamma
            )
            if np.isfinite(A) and np.isfinite(AB):
                A_list.append(A)
                B_list.append(AB - A)
        except Exception:
            pass
    if not A_list:
        return np.nan, np.nan
    return float(np.mean(A_list)), float(np.mean(B_list))


print("Computing A_j and B_j …")
AB = {}  # {domain: {model: (A, B)}}
for domain in DOMAIN_ORDER:
    df = load_domain(PRED_FILES[domain])
    AB[domain] = {}
    for model in MODEL_ORDER:
        A, B = compute_AB(df, model)
        AB[domain][model] = (A, B)
        print(f"  {domain}/{model}: A={A:.5f}  B={B:.5f}")


def crossing_lambda(Ai, Bi, Aj, Bj):
    """λ* where CAS_i(λ) = CAS_j(λ).  None if parallel or non-positive."""
    dB = Bi - Bj
    if abs(dB) < 1e-12:
        return None
    lam = (Aj - Ai) / dB
    return float(lam) if lam > 0 else None


# ── figure ────────────────────────────────────────────────────────────────────
with plt.rc_context({"figure.constrained_layout.use": False}):
    fig = plt.figure(figsize=(15, 9))
fig.set_constrained_layout(False)
fig.subplots_adjust(
    left=0.07, right=0.97, top=0.88, bottom=0.10, hspace=0.52, wspace=0.32
)

outer_gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.52)
top_gs = outer_gs[0].subgridspec(1, 3, wspace=0.32)
bot_ax = fig.add_subplot(outer_gs[1])

# ── row 1: CAS(λ) fan per domain ─────────────────────────────────────────────
for di, domain in enumerate(DOMAIN_ORDER):
    ax = fig.add_subplot(top_gs[di])
    col = DOMAIN_COLOR[domain]
    ax.set_facecolor(col + "0D")

    # Shade the practical λ range
    ax.axvspan(
        *LAM_PRACTICAL,
        color=col,
        alpha=0.07,
        zorder=0,
        label=f"Practical range [{LAM_PRACTICAL[0]}, {LAM_PRACTICAL[1]}]",
    )
    ax.axvline(LAM_DEFAULT, color="0.45", lw=0.9, ls=":", zorder=1)

    lines_plotted = []
    for model in MODEL_ORDER:
        A, B = AB[domain][model]
        if not (np.isfinite(A) and np.isfinite(B)):
            continue
        cas_lam = A + LAM_RANGE * B

        # Darken by 30–70% for model visual separation
        alpha_lc = {"qar": 1.0, "qgbm": 0.65, "xtft": 0.38}[model]
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        lc = f"#{int(r * alpha_lc):02x}{int(g * alpha_lc):02x}{int(b * alpha_lc):02x}"

        (ln,) = ax.plot(
            LAM_RANGE,
            cas_lam,
            ls=MODEL_STYLE[model],
            lw=2.0,
            color=lc,
            label=MODEL_LABEL[model],
            zorder=3,
        )
        lines_plotted.append((model, A, B, lc, ln))

    # Mark crossing points within [0, 3]
    models_plotted = [m for m, *_ in lines_plotted]
    for i in range(len(models_plotted)):
        for j in range(i + 1, len(models_plotted)):
            mi, mj = models_plotted[i], models_plotted[j]
            Ai, Bi = AB[domain][mi]
            Aj, Bj = AB[domain][mj]
            lam_x = crossing_lambda(Ai, Bi, Aj, Bj)
            if lam_x is not None and LAM_RANGE[0] <= lam_x <= LAM_RANGE[-1]:
                cas_x = Ai + lam_x * Bi
                ax.scatter(
                    lam_x,
                    cas_x,
                    s=80,
                    zorder=6,
                    color="0.2",
                    marker="X",
                    edgecolors="white",
                    lw=0.8,
                )
                ax.annotate(
                    f"$\\lambda^*={lam_x:.2f}$",
                    xy=(lam_x, cas_x),
                    xytext=(lam_x + 0.12, cas_x * 1.06),
                    fontsize=7.5,
                    color="0.2",
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6),
                )

    # Log y if range spans an order of magnitude
    cas_at_default = [
        AB[domain][m][0] + LAM_DEFAULT * AB[domain][m][1]
        for m in MODEL_ORDER
        if np.isfinite(AB[domain][m][0])
        and np.isfinite(AB[domain][m][1])
        and AB[domain][m][0] + LAM_DEFAULT * AB[domain][m][1] > 0
    ]
    use_log_y = (
        len(cas_at_default) > 1
        and max(cas_at_default) / min(cas_at_default) > 10
    )
    if use_log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            mtick.LogFormatterSciNotation(labelOnlyBase=False)
        )
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

    ax.set_title(
        DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12
    )
    ax.set_xlabel("Cluster weight $\\lambda$", fontsize=10)
    if di == 0:
        ax.set_ylabel("CAS$(\\lambda)$", fontsize=10)
    ax.set_xlim(LAM_RANGE[0] - 0.05, LAM_RANGE[-1] + 0.05)
    ax.grid(True, axis="y", alpha=0.20, ls="--")
    ax.legend(fontsize=9, frameon=True, framealpha=0.92, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# ── row 2: A_j vs B_j scatter (all domains together) ─────────────────────────
for domain in DOMAIN_ORDER:
    col = DOMAIN_COLOR[domain]
    for model in MODEL_ORDER:
        A, B = AB[domain][model]
        if not (np.isfinite(A) and np.isfinite(B)):
            continue
        bot_ax.scatter(
            A,
            B,
            s=110,
            color=col,
            marker=MODEL_MARK[model],
            edgecolors="white",
            lw=0.9,
            zorder=4,
            label=f"{DOMAIN_LABEL[domain]} / {MODEL_LABEL[model]}",
        )
        bot_ax.annotate(
            MODEL_LABEL[model],
            xy=(A, B),
            xytext=(A, B + max(0.0002, B * 0.04)),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=col,
        )

# Use log scale when A or B values span more than one order of magnitude
all_A = [
    AB[d][m][0]
    for d in DOMAIN_ORDER
    for m in MODEL_ORDER
    if np.isfinite(AB[d][m][0]) and AB[d][m][0] > 0
]
all_B = [
    AB[d][m][1]
    for d in DOMAIN_ORDER
    for m in MODEL_ORDER
    if np.isfinite(AB[d][m][1]) and AB[d][m][1] > 0
]
if all_A and all_B:
    use_log = (max(all_A) / min(all_A) > 10) or (max(all_B) / min(all_B) > 10)
else:
    use_log = False

if use_log:
    bot_ax.set_xscale("log")
    bot_ax.set_yscale("log")
else:
    bot_ax.set_xlim(left=0)
    bot_ax.set_ylim(bottom=0)

bot_ax.set_xlabel("Baseline CAS   $A_j$  (at $\\lambda=0$)", fontsize=10)
bot_ax.set_ylabel(
    "Density slope  $B_j$  (gain per unit $\\lambda$)", fontsize=10
)
bot_ax.set_title(
    "Baseline vs. density-slope decomposition of CAS$(\\lambda) = A_j + \\lambda B_j$",
    fontweight="bold",
    fontsize=11,
)
bot_ax.grid(True, alpha=0.20, ls="--")
for sp in ("top", "right"):
    bot_ax.spines[sp].set_visible(False)

# Domain + model legend (deduped)
legend_handles = [
    Patch(
        facecolor=DOMAIN_COLOR[d],
        edgecolor="white",
        alpha=0.85,
        label=DOMAIN_LABEL[d],
    )
    for d in DOMAIN_ORDER
] + [
    Line2D(
        [0],
        [0],
        marker=MODEL_MARK[m],
        color="0.3",
        lw=0,
        ms=7,
        label=MODEL_LABEL[m],
    )
    for m in MODEL_ORDER
]
bot_ax.legend(
    handles=legend_handles,
    ncol=6,
    fontsize=9,
    loc="upper left",
    frameon=True,
    framealpha=0.92,
)

fig.suptitle(
    r"CAS$(\lambda) = A_j + \lambda B_j$: linear fan diagram and crossing analysis"
    "\n(default $h=5$, triangular kernel, $\\gamma=1$; "
    r"$\times$ = crossing point $\lambda^*_{ij}$)",
    fontsize=11.5,
    fontweight="bold",
    y=1.00,
)

out_png = OUTDIR / "figA2_lambda_crossing.png"
out_pdf = OUTDIR / "figA2_lambda_crossing.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"\n[Saved] {out_png}")
print(f"[Saved] {out_pdf}")
print("[Done] figA2_lambda_crossing complete.")
