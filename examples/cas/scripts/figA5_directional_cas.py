"""
Appendix Figure A5 – Directional CAS: upper vs lower tail
==========================================================

Decomposes CAS violations into upper (y > q90) and lower (y < q10) tails.
Shows whether prediction intervals are systematically too narrow on one side.

Layout:
  Row 1 – Stacked proportion bars (% upper vs % lower violations) per
    (domain, model). A bar that is 100% upper means all violations are
    above q90; 50/50 indicates symmetric undercoverage.
  Row 2 – Side-by-side bars of mean upper-severity and mean lower-severity
    per (domain, model), showing asymmetry in violation magnitude.

Computes from raw prediction CSVs (h=5, triangular kernel, λ=1, γ=1).

Outputs (data/cas/outputs/):
  figA5_directional_cas.png / .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

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

H_DEFAULT = 5
LAMBDA_ = 1.0
GAMMA_ = 1.0
KERNEL = "triangular"


# ── helpers ───────────────────────────────────────────────────────────────────
def load_domain(pred_file):
    df = pd.read_csv(pred_file)
    q10, q50, q90 = enforce_non_crossing(
        df["q10"].to_numpy(), df["q50"].to_numpy(), df["q90"].to_numpy()
    )
    df["q10"], df["q50"], df["q90"] = q10, q50, q90
    df = df[df["q90"] - df["q10"] > 1e-6].reset_index(drop=True)
    return df


def directional_stats(df, model, h=H_DEFAULT):
    """Return (n_over, n_under, sev_over, sev_under) averaged over horizons."""
    sub = df[df["model"] == model].dropna(subset=["q10", "q90", "y"])
    if sub.empty:
        return np.nan, np.nan, np.nan, np.nan

    over_counts, under_counts = [], []
    over_sevs, under_sevs = [], []

    for _, grp in sub.groupby("horizon"):
        y = grp["y"].to_numpy()
        yp = np.column_stack([grp["q10"].to_numpy(), grp["q90"].to_numpy()])
        if len(y) < 3:
            continue
        try:
            _, det = cluster_aware_severity_score(
                y,
                yp,
                window_size=h,
                kernel=KERNEL,
                lambda_=LAMBDA_,
                gamma=GAMMA_,
                return_details=True,
            )
        except Exception:
            continue

        over = det[det["type"] == "over"]
        under = det[det["type"] == "under"]
        over_counts.append(len(over))
        under_counts.append(len(under))
        over_sevs.append(over["severity"].mean() if len(over) > 0 else 0.0)
        under_sevs.append(under["severity"].mean() if len(under) > 0 else 0.0)

    if not over_counts:
        return np.nan, np.nan, np.nan, np.nan

    n_over = float(np.mean(over_counts))
    n_under = float(np.mean(under_counts))
    s_over = float(np.mean(over_sevs))
    s_under = float(np.mean(under_sevs))
    return n_over, n_under, s_over, s_under


# ── compute ───────────────────────────────────────────────────────────────────
print("Computing directional CAS …")
dir_stats = {}  # {domain: {model: (n_over, n_under, sev_over, sev_under)}}

for domain in DOMAIN_ORDER:
    df = load_domain(PRED_FILES[domain])
    dir_stats[domain] = {}
    for model in MODEL_ORDER:
        stats = directional_stats(df, model)
        dir_stats[domain][model] = stats
        n_o, n_u, s_o, s_u = stats
        if np.isfinite(n_o):
            print(
                f"  {domain}/{model}: over={n_o:.1f}({s_o:.4f})  "
                f"under={n_u:.1f}({s_u:.4f})"
            )
        else:
            print(f"  {domain}/{model}: no data")


# ── colour helpers ────────────────────────────────────────────────────────────
def darken(hex_col, factor):
    r, g, b = (
        int(hex_col[1:3], 16),
        int(hex_col[3:5], 16),
        int(hex_col[5:7], 16),
    )
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


def lighten(hex_col, factor):
    r = int(hex_col[1:3], 16)
    g = int(hex_col[3:5], 16)
    b = int(hex_col[5:7], 16)
    r2 = int(r + (255 - r) * factor)
    g2 = int(g + (255 - g) * factor)
    b2 = int(b + (255 - b) * factor)
    return f"#{r2:02x}{g2:02x}{b2:02x}"


# ── figure ────────────────────────────────────────────────────────────────────
with plt.rc_context({"figure.constrained_layout.use": False}):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.set_constrained_layout(False)
fig.subplots_adjust(
    left=0.07, right=0.97, top=0.90, bottom=0.09, hspace=0.52, wspace=0.33
)

BW = 0.22  # bar width per model

# ── row 0: stacked proportion bars (% over vs % under) ───────────────────────
for di, domain in enumerate(DOMAIN_ORDER):
    ax = axes[0, di]
    col = DOMAIN_COLOR[domain]
    ax.set_facecolor(col + "0D")

    x_pos = []
    labels = []
    over_pcts, under_pcts = [], []

    for mi, model in enumerate(MODEL_ORDER):
        n_o, n_u, *_ = dir_stats[domain][model]
        if not np.isfinite(n_o):
            continue
        total = n_o + n_u
        pct_o = 100.0 * n_o / total if total > 0 else 0.0
        pct_u = 100.0 * n_u / total if total > 0 else 0.0
        x_pos.append(mi)
        labels.append(MODEL_LABEL[model])
        over_pcts.append(pct_o)
        under_pcts.append(pct_u)

    x = np.array(x_pos, dtype=float)
    col_over = darken(col, 0.70)
    col_under = lighten(col, 0.55)

    if len(x) > 0:
        b_over = ax.bar(
            x,
            over_pcts,
            BW,
            color=col_over,
            alpha=0.88,
            edgecolor="white",
            lw=0.8,
            label="Upper (over $q_{90}$)",
        )
        b_under = ax.bar(
            x,
            under_pcts,
            BW,
            bottom=over_pcts,
            color=col_under,
            alpha=0.88,
            edgecolor="white",
            lw=0.8,
            hatch="...",
            label="Lower (under $q_{10}$)",
        )

        # Annotate percentages inside the bars
        for i, (po, pu) in enumerate(zip(over_pcts, under_pcts)):
            if po > 5:
                ax.text(
                    x[i],
                    po / 2,
                    f"{po:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
            if pu > 5:
                ax.text(
                    x[i],
                    po + pu / 2,
                    f"{pu:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=darken(col, 0.5),
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(
        DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12
    )
    if di == 0:
        ax.set_ylabel("Share of violations", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92, loc="upper right")
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# ── row 1: side-by-side bars of upper/lower mean severity ─────────────────────
GAP = 0.05
GROUP_W = 2 * BW + GAP

for di, domain in enumerate(DOMAIN_ORDER):
    ax = axes[1, di]
    col = DOMAIN_COLOR[domain]
    ax.set_facecolor(col + "0D")

    col_over = darken(col, 0.70)
    col_under = lighten(col, 0.35)

    sev_vals = []
    x_centers = np.arange(len(MODEL_ORDER)) * (GROUP_W + 0.08)
    valid_x, valid_lbls = [], []

    for mi, model in enumerate(MODEL_ORDER):
        n_o, n_u, s_o, s_u = dir_stats[domain][model]
        if not np.isfinite(s_o):
            continue
        x0 = x_centers[mi]
        valid_x.append(x0 + (BW + GAP / 2) / 2)
        valid_lbls.append(MODEL_LABEL[model])

        b1 = ax.bar(
            x0,
            s_o,
            BW,
            color=col_over,
            alpha=0.88,
            edgecolor="white",
            lw=0.8,
            label="Upper severity" if mi == 0 else "",
        )
        b2 = ax.bar(
            x0 + BW + GAP,
            s_u,
            BW,
            color=col_under,
            alpha=0.88,
            edgecolor="white",
            lw=0.8,
            hatch="...",
            label="Lower severity" if mi == 0 else "",
        )

        # Value annotations
        for bar, v in [(b1, s_o), (b2, s_u)]:
            if v > 0:
                ax.text(
                    bar[0].get_x() + bar[0].get_width() / 2,
                    v + max(v * 0.02, 1e-5),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90,
                )
        sev_vals.extend([s_o, s_u])

    # Log scale when range spans an order of magnitude
    finite_sev = [v for v in sev_vals if np.isfinite(v) and v > 0]
    use_log = len(finite_sev) > 1 and max(finite_sev) / min(finite_sev) > 10
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            mtick.LogFormatterSciNotation(labelOnlyBase=False)
        )
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

    if valid_x:
        ax.set_xticks(valid_x)
        ax.set_xticklabels(valid_lbls, fontsize=10)
    ax.set_title(
        DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12
    )
    if di == 0:
        ax.set_ylabel("Mean violation severity", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.suptitle(
    "Directional CAS: upper ($y > q_{90}$) vs. lower ($y < q_{10}$) tail breakdown\n"
    "(top: violation share; bottom: mean severity per direction; "
    r"$h=5$, triangular kernel, $\lambda=1$, $\gamma=1$)",
    fontsize=11,
    fontweight="bold",
    y=1.01,
)

out_png = OUTDIR / "figA5_directional_cas.png"
out_pdf = OUTDIR / "figA5_directional_cas.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"\n[Saved] {out_png}")
print(f"[Saved] {out_pdf}")
print("[Done] figA5_directional_cas complete.")
