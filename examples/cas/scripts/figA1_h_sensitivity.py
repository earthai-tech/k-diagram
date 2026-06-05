"""
Appendix Figure A1 – CAS sensitivity to neighbourhood scale h
==============================================================

Shows how CAS changes as the neighbourhood scale h varies from small to
large, for each domain × model combination.  The key message is that
absolute CAS values increase with h (more violations are considered
locally connected) while model rankings remain stable.

h grid: {3, 5, 7, 11, 15, 21}  (triangular kernel, λ=1, γ=1)

Layout: 1 row × 3 columns (one per domain).
Each panel: CAS vs h for QAR / QGBM / XTFT as styled lines,
with domain-coloured background shading.

Outputs (data/cas/outputs/):
  figA1_h_sensitivity.png / .pdf
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

# ── config ────────────────────────────────────────────────────────────────────
H_GRID = [3, 5, 7, 11, 15, 21]
H_DEFAULT = 5
KERNEL = "triangular"
LAMBDA_ = 1.0
GAMMA_ = 1.0

MODEL_COLOR = {  # tint within each panel (domain colour at 3 opacities)
    "qar": 1.00,
    "qgbm": 0.65,
    "xtft": 0.35,
}


# ── helpers ───────────────────────────────────────────────────────────────────
def load_domain(pred_file):
    df = pd.read_csv(pred_file)
    q10, q50, q90 = enforce_non_crossing(
        df["q10"].to_numpy(), df["q50"].to_numpy(), df["q90"].to_numpy()
    )
    df["q10"], df["q50"], df["q90"] = q10, q50, q90
    df = df[df["q90"] - df["q10"] > 1e-6].reset_index(drop=True)
    return df


def mean_cas_for_h(df, model, h):
    sub = df[df["model"] == model].dropna(subset=["q10", "q90", "y"])
    if sub.empty:
        return np.nan
    scores = []
    for _, grp in sub.groupby("horizon"):
        y = grp["y"].to_numpy()
        yp = np.column_stack([grp["q10"].to_numpy(), grp["q90"].to_numpy()])
        if len(y) < 3:
            continue
        try:
            s = cluster_aware_severity_score(
                y,
                yp,
                window_size=h,
                kernel=KERNEL,
                lambda_=LAMBDA_,
                gamma=GAMMA_,
            )
            if np.isfinite(s):
                scores.append(s)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else np.nan


# ── compute ───────────────────────────────────────────────────────────────────
print("Computing CAS sensitivity to h …")
records = {}  # {domain: {model: [cas per h]}}

for domain in DOMAIN_ORDER:
    df = load_domain(PRED_FILES[domain])
    records[domain] = {}
    for model in MODEL_ORDER:
        cas_vals = [mean_cas_for_h(df, model, h) for h in H_GRID]
        records[domain][model] = cas_vals
        finite = [f"{v:.4f}" if np.isfinite(v) else "nan" for v in cas_vals]
        print(f"  {domain}/{model}: {finite}")


# ── figure ────────────────────────────────────────────────────────────────────
with plt.rc_context({"figure.constrained_layout.use": False}):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
fig.set_constrained_layout(False)
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.14, wspace=0.30)

x = np.array(H_GRID)

for ax, domain in zip(axes, DOMAIN_ORDER):
    col = DOMAIN_COLOR[domain]

    # Light domain-coloured background
    ax.set_facecolor(col + "0D")  # 5 % opacity hex

    # Vertical dashed line at default h
    ax.axvline(
        H_DEFAULT,
        color="0.55",
        lw=0.9,
        ls=":",
        label=f"Default $h={H_DEFAULT}$",
        zorder=1,
    )

    for model in MODEL_ORDER:
        cas_vals = np.array(records[domain][model], dtype=float)
        if np.all(np.isnan(cas_vals)):
            continue

        # Shade the colour proportional to model rank ordering clarity
        alpha = MODEL_COLOR[model]
        # Darken the line colour by mixing with black
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        lc = f"#{int(r * alpha):02x}{int(g * alpha):02x}{int(b * alpha):02x}"

        ax.plot(
            x,
            cas_vals,
            ls=MODEL_STYLE[model],
            lw=2.2,
            marker=MODEL_MARK[model],
            ms=6.5,
            color=lc,
            alpha=0.92,
            label=MODEL_LABEL[model],
            zorder=3,
        )

        # Fill under line with very light shading
        ax.fill_between(x, cas_vals, alpha=0.07, color=lc, zorder=2)

    # Use log scale when values span more than one order of magnitude
    all_finite = [
        v
        for m in MODEL_ORDER
        for v in records[domain][m]
        if np.isfinite(v) and v > 0
    ]
    use_log = all_finite and (max(all_finite) / min(all_finite) > 10)
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            mtick.LogFormatterSciNotation(labelOnlyBase=False)
        )
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

    ax.set_title(
        DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12
    )
    ax.set_xlabel("Neighbourhood scale $h$", fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel("Mean CAS (averaged over horizons)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in H_GRID], fontsize=9)
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    ax.legend(fontsize=9, frameon=True, framealpha=0.92, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.suptitle(
    "CAS sensitivity to neighbourhood scale $h$\n"
    r"(triangular kernel, $\lambda=1$, $\gamma=1$; "
    "dashed = default; higher $h$ connects more distant violations)",
    fontsize=11.5,
    fontweight="bold",
    y=1.00,
)

out_png = OUTDIR / "figA1_h_sensitivity.png"
out_pdf = OUTDIR / "figA1_h_sensitivity.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"\n[Saved] {out_png}")
print(f"[Saved] {out_pdf}")
print("[Done] figA1_h_sensitivity complete.")
