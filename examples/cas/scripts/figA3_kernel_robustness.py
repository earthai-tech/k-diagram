"""
Appendix Figure A3 – Kernel robustness: triangular vs box
==========================================================

Compares CAS under two kernels (triangular = paper default; box = robustness
check) for every (domain, model, horizon) combination.  Shows that:
  (a) absolute CAS is slightly higher under the box kernel (all violations
      in the window have equal weight),
  (b) model rankings are preserved across both kernels.

Layout:
  Row 1 – 3 scatter panels (one per domain): triangular CAS (x) vs
           box CAS (y) per (model, horizon) point. Points above the diagonal
           mean box > triangular.
  Row 2 – grouped-bar comparison of horizon-averaged CAS per (domain, model)
           for both kernels side-by-side.

Outputs (data/cas/outputs/):
  figA3_kernel_robustness.png / .pdf
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
KERNELS = ["triangular", "box"]
KERNEL_LABEL = {"triangular": "Triangular", "box": "Box"}
KERNEL_STYLE = {"triangular": "-", "box": "--"}
KERNEL_MARKER = {"triangular": "o", "box": "s"}


# ── compute ───────────────────────────────────────────────────────────────────
def load_domain(pred_file):
    df = pd.read_csv(pred_file)
    q10, q50, q90 = enforce_non_crossing(
        df["q10"].to_numpy(), df["q50"].to_numpy(), df["q90"].to_numpy()
    )
    df["q10"], df["q50"], df["q90"] = q10, q50, q90
    df = df[df["q90"] - df["q10"] > 1e-6].reset_index(drop=True)
    return df


def cas_per_horizon(df, model, kernel, h=H_DEFAULT):
    sub = df[df["model"] == model].dropna(subset=["q10", "q90", "y"])
    rows = []
    for horizon, grp in sub.groupby("horizon"):
        y = grp["y"].to_numpy()
        yp = np.column_stack([grp["q10"].to_numpy(), grp["q90"].to_numpy()])
        if len(y) < 3:
            continue
        try:
            s = cluster_aware_severity_score(
                y,
                yp,
                window_size=h,
                kernel=kernel,
                lambda_=LAMBDA_,
                gamma=GAMMA_,
            )
            if np.isfinite(s):
                rows.append({"horizon": horizon, "cas": s})
        except Exception:
            pass
    return pd.DataFrame(rows)


print("Computing kernel-robustness CAS …")
results = {}  # {domain: {model: {kernel: mean_cas}}}
point_data = {}  # {domain: DataFrame(model, horizon, tri_cas, box_cas)}

for domain in DOMAIN_ORDER:
    df = load_domain(PRED_FILES[domain])
    results[domain] = {}
    frames = []
    for model in MODEL_ORDER:
        df_tri = cas_per_horizon(df, model, "triangular")
        df_box = cas_per_horizon(df, model, "box")
        if df_tri.empty or df_box.empty:
            results[domain][model] = {"triangular": np.nan, "box": np.nan}
            continue
        merged = df_tri.merge(df_box, on="horizon", suffixes=("_tri", "_box"))
        merged["model"] = model
        frames.append(merged)
        results[domain][model] = {
            "triangular": df_tri["cas"].mean(),
            "box": df_box["cas"].mean(),
        }
        print(
            f"  {domain}/{model}: tri={results[domain][model]['triangular']:.4f}"
            f"  box={results[domain][model]['box']:.4f}"
        )
    point_data[domain] = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )


# ── figure ────────────────────────────────────────────────────────────────────
with plt.rc_context({"figure.constrained_layout.use": False}):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.set_constrained_layout(False)
fig.subplots_adjust(
    left=0.07, right=0.97, top=0.90, bottom=0.09, hspace=0.48, wspace=0.32
)

# ── row 0: scatter triangular vs box per (model, horizon) point ───────────────
for di, domain in enumerate(DOMAIN_ORDER):
    ax = axes[0, di]
    col = DOMAIN_COLOR[domain]
    ax.set_facecolor(col + "0D")

    pts = point_data[domain]
    if pts.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.5",
        )
        continue

    # Diagonal y = x reference
    all_vals = pd.concat([pts["cas_tri"], pts["cas_box"]]).dropna()
    if all_vals.empty:
        continue
    vmin, vmax = all_vals.min() * 0.9, all_vals.max() * 1.1
    ax.plot(
        [vmin, vmax],
        [vmin, vmax],
        color="0.55",
        lw=0.9,
        ls="--",
        label="$y = x$",
        zorder=1,
    )

    for model in MODEL_ORDER:
        sub = pts[pts["model"] == model]
        if sub.empty:
            continue
        alpha_lc = {"qar": 1.0, "qgbm": 0.65, "xtft": 0.38}[model]
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        lc = f"#{int(r * alpha_lc):02x}{int(g * alpha_lc):02x}{int(b * alpha_lc):02x}"
        ax.scatter(
            sub["cas_tri"],
            sub["cas_box"],
            s=65,
            color=lc,
            marker=MODEL_MARK[model],
            edgecolors="white",
            lw=0.7,
            zorder=4,
            label=MODEL_LABEL[model],
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(
        DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12
    )
    ax.set_xlabel("Triangular CAS", fontsize=9.5)
    if di == 0:
        ax.set_ylabel("Box CAS", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92, loc="upper left")
    ax.grid(True, alpha=0.20, ls="--")
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# ── row 1: grouped bar – mean CAS per model by kernel ─────────────────────────
# Collect means
bar_data = {}  # {domain: {model: {kernel: mean}}}
for domain in DOMAIN_ORDER:
    bar_data[domain] = results[domain]

n_models = len(MODEL_ORDER)
BW = 0.18  # single bar width
GAP = 0.05  # gap between kernel pair
GROUP_W = 2 * BW + GAP
TOTAL_W = n_models * GROUP_W + 0.25


# Model colour darkenings
def darken(hex_col, factor):
    r, g, b = (
        int(hex_col[1:3], 16),
        int(hex_col[3:5], 16),
        int(hex_col[5:7], 16),
    )
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


for di, domain in enumerate(DOMAIN_ORDER):
    ax = axes[1, di]
    col = DOMAIN_COLOR[domain]
    ax.set_facecolor(col + "0D")

    x_centers = np.arange(n_models) * GROUP_W
    for mi, model in enumerate(MODEL_ORDER):
        tri = bar_data[domain][model].get("triangular", np.nan)
        box = bar_data[domain][model].get("box", np.nan)
        x0 = x_centers[mi]
        lc_tri = darken(col, 0.85)
        lc_box = darken(col, 0.50)

        b1 = ax.bar(
            x0,
            tri,
            BW,
            color=lc_tri,
            alpha=0.88,
            edgecolor="white",
            lw=0.8,
            label="Triangular" if mi == 0 else "",
        )
        b2 = ax.bar(
            x0 + BW + GAP,
            box,
            BW,
            color=lc_box,
            alpha=0.88,
            edgecolor="white",
            lw=0.8,
            hatch="//",
            label="Box" if mi == 0 else "",
        )

        # Value labels
        for bar, v in [(b1, tri), (b2, box)]:
            if np.isfinite(v):
                ax.text(
                    bar[0].get_x() + bar[0].get_width() / 2,
                    v + max(v * 0.02, 1e-4),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    # Log y when values span an order of magnitude
    bar_vals = [
        bar_data[domain][m].get(k, np.nan)
        for m in MODEL_ORDER
        for k in KERNELS
        if np.isfinite(bar_data[domain][m].get(k, np.nan))
        and bar_data[domain][m].get(k, np.nan) > 0
    ]
    use_log = len(bar_vals) > 1 and max(bar_vals) / min(bar_vals) > 10
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            mtick.LogFormatterSciNotation(labelOnlyBase=False)
        )
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

    ax.set_xticks(x_centers + (BW + GAP / 2) / 2)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=10)
    ax.set_title(
        DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12
    )
    if di == 0:
        ax.set_ylabel("Mean CAS (horizons avg.)", fontsize=9.5)
    ax.legend(fontsize=9, frameon=True, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.suptitle(
    "Kernel robustness: triangular (paper default) vs. box kernel\n"
    "(top: scatter per horizon; bottom: horizon-averaged mean CAS;\n"
    " rankings preserved across both kernels when points lie close to the diagonal)",
    fontsize=11,
    fontweight="bold",
    y=1.01,
)

out_png = OUTDIR / "figA3_kernel_robustness.png"
out_pdf = OUTDIR / "figA3_kernel_robustness.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"\n[Saved] {out_png}")
print(f"[Saved] {out_pdf}")
print("[Done] figA3_kernel_robustness complete.")
