"""
Appendix Figure A4 – CAS by forecast horizon
=============================================

Shows how CAS changes across forecast horizons for each domain and model.
This reveals whether short-term or long-term forecasts drive the overall CAS.

Layout:
  Row 1 (line plots) – CAS vs horizon for each domain (3 panels),
    one line per model.
  Row 2 (bar + heatmap) – Heat-encoded CAS table (domain × model × horizon)
    as a colour matrix, plus a summary bar of mean CAS.

Data source: metrics_all_domains.csv (already computed).

Outputs (data/cas/outputs/):
  figA4_horizon_profile.png / .pdf
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

import os as _os
_REPO_ROOT = _HERE.parents[2]
_REAL_DATA = _REPO_ROOT / "data" / "cas"
if _REAL_DATA.exists():
    _os.environ.setdefault("KDIAGRAM_DATA_DIR", str(_REAL_DATA))

try:
    from results_config import (
        DOMAIN_COLOR, DOMAIN_LABEL, DOMAIN_ORDER,
        METRICS_PATH,
        MODEL_LABEL, MODEL_MARK, MODEL_ORDER, MODEL_STYLE,
        OUTDIR,
    )
except ModuleNotFoundError:
    _DATA = _REAL_DATA
    OUTDIR = _DATA / "outputs"; OUTDIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH = _DATA / "modeling_results_ok" / "metrics_all_domains.csv"
    DOMAIN_COLOR  = {"hydro": "#0072B2", "wind": "#E69F00", "subsidence": "#009E73"}
    DOMAIN_LABEL  = {"hydro": "Hydrology", "wind": "Wind", "subsidence": "Subsidence"}
    DOMAIN_ORDER  = ["hydro", "wind", "subsidence"]
    MODEL_ORDER   = ["qar", "qgbm", "xtft"]
    MODEL_LABEL   = {"qar": "QAR", "qgbm": "QGBM", "xtft": "XTFT"}
    MODEL_MARK    = {"qar": "o", "qgbm": "s", "xtft": "^"}
    MODEL_STYLE   = {"qar": "-", "qgbm": "--", "xtft": "-."}

# ── load data ─────────────────────────────────────────────────────────────────
metrics = pd.read_csv(METRICS_PATH)
metrics["cas"] = pd.to_numeric(metrics["cas"], errors="coerce")

# horizon labels per domain
DOMAIN_HORIZONS = {
    d: sorted(metrics[metrics["domain"] == d]["horizon"].unique())
    for d in DOMAIN_ORDER
}

def darken(hex_col, factor):
    r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(int(r*factor), int(g*factor), int(b*factor))


# ── figure ────────────────────────────────────────────────────────────────────
with plt.rc_context({"figure.constrained_layout.use": False}):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.set_constrained_layout(False)
fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.09,
                    hspace=0.52, wspace=0.33)

# ── row 0: line plots (CAS vs horizon) ───────────────────────────────────────
for di, domain in enumerate(DOMAIN_ORDER):
    ax  = axes[0, di]
    col = DOMAIN_COLOR[domain]
    ax.set_facecolor(col + "0D")

    horizons = DOMAIN_HORIZONS[domain]
    sub = metrics[metrics["domain"] == domain]

    for model in MODEL_ORDER:
        subm = sub[sub["model"] == model].sort_values("horizon")
        cas_vals = subm.set_index("horizon")["cas"].reindex(horizons).values

        if np.all(np.isnan(cas_vals)):
            continue

        alpha_lc = {"qar": 1.0, "qgbm": 0.65, "xtft": 0.40}[model]
        r, g, b  = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        lc = "#{:02x}{:02x}{:02x}".format(
            int(r * alpha_lc), int(g * alpha_lc), int(b * alpha_lc))

        ax.plot(horizons, cas_vals,
                ls=MODEL_STYLE[model], lw=2.2,
                marker=MODEL_MARK[model], ms=7,
                color=lc, alpha=0.93,
                label=MODEL_LABEL[model], zorder=3)
        ax.fill_between(horizons, cas_vals, alpha=0.07, color=lc, zorder=2)

        # Annotate last point
        last_h  = horizons[-1]
        last_v  = subm[subm["horizon"] == last_h]["cas"].values
        if len(last_v) > 0 and np.isfinite(last_v[0]):
            ax.annotate(
                MODEL_LABEL[model],
                xy=(last_h, last_v[0]),
                xytext=(last_h + 0.4, last_v[0]),
                fontsize=8, color=lc, va="center",
            )

    ax.set_title(DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12)
    ax.set_xlabel("Forecast horizon (steps)", fontsize=9.5)
    if di == 0:
        ax.set_ylabel("CAS", fontsize=9.5)
    ax.set_xticks(horizons)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# ── row 1: heatmap (model × horizon per domain) ───────────────────────────────
CAS_CMAP = plt.cm.YlOrRd   # low = light yellow, high = dark red

# Global CAS range for shared colour scale
all_cas = metrics["cas"].dropna()
vmin, vmax = 0.0, np.percentile(all_cas, 97) if len(all_cas) > 0 else 1.0

for di, domain in enumerate(DOMAIN_ORDER):
    ax  = axes[1, di]
    col = DOMAIN_COLOR[domain]

    horizons = DOMAIN_HORIZONS[domain]
    sub = metrics[metrics["domain"] == domain]

    # Build matrix: rows=models, cols=horizons
    mat = np.full((len(MODEL_ORDER), len(horizons)), np.nan)
    for mi, model in enumerate(MODEL_ORDER):
        subm = sub[sub["model"] == model].sort_values("horizon")
        cas_vals = subm.set_index("horizon")["cas"].reindex(horizons).values
        mat[mi, :] = cas_vals

    im = ax.imshow(mat, aspect="auto", cmap=CAS_CMAP,
                   vmin=vmin, vmax=vmax, origin="upper")

    # Cell annotations
    for mi in range(len(MODEL_ORDER)):
        for hi_idx in range(len(horizons)):
            v = mat[mi, hi_idx]
            txt = f"{v:.3f}" if np.isfinite(v) else "—"
            # Pick text colour for contrast
            norm_v = (v - vmin) / max(vmax - vmin, 1e-9) if np.isfinite(v) else 0
            tc = "white" if norm_v > 0.55 else "black"
            ax.text(hi_idx, mi, txt, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=tc)

    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=10)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"h={h}" for h in horizons], fontsize=9)
    ax.set_title(DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12)
    if di == 0:
        ax.set_ylabel("Model", fontsize=9.5)
    ax.set_xlabel("Horizon", fontsize=9.5)

    # Per-panel border in domain colour
    for sp in ax.spines.values():
        sp.set_edgecolor(col)
        sp.set_linewidth(1.5)

# Shared colorbar — position computed after render to avoid overlap
fig.canvas.draw()
heat_axes = [axes[1, di] for di in range(len(DOMAIN_ORDER))]
heat_pos   = [ax.get_position() for ax in heat_axes]
cbar_left   = 0.905
cbar_bottom = min(p.y0 for p in heat_pos)
cbar_height = max(p.y1 for p in heat_pos) - cbar_bottom

cbar_ax = fig.add_axes([cbar_left, cbar_bottom, 0.016, cbar_height])
sm = plt.cm.ScalarMappable(cmap=CAS_CMAP,
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("CAS", fontsize=9, labelpad=6)
cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    "CAS by forecast horizon\n"
    "(top: line profiles per domain; bottom: horizon × model heat map)",
    fontsize=11.5, fontweight="bold", y=1.01,
)

out_png = OUTDIR / "figA4_horizon_profile.png"
out_pdf = OUTDIR / "figA4_horizon_profile.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] {out_png}")
print(f"[Saved] {out_pdf}")
print("[Done] figA4_horizon_profile complete.")
