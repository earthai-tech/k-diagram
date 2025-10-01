"""
R9: Error taxonomy & links to standard scores (succinct)
Figure 11: log(1+|CAS|) vs CRPS scatter (color = under-coverage)
+ small partials vs horizon / under-coverage

Reads:
  modeling_results_ok/metrics_all_domains.csv

Writes:
  outputs/figure11_error_taxonomy.{png,pdf}
  outputs/figure11_correlations.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------------------
# Config import (+ safe fallbacks)
# ----------------------------
try:
    from results_config import (  # type: ignore
        BASE_DIR,
        DOMAIN_COLOR,
        DOMAIN_ORDER,
        MODEL_LABEL,
        MODEL_MARK,
        NOMINAL_COVERAGE,
        OUTDIR,
    )
except Exception:
    BASE_DIR = Path(__file__).resolve().parent
    OUTDIR = BASE_DIR / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    DOMAIN_COLOR = {
        "hydro": "#0072B2",
        "wind": "#E69F00",
        "subsidence": "#009E73",
    }
    MODEL_LABEL = {"qar": "QAR", "qgbm": "QGBM", "xtft": "XTFT"}
    # try to import MODEL_MARKER name if present
    try:
        from results_config import MODEL_MARKER as MODEL_MARK  # type: ignore
    except Exception:
        MODEL_MARK = {"qar": "o", "qgbm": "s", "xtft": "^"}
    try:
        from results_config import DOMAIN_ORDER as _DO  # type: ignore

        DOMAIN_ORDER = _DO
    except Exception:
        DOMAIN_ORDER = ["hydro", "wind", "subsidence"]
    try:
        from results_config import NOMINAL_COVERAGE as _NC  # type: ignore

        NOMINAL_COVERAGE = _NC
    except Exception:
        NOMINAL_COVERAGE = 0.90

IN_AGG = BASE_DIR / "modeling_results_ok" / "metrics_all_domains.csv"

# ----------------------------
# Style
# ----------------------------
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 20,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.2,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "figure.constrained_layout.use": False,
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
            "Times New Roman",
            "Times",
            "Computer Modern Roman",
        ],
    }
)


# ----------------------------
# Helpers
# ----------------------------
def _find_col(df: pd.DataFrame, candidates):
    """Return first present column (case-insensitive) from candidates."""
    lookup = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    for c in df.columns:
        lc = c.lower()
        if any(lc.startswith(name.lower()) for name in candidates):
            return c
    raise KeyError(f"None of {candidates} in columns: {list(df.columns)}")


def _spearman_np(x, y):
    """Spearman rho via numpy (rank correlation)."""
    xr = pd.Series(x).rank(method="average")
    yr = pd.Series(y).rank(method="average")
    if xr.nunique() < 2 or yr.nunique() < 2:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _bin_curve(x, y, nbins=8):
    """Quantile-binned smoothing; returns bin means."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return np.array([]), np.array([])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.digitize(x, edges[1:-1], right=False)
    xb, yb = [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.sum() >= 2:
            xb.append(x[sel].mean())
            yb.append(y[sel].mean())
    return np.array(xb), np.array(yb)


# ----------------------------
# Load & standardize columns
# ----------------------------
agg = pd.read_csv(IN_AGG)

crps_col = _find_col(agg, ["crps", "mean_crps", "mean_crp"])
if "coverage" in [c.lower() for c in agg.columns]:
    cov_col = _find_col(agg, ["coverage"])
else:
    dc = _find_col(agg, ["delta_cov"])
    agg["coverage"] = NOMINAL_COVERAGE + agg[dc].astype(float)
    cov_col = "coverage"

if any(c.lower() in ("abs_cas", "mean_abs_cas") for c in agg.columns):
    abs_col = _find_col(agg, ["abs_cas", "mean_abs_cas"])
else:
    cas_signed = _find_col(agg, ["cas", "mean_cas"])
    agg["abs_cas"] = agg[cas_signed].astype(float).abs()
    abs_col = "abs_cas"

h_col = _find_col(agg, ["horizon", "h"])
m_col = _find_col(agg, ["model"])
d_col = _find_col(agg, ["domain"])

agg_std = agg.rename(
    columns={
        d_col: "domain",
        m_col: "model",
        h_col: "horizon",
        cov_col: "coverage",
        crps_col: "crps",
        abs_col: "abs_cas",
    }
)
agg_std["model"] = agg_std["model"].astype(str).str.lower()
agg_std["under_cov"] = (
    NOMINAL_COVERAGE - agg_std["coverage"].astype(float)
).clip(lower=0)
agg_std["log1p_abs_cas"] = np.log1p(agg_std["abs_cas"].astype(float))
perh = agg_std.copy()

# ----------------------------
# Figure
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), sharey=True)
fig.subplots_adjust(left=0.07, right=0.88, top=0.92, bottom=0.18, wspace=0.11)

uc = agg_std["under_cov"].to_numpy(float)
uc_min = float(np.nanmin(uc)) if np.isfinite(uc).any() else 0.0
uc_max = float(np.nanmax(uc)) if np.isfinite(uc).any() else 0.0
show_colorbar = (uc_max - uc_min) > 1e-8
norm = Normalize(vmin=max(0.0, uc_min), vmax=max(1e-6, uc_max))
cmap = plt.cm.Blues

rho_rows = []
for j, domain in enumerate(["hydro", "wind", "subsidence"]):
    ax = axes[j]
    sub = agg_std[agg_std["domain"] == domain]
    if sub.empty:
        ax.set_visible(False)
        continue

    # main scatter
    for m, g in sub.groupby("model"):
        ax.scatter(
            g["crps"],
            g["log1p_abs_cas"],
            marker=MODEL_MARK.get(m, "o"),
            s=72,
            alpha=0.95,
            zorder=3,
            c=(
                cmap(norm(g["under_cov"]))
                if show_colorbar
                else DOMAIN_COLOR[domain]
            ),
            edgecolor="black",
            linewidth=0.6,
            label=MODEL_LABEL.get(m, m.upper()),
        )

    # domain-level Spearman (invariant to log1p)
    rho = _spearman_np(sub["crps"], sub["abs_cas"])
    rho_rows.append(
        {
            "domain": domain,
            "spearman_rho_absCAS_vs_CRPS": rho,
        }
    )
    ax.text(
        0.03,
        0.96,
        rf"$\rho_s$={rho:0.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="k",
        bbox=dict(
            boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.95
        ),
    )

    # cosmetics
    title_map = {
        "hydro": "Hydro",
        "wind": "Wind",
        "subsidence": "Subsidence",
    }
    ax.set_title(title_map[domain], color=DOMAIN_COLOR[domain], pad=8)
    ax.set_xlabel("CRPS (lower is better)")
    if j == 0:
        ax.set_ylabel(r"$\log(1 + |\mathrm{CAS}|)$ (lower is better)")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- inset: vs under-coverage (if varied)
    ph = perh[perh["domain"] == domain]
    has_uc = (
        ph["under_cov"].std(skipna=True) > 1e-6
        and ph["under_cov"].count() >= 6
    )
    if has_uc:
        ax1 = inset_axes(
            ax,
            width="58%",
            height="48%",
            loc="lower right",
            bbox_to_anchor=(0.04, 0.04, 0.58, 0.48),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        drew = False
        for m, g in ph.groupby("model"):
            xb, yb = _bin_curve(
                g["under_cov"], np.log1p(g["abs_cas"]), nbins=8
            )
            if xb.size:
                ax1.plot(
                    xb,
                    yb,
                    color=DOMAIN_COLOR[domain],
                    linestyle={"qar": "-", "qgbm": "--", "xtft": ":"}.get(
                        m, "-"
                    ),
                    linewidth=2.0,
                    alpha=0.95,
                )
                drew = True
        if drew:
            ax1.set_title("vs under-coverage", fontsize=10, pad=2)
            ax1.tick_params(labelsize=8)
            ax1.grid(True, linestyle=":", alpha=0.3)
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
        else:
            ax1.remove()

    # --- inset: vs horizon
    ax2 = inset_axes(
        ax,
        width="58%",
        height="48%",
        loc="upper right",
        bbox_to_anchor=(0.52, 0.54, 0.58, 0.48),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    drew_any = False
    for m, g in ph.groupby("model"):
        g2 = g.sort_values("horizon")
        if g2.empty:
            continue
        ax2.plot(
            g2["horizon"],
            np.log1p(g2["abs_cas"]),
            color=DOMAIN_COLOR[domain],
            linestyle={"qar": "-", "qgbm": "--", "xtft": ":"}.get(m, "-"),
            linewidth=2.0,
            alpha=0.95,
        )
        drew_any = True
    if drew_any:
        ax2.set_title("vs horizon", fontsize=10, pad=2)
        ax2.tick_params(labelsize=8)
        ax2.grid(True, linestyle=":", alpha=0.3)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        ax2.remove()


legend_handles = [
    Line2D(
        [0],
        [0],
        marker=MODEL_MARK.get(m, "o"),
        linestyle="",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        label=MODEL_LABEL.get(m, m.upper()),
    )
    for m in ["qar", "qgbm", "xtft"]
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, 0.03),
)

# colorbar if there is variation
if show_colorbar:
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f"Under-coverage ({NOMINAL_COVERAGE} − coverage)")

# Save
png = OUTDIR / "figure11_error_taxonomy.png"
pdf = OUTDIR / "figure11_error_taxonomy.pdf"
fig.savefig(png, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

# Domain-level Spearman rhos (|CAS| vs CRPS)
corr_path = OUTDIR / "figure11_correlations.csv"
pd.DataFrame(rho_rows).to_csv(corr_path, index=False)

print(f"[Saved] {png}")
print(f"[Saved] {pdf}")
print(f"[Saved] {corr_path}")
