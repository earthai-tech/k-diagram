"""
R3: Per-horizon analysis — Table 3 + Figure 4

Reads (via results_config):
  metrics_all_domains.csv
    cols: domain, model, horizon, n, coverage, delta_cov,
          winkler, crps, cas

Writes (under data/cas/outputs/):
  table3_per_horizon_winners.(csv|tex)
  figure4a_cas_vs_horizon.(png|pdf)
  figure4b_cas_vs_horizon_log.(png|pdf)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from results_config import (
    DOMAIN_COLOR,
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    METRICS_PATH,
    MODEL_LABEL,
    MODEL_MARK,
    MODEL_ORDER,
    OUTDIR,
)

# ----------------------------
# Config & aesthetics
# ----------------------------
HORIZONS = [1, 3, 7, 14, 28]

# neutral line styles per model
MODEL_STYLE = {"qar": "-", "qgbm": "--", "xtft": "-."}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "figure.constrained_layout.use": True,
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
            "Times New Roman",
            "Times",
            "Computer Modern Roman",
        ],
    }
)

OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load & prepare
# ----------------------------
met = pd.read_csv(METRICS_PATH)
met["abs_cas"] = met["cas"].abs()

# ----------------------------
# Table 3: winners by min |CAS|
# per (domain, horizon)
# ----------------------------
w_idx = met.groupby(["domain", "horizon"])["abs_cas"].idxmin()

winners = (
    met.loc[
        w_idx,
        [
            "domain",
            "horizon",
            "model",
            "abs_cas",
            "cas",
            "coverage",
            "delta_cov",
            "crps",
            "winkler",
        ],
    ]
    .sort_values(["domain", "horizon"])
    .reset_index(drop=True)
)

csv_path = OUTDIR / "table3_per_horizon_winners.csv"
winners.to_csv(csv_path, index=False)


def winners_to_latex(df: pd.DataFrame) -> str:
    df2 = df.copy()
    df2["model"] = (
        df2["model"].map(MODEL_LABEL).apply(lambda s: f"\\textbf{{{s}}}")
    )
    return df2.to_latex(
        index=False,
        float_format="%.4g",
        columns=[
            "domain",
            "horizon",
            "model",
            "abs_cas",
            "cas",
            "coverage",
            "delta_cov",
            "crps",
            "winkler",
        ],
        header=[
            "domain",
            "h",
            "winner",
            "|CAS|",
            "CAS",
            "cov",
            "Δcov",
            "CRPS",
            "Winkler",
        ],
    )


tex_path = OUTDIR / "table3_per_horizon_winners.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(winners_to_latex(winners))

print(f"[Saved] {csv_path}")
print(f"[Saved] {tex_path}")


# ----------------------------
# Fig 4a: |CAS| vs horizon
# domain panels; robust y-limits
# + clipped outlier flags
# ----------------------------
def robust_ylim_domain(
    df_dom: pd.DataFrame,
    q: float = 0.95,
    pad: float = 1.10,
    floor: float = 0.2,
):
    y = df_dom["abs_cas"].to_numpy()
    top = np.quantile(y, q) if y.size else floor
    top = max(top, floor) * pad
    return 0.0, float(top)


fig4a, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), sharey=False)

for j, d in enumerate(DOMAIN_ORDER):
    ax = axes[j]
    sub = met[met["domain"] == d].copy()
    sub["abs_cas"] = sub["cas"].abs()

    ymin, ymax = robust_ylim_domain(sub, q=0.95, pad=1.10, floor=0.2)

    for m in MODEL_ORDER:
        sm = sub[sub["model"] == m].copy()
        sm = sm.sort_values("horizon")
        xs = sm["horizon"].to_numpy()
        ys = sm["abs_cas"].to_numpy()

        yplot = np.minimum(ys, ymax)
        ax.plot(
            xs,
            yplot,
            MODEL_STYLE[m],
            color="0.2",
            alpha=0.9,
        )
        ax.scatter(
            xs,
            yplot,
            s=50,
            marker=MODEL_MARK[m],
            color="0.2",
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

        if xs.size:
            ax.text(
                xs[-1] + 0.35,
                yplot[-1],
                MODEL_LABEL[m],
                fontsize=10,
                va="center",
                ha="left",
                color="0.2",
            )

        clipped = ys > ymax
        for xi, yi in zip(xs[clipped], ys[clipped]):
            ax.plot(
                [xi, xi],
                [ymax * 0.98, ymax],
                color="0.2",
                lw=1.0,
            )
            ax.text(
                xi,
                ymax,
                f"{yi:.0f}",
                fontsize=8,
                ha="center",
                va="bottom",
                color="0.2",
            )

    ax.set_title(DOMAIN_LABEL[d], color=DOMAIN_COLOR[d])
    ax.set_xlabel("Horizon h")
    if j == 0:
        ax.set_ylabel(r"$|\mathrm{CAS}|$ (lower is better)")
    ax.set_xticks(HORIZONS)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    ax.spines["bottom"].set_color(DOMAIN_COLOR[d])
    ax.spines["bottom"].set_linewidth(2.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

handles = [
    Line2D(
        [0],
        [0],
        marker=MODEL_MARK[m],
        color="0.2",
        linestyle=MODEL_STYLE[m],
        markersize=7,
        label=MODEL_LABEL[m],
    )
    for m in MODEL_ORDER
]
fig4a.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

p_png = OUTDIR / "figure4a_cas_vs_horizon.png"
p_pdf = OUTDIR / "figure4a_cas_vs_horizon.pdf"
fig4a.savefig(p_png, bbox_inches="tight")
fig4a.savefig(p_pdf, bbox_inches="tight")
plt.close(fig4a)
print(f"[Saved] {p_png}")
print(f"[Saved] {p_pdf}")


# ----------------------------
# Fig 4b: log-scale companion
# ----------------------------
def log1p_fmt(val, _pos=None):
    raw = np.expm1(val)
    if raw >= 1000:
        return f"{raw / 1000:.1f}k"
    if raw >= 100:
        return f"{raw:.0f}"
    if raw >= 10:
        return f"{raw:.1f}"
    return f"{raw:.2f}"


fig4b, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), sharey=False)

for j, d in enumerate(DOMAIN_ORDER):
    ax = axes[j]
    sub = met[met["domain"] == d].copy()
    sub["abs_cas"] = sub["cas"].abs()
    for m in MODEL_ORDER:
        sm = sub[sub["model"] == m].copy()
        sm = sm.sort_values("horizon")
        xs = sm["horizon"].to_numpy()
        ys = np.log1p(sm["abs_cas"].to_numpy())
        ax.plot(
            xs,
            ys,
            MODEL_STYLE[m],
            color="0.2",
            alpha=0.9,
        )
        ax.scatter(
            xs,
            ys,
            s=50,
            marker=MODEL_MARK[m],
            color="0.2",
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        if xs.size:
            ax.text(
                xs[-1] + 0.35,
                ys[-1],
                MODEL_LABEL[m],
                fontsize=10,
                va="center",
                ha="left",
                color="0.2",
            )

    ax.set_title(DOMAIN_LABEL[d], color=DOMAIN_COLOR[d])
    ax.set_xlabel("Horizon h")
    if j == 0:
        ax.set_ylabel(r"$\log(1+|\mathrm{CAS}|)$")
    ax.set_xticks(HORIZONS)
    ax.grid(True)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(log1p_fmt))
    ax.spines["bottom"].set_color(DOMAIN_COLOR[d])
    ax.spines["bottom"].set_linewidth(2.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig4b.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

p_png = OUTDIR / "figure4b_cas_vs_horizon_log.png"
p_pdf = OUTDIR / "figure4b_cas_vs_horizon_log.pdf"
fig4b.savefig(p_png, bbox_inches="tight")
fig4b.savefig(p_pdf, bbox_inches="tight")
plt.close(fig4b)
print(f"[Saved] {p_png}")
print(f"[Saved] {p_pdf}")
