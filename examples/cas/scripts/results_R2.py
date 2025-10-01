"""
R2: Cluster-Aware Severity (CAS)  figures & table

Generates
  * Table 2: CAS leaderboard by (domain, model)
  * Fig 2: Coverage vs |CAS| (panel per horizon)
  * Fig 2a: Coverage vs |CAS| with robust y-limits
  * Fig 2b: Coverage vs log1p(|CAS|)
  * Fig 3: Severity calendars per domain (with insets)

Inputs (via results_config):
  metrics_all_domains.csv
  predictions_wind.csv
  predictions_hydro.csv
  predictions_subsidence.csv

Outputs (under data/cas/outputs/):
  table2_cas_leaderboard.(csv|tex)
  figure2_tradeoff_coverage_vs_abscas.(png|pdf)
  figure2a_tradeoff_coverage_vs_abscas_robust.(png|pdf)
  figure2b_tradeoff_coverage_vs_abscas_log.(png|pdf)
  figure3_severity_calendar_<domain>.(png|pdf)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from results_config import (
    DOMAIN_COLOR,
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    METRICS_PATH,
    MODEL_LABEL,
    MODEL_MARK,
    MODEL_ORDER,
    NOMINAL_COVERAGE,
    OUTDIR,
    PRED_HYDRO,
    PRED_SUBS,
    PRED_WIND,
    RASTER_CMAP,
    RASTER_NORM,
    label_points,
    load_preds,
)

# -------------------------
# Local configuration
# -------------------------
HORIZONS = [1, 3, 7, 14, 28]
TOP_SERIES = 24  # top S series by exceed rate


# -------------------------
# Helpers
# -------------------------
def with_exceed(df: pd.DataFrame) -> pd.DataFrame:
    """Add {-1,0,+1} exceedance sign."""
    out = df.copy()
    out["exceed"] = np.where(
        out["y"] < out["q10"],
        -1,
        np.where(out["y"] > out["q90"], 1, 0),
    )
    return out


def run_lengths(sig: np.ndarray) -> np.ndarray:
    """Run lengths for nonzero segments in sign array."""
    if sig.size == 0:
        return np.array([], dtype=int)
    runs, cur = [], 0
    for v in sig:
        if v != 0:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    return np.asarray(runs, int)


def series_calendar_block(df: pd.DataFrame, model: str, S: int):
    """
    Build an (S×T) raster for one model:
      rows = top S series by exceed rate,
      cols = sorted time stamps, vals in {-1,0,1}.
    """
    sub = df[df["model"] == model].copy()
    rates = (
        sub.assign(exc=(sub["exceed"] != 0).astype(int))
        .groupby("series_id")["exc"]
        .mean()
        .sort_values(ascending=False)
    )
    keep = rates.index[:S].tolist()
    sub = sub[sub["series_id"].isin(keep)]
    ts = np.sort(sub["t"].unique())
    ras = np.zeros((len(keep), len(ts)), dtype=int)
    for i, sid in enumerate(keep):
        row = sub[sub["series_id"] == sid]
        mp = dict(zip(row["t"].to_numpy(), row["exceed"].to_numpy()))
        ras[i, :] = [mp.get(tt, 0) for tt in ts]
    return ras, ts, np.array(keep)


# -------------------------
# Load data
# -------------------------
metrics = pd.read_csv(METRICS_PATH)

preds_wind = with_exceed(load_preds(PRED_WIND, "wind"))
preds_hydro = with_exceed(load_preds(PRED_HYDRO, "hydro"))
preds_subs = with_exceed(load_preds(PRED_SUBS, "subsidence"))

preds = pd.concat(
    [preds_hydro, preds_wind, preds_subs],
    ignore_index=True,
)

# -------------------------
# Table 2: CAS leaderboard
# -------------------------
tbl2 = (
    metrics.groupby(["domain", "model"], as_index=False)
    .agg(
        mean_abs_cas=("cas", lambda s: float(np.mean(np.abs(s)))),
        mean_cas=("cas", "mean"),
    )
    .sort_values(["domain", "mean_abs_cas"])
)

t2_csv = OUTDIR / "table2_cas_leaderboard.csv"
t2_tex = OUTDIR / "table2_cas_leaderboard.tex"

tbl2.to_csv(t2_csv, index=False)
with open(t2_tex, "w", encoding="utf-8") as f:
    f.write(tbl2.to_latex(index=False, float_format="%.4g"))

print(f"[Saved] {t2_csv}")
print(f"[Saved] {t2_tex}")

# -------------------------
# Figure 2: Coverage vs |CAS|
# -------------------------
fig2, axes2 = plt.subplots(
    1,
    len(HORIZONS),
    figsize=(3.8 * len(HORIZONS), 4.6),
    sharey=True,
)

for j, h in enumerate(HORIZONS):
    ax = axes2[j]
    sub = metrics[metrics["horizon"] == h].copy()
    sub["abs_cas"] = np.abs(sub["cas"])
    for d in DOMAIN_ORDER:
        sd = sub[sub["domain"] == d]
        for m in MODEL_ORDER:
            sm = sd[sd["model"] == m]
            if sm.empty:
                continue
            x = sm["coverage"]
            y = sm["abs_cas"]
            ax.scatter(
                x,
                y,
                s=46,
                color=DOMAIN_COLOR[d],
                marker=MODEL_MARK[m],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            label_points(
                ax,
                x,
                y,
                [MODEL_LABEL[m]] * len(sm),
                dx=0.004,
                dy=0.004,
            )
    ax.axvline(NOMINAL_COVERAGE, ls="--", lw=1.0, color="0.6")
    ax.set_title(f"h = {h}")
    ax.set_xlabel("Coverage")
    if j == 0:
        ax.set_ylabel(r"$|\mathrm{CAS}|$ (lower is better)")
    ax.grid(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

dom_handles = [
    Line2D(
        [0],
        [0],
        color=DOMAIN_COLOR[d],
        marker="o",
        lw=0,
        markersize=8,
        label=DOMAIN_LABEL[d],
    )
    for d in DOMAIN_ORDER
]
mod_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        marker=MODEL_MARK[m],
        lw=0,
        markersize=7,
        label=MODEL_LABEL[m],
    )
    for m in MODEL_ORDER
]

fig2.legend(
    handles=dom_handles + mod_handles,
    ncol=6,
    loc="lower center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.03),
)

f2_png = OUTDIR / "figure2_tradeoff_coverage_vs_abscas.png"
f2_pdf = OUTDIR / "figure2_tradeoff_coverage_vs_abscas.pdf"

fig2.savefig(f2_png, bbox_inches="tight")
fig2.savefig(f2_pdf, bbox_inches="tight")
plt.close(fig2)

print(f"[Saved] {f2_png}")
print(f"[Saved] {f2_pdf}")


# -------------------------
# Figure 3: Severity calendars
# -------------------------
def figure3_for_domain(preds_dom: pd.DataFrame, domain: str):
    k = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 6.0), sharey=True)

    for j, m in enumerate(MODEL_ORDER):
        ax = axes[j]
        ras, ts, sid = series_calendar_block(preds_dom, m, TOP_SERIES)
        if ras.size == 0:
            ax.set_visible(False)
            continue

        ax.imshow(
            ras,
            aspect="auto",
            cmap=RASTER_CMAP,
            norm=RASTER_NORM,
            interpolation="nearest",
            origin="lower",
        )
        ax.set_title(MODEL_LABEL[m])
        ax.set_xlabel("Time index")
        if j == 0:
            ax.set_ylabel(f"Top {TOP_SERIES} series by exceed rate")
        ax.set_xticks(np.linspace(0, ras.shape[1] - 1, 5))
        ax.set_xticklabels(
            [f"{int(v)}" for v in np.linspace(0, ras.shape[1] - 1, 5)]
        )
        ax.set_yticks(np.linspace(0, ras.shape[0] - 1, 5))
        ax.set_yticklabels(
            [f"{int(v)}" for v in np.linspace(1, ras.shape[0], 5)]
        )
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # inset: run-length histogram over selected rows
        all_runs = [run_lengths(ras[i, :]) for i in range(ras.shape[0])]
        runs = (
            np.concatenate([r for r in all_runs if r.size > 0])
            if any(r.size for r in all_runs)
            else np.array([], dtype=int)
        )

        if runs.size:
            axins = inset_axes(
                ax,
                width="36%",
                height="36%",
                loc="upper right",
                borderpad=0.8,
            )
            bins = np.arange(1, max(10, runs.max()) + 1)
            axins.hist(
                runs,
                bins=bins,
                color=DOMAIN_COLOR[domain],
                edgecolor="black",
                linewidth=0.5,
            )
            axins.set_title("Run length", fontsize=9)
            axins.tick_params(axis="both", labelsize=8)
            for sp in ("top", "right"):
                axins.spines[sp].set_visible(False)

    # shared colorbar
    cax = fig.add_axes([0.33, -0.045, 0.34, 0.025])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=RASTER_NORM, cmap=RASTER_CMAP),
        cax=cax,
        orientation="horizontal",
        ticks=[-1, 0, 1],
    )
    cb.ax.set_xticklabels(["below", "inside", "above"])
    cb.ax.tick_params(labelsize=10)

    # domain accent on bottom spines
    for ax in axes:
        ax.spines["bottom"].set_color(DOMAIN_COLOR[domain])
        ax.spines["bottom"].set_linewidth(2.0)

    fig.suptitle(
        f"Severity calendar — {DOMAIN_LABEL[domain]}",
        y=1.02,
        fontsize=14,
    )

    png = OUTDIR / f"figure3_severity_calendar_{domain}.png"
    pdf = OUTDIR / f"figure3_severity_calendar_{domain}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png}")
    print(f"[Saved] {pdf}")


for d in DOMAIN_ORDER:
    figure3_for_domain(preds[preds["domain"] == d].copy(), d)

print("[Info] R2 main assets written to outputs/")

# ============================================================
# Figure 2a: Coverage vs |CAS| with robust per-panel y-limits
# ============================================================


def robust_ylim_for_horizon(
    df_h: pd.DataFrame,
    q: float = 0.95,
    pad: float = 1.10,
    floor: float = 0.25,
):
    """Top = max(floor, q-quantile of |CAS|) * pad."""
    y = np.abs(df_h["cas"].to_numpy())
    top = np.quantile(y, q) if y.size else floor
    top = max(top, floor) * pad
    return 0.0, float(top)


fig2a, axes2a = plt.subplots(
    1,
    len(HORIZONS),
    figsize=(3.8 * len(HORIZONS), 4.6),
    sharey=False,
)

for j, h in enumerate(HORIZONS):
    ax = axes2a[j]
    sub = metrics[metrics["horizon"] == h].copy()
    sub["abs_cas"] = np.abs(sub["cas"])
    ymin, ymax = robust_ylim_for_horizon(sub, q=0.95, pad=1.10, floor=0.25)

    # scatter points
    for d in DOMAIN_ORDER:
        sd = sub[sub["domain"] == d]
        for m in MODEL_ORDER:
            sm = sd[sd["model"] == m]
            if sm.empty:
                continue
            x = sm["coverage"].to_numpy()
            y = sm["abs_cas"].to_numpy()

            in_rng = y <= ymax
            ax.scatter(
                x[in_rng],
                y[in_rng],
                s=46,
                color=DOMAIN_COLOR[d],
                marker=MODEL_MARK[m],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            label_points(
                ax,
                x[in_rng],
                y[in_rng],
                [MODEL_LABEL[m]] * int(in_rng.sum()),
                dx=0.004,
                dy=0.004,
            )

            # outliers clipped at top
            if np.any(~in_rng):
                x_o = x[~in_rng]
                y_true = y[~in_rng]
                y_cap = np.full_like(x_o, ymax)
                ax.scatter(
                    x_o,
                    y_cap,
                    s=70,
                    facecolors="none",
                    edgecolors=DOMAIN_COLOR[d],
                    marker="^",
                    linewidth=1.5,
                    zorder=4,
                )
                for xi, yt, yc in zip(x_o, y_true, y_cap):
                    ax.text(
                        xi,
                        yc,
                        f"{yt:.0f}",
                        fontsize=8,
                        ha="center",
                        va="bottom",
                        color=DOMAIN_COLOR[d],
                        zorder=5,
                    )

    ax.axvline(NOMINAL_COVERAGE, ls="--", lw=1.0, color="0.6")
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"h = {h}")
    ax.set_xlabel("Coverage")
    if j == 0:
        ax.set_ylabel(r"$|\mathrm{CAS}|$ (lower is better)")
    ax.grid(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig2a.legend(
    handles=dom_handles + mod_handles,
    ncol=6,
    loc="lower center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.03),
)

f2a_png = OUTDIR / "figure2a_tradeoff_coverage_vs_abscas_robust.png"
f2a_pdf = OUTDIR / "figure2a_tradeoff_coverage_vs_abscas_robust.pdf"

fig2a.savefig(f2a_png, bbox_inches="tight")
fig2a.savefig(f2a_pdf, bbox_inches="tight")
plt.close(fig2a)

print(f"[Saved] {f2a_png}")
print(f"[Saved] {f2a_pdf}")


# ============================================================
# Figure 2b: same plot on log scale (log1p), ticks formatted
# ============================================================
def log1p_formatter(val, _pos=None):
    """Format tick of log1p(|CAS|) on original scale."""
    raw = np.expm1(val)
    if raw >= 1000:
        return f"{raw / 1000:.1f}k"
    if raw >= 100:
        return f"{raw:.0f}"
    if raw >= 10:
        return f"{raw:.1f}"
    return f"{raw:.2f}"


fig2b, axes2b = plt.subplots(
    1,
    len(HORIZONS),
    figsize=(3.8 * len(HORIZONS), 4.6),
    sharey=False,
)

for j, h in enumerate(HORIZONS):
    ax = axes2b[j]
    sub = metrics[metrics["horizon"] == h].copy()
    sub["abs_cas"] = np.abs(sub["cas"])
    sub["logy"] = np.log1p(sub["abs_cas"])

    for d in DOMAIN_ORDER:
        sd = sub[sub["domain"] == d]
        for m in MODEL_ORDER:
            sm = sd[sd["model"] == m]
            if sm.empty:
                continue
            ax.scatter(
                sm["coverage"],
                sm["logy"],
                s=46,
                color=DOMAIN_COLOR[d],
                marker=MODEL_MARK[m],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            label_points(
                ax,
                sm["coverage"],
                sm["logy"],
                [MODEL_LABEL[m]] * len(sm),
                dx=0.004,
                dy=0.004,
            )

    ax.axvline(NOMINAL_COVERAGE, ls="--", lw=1.0, color="0.6")
    ax.set_title(f"h = {h}")
    ax.set_xlabel("Coverage")
    if j == 0:
        ax.set_ylabel(r"$\log(1+|\mathrm{CAS}|)$")
    ax.grid(True)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(log1p_formatter))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig2b.legend(
    handles=dom_handles + mod_handles,
    ncol=6,
    loc="lower center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.03),
)

f2b_png = OUTDIR / "figure2b_tradeoff_coverage_vs_abscas_log.png"
f2b_pdf = OUTDIR / "figure2b_tradeoff_coverage_vs_abscas_log.pdf"

fig2b.savefig(f2b_png, bbox_inches="tight")
fig2b.savefig(f2b_pdf, bbox_inches="tight")
plt.close(fig2b)

print(f"[Saved] {f2b_png}")
print(f"[Saved] {f2b_pdf}")
print("[Done] R2 extended assets written to outputs/")
