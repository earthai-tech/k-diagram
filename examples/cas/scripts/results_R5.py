"""
R5: Robustness & sensitivity of CAS

Generates:
  - Fig 8: heatmaps of mean |CAS| vs (λ, γ) for two h per
    domain.
  - Table 4: Kendall's τ stability of model ranks across
    settings.

Inputs (via results_config):
  predictions_wind.csv
  predictions_hydro.csv
  predictions_subsidence.csv

Outputs (under ./outputs):
  figure8_heatmap_wind.(png|pdf)
  figure8_heatmap_hydro.(png|pdf)
  figure8_heatmap_subsidence.(png|pdf)
  table4_kendall_tau_rank_stability.(csv|tex)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from results_config import (
    DOMAIN_COLOR,
    OUTDIR,
    PRED_HYDRO,
    PRED_SUBS,
    PRED_WIND,
    enforce_non_crossing,
)

# ----------------------------
# Config
# ----------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)

# Grids for sensitivity
LAMBDAS = [0.5, 1.0, 1.5]
GAMMAS = [1.0, 1.5, 2.0]
KERNEL = "gaussian"  # or 'triangular'/'epanechnikov'

# Two bandwidths h per domain. For regular series they are in
# steps; for irregular (subsidence) they are in days.
H_GRID = {
    "wind": {"low": 6, "high": 24},
    "hydro": {"low": 7, "high": 21},
    "subsidence": {"low": 30, "high": 90},
}

# Aesthetics
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


# ----------------------------
# Small helpers
# ----------------------------
def ensure_time_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["t"].dtype == object:
        try:
            df["t"] = pd.to_datetime(df["t"])
        except Exception:
            pass
    return df


def enforce_non_crossing_df(df: pd.DataFrame) -> pd.DataFrame:
    q10, q50, q90 = enforce_non_crossing(
        df["q10"].to_numpy(),
        df["q50"].to_numpy(),
        df["q90"].to_numpy(),
    )
    out = df.copy()
    out["q10"], out["q50"], out["q90"] = q10, q50, q90
    return out


def miss_and_sign(
    y: np.ndarray, q10: np.ndarray, q90: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized miss m>=0 and sign in {-1,0,+1}."""
    bw = np.maximum(q90 - q10, 1e-12)
    below = y < q10
    above = y > q90
    m = np.zeros_like(y, dtype=float)
    s = np.zeros_like(y, dtype=int)
    m[below] = (q10[below] - y[below]) / bw[below]
    m[above] = (y[above] - q90[above]) / bw[above]
    s[below] = -1
    s[above] = +1
    return m, s


def kernel_vec(h: float, kind: str = "gaussian") -> np.ndarray:
    """Symmetric 1D kernel for regular grids (unnormalized)."""
    if h <= 0:
        return np.array([1.0])
    if kind == "gaussian":
        R = max(1, int(np.ceil(3 * h)))
        x = np.arange(-R, R + 1, dtype=float)
        return np.exp(-0.5 * (x / h) ** 2)
    if kind == "triangular":
        R = max(1, int(np.ceil(h)))
        x = np.arange(-R, R + 1, dtype=float)
        return np.maximum(0.0, 1.0 - np.abs(x) / h)
    if kind == "epanechnikov":
        R = max(1, int(np.ceil(h)))
        x = np.arange(-R, R + 1, dtype=float)
        z = (x / h) ** 2
        return np.maximum(0.0, 1.0 - z)
    raise ValueError("Unknown kernel kind")


def local_density_regular(
    I: np.ndarray, h: float, kind: str = "gaussian"
) -> np.ndarray:
    """Local exceedance density via convolution."""
    w = kernel_vec(h, kind=kind)
    den = np.convolve(np.ones_like(I, float), w, mode="same")
    num = np.convolve(I.astype(float), w, mode="same")
    d = np.divide(num, np.maximum(den, 1e-12))
    return np.clip(d, 0.0, 1.0)


def to_seconds(arr) -> np.ndarray:
    s = pd.to_datetime(pd.Series(arr), errors="coerce")
    if s.notna().any():
        return (s.view("int64") / 1e9).to_numpy()
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(
        dtype=float
    )


def local_density_irregular(
    times, I: np.ndarray, h_days: float, kind: str = "gaussian"
) -> np.ndarray:
    """Local density for irregular sampling.
    If datetime, h_days is in days; else uses same units as t.
    """
    t = np.asarray(times)
    if np.issubdtype(t.dtype, np.datetime64) or isinstance(
        t[0], (pd.Timestamp,)
    ):
        tsec = to_seconds(t)
        h = h_days * 86400.0
    else:
        tsec = t.astype(float)
        h = float(h_days)

    n = len(I)
    d = np.zeros(n, dtype=float)
    for i in range(n):
        dt = tsec - tsec[i]
        if kind == "gaussian":
            w = np.exp(-0.5 * (dt / h) ** 2)
        elif kind == "triangular":
            w = np.maximum(0.0, 1.0 - np.abs(dt) / h)
        elif kind == "epanechnikov":
            z = (dt / h) ** 2
            w = np.maximum(0.0, 1.0 - z)
        else:
            raise ValueError("Unknown kernel")
        den = np.sum(w)
        num = np.sum(w * I)
        d[i] = num / (den if den > 1e-12 else 1.0)
    return np.clip(d, 0.0, 1.0)


def local_cas(
    m: np.ndarray, d: np.ndarray, lam: float = 1.0, gamma: float = 1.25
) -> np.ndarray:
    return m * (1.0 + lam * (d**gamma))


def kendall_tau(rank_a: dict, rank_b: dict) -> float:
    """Kendall's τ for two total orders (no ties)."""
    labels = list(rank_a.keys())
    n = len(labels)
    pairs = 0
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i = rank_a[labels[i]]
            a_j = rank_a[labels[j]]
            b_i = rank_b[labels[i]]
            b_j = rank_b[labels[j]]
            pairs += 1
            s_a = np.sign(a_i - a_j)
            s_b = np.sign(b_i - b_j)
            if s_a == s_b:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / pairs if pairs else 1.0


# ----------------------------
# Load predictions (once)
# ----------------------------
wind = pd.read_csv(PRED_WIND)
hydro = pd.read_csv(PRED_HYDRO)
subs = pd.read_csv(PRED_SUBS)

wind = enforce_non_crossing_df(ensure_time_col(wind)).assign(domain="wind")
hydro = enforce_non_crossing_df(ensure_time_col(hydro)).assign(domain="hydro")
subs = enforce_non_crossing_df(ensure_time_col(subs)).assign(
    domain="subsidence"
)

preds_all = pd.concat([hydro, wind, subs], ignore_index=True)


# ----------------------------
# Mean |CAS| for one config
# ----------------------------
def compute_mean_abs_cas(
    preds_domain: pd.DataFrame,
    h_val: float,
    kernel: str,
    lam: float,
    gamma: float,
) -> pd.DataFrame:
    """Return mean |CAS| by (model,horizon) and the model
    mean across horizons (mean_over_h)."""
    is_irreg = preds_domain["domain"].iloc[0] == "subsidence"
    rows = []
    gb = ["model", "horizon", "series_id"]
    for (model, horizon, sid), g in preds_domain.groupby(gb):
        g = g.sort_values("t")
        y = g["y"].to_numpy()
        q10 = g["q10"].to_numpy()
        q90 = g["q90"].to_numpy()
        m, sgn = miss_and_sign(y, q10, q90)
        I = (sgn != 0).astype(int)
        if is_irreg:
            d = local_density_irregular(
                g["t"].to_numpy(), I, h_days=h_val, kind=kernel
            )
        else:
            d = local_density_regular(I, h=h_val, kind=kernel)
        c = local_cas(m, d, lam=lam, gamma=gamma)
        rows.append(
            {
                "model": model,
                "horizon": horizon,
                "series_id": sid,
                "cas_contrib_mean": float(np.mean(c)),
            }
        )

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["model", "horizon"], as_index=False)["cas_contrib_mean"]
        .mean()
        .rename(columns={"cas_contrib_mean": "mean_abs_cas"})
    )
    overall = (
        agg.groupby("model")["mean_abs_cas"]
        .mean()
        .rename("mean_over_h")
        .reset_index()
    )
    return agg.merge(overall, on="model", how="left")


# ----------------------------
# Sweep the grid
# ----------------------------
records = []
for domain, df_dom in preds_all.groupby("domain"):
    h_low = H_GRID[domain]["low"]
    h_high = H_GRID[domain]["high"]
    for h_label, h_val in [("low", h_low), ("high", h_high)]:
        for lam in LAMBDAS:
            for gam in GAMMAS:
                agg = compute_mean_abs_cas(
                    df_dom,
                    h_val=h_val,
                    kernel=KERNEL,
                    lam=lam,
                    gamma=gam,
                )
                for _, r in agg.iterrows():
                    records.append(
                        {
                            "domain": domain,
                            "h_label": h_label,
                            "h_value": h_val,
                            "kernel": KERNEL,
                            "lambda": lam,
                            "gamma": gam,
                            "model": r["model"],
                            "horizon": r["horizon"],
                            "mean_abs_cas": r["mean_abs_cas"],
                            "mean_over_h": r["mean_over_h"],
                        }
                    )

sens = pd.DataFrame(records)


# ----------------------------
# Figure 8 — heatmaps
# ----------------------------
def plot_heatmaps_for_domain(sens_df: pd.DataFrame, domain: str):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    for j, h_label in enumerate(["low", "high"]):
        ax = axes[j]
        sub = sens_df[
            (sens_df["domain"] == domain) & (sens_df["h_label"] == h_label)
        ]
        table = (
            sub.groupby(["lambda", "gamma"])["mean_abs_cas"]
            .mean()
            .unstack("gamma")
            .reindex(index=LAMBDAS, columns=GAMMAS)
        )
        im = ax.imshow(table.values, origin="lower", cmap="Greys")
        ttl = (
            f"{domain.capitalize()} — h={h_label} "
            f"({H_GRID[domain][h_label]})\n"
            f"kernel={KERNEL}"
        )
        ax.set_title(ttl, color=DOMAIN_COLOR[domain])
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\lambda$")
        ax.set_xticks(np.arange(len(GAMMAS)))
        ax.set_xticklabels([f"{g:g}" for g in GAMMAS])
        ax.set_yticks(np.arange(len(LAMBDAS)))
        ax.set_yticklabels([f"{l:g}" for l in LAMBDAS])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("mean |CAS|", rotation=270, labelpad=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Sensitivity of mean |CAS| to (λ, γ)", y=1.03, fontsize=14)
    png = OUTDIR / f"figure8_heatmap_{domain}.png"
    pdf = OUTDIR / f"figure8_heatmap_{domain}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png}\n[Saved] {pdf}")


for d in ["wind", "hydro", "subsidence"]:
    plot_heatmaps_for_domain(sens, d)

# ----------------------------
# Table 4 — Kendall's τ
# ----------------------------
rows_tau = []
for domain in ["wind", "hydro", "subsidence"]:
    base_h = "low"
    base = (
        sens[
            (sens["domain"] == domain)
            & (sens["h_label"] == base_h)
            & (sens["lambda"] == 1.0)
            & (sens["gamma"] == 1.0)
        ]
        .groupby("model")["mean_over_h"]
        .mean()
        .sort_values(ascending=True)
    )
    base_rank = {m: i + 1 for i, m in enumerate(base.index)}
    taus = []
    grp = sens[sens["domain"] == domain]
    for (_h_label, _lam, _gam), g in grp.groupby(
        ["h_label", "lambda", "gamma"]
    ):
        ranks = (
            g.groupby("model")["mean_over_h"]
            .mean()
            .sort_values(ascending=True)
        )
        if set(ranks.index) != set(base_rank.keys()):
            continue
        r = {m: i + 1 for i, m in enumerate(ranks.index)}
        taus.append(kendall_tau(base_rank, r))

    if not taus:
        continue

    rows_tau.append(
        {
            "domain": domain,
            "kernel": KERNEL,
            "h_baseline": f"{base_h} ({H_GRID[domain][base_h]})",
            "tau_mean": float(np.mean(taus)),
            "tau_median": float(np.median(taus)),
            "tau_min": float(np.min(taus)),
            "n_settings_compared": int(len(taus)),
        }
    )

table4 = pd.DataFrame(rows_tau)
csv_t4 = OUTDIR / "table4_kendall_tau_rank_stability.csv"
tex_t4 = OUTDIR / "table4_kendall_tau_rank_stability.tex"
table4.to_csv(csv_t4, index=False)
with open(tex_t4, "w", encoding="utf-8") as f:
    f.write(table4.to_latex(index=False, float_format="%.4g"))
print(f"[Saved] {csv_t4}\n[Saved] {tex_t4}")

print("\n[Done] R5 robustness assets saved in ./outputs/")
