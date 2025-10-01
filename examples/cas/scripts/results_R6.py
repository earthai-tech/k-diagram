"""
R6: Calibration fixes and their effect on CAS

Generates:
  - Fig 9: before/after reliability curves (3 cols = models;
    colored by domain)
  - Table 5: Δcoverage, ΔCRPS_proxy, Δ|CAS| by model×domain
    (plus before/after levels)

Reads (via results_config):
  predictions_wind.csv
  predictions_hydro.csv
  predictions_subsidence.csv

Notes:
  * CRPS is approximated by averaging pinball at τ∈{0.1,0.5,0.9}.
    We call it 'CRPS_proxy'.
  * CAS defaults (as earlier): kernel='gaussian', λ=1.0,
    γ=1.25; bandwidth per-domain.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from results_config import (
    DOMAIN_COLOR,
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    MODEL_LABEL,
    MODEL_ORDER,
    OUTDIR,
    PRED_HYDRO,
    PRED_SUBS,
    PRED_WIND,
    enforce_non_crossing,
)
from sklearn.isotonic import IsotonicRegression

# ----------------------------
# Aesthetics
# ----------------------------
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
# CAS parameters
# ----------------------------
KERNEL_KIND = "gaussian"
LAMBDA = 1.0
GAMMA = 1.25
# steps (hours for wind, days for hydro)
H_REGULAR = {"wind": 6, "hydro": 7}
# days for subsidence (if datetime), else time units of t
H_IRREG_DAYS = 30.0


# ----------------------------
# Helpers
# ----------------------------
def enforce_non_crossing_df(df: pd.DataFrame) -> pd.DataFrame:
    q10, q50, q90 = enforce_non_crossing(
        df["q10"].to_numpy(),
        df["q50"].to_numpy(),
        df["q90"].to_numpy(),
    )
    out = df.copy()
    out["q10"], out["q50"], out["q90"] = q10, q50, q90
    return out


def ensure_time_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "t" in df and df["t"].dtype == object:
        try:
            df["t"] = pd.to_datetime(df["t"])
        except Exception:
            pass
    return df


def reliability_points(
    y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    taus = np.array([0.10, 0.50, 0.90], dtype=float)
    emp = np.array(
        [
            np.mean(y <= q10),
            np.mean(y <= q50),
            np.mean(y <= q90),
        ],
        dtype=float,
    )
    return taus, emp


def pinball_loss(y, q, tau):
    u = y - q
    return ((tau - (u < 0).astype(float)) * u).astype(float)


def crps_proxy(y, q10, q50, q90):
    return (
        pinball_loss(y, q10, 0.10)
        + pinball_loss(y, q50, 0.50)
        + pinball_loss(y, q90, 0.90)
    ) / 3.0


def miss_and_sign(
    y: np.ndarray, q10: np.ndarray, q90: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
    raise ValueError("Unknown kernel kind")


def local_density_regular(
    I: np.ndarray, h: float, kind: str = "gaussian"
) -> np.ndarray:
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
        else:
            raise ValueError("Unknown kernel")
        den = np.sum(w)
        d[i] = np.sum(w * I) / (den if den > 1e-12 else 1.0)
    return np.clip(d, 0.0, 1.0)


def local_cas(
    m: np.ndarray, d: np.ndarray, lam: float = 1.0, gamma: float = 1.25
) -> np.ndarray:
    return m * (1.0 + lam * (d**gamma))


# ----------------------------
# Isotonic recalibration
# ----------------------------
def fit_isotonic_map(x, y, min_pts: int = 8):
    """Monotone ψ: x→y on finite pairs. Safe predictor."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_pts or np.nanstd(x[mask]) == 0.0:

        def predict_safe(z):
            z = np.asarray(z, dtype=float)
            return z

        return predict_safe

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(x[mask], y[mask])

    def predict_safe(z):
        z = np.asarray(z, dtype=float)
        out = z.copy()
        m = np.isfinite(z)
        if m.any():
            out[m] = ir.predict(z[m])
        return out

    return predict_safe


def recalibrate_quantiles_block(df_block: pd.DataFrame) -> pd.DataFrame:
    """Fit ψ10, ψ50, ψ90 on (qτ, y) and map qτ→qτ_c. Enforce
    non-crossing; keep NaNs unchanged."""
    out = df_block.copy()
    q10 = out["q10"].to_numpy(float)
    q50 = out["q50"].to_numpy(float)
    q90 = out["q90"].to_numpy(float)
    yt = out["y"].to_numpy(float)

    f10 = fit_isotonic_map(q10, yt)
    f50 = fit_isotonic_map(q50, yt)
    f90 = fit_isotonic_map(q90, yt)

    q10c = f10(q10)
    q50c = f50(q50)
    q90c = f90(q90)

    q10c = np.where(np.isfinite(q10c), q10c, q10)
    q50c = np.where(np.isfinite(q50c), q50c, q50)
    q90c = np.where(np.isfinite(q90c), q90c, q90)

    q50c = np.maximum(q50c, q10c)
    q90c = np.maximum(q90c, q50c)

    out["q10_c"], out["q50_c"], out["q90_c"] = q10c, q50c, q90c
    return out


# ----------------------------
# Load predictions
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
# Build calibrated copies
# ----------------------------
blocks = []
for (_d, _m), g in preds_all.groupby(["domain", "model"], sort=False):
    blocks.append(recalibrate_quantiles_block(g))
preds_cal = pd.concat(blocks, ignore_index=True)


# ----------------------------------
# Fig 9: reliability before/after
# ----------------------------------
def plot_reliability_r6(df0: pd.DataFrame, df1: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.2), sharey=True)
    taus_nom = np.array([0.10, 0.50, 0.90], dtype=float)

    for j, model in enumerate(MODEL_ORDER):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="0.6", zorder=1)
        for d in DOMAIN_ORDER:
            col = DOMAIN_COLOR[d]
            g0 = df0[(df0["model"] == model) & (df0["domain"] == d)]
            if not g0.empty:
                t0, e0 = reliability_points(
                    g0["y"].to_numpy(),
                    g0["q10"].to_numpy(),
                    g0["q50"].to_numpy(),
                    g0["q90"].to_numpy(),
                )
                ax.plot(
                    t0,
                    e0,
                    ls="--",
                    marker="o",
                    mfc="white",
                    mec=col,
                    color=col,
                    label=f"{DOMAIN_LABEL[d]} (before)",
                    alpha=0.9,
                )

            g1 = df1[(df1["model"] == model) & (df1["domain"] == d)]
            if not g1.empty:
                t1, e1 = reliability_points(
                    g1["y"].to_numpy(),
                    g1["q10_c"].to_numpy(),
                    g1["q50_c"].to_numpy(),
                    g1["q90_c"].to_numpy(),
                )
                ax.plot(
                    t1,
                    e1,
                    ls="-",
                    marker="o",
                    mfc=col,
                    mec="white",
                    color=col,
                    label=f"{DOMAIN_LABEL[d]} (after)",
                    alpha=0.95,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(taus_nom)
        ax.set_yticks(taus_nom)
        ax.set_xlabel("Nominal probability $\\tau$")
        if j == 0:
            ax.set_ylabel("Empirical $\\Pr(Y \\le q_\\tau)$")
        ax.set_title(MODEL_LABEL.get(model, model.upper()))
        ax.grid(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    handles = []
    for d in DOMAIN_ORDER:
        col = DOMAIN_COLOR[d]
        handles.append(
            Line2D(
                [0],
                [0],
                color=col,
                marker="o",
                mfc="white",
                mec=col,
                ls="--",
                label=f"{DOMAIN_LABEL[d]} (before)",
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=col,
                marker="o",
                mfc=col,
                mec="white",
                ls="-",
                label=f"{DOMAIN_LABEL[d]} (after)",
            )
        )
    fig.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    p_png = OUTDIR / "figure9_reliability_before_after.png"
    p_pdf = OUTDIR / "figure9_reliability_before_after.pdf"
    fig.savefig(p_png, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {p_png}\n[Saved] {p_pdf}")


plot_reliability_r6(preds_all, preds_cal)


# ----------------------------
# Table 5: deltas by model×domain
# ----------------------------
def mean_abs_cas_for_block(
    df_block: pd.DataFrame, domain: str, use_calibrated: bool
) -> float:
    """Mean |CAS| for one (domain, model) block."""
    is_irreg = domain == "subsidence"
    qlo = "q10_c" if use_calibrated else "q10"
    qhi = "q90_c" if use_calibrated else "q90"

    acc_sum = 0.0
    acc_n = 0
    if is_irreg:
        for _, g in df_block.groupby(["horizon", "series_id"]):
            g = g.sort_values("t")
            y = g["y"].to_numpy()
            m, s = miss_and_sign(y, g[qlo].to_numpy(), g[qhi].to_numpy())
            I = (s != 0).astype(int)
            d = local_density_irregular(
                g["t"].to_numpy(),
                I,
                h_days=H_IRREG_DAYS,
                kind=KERNEL_KIND,
            )
            c = local_cas(m, d, lam=LAMBDA, gamma=GAMMA)
            acc_sum += np.sum(np.abs(c))
            acc_n += c.size
    else:
        h = H_REGULAR[domain]
        for _, g in df_block.groupby(["horizon", "series_id"]):
            g = g.sort_values("t")
            y = g["y"].to_numpy()
            m, s = miss_and_sign(y, g[qlo].to_numpy(), g[qhi].to_numpy())
            I = (s != 0).astype(int)
            d = local_density_regular(I, h=h, kind=KERNEL_KIND)
            c = local_cas(m, d, lam=LAMBDA, gamma=GAMMA)
            acc_sum += np.sum(np.abs(c))
            acc_n += c.size

    return float(acc_sum / max(acc_n, 1))


rows = []
for d in DOMAIN_ORDER:
    for m in MODEL_ORDER:
        b0 = preds_all[(preds_all["domain"] == d) & (preds_all["model"] == m)]
        if b0.empty:
            continue
        b1 = preds_cal[(preds_cal["domain"] == d) & (preds_cal["model"] == m)]

        cov0 = np.mean((b0["y"] >= b0["q10"]) & (b0["y"] <= b0["q90"]))
        cov1 = np.mean((b1["y"] >= b1["q10_c"]) & (b1["y"] <= b1["q90_c"]))

        crps0 = np.mean(
            crps_proxy(
                b0["y"].to_numpy(),
                b0["q10"].to_numpy(),
                b0["q50"].to_numpy(),
                b0["q90"].to_numpy(),
            )
        )
        crps1 = np.mean(
            crps_proxy(
                b1["y"].to_numpy(),
                b1["q10_c"].to_numpy(),
                b1["q50_c"].to_numpy(),
                b1["q90_c"].to_numpy(),
            )
        )

        ac0 = mean_abs_cas_for_block(b0, domain=d, use_calibrated=False)
        ac1 = mean_abs_cas_for_block(b1, domain=d, use_calibrated=True)

        rows.append(
            {
                "domain": d,
                "model": m,
                "before_coverage": cov0,
                "after_coverage": cov1,
                "delta_coverage": cov1 - cov0,
                "before_crps_proxy": crps0,
                "after_crps_proxy": crps1,
                "delta_crps_proxy": crps1 - crps0,
                "before_abs_cas": ac0,
                "after_abs_cas": ac1,
                "delta_abs_cas": ac1 - ac0,
            }
        )

table5 = (
    pd.DataFrame(rows).sort_values(["domain", "model"]).reset_index(drop=True)
)

csv5 = OUTDIR / "table5_recalibration_deltas.csv"
tex5 = OUTDIR / "table5_recalibration_deltas.tex"
table5.to_csv(csv5, index=False)

with open(tex5, "w", encoding="utf-8") as f:
    f.write(
        table5.rename(
            columns={
                "domain": "domain",
                "model": "model",
                "before_coverage": "before_cov",
                "after_coverage": "after_cov",
                "delta_coverage": "Δcov",
                "before_crps_proxy": "before_CRPSproxy",
                "after_crps_proxy": "after_CRPSproxy",
                "delta_crps_proxy": "ΔCRPSproxy",
                "before_abs_cas": "before_|CAS|",
                "after_abs_cas": "after_|CAS|",
                "delta_abs_cas": "Δ|CAS|",
            }
        ).to_latex(index=False, float_format="%.4g")
    )

print(f"[Saved] {csv5}\n[Saved] {tex5}")
print("\n[Done] R6 outputs written to ./outputs/")
