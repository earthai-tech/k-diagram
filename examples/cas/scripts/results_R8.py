# %%
"""
R8 (operational): Practical impact — burst-aware decision metrics (ex-ante)

Changes vs diagnostic version:
- Alerts use a FORECAST-ONLY risk index R_t at time t, using current
  quantiles and *past* exceedance history only.
- A burst is "caught" if an alert fires within k steps of its start.

Outputs
-------
- outputs/figure10_burst_roc_operational.{png,pdf}
- outputs/table_r8_operational_impact.csv

Inputs
------
- modeling_results_ok/predictions_{wind,hydro,subsidence}.csv
"""

# from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from results_config import (
    DOMAIN_ORDER,
    MODEL_ORDER,
    OUTDIR,
    PRED_HYDRO,
    PRED_SUBS,
    PRED_WIND,
)

# ----------------------------
# Paths / style
# ----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# PRED_WIND = BASE_DIR / "modeling_results_ok" / "predictions_wind.csv"
# PRED_HYDRO = BASE_DIR / "modeling_results_ok" / "predictions_hydro.csv"
# PRED_SUBS = BASE_DIR / "modeling_results_ok" / "predictions_subsidence.csv"
# OUTDIR = BASE_DIR / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

DOMAIN_COLOR = {
    "hydro": "#0072B2",
    "wind": "#E69F00",
    "subsidence": "#009E73",
}
MODEL_MARKER = {"qar": "o", "qgbm": "s", "xtft": "^"}
# MODEL_ORDER = ["qar", "qgbm", "xtft"]
# DOMAIN_ORDER = ["hydro", "wind", "subsidence"]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 16,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
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
# CAS ingredients (reused)
# ----------------------------
KERNEL_KIND = "gaussian"
LAMBDA = 1.0
GAMMA = 1.25

# density bandwidths (regular in steps, irregular in days)
H_REGULAR = {"wind": 6, "hydro": 7}
H_IRREG_DAYS = 30.0  # subsidence window in days

# detection window (early-warning) and long-burst threshold
K_DET = {"hydro": 2, "wind": 2, "subsidence": 1}
L_MIN = {"hydro": 3, "wind": 5, "subsidence": 2}


# ----------------------------
# Utilities
# ----------------------------
def enforce_non_crossing(df: pd.DataFrame) -> pd.DataFrame:
    q10 = df["q10"].to_numpy()
    q50 = np.maximum(df["q50"].to_numpy(), q10)
    q90 = np.maximum(df["q90"].to_numpy(), q50)
    out = df.copy()
    out["q10"], out["q50"], out["q90"] = q10, q50, q90
    return out


def _infer_series_id(df):
    for c in ["series_id", "series", "id", "zone", "station"]:
        if c in df.columns:
            return c
    raise KeyError(
        "Need a series identifier column (series_id/series/id/zone/station)."
    )


def kernel_vec(h, kind="gaussian"):
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
    raise ValueError("Unknown kernel")


def causal_density_regular(I: np.ndarray, h: float, kind="gaussian"):
    """
    Causal exceedance density d_{t-1}: only past indicators contribute.
    Returns array aligned with I (d_prev for each t).
    """
    w_full = kernel_vec(h, kind=kind)
    R = len(w_full) // 2  # center index (lag 0)
    w_past = w_full[:R][::-1]  # weights for lags 1..R
    n, r = len(I), len(w_past)
    num = np.zeros(n, dtype=float)
    den = np.zeros(n, dtype=float)
    for t in range(n):
        m = min(t, r)
        if m == 0:
            num[t] = 0.0
            den[t] = 1.0
        else:
            x = I[t - m : t].astype(float)
            ww = w_past[:m]
            num[t] = np.dot(ww, x[::-1])
            den[t] = np.sum(ww)
    dprev = np.divide(num, np.maximum(den, 1e-12))
    return np.clip(dprev, 0.0, 1.0)


def to_seconds(arr) -> np.ndarray:
    s = pd.to_datetime(pd.Series(arr), errors="coerce")
    if s.notna().any():
        return (s.view("int64") / 1e9).to_numpy()
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(float)


def causal_density_irregular(
    times, I: np.ndarray, h_days: float, kind="gaussian"
):
    """
    Causal density using only t' < t. times can be datetime or numeric.
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
    dprev = np.zeros(n, dtype=float)
    for i in range(n):
        dt = tsec[:i] - tsec[i]
        if dt.size == 0:
            dprev[i] = 0.0
            continue
        if kind == "gaussian":
            w = np.exp(-0.5 * (dt / h) ** 2)
        elif kind == "triangular":
            w = np.maximum(0.0, 1.0 - np.abs(dt) / h)
        else:
            raise ValueError("Unknown kernel")
        num = np.sum(w * I[:i].astype(float))
        den = np.sum(w) if np.sum(w) > 1e-12 else 1.0
        dprev[i] = num / den
    return np.clip(dprev, 0.0, 1.0)


def exceed_indicator(y, q10, q90):
    return ((y < q10) | (y > q90)).astype(int)


def skew_toward_edge(q10, q50, q90, eps=1e-12):
    """
    Forecast-only 'pressure' toward an edge in [q10, q90].
    psi in [0,1], 0 when centered, ->1 near either bound.
    """
    bw = np.maximum(q90 - q10, eps)
    left = q50 - q10
    rght = q90 - q50
    psi = 1.0 - 2.0 * np.minimum(left, rght) / bw
    return np.clip(psi, 0.0, 1.0)


def risk_index_ex_ante(y, q10, q50, q90, domain, times=None):
    """
    Compute (R_t, I_t) for one series block.
    R_t = (1 + λ * d_{t-1}^γ) * psi_t, with d_{t-1} using past only.
    """
    y = np.asarray(y, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)

    I = exceed_indicator(y, q10, q90)

    if domain == "subsidence":
        dprev = causal_density_irregular(
            times, I, h_days=H_IRREG_DAYS, kind=KERNEL_KIND
        )
    else:
        dprev = causal_density_regular(
            I, h=H_REGULAR[domain], kind=KERNEL_KIND
        )

    psi = skew_toward_edge(q10, q50, q90)
    R = (1.0 + LAMBDA * (dprev**GAMMA)) * psi
    return R, I


def run_lengths(I: np.ndarray):
    """List of (start, end, length) for runs of ones in I."""
    out = []
    n = len(I)
    i = 0
    while i < n:
        if I[i] == 1:
            j = i
            while j + 1 < n and I[j + 1] == 1:
                j += 1
            out.append((i, j, j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


# ----------------------------
# Load predictions
# ----------------------------
def load_predictions():
    wind = enforce_non_crossing(pd.read_csv(PRED_WIND)).assign(domain="wind")
    hydro = enforce_non_crossing(pd.read_csv(PRED_HYDRO)).assign(
        domain="hydro"
    )
    subs = enforce_non_crossing(pd.read_csv(PRED_SUBS)).assign(
        domain="subsidence"
    )
    if "t" in subs.columns and subs["t"].dtype == object:
        subs["t"] = pd.to_datetime(subs["t"], errors="coerce")
    return pd.concat([hydro, wind, subs], ignore_index=True)


# ----------------------------
# Evaluation (ROC-like in burst space)
# ----------------------------
def _sort_block(g: pd.DataFrame) -> pd.DataFrame:
    if "t" in g.columns:
        return g.sort_values("t")
    # fallback: keep index order if no explicit time
    return g.sort_index()


def eval_block_operational(g: pd.DataFrame, domain: str, k_det: int):
    g = _sort_block(g)
    y = g["y"].to_numpy(float)
    q10 = g["q10"].to_numpy(float)
    q50 = g["q50"].to_numpy(float)
    q90 = g["q90"].to_numpy(float)
    mask = (
        np.isfinite(y)
        & np.isfinite(q10)
        & np.isfinite(q50)
        & np.isfinite(q90)
    )
    y, q10, q50, q90 = y[mask], q10[mask], q50[mask], q90[mask]
    if y.size == 0:
        return None

    times = (
        g.loc[mask, "t"].to_numpy()
        if ("t" in g and domain == "subsidence")
        else None
    )
    R, I = risk_index_ex_ante(y, q10, q50, q90, domain, times=times)

    # true bursts & long bursts
    runs = run_lengths(I)
    long_runs = [(i0, i1, L) for (i0, i1, L) in runs if L >= L_MIN[domain]]
    nonburst_time = int((I == 0).sum())

    return {
        "R": R,
        "I": I,
        "long_runs": long_runs,
        "nonburst_time": nonburst_time,
    }


def aggregate_curves(blocks, theta_grid, domain):
    K = len(theta_grid)
    TPB = np.zeros(K)  # true bursts caught
    FNB = np.zeros(K)  # missed bursts
    FP = np.zeros(K)  # non-burst alerts
    TN = np.zeros(K)  # non-burst quiet time
    NB = np.zeros(K)  # number of long bursts
    k_det = K_DET[domain]

    def burst_detected(pred, i0, i1):
        stop = min(i0 + k_det - 1, i1)
        return pred[i0 : stop + 1].any()

    for b in blocks:
        R, I = b["R"], b["I"]
        long_runs = b["long_runs"]
        nonburst_time = b["nonburst_time"]
        NB_block = len(long_runs)

        for k, th in enumerate(theta_grid):
            pred = (R >= th).astype(int)

            # burst recall (within early-warning window)
            tp = 0
            for i0, i1, _L in long_runs:
                if burst_detected(pred, i0, i1):
                    tp += 1
            TPB[k] += tp
            FNB[k] += NB_block - tp
            NB[k] += NB_block

            # false-burst time (alerts outside ANY exceedance)
            outside = np.ones_like(I, dtype=bool)
            for i0, i1, _ in run_lengths(I):
                outside[i0 : i1 + 1] = False
            fp = int(pred[outside].sum())
            FP[k] += fp
            TN[k] += nonburst_time - fp

    recall = np.divide(TPB, np.maximum(NB, 1e-12))
    fbr = np.divide(FP, np.maximum(FP + TN, 1e-12))
    return fbr, recall


def _unique_preserve(arr):
    """Unique while preserving order."""
    out = []
    seen = set()
    for x in arr:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return np.asarray(out)


def summarize_at_fbr(blocks, theta_grid, fbr, recall, targets, domain):
    rows = []
    k_det = K_DET[domain]

    for fbr_star in targets:
        # choose the smallest threshold with fbr <= target
        idx = np.where(fbr <= fbr_star)[0]
        if idx.size == 0:
            k_sel = len(theta_grid) - 1
        else:
            k_sel = idx[0]
        th = theta_grid[k_sel]
        rec = float(recall[k_sel])

        # time-to-alert & expected missed length
        tta_list, miss_list = [], []
        for b in blocks:
            R, _, long_runs = b["R"], b["I"], b["long_runs"]
            if not long_runs:
                continue
            pred = (R >= th).astype(int)
            for i0, i1, L in long_runs:
                stop = min(i0 + k_det - 1, i1)
                idx_fire = np.where(pred[i0 : stop + 1] == 1)[0]
                if idx_fire.size:
                    tta_list.append(int(idx_fire[0]))
                else:
                    miss_list.append(int(L))
        med_tta = float(np.median(tta_list)) if tta_list else np.nan
        exp_missed = float(np.mean(miss_list)) if miss_list else 0.0

        rows.append((fbr_star, float(fbr[k_sel]), rec, med_tta, exp_missed))
    return rows


# ----------------------------
# Main
# ----------------------------
def main():
    df = load_predictions()
    series_col = _infer_series_id(df)
    keep = {
        "domain",
        "model",
        "horizon",
        series_col,
        "t",
        "y",
        "q10",
        "q50",
        "q90",
    }
    df = df[[c for c in df.columns if c in keep]].dropna(
        subset=["y", "q10", "q50", "q90"]
    )

    curves = {}
    summaries = []

    for domain in DOMAIN_ORDER:
        df_dom = df[df["domain"] == domain]
        for model in MODEL_ORDER:
            df_m = df_dom[df_dom["model"] == model]

            # Build series blocks
            blocks = []
            for (_h, _sid), g in df_m.groupby(["horizon", series_col]):
                b = eval_block_operational(g, domain, K_DET[domain])
                if b is not None:
                    blocks.append(b)
            if len(blocks) == 0:
                continue

            # pooled thresholds from R distribution
            R_all = np.concatenate([b["R"] for b in blocks])
            R_all = R_all[np.isfinite(R_all)]
            q = np.linspace(0.99, 0.01, 60)  # high->low thresholds
            thetas = np.quantile(R_all, q)
            thetas = _unique_preserve(thetas)  # keep descending order

            fbr, rec = aggregate_curves(blocks, thetas, domain)
            curves.setdefault(domain, {})[model] = (fbr, rec)

            # summaries at target FBRs
            rows = summarize_at_fbr(
                blocks, thetas, fbr, rec, targets=(0.05, 0.10), domain=domain
            )
            for fbr_star, fbr_ach, recall_val, med_tta, exp_missed in rows:
                summaries.append(
                    {
                        "domain": domain,
                        "model": model,
                        "target_FBR": fbr_star,
                        "achieved_FBR": fbr_ach,
                        "recall": recall_val,
                        "median_time_to_alert_steps": med_tta,
                        "expected_missed_length_steps": exp_missed,
                    }
                )

    # Save table
    tab = pd.DataFrame(summaries)
    tab_path = OUTDIR / "table_r8_operational_impact.csv"
    tab.to_csv(tab_path, index=False)
    print(f"[Saved] {tab_path}")

    # Figure 10 (operational)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, domain in zip(axes, DOMAIN_ORDER):
        if domain not in curves:
            ax.axis("off")
            continue
        color = DOMAIN_COLOR[domain]
        for m in MODEL_ORDER:
            if m not in curves[domain]:
                continue
            fbr, rec = curves[domain][m]
            ax.plot(
                fbr,
                rec,
                label=m.upper(),
                color=color,
                alpha=0.9,
                linestyle={"qar": "-", "qgbm": "--", "xtft": ":"}[m],
                marker=MODEL_MARKER[m],
                markevery=6,
                mec="k",
                mew=0.6,
            )
        ax.set_xlim(0, 0.30)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_xlabel("False-burst rate (time)")
        ax.set_ylabel(f"Burst recall (first {K_DET[domain]} steps)")
        ax.set_title(domain.capitalize(), color=color)
        ax.legend(title="Model", frameon=False, loc="upper left")

    fig.suptitle(
        "Figure 10 — ROC-like curves in burst space (operational, ex-ante)",
        y=1.02,
        fontsize=16,
    )
    png = OUTDIR / "figure10_burst_roc_operational.png"
    pdf = OUTDIR / "figure10_burst_roc_operational.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png}\n[Saved] {pdf}")


main()
