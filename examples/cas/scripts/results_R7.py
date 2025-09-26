"""
R7: Statistical testing & uncertainty

Outputs
-------
- outputs/table6_dm_tests.{csv,tex}
  DM tests on CRPS_proxy and Winkler per horizon
  (Holm-adjusted p-values).
- outputs/table7_mcs_cas.{csv,tex}
  MCS membership (alpha=0.10) for mean |CAS| by domain × horizon.

Notes
-----
* CRPS_proxy = mean pinball at τ∈{0.1,0.5,0.9}.
* DM uses Newey–West (Bartlett) HAC with lag L=max(1, h-1).
* MCS on series-level mean |CAS|; resample series (B=1000).
"""

import itertools as it
from math import sqrt

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
from scipy.stats import norm

# ----------------------------
# Paths / config
# ----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# PRED_WIND = BASE_DIR / "modeling_results_ok" / "predictions_wind.csv"
# PRED_HYDRO = BASE_DIR / "modeling_results_ok" / "predictions_hydro.csv"
# PRED_SUBS = BASE_DIR / "modeling_results_ok" / "predictions_subsidence.csv"
# OUTDIR = BASE_DIR / "outputs"
# OUTDIR.mkdir(parents=True, exist_ok=True)

# DOMAIN_ORDER = ["hydro", "wind", "subsidence"]
# MODEL_ORDER = ["qar", "qgbm", "xtft"]

# CAS params (as used in R5/R6)
KERNEL_KIND = "gaussian"
LAMBDA = 1.0
GAMMA = 1.25
H_REGULAR = {"wind": 6, "hydro": 7}  # steps
H_IRREG_DAYS = 30.0  # subsidence (days)


# ----------------------------
# Helpers (compact copies)
# ----------------------------
def enforce_non_crossing_df(df: pd.DataFrame) -> pd.DataFrame:
    q10 = df["q10"].to_numpy()
    q50 = np.maximum(df["q50"].to_numpy(), q10)
    q90 = np.maximum(df["q90"].to_numpy(), q50)
    out = df.copy()
    out["q10"], out["q50"], out["q90"] = q10, q50, q90
    return out


def pinball_loss(y, q, tau):
    u = y - q
    return (tau - (u < 0).astype(float)) * u


def crps_proxy_vec(y, q10, q50, q90):
    y = np.asarray(y, float)
    q10 = np.asarray(q10, float)
    q50 = np.asarray(q50, float)
    q90 = np.asarray(q90, float)
    p10 = pinball_loss(y, q10, 0.10)
    p50 = pinball_loss(y, q50, 0.50)
    p90 = pinball_loss(y, q90, 0.90)
    out = (p10 + p50 + p90) / 3.0
    out[~np.isfinite(out)] = np.nan
    return out


def winkler80_vec(y, q10, q90, alpha=0.2):
    """
    Winkler / interval score for central (1-alpha) interval.
    Here alpha=0.2 for q10–q90.
    """
    y = np.asarray(y, float)
    q10 = np.asarray(q10, float)
    q90 = np.asarray(q90, float)
    score = (q90 - q10).astype(float)
    below = y < q10
    above = y > q90
    score[below] += (2.0 / alpha) * (q10[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - q90[above])
    score[~np.isfinite(score)] = np.nan
    return score


def miss_and_sign(y, q10, q90):
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


def local_density_regular(I: np.ndarray, h: float, kind="gaussian"):
    w = kernel_vec(h, kind=kind)
    den = np.convolve(np.ones_like(I, float), w, mode="same")
    num = np.convolve(I.astype(float), w, mode="same")
    d = np.divide(num, np.maximum(den, 1e-12))
    return np.clip(d, 0.0, 1.0)


def to_seconds(arr) -> np.ndarray:
    s = pd.to_datetime(pd.Series(arr), errors="coerce")
    if s.notna().any():
        return (s.view("int64") / 1e9).to_numpy()
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(float)


def local_density_irregular(
    times, I: np.ndarray, h_days: float, kind="gaussian"
):
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


def local_cas(m, d, lam=1.0, gamma=1.25):
    return m * (1.0 + lam * (d**gamma))


# ----------------------------
# Load predictions
# ----------------------------
def load_preds():
    wind = pd.read_csv(PRED_WIND)
    hydro = pd.read_csv(PRED_HYDRO)
    subs = pd.read_csv(PRED_SUBS)

    wind = enforce_non_crossing_df(wind).assign(domain="wind")
    hydro = enforce_non_crossing_df(hydro).assign(domain="hydro")
    subs = enforce_non_crossing_df(subs).assign(domain="subsidence")

    if "t" in subs and subs["t"].dtype == object:
        subs["t"] = pd.to_datetime(subs["t"], errors="coerce")

    return pd.concat([hydro, wind, subs], ignore_index=True)


preds = load_preds()


# ----------------------------
# Per-observation losses for DM
# ----------------------------
def losses_for_dm(df_block):
    """
    Return dict model -> {'crps': arr, 'wink': arr}
    with NaNs dropped.
    """
    out = {}
    for m, g in df_block.groupby("model"):
        y = g["y"].to_numpy(float)
        q10 = g["q10"].to_numpy(float)
        q50 = g["q50"].to_numpy(float)
        q90 = g["q90"].to_numpy(float)
        crps = crps_proxy_vec(y, q10, q50, q90)
        win = winkler80_vec(y, q10, q90, alpha=0.2)
        out[m] = {
            "crps": crps[np.isfinite(crps)],
            "wink": win[np.isfinite(win)],
        }
    return out


# ----------------------------
# Diebold–Mariano with HAC NW
# ----------------------------
def newey_west_se(d, L):
    """
    HAC-robust SE for mean(d) with Bartlett up to lag L.
    """
    d = d[np.isfinite(d)]
    n = d.size
    if n < 5:
        return np.nan
    mu = d.mean()
    v0 = np.mean((d - mu) ** 2)
    gamma = 0.0
    for l in range(1, min(L, n - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        cov = np.mean((d[l:] - mu) * (d[:-l] - mu))
        gamma += 2.0 * w * cov
    var = v0 + gamma
    se = sqrt(max(var, 0.0) / n)
    return se


def dm_test(loss1, loss2, horizon_h, alternative="two-sided"):
    """
    DM test with NW HAC. L = max(1, h-1).
    Returns (t, p, mean_diff).
    """
    l1 = loss1[np.isfinite(loss1)]
    l2 = loss2[np.isfinite(loss2)]
    n = min(l1.size, l2.size)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    d = (l1[:n] - l2[:n]).astype(float)
    d_bar = np.mean(d)
    L = max(1, int(horizon_h) - 1)
    se = newey_west_se(d, L)
    if not np.isfinite(se) or se == 0.0:
        return (np.nan, np.nan, d_bar)
    t = d_bar / se
    if alternative == "two-sided":
        p = 2.0 * (1.0 - norm.cdf(abs(t)))
    elif alternative == "less":
        p = norm.cdf(t)
    else:
        p = 1.0 - norm.cdf(t)
    return (t, p, d_bar)


def holm_adjust(pvals):
    """
    Holm step-down. Input: dict key->p.
    Return dict key->p_holm.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    for i, (k, p) in enumerate(items, start=1):
        out[k] = min(1.0, (m - i + 1) * p)
    for i in range(m - 1, 0, -1):
        ki = items[i][0]
        kj = items[i - 1][0]
        out[kj] = min(out[kj], out[ki])
    return out


# ----------------------------
# CAS per-series and MCS
# ----------------------------
def abs_cas_series(g, domain):
    """
    |CAS| vector for a single series block (sorted by time).
    """
    if "t" in g:
        g = g.sort_values("t")
    y = g["y"].to_numpy(float)
    q10 = g["q10"].to_numpy(float)
    q90 = g["q90"].to_numpy(float)
    m, s = miss_and_sign(y, q10, q90)
    I = (s != 0).astype(int)
    if domain == "subsidence":
        t = g["t"].to_numpy() if "t" in g else np.arange(y.size)
        d = local_density_irregular(
            t, I, h_days=H_IRREG_DAYS, kind=KERNEL_KIND
        )
    else:
        h = H_REGULAR[domain]
        d = local_density_regular(I, h=h, kind=KERNEL_KIND)
    c = local_cas(m, d, lam=LAMBDA, gamma=GAMMA)
    return np.abs(c)


def mcs_membership(loss_matrix, models, B=1000, alpha=0.10, rng_seed=42):
    """
    Hansen et al. MCS (simplified). Rows=series, cols=models.
    Lower is better. Bootstrap over series.
    """
    rng = np.random.default_rng(rng_seed)
    S, M = loss_matrix.shape
    current = list(range(M))

    def boot_Tmax(cols):
        idx = rng.integers(0, S, size=S)
        Lb = loss_matrix[idx][:, cols]
        mean_b = np.nanmean(Lb, axis=0)
        return float(np.nanmax(mean_b - np.nanmin(mean_b)))

    while len(current) > 1:
        Lc = loss_matrix[:, current]
        mean_obs = np.nanmean(Lc, axis=0)
        T_obs = float(np.nanmax(mean_obs - np.nanmin(mean_obs)))
        boots = np.array([boot_Tmax(current) for _ in range(B)])
        pval = np.mean(boots >= T_obs)
        if pval < alpha:
            worst = int(np.nanargmax(mean_obs))
            del current[worst]
        else:
            break

    return {models[j] for j in current}


# ----------------------------
# Table 6: DM by domain×horizon×metric
# ----------------------------
rows_dm = []
for d in DOMAIN_ORDER:
    df_d = preds[preds["domain"] == d]
    for h, df_h in df_d.groupby("horizon"):
        L = losses_for_dm(df_h)
        pairs = list(it.combinations([m for m in MODEL_ORDER if m in L], 2))
        for metric in ["crps", "wink"]:
            p_for_holm = {}
            tmp = {}
            for m1, m2 in pairs:
                t, p, dbar = dm_test(
                    L[m1][metric],
                    L[m2][metric],
                    horizon_h=int(h),
                    alternative="two-sided",
                )
                key = f"{metric}:{m1} vs {m2}"
                p_for_holm[key] = p if np.isfinite(p) else 1.0
                tmp[key] = (t, p, dbar)
            ph = holm_adjust(p_for_holm) if p_for_holm else {}
            for key, (t, p, dbar) in tmp.items():
                m1, m2 = key.split(":")[1].split(" vs ")
                rows_dm.append(
                    {
                        "domain": d,
                        "horizon": int(h),
                        "metric": metric.upper(),
                        "model1": m1,
                        "model2": m2,
                        "mean_diff_l1_minus_l2": dbar,
                        "t_DM": t,
                        "p_value": p,
                        "p_holm": ph.get(key, np.nan),
                    }
                )

table6 = pd.DataFrame(rows_dm).sort_values(
    ["domain", "metric", "horizon", "model1", "model2"]
)
t6_csv = OUTDIR / "table6_dm_tests.csv"
t6_tex = OUTDIR / "table6_dm_tests.tex"
table6.to_csv(t6_csv, index=False)
with open(t6_tex, "w", encoding="utf-8") as f:
    f.write(table6.to_latex(index=False, float_format="%.4g"))
print(f"[Saved] {t6_csv}\n[Saved] {t6_tex}")

# ----------------------------
# Table 7: MCS for |CAS| by domain×horizon
# ----------------------------
rows_mcs = []
for d in DOMAIN_ORDER:
    df_d = preds[preds["domain"] == d]
    for h, df_h in df_d.groupby("horizon"):
        Ms_all = sorted(
            [m for m in MODEL_ORDER if m in df_h["model"].unique()],
            key=lambda x: MODEL_ORDER.index(x),
        )
        if len(Ms_all) < 2:
            continue
        series_ids = sorted(df_h["series_id"].unique())
        Lmat = np.full((len(series_ids), len(Ms_all)), np.nan, float)
        for si, sid in enumerate(series_ids):
            g_ser = df_h[df_h["series_id"] == sid]
            for mj, m in enumerate(Ms_all):
                g_sm = g_ser[g_ser["model"] == m]
                if g_sm.empty:
                    continue
                ac = abs_cas_series(g_sm, domain=d)
                if ac.size == 0:
                    continue
                Lmat[si, mj] = np.nanmean(ac)
        members = mcs_membership(
            Lmat, Ms_all, B=1000, alpha=0.10, rng_seed=123
        )
        for m in Ms_all:
            rows_mcs.append(
                {
                    "domain": d,
                    "horizon": int(h),
                    "model": m,
                    "in_MCS_alpha0.10": 1 if m in members else 0,
                }
            )

table7 = pd.DataFrame(rows_mcs).sort_values(["domain", "horizon", "model"])
t7_csv = OUTDIR / "table7_mcs_cas.csv"
t7_tex = OUTDIR / "table7_mcs_cas.tex"
table7.to_csv(t7_csv, index=False)
with open(t7_tex, "w", encoding="utf-8") as f:
    wide = (
        table7.pivot(
            index=["domain", "horizon"],
            columns="model",
            values="in_MCS_alpha0.10",
        )
        .fillna(0)
        .astype(int)
    )
    f.write(wide.to_latex(float_format="%.0f"))

print(f"[Saved] {t7_csv}\n[Saved] {t7_tex}")
