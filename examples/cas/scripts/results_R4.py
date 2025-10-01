"""
R4: Case studies — fan charts with clustered-failure overlays
    + local CAS stems.

Reads (via results_config):
  predictions_wind.csv
  predictions_hydro.csv
  predictions_subsidence.csv

Writes (under data/cas/outputs/):
  figure5_wind_cases.(png|pdf)
  figure6_hydro_cases.(png|pdf)
  figure7_subsidence_cases.(png|pdf)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from results_config import (
    DOMAIN_COLOR,  # shared palette/labels
    OUTDIR,
    PRED_HYDRO,
    PRED_SUBS,
    PRED_WIND,
    enforce_non_crossing,  # q10<=q50<=q90 guard
)

# ----------------------------
# Config
# ----------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)

# Domain colors (consistency)
COL_DOMAIN = {
    "hydro": DOMAIN_COLOR["hydro"],
    "wind": DOMAIN_COLOR["wind"],
    "subsidence": DOMAIN_COLOR["subsidence"],
}

# Exceedance colors
COL_ABOVE = "#D55E00"  # above upper band
COL_BELOW = "#56B4E9"  # below lower band
COL_BAND = "#BDBDBD"  # fan interior (neutral gray)

# Default horizon to visualize
DEFAULT_H = 7

# Local CAS parameters
KERNEL = "gaussian"  # 'gaussian' or 'triangular'
H_WIND_STEPS = 12  # bandwidth (steps) for wind (hourly)
H_HYDRO_STEPS = 21  # bandwidth (steps) for hydro (daily)
H_SUBS_TIME = 10.0  # bandwidth in *time units* for subsidence
LAMBDA = 1.0
GAMMA = 1.25

# Models to show per domain
MODELS_TO_SHOW = {
    "wind": ["qar", "xtft"],
    "hydro": ["xtft", "qgbm"],
    "subsidence": ["qar", "xtft"],
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
        "lines.linewidth": 1.8,
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
# IO helpers
# ----------------------------
def ensure_time_col(df: pd.DataFrame) -> pd.DataFrame:
    """Parse t to datetime if possible; otherwise leave as-is."""
    if df["t"].dtype == object:
        df = df.copy()
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


wind = ensure_time_col(pd.read_csv(PRED_WIND))
hydro = ensure_time_col(pd.read_csv(PRED_HYDRO))
subs = ensure_time_col(pd.read_csv(PRED_SUBS))

wind = enforce_non_crossing_df(wind)
hydro = enforce_non_crossing_df(hydro)
subs = enforce_non_crossing_df(subs)


# ----------------------------
# Utilities
# ----------------------------
def miss_and_sign(
    y: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized miss m in [0,inf) and sign in {-1,0,+1}.
    Normalization uses band width.
    """
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


def kernel_weights(
    dt: np.ndarray,
    h: float,
    kind: str = "gaussian",
) -> np.ndarray:
    dt = np.asarray(dt, dtype=float)
    if h <= 0:
        return (dt == 0).astype(float)
    if kind == "gaussian":
        z = dt / h
        return np.exp(-0.5 * z * z)
    if kind == "triangular":
        return np.maximum(0.0, 1.0 - np.abs(dt) / h)
    raise ValueError("Unknown kernel kind")


def local_density(
    times,
    exc_indicator: np.ndarray,
    h: float,
    kernel_kind: str = "gaussian",
    irregular: bool = False,
) -> np.ndarray:
    """d_t in [0,1]. If irregular=False, use index distances.
    If irregular=True, convert times to seconds if datetime.
    """
    IND = np.asarray(exc_indicator, dtype=float)
    n = len(IND)
    d = np.zeros(n, dtype=float)
    if n == 0:
        return d

    if irregular:
        ts = pd.to_datetime(pd.Series(times), errors="coerce")
        if ts.notna().any():
            # seconds since epoch
            t = (ts.view("int64") / 1e9).to_numpy()
        else:
            t = pd.to_numeric(pd.Series(times), errors="coerce").to_numpy(
                dtype=float
            )
        for i in range(n):
            dt = t - t[i]
            w = kernel_weights(dt, h, kind=kernel_kind)
            den = np.sum(w)
            d[i] = np.sum(w * IND) / (den if den > 1e-12 else 1.0)
    else:
        idx = np.arange(n, dtype=float)
        for i in range(n):
            dt = idx - i
            w = kernel_weights(dt, h, kind=kernel_kind)
            den = np.sum(w)
            d[i] = np.sum(w * IND) / (den if den > 1e-12 else 1.0)

    return np.clip(d, 0.0, 1.0)


def local_cas(
    m: np.ndarray,
    d: np.ndarray,
    lam: float = LAMBDA,
    gamma: float = GAMMA,
    s: np.ndarray | None = None,
) -> np.ndarray:
    base = m * (1.0 + lam * (d**gamma))
    return base if s is None else base * np.sign(s)


def find_runs(sign_arr: np.ndarray):
    """Return list of (i0, i1, sign, length). end inclusive."""
    runs = []
    n = len(sign_arr)
    i = 0
    while i < n:
        if sign_arr[i] == 0:
            i += 1
            continue
        s = int(sign_arr[i])
        j = i
        while (j + 1 < n) and (sign_arr[j + 1] == s):
            j += 1
        runs.append((i, j, s, j - i + 1))
        i = j + 1
    return runs


def longest_run_and_gap(runs):
    if not runs:
        return None, None
    lr = max(runs, key=lambda r: r[3])
    _, end, _, _ = lr
    gaps = [r[0] - end - 1 for r in runs if r[0] > end]
    return lr, (min(gaps) if gaps else None)


def choose_h_available(df: pd.DataFrame, target_h: int) -> int:
    hs = np.sort(df["horizon"].unique())
    if target_h in hs:
        return int(target_h)
    return int(hs[np.argmin(np.abs(hs - target_h))])


def select_representative_series(
    df_dom: pd.DataFrame,
    model: str,
    horizon: int,
    strategy: str = "max_cas",
    k: int = 1,
    h_band: float | None = None,
    irregular: bool = False,
) -> list:
    """Pick k series_ids that best illustrate bursts."""
    sub = df_dom[
        (df_dom["model"] == model) & (df_dom["horizon"] == horizon)
    ].copy()
    if sub.empty:
        return []

    scores = []
    for sid, g in sub.groupby("series_id"):
        g = g.sort_values("t")
        m, s = miss_and_sign(
            g["y"].to_numpy(),
            g["q10"].to_numpy(),
            g["q90"].to_numpy(),
        )
        Ib = (s != 0).astype(int)
        d = local_density(
            g["t"].to_numpy(),
            Ib,
            h=h_band,
            kernel_kind=KERNEL,
            irregular=irregular,
        )
        c = local_cas(m, d, lam=LAMBDA, gamma=GAMMA)
        if strategy == "max_cas":
            score = c.sum()
        else:
            rs = find_runs(s)
            score = rs[-1][3] if rs else 0
        scores.append((sid, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:k]]


def fan_with_bursts(
    ax,
    g: pd.DataFrame,
    domain_color: str,
    irregular: bool = False,
    title_suffix: str = "",
):
    g = g.sort_values("t")
    t = g["t"].to_numpy()
    y = g["y"].to_numpy()
    q10 = g["q10"].to_numpy()
    q50 = g["q50"].to_numpy()
    q90 = g["q90"].to_numpy()

    # Fan and series
    ax.fill_between(t, q10, q90, color=COL_BAND, alpha=0.35, linewidth=0)
    ax.plot(t, q50, ls="--", color="0.35", lw=1.2)
    ax.plot(t, y, color="black", lw=1.4)

    # Burst shading
    m, sgn = miss_and_sign(y, q10, q90)
    runs = find_runs(sgn)
    for i0, i1, s, _L in runs:
        ax.axvspan(
            t[i0],
            t[i1],
            facecolor=(COL_ABOVE if s > 0 else COL_BELOW),
            alpha=0.15,
            zorder=0,
        )

    # Bandwidth for density
    if irregular:
        ts = pd.to_datetime(pd.Series(t), errors="coerce")
        if ts.notna().any():
            step = np.median(np.diff(ts.view("int64"))) / 1e9
            h = max(step, 1.0) * 10.0
        else:
            step = np.median(np.diff(pd.to_numeric(t, errors="coerce")))
            h = max(step, 1.0) * 10.0
    else:
        # if domain not attached, fall back to wind
        dom = g.get("domain", pd.Series(["wind"])).iloc[0]
        h = H_WIND_STEPS if dom == "wind" else H_HYDRO_STEPS

    # Local CAS stems
    I = (sgn != 0).astype(int)
    d = local_density(t, I, h=h, kernel_kind=KERNEL, irregular=irregular)
    c = local_cas(m, d, lam=LAMBDA, gamma=GAMMA)

    ax2 = ax.twinx()
    ax2.vlines(
        t[sgn != 0],
        0.0,
        c[sgn != 0],
        colors=[COL_ABOVE if s > 0 else COL_BELOW for s in sgn[sgn != 0]],
        linewidth=1.2,
        alpha=0.9,
    )
    ax2.set_ylim(0, max(1e-3, np.percentile(c, 99) * 1.1))
    ax2.set_ylabel("Local CAS", color="0.25")
    ax2.tick_params(axis="y", labelsize=9, colors="0.25")
    ax2.spines["top"].set_visible(False)

    ax.set_title(title_suffix, color=domain_color)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax, ax2


# ----------------------------
# Build per-domain figures
# ----------------------------
def build_domain_cases(
    df_dom: pd.DataFrame,
    domain: str,
    out_stem: str,
    irregular: bool = False,
):
    models = MODELS_TO_SHOW[domain]
    h = choose_h_available(df_dom, DEFAULT_H)
    h_band = (
        H_SUBS_TIME
        if irregular
        else (H_WIND_STEPS if domain == "wind" else H_HYDRO_STEPS)
    )

    chosen: list[tuple[str, str]] = []
    for m in models:
        ids = select_representative_series(
            df_dom,
            m,
            h,
            strategy="max_cas",
            k=1,
            h_band=h_band,
            irregular=irregular,
        )
        if ids:
            chosen.append((m, ids[0]))

    if not chosen:
        print(f"[warn] no cases selected for {domain}")
        return

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), sharey=False)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]

    for ax, (m, sid) in zip(axes, chosen):
        g = df_dom[
            (df_dom["model"] == m)
            & (df_dom["horizon"] == h)
            & (df_dom["series_id"] == sid)
        ].copy()
        title = f"{m.upper()} — series {sid} (h={h})"
        fan_with_bursts(
            ax,
            g,
            domain_color=COL_DOMAIN[domain],
            irregular=irregular,
            title_suffix=title,
        )

    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COL_BAND,
            alpha=0.35,
            edgecolor="none",
            label="q10–q90",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COL_ABOVE,
            alpha=0.15,
            edgecolor="none",
            label="above-run",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COL_BELOW,
            alpha=0.15,
            edgecolor="none",
            label="below-run",
        ),
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    png = OUTDIR / f"{out_stem}.png"
    pdf = OUTDIR / f"{out_stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png}\n[Saved] {pdf}")


# Wind (hourly; regular grid)
build_domain_cases(
    wind.assign(domain="wind"),
    "wind",
    "figure5_wind_cases",
    irregular=False,
)

# Hydro (daily; regular grid)
build_domain_cases(
    hydro.assign(domain="hydro"),
    "hydro",
    "figure6_hydro_cases",
    irregular=False,
)

# Subsidence (irregular sampling)
build_domain_cases(
    subs.assign(domain="subsidence"),
    "subsidence",
    "figure7_subsidence_cases",
    irregular=True,
)

print("\n[Done] Case-study figures written to ./outputs/")
