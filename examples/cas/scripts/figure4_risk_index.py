"""
Figure 4 – Operational risk-index and early-warning curves
===========================================================

Implements the causal risk index R_t from §5.4 of the revised CAS paper:

  R_t = (1 + λ · d^past_{t−1}^γ) · ψ_t

where:
  d^past_{t−1} = one-sided causal local density (past violations only)
  ψ_t = forecast-geometry pressure term
        = max(q50−q10, q90−q50) / (q90−q10 + ε)

Alert rule: raise an alert at t when R_t ≥ θ.

Evaluation (§5.4, Table 5):
  "Long episode" = run of ≥ L_MIN consecutive violations.
  Detection: alert fired within the first L_NEAR steps of an episode start.
  Sweep θ across quantiles of R to obtain (achieved FBR, burst recall) curve.

Figure layout:
  Three panels (one per domain), each showing burst-recall vs achieved-FBR
  curves for QAR, QGBM, XTFT.  Reference line at FBR = 0.10.

Outputs (data/cas/outputs/):
  figure4_risk_index.png / .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
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
    _REPO = _HERE.parents[2]
    DATA_ROOT = _REPO / "data" / "cas"
    OUTDIR = DATA_ROOT / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PRED_WIND = DATA_ROOT / "modeling_results_ok" / "predictions_wind.csv"
    PRED_HYDRO = DATA_ROOT / "modeling_results_ok" / "predictions_hydro.csv"
    PRED_SUBS = (
        DATA_ROOT / "modeling_results_ok" / "predictions_subsidence.csv"
    )
    DOMAIN_COLOR = {
        "hydro": "#0072B2",
        "wind": "#E69F00",
        "subsidence": "#009E73",
    }
    DOMAIN_LABEL = {
        "hydro": "Hydro",
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
        q90c = np.maximum(q90, q50c)
        return q10, q50c, q90c


PRED_FILES = {
    "hydro": PRED_HYDRO,
    "wind": PRED_WIND,
    "subsidence": PRED_SUBS,
}

# ---------------------------------------------------------------------------
# Risk-index parameters
# ---------------------------------------------------------------------------
H_PAST = 5  # causal window width (past h steps)
LAMBDA = 1.0
GAMMA = 1.0
L_MIN = 3  # minimum run length to be a "long episode"
L_NEAR = 2  # alert counts as "near onset" if within first L_NEAR steps
N_THETA = 80  # number of threshold values to sweep
TARGET_FBR = 0.10  # target false-burst rate for the summary annotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def causal_density(v: np.ndarray, h: int) -> np.ndarray:
    """
    One-sided (causal) local violation density using a triangular kernel:
      d_past_t = Σ_{k=1}^{h} K(k/h) · v_{t-k}  /  Σ_{k=1}^{h} K(k/h)
    where K(u) = (1−u)₊.
    Returns d_past with d_past[t] computed from v[0..t-1] only.
    """
    n = len(v)
    d = np.zeros(n, dtype=float)
    # Build triangular weights for lags 1..h
    lags = np.arange(1, h + 1)  # 1, 2, …, h
    weights = np.maximum(0.0, 1.0 - lags / h)  # K(1/h), K(2/h), …, K(1)
    weight_sum = weights.sum()
    if weight_sum == 0:
        return d
    weights = weights / weight_sum

    for t in range(1, n):
        # Look back up to h steps
        max_lag = min(t, h)
        past_v = v[t - max_lag : t][::-1]  # lag 1 first
        w = weights[:max_lag]
        w = w / w.sum() if w.sum() > 0 else w
        d[t] = np.dot(w, past_v)
    return d


def geometry_pressure(
    q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    """
    ψ_t = 2·(max(q50−q10, q90−q50) / (q90−q10) − 0.5)
    Ranges from 0 (perfectly symmetric) to 1 (median at one edge).
    Centred so that symmetric intervals contribute nothing.
    """
    bw = q90 - q10 + eps
    half_up = q90 - q50
    half_dn = q50 - q10
    return np.clip(2.0 * np.maximum(half_up, half_dn) / bw - 1.0, 0.0, 1.0)


def compute_risk_index(df_sub: pd.DataFrame):
    """Compute R_t for one (model, horizon, series) block."""
    q10, q50, q90 = enforce_non_crossing(
        df_sub["q10"].to_numpy(),
        df_sub["q50"].to_numpy(),
        df_sub["q90"].to_numpy(),
    )
    y = df_sub["y"].to_numpy()
    v = ((y < q10) | (y > q90)).astype(float)

    d_past = causal_density(v, H_PAST)
    psi = geometry_pressure(q10, q50, q90)

    # R = 0 when no past violations AND symmetric forecast
    R = (1.0 + LAMBDA * d_past**GAMMA) * (1.0 + psi) - 1.0
    return R, v


def find_episodes(v: np.ndarray, l_min: int) -> list[tuple[int, int]]:
    """
    Return list of (start, end) inclusive indices for runs of consecutive
    violations of length ≥ l_min.
    """
    episodes = []
    n = len(v)
    i = 0
    while i < n:
        if v[i] == 1:
            j = i
            while j < n and v[j] == 1:
                j += 1
            if (j - i) >= l_min:
                episodes.append((i, j - 1))
            i = j
        else:
            i += 1
    return episodes


def _alert_bursts(alert_times: np.ndarray) -> list[tuple[int, int]]:
    """Group consecutive alert indices into (start, end) burst tuples."""
    if len(alert_times) == 0:
        return []
    bursts = []
    bstart = alert_times[0]
    bprev = alert_times[0]
    for a in alert_times[1:]:
        if a == bprev + 1:
            bprev = a
        else:
            bursts.append((bstart, bprev))
            bstart = a
            bprev = a
    bursts.append((bstart, bprev))
    return bursts


def detection_curve(
    R: np.ndarray,
    v: np.ndarray,
    episodes: list[tuple[int, int]],
    l_near: int,
    n_theta: int,
):
    """
    Sweep θ and return (fbr_arr, recall_arr) arrays.

    Burst-level FBR: group consecutive alert times into alert bursts.
      false-burst rate = (alert bursts not overlapping any episode onset) /
                         (total alert bursts)
      burst recall     = (long episodes with an alert within first l_near steps) /
                         (total long episodes)
    """
    # Use quantile-spaced thresholds to cover the range better
    R_pos = R[R > 0]
    if len(R_pos) == 0:
        return np.zeros(1), np.zeros(1)
    thresholds = np.unique(
        np.percentile(R_pos, np.linspace(0, 100, n_theta + 1))
    )
    n_ep = len(episodes)

    fbr_arr = []
    rec_arr = []

    # Build episode onset sets
    onset_sets = [set(range(s, min(s + l_near, e + 1))) for s, e in episodes]
    all_onset = set().union(*onset_sets) if onset_sets else set()

    for theta in thresholds:
        alert_times = np.where(R >= theta)[0]
        if len(alert_times) == 0:
            fbr_arr.append(0.0)
            rec_arr.append(0.0)
            continue

        # Group into bursts
        bursts = _alert_bursts(alert_times)
        n_bursts = len(bursts)

        # True burst: any alert in burst overlaps an episode onset window
        true_bursts = sum(
            1
            for bs, be in bursts
            if any(a in all_onset for a in range(bs, be + 1))
        )
        false_bursts = n_bursts - true_bursts
        fbr = false_bursts / n_bursts

        # Detected episodes
        alert_set = set(alert_times.tolist())
        detected = sum(1 for onset in onset_sets if onset & alert_set)
        recall = detected / n_ep if n_ep > 0 else 0.0

        fbr_arr.append(fbr)
        rec_arr.append(recall)

    return np.array(fbr_arr), np.array(rec_arr)


def read_at_target_fbr(fbr_arr, rec_arr, target=TARGET_FBR):
    """Return (achieved_fbr, recall) at the best operating point with FBR <= target.

    If no point satisfies FBR <= target (i.e. the minimum achievable FBR is
    already above the target), return the minimum-FBR point so the table still
    reports a meaningful value rather than NaN.
    """
    order = np.argsort(fbr_arr)
    fbr_s = fbr_arr[order]
    rec_s = rec_arr[order]
    under = fbr_s <= target
    if under.any():
        # Last (= highest recall) operating point that stays within the FBR budget
        idx = int(np.where(under)[0][-1])
    else:
        # Constraint can't be met; report the minimum-FBR operating point
        idx = 0
    return float(fbr_s[idx]), float(rec_s[idx])


# ---------------------------------------------------------------------------
# Main computation per (domain, model)
# ---------------------------------------------------------------------------
print("Computing risk-index curves …")

curves = {}  # {(domain, model): (fbr_arr, rec_arr)}
summary_rows = []

for domain in DOMAIN_ORDER:
    df_all = pd.read_csv(PRED_FILES[domain])
    df_all = df_all.dropna(subset=["q10", "q50", "q90", "y"])

    for model in MODEL_ORDER:
        sub = df_all[df_all["model"] == model]
        if sub.empty or sub["q10"].isna().all():
            curves[(domain, model)] = (
                np.array([0.0, 1.0]),
                np.array([0.0, 0.0]),
            )
            summary_rows.append(
                dict(
                    domain=domain,
                    model=model,
                    achieved_fbr=np.nan,
                    burst_recall=np.nan,
                    n_episodes=0,
                )
            )
            continue

        # Concatenate all horizons into one time stream per domain
        all_R, all_v, all_ep = [], [], []
        offset = 0
        for _horizon, grp in sub.groupby("horizon"):
            grp = grp.reset_index(drop=True)
            R_h, v_h = compute_risk_index(grp)
            ep_h = find_episodes(v_h, L_MIN)
            # Adjust episode indices by offset
            ep_h_off = [(s + offset, e + offset) for s, e in ep_h]
            all_R.append(R_h)
            all_v.append(v_h)
            all_ep.extend(ep_h_off)
            offset += len(grp)

        R_full = np.concatenate(all_R)
        v_full = np.concatenate(all_v)

        fbr_arr, rec_arr = detection_curve(
            R_full, v_full, all_ep, L_NEAR, N_THETA
        )
        curves[(domain, model)] = (fbr_arr, rec_arr)

        ach_fbr, ach_rec = read_at_target_fbr(fbr_arr, rec_arr, TARGET_FBR)
        summary_rows.append(
            dict(
                domain=domain,
                model=model,
                achieved_fbr=ach_fbr,
                burst_recall=ach_rec,
                n_episodes=len(all_ep),
            )
        )

        print(
            f"  {domain}/{model}: "
            f"n_episodes={len(all_ep)}, "
            f"FBR@{TARGET_FBR}~={ach_fbr:.3f}, "
            f"recall~={ach_rec:.3f}"
        )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTDIR / "table4_risk_index_summary.csv", index=False)
print("\n  Summary saved.")

# ---------------------------------------------------------------------------
# Figure: 1 row × 3 columns (one per domain)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax_idx, domain in enumerate(DOMAIN_ORDER):
    ax = axes[ax_idx]
    col = DOMAIN_COLOR[domain]

    for model in MODEL_ORDER:
        fbr_arr, rec_arr = curves.get(
            (domain, model), (np.array([]), np.array([]))
        )
        if len(fbr_arr) == 0:
            continue

        # Sort and smooth for clean curve
        order = np.argsort(fbr_arr)
        fbr_s = fbr_arr[order]
        rec_s = rec_arr[order]

        # Slightly lighten model color relative to domain
        lw = 2.2 if model == "qar" else 1.8
        zord = 4 if model == "qar" else 3

        ax.plot(
            fbr_s,
            rec_s,
            ls=MODEL_STYLE.get(model, "-"),
            lw=lw,
            marker=MODEL_MARK.get(model, "o"),
            ms=5,
            color=col,
            alpha=0.88,
            label=MODEL_LABEL.get(model, model.upper()),
            zorder=zord,
            markevery=max(1, len(fbr_s) // 10),
        )

        # Annotate the operating point at target FBR
        ach_fbr, ach_rec = read_at_target_fbr(fbr_arr, rec_arr, TARGET_FBR)
        if np.isfinite(ach_fbr) and np.isfinite(ach_rec):
            ax.scatter(
                ach_fbr,
                ach_rec,
                s=70,
                color=col,
                marker=MODEL_MARK.get(model, "o"),
                edgecolors="white",
                lw=1.2,
                zorder=6,
            )

    # Reference FBR line
    ax.axvline(
        TARGET_FBR,
        color="0.4",
        lw=1.0,
        ls=":",
        label=f"Target FBR={TARGET_FBR}",
    )

    # Diagonal (random alert)
    ax.plot([0, 1], [0, 1], color="0.80", lw=0.8, ls="--", label="Random")

    # Decorations
    ax.set_title(
        DOMAIN_LABEL[domain], fontweight="bold", color=col, fontsize=12
    )
    ax.set_xlabel("False-burst rate (FBR)")
    if ax_idx == 0:
        ax.set_ylabel("Burst recall")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.09)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.92)
    ax.grid(True, alpha=0.20, ls="--")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Shade background lightly with domain colour
    ax.set_facecolor(col + "08")

# ---------------------------------------------------------------------------
# Inset summary bar chart (achieved recall at target FBR)
# ---------------------------------------------------------------------------
fig.suptitle(
    "Operational risk-index: burst recall vs. false-burst rate\n"
    rf"(causal window $h={H_PAST}$, $\lambda={LAMBDA}$, $\gamma={GAMMA}$; "
    rf"episode $\geq {L_MIN}$ consecutive violations; "
    rf"detection within first {L_NEAR} steps of onset)",
    y=1.05,
    fontsize=12,
    fontweight="bold",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_png = OUTDIR / "figure4_risk_index.png"
out_pdf = OUTDIR / "figure4_risk_index.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"\n[Saved] {out_png}")
print(f"[Saved] {out_pdf}")
print("[Done] figure4_risk_index complete.")
