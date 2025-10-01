# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Shared config for Results scripts (R1-R9).

This module defines:

  * Repo-aware paths to inputs/outputs under data/cas/
  * Domain & model display order, labels, colors/markers
  * Global rcParams for a journal-style look
  * Common constants (coverage level, PIT bins, RNG, etc.)
  * Lightweight utilities reused across figures/tables:
      - enforce_non_crossing, PIT from three quantiles
      - reliability points + bootstrap CIs
      - load_preds() with PIT + coverage indicator
      - label_points() for unobtrusive annotations

Usage
-----
from results_config import (
    METRICS_PATH, PRED_WIND, PRED_HYDRO, PRED_SUBS, OUTDIR,
    DOMAIN_ORDER, DOMAIN_LABEL, DOMAIN_COLOR, DOMAIN_MARKER,
    MODEL_ORDER, MODEL_LABEL, MODEL_MARK, MODEL_STYLE,
    NOMINAL_COVERAGE, TAUS, PIT_BINS, N_BOOT, rng,
    enforce_non_crossing, pit_from_tri_quantiles,
    reliability_points, reliability_ci, load_preds, label_points,
)

Paths default to:
  data/cas/modeling_results_ok/*.csv  -> inputs
  data/cas/outputs/                   -> figures/tables
Override base via env var:
  KDIAGRAM_DATA_DIR=/custom/path/to/data/cas
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import (  # optional, used in R2
    BoundaryNorm,
    ListedColormap,
)
from matplotlib.lines import Line2D  # re-export convenience
from matplotlib.patches import Patch  # re-export convenience

# =========================
# Repo-aware PATHS
# =========================


def _find_repo_root(start: Path) -> Path:
    """Walk up a few levels to guess the repo root (has data/ or .git)."""
    markers = ("pyproject.toml", "README.md", ".git")
    p = start.resolve()
    for _ in range(6):
        if (p / "data").exists() or any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.resolve()


# 1) Optional environment override
_DATA_ROOT_ENV = os.getenv("KDIAGRAM_DATA_DIR")
if _DATA_ROOT_ENV:
    DATA_ROOT = Path(_DATA_ROOT_ENV).expanduser().resolve()
else:
    # 2) Infer repo root from this file location
    _HERE = Path(__file__).resolve().parent
    _REPO = _find_repo_root(_HERE)
    DATA_ROOT = (_REPO / "data" / "cas").resolve()

# Inputs produced by cas_modeling.py
METRICS_PATH = DATA_ROOT / "modeling_results_ok" / "metrics_all_domains.csv"
PRED_WIND = DATA_ROOT / "modeling_results_ok" / "predictions_wind.csv"
PRED_HYDRO = DATA_ROOT / "modeling_results_ok" / "predictions_hydro.csv"
PRED_SUBS = DATA_ROOT / "modeling_results_ok" / "predictions_subsidence.csv"

# Output directory for all Results figures/tables
OUTDIR = DATA_ROOT / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =========================
# Constants & Ordering
# =========================

NOMINAL_COVERAGE: float = 0.90
TAUS = np.array([0.10, 0.50, 0.90], dtype=float)
PIT_BINS: int = 20
N_BOOT: int = 200
RANDOM_SEED: int = 13
rng = np.random.default_rng(RANDOM_SEED)

# Domain palette (Ok to color-blind safe)
DOMAIN_ORDER = ["hydro", "wind", "subsidence"]
DOMAIN_LABEL = {"hydro": "Hydro", "wind": "Wind", "subsidence": "Subsidence"}
DOMAIN_COLOR = {
    "hydro": "#0072B2",  # blue
    "wind": "#E69F00",  # orange
    "subsidence": "#009E73",  # green
}
DOMAIN_MARKER = {"hydro": "o", "wind": "s", "subsidence": "^"}

# Model display
MODEL_ORDER = ["qar", "qgbm", "xtft"]
MODEL_LABEL = {"qar": "QAR", "qgbm": "QGBM", "xtft": "XTFT"}
MODEL_MARK = {"qar": "o", "qgbm": "s", "xtft": "^"}
MODEL_STYLE = {
    "qar": "-",
    "qgbm": "--",
    "xtft": "-.",
}  # neutral styles for trend lines

# Optional: raster colors used by R2 severity calendars
RASTER_CMAP = ListedColormap(
    ["#56B4E9", "#EBEBEB", "#D55E00"]
)  # below / inside / above
RASTER_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], RASTER_CMAP.N)

# =========================
# Global style (journal look)
# =========================

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
        "errorbar.capsize": 3.0,
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

# =========================
# Shared utilities
# =========================


def enforce_non_crossing(
    q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal safeguard to ensure q10 ,q50 , q90 elementwise."""
    q50c = np.maximum(q50, q10)
    q90c = np.maximum(q90, q50c)
    return q10, q50c, q90c


def pit_from_tri_quantiles(
    y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
) -> np.ndarray:
    """
    Piecewise-linear CDF from (\tau; q_\tau) at \tau \in {0.1, 0.5, 0.9};
    returns PIT = F(Y).
    """

    eps = 1e-12
    s1 = (0.5 - 0.1) / np.maximum(q50 - q10, eps)
    s2 = (0.9 - 0.5) / np.maximum(q90 - q50, eps)

    F = np.empty_like(y, dtype=float)
    left = y < q10
    mid1 = (y >= q10) & (y < q50)
    mid2 = (y >= q50) & (y < q90)
    right = y >= q90

    F[left] = np.clip(0.1 - s1[left] * (q10[left] - y[left]), 0.0, 1.0)
    F[mid1] = 0.1 + s1[mid1] * (y[mid1] - q10[mid1])
    F[mid2] = 0.5 + s2[mid2] * (y[mid2] - q50[mid2])
    F[right] = np.clip(0.9 + s2[right] * (y[right] - q90[right]), 0.0, 1.0)
    return np.clip(F, 0.0, 1.0)


def reliability_points(
    y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
):
    """Return (taus, empirical) where empirical[k] = mean(Y <= q_{tau_k})."""

    emp = np.array(
        [np.mean(y <= q10), np.mean(y <= q50), np.mean(y <= q90)], dtype=float
    )
    return TAUS, emp


def reliability_ci(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    n_boot: int = N_BOOT,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap 95% CI for the three reliability points."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(y)
    idx = np.arange(n)
    boot = np.empty((n_boot, 3), dtype=float)
    for b in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        _, emp = reliability_points(y[take], q10[take], q50[take], q90[take])
        boot[b] = emp
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return lo, hi


def load_preds(pred_path: str | Path, domain_name: str) -> pd.DataFrame:
    """
    Load a predictions CSV (q10/q50/q90,y,model,horizon,series_id,t...),
    enforce non-crossing,
    and add helper columns: domain, covered(inside q10, q90), PIT.
    """
    df = pd.read_csv(pred_path)
    q10c, q50c, q90c = enforce_non_crossing(
        df["q10"].to_numpy(), df["q50"].to_numpy(), df["q90"].to_numpy()
    )
    out = df.copy()
    out["q10"], out["q50"], out["q90"] = q10c, q50c, q90c
    out["domain"] = domain_name
    out["covered"] = (out["y"] >= out["q10"]) & (out["y"] <= out["q90"])
    out["pit"] = pit_from_tri_quantiles(out["y"].to_numpy(), q10c, q50c, q90c)
    return out


def label_points(ax, x, y, labels, dx=0.006, dy=0.006):
    """Small text labels with tiny offset to reduce overlaps."""
    for xi, yi, s in zip(np.asarray(x), np.asarray(y), labels):
        ax.text(
            float(xi) + dx,
            float(yi) + dy,
            str(s),
            fontsize=9,
            va="bottom",
            ha="left",
        )


# What this module exports (helps static checkers and docs)
__all__ = [
    # paths
    "DATA_ROOT",
    "METRICS_PATH",
    "PRED_WIND",
    "PRED_HYDRO",
    "PRED_SUBS",
    "OUTDIR",
    # orders & styles
    "DOMAIN_ORDER",
    "DOMAIN_LABEL",
    "DOMAIN_COLOR",
    "DOMAIN_MARKER",
    "MODEL_ORDER",
    "MODEL_LABEL",
    "MODEL_MARK",
    "MODEL_STYLE",
    "RASTER_CMAP",
    "RASTER_NORM",
    # constants
    "NOMINAL_COVERAGE",
    "TAUS",
    "PIT_BINS",
    "N_BOOT",
    "RANDOM_SEED",
    "rng",
    # utilities
    "enforce_non_crossing",
    "pit_from_tri_quantiles",
    "reliability_points",
    "reliability_ci",
    "load_preds",
    "label_points",
    # convenience for callers who want to import these Matplotlib types
    "Line2D",
    "Patch",
]

if __name__ == "__main__":
    # quick sanity check
    print("[results_config] DATA_ROOT:", DATA_ROOT)
    print("[results_config] METRICS:", METRICS_PATH.exists())
    print("[results_config] PRED_WIND:", PRED_WIND.exists())
    print("[results_config] OUTDIR:", OUTDIR)
