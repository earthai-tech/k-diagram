from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection, PolyCollection

# Use a headless backend for CI
matplotlib.use("Agg")

from kdiagram.plot.probabilistic import (
    plot_calibration_sharpness,
    plot_credibility_bands,
)

ACOV_DEG = {
    "default": 360.0,
    "half_circle": 180.0,
    "quarter_circle": 90.0,
    "eighth_circle": 45.0,
}


def _is_polar(ax) -> bool:
    return getattr(ax, "name", None) == "polar"


def _get_thetamax(ax) -> float:
    if hasattr(ax, "get_thetamax"):
        return float(ax.get_thetamax())
    return float(getattr(ax, "thetamax", 360.0))


def _close(ax):
    plt.close(ax.figure)


def _synth_df_cred(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    theta = rng.uniform(0.0, 24.0, size=n)  # arbitrary units
    base = rng.normal(10.0, 2.0, size=n)
    width = np.abs(rng.normal(2.0, 0.8, size=n))
    q_med = base
    q_low = q_med - 0.5 * width - np.abs(rng.normal(0, 0.2, n))
    q_up = q_med + 0.5 * width + np.abs(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "q_low": q_low,
            "q_med": q_med,
            "q_up": q_up,
            "theta": theta,
        }
    )


def _synth_quantile_models(
    n: int = 200, q=(0.1, 0.5, 0.9)
) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(7)
    y_true = rng.normal(0.0, 1.0, size=n)

    # Build two models with different spread
    sigma_mu = 0.8
    z = np.array([-1.28155, 0.0, 1.28155])  # 10/50/90 normal
    mu = y_true + rng.normal(0.0, sigma_mu, size=n)

    preds = []
    for s in (0.8, 1.4):
        Q = np.stack([mu + z_i * s for z_i in z], axis=1)
        preds.append(Q)

    return y_true, preds, np.array(q, dtype=float)


def test_credibility_bands_respects_acov():
    df = _synth_df_cred()

    for acov, deg in ACOV_DEG.items():
        ax = plot_credibility_bands(
            df=df,
            q_cols=("q_low", "q_med", "q_up"),
            theta_col="theta",
            acov=acov,
            theta_period=24.0,
            theta_bins=16,
            color="#1f77b4",
            show_grid=False,
        )
        assert ax is not None
        assert _is_polar(ax)
        got = _get_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)

        # median line exists
        assert len(ax.lines) >= 1
        # fill_between produced a PolyCollection
        assert any(isinstance(c, PolyCollection) for c in ax.collections)

        _close(ax)


def test_calibration_sharpness_respects_acov_and_scatter():
    y_true, preds_list, q = _synth_quantile_models()

    for acov, deg in ACOV_DEG.items():
        ax = plot_calibration_sharpness(
            y_true,
            *preds_list,
            quantiles=q,
            names=["A", "B"],
            acov=acov,
            cmap="viridis",
            show_grid=True,
        )
        assert ax is not None
        assert _is_polar(ax)
        got = _get_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)

        # scatter produced a PathCollection
        assert any(isinstance(c, PathCollection) for c in ax.collections)
        _close(ax)
