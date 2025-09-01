from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

# Headless for CI
matplotlib.use("Agg")

from kdiagram.plot.probabilistic import (
    plot_crps_comparison,
    plot_pit_histogram,
    plot_polar_sharpness,
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


def _synth_quantile_data(
    n: int = 300,
    qs=(0.1, 0.5, 0.9),
    spreads=(0.8, 1.2),
):
    rng = np.random.default_rng(123)
    y_true = rng.normal(0.0, 1.0, size=n)

    q = np.array(qs, dtype=float)
    # z-scores approx for 10/50/90 under normal
    z = np.array([-1.28155, 0.0, 1.28155], dtype=float)

    # one shared model mean with some error
    mu = y_true + rng.normal(0.0, 0.9, size=n)

    preds = []
    for s in spreads:
        Q = np.stack([mu + z_i * s for z_i in z], axis=1)
        preds.append(Q)

    return y_true, preds, q


def test_pit_histogram_respects_acov_and_bars():
    y_true, preds_list, q = _synth_quantile_data(
        n=400,
        spreads=(0.9,),  # single model for PIT
    )
    yq = preds_list[0]

    for acov, deg in ACOV_DEG.items():
        ax = plot_pit_histogram(
            y_true=y_true,
            y_preds_quantiles=yq,
            quantiles=q,
            acov=acov,
            n_bins=16,
            title="PIT",
            color="#1f77b4",
            edgecolor="black",
            show_uniform_line=True,
            show_grid=False,
        )
        assert ax is not None
        assert _is_polar(ax)
        got = _get_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)

        # Bars should exist (polar bar adds Rectangle patches)
        assert len(ax.patches) >= 16

        # Reference uniform line present (look for dashed line)
        has_dashed = any(
            getattr(lo, "get_linestyle", lambda: "")() == "--"
            for lo in ax.lines
        )
        assert has_dashed

        _close(ax)


def test_crps_comparison_respects_acov_and_scatter():
    y_true, preds_list, q = _synth_quantile_data(
        n=250,
        spreads=(0.7, 1.4),
    )

    for acov, deg in ACOV_DEG.items():
        ax = plot_crps_comparison(
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

        # scatter -> PathCollection in ax.collections
        assert any(isinstance(c, PathCollection) for c in ax.collections)

        _close(ax)


def test_polar_sharpness_respects_acov_and_scatter():
    _, preds_list, q = _synth_quantile_data(
        n=220,
        spreads=(0.6, 1.3, 1.8),
    )

    for acov, deg in ACOV_DEG.items():
        ax = plot_polar_sharpness(
            *preds_list,
            quantiles=q,
            names=["M1", "M2", "M3"],
            acov=acov,
            cmap="plasma",
            show_grid=False,
        )
        assert ax is not None
        assert _is_polar(ax)
        got = _get_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)

        # scatter -> PathCollection present
        assert any(isinstance(c, PathCollection) for c in ax.collections)

        # Text labels for each model
        assert len(ax.texts) >= 3

        _close(ax)
