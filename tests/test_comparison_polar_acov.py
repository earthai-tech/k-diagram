from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Headless backend for CI
matplotlib.use("Agg")

from kdiagram.plot.comparison import (
    plot_model_comparison,
    plot_polar_reliability,
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


def _synth_regression(n: int = 200):
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    y_true = 3.0 * x + rng.normal(scale=0.6, size=n)
    y_pred_a = 3.1 * x + rng.normal(scale=0.7, size=n)
    y_pred_b = 2.7 * x + rng.normal(scale=0.8, size=n)
    return y_true, [y_pred_a, y_pred_b]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _synth_reliability(n: int = 800):
    rng = np.random.default_rng(123)
    x = rng.normal(size=n)
    # latent probability & outcome
    p_true = _sigmoid(0.8 * x)
    y_true = rng.binomial(1, p_true, size=n)
    # two probabilistic models
    p1 = _sigmoid(0.75 * x + rng.normal(scale=0.2, size=n))  # decent
    p2 = _sigmoid(1.10 * x + 0.2)  # over-confident, biased
    return y_true, [p1, p2]


def test_model_comparison_respects_acov_and_warns():
    y_true, preds = _synth_regression()
    metrics = ["r2", "mae", "rmse"]

    # Non-default acov should warn once per call
    for acov, deg in [
        ("half_circle", 180.0),
        ("quarter_circle", 90.0),
        ("eighth_circle", 45.0),
    ]:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ax = plot_model_comparison(
                y_true,
                *preds,
                metrics=metrics,
                names=["A", "B"],
                acov=acov,
                scale="norm",
                legend=False,
                show_grid=False,
            )
        assert ax is not None
        assert _is_polar(ax)
        got = _get_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)
        # at least one warning mentioning acov
        msgs = [str(w.message).lower() for w in rec]
        assert any("acov" in m for m in msgs)
        # at least as many model line polygons as models
        assert len(ax.lines) >= 2
        _close(ax)


def test_model_comparison_default_acov_no_warn():
    y_true, preds = _synth_regression()
    metrics = ["r2", "mae", "rmse"]

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ax = plot_model_comparison(
            y_true,
            *preds,
            metrics=metrics,
            names=["A", "B"],
            acov="default",
            scale="norm",
        )
    assert ax is not None
    assert _is_polar(ax)
    got = _get_thetamax(ax)
    assert np.isclose(got, 360.0, atol=0.5)
    msgs = [str(w.message).lower() for w in rec]
    assert not any("acov" in m for m in msgs)
    _close(ax)


def test_polar_reliability_respects_acov_and_has_collections():
    y_true, p_list = _synth_reliability()

    for acov, deg in ACOV_DEG.items():
        ax = plot_polar_reliability(
            y_true,
            *p_list,
            names=["Calib", "Overconf"],
            n_bins=12,
            strategy="uniform",
            acov=acov,
            cmap="coolwarm",
            show_grid=False,
            show_cbar=False,  # avoid extra axes in tests
        )
        assert ax is not None
        assert _is_polar(ax)
        got = _get_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)
        # should have at least one LineCollection (colored spiral)
        assert any(isinstance(c, LineCollection) for c in ax.collections)
        # perfect calibration line is drawn
        assert len(ax.lines) >= 1
        _close(ax)
