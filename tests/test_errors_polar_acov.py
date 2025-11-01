from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from kdiagram.plot.errors import (
    plot_error_bands,
    plot_error_ellipses,
    plot_error_violins,
)

ACOV_DEG = {
    "default": 360.0,
    "half_circle": 180.0,
    "quarter_circle": 90.0,
    "eighth_circle": 45.0,
}

VIOLIN_MODE = "basic"


def _is_polar(ax) -> bool:
    return getattr(ax, "name", None) == "polar"


def _get_thetamax(ax) -> float:
    if hasattr(ax, "get_thetamax"):
        return float(ax.get_thetamax())
    return float(getattr(ax, "thetamax", 360.0))


def _close(ax):
    plt.close(ax.figure)


def _df_for_ellipses(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    r = rng.uniform(2.0, 8.0, size=n)
    theta_deg = rng.uniform(0.0, 360.0, size=n)
    r_std = rng.uniform(0.2, 0.9, size=n)
    theta_std = rng.uniform(0.05, 0.25, size=n)  # radians
    c = rng.normal(0.0, 1.0, size=n)
    return pd.DataFrame(
        {
            "r": r,
            "theta": theta_deg,
            "r_std": r_std,
            "theta_std": theta_std,
            "c": c,
        }
    )


def _df_for_bands(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    theta_deg = rng.uniform(0.0, 360.0, size=n)
    err = rng.normal(
        loc=0.0,
        scale=0.8 + 0.4 * np.sin(np.deg2rad(theta_deg)) ** 2,
        size=n,
    )
    return pd.DataFrame({"err": err, "theta": theta_deg})


def _df_for_violins(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    e1 = rng.normal(0.0, 1.0, size=n)
    e2 = rng.laplace(0.0, 0.8, size=n)
    # FIX: Generator.standard_t for Student-t variates (heavier tails)
    e3 = rng.standard_t(df=5, size=n) * 0.7
    return pd.DataFrame({"e1": e1, "e2": e2, "e3": e3})


@pytest.mark.parametrize("acov,deg", ACOV_DEG.items())
def test_plot_error_ellipses_respects_acov_and_draws_patches(acov, deg):
    df = _df_for_ellipses(n=10)
    ax = plot_error_ellipses(
        df=df,
        r_col="r",
        theta_col="theta",
        r_std_col="r_std",
        theta_std_col="theta_std",
        color_col="c",
        n_std=2.0,
        acov=acov,
        show_grid=False,
    )
    assert ax is not None
    assert _is_polar(ax)
    got = _get_thetamax(ax)
    assert np.isclose(got, deg, atol=0.5)
    assert len(ax.patches) >= len(df)
    assert len(ax.figure.axes) >= 2
    _close(ax)


@pytest.mark.parametrize("acov,deg", ACOV_DEG.items())
def test_plot_error_bands_respects_acov_and_fills_band(acov, deg):
    df = _df_for_bands()
    ax = plot_error_bands(
        df=df,
        error_col="err",
        theta_col="theta",
        theta_period=360.0,
        theta_bins=32,
        n_std=1.5,
        acov=acov,
        show_grid=False,
    )
    assert ax is not None
    assert _is_polar(ax)
    got = _get_thetamax(ax)
    assert np.isclose(got, deg, atol=0.5)
    assert len(ax.lines) >= 2
    assert any(
        c.__class__.__name__.endswith("Collection") for c in ax.collections
    )
    _close(ax)


def test_plot_error_bands_masks_angle_ticks():
    df = _df_for_bands()
    ax = plot_error_bands(
        df=df,
        error_col="err",
        theta_col="theta",
        theta_period=360.0,
        mask_angle=True,
        show_grid=False,
    )
    assert ax is not None
    xtlbl = [t.get_text() for t in ax.get_xticklabels()]
    assert all(lbl == "" for lbl in xtlbl)
    _close(ax)


@pytest.mark.parametrize("acov,deg", ACOV_DEG.items())
def test_plot_error_violins_respects_acov_and_builds_violins(acov, deg):
    df = _df_for_violins()
    names = ["A", "B", "C"]
    ax = plot_error_violins(
        df,
        *["e1", "e2", "e3"],
        names=names,
        acov=acov,
        mode=VIOLIN_MODE,
        show_grid=False,
    )
    assert ax is not None
    assert _is_polar(ax)
    got = _get_thetamax(ax)
    assert np.isclose(got, deg, atol=0.5)
    assert len(ax.patches) >= 3
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == names
    _close(ax)
