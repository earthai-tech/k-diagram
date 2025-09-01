from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh

# Use non-interactive backend for tests
matplotlib.use("Agg")

# Adjust import path to match your package layout
from kdiagram.plot.uncertainty import (
    plot_polar_heatmap,
    plot_polar_quiver,
    plot_radial_density_ring,
)

ACOV_TO_DEG = {
    "default": 360.0,
    "half_circle": 180.0,
    "quarter_circle": 90.0,
    "eighth_circle": 45.0,
}


def _expect_thetamax(ax) -> float:
    # PolarAxes has get_thetamax; fall back to private attr.
    if hasattr(ax, "get_thetamax"):
        return float(ax.get_thetamax())
    return float(getattr(ax, "thetamax", 360.0))


def _is_polar(ax) -> bool:
    return getattr(ax, "name", None) == "polar"


def _has_quiver(ax) -> bool:
    # Quiver is a special collection; check by class name.
    names = [c.__class__.__name__ for c in ax.collections]
    names += [a.__class__.__name__ for a in ax.artists]
    return any(n == "Quiver" for n in names)


def _has_quadmesh(ax) -> bool:
    return any(isinstance(c, QuadMesh) for c in ax.collections)


def _rand_df_for_heatmap(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    r = np.abs(rng.normal(loc=2.0, scale=0.7, size=n))
    # Pretend theta is "hours" in [0, 24)
    th = rng.uniform(0.0, 24.0, size=n)
    return pd.DataFrame({"r": r, "theta": th})


def _rand_df_for_quiver(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    r = np.abs(rng.normal(2.0, 0.6, n))
    th = rng.uniform(0.0, 100.0, n)
    # small vector field
    u = rng.normal(0.0, 0.2, n)  # radial
    v = rng.normal(0.0, 0.2, n)  # angular
    mag = np.hypot(u, v)
    return pd.DataFrame({"r": r, "theta": th, "u": u, "v": v, "mag": mag})


def _rand_df_for_ring(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.2, n)
    y = x + rng.normal(0.5, 0.6, n)  # wider than x on avg
    return pd.DataFrame({"x": x, "y": y})


def test_heatmap_respects_acov_param():
    df = _rand_df_for_heatmap()
    for acov, deg in ACOV_TO_DEG.items():
        ax = plot_polar_heatmap(
            df=df,
            r_col="r",
            theta_col="theta",
            acov=acov,
            theta_period=24.0,
            r_bins=10,
            theta_bins=12,
            cmap="viridis",
            show_grid=False,
        )
        assert ax is not None
        assert _is_polar(ax)
        got = _expect_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)
        assert _has_quadmesh(ax)
        plt.close(ax.figure)


def test_quiver_respects_acov_param():
    df = _rand_df_for_quiver()
    for acov, deg in ACOV_TO_DEG.items():
        ax = plot_polar_quiver(
            df=df,
            r_col="r",
            theta_col="theta",
            u_col="u",
            v_col="v",
            color_col="mag",
            acov=acov,
            theta_period=None,
            cmap="plasma",
            show_grid=True,
            headwidth=3.0,
            headlength=4.0,
            headaxislength=3.5,
            pivot="middle",
            scale_units="xy",
            scale=10.0,
        )
        assert ax is not None
        assert _is_polar(ax)
        got = _expect_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)
        assert _has_quiver(ax)
        plt.close(ax.figure)


def test_ring_warns_on_non_default_acov_and_respects_span():
    df = _rand_df_for_ring()

    # Non-default acov should warn once per call
    for acov, deg in [
        ("half_circle", 180.0),
        ("quarter_circle", 90.0),
        ("eighth_circle", 45.0),
    ]:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ax = plot_radial_density_ring(
                df=df,
                kind="width",
                target_cols=["x", "y"],
                acov=acov,
                cmap="magma",
                show_grid=False,
                alpha=0.9,
            )
        assert ax is not None
        assert _is_polar(ax)
        got = _expect_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)
        # must have a QuadMesh from pcolormesh
        assert _has_quadmesh(ax)
        # at least one UserWarning mentioning acov
        msgs = [str(w.message) for w in rec]
        assert any("acov" in m.lower() for m in msgs)
        plt.close(ax.figure)

    # Default acov: no warning expected
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ax = plot_radial_density_ring(
            df=df,
            kind="direct",
            target_cols="x",
            acov="default",
            cmap="viridis",
            show_grid=True,
        )
    assert ax is not None
    assert _is_polar(ax)
    got = _expect_thetamax(ax)
    assert np.isclose(got, 360.0, atol=0.5)
    # no acov warning
    msgs = [str(w.message) for w in rec]
    assert not any("acov" in m.lower() for m in msgs)
    assert _has_quadmesh(ax)
    plt.close(ax.figure)
