import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from kdiagram.plot.feature_based import (
    plot_feature_fingerprint,
    plot_feature_interaction,
)
from kdiagram.utils.plot import resolve_polar_span

matplotlib.use("Agg")

ACOV_VALUES = [
    "default",
    "full",
    "full-circle",
    "full_circle",
    "half_circle",
    "half",
    "half-circle",
    "quarter_circle",
    "quarter",
    "quarter-circle",
    "eighth_circle",
    "eighth",
    "eighth-circle",
]


@pytest.mark.parametrize("acov", ACOV_VALUES)
def test_plot_feature_fingerprint_span_respects_acov(acov, tmp_path):
    # 3 layers x 5 features importance matrix
    imp = np.array(
        [
            [0.2, 0.8, 0.4, 0.9, 0.3],
            [0.1, 0.3, 0.7, 0.2, 0.5],
            [0.6, 0.5, 0.2, 0.4, 0.8],
        ]
    )
    features = [f"f{i}" for i in range(1, 6)]
    labels = ["L1", "L2", "L3"]

    if acov not in ("half", "half-circle", "half_circle"):
        with pytest.warns(UserWarning):
            ax = plot_feature_fingerprint(
                imp,
                features=features,
                labels=labels,
                normalize=True,
                fill=True,
                cmap="tab10",
                title="fingerprint",
                figsize=(6, 6),
                show_grid=True,
                savefig=str(tmp_path / f"fingerprint_{acov}.png"),
                acov=acov,
            )
    else:
        ax = plot_feature_fingerprint(
            imp,
            features=features,
            labels=labels,
            normalize=True,
            fill=True,
            cmap="tab10",
            title="fingerprint",
            figsize=(6, 6),
            show_grid=True,
            savefig=str(tmp_path / f"fingerprint_{acov}.png"),
            acov=acov,
        )

    assert ax is not None
    # Compare angular span in degrees
    expected_deg = np.degrees(resolve_polar_span(acov))
    assert np.isclose(ax.get_thetamax(), expected_deg, atol=1e-6)


def _make_interaction_df(n=400, seed=7):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 100.0, n)  # arbitrary "angle-like" feature
    rvals = rng.uniform(1.0, 10.0, size=n)  # positive radial variable
    color = np.sin(theta / 10.0) + 0.2 * rng.normal(size=n)
    return pd.DataFrame(
        {"theta_feat": theta, "r_feat": rvals, "signal": color}
    )


@pytest.mark.parametrize("acov", ACOV_VALUES)
def test_plot_feature_interaction_span_respects_acov(acov, tmp_path):
    df = _make_interaction_df()

    if acov not in ("full", "full-circle", "default", "full_circle"):
        with pytest.warns(UserWarning):
            ax = plot_feature_interaction(
                df=df,
                theta_col="theta_feat",
                r_col="r_feat",
                color_col="signal",
                statistic="mean",
                theta_bins=24,
                r_bins=10,
                title="interaction",
                figsize=(6, 6),
                cmap="viridis",
                show_grid=True,
                savefig=str(tmp_path / f"interaction_{acov}.png"),
                acov=acov,
            )
    else:
        ax = plot_feature_interaction(
            df=df,
            theta_col="theta_feat",
            r_col="r_feat",
            color_col="signal",
            statistic="mean",
            theta_bins=24,
            r_bins=10,
            title="interaction",
            figsize=(6, 6),
            cmap="viridis",
            show_grid=True,
            savefig=str(tmp_path / f"interaction_{acov}.png"),
            acov=acov,
        )

    assert ax is not None
    expected_deg = np.degrees(resolve_polar_span(acov))
    assert np.isclose(ax.get_thetamax(), expected_deg, atol=1e-6)


def test_plot_feature_fingerprint_uses_external_axes_and_sets_span(tmp_path):
    imp = np.array([[1, 2, 3], [2, 1, 0.5]])
    features = ["a", "b", "c"]
    labels = ["L1", "L2"]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))

    with pytest.warns(UserWarning):
        out_ax = plot_feature_fingerprint(
            imp,
            features=features,
            labels=labels,
            normalize=True,
            fill=True,
            cmap="tab10",
            title="fp-ext-ax",
            show_grid=False,
            savefig=str(tmp_path / "fp_ext_ax.png"),
            acov="quarter",  # alias should work
            ax=ax,  # external axes
        )
    assert out_ax is ax  # same axes object
    expected_deg = np.degrees(resolve_polar_span("quarter"))
    assert np.isclose(ax.get_thetamax(), expected_deg, atol=1e-6)


def test_plot_feature_interaction_uses_external_axes_and_sets_span(tmp_path):
    df = _make_interaction_df()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))

    with pytest.warns(UserWarning):
        out_ax = plot_feature_interaction(
            df=df,
            theta_col="theta_feat",
            r_col="r_feat",
            color_col="signal",
            statistic="mean",
            theta_bins=12,
            r_bins=6,
            title="int-ext-ax",
            cmap="plasma",
            show_grid=False,
            savefig=str(tmp_path / "int_ext_ax.png"),
            acov="eighth_circle",
            ax=ax,  # external axes
        )
    assert out_ax is ax
    expected_deg = np.degrees(resolve_polar_span("eighth_circle"))
    assert np.isclose(ax.get_thetamax(), expected_deg, atol=1e-6)
