import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from kdiagram.plot.comparison import plot_horizon_metrics

# Assume the functions are in these locations
from kdiagram.plot.uncertainty import (
    plot_polar_heatmap,
    plot_polar_quiver,
    plot_radial_density_ring,
)

# Use a non-interactive backend for testing
plt.switch_backend("Agg")

# --- Fixtures for generating test data ---


@pytest.fixture
def heatmap_data():
    """Fixture for plot_polar_heatmap and plot_radial_density_ring."""
    np.random.seed(42)
    n_points = 500
    hour = np.random.uniform(0, 24, n_points)
    value = np.random.gamma(2, 5, n_points) + (hour > 12) * 5
    df = pd.DataFrame(
        {
            "hour": hour,
            "value": value,
            "width_start": value,
            "width_end": value + np.random.uniform(5, 10, n_points),
            "velocity_start": value,
            "velocity_end": value + np.random.normal(0, 5, n_points),
        }
    )
    return df


@pytest.fixture
def quiver_data():
    """Fixture for plot_polar_quiver."""
    np.random.seed(0)
    n_points = 20
    df = pd.DataFrame(
        {
            "angle": np.linspace(0, 360, n_points, endpoint=False),
            "radius": 10 + 5 * np.sin(np.deg2rad(np.linspace(0, 1080, n_points))),
            "radial_change": np.random.normal(0, 1, n_points),
            "tangential_change": np.random.normal(0, 0.1, n_points),
            "magnitude": np.random.rand(n_points) * 5,
        }
    )
    return df


@pytest.fixture
def horizon_data():
    """Fixture for plot_horizon_metrics."""
    np.random.seed(1)
    horizons = ["H+1", "H+2", "H+3", "H+4"]
    df = pd.DataFrame(
        {
            "q10_s1": [1, 2, 3, 4],
            "q10_s2": [1.2, 2.3, 3.4, 4.5],
            "q90_s1": [3, 4, 5.5, 7],
            "q90_s2": [3.1, 4.2, 5.7, 7.3],
            "q50_s1": [2, 3, 4.2, 5.7],
            "q50_s2": [2.1, 3.2, 4.4, 5.9],
        },
        index=horizons,
    )
    return df


# --- Tests for plot_polar_heatmap ---


def test_plot_polar_heatmap_runs_successfully(heatmap_data):
    """Test that plot_polar_heatmap runs without errors."""
    ax = plot_polar_heatmap(
        df=heatmap_data, r_col="value", theta_col="hour", theta_period=24
    )
    assert isinstance(ax, Axes)
    plt.close()


@pytest.mark.parametrize(
    "mask_angle, mask_radius", [(True, False), (False, True), (True, True)]
)
def test_plot_polar_heatmap_masking(heatmap_data, mask_angle, mask_radius):
    """Test masking parameters for heatmap."""
    ax = plot_polar_heatmap(
        df=heatmap_data,
        r_col="value",
        theta_col="hour",
        mask_angle=mask_angle,
        mask_radius=mask_radius,
    )
    if mask_angle:
        assert all(label.get_text() == "" for label in ax.get_xticklabels())
    if mask_radius:
        assert all(label.get_text() == "" for label in ax.get_yticklabels())
    plt.close()


# --- Tests for plot_radial_density_ring ---


@pytest.mark.parametrize(
    "kind, cols",
    [
        ("direct", "value"),
        ("width", ["width_start", "width_end"]),
        ("velocity", ["velocity_start", "velocity_end"]),
    ],
)
def test_plot_radial_density_ring_runs_successfully(heatmap_data, kind, cols):
    """Test that plot_radial_density_ring runs for all kinds."""
    ax = plot_radial_density_ring(df=heatmap_data, kind=kind, target_cols=cols)
    assert isinstance(ax, Axes)
    plt.close()


def test_plot_radial_density_ring_raises_error(heatmap_data):
    """Test that plot_radial_density_ring raises ValueError for wrong column counts."""
    with pytest.raises(ValueError):
        plot_radial_density_ring(
            df=heatmap_data, kind="direct", target_cols=["value", "hour"]
        )
    with pytest.raises(ValueError):
        plot_radial_density_ring(df=heatmap_data, kind="width", target_cols=["value"])


# --- Tests for plot_polar_quiver ---


def test_plot_polar_quiver_runs_successfully(quiver_data):
    """Test that plot_polar_quiver runs without errors."""
    ax = plot_polar_quiver(
        df=quiver_data,
        r_col="radius",
        theta_col="angle",
        u_col="radial_change",
        v_col="tangential_change",
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_plot_polar_quiver_with_color_col(quiver_data):
    """Test the color_col parameter for quiver plot."""
    ax = plot_polar_quiver(
        df=quiver_data,
        r_col="radius",
        theta_col="angle",
        u_col="radial_change",
        v_col="tangential_change",
        color_col="magnitude",
        scale=30,  # Test pass-through kwarg
    )
    assert isinstance(ax, Axes)
    assert len(ax.figure.axes) > 1  # Check for colorbar
    plt.close()


def test_plot_horizon_metrics_runs_successfully(horizon_data):
    """Test that plot_horizon_metrics runs without errors."""
    ax = plot_horizon_metrics(
        df=horizon_data,
        qlow_cols=["q10_s1", "q10_s2"],
        qup_cols=["q90_s1", "q90_s2"],
        q50_cols=["q50_s1", "q50_s2"],
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_plot_horizon_metrics_without_q50(horizon_data):
    """Test plot_horizon_metrics without optional q50_cols."""
    ax = plot_horizon_metrics(
        df=horizon_data,
        qlow_cols=["q10_s1", "q10_s2"],
        qup_cols=["q90_s1", "q90_s2"],
        xtick_labels=horizon_data.index.tolist(),
        show_value_labels=True,
    )
    assert isinstance(ax, Axes)
    # Check if value labels were created
    assert len(ax.texts) > 0
    plt.close()


def test_plot_horizon_metrics_raises_error_for_mismatched_cols(horizon_data):
    """Test that ValueError is raised for mismatched column list lengths."""
    with pytest.raises(ValueError):
        plot_horizon_metrics(
            df=horizon_data,
            qlow_cols=["q10_s1"],  # Mismatched length
            qup_cols=["q90_s1", "q90_s2"],
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
