import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from kdiagram.plot.comparison import plot_polar_reliability
from kdiagram.plot.feature_based import plot_feature_interaction

# Use a non-interactive backend for testing
plt.switch_backend("Agg")

# --- Fixtures for generating test data ---


@pytest.fixture(scope="module")
def reliability_data():
    """Fixture for plot_polar_reliability data."""
    np.random.seed(0)
    n_samples = 500
    y_true = (np.random.rand(n_samples) < 0.4).astype(int)
    # Well-calibrated model
    p1 = np.clip(0.4 + np.random.normal(0, 0.15, n_samples), 0, 1)
    # Over-confident model
    p2 = np.clip(0.4 + np.random.normal(0, 0.3, n_samples), 0, 1)
    return {
        "y_true": y_true,
        "preds": [p1, p2],
        "names": ["Calibrated", "Overconfident"],
    }


@pytest.fixture(scope="module")
def interaction_data():
    """Fixture for plot_feature_interaction data."""
    np.random.seed(1)
    n_points = 1000
    df = pd.DataFrame(
        {
            "hour": np.random.uniform(0, 24, n_points),
            "temp": np.random.uniform(10, 30, n_points),
            "load": np.random.rand(n_points) * 100,
        }
    )
    return df


# --- Tests for plot_polar_reliability (in comparison.py) ---


def test_plot_polar_reliability_runs(reliability_data):
    """Test that plot_polar_reliability runs without errors."""
    ax = plot_polar_reliability(
        reliability_data["y_true"],
        *reliability_data["preds"],
        names=reliability_data["names"],
    )
    assert isinstance(ax, Axes)
    # Check that a legend was created with the correct number of entries
    assert len(ax.get_legend().get_texts()) == 3  # 2 models + 1 perfect calib
    plt.close()


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_plot_polar_reliability_strategies(reliability_data, strategy):
    """Test both binning strategies for polar reliability."""
    ax = plot_polar_reliability(
        reliability_data["y_true"],
        reliability_data["preds"][0],
        strategy=strategy,
        title=f"Strategy: {strategy}",
    )
    assert isinstance(ax, Axes)
    assert f"Strategy: {strategy}" in ax.get_title()
    plt.close()


def test_plot_polar_reliability_warns_on_mismatched_names(reliability_data):
    """Test that a warning is issued for mismatched names."""

    ax = plot_polar_reliability(
        reliability_data["y_true"],
        *reliability_data["preds"],
        names=["Only one name"],  # Mismatched length
    )
    assert isinstance(ax, Axes)
    plt.close()


# --- Tests for plot_feature_interaction (in feature_based.py) ---


def test_plot_feature_interaction_runs(interaction_data):
    """Test that plot_feature_interaction runs without errors."""
    ax = plot_feature_interaction(
        df=interaction_data,
        theta_col="hour",
        r_col="temp",
        color_col="load",
        theta_period=24,
    )
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "hour"
    assert ax.get_ylabel() == "temp"
    plt.close()


@pytest.mark.parametrize("statistic", ["mean", "median", "std"])
def test_plot_feature_interaction_statistics(interaction_data, statistic):
    """Test different aggregation statistics."""
    ax = plot_feature_interaction(
        df=interaction_data,
        theta_col="hour",
        r_col="temp",
        color_col="load",
        statistic=statistic,
        title=f"Statistic: {statistic}",
    )
    assert isinstance(ax, Axes)
    assert f"Statistic: {statistic}" in ax.get_title()
    plt.close()


def test_plot_feature_interaction_raises_error_on_missing_col(
    interaction_data,
):
    """Test that ValueError is raised if a specified column is missing."""
    with pytest.raises(ValueError, match=r"not_a_column"):
        plot_feature_interaction(
            df=interaction_data,
            theta_col="hour",
            r_col="temp",
            color_col="not_a_column",  # This column does not exist
        )
