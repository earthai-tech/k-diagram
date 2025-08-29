import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from scipy.stats import norm

from kdiagram.plot.probabilistic import (
    plot_calibration_sharpness,
    plot_credibility_bands,
    plot_crps_comparison,
    plot_pit_histogram,
    plot_polar_sharpness,
)

# Use a non-interactive backend for testing
plt.switch_backend("Agg")


@pytest.fixture(scope="module")
def probabilistic_data():
    """
    Generates a consistent set of probabilistic forecasts for testing.
    Includes a good model, an overconfident (too sharp) model, and an
    underconfident (not sharp) model.
    """
    np.random.seed(42)
    n_samples = 500
    y_true = np.random.normal(loc=10, scale=5, size=n_samples)
    quantiles = np.linspace(
        0.05, 0.95, 19
    )  # 90% interval covered by 19 quantiles

    # Model 1: Good model
    good_preds = norm.ppf(quantiles, loc=y_true[:, np.newaxis], scale=5)
    # Model 2: Overconfident model
    overconfident_preds = norm.ppf(
        quantiles, loc=y_true[:, np.newaxis], scale=2.5
    )
    # Model 3: Underconfident model
    underconfident_preds = norm.ppf(
        quantiles, loc=y_true[:, np.newaxis] + 2, scale=8
    )

    return {
        "y_true": y_true,
        "quantiles": quantiles,
        "preds": [good_preds, overconfident_preds, underconfident_preds],
        "names": ["Good", "Overconfident", "Underconfident"],
    }


# --- Tests for plot_pit_histogram ---


def test_plot_pit_histogram_runs(probabilistic_data):
    """Test that plot_pit_histogram runs without errors."""
    ax = plot_pit_histogram(
        probabilistic_data["y_true"],
        probabilistic_data["preds"][0],  # Good model
        probabilistic_data["quantiles"],
        title="Test PIT",
    )
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Test PIT"
    plt.close()


def test_plot_pit_histogram_raises_error_on_shape_mismatch(
    probabilistic_data,
):
    """Test for ValueError with mismatched quantile and prediction shapes."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        plot_pit_histogram(
            probabilistic_data["y_true"],
            probabilistic_data["preds"][0],
            probabilistic_data["quantiles"][:-1],  # Mismatched length
        )


# --- Tests for plot_polar_sharpness ---


def test_plot_polar_sharpness_runs(probabilistic_data):
    """Test that plot_polar_sharpness runs with multiple models."""
    ax = plot_polar_sharpness(
        *probabilistic_data["preds"],
        quantiles=probabilistic_data["quantiles"],
        names=probabilistic_data["names"],
    )
    assert isinstance(ax, Axes)
    # Check that three points were plotted
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape[0] == 3
    plt.close()


# --- Tests for plot_crps_comparison ---


def test_plot_crps_comparison_runs(probabilistic_data):
    """Test that plot_crps_comparison runs and returns an Axes object."""
    ax = plot_crps_comparison(
        probabilistic_data["y_true"],
        *probabilistic_data["preds"],
        quantiles=probabilistic_data["quantiles"],
        names=probabilistic_data["names"],
    )
    assert isinstance(ax, Axes)
    plt.close()


# --- Tests for plot_credibility_bands ---


def test_plot_credibility_bands_runs(probabilistic_data):
    """Test that plot_credibility_bands runs without errors."""
    df = pd.DataFrame(
        {
            "q10": probabilistic_data["preds"][0][:, 0],  # Using good model
            "q50": probabilistic_data["preds"][0][:, 9],
            "q90": probabilistic_data["preds"][0][:, -1],
            "feature": np.tile(np.arange(50), 10),
        }
    )
    ax = plot_credibility_bands(
        df=df,
        q_cols=("q10", "q50", "q90"),
        theta_col="feature",
        theta_bins=10,
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_plot_credibility_bands_raises_error_on_wrong_qcols():
    """Test ValueError for incorrect number of q_cols."""
    df = pd.DataFrame({"q10": [1], "q50": [2], "q90": [3], "f": [1]})
    with pytest.raises(ValueError, match="must be a tuple of three"):
        plot_credibility_bands(df=df, q_cols=("q10", "q90"), theta_col="f")


# --- Tests for plot_calibration_sharpness ---


def test_plot_calibration_sharpness_runs(probabilistic_data):
    """Test that plot_calibration_sharpness runs with multiple models."""
    ax = plot_calibration_sharpness(
        probabilistic_data["y_true"],
        *probabilistic_data["preds"],
        quantiles=probabilistic_data["quantiles"],
        names=probabilistic_data["names"],
    )
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Calibration Error (Lower is Better)"
    assert ax.get_ylabel() == "Sharpness (Lower is Better)"
    plt.close()


def test_mask_radius_param(probabilistic_data):
    """Test that the mask_radius parameter works on a sample plot."""
    ax = plot_calibration_sharpness(
        probabilistic_data["y_true"],
        probabilistic_data["preds"][0],
        quantiles=probabilistic_data["quantiles"],
        mask_radius=True,
    )
    # Check that radial tick labels are empty
    assert all(label.get_text() == "" for label in ax.get_yticklabels())
    plt.close()
