import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from kdiagram.plot.relationship import (
    plot_conditional_quantiles,
    plot_error_relationship,
    plot_residual_relationship,
)

# Use a non-interactive backend for testing
plt.switch_backend("Agg")

# --- Fixture for generating relationship data ---

@pytest.fixture(scope="module")
def relationship_data():
    """
    Generates a consistent set of data for testing relationship plots.
    The data has known heteroscedasticity and conditional bias.
    """
    np.random.seed(0)
    n_samples = 100
    y_true = np.linspace(0, 20, n_samples)**1.5
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    # Model's median forecast has a conditional bias
    y_pred_median = y_true * 0.9 + np.random.normal(0, 1, n_samples)
    
    # Model's uncertainty increases with the true value
    interval_width = 5 + (y_true / y_true.max()) * 15
    
    y_preds_quantiles = np.zeros((n_samples, len(quantiles)))
    y_preds_quantiles[:, 2] = y_pred_median
    y_preds_quantiles[:, 1] = y_pred_median - interval_width * 0.25
    y_preds_quantiles[:, 3] = y_pred_median + interval_width * 0.25
    y_preds_quantiles[:, 0] = y_pred_median - interval_width * 0.5
    y_preds_quantiles[:, 4] = y_pred_median + interval_width * 0.5
    
    return {
        "y_true": y_true,
        "y_pred_median": y_pred_median,
        "y_preds_quantiles": y_preds_quantiles,
        "quantiles": quantiles
    }

# --- Tests for plot_conditional_quantiles ---

def test_plot_conditional_quantiles_runs(relationship_data):
    """Test that plot_conditional_quantiles runs without errors."""
    ax = plot_conditional_quantiles(
        relationship_data["y_true"],
        relationship_data["y_preds_quantiles"],
        relationship_data["quantiles"],
        bands=[80, 50]
    )
    assert isinstance(ax, Axes)
    # Check if a legend was created with the correct number of entries
    assert len(ax.get_legend().get_texts()) == 3 # 2 bands + 1 median
    plt.close()

def test_plot_conditional_quantiles_warns_on_missing_band(relationship_data):
    """Test that a warning is issued if quantiles for a band are missing."""
    with pytest.warns(UserWarning, match="Quantiles for 95% interval not found"):
        ax = plot_conditional_quantiles(
            relationship_data["y_true"],
            relationship_data["y_preds_quantiles"],
            relationship_data["quantiles"],
            bands=[95, 50] # 95% interval requires 0.025 quantile, which is missing
        )
    assert isinstance(ax, Axes)
    plt.close()

# --- Tests for plot_error_relationship ---

def test_plot_error_relationship_runs(relationship_data):
    """Test that plot_error_relationship runs with a single model."""
    ax = plot_error_relationship(
        relationship_data["y_true"],
        relationship_data["y_pred_median"],
        names=["Test Model"]
    )
    assert isinstance(ax, Axes)
    assert ax.get_ylabel() == "Forecast Error (Actual - Predicted)"
    plt.close()

def test_plot_error_relationship_multi_model(relationship_data):
    """Test that plot_error_relationship runs with multiple models."""
    y_pred2 = relationship_data["y_pred_median"] * 1.1
    ax = plot_error_relationship(
        relationship_data["y_true"],
        relationship_data["y_pred_median"],
        y_pred2,
        names=["Model 1", "Model 2"]
    )
    assert isinstance(ax, Axes)
    # Check that two scatter plots were created
    assert len(ax.collections) == 2
    plt.close()

# --- Tests for plot_residual_relationship ---

def test_plot_residual_relationship_runs(relationship_data):
    """Test that plot_residual_relationship runs without errors."""
    ax = plot_residual_relationship(
        relationship_data["y_true"],
        relationship_data["y_pred_median"],
        names=["Test Model"]
    )
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Based on Predicted Value"
    plt.close()

def test_plot_residual_relationship_no_zero_line(relationship_data):
    """Test the show_zero_line=False parameter."""
    ax = plot_residual_relationship(
        relationship_data["y_true"],
        relationship_data["y_pred_median"],
        show_zero_line=False
    )
    assert isinstance(ax, Axes)
    # Check that only the scatter plot was created (no line)
    assert len(ax.lines) == 0
    plt.close()

