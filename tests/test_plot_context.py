import builtins
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import kdiagram.utils._deps as deps
from kdiagram.plot.context import (
    plot_error_autocorrelation,
    plot_error_distribution,
    plot_error_pacf,
    plot_qq,
    plot_scatter_correlation,
    plot_time_series,
)

_real_import = builtins.__import__


# Use a non-interactive backend for testing
plt.switch_backend("Agg")


# Note: scope="function" is the default for pytest fixtures and could be omitted.
# It is included here explicitly for clarity, emphasizing that this fixture
# is re-created for each test to ensure proper isolation and prevent
# state from leaking between them (test pollution).
@pytest.fixture(scope="function")
def context_data():
    # ... rest of your fixture code
    """Provides a consistent DataFrame for all context plot tests."""
    np.random.seed(0)
    n_samples = 100

    # FIX: Construct the DatetimeIndex manually to avoid the internal
    # pandas function that is being affected by test pollution.
    start_date = pd.Timestamp("2023-01-01")
    time_index = pd.to_datetime(
        [start_date + pd.Timedelta(days=i) for i in range(n_samples)]
    )

    y_true = (
        50 + np.linspace(0, 20, n_samples) + np.random.normal(0, 2, n_samples)
    )

    # Model 1: Good prediction
    y_pred1 = y_true + np.random.normal(0, 1.5, n_samples)
    # Model 2: Biased prediction
    y_pred2 = y_true * 0.9 + 5

    df = pd.DataFrame(
        {
            "time": time_index,
            "actual": y_true,
            "pred_good": y_pred1,
            "pred_biased": y_pred2,
            "q10": y_pred1 - 5,
            "q90": y_pred1 + 5,
        }
    )
    return df


# --- Tests for plot_time_series ---


def test_plot_time_series_runs(context_data):
    """Test basic execution of plot_time_series."""
    ax = plot_time_series(
        df=context_data,
        x_col="time",
        actual_col="actual",
        pred_cols=["pred_good", "pred_biased"],
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_plot_time_series_with_bands(context_data):
    """Test plot_time_series with uncertainty bands."""
    ax = plot_time_series(
        df=context_data,
        x_col="time",
        actual_col="actual",
        pred_cols=["pred_good"],
        q_lower_col="q10",
        q_upper_col="q90",
    )
    assert isinstance(ax, Axes)
    # Check that a fill_between collection was created
    assert len(ax.collections) > 0
    plt.close()


# --- Tests for plot_scatter_correlation ---


def test_plot_scatter_correlation_runs(context_data):
    """Test basic execution of plot_scatter_correlation."""
    ax = plot_scatter_correlation(
        df=context_data,
        actual_col="actual",
        pred_cols=["pred_good", "pred_biased"],
    )
    assert isinstance(ax, Axes)
    # Check for identity line + 2 scatter plots
    assert len(ax.lines) == 1
    assert len(ax.collections) == 2
    plt.close()


# --- Tests for plot_error_autocorrelation ---


def test_plot_error_autocorrelation_runs(context_data):
    """Test basic execution of plot_error_autocorrelation."""
    ax = plot_error_autocorrelation(
        df=context_data, actual_col="actual", pred_col="pred_good"
    )
    assert isinstance(ax, Axes)
    plt.close()


# --- Tests for plot_error_pacf ---
def test_plot_error_pacf_runs(context_data):
    """Test basic execution of plot_error_pacf."""
    # This test now receives a clean, 100-sample DataFrame
    # and will no longer raise the ValueError from statsmodels.
    try:
        import statsmodels  # noqa

        ax = plot_error_pacf(
            df=context_data,
            actual_col="actual",
            pred_col="pred_biased",
        )
        assert isinstance(ax, Axes)
        plt.close()
    except ImportError:
        pytest.skip("statsmodels not installed, skipping PACF test.")


def _blocked_import(name, *args, **kwargs):
    if name == "statsmodels" or name.startswith("statsmodels."):
        raise ImportError("statsmodels is required")
    return _real_import(name, *args, **kwargs)


def test_plot_error_pacf_raises_import_error_if_not_installed():
    # ensure ensure_pkg re-checks
    deps._REQUIREMENT_CACHE.clear()

    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10)})

    # Don't poison sys.modules with None; just block imports
    with patch("builtins.__import__", side_effect=_blocked_import):
        with pytest.raises(ImportError, match=r"statsmodels is required"):
            plot_error_pacf(df, "a", "b")


# --- Tests for plot_qq ---


def test_plot_qq_runs(context_data):
    """Test basic execution of plot_qq."""
    ax = plot_qq(df=context_data, actual_col="actual", pred_col="pred_good")
    assert isinstance(ax, Axes)
    # Q-Q plot creates 2 lines: the data points and the reference line
    assert len(ax.get_lines()) == 2
    plt.close()


# --- Tests for plot_error_distribution ---
def test_plot_error_distribution_runs(context_data):
    """Test basic execution of plot_error_distribution."""
    ax = plot_error_distribution(
        df=context_data, actual_col="actual", pred_col="pred_biased"
    )
    assert isinstance(ax, Axes)
    assert "Distribution of Forecast Errors" in ax.get_title()
    plt.close()


def test_plot_error_distribution_passes_kwargs(context_data):
    """Test that kwargs are passed to the underlying hist_kde plot."""
    ax = plot_error_distribution(
        df=context_data,
        actual_col="actual",
        pred_col="pred_good",
        bins=15,
        kde_color="red",  # Pass a kwarg
    )
    assert isinstance(ax, Axes)
    # A simple check: the legend will have a line artist for the KDE
    assert len(ax.get_legend().get_lines()) > 0
    plt.close()
