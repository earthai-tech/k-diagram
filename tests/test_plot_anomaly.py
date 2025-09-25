import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from kdiagram.plot.anomaly import (
    plot_anomaly_glyphs,
    plot_anomaly_profile,
    plot_anomaly_severity,
    plot_cas_profile,
)

# A list of all functions to be tested
ALL_ANOMALY_PLOTS = [
    plot_anomaly_severity,
    plot_anomaly_profile,
    plot_anomaly_glyphs,
    plot_cas_profile,
]

# ---- Test Data Fixtures ----


@pytest.fixture
def mixed_anomaly_data():
    """
    Generates a DataFrame with a mix of covered points,
    over-predictions, and under-predictions.
    """
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.normal(50, 10, n_samples)
    y_qlow = y_true - 5
    y_qup = y_true + 5

    # Add over-predictions
    over_indices = np.random.choice(n_samples, 15, replace=False)
    y_true[over_indices] += np.random.uniform(6, 10, 15)

    # Add under-predictions
    under_indices = np.random.choice(
        list(set(range(n_samples)) - set(over_indices)),
        10,
        replace=False,
    )
    y_true[under_indices] -= np.random.uniform(6, 10, 10)

    return pd.DataFrame(
        {
            "actual": y_true,
            "q_low": y_qlow,
            "q_up": y_qup,
        }
    )


@pytest.fixture
def no_anomaly_data():
    """Generates data with no anomalies."""
    y_true = np.array([10, 20, 30])
    y_qlow = np.array([8, 18, 28])
    y_qup = np.array([12, 22, 32])
    return pd.DataFrame({"actual": y_true, "q_low": y_qlow, "q_up": y_qup})


# ---- Pytest Functions ----


@pytest.mark.parametrize("plot_func", ALL_ANOMALY_PLOTS)
def test_anomaly_plots_smoke_and_return_type(plot_func, mixed_anomaly_data):
    """
    Smoke test: ensure plots run without error and return an Axes.
    """
    # Ensure a clean state before each plot
    plt.close("all")

    ax = plot_func(
        df=mixed_anomaly_data,
        actual_col="actual",
        q_low_col="q_low",
        q_up_col="q_up",
        # Use savefig to prevent GUI pop-ups during testing
        savefig="temp_test_plot.png",
    )

    assert isinstance(
        ax, Axes
    ), f"{plot_func.__name__} did not return a matplotlib Axes object."
    assert ax.get_title() != "", f"{plot_func.__name__} did not set a title."


@pytest.mark.parametrize("plot_func", ALL_ANOMALY_PLOTS)
def test_anomaly_plots_no_anomalies(plot_func, no_anomaly_data):
    """
    Test that functions warn and return None if no anomalies
    are found.
    """
    plt.close("all")

    with pytest.warns(UserWarning, match="No anomalies detected"):
        ax = plot_func(
            df=no_anomaly_data,
            actual_col="actual",
            q_low_col="q_low",
            q_up_col="q_up",
        )

    assert ax is None, (
        f"{plot_func.__name__} should return None when there"
        " are no anomalies."
    )


@pytest.mark.parametrize("plot_func", ALL_ANOMALY_PLOTS)
def test_anomaly_plots_empty_input(plot_func):
    """
    Test that functions warn and return None for empty or all-NaN
    DataFrames.
    """
    plt.close("all")

    df_nan = pd.DataFrame(
        {
            "actual": [np.nan, np.nan],
            "q_low": [1, 2],
            "q_up": [3, 4],
        }
    )

    with pytest.warns(UserWarning, match="empty after dropping NaNs"):
        ax = plot_func(
            df=df_nan,
            actual_col="actual",
            q_low_col="q_low",
            q_up_col="q_up",
        )

    assert (
        ax is None
    ), f"{plot_func.__name__} should return None for empty data."
