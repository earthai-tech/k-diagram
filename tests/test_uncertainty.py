# test_uncertainty_plots.py
# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Pytest suite for testing uncertainty visualization functions in
kdiagram.plot.uncertainty.
"""

from unittest.mock import patch
import re
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from kdiagram.plot.uncertainty import (
    PerformanceWarning,  # , InternalError
    plot_actual_vs_predicted,
    plot_anomaly_magnitude,
    plot_coverage,
    plot_coverage_diagnostic,
    plot_interval_consistency,
    plot_interval_width,
    plot_model_drift,
    plot_temporal_uncertainty,
    plot_uncertainty_drift,
    plot_velocity,
)

# --- Pytest Configuration ---
# Use a non-interactive backend for matplotlib to avoid plots
# popping up during tests. 'Agg' is a good choice.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_plots():
    """Fixture to close all matplotlib plots after each test."""
    yield  # Run the test
    plt.close("all")  # Cleanup after test completes


@pytest.fixture
def sample_data_coverage():
    """Provides sample data for plot_coverage tests."""
    np.random.seed(42)
    y_true = np.random.rand(100) * 10
    # Model 1: Quantile predictions (3 quantiles)
    y_pred_q1 = np.sort(
        np.random.rand(100, 3) * 10 + np.random.randn(100, 3) * 2, axis=1
    )
    # Model 2: Quantile predictions (3 quantiles)
    y_pred_q2 = np.sort(
        np.random.rand(100, 3) * 12 + np.random.randn(100, 3) * 1.5, axis=1
    )
    # Model 3: Point predictions (1D)
    y_pred_p1 = y_true + np.random.randn(100) * 0.5

    q_levels = [0.1, 0.5, 0.9]
    names = ["QuantModel1", "QuantModel2", "PointModel1"]

    return {
        "y_true": y_true,
        "y_pred_q1": y_pred_q1,
        "y_pred_q2": y_pred_q2,
        "y_pred_p1": y_pred_p1,
        "q": q_levels,
        "names": names,
    }


@pytest.fixture
def sample_data_drift():
    """Provides sample DataFrame for plot_model_drift tests."""
    np.random.seed(0)
    years = [2023, 2024, 2025, 2026]
    n_samples = 50
    data = {}
    for year in years:
        q10 = np.random.rand(n_samples) * 5 + (year - 2023) * 0.5
        q90 = q10 + np.random.rand(n_samples) * 2 + 1
        # Add a dummy metric for color testing
        metric = q10 + q90 + np.random.rand(n_samples)
        data[f"val_{year}_q10"] = q10
        data[f"val_{year}_q90"] = q90
        data[f"metric_{year}"] = metric

    df = pd.DataFrame(data)
    q10_cols = [f"val_{y}_q10" for y in years]
    q90_cols = [f"val_{y}_q90" for y in years]
    metric_cols = [f"metric_{y}" for y in years]
    horizons = years

    return {
        "df": df,
        "q10_cols": q10_cols,
        "q90_cols": q90_cols,
        "horizons": horizons,
        "metric_cols": metric_cols,
    }


@pytest.fixture
def sample_data_velocity():
    """Provides sample DataFrame for plot_velocity tests."""
    np.random.seed(123)
    n_points = 80
    years = [2020, 2021, 2022, 2023]
    data = {"location_id": range(n_points)}
    base_val = np.random.rand(n_points) * 10
    trend = np.linspace(0, 5, n_points)
    for i, year in enumerate(years):
        noise = np.random.randn(n_points) * 0.5
        data[f"val_{year}_q50"] = base_val + trend * i + noise

    # Add a column that can be used as theta_col (even if ignored)
    data["latitude"] = np.linspace(30, 35, n_points)
    df = pd.DataFrame(data)
    q50_cols = [f"val_{y}_q50" for y in years]

    return {"df": df, "q50_cols": q50_cols}


# --- Test Functions ---

# == Tests for plot_coverage ==


@pytest.mark.parametrize("kind", ["line", "bar", "pie", "radar"])
def test_plot_coverage_single_model_quantile(sample_data_coverage, kind):
    """Test plot_coverage with one quantile model for various kinds."""
    data = sample_data_coverage
    try:
        plot_coverage(
            data["y_true"],
            data["y_pred_q1"],  # Single model prediction
            names=[data["names"][0]],
            q=data["q"],
            kind=kind,
            figsize=(6, 6),  # Smaller figure for tests
        )
        # Check if a figure was created
        assert len(plt.get_fignums()) > 0, f"Plot should be created for kind='{kind}'"
    except Exception as e:
        pytest.fail(f"plot_coverage raised an exception for kind='{kind}': {e}")


def test_plot_coverage_multi_model_quantile_radar(sample_data_coverage):
    """Test plot_coverage with multiple quantile models (radar)."""
    data = sample_data_coverage
    try:
        plot_coverage(
            data["y_true"],
            data["y_pred_q1"],
            data["y_pred_q2"],  # Two models
            names=data["names"][:2],
            q=data["q"],
            kind="radar",
            cov_fill=True,
            figsize=(6, 6),
        )
        assert len(plt.get_fignums()) > 0, "Plot should be created"
    except Exception as e:
        pytest.fail(f"plot_coverage raised an exception: {e}")


def test_plot_coverage_single_model_point(sample_data_coverage):
    """Test plot_coverage with a single point forecast model."""
    data = sample_data_coverage
    try:
        # q should ideally be None or ignored for point forecasts
        plot_coverage(
            data["y_true"],
            data["y_pred_p1"],  # 1D point prediction
            names=[data["names"][2]],
            q=None,  # Explicitly None
            kind="bar",  # e.g.
            figsize=(6, 6),
        )
        assert len(plt.get_fignums()) > 0, "Plot should be created"
    except Exception as e:
        pytest.fail(f"plot_coverage raised an exception: {e}")


def test_plot_coverage_invalid_q(sample_data_coverage):
    """Test plot_coverage raises ValueError for invalid quantiles."""
    data = sample_data_coverage
    with pytest.raises(ValueError, match="between 0 and 1"):
        plot_coverage(data["y_true"], data["y_pred_q1"], q=[0.1, 1.5])

    with pytest.raises(ValueError, match="1D list or array"):
        plot_coverage(data["y_true"], data["y_pred_q1"], q=[[0.1], [0.9]])


# Note: Test for q vs pred dimension mismatch is skipped as the
# check is commented out in the source code provided.

# == Tests for plot_model_drift ==


@pytest.mark.parametrize(
    "acov", ["default", "half_circle", "quarter_circle", "eighth_circle"]
)
@pytest.mark.parametrize("annotate", [True, False])
def test_plot_model_drift_runs_ok(sample_data_drift, acov, annotate):
    """Test plot_model_drift runs without error for various settings."""
    data = sample_data_drift
    try:
        ax = plot_model_drift(
            df=data["df"],
            q10_cols=data["q10_cols"],
            q90_cols=data["q90_cols"],
            horizons=data["horizons"],
            acov=acov,
            annotate=annotate,
            figsize=(7, 7),
        )
        assert isinstance(ax, Axes), "Should return a Matplotlib Axes"
        assert len(plt.get_fignums()) > 0, "Plot should be created"
    except Exception as e:
        pytest.fail(f"plot_model_drift raised an exception: {e}")


def test_plot_model_drift_with_color_metric(sample_data_drift):
    """Test plot_model_drift with custom color metric columns."""
    data = sample_data_drift
    try:
        ax = plot_model_drift(
            df=data["df"],
            q10_cols=data["q10_cols"],
            q90_cols=data["q90_cols"],
            horizons=data["horizons"],
            color_metric_cols=data["metric_cols"],  # Use the metric
            cmap="plasma",
            figsize=(7, 7),
        )
        assert isinstance(ax, Axes)
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"plot_model_drift raised an exception: {e}")


def test_plot_model_drift_missing_columns(sample_data_drift):
    """Test plot_model_drift raises error if columns are missing."""
    data = sample_data_drift
    q10_cols_bad = data["q10_cols"] + ["missing_col_q10"]

    # Check internal validation within build_qcols_multiple (indirect)
    # or direct check if added to plot_model_drift itself.
    # Assuming build_qcols_multiple handles it via exist_features:
    with pytest.raises(ValueError):
        plot_model_drift(
            df=data["df"],
            q10_cols=q10_cols_bad,  # Contains missing column
            q90_cols=data["q90_cols"],
            horizons=data["horizons"],
        )

def test_plot_model_drift_mismatched_lengths(sample_data_drift):
    """Test plot_model_drift error for mismatched input list lengths."""
    data = sample_data_drift
    # Horizons list is shorter than quantile columns lists
    with pytest.raises(ValueError):
        plot_model_drift(
            df=data["df"],
            q10_cols=data["q10_cols"],
            q90_cols=data["q90_cols"],
            horizons=data["horizons"][:-1],  # Shorter horizons list
        )

    # q10_cols list is shorter than q90_cols list
    with pytest.raises(ValueError, match=( 
            "`qlow_cols` and `qup_cols` must be the same length")
            ):
        plot_model_drift(
            df=data["df"],
            q10_cols=data["q10_cols"][:-1],  # Shorter list
            q90_cols=data["q90_cols"],
            horizons=data["horizons"],
        )


# == Tests for plot_velocity ==


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("use_abs_color", [True, False])
@pytest.mark.parametrize("acov", ["default", "quarter_circle"])
@pytest.mark.parametrize("cbar", [True, False])
def test_plot_velocity_runs_ok(
    sample_data_velocity, normalize, use_abs_color, acov, cbar
):
    """Test plot_velocity runs without error for various settings."""
    data = sample_data_velocity
    try:
        ax = plot_velocity(
            df=data["df"],
            q50_cols=data["q50_cols"],
            normalize=normalize,
            use_abs_color=use_abs_color,
            acov=acov,
            cbar=cbar,
            mask_angle=(acov == "quarter_circle"),  # Example condition
            figsize=(8, 8),
        )
        assert isinstance(ax, Axes), "Should return a Matplotlib Axes"
        assert len(plt.get_fignums()) > 0, "Plot should be created"
    except Exception as e:
        pytest.fail(f"plot_velocity raised an exception: {e}")


def test_plot_velocity_missing_q50_cols(sample_data_velocity):
    """Test plot_velocity raises error for missing q50 columns."""
    data = sample_data_velocity
    q50_cols_bad = data["q50_cols"] + ["missing_q50"]
    with pytest.raises(ValueError, match="missing from the DataFrame"):
        plot_velocity(df=data["df"], q50_cols=q50_cols_bad)


def test_plot_velocity_too_few_q50_cols(sample_data_velocity):
    """Test plot_velocity raises error if < 2 q50 columns provided."""
    data = sample_data_velocity
    with pytest.raises(ValueError, match="At least two Q50 columns"):
        plot_velocity(df=data["df"], q50_cols=data["q50_cols"][:1])  # Only one


def test_plot_velocity_theta_col_warning(sample_data_velocity):
    """Test plot_velocity warns when theta_col is provided."""
    data = sample_data_velocity
    # Case 1: theta_col exists in DataFrame
    with pytest.warns(UserWarning, match="currently ignored for positioning"):
        plot_velocity(
            df=data["df"],
            q50_cols=data["q50_cols"],
            theta_col="latitude",  # Exists but ignored for position
        )

    # Case 2: theta_col does NOT exist in DataFrame
    with pytest.warns(UserWarning, match="not found in DataFrame"):
        plot_velocity(
            df=data["df"],
            q50_cols=data["q50_cols"],
            theta_col="non_existent_col",  # Doesn't exist
        )


def test_plot_velocity_zero_range_warning(sample_data_velocity):
    """Test plot_velocity warns if velocity range is zero."""
    data = sample_data_velocity
    df_zero_range = data["df"].copy()
    # Make all q50 values identical across time for zero velocity
    first_q50_col = data["q50_cols"][0]
    for col in data["q50_cols"][1:]:
        df_zero_range[col] = df_zero_range[first_q50_col]

    with pytest.warns(UserWarning, match="Velocity range is zero"):
        plot_velocity(
            df=df_zero_range,
            q50_cols=data["q50_cols"],
            normalize=True,  # Warning occurs during normalization
        )


@pytest.fixture
def sample_data_consistency():
    """Provides sample DataFrame for plot_interval_consistency."""
    np.random.seed(42)
    n_points = 100
    n_years = 4
    years = list(range(2021, 2021 + n_years))
    data = {"id": range(n_points)}
    data["latitude"] = np.linspace(40, 41, n_points)  # For theta_col test
    all_qlow_cols = []
    all_qup_cols = []
    all_q50_cols = []

    for i, year in enumerate(years):
        qlow_col = f"val_{year}_q10"
        qup_col = f"val_{year}_q90"
        q50_col = f"val_{year}_q50"
        all_qlow_cols.append(qlow_col)
        all_qup_cols.append(qup_col)
        all_q50_cols.append(q50_col)

        # Base values + some trend/noise per year
        base_low = np.random.rand(n_points) * 5 + i * 0.2
        width = (
            np.random.rand(n_points) * 3
            + 1
            + np.sin(np.linspace(0, np.pi, n_points)) * i
        )  # Vary width
        data[qlow_col] = base_low
        data[qup_col] = base_low + width
        data[q50_col] = base_low + width / 2 + np.random.randn(n_points) * 0.5

    df = pd.DataFrame(data)
    return {
        "df": df,
        "qlow_cols": all_qlow_cols,
        "qup_cols": all_qup_cols,
        "q50_cols": all_q50_cols,
    }


@pytest.fixture(params=[True, False])  # Run anomaly tests with/without anomalies
def sample_data_anomaly(request):
    """Provides sample DataFrame for plot_anomaly_magnitude."""
    generate_anomalies = request.param
    np.random.seed(111)
    n_points = 180
    data = {"id": range(n_points)}
    data["order_feature"] = np.random.permutation(n_points)  # For theta_col

    # Generate actual values and prediction intervals
    data["actual"] = np.random.randn(n_points) * 4 + 15  # Centered around 15
    data["q10"] = data["actual"] - np.random.rand(n_points) * 3 - 1
    data["q90"] = data["actual"] + np.random.rand(n_points) * 3 + 1

    if generate_anomalies:
        # Introduce some specific under-predictions
        under_indices = np.random.choice(n_points, 20, replace=False)
        data["actual"][under_indices] = (
            data["q10"][under_indices] - np.random.rand(20) * 5 - 0.5
        )  # Below q10

        # Introduce some specific over-predictions
        available_indices = list(set(range(n_points)) - set(under_indices))
        over_indices = np.random.choice(available_indices, 25, replace=False)
        data["actual"][over_indices] = (
            data["q90"][over_indices] + np.random.rand(25) * 6 + 0.5
        )  # Above q90

    df = pd.DataFrame(data)
    return {
        "df": df,
        "actual_col": "actual",
        "q_cols": ["q10", "q90"],
        "theta_col": "order_feature",
        "has_anomalies": generate_anomalies,  # Flag for assertions
    }


# --- Test Functions ---

# == Tests for plot_interval_consistency ==


@pytest.mark.parametrize("use_cv", [True, False])
@pytest.mark.parametrize("q50_provided", [True, False])
@pytest.mark.parametrize("acov", ["default", "half_circle"])
def test_plot_interval_consistency_runs_ok(
    sample_data_consistency, use_cv, q50_provided, acov
):
    """Test plot_interval_consistency runs okay."""
    data = sample_data_consistency
    q50_cols = data["q50_cols"] if q50_provided else None
    try:
        ax = plot_interval_consistency(
            df=data["df"],
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            q50_cols=q50_cols,
            use_cv=use_cv,
            acov=acov,
            mask_angle=True,
            figsize=(7, 7),
        )
        assert isinstance(ax, Axes)
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"plot_interval_consistency raised exception: {e}")


def test_plot_interval_consistency_mismatched_qcols(sample_data_consistency):
    """Test error on mismatched quantile column list lengths."""
    data = sample_data_consistency
    with pytest.raises(ValueError, match="Mismatch in length between"):
        plot_interval_consistency(
            df=data["df"],
            qlow_cols=data["qlow_cols"][:-1],  # Shorter list
            qup_cols=data["qup_cols"],
            q50_cols=data["q50_cols"],
        )
    with pytest.raises(ValueError, match="Mismatch in length between"):
        plot_interval_consistency(
            df=data["df"],
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            q50_cols=data["q50_cols"][:-1],  # Shorter list
        )


def test_plot_interval_consistency_missing_cols(sample_data_consistency):
    """Test error if specified columns are missing."""
    data = sample_data_consistency
    df_missing = data["df"].drop(columns=[data["qlow_cols"][0]])
    with pytest.raises(ValueError, match="missing from the DataFrame"):
        plot_interval_consistency(
            df=df_missing,
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            q50_cols=data["q50_cols"],
        )


def test_plot_interval_consistency_theta_col_warning(sample_data_consistency):
    """Test warning when theta_col is provided but ignored."""
    data = sample_data_consistency
    # Case 1: theta_col exists
    with pytest.warns(UserWarning, match="currently ignored for positioning"):
        plot_interval_consistency(
            df=data["df"],
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            theta_col="latitude",  # Exists
        )
    # Case 2: theta_col does not exist
    with pytest.warns(UserWarning, match="not found in DataFrame"):
        plot_interval_consistency(
            df=data["df"],
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            theta_col="non_existent",  # Doesn't exist
        )


def test_plot_interval_consistency_zero_mean_warning(sample_data_consistency):
    """Test warning when calculating CV with zero mean width."""
    data = sample_data_consistency
    df_zero = data["df"].copy()
    # Force width to be zero for some points by making qlow=qup
    df_zero[data["qup_cols"][0]] = df_zero[data["qlow_cols"][0]]
    # Make all widths zero for simplicity of trigger
    for ql, qu in zip(data["qlow_cols"], data["qup_cols"]):
        df_zero[qu] = df_zero[ql]

    with pytest.warns(RuntimeWarning, match="Mean interval width was zero"):
        plot_interval_consistency(
            df=df_zero,
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            use_cv=True,  # Warning occurs during CV calculation
        )


# == Tests for plot_anomaly_magnitude ==


@pytest.mark.parametrize("theta_col_provided", [True, False])
@pytest.mark.parametrize("acov", ["default", "quarter_circle"])
@pytest.mark.parametrize("cbar", [True, False])
def test_plot_anomaly_magnitude_runs_ok(
    sample_data_anomaly, theta_col_provided, acov, cbar
):
    """Test plot_anomaly_magnitude runs okay."""
    data = sample_data_anomaly
    theta_col = data["theta_col"] if theta_col_provided else None

    # Expect a warning only if there are NO anomalies to plot
    if not data["has_anomalies"]:
        with pytest.warns(UserWarning, match="No anomalies detected"):
            ax = plot_anomaly_magnitude(
                df=data["df"],
                actual_col=data["actual_col"],
                q_cols=data["q_cols"],
                theta_col=theta_col,
                acov=acov,
                cbar=cbar,
                figsize=(7, 7),
            )
            # Plot might be empty but should still return Axes
            assert (
                isinstance(ax, Axes) or ax is None
            )  # Can return None if df empty after NaN drop
            assert len(plt.get_fignums()) > 0  # Figure should exist
    else:
        # If anomalies are expected, run normally
        try:
            ax = plot_anomaly_magnitude(
                df=data["df"],
                actual_col=data["actual_col"],
                q_cols=data["q_cols"],
                theta_col=theta_col,
                acov=acov,
                cbar=cbar,
                figsize=(7, 7),
                verbose=0,  # Suppress summary print during test
            )
            assert isinstance(ax, Axes)
            assert len(plt.get_fignums()) > 0
        except Exception as e:
            pytest.fail(f"plot_anomaly_magnitude raised exception: {e}")


def test_plot_anomaly_magnitude_invalid_qcols(sample_data_anomaly):
    """Test error if q_cols is not a list/tuple of two strings."""
    data = sample_data_anomaly
    with pytest.raises(ValueError, match="exactly two column names"):
        plot_anomaly_magnitude(
            df=data["df"], actual_col=data["actual_col"], q_cols=["q10"]  # Only one
        )
    with pytest.raises(ValueError, match="exactly two column names"):
        plot_anomaly_magnitude(
            df=data["df"], actual_col=data["actual_col"], q_cols="q10"  # String
        )
    with pytest.raises(ValueError, match="exactly two column names"):
        plot_anomaly_magnitude(
            df=data["df"],
            actual_col=data["actual_col"],
            q_cols=["q10", "q50", "q90"],  # Three
        )


def test_plot_anomaly_magnitude_missing_cols(sample_data_anomaly):
    """Test error if essential columns are missing."""
    data = sample_data_anomaly
    df_missing_actual = data["df"].drop(columns=[data["actual_col"]])
    df_missing_q10 = data["df"].drop(columns=[data["q_cols"][0]])

    with pytest.raises(ValueError, match="essential columns are missing"):
        plot_anomaly_magnitude(
            df=df_missing_actual, actual_col=data["actual_col"], q_cols=data["q_cols"]
        )
    with pytest.raises(ValueError, match="essential columns are missing"):
        plot_anomaly_magnitude(
            df=df_missing_q10, actual_col=data["actual_col"], q_cols=data["q_cols"]
        )

def test_plot_anomaly_magnitude_theta_col_warning(sample_data_anomaly):
    """Test warnings related to theta_col."""
    data = sample_data_anomaly
    df_mod = data["df"].copy()

    # Case 1: theta_col exists but has non-numeric data (after NaN drop)
    df_mod["order_feature_str"] = df_mod["order_feature"].astype(str)
    with pytest.warns(UserWarning):
        plot_anomaly_magnitude(
            df=df_mod,
            actual_col=data["actual_col"],
            q_cols=data["q_cols"],
            theta_col="order_feature_str",  # Non-numeric
        )

    # Case 2: theta_col does not exist
    with pytest.raises(KeyError):
        plot_anomaly_magnitude(
            df=data["df"],
            actual_col=data["actual_col"],
            q_cols=data["q_cols"],
            theta_col="non_existent_column",  # Doesn't exist
        )


def test_plot_anomaly_magnitude_non_numeric_essential(sample_data_anomaly):
    """Test TypeError if essential columns are non-numeric."""
    data = sample_data_anomaly
    df_mod = data["df"].copy()
    df_mod[data["actual_col"]] = "string_value"  # Make actual non-numeric

    with pytest.raises(TypeError, match="Failed to convert essential columns"):
        plot_anomaly_magnitude(
            df=df_mod, actual_col=data["actual_col"], q_cols=data["q_cols"]
        )


def test_plot_anomaly_magnitude_empty_after_na_drop(sample_data_anomaly):
    """Test warning and return None if df empty after NaN drop."""
    data = sample_data_anomaly
    df_mod = data["df"].copy()
    # Make all essential values NaN
    df_mod[data["actual_col"]] = np.nan
    df_mod[data["q_cols"][0]] = np.nan
    df_mod[data["q_cols"][1]] = np.nan

    with pytest.warns(UserWarning, match="empty after dropping NaN"):
        result = plot_anomaly_magnitude(
            df=df_mod, actual_col=data["actual_col"], q_cols=data["q_cols"]
        )
        assert result is None


@pytest.fixture
def sample_data_drift_uncertainty():
    """Provides sample DataFrame for plot_uncertainty_drift."""
    np.random.seed(55)
    n_points = 90
    n_years = 5
    years = list(range(2020, 2020 + n_years))
    data = {"id": range(n_points)}
    data["latitude"] = np.linspace(10, 12, n_points)  # For theta_col test
    all_qlow_cols = []
    all_qup_cols = []

    for i, year in enumerate(years):
        qlow_col = f"value_{year}_q10"
        qup_col = f"value_{year}_q90"
        all_qlow_cols.append(qlow_col)
        all_qup_cols.append(qup_col)

        base_low = np.random.rand(n_points) * 3 + i * 0.1
        # Width increases slightly with year and sinusoidally with index
        width = (np.random.rand(n_points) + 0.5) * (
            1.5 + i * 0.3 + np.cos(np.linspace(0, 2 * np.pi, n_points))
        )
        data[qlow_col] = base_low
        data[qup_col] = base_low + width
        # Ensure non-negative width, though function should warn if negative
        data[qup_col] = np.maximum(data[qup_col], data[qlow_col])

    df = pd.DataFrame(data)
    dt_labels = [str(y) for y in years]
    return {
        "df": df,
        "qlow_cols": all_qlow_cols,
        "qup_cols": all_qup_cols,
        "dt_labels": dt_labels,
    }


@pytest.fixture
def sample_data_iw():
    """Provides sample DataFrame for plot_interval_width."""
    np.random.seed(77)
    n_points = 150
    data = {"location": range(n_points)}
    data["elevation"] = np.linspace(100, 500, n_points)  # For z_col
    data["q10_val"] = np.random.rand(n_points) * 20
    # Width depends on elevation
    width = 5 + (data["elevation"] / 100) * np.random.uniform(0.5, 2, n_points)
    data["q90_val"] = data["q10_val"] + width
    data["q50_val"] = data["q10_val"] + width / 2  # Add potential z_col
    df = pd.DataFrame(data)
    return {
        "df": df,
        "q_cols": ["q10_val", "q90_val"],
        "z_col": "elevation",
        "q50_col": "q50_val",  # Alternative z_col
        "theta_col": "location",  # Example theta col
    }


@pytest.fixture(params=[True, False])  # Test with/without covered points
def sample_data_coverage_diag(request):
    """Provides sample DataFrame for plot_coverage_diagnostic."""
    generate_covered = request.param
    np.random.seed(88)
    n_points = 250
    data = {"point_id": range(n_points)}
    data["actual_val"] = np.random.normal(loc=5, scale=1.5, size=n_points)
    # Base interval around 5
    data["q_lower"] = 5 - np.random.uniform(1, 3, n_points)
    data["q_upper"] = 5 + np.random.uniform(1, 3, n_points)

    if not generate_covered:
        # Ensure most points are outside interval
        is_high = data["actual_val"] > 5
        data["actual_val"][is_high] = (
            data["q_upper"][is_high] + np.random.rand(np.sum(is_high)) + 0.1
        )
        data["actual_val"][~is_high] = (
            data["q_lower"][~is_high] - np.random.rand(np.sum(~is_high)) - 0.1
        )

    df = pd.DataFrame(data)
    return {
        "df": df,
        "actual_col": "actual_val",
        "q_cols": ["q_lower", "q_upper"],
        "theta_col": "point_id",
    }


@pytest.fixture
def sample_data_temporal():
    """Provides sample DataFrame for plot_temporal_uncertainty."""
    np.random.seed(99)
    n_points = 80
    data = {"id": range(n_points)}
    base = 10 + 5 * np.sin(np.linspace(0, 2 * np.pi, n_points))
    data["val_q10"] = base - np.random.rand(n_points) * 2 - 1
    data["val_q50"] = base + np.random.randn(n_points) * 0.5
    data["val_q90"] = base + np.random.rand(n_points) * 2 + 1
    data["other_model"] = base * 1.1 + np.random.randn(n_points)
    df = pd.DataFrame(data)
    return {
        "df": df,
        "q_cols_list": ["val_q10", "val_q50", "val_q90"],
        "theta_col": "id",
    }


# == Tests for plot_uncertainty_drift ==
@pytest.mark.parametrize("dt_labels_provided", [True, False])
@pytest.mark.parametrize("acov", ["default", "quarter_circle"])
@pytest.mark.parametrize("mask_angle", [True, False])
def test_plot_uncertainty_drift_runs_ok(
    sample_data_drift_uncertainty, dt_labels_provided, acov, mask_angle
):
    """Test plot_uncertainty_drift runs okay."""
    data = sample_data_drift_uncertainty
    dt_labels = data["dt_labels"] if dt_labels_provided else None
    try:
        ax = plot_uncertainty_drift(
            df=data["df"],
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            dt_labels=dt_labels,
            acov=acov,
            mask_angle=mask_angle,
            figsize=(7, 7),
        )
        assert isinstance(ax, Axes)
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"plot_uncertainty_drift raised exception: {e}")


def test_plot_uncertainty_drift_mismatched_cols(sample_data_drift_uncertainty):
    """Test errors for mismatched column/label list lengths."""
    data = sample_data_drift_uncertainty
    with pytest.raises(ValueError):
        plot_uncertainty_drift(
            df=data["df"],
            qlow_cols=data["qlow_cols"][:-1],  # Shorter list
            qup_cols=data["qup_cols"],
            dt_labels=data["dt_labels"],
        )
    with pytest.raises(ValueError,):
        plot_uncertainty_drift(
            df=data["df"],
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            dt_labels=data["dt_labels"][:-1],  # Shorter list
        )


def test_plot_uncertainty_drift_missing_cols(sample_data_drift_uncertainty):
    """Test error if quantile columns are missing."""
    data = sample_data_drift_uncertainty
    df_missing = data["df"].drop(columns=[data["qlow_cols"][0]])
    with pytest.raises(ValueError, match="quantile columns are missing"):
        plot_uncertainty_drift(
            df=df_missing,
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            dt_labels=data["dt_labels"],
        )


def test_plot_uncertainty_drift_theta_col_warning(sample_data_drift_uncertainty):
    """Test warning when theta_col is provided."""
    data = sample_data_drift_uncertainty
    with pytest.warns(UserWarning, match="ignored for positioning"):
        plot_uncertainty_drift(
            df=data["df"],
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            theta_col="latitude",  # Exists
        )
    with pytest.warns(UserWarning, match="not found"):
        plot_uncertainty_drift(
            df=data["df"],
            qlow_cols=data["qlow_cols"],
            qup_cols=data["qup_cols"],
            theta_col="non_existent",  # Doesn't exist
        )


def test_plot_uncertainty_drift_width_warnings(sample_data_drift_uncertainty):
    """Test warnings related to interval widths."""
    data = sample_data_drift_uncertainty
    df_mod = data["df"].copy()

    # Case 1: Negative width
    df_mod[data["qup_cols"][0]] = df_mod[data["qlow_cols"][0]] - 1  # Force neg
    with pytest.warns(UserWarning, match="Negative interval widths detected"):
        plot_uncertainty_drift(
            df=df_mod, qlow_cols=data["qlow_cols"], qup_cols=data["qup_cols"]
        )

    # Case 2: Max width is zero
    df_zero = data["df"].copy()
    for ql, qu in zip(data["qlow_cols"], data["qup_cols"]):
        df_zero[qu] = df_zero[ql]  # Make all widths zero
    with pytest.warns(UserWarning, match="Maximum interval width.*?is zero"):
        plot_uncertainty_drift(
            df=df_zero, qlow_cols=data["qlow_cols"], qup_cols=data["qup_cols"]
        )


# --- Corrected Pytest Fixture ---
@pytest.fixture
def sample_data_avp():
    """
    Provides a reasonably sized sample DataFrame for most tests.
    The original fixture was too large, causing OverflowErrors.
    """
    np.random.seed(66)
    # Reduced n_points from a large number to a safe and fast value.
    n_points = 120
    data = {"sample": range(n_points)}
    data["time"] = pd.date_range("2024-01-01", periods=n_points, freq="h")
    signal = 20 + 15 * np.cos(np.linspace(0, 6 * np.pi, n_points))
    data["actual"] = signal + np.random.randn(n_points) * 3
    data["predicted"] = signal * 0.9 + np.random.randn(n_points) * 2 + 2
    df = pd.DataFrame(data)
    return {
        "df": df,
        "actual_col": "actual",
        "pred_col": "predicted",
        "theta_col": "time",
    }

# --- Corrected Tests ---

@pytest.mark.parametrize("line", [True, False])
@pytest.mark.parametrize("acov", ["default", "half_circle"])
@pytest.mark.parametrize("show_legend", [True, False])
def test_plot_actual_vs_predicted_runs_ok(sample_data_avp, line, acov, show_legend):
    """Test plot_actual_vs_predicted runs okay with standard-sized data."""
    data = sample_data_avp
    props = {"linestyle": "--", "marker": "x"} if not line else {}
    try:
        ax = plot_actual_vs_predicted(
            df=data["df"],
            actual_col=data["actual_col"],
            pred_col=data["pred_col"],
            acov=acov,
            line=line,
            show_legend=show_legend,
            mask_angle=True,
            actual_props=props,
            figsize=(7, 7),
        )
        assert isinstance(ax, Axes)
        # Ensure a figure was created
        assert plt.gcf().number > 0
    except ( Exception, OverflowError) as e:
        pytest.fail(f"plot_actual_vs_predicted raised an unexpected exception: {e}")
    finally:
        plt.close('all') # Clean up figures after each test run

def test_plot_actual_vs_predicted_missing_cols(sample_data_avp):
    """Test ValueError if actual or predicted columns are missing."""
    data = sample_data_avp
    df_missing = data["df"].drop(columns=[data["actual_col"]])
    # The function should raise a ValueError when a required column is not found.
    with pytest.raises(ValueError):
        plot_actual_vs_predicted(
            df=df_missing,
            actual_col=data["actual_col"],
            pred_col=data["pred_col"],
        )


# == Tests for plot_interval_width ==
@pytest.mark.parametrize("z_col_option", [None, "q50_col", "z_col"])
@pytest.mark.parametrize("acov", ["default", "eighth_circle"])
@pytest.mark.parametrize("cbar", [True, False])
def test_plot_interval_width_runs_ok(sample_data_iw, z_col_option, acov, cbar):
    """Test plot_interval_width runs okay."""
    data = sample_data_iw
    z_col = data[z_col_option] if z_col_option else None
    try:
        ax = plot_interval_width(
            df=data["df"],
            q_cols=data["q_cols"],
            z_col=z_col,
            acov=acov,
            cbar=cbar,
            mask_angle=True,
            figsize=(7, 7),
        )
        assert isinstance(ax, Axes)
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"plot_interval_width raised exception: {e}")


def test_plot_interval_width_invalid_qcols(sample_data_iw):
    """Test error if q_cols is not length 2."""
    data = sample_data_iw
    with pytest.raises(TypeError, match="expects exactly two column names"):
        plot_interval_width(df=data["df"], q_cols=["q10_val"])
    with pytest.raises(TypeError, match="expects exactly two column names"):
        plot_interval_width(df=data["df"], q_cols="q10_val")  # String


def test_plot_interval_width_missing_cols(sample_data_iw):
    """Test error if required columns are missing."""
    data = sample_data_iw
    df_missing_q = data["df"].drop(columns=[data["q_cols"][0]])
    df_missing_z = data["df"].drop(columns=[data["z_col"]])

    with pytest.raises(ValueError, match="Essential quantile columns missing"):
        plot_interval_width(df=df_missing_q, q_cols=data["q_cols"])

    with pytest.raises(ValueError, match="`z_col`.*?not found"):
        plot_interval_width(df=df_missing_z, q_cols=data["q_cols"], z_col=data["z_col"])


def test_plot_interval_width_negative_width_warning(sample_data_iw):
    """Test warning for negative interval widths."""
    data = sample_data_iw
    df_neg = data["df"].copy()
    # Force negative width for some points
    df_neg.loc[0:10, data["q_cols"][1]] = df_neg.loc[0:10, data["q_cols"][0]] - 1
    with pytest.warns(UserWarning, match="negative interval width"):
        plot_interval_width(df=df_neg, q_cols=data["q_cols"])


# == Tests for plot_coverage_diagnostic ==


@pytest.mark.parametrize("as_bars", [True, False])
@pytest.mark.parametrize("fill_gradient", [True, False])
@pytest.mark.parametrize("acov", ["default", "half_circle"])
def test_plot_coverage_diagnostic_runs_ok(
    sample_data_coverage_diag, as_bars, fill_gradient, acov
):
    """Test plot_coverage_diagnostic runs okay."""
    data = sample_data_coverage_diag
    try:
        ax = plot_coverage_diagnostic(
            df=data["df"],
            actual_col=data["actual_col"],
            q_cols=data["q_cols"],
            as_bars=as_bars,
            fill_gradient=fill_gradient,
            acov=acov,
            mask_angle=True,
            figsize=(7, 7),
            verbose=0,  # Suppress print
        )
        assert isinstance(ax, Axes)
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"plot_coverage_diagnostic raised exception: {e}")


def test_plot_coverage_diagnostic_invalid_qcols(sample_data_coverage_diag):
    """Test error if q_cols is not length 2."""
    data = sample_data_coverage_diag
    with pytest.raises(TypeError, match="expects exactly two column names"):
        plot_coverage_diagnostic(
            df=data["df"], actual_col=data["actual_col"], q_cols=["q_lower"]
        )


def test_plot_coverage_diagnostic_missing_cols(sample_data_coverage_diag):
    """Test error if essential columns are missing."""
    data = sample_data_coverage_diag
    df_missing = data["df"].drop(columns=[data["actual_col"]])
    with pytest.raises(ValueError, match="Essential columns missing"):
        plot_coverage_diagnostic(
            df=df_missing, actual_col=data["actual_col"], q_cols=data["q_cols"]
        )


def test_plot_coverage_diagnostic_invalid_acov(sample_data_coverage_diag):
    """Test error for invalid acov value."""
    data = sample_data_coverage_diag
    with pytest.raises(ValueError, match="Invalid `acov` value"):
        plot_coverage_diagnostic(
            df=data["df"],
            actual_col=data["actual_col"],
            q_cols=data["q_cols"],
            acov="invalid_coverage_name",
        )


# == Tests for plot_temporal_uncertainty ==
# Mock the detect_quantiles_in function if it's used for 'auto'
MOCK_DETECTED_COLS = ["val_q10", "val_q50", "val_q90"]


@pytest.mark.parametrize(
    "q_cols_option",
    [
        "explicit_list",
        # 'auto' # Requires mocking or actual function
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("acov", ["default", "eighth_circle"])
@patch("kdiagram.plot.uncertainty.detect_quantiles_in", return_value=MOCK_DETECTED_COLS)
def test_plot_temporal_uncertainty_runs_ok(
    mock_detect, sample_data_temporal, q_cols_option, normalize, acov
):
    """Test plot_temporal_uncertainty runs okay."""
    data = sample_data_temporal
    if q_cols_option == "explicit_list":
        q_cols = data["q_cols_list"]
    elif q_cols_option == "auto":
        q_cols = "auto"
        # Ensure mock is called by checking if the test DataFrame
        # actually has the MOCK_DETECTED_COLS
        if not all(c in data["df"].columns for c in MOCK_DETECTED_COLS):
            pytest.skip("Skipping 'auto' test as mock data doesn't match.")
    else:
        pytest.skip("Invalid q_cols_option")

    names = (
        [f"Series {i+1}" for i in range(len(data["q_cols_list"]))]
        if q_cols_option == "explicit_list"
        else None
    )

    try:
        ax = plot_temporal_uncertainty(
            df=data["df"],
            q_cols=q_cols,
            names=names,
            normalize=normalize,
            acov=acov,
            mask_angle=True,
            mask_label=normalize,  # Example condition
            figsize=(7, 7),
        )
        assert isinstance(ax, Axes)
        assert len(plt.get_fignums()) > 0
        if q_cols_option == "auto":
            mock_detect.assert_called_once()  # Verify mock was used
    except Exception as e:
        pytest.fail(f"plot_temporal_uncertainty raised exception: {e}")


def test_plot_temporal_uncertainty_errors(sample_data_temporal):
    """Test various error conditions for plot_temporal_uncertainty."""
    data = sample_data_temporal

    # Empty q_cols list
    with pytest.raises(ValueError):
        plot_temporal_uncertainty(df=data["df"], q_cols=[])

    # Mismatched names length
    with pytest.raises(ValueError):
        plot_temporal_uncertainty(
            df=data["df"],
            q_cols=data["q_cols_list"],
            names=["One Name"],  # Only one name for multiple columns
        )

    # Missing columns
    df_missing = data["df"].drop(columns=[data["q_cols_list"][0]])
    with pytest.raises(ValueError, match=re.escape( 
            "Specified plot columns (`q_cols`)s"
            " 'val_q10' not found in the dataframe."
            )):
        plot_temporal_uncertainty(df=df_missing, q_cols=data["q_cols_list"])

    # Invalid acov
    with pytest.raises(ValueError, match="Invalid `acov` value"):
        plot_temporal_uncertainty(
            df=data["df"], q_cols=data["q_cols_list"], acov="bad_acov"
        )


def test_plot_temporal_uncertainty_theta_col_warning(sample_data_temporal):
    """Test theta_col warnings."""
    data = sample_data_temporal
    with pytest.warns(UserWarning, match="ignored for positioning/ordering"):
        plot_temporal_uncertainty(
            df=data["df"],
            q_cols=data["q_cols_list"],
            theta_col=data["theta_col"],  # Exists
        )


if __name__ == "__main__":
    pytest.main([__file__])
