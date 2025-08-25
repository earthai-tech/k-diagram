import numpy as np
import pandas as pd
import pytest

from kdiagram.utils.forecast_utils import (
    bin_by_feature,
    calculate_probabilistic_scores,
    compute_forecast_errors,
    compute_interval_width,
    pivot_forecasts_long,
)

# --- Fixtures for generating test data ---


@pytest.fixture
def sample_data():
    """Provides a consistent DataFrame for all utility tests."""
    np.random.seed(1)
    n_samples = 20
    df = pd.DataFrame(
        {
            "actual": np.arange(n_samples),
            "pred_A": np.arange(n_samples)
            + np.random.normal(0, 1, n_samples),
            "pred_B": np.arange(n_samples)
            - np.random.normal(0, 1, n_samples),
            "q10_A": np.arange(n_samples) - 1,
            "q90_A": np.arange(n_samples) + 1,
            "q10_B": np.arange(n_samples) - 2,
            "q90_B": np.arange(n_samples) + 2,
            "feature_for_binning": np.random.rand(n_samples) * 100,
        }
    )
    return df


# --- Tests for compute_forecast_errors ---


def test_compute_forecast_errors_all_types(sample_data):
    """Test all error_type options."""
    df_sq = compute_forecast_errors(
        sample_data, "actual", "pred_A", error_type="squared"
    )
    assert "error_pred_A" in df_sq.columns
    assert np.all(df_sq["error_pred_A"] >= 0)

    df_pct = compute_forecast_errors(
        sample_data, "actual", "pred_A", error_type="percentage"
    )
    assert "error_pred_A" in df_pct.columns
    # Check for NaNs where actual is 0
    assert df_pct["error_pred_A"].isnull().sum() == (
        1 if 0 in sample_data["actual"].values else 0
    )


def test_compute_forecast_errors_no_preds_raises_error(sample_data):
    """Test that an error is raised if no pred_cols are provided."""
    with pytest.raises(
        ValueError, match=r"At least one prediction column must be provided"
    ):
        compute_forecast_errors(sample_data, "actual")


def test_compute_forecast_errors_invalid_type_raises_error(sample_data):
    """Test that an error is raised for an invalid error_type."""
    with pytest.raises(ValueError, match=r"Unknown error_type"):
        compute_forecast_errors(
            sample_data, "actual", "pred_A", error_type="invalid_type"
        )


# --- Tests for compute_interval_width ---


def test_compute_interval_width_no_pairs_raises_error(sample_data):
    """Test that an error is raised if no quantile_pairs are provided."""
    with pytest.raises(
        ValueError,
        match=r"At least one pair of quantile columns must be provided",
    ):
        compute_interval_width(sample_data)


# --- Tests for calculate_probabilistic_scores ---


def test_calculate_probabilistic_scores_edge_cases():
    """Test edge cases for probabilistic scores."""
    # Test with empty arrays
    scores_df = calculate_probabilistic_scores(
        np.array([]), np.array([]).reshape(0, 2), np.array([0.1, 0.9])
    )
    assert scores_df.empty


# --- Tests for pivot_forecasts_long ---


def test_pivot_forecasts_long_mismatched_labels_raises_error(sample_data):
    """Test that an error is raised for mismatched horizon_labels length."""
    with pytest.raises(
        ValueError, match="Length of horizon_labels must match"
    ):
        pivot_forecasts_long(
            sample_data,
            qlow_cols=["q10_A", "q10_B"],
            q50_cols=["pred_A", "pred_B"],
            qup_cols=["q90_A", "q90_B"],
            horizon_labels=["H1"],  # Mismatched length
        )


# --- Tests for bin_by_feature ---


def test_bin_by_feature_dict_agg(sample_data):
    """Test binning with a dictionary of aggregation functions."""
    stats_df = bin_by_feature(
        sample_data,
        bin_on_col="feature_for_binning",
        target_cols=["pred_A", "pred_B"],
        n_bins=3,
        agg_funcs={"pred_A": "mean", "pred_B": "std"},
    )
    assert stats_df.shape == (3, 3)
    assert "pred_A" in stats_df.columns
    assert "pred_B" in stats_df.columns
