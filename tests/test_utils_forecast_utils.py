import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from kdiagram.utils.forecast_utils import (
    bin_by_feature,
    calculate_probabilistic_scores,
    compute_interval_width,
    pivot_forecasts_long,
)


# --- Fixtures for generating test data ---
@pytest.fixture(scope="module")
def wide_forecast_data():
    """Fixture for a wide-format DataFrame."""
    np.random.seed(0)
    n_samples = 50
    df = pd.DataFrame(
        {
            "location_id": [f"loc_{i}" for i in range(n_samples)],
            "q10_2023": np.random.rand(n_samples) * 10,
            "q50_2023": np.random.rand(n_samples) * 10 + 10,
            "q90_2023": np.random.rand(n_samples) * 10 + 20,
            "q10_2024": np.random.rand(n_samples) * 10 + 5,
            "q50_2024": np.random.rand(n_samples) * 10 + 15,
            "q90_2024": np.random.rand(n_samples) * 10 + 25,
        }
    )
    return df


@pytest.fixture(scope="module")
def probabilistic_data():
    """Fixture for probabilistic forecast data."""
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.normal(loc=10, scale=5, size=n_samples)
    quantiles = np.linspace(0.05, 0.95, 19)
    preds = norm.ppf(quantiles, loc=y_true[:, np.newaxis], scale=5)
    return {"y_true": y_true, "preds": preds, "quantiles": quantiles}


# --- Tests for compute_interval_width ---


def test_compute_interval_width_single_pair(wide_forecast_data):
    """Test computing a single interval width."""
    df = compute_interval_width(wide_forecast_data, ["q10_2023", "q90_2023"])
    assert "width_q90_2023" in df.columns
    expected_width = (
        wide_forecast_data["q90_2023"] - wide_forecast_data["q10_2023"]
    )
    pd.testing.assert_series_equal(
        df["width_q90_2023"], expected_width, check_names=False
    )


def test_compute_interval_width_multi_pair_inplace(wide_forecast_data):
    """Test computing multiple widths with inplace modification."""
    df_copy = wide_forecast_data.copy()
    result_df = compute_interval_width(
        df_copy,
        ["q10_2023", "q90_2023"],
        ["q10_2024", "q90_2024"],
        inplace=True,
    )
    assert "width_q90_2023" in df_copy.columns
    assert "width_q90_2024" in df_copy.columns
    assert id(df_copy) == id(result_df)  # Check that it was modified in place


def test_compute_interval_width_raises_error_on_bad_pair():
    """Test that an error is raised for incorrectly formatted pairs."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    with pytest.raises(ValueError, match=r"must contain exactly two columns"):
        compute_interval_width(df, ["a", "b", "c"])


# --- Tests for bin_by_feature ---


def test_bin_by_feature_single_agg(wide_forecast_data):
    """Test binning with a single aggregation function."""
    stats_df = bin_by_feature(
        wide_forecast_data,
        bin_on_col="q50_2023",
        target_cols="q10_2023",
        n_bins=5,
        agg_funcs="mean",
    )
    assert stats_df.shape == (5, 2)
    assert "q10_2023" in stats_df.columns


def test_bin_by_feature_multi_agg(wide_forecast_data):
    """Test binning with multiple aggregation functions."""
    stats_df = bin_by_feature(
        wide_forecast_data,
        bin_on_col="q50_2023",
        target_cols=["q10_2023", "q90_2023"],
        n_bins=4,
        agg_funcs=["mean", "std"],
    )
    assert stats_df.shape == (4, 5)  # 1 bin col + 2 targets * 2 aggs
    assert ("q10_2023", "mean") in stats_df.columns
    assert ("q90_2023", "std") in stats_df.columns


# --- Tests for calculate_probabilistic_scores ---
def test_calculate_probabilistic_scores_returns_correct_df(
    probabilistic_data,
):
    """Test that the function returns a DataFrame with the correct columns and shape."""
    scores_df = calculate_probabilistic_scores(
        probabilistic_data["y_true"],
        probabilistic_data["preds"],
        probabilistic_data["quantiles"],
    )
    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape == (len(probabilistic_data["y_true"]), 3)
    assert all(
        col in scores_df.columns for col in ["pit_value", "sharpness", "crps"]
    )
    assert scores_df["pit_value"].between(0, 1).all()


# --- Tests for pivot_forecasts_long ---
def test_pivot_forecasts_long_basic(wide_forecast_data):
    """Test the basic wide-to-long transformation."""
    long_df = pivot_forecasts_long(
        wide_forecast_data,
        qlow_cols=["q10_2023", "q10_2024"],
        q50_cols=["q50_2023", "q50_2024"],
        qup_cols=["q90_2023", "q90_2024"],
    )
    assert long_df.shape[0] == wide_forecast_data.shape[0] * 2
    assert all(
        col in long_df.columns
        for col in ["horizon", "q_low", "q_median", "q_high"]
    )
    assert set(long_df["horizon"].unique()) == {"H1", "H2"}


def test_pivot_forecasts_long_with_ids_and_labels(wide_forecast_data):
    """Test pivoting with id_vars and custom horizon labels."""
    long_df = pivot_forecasts_long(
        wide_forecast_data,
        qlow_cols=["q10_2023", "q10_2024"],
        q50_cols=["q50_2023", "q50_2024"],
        qup_cols=["q90_2023", "q90_2024"],
        horizon_labels=["2023", "2024"],
        id_vars="location_id",
    )
    assert "location_id" in long_df.columns
    assert set(long_df["horizon"].unique()) == {"2023", "2024"}


def test_pivot_forecasts_long_raises_error_on_mismatch(wide_forecast_data):
    """Test that an error is raised for mismatched column list lengths."""
    with pytest.raises(ValueError, match=r"must have the same length"):
        pivot_forecasts_long(
            wide_forecast_data,
            qlow_cols=["q10_2023"],  # Mismatched length
            q50_cols=["q50_2023", "q50_2024"],
            qup_cols=["q90_2023", "q90_2024"],
        )


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
