import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from kdiagram.utils.mathext import (
    compute_crps,
    compute_pit,
    get_forecast_arrays,
)

# --- Fixtures for generating test data ---


@pytest.fixture(scope="module")
def sample_df():
    """Provides a sample DataFrame for extraction tests."""
    return pd.DataFrame(
        {
            "actual": [10, 20, 30, 40, 50],
            "pred_point": [12, 18, 33, 42, 48],
            "q10": [8, 15, 25, 35, 45],
            "q50": [10, 20, 30, 40, 50],
            "q90": [12, 25, 35, 45, 55],
            "with_nan": [5, np.nan, 15, 25, 35],
        }
    )


@pytest.fixture(scope="module")
def probabilistic_data():
    """Provides more extensive probabilistic data for testing."""
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.normal(loc=10, scale=5, size=n_samples)
    quantiles = np.linspace(0.05, 0.95, 19)

    # Good and bad models
    good_preds = norm.ppf(quantiles, loc=y_true[:, np.newaxis], scale=5)
    bad_preds = norm.ppf(quantiles, loc=y_true[:, np.newaxis] + 2, scale=8)

    return {
        "y_true": y_true,
        "quantiles": quantiles,
        "preds": [good_preds, bad_preds],
    }


# --- Tests for get_forecast_arrays ---


def test_get_forecast_arrays_both_numpy(sample_df):
    """Test extracting both actual and prediction arrays as numpy."""
    y_true, y_preds = get_forecast_arrays(
        sample_df, actual_col="actual", pred_cols=["q10", "q50", "q90"]
    )
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_preds, np.ndarray)
    assert y_true.shape == (5,)
    assert y_preds.shape == (5, 3)


def test_get_forecast_arrays_both_pandas(sample_df):
    """Test extracting both as pandas objects."""
    y_true, y_preds = get_forecast_arrays(
        sample_df,
        actual_col="actual",
        pred_cols="pred_point",
        return_as="pandas",
    )
    assert isinstance(y_true, pd.Series)
    assert isinstance(y_preds, pd.Series)  # Squeezed to Series
    assert y_true.name == "actual"


def test_get_forecast_arrays_only_actual(sample_df):
    """Test extracting only the actual values."""
    y_true = get_forecast_arrays(sample_df, actual_col="actual")
    assert isinstance(y_true, np.ndarray)
    np.testing.assert_array_equal(y_true, sample_df["actual"].values)


def test_get_forecast_arrays_only_preds(sample_df):
    """Test extracting only the prediction values."""
    y_preds = get_forecast_arrays(sample_df, pred_cols=["q10", "q90"])
    assert isinstance(y_preds, np.ndarray)
    assert y_preds.shape == (5, 2)


def test_get_forecast_arrays_drop_na(sample_df):
    """Test the drop_na functionality."""
    y_true, y_preds = get_forecast_arrays(
        sample_df, actual_col="actual", pred_cols="with_nan", drop_na=True
    )
    assert len(y_true) == 4
    assert len(y_preds) == 4


def test_get_forecast_arrays_no_drop_na(sample_df):
    """Test disabling drop_na."""
    y_true, y_preds = get_forecast_arrays(
        sample_df, actual_col="actual", pred_cols="with_nan", drop_na=False
    )
    assert len(y_true) == 5
    assert np.isnan(y_preds).any()


def test_get_forecast_arrays_raises_error_on_no_cols(sample_df):
    """Test that a ValueError is raised if no columns are provided."""
    with pytest.raises(ValueError, match=r"at least one of"):
        get_forecast_arrays(sample_df)


# --- Tests for compute_pit ---


def test_compute_pit(probabilistic_data):
    """Test the PIT value calculation."""
    pit_values = compute_pit(
        probabilistic_data["y_true"],
        probabilistic_data["preds"][0],  # Good model
        probabilistic_data["quantiles"],
    )
    assert pit_values.shape == (len(probabilistic_data["y_true"]),)
    assert pit_values.min() >= 0
    assert pit_values.max() <= 1
    # For a well-calibrated model, the mean PIT should be close to 0.5
    assert np.mean(pit_values) == pytest.approx(0.5, abs=0.1)


# --- Tests for compute_crps ---


def test_compute_crps(probabilistic_data):
    """Test the CRPS calculation."""
    crps_good = compute_crps(
        probabilistic_data["y_true"],
        probabilistic_data["preds"][0],  # Good model
        probabilistic_data["quantiles"],
    )
    crps_bad = compute_crps(
        probabilistic_data["y_true"],
        probabilistic_data["preds"][1],  # Bad model
        probabilistic_data["quantiles"],
    )
    assert crps_good > 0
    # The bad model should have a higher (worse) CRPS score
    assert crps_bad > crps_good


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
