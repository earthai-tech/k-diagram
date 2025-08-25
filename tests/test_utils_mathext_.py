import numpy as np
import pytest
from scipy.stats import norm

from kdiagram.utils.mathext import (
    build_cdf_interpolator,
    calculate_calibration_error,
    compute_coverage_score,
    compute_pinball_loss,
    compute_winkler_score,
)


# --- Fixture for generating test data ---
@pytest.fixture(scope="module")
def probabilistic_data():
    """
    Generates a consistent set of probabilistic forecasts for testing.
    Includes a good model, an overconfident (too sharp) model, and an
    underconfident (not sharp) model.
    """
    np.random.seed(42)
    n_samples = 500
    true_loc, true_scale = 10, 5

    # The true values are random draws from the true distribution
    y_true = np.random.normal(loc=true_loc, scale=true_scale, size=n_samples)
    quantiles = np.linspace(0.05, 0.95, 19)

    # Model 1: Good model - its forecast matches the true distribution
    good_forecast_dist = norm.ppf(quantiles, loc=true_loc, scale=true_scale)
    good_preds = np.tile(good_forecast_dist, (n_samples, 1))

    # Model 2: Overconfident model - its forecast is too narrow
    overconfident_dist = norm.ppf(quantiles, loc=true_loc, scale=2.5)
    overconfident_preds = np.tile(overconfident_dist, (n_samples, 1))

    # Model 3: Underconfident model - its forecast is too wide and biased
    underconfident_dist = norm.ppf(quantiles, loc=true_loc + 2, scale=8)
    underconfident_preds = np.tile(underconfident_dist, (n_samples, 1))

    return {
        "y_true": y_true,
        "quantiles": quantiles,
        "preds": [good_preds, overconfident_preds, underconfident_preds],
        "names": ["Good", "Overconfident", "Underconfident"],
    }


def test_calculate_calibration_error(probabilistic_data):
    """Test the calibration error calculation."""
    # A well-calibrated forecast should have a low KS statistic (close to 0)
    calib_error_good = calculate_calibration_error(
        probabilistic_data["y_true"],
        probabilistic_data["preds"][0],  # Good model
        probabilistic_data["quantiles"],
    )
    assert 0 <= calib_error_good < 0.1

    # A poorly calibrated forecast should have a high KS statistic
    calib_error_bad = calculate_calibration_error(
        probabilistic_data["y_true"],
        probabilistic_data["preds"][1],  # Overconfident model
        probabilistic_data["quantiles"],
    )
    assert calib_error_bad > 0.1


@pytest.fixture(scope="module")
def forecast_data():
    """Provides consistent data for testing math utilities."""
    np.random.seed(0)
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # A forecast that is sometimes too narrow
    y_pred_lower = np.array([0, 2, 2, 5, 4, 7, 6, 9, 8, 9])
    y_pred_upper = np.array([2, 3, 4, 6, 6, 8, 8, 10, 11, 11])

    # Probabilistic data
    quantiles = np.array([0.1, 0.5, 0.9])
    preds_quantiles = np.array(
        [
            [8, 10, 12],  # y_true = 10 -> PIT = 0.66
            [0, 1, 2],  # y_true = 1 -> PIT = 0.66
            [4, 5, 6],  # y_true = 5 -> PIT = 0.66
        ]
    )
    y_true_prob = np.array([10, 1, 5])

    return {
        "y_true": y_true,
        "lower": y_pred_lower,
        "upper": y_pred_upper,
        "y_true_prob": y_true_prob,
        "preds_quantiles": preds_quantiles,
        "quantiles": quantiles,
    }


# --- Tests for compute_coverage_score ---


def test_compute_coverage_score_within(forecast_data):
    """Test standard 'within' coverage calculation."""
    # Expected: [T, T, T, F, T, F, T, F, T, T] -> 7/10 = 0.7
    score = compute_coverage_score(
        forecast_data["y_true"],
        forecast_data["lower"],
        forecast_data["upper"],
    )
    assert score == pytest.approx(0.7)


def test_compute_coverage_score_above_and_below(forecast_data):
    """Test 'above' and 'below' methods."""
    # y_true > upper: [F, F, F, F, F, F, F, F, F, F] -> 0
    above_score = compute_coverage_score(
        forecast_data["y_true"],
        forecast_data["lower"],
        forecast_data["upper"],
        method="above",
    )
    assert above_score == pytest.approx(0.0)

    # y_true < lower: [F, F, F, F, F, F, F, F, F, F] -> This is wrong, let's re-check
    # y_true: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # lower:  [0, 2, 2, 5, 4, 7, 6, 9, 8, 9]
    # below: [F, F, F, T, F, T, F, T, F, F] -> 3/10 = 0.3... wait, y_true < lower
    # 1<0(F), 2<2(F), 3<2(F), 4<5(T), 5<4(F), 6<7(T), 7<6(F), 8<9(T), 9<8(F), 10<9(F)
    # Ah, the logic was based on a different example. Let's re-calculate.
    # y_true: [1, 2, 3, 4, 5, 6]
    # lower:  [2, 2, 4, 4, 6, 6]
    # upper:  [3, 3, 5, 5, 7, 7]
    # y_true < lower -> [T, F, T, F, T, F] -> 3/6 = 0.5
    y_true = np.array([1, 2, 3, 4, 5, 6])
    lower = np.array([2, 2, 4, 4, 6, 6])
    upper = np.array([3, 3, 5, 5, 7, 7])

    below_score = compute_coverage_score(y_true, lower, upper, method="below")
    assert below_score == pytest.approx(0.5)


def test_compute_coverage_score_return_counts(forecast_data):
    """Test the return_counts parameter."""
    count = compute_coverage_score(
        forecast_data["y_true"],
        forecast_data["lower"],
        forecast_data["upper"],
        return_counts=True,
    )
    assert isinstance(count, int)
    assert count == 7


# --- Tests for compute_winkler_score ---


def test_compute_winkler_score_basic(forecast_data):
    """Test the Winkler score calculation."""
    # For the 8 covered points, score is width.
    # For the 2 uncovered, penalty is added.
    score = compute_winkler_score(
        forecast_data["y_true"],
        forecast_data["lower"],
        forecast_data["upper"],
        alpha=0.2,  # 80% interval
    )
    assert score > 0
    # A perfect forecast (width=0, all covered) would have a score of 0
    perfect_score = compute_winkler_score(
        np.array([5]), np.array([5]), np.array([5])
    )
    assert perfect_score == 0


# --- Tests for build_cdf_interpolator ---
def test_build_cdf_interpolator(forecast_data):
    """Test that the CDF interpolator works correctly."""
    cdf_func = build_cdf_interpolator(
        forecast_data["preds_quantiles"], forecast_data["quantiles"]
    )
    assert callable(cdf_func)

    # Test interpolation using the 3 corresponding true values
    # y_true_prob is [10, 1, 5]
    # preds_quantiles are [[8, 10, 12], [0, 1, 2], [4, 5, 6]]
    pit_values = cdf_func(forecast_data["y_true_prob"])

    # Case 1: y_true=10 vs [8, 10, 12] -> interp(10, ...) -> 0.5
    assert pit_values[0] == pytest.approx(0.5)

    # Case 2: y_true=1 vs [0, 1, 2] -> interp(1, ...) -> 0.5
    assert pit_values[1] == pytest.approx(0.5)

    # Case 3: y_true=5 vs [4, 5, 6] -> interp(5, ...) -> 0.5
    assert pit_values[2] == pytest.approx(0.5)

    # To test an intermediate value like 5.5, we test it against a
    # single forecast distribution.
    single_forecast = forecast_data["preds_quantiles"][2:3]  # Shape (1, 3)
    cdf_func_single = build_cdf_interpolator(
        single_forecast, forecast_data["quantiles"]
    )

    # Test y_true=5.5 against the forecast [4, 5, 6].
    # The result should be halfway between 0.5 and 0.9 -> 0.7
    pit_single = cdf_func_single(np.array([5.5]))
    assert pit_single[0] == pytest.approx(0.7)
    pattern = r"number of forecast distributions the interpolator was "
    with pytest.raises(ValueError, match=pattern):
        pit_values = cdf_func([1.0, 0.5, 5, 5.5])


def test_compute_pinball_loss(forecast_data):
    """Test the Pinball Loss calculation."""
    y_true = np.array([10, 10])
    y_pred = np.array([8, 12])  # one under, one over
    quantile = 0.9
    # Loss for under-prediction (10 > 8): (10-8) * 0.9 = 1.8
    # Loss for over-prediction (10 < 12): (12-10) * (1-0.9) = 0.2
    # Average = (1.8 + 0.2) / 2 = 1.0
    loss = compute_pinball_loss(y_true, y_pred, quantile)
    assert loss == pytest.approx(1.0)


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
