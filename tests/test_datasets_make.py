import numpy as np
import pandas as pd
import pytest

from kdiagram.datasets import make as mk

def test_make_cyclical_data_bunch_and_frame_shapes():
    n = 24
    s = 2
    bunch = mk.make_cyclical_data(n_samples=n, n_series=s, seed=7, as_frame=False)
    assert hasattr(bunch, "frame")
    assert bunch.frame.shape[0] == n
    # expected columns: y_true, time_step, s prediction cols
    assert len(bunch.prediction_columns) == s
    assert set(bunch.feature_names) == {"time_step"}
    assert set(bunch.target_names) == {"y_true"}
    # as_frame=True ordering check
    df = mk.make_cyclical_data(n_samples=n, n_series=s, seed=7, as_frame=True)
    assert list(df.columns[:2]) == ["y_true", "time_step"]
    assert sum(c.startswith("model_") for c in df.columns) == s


def test_make_cyclical_data_series_and_param_validation_errors():
    # series_names length must match n_series
    with pytest.raises(ValueError, match=r"Length of series_names .* must match"):
        mk.make_cyclical_data(n_samples=10, n_series=2, series_names=["only_one"])
    # list length mismatch for scalar/list params
    with pytest.raises(ValueError, match=r"Length of 'pred_bias'"):
        mk.make_cyclical_data(n_samples=10, n_series=3, pred_bias=[0.0, 1.0])
    # wrong type for param
    with pytest.raises(TypeError, match=r"'pred_noise_factor' must be float or list"):
        mk.make_cyclical_data(n_samples=10, n_series=2, pred_noise_factor=(1.0, 1.5))


def test_make_cyclical_data_seed_reproducibility():
    a = mk.make_cyclical_data(n_samples=16, seed=123, as_frame=True)
    b = mk.make_cyclical_data(n_samples=16, seed=123, as_frame=True)
    np.testing.assert_allclose(a["y_true"].to_numpy(), b["y_true"].to_numpy())

def test_make_fingerprint_data_bunch_and_frame_and_validations():
    # basic bunch
    bunch = mk.make_fingerprint_data(n_layers=4, n_features=6, seed=3, as_frame=False)
    assert bunch.importances.shape == (4, 6)
    assert bunch.frame.shape == (4, 6)
    assert list(bunch.frame.index) == bunch.layer_names
    assert list(bunch.frame.columns) == bunch.feature_names

    # as_frame
    df = mk.make_fingerprint_data(n_layers=2, n_features=3, seed=4, as_frame=True)
    assert df.shape == (2, 3)

    # validations
    with pytest.raises(ValueError, match="sparsity must be between 0.0 and 1.0"):
        mk.make_fingerprint_data(sparsity=1.5)
    with pytest.raises(ValueError, match=r"value_range must be a tuple"):
        mk.make_fingerprint_data(value_range=(2.0, 1.0))
    with pytest.raises(ValueError, match=r"feature_names .* must match"):
        mk.make_fingerprint_data(n_features=3, feature_names=["F1"])
    with pytest.raises(ValueError, match=r"layer_names .* must match"):
        mk.make_fingerprint_data(n_layers=2, layer_names=["A"])


def test_make_uncertainty_data_bunch_and_lists_and_nperiods_zero():
    n_periods = 3
    bunch = mk.make_uncertainty_data(
        n_samples=30, n_periods=n_periods, seed=10, as_frame=False
    )
    # q lists lengths == n_periods
    assert len(bunch.q10_cols) == n_periods
    assert len(bunch.q50_cols) == n_periods
    assert len(bunch.q90_cols) == n_periods
    assert set(bunch.quantile_cols.keys()) == {"q0.1", "q0.5", "q0.9"}
    # frame has target and spatial features
    cols = set(bunch.frame.columns)
    assert {"location_id", "longitude", "latitude", "elevation"}.issubset(cols)
    assert any(c.endswith("_q0.1") for c in cols)

    # n_periods = 0 returns only features + target
    df0 = mk.make_uncertainty_data(n_samples=5, n_periods=0, seed=0, as_frame=True)
    assert set(df0.columns).issuperset(
        {"location_id", "longitude", "latitude", "elevation", "value_actual"}
    )
    assert not any("_q" in c for c in df0.columns)


def test_make_uncertainty_data_anomalies_outside_interval():
    df = mk.make_uncertainty_data(
        n_samples=80, n_periods=2, anomaly_frac=0.4, seed=123, as_frame=True
    )
    q10 = df["value_2022_q0.1"]
    q90 = df["value_2022_q0.9"]
    actual = df["value_actual"]
    outside = (actual < q10) | (actual > q90)
    # at least some anomalies exist
    assert outside.sum() > 0



def test_make_taylor_data_bunch_and_frame_and_warnings():
    # as bunch
    bunch = mk.make_taylor_data(n_samples=50, n_models=2, seed=5, as_frame=False)
    assert bunch.frame.shape == (50, 3)  # reference + 2 models
    assert set(bunch.stats.columns) == {"stddev", "corrcoef"}
    assert len(bunch.model_names) == 2

    # as frame
    df = mk.make_taylor_data(n_samples=20, n_models=3, seed=6, as_frame=True)
    assert df.shape == (20, 4)
    assert "reference" in df.columns

    # corr_range outside [0,1] -> warning + adjustment
    with pytest.warns(
        UserWarning, match="corr_range limits should ideally be between 0 and 1"
    ):
        _ = mk.make_taylor_data(corr_range=(-0.5, 1.5), seed=1)

    # std_range invalid -> warning + defaults
    with pytest.warns(UserWarning, match="std_range factors should be non-negative"):
        _ = mk.make_taylor_data(std_range=(-1.0, 0.5), seed=2)

    # noise_level zero while correlations < 1 allowed -> ValueError
    with pytest.raises(ValueError, match="noise_level cannot be zero"):
        _ = mk.make_taylor_data(noise_level=0.0, corr_range=(0.0, 0.9), seed=3)

def test_make_multi_model_quantile_data_bunch_and_frame_and_ordering():
    bunch = mk.make_multi_model_quantile_data(
        n_samples=40, n_models=2, seed=11, as_frame=False
    )
    assert hasattr(bunch, "frame") and hasattr(bunch, "data")
    # per-model columns exist
    for name in bunch.model_names:
        cols = bunch.prediction_columns[name]
        assert len(cols) == len(bunch.quantile_levels)
        assert all(col.startswith(f"{bunch.prefix}_{name}_q") for col in cols)

    # check quantile ordering for default [0.1, 0.5, 0.9]
    df = bunch.frame
    for name in bunch.model_names:
        q10 = df[f"{bunch.prefix}_{name}_q0.1"].to_numpy()
        q50 = df[f"{bunch.prefix}_{name}_q0.5"].to_numpy()
        q90 = df[f"{bunch.prefix}_{name}_q0.9"].to_numpy()
        assert np.all(q10 <= q50 + 1e-8)
        assert np.all(q50 <= q90 + 1e-8)

    # as_frame
    df2 = mk.make_multi_model_quantile_data(n_samples=10, as_frame=True, seed=12)
    assert "y_true" in df2.columns
    assert any(c.startswith("pred_Model_") for c in df2.columns)


def test_make_multi_model_quantile_data_validations_and_names():
    # must include 0.5
    with pytest.raises(ValueError, match="must contain 0.5"):
        mk.make_multi_model_quantile_data(quantiles=[0.1, 0.9])
    # width_range invalid
    with pytest.raises(ValueError, match="width_range must be"):
        mk.make_multi_model_quantile_data(width_range=(-1.0, 1.0))
    # bias_range invalid
    with pytest.raises(ValueError, match="bias_range must be"):
        mk.make_multi_model_quantile_data(bias_range=(2.0, -2.0))
    # custom names length must match
    with pytest.raises(ValueError, match="Length of model_names .* must match"):
        mk.make_multi_model_quantile_data(n_models=2, model_names=["A"])

    # custom names applied
    df = mk.make_multi_model_quantile_data(
        n_models=2, model_names=["Alpha", "Beta"], as_frame=True, seed=9
    )
    assert any(c.startswith("pred_Alpha_q") for c in df.columns)
    assert any(c.startswith("pred_Beta_q") for c in df.columns)


def test_make_multi_model_quantile_data_subset_quantiles():
    # Use a subset with only two quantiles including median
    bunch = mk.make_multi_model_quantile_data(
        n_samples=15, n_models=1, quantiles=[0.5, 0.9], seed=8, as_frame=False
    )
    cols = [c for c in bunch.frame.columns if c.startswith("pred_Model_A_q")]
    assert set(c.split("_q")[-1] for c in cols) == {"0.5", "0.9"}


if __name__ == "__main__":  # pragma : no-cover
    pytest.main([__file__])
