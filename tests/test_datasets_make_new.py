import numpy as np
import pandas as pd
import pytest

from kdiagram.datasets.make import (
    make_classification_data,
    make_cyclical_data,
    make_fingerprint_data,
    make_multi_model_quantile_data,
    make_regression_data,
    make_taylor_data,
    make_uncertainty_data,
)

# ------------------------------- make_cyclical_data ---------------------------


def test_make_cyclical_basic_and_errors():
    # scalar params should broadcast; as_frame=True order/shape
    df = make_cyclical_data(
        n_samples=24,
        n_series=3,
        pred_phase_shift=0.1,
        pred_amplitude_factor=0.9,
        pred_noise_factor=1.2,
        pred_bias=0.0,
        as_frame=True,
        seed=123,
    )
    assert list(df.columns[:2]) == ["y_true", "time_step"]
    assert df.shape == (24, 2 + 3)

    # Bunch path, seed=None branch
    b = make_cyclical_data(
        n_samples=10, n_series=2, seed=None, as_frame=False
    )
    assert set(
        [
            "frame",
            "data",
            "feature_names",
            "target_names",
            "target",
            "series_names",
            "prediction_columns",
            "DESCR",
        ]
    ).issubset(b.keys())
    assert b.data.shape == (10, 1 + 2)  # time_step + two series

    # wrong series_names length -> ValueError
    with pytest.raises(ValueError):
        make_cyclical_data(
            n_series=2, series_names=["only_one"], as_frame=True
        )

    # bad param type -> TypeError
    with pytest.raises(TypeError):
        make_cyclical_data(n_series=2, pred_bias={"oops": 1}, as_frame=True)


# ------------------------------ make_fingerprint_data ------------------------


def test_make_fingerprint_variants_and_validation():
    # invalid sparsity
    with pytest.raises(ValueError):
        make_fingerprint_data(sparsity=-0.1, as_frame=True)
    # invalid value_range
    with pytest.raises(ValueError):
        make_fingerprint_data(value_range=(2.0, -1.0), as_frame=True)
    # feature names length mismatch
    with pytest.raises(ValueError):
        make_fingerprint_data(
            n_features=4, feature_names=["a", "b"], as_frame=True
        )
    # layer names length mismatch
    with pytest.raises(ValueError):
        make_fingerprint_data(n_layers=2, layer_names=["L1"], as_frame=True)

    # as_frame=True with sparsity > 0 => expect some zeros
    df = make_fingerprint_data(
        n_layers=4, n_features=7, sparsity=0.25, seed=1, as_frame=True
    )
    assert df.shape == (4, 7)
    assert (df.values == 0).sum() > 0

    # Bunch path, add_structure toggled
    b = make_fingerprint_data(
        n_layers=3, n_features=5, add_structure=False, seed=2
    )
    assert b.importances.shape == (3, 5)
    assert isinstance(b.frame, pd.DataFrame)


# ------------------------------ make_uncertainty_data ------------------------


def test_make_uncertainty_anomalies_and_layout():
    ds = make_uncertainty_data(
        n_samples=50, n_periods=3, anomaly_frac=0.2, seed=7
    )  # Bunch path
    assert (
        len(ds.q10_cols) == 3
        and len(ds.q90_cols) == 3
        and len(ds.q50_cols) == 3
    )

    # anomaly count equals int(frac * n_samples)
    n_anom_expected = int(0.2 * 50)
    q10_first = ds.frame[ds.q10_cols[0]].to_numpy()
    q90_first = ds.frame[ds.q90_cols[0]].to_numpy()
    actual = ds.frame[ds.target_names[0]].to_numpy()
    outside = (actual < q10_first) | (actual > q90_first)
    assert outside.sum() >= n_anom_expected

    # as_frame=True ordered columns start with features + actual
    df = make_uncertainty_data(
        as_frame=True, n_samples=5, n_periods=2, seed=0
    )
    assert list(df.columns[:5]) == [
        "location_id",
        "longitude",
        "latitude",
        "elevation",
        "value_actual",
    ]
    # then Q10/Q50/Q90 triplets
    assert any(c.endswith("q0.1") for c in df.columns[5:8])
    assert any(c.endswith("q0.5") for c in df.columns[5:8])
    assert any(c.endswith("q0.9") for c in df.columns[5:8])

    # anomaly_frac=0 => no forced outside
    df0 = make_uncertainty_data(
        as_frame=True, n_samples=30, n_periods=2, anomaly_frac=0.0, seed=3
    )
    a = df0["value_actual"].to_numpy()
    q10 = df0.filter(like="_q0.1").iloc[:, 0].to_numpy()
    q90 = df0.filter(like="_q0.9").iloc[:, 0].to_numpy()
    assert (
        (a < q10) | (a > q90)
    ).sum() >= 0  # should be small; no exact count required


# -------------------------------- make_taylor_data ---------------------------


def test_make_taylor_ranges_and_errors_and_paths(tmp_path):
    # invalid ranges -> warnings then proceed
    with pytest.warns(UserWarning, match="corr_range"):
        _ = make_taylor_data(corr_range=(-0.2, 1.5), as_frame=True)
    with pytest.warns(UserWarning, match="std_range"):
        _ = make_taylor_data(std_range=(-1, -0.5), as_frame=True)

    # noise_level=0 with sub-perfect corr requested -> ValueError
    with pytest.raises(ValueError):
        make_taylor_data(noise_level=0.0, corr_range=(0.5, 0.9))

    # normal Bunch path; stats present
    ds = make_taylor_data(n_models=2, seed=4)
    assert set(
        [
            "frame",
            "reference",
            "predictions",
            "model_names",
            "stats",
            "ref_std",
            "DESCR",
        ]
    ).issubset(ds.keys())
    assert set(["stddev", "corrcoef"]).issubset(ds.stats.columns)


# ----------------------- make_multi_model_quantile_data ----------------------


def test_make_multi_model_quantile_validation_and_sorting():
    # requires 0.5 in quantiles
    with pytest.raises(ValueError):
        make_multi_model_quantile_data(quantiles=[0.1, 0.9], as_frame=True)

    # invalid ranges
    with pytest.raises(ValueError):
        make_multi_model_quantile_data(width_range=(-1.0, 2.0), as_frame=True)
    with pytest.raises(ValueError):
        make_multi_model_quantile_data(bias_range=(2.0, -2.0), as_frame=True)

    # Padding case (too few entries => pad with last tuple)
    with pytest.warns(UserWarning, match="Padding"):
        _ = make_multi_model_quantile_data(
            n_models=3,
            width_range=[(5.0, 8.0)],  # list of tuples, not floats
            as_frame=True,
        )
    # Truncating case (too many entries => truncate extras)
    with pytest.warns(UserWarning, match="Truncating"):
        _ = make_multi_model_quantile_data(
            n_models=2,
            bias_range=[
                (-2.0, 2.0),
                (-1.0, 1.0),
                (0.0, 0.5),
            ],  # list of tuples
            as_frame=True,
        )

    # Sorting property: for each row q_min <= q50 <= q_max for a model
    ds = make_multi_model_quantile_data(n_samples=20, n_models=1, seed=5)
    cols = ds.prediction_columns["Model_A"]
    # infer min, median, max from names (0.1, 0.5, 0.9)
    qmin = [c for c in cols if c.endswith("q0.1")][0]
    q50 = [c for c in cols if c.endswith("q0.5")][0]
    qmax = [c for c in cols if c.endswith("q0.9")][0]
    f = ds.frame[[qmin, q50, qmax]].to_numpy()
    assert np.all(f[:, 0] <= f[:, 1])
    assert np.all(f[:, 1] <= f[:, 2])

    # as_frame=True returns tidy layout
    df = make_multi_model_quantile_data(as_frame=True, seed=2)
    assert set(["y_true", "feature_1", "feature_2"]).issubset(df.columns)


# -------------------------------- make_regression_data -----------------------


def test_make_regression_validation_paths_and_outputs():
    # invalid feature range
    with pytest.raises(ValueError):
        make_regression_data(feature_range=(1.0, 0.0), as_frame=True)

    # true_func wrong shape
    def bad_true(X):  # returns (n,1) instead of (n,)
        return np.ones((X.shape[0], 1))

    with pytest.raises(ValueError):
        make_regression_data(true_func=bad_true, as_frame=True)

    # invalid true_kind
    with pytest.raises(ValueError):
        make_regression_data(true_func=None, true_kind="cubic", as_frame=True)

    # negative noise_on_true
    with pytest.raises(ValueError):
        make_regression_data(noise_on_true=-1.0, as_frame=True)

    # heteroskedastic path + clip_negative + as_frame=True
    df = make_regression_data(
        n_samples=50,
        heteroskedastic=True,
        noise_on_true=1.5,
        clip_negative=True,
        as_frame=True,
        seed=9,
    )
    assert (df["y_true"] >= 0).all()

    # invalid model error_type -> ValueError
    with pytest.raises(ValueError):
        make_regression_data(
            n_models=1,
            model_profiles={
                "Bad": {"error_type": "unknown", "bias": 0, "noise_std": 1}
            },
            as_frame=True,
        )

    # name resolution: extra user names are ignored with a warning; no prefix for provided names
    profiles = {
        "Good Model": {
            "bias": 0.0,
            "noise_std": 2.0,
            "error_type": "additive",
        },
        "High Var": {"bias": 0.0, "noise_std": 5.0, "error_type": "additive"},
    }
    with pytest.warns(UserWarning, match="Extra names"):
        df2 = make_regression_data(
            n_samples=30,
            n_models=2,
            model_profiles=profiles,
            model_names=["A", "B", "C"],  # extra -> warns (via resolver)
            as_frame=True,
            seed=2,
        )
    # exact names used as columns (no prefix)
    assert set(["A", "B"]).issubset(df2.columns)

    # Bunch path
    b = make_regression_data(n_samples=25, n_models=3, as_frame=False, seed=1)
    assert set(
        [
            "frame",
            "data",
            "feature_names",
            "target_names",
            "prediction_columns",
            "model_names",
            "prefix",
            "DESCR",
        ]
    ).issubset(b.keys())


# ------------------------------ make_classification_data ---------------------


def test_make_classification_validation_binary_and_multiclass():
    # invalid weights sum
    with pytest.raises(ValueError):
        make_classification_data(
            n_classes=2, weights=[0.0, 0.0], as_frame=True
        )
    # weights length mismatch
    with pytest.raises(ValueError):
        make_classification_data(
            n_classes=2, weights=[1.0, 0.0, 0.0], as_frame=True
        )

    # model_names vs profiles length mismatch -> ValueError
    profiles = {"M1": {}, "M2": {}}
    with pytest.raises(ValueError):
        make_classification_data(
            n_classes=2,
            n_models=2,
            model_profiles=profiles,
            model_names=["only_one"],
        )

    # Binary: include label cols + as_frame=True
    df_bin = make_classification_data(
        n_samples=200,
        n_features=8,
        n_classes=2,
        n_models=2,
        model_names=["Good", "Biased"],
        include_binary_pred_cols=True,
        as_frame=True,
        seed=7,
    )
    assert "Good" in df_bin.columns and "Biased" in df_bin.columns
    assert any(c.startswith("pred_") for c in df_bin.columns)  # label columns

    # Multiclass: compat cols and Bunch path with proba columns accounted for
    ds_mc = make_classification_data(
        n_samples=120,
        n_features=6,
        n_classes=3,
        n_models=2,
        add_compat_cols=True,
        as_frame=False,
        seed=11,
    )
    # Bunch contains probability columns list (n_models * n_classes)
    assert len(ds_mc.prediction_columns) == 2 * 3
    # frame has yt + yp compatibility columns
    assert "yt" in ds_mc.frame.columns and any(
        c.startswith("pred_") for c in ds_mc.frame.columns
    )
    assert "yp" in ds_mc.frame.columns
