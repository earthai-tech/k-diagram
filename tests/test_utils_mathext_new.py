import numpy as np
import pandas as pd
import pytest

from kdiagram.utils.mathext import (
    build_cdf_interpolator,
    calculate_calibration_error,
    compute_coverage_score,
    compute_crps,
    compute_pinball_loss,
    compute_pit,
    compute_winkler_score,
    get_forecast_arrays,
    minmax_scaler,
)


def _mk_df():
    return pd.DataFrame(
        {
            "y": [1.0, np.nan, 3.0, 4.0],
            "p": [1.1, 2.0, np.nan, 5.0],
            "q10": [0.5, 1.0, 2.0, 3.0],
            "q90": [1.5, 3.0, 4.0, 6.0],
            "txt": ["1", "x", "3", "4"],
        },
        index=[2, 1, 3, 4],
    )


# ---------------- get_forecast_arrays ----------------


def test_gfa_errors_and_basic_paths():
    df = _mk_df()
    with pytest.raises(ValueError):
        get_forecast_arrays(df)

    y, q = get_forecast_arrays(
        df,
        actual_col="y",
        pred_cols=["q10", "q90"],
    )
    assert y.ndim == 1 and q.shape == (3, 2)

    y2 = get_forecast_arrays(
        df,
        actual_col="y",
        drop_na=False,
        return_as="numpy",
    )
    assert y2.shape[0] == 4

    p = get_forecast_arrays(
        df,
        pred_cols="p",
        return_as="numpy",
        squeeze=False,
    )
    assert p.shape == (3, 1)


def test_gfa_fillna_drop_policies_and_sort_copy():
    df = _mk_df()
    # ffill path
    _ = get_forecast_arrays(
        df,
        actual_col="y",
        pred_cols="p",
        fillna="ffill",
    )
    # bfill path + none policy, keep NaNs
    idx, y, p = get_forecast_arrays(
        df,
        actual_col="y",
        pred_cols="p",
        na_policy="none",
        with_index=True,
        sort_index=True,
    )
    assert (idx == np.array([1, 2, 3, 4])).all()
    assert len(y) == 4 and len(p) == 4

    # copy=False smoke (exercise branch)
    _ = get_forecast_arrays(
        df,
        actual_col="y",
        pred_cols=["q10", "q90"],
        copy=False,
    )


def test_gfa_numeric_enforce_and_dtype_and_pandas():
    df = _mk_df()
    # ensure_numeric raise
    with pytest.raises(ValueError):
        get_forecast_arrays(
            df,
            pred_cols="txt",
            ensure_numeric=True,
        )
    # coerce path
    s = get_forecast_arrays(
        df,
        pred_cols="txt",
        ensure_numeric=True,
        coerce_numeric=True,
        return_as="pandas",
    )
    assert isinstance(s, pd.Series)

    # pandas dtype cast with both outputs + index
    idx, y, p = get_forecast_arrays(
        df,
        actual_col="y",
        pred_cols="p",
        return_as="pandas",
        dtype=float,
        with_index=True,
    )
    assert isinstance(idx, pd.Index)
    assert y.dtype == float and p.dtype == float


# ---------------- probabilistic metrics ----------------


def test_compute_pit_and_crps_and_calib_error():
    y = np.array([10.0, 1.0, 5.5])
    qs = np.array([0.5, 0.1, 0.9])  # unsorted
    preds = np.array(
        [
            [11.0, 8.0, 13.0],
            [0.5, 0.0, 2.0],
            [5.0, 4.0, 6.0],
        ]
    )
    pit = compute_pit(y, preds, qs)
    assert pit.shape == (3,) and np.all(pit >= 0) and np.all(pit <= 1)

    crps = compute_crps(y, preds, np.sort(qs))
    assert crps > 0

    # calib error, small n => 1.0 path
    ce_small = calculate_calibration_error(
        np.array([1.0]),
        np.array([[0.0, 1.0, 2.0]]),
        np.array([0.1, 0.5, 0.9]),
    )
    assert ce_small == 1.0

    # larger n, just check in [0,1]
    y2 = np.linspace(0, 1, 20)
    preds2 = np.stack(
        [
            y2 - 0.1,
            y2,
            y2 + 0.1,
        ],
        axis=1,
    )
    ce = calculate_calibration_error(
        y2,
        preds2,
        np.array([0.1, 0.5, 0.9]),
    )
    assert 0 <= ce <= 1


def test_build_cdf_interpolator_and_errors():
    preds = np.array(
        [
            [8.0, 10.0, 12.0],
            [0.0, 1.0, 2.0],
        ]
    )
    qs = np.array([0.1, 0.5, 0.9])
    f = build_cdf_interpolator(preds, qs)
    out = f(np.array([10.0, 0.5]))
    assert np.all((out >= 0) & (out <= 1))

    with pytest.raises(ValueError):
        _ = f(np.array([1.0]))


# ---------------- interval / scores ----------------


def test_compute_coverage_score_modes_and_nan_and_errors():
    y = np.array([1, 2, 3, np.nan])
    lo = np.array([0, 2, 4, np.nan])
    hi = np.array([2, 3, 5, np.nan])

    cov = compute_coverage_score(y, lo, hi)
    assert 0 <= cov <= 1

    above = compute_coverage_score(
        y, lo, hi, method="above", return_counts=True
    )
    below = compute_coverage_score(
        y, lo, hi, method="below", return_counts=True
    )
    assert isinstance(above, int) and isinstance(below, int)

    z = np.array([np.nan, np.nan])
    assert compute_coverage_score(z, z, z) == 0.0
    assert compute_coverage_score(z, z, z, return_counts=True) == 0

    with pytest.raises(ValueError):
        compute_coverage_score(y, lo, hi, method="nope")


def test_compute_pinball_loss_and_edges():
    y = np.array([10, 10, 5], dtype=float)
    q = np.array([8, 12, 5], dtype=float)
    loss = compute_pinball_loss(y, q, 0.9)
    assert loss > 0

    with pytest.raises(ValueError):
        compute_pinball_loss(y, q, 1.1)

    nan_arr = np.array([np.nan, np.nan])
    res = compute_pinball_loss(nan_arr, nan_arr, 0.5)
    assert np.isnan(res)


def test_compute_winkler_score_normal_and_nan():
    y = np.array([1, 5, 12], dtype=float)
    lo = np.array([2, 4, 8], dtype=float)
    hi = np.array([8, 6, 10], dtype=float)
    s = compute_winkler_score(y, lo, hi, alpha=0.1)
    assert s > 0

    z = np.array([np.nan, np.nan])
    s_nan = compute_winkler_score(z, z, z)
    assert np.isnan(s_nan)


# ---------------- scaling ----------------


def test_minmax_scaler_2d_1d_y_and_errors_and_df():
    X = np.array([[1.0, 10.0], [1.0, 10.0], [3.0, 30.0]])
    Xs = minmax_scaler(X)
    assert Xs.shape == X.shape
    # zero variance col handled by eps (first col partly const)
    assert np.all((Xs >= 0) & (Xs <= 1))

    x1d = np.array([5.0, 6.0, 7.0])
    xs1d = minmax_scaler(x1d)
    assert xs1d.ndim == 1 and xs1d.shape == (3,)

    X2, y2 = minmax_scaler(
        X,
        x1d,
        feature_range=(-1.0, 1.0),
    )
    assert X2.min() >= -1 and X2.max() <= 1
    assert y2.min() >= -1 and y2.max() <= 1

    with pytest.raises(ValueError):
        _ = minmax_scaler(X, feature_range=(1.0,))

    # pandas input
    dfx = pd.DataFrame({"a": [1, 1, 2], "b": [0, 5, 5]})
    s = pd.Series([10, 20, 30])
    Xp, yp = minmax_scaler(dfx, s)
    assert Xp.shape == (3, 2) and yp.shape == (3,)
