# Tests for kdiagram.metrics (CAS)
# Aim: >= 98% coverage

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kdiagram.metrics import (
    cluster_aware_severity_score,
    clustered_anomaly_severity,
)

# -----------------------------
# Helpers
# -----------------------------


def _toy_series(n=32, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1.0, n)
    lo = y - 0.5
    up = y + 0.5
    return y, lo, up


def _inject_runs(y, lo, up, idx):
    # push these points outside the band
    y2 = y.copy()
    y2[idx] = up[idx] + 0.8
    return y2, lo, up


# -----------------------------
# _split_bounds coverage via API
# -----------------------------


def test_split_bounds_2d():
    y = np.array([1, 2, 3, 4])
    yp = np.c_[y - 1, y + 1]
    s = cluster_aware_severity_score(y, yp)
    assert isinstance(s, float)


def test_split_bounds_3d_multioutput_raw_values():
    n = 16
    y = np.arange(n, dtype=float)
    lo = np.c_[y - 1, y - 2]
    up = np.c_[y + 1, y + 2]
    y_true = np.c_[y, y]  # 2 outputs
    y_pred = np.stack(
        [np.c_[lo[:, 0], up[:, 0]], np.c_[lo[:, 1], up[:, 1]]], axis=1
    )
    s_raw, details = cluster_aware_severity_score(
        y_true,
        y_pred,
        multioutput="raw_values",
        return_details=True,
    )
    assert isinstance(s_raw, np.ndarray)
    assert s_raw.shape == (2,)
    assert isinstance(details, list) and len(details) == 2
    assert {"y_true", "y_qlow", "y_qup"}.issubset(set(details[0].columns))


def test_split_bounds_wide_matrix():
    n = 10
    y = np.arange(n, dtype=float)
    y_true = y
    # (n, 4) -> two outputs (L1,U1,L2,U2)
    y_pred = np.c_[y - 1.0, y + 1.0, y - 2.0, y + 2.0]
    s = cluster_aware_severity_score(
        y_true, y_pred, multioutput="uniform_average"
    )
    assert isinstance(s, float)


def test_split_bounds_mismatch_raises():
    n = 8
    y_true = np.arange(n, dtype=float).reshape(-1, 1)
    # single set of bounds -> mismatch with 1 output?
    # here we trick with 2 outputs requested by shape (n,2,2)
    # but pass only one pair; should raise
    with pytest.raises(ValueError):
        cluster_aware_severity_score(
            y_true,
            np.c_[y_true.ravel() - 1],  # y_true.ravel() + 1
        )


# -----------------------------
# clustered_anomaly_severity API
# -----------------------------


def test_clustered_anomaly_arrays_basic_and_details():
    y, lo, up = _toy_series(n=16, seed=1)
    # no anomalies -> near zero
    s0 = clustered_anomaly_severity(y, lo, up, window_size=3)
    assert isinstance(s0, float) and s0 >= 0.0

    # inject a run -> positive and details
    idx = np.arange(6, 10)
    y2, lo2, up2 = _inject_runs(y, lo, up, idx)
    s, det = clustered_anomaly_severity(
        y2, lo2, up2, window_size=3, return_details=True
    )
    assert s > s0
    assert {"is_anomaly", "local_density", "severity"}.issubset(
        set(det.columns)
    )
    assert det.loc[idx, "is_anomaly"].all()


def test_clustered_anomaly_dataframe_mode():
    y, lo, up = _toy_series(n=12, seed=2)
    df = pd.DataFrame({"yt": y, "ql": lo, "qu": up})
    s, det = clustered_anomaly_severity(
        "yt",
        "ql",
        "qu",
        data=df,
        window_size=5,
        return_details=True,
    )
    assert isinstance(s, float)
    assert len(det) == len(df)


def test_clustered_anomaly_bad_signature_raises():
    y, lo, up = _toy_series(n=8, seed=3)
    df = pd.DataFrame({"yt": y, "ql": lo, "qu": up})
    with pytest.raises(TypeError):
        clustered_anomaly_severity(y, "ql", "qu", data=None)
    with pytest.raises(TypeError):
        clustered_anomaly_severity("yt", lo, up, data=df)


# -----------------------------
# cluster_aware_severity_score
# branches + options
# -----------------------------


@pytest.mark.parametrize("normalize", ["band", "mad", "none"])
@pytest.mark.parametrize("kernel", ["box", "triangular", "epan", "gaussian"])
@pytest.mark.parametrize("density_source", ["indicator", "magnitude"])
def test_cas_core_options(normalize, kernel, density_source):
    y, lo, up = _toy_series(n=64, seed=4)
    # make two separated runs for signal
    idx = np.r_[10:16, 40:46]
    y2, lo2, up2 = _inject_runs(y, lo, up, idx)
    s = cluster_aware_severity_score(
        y2,
        np.c_[lo2, up2],
        window_size=7,
        normalize=normalize,
        density_source=density_source,
        kernel=kernel,
        lambda_=1.3,
        gamma=1.7,
    )
    assert isinstance(s, float) and s >= 0.0


def test_kernel_invalid_raises():
    y, lo, up = _toy_series(n=16, seed=5)
    with pytest.raises(ValueError):
        cluster_aware_severity_score(y, np.c_[lo, up], kernel="bad")


def test_sorting_changes_clustering_signal():
    # same anomalies, different order -> CAS should change
    y, lo, up = _toy_series(n=30, seed=6)
    idx = np.r_[5:9, 20:24]
    y2, lo2, up2 = _inject_runs(y, lo, up, idx)
    # unsorted
    s_uns = cluster_aware_severity_score(y2, np.c_[lo2, up2], window_size=5)
    # sort key brings runs together
    # place all anomaly idx at the end
    sort_key = np.arange(len(y2))
    sort_key[idx] = np.arange(len(y2) - len(idx), len(y2))
    s_sort = cluster_aware_severity_score(
        y2,
        np.c_[lo2, up2],
        window_size=5,
        sort_by=sort_key,
    )
    assert s_sort != s_uns


def test_sample_weight_effect():
    y, lo, up = _toy_series(n=12, seed=7)
    idx = np.r_[4:8]
    y2, lo2, up2 = _inject_runs(y, lo, up, idx)
    w = np.ones_like(y2)
    w[idx] = 10.0
    s_w = cluster_aware_severity_score(y2, np.c_[lo2, up2], sample_weight=w)
    s = cluster_aware_severity_score(y2, np.c_[lo2, up2])
    # weighted should be >= unweighted
    assert s_w >= s


def test_multioutput_aggregation_modes():
    n = 20
    y = np.linspace(0.0, 1.0, n)
    y_true = np.c_[y, y]
    lo1, up1 = y - 0.2, y + 0.2
    lo2, up2 = y - 0.1, y + 0.1
    y_pred = np.stack([np.c_[lo1, up1], np.c_[lo2, up2]], axis=1)
    s_raw = cluster_aware_severity_score(
        y_true, y_pred, multioutput="raw_values"
    )
    s_avg = cluster_aware_severity_score(
        y_true, y_pred, multioutput="uniform_average"
    )
    assert isinstance(s_raw, np.ndarray) and s_raw.shape == (2,)
    assert math.isclose(float(np.mean(s_raw)), s_avg, rel_tol=1e-12)


def test_nan_policy_variants():
    y, lo, up = _toy_series(n=10, seed=8)
    y[3] = np.nan
    with pytest.raises(ValueError):
        cluster_aware_severity_score(y, np.c_[lo, up], nan_policy="raise")
    s = cluster_aware_severity_score(y, np.c_[lo, up], nan_policy="omit")
    assert isinstance(s, float) and not math.isnan(s)
    s2, det = cluster_aware_severity_score(
        y,
        np.c_[lo, up],
        nan_policy="propagate",
        return_details=True,
    )
    assert math.isnan(s2) and det is None


def test_eps_avoids_div_zero_when_band_zero():
    n = 12
    y = np.zeros(n)
    lo = np.zeros(n)
    up = np.zeros(n)  # width 0
    # push two points outside
    y[2] = 1.0
    y[9] = -1.0
    s = cluster_aware_severity_score(
        y, np.c_[lo, up], window_size=3, eps=1e-9
    )
    assert s >= 0.0 and np.isfinite(s)


def test_density_source_magnitude_vs_indicator():
    n = 24
    y = np.zeros(n)
    lo = -np.ones(n) * 0.1
    up = np.ones(n) * 0.1
    # single large miss
    y[12] = 10.0
    s_ind = cluster_aware_severity_score(
        y,
        np.c_[lo, up],
        density_source="indicator",
        window_size=5,
    )
    s_mag = cluster_aware_severity_score(
        y,
        np.c_[lo, up],
        density_source="magnitude",
        window_size=5,
    )
    # magnitude source should penalize large single miss more
    assert s_mag >= s_ind


def test_return_details_df_schema_and_types():
    y, lo, up = _toy_series(n=18, seed=9)
    s, det = cluster_aware_severity_score(
        y,
        np.c_[lo, up],
        return_details=True,
    )
    assert isinstance(s, float)
    cols = {
        "y_true",
        "y_qlow",
        "y_qup",
        "is_anomaly",
        "type",
        "magnitude",
        "local_density",
        "severity",
    }
    assert cols.issubset(set(det.columns))
    assert det["is_anomaly"].dtype == bool
    assert (det["type"].isin(["none", "under", "over"])).all()
