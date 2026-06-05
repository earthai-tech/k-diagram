import numpy as np
import pandas as pd
import pytest

from kdiagram.metrics import (
    cluster_aware_severity_score,
    clustered_anomaly_severity,
)

# ---- Test Data Fixtures ----


@pytest.fixture
def basic_data():
    """A simple case with a mix of anomalies and correct points."""
    return {
        "y_true": np.array([10, 15, 20, 25, 30, 35, 40]),
        "y_qlow": np.array([8, 14, 22, 24, 28, 36, 38]),
        "y_qup": np.array([12, 16, 23, 26, 32, 37, 42]),
    }


@pytest.fixture
def clustered_data():
    """Data where anomalies are clustered together by default."""
    y_true = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    # Anomalies at indices 2, 3, 4
    y_qlow = np.array([8, 18, 32, 42, 52, 58, 68, 78])
    y_qup = np.array([12, 22, 33, 43, 53, 62, 72, 82])
    # sort_by will spread the anomalies out
    sort_by = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    return {
        "y_true": y_true,
        "y_qlow": y_qlow,
        "y_qup": y_qup,
        "sort_by": sort_by,
    }


# ---- Tests for compute_clustered_anomaly_severity ----


def test_compute_cas_no_anomalies():
    """Score should be 0.0 if all points are covered."""
    y_true = np.array([10, 20, 30])
    y_qlow = np.array([8, 18, 28])
    y_qup = np.array([12, 22, 32])
    score = clustered_anomaly_severity(y_true, y_qlow, y_qup)
    assert score == 0.0


def test_compute_cas_basic_calculation(basic_data):
    """Test the score calculation on a simple case."""
    score = clustered_anomaly_severity(
        basic_data["y_true"],
        basic_data["y_qlow"],
        basic_data["y_qup"],
        window_size=3,
    )
    # Anomalies at t=2 (under, e=2) and t=5 (under, e=1); widths=1.
    # Triangular kernel w=3 → off-center weights are 0 after normalisation,
    # so self-exclusion leaves d=0 for both (isolated).
    # Severity=[0,0,2,0,0,1,0]; CAS = (2+1)/7 = 3/7.
    assert np.isclose(score, 3 / 7, atol=1e-5)


def test_compute_cas_return_details(basic_data):
    """Test the return_details flag."""
    result = clustered_anomaly_severity(
        basic_data["y_true"],
        basic_data["y_qlow"],
        basic_data["y_qup"],
        return_details=True,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], pd.DataFrame)
    assert "severity" in result[1].columns


def test_compute_cas_input_type_validation():
    """Test that non-numpy array inputs raise TypeError."""
    with pytest.raises(TypeError):
        clustered_anomaly_severity("ten", [8], [12])  # mixing str and numbers


def test_cas_score_no_anomalies():
    """Score should be 0.0 for the scikit-learn API version."""
    y_true = np.array([10, 20, 30])
    y_pred = np.array([[8, 12], [18, 22], [28, 32]])
    score = cluster_aware_severity_score(y_true, y_pred)
    assert score == 0.0


def test_cas_score_basic_calculation(basic_data):
    """Test basic score matches the helper function."""
    y_pred = np.c_[basic_data["y_qlow"], basic_data["y_qup"]]
    score = cluster_aware_severity_score(
        basic_data["y_true"], y_pred, window_size=3
    )
    assert np.isclose(score, 3 / 7, atol=1e-5)


def test_cas_score_incorrect_pred_shape():
    """Test that y_pred with wrong shape raises ValueError."""
    y_true = np.array([10, 20])
    with pytest.raises(ValueError):
        # 1D y_pred
        cluster_aware_severity_score(y_true, np.array([8, 18]))
    with pytest.raises(ValueError):
        # 3-column y_pred
        cluster_aware_severity_score(
            y_true, np.array([[8, 10, 12], [18, 20, 22]])
        )


def test_cas_score_sample_weight(basic_data):
    """Test that sample_weight correctly influences the score."""
    y_true = basic_data["y_true"]
    y_pred = np.c_[basic_data["y_qlow"], basic_data["y_qup"]]
    weights = np.array([1, 1, 100, 1, 1, 1, 1])

    # Use box kernel explicitly so density is non-zero and score is
    # independent of default kernel changes.
    score_unweighted = cluster_aware_severity_score(
        y_true, y_pred, window_size=3, kernel="box"
    )
    score_weighted = cluster_aware_severity_score(
        y_true, y_pred, sample_weight=weights, window_size=3, kernel="box"
    )

    assert score_weighted > score_unweighted

    # Violations at t=2 (e=2, d=0) and t=5 (e=1, d=0); box w=3 self-excl.
    # S=[0,0,2,0,0,1,0]; weighted avg = (100*2 + 1*1) / 106 = 201/106.
    assert np.isclose(score_weighted, 201 / 106, atol=1e-5)


def test_cas_score_sort_by_effect(clustered_data):
    """Test that `sort_by` correctly changes the score."""
    y_true = clustered_data["y_true"]
    y_pred = np.c_[clustered_data["y_qlow"], clustered_data["y_qup"]]
    sort_by_vec = clustered_data["sort_by"]

    # Use box kernel (w=3) so neighbours are visible and clustering matters.
    # Without sorting, anomalies at idx 2,3,4 are adjacent → positive density.
    score_clustered = cluster_aware_severity_score(
        y_true, y_pred, window_size=3, kernel="box"
    )

    # With sorting, anomalies are spread >1 step apart → density=0.
    score_scattered = cluster_aware_severity_score(
        y_true, y_pred, sort_by=sort_by_vec, window_size=3, kernel="box"
    )

    assert score_clustered > score_scattered

    # box w=3 self-excl: d[2]=0.5, d[3]=1.0, d[4]=0.5; e=2 everywhere.
    # S[2]=2*1.5=3, S[3]=2*2=4, S[4]=2*1.5=3; CAS=(3+4+3)/8=10/8=1.25.
    assert np.isclose(score_clustered, 1.25, atol=1e-5)
    # Scattered: d=0 for all three; S=2*1=2 each; CAS=(2+2+2)/8=0.75.
    assert np.isclose(score_scattered, 0.75, atol=1e-5)


def test_cas_score_return_details_sklearn(basic_data):
    """Test the return_details flag for the sklearn API."""
    y_pred = np.c_[basic_data["y_qlow"], basic_data["y_qup"]]
    result = cluster_aware_severity_score(
        basic_data["y_true"], y_pred, return_details=True
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], pd.DataFrame)
    assert "severity" in result[1].columns
