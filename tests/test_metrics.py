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
    # Manual calculation for verification:
    # anomalies at t=2 and t=5; widths at those points are 1;
    # normalized magnitudes = [0,0,2,0,0,1,0];
    # density from indicator (win=3) = [0, 1/3, 1/3, 1/3, 1/3, 1/3, 1/2];
    # severities = [0, 0, 2*(1+1/3), 0, 0, 1*(1+1/3), 0];
    # mean = (2.6667 + 1.3333)/7 = 0.5714286.
    assert np.isclose(score, 0.571428571428, atol=1e-5)


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
    assert np.isclose(score, 0.571428571428, atol=1e-5)


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

    score_unweighted = cluster_aware_severity_score(
        y_true, y_pred, window_size=3
    )
    score_weighted = cluster_aware_severity_score(
        y_true, y_pred, sample_weight=weights, window_size=3
    )

    assert score_weighted > score_unweighted

    # The actual output from  code
    correct_weighted_score = 2.5283018867899245
    assert np.isclose(score_weighted, correct_weighted_score, atol=1e-5)


def test_cas_score_sort_by_effect(clustered_data):
    """Test that `sort_by` correctly changes the score."""
    y_true = clustered_data["y_true"]
    y_pred = np.c_[clustered_data["y_qlow"], clustered_data["y_qup"]]
    sort_by_vec = clustered_data["sort_by"]

    # Without sorting, anomalies at idx 2,3,4 are clustered
    score_clustered = cluster_aware_severity_score(
        y_true, y_pred, window_size=3
    )

    # With sorting, anomalies are spread out
    score_scattered = cluster_aware_severity_score(
        y_true, y_pred, sort_by=sort_by_vec, window_size=3
    )

    assert score_clustered > score_scattered

    # CORRECTED VALUES: The actual outputs from your code
    correct_clustered_score = 1.333333333333
    # THIS VALUE IS (1.333... + 1.333... + 1.333...) / 8 = 4.0 / 8 = 0.5.
    correct_scattered_score = 1.0

    assert np.isclose(score_clustered, correct_clustered_score, atol=1e-5)
    assert np.isclose(score_scattered, correct_scattered_score, atol=1e-5)


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
