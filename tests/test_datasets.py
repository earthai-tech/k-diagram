# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

"""
Pytest suite for testing dataset generation and loading functions
in kdiagram.datasets.
"""

import os
from unittest.mock import MagicMock, patch  # noqa

import numpy as np
import pandas as pd
import pytest

import kdiagram.datasets as kdd
from kdiagram.api.bunch import Bunch

@pytest.mark.parametrize("n_samples, n_periods", [(50, 3), (10, 1)])
def test_make_uncertainty_data(n_samples, n_periods):
    """Test make_uncertainty_data runs and returns DataFrame."""
    df = kdd.make_uncertainty_data(
        n_samples=n_samples, n_periods=n_periods, seed=42
    ).frame
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples
    # Check if expected number of columns are generated
    # 4 features + 1 actual + n_periods * 3 quantiles
    expected_cols = 4 + 1 + n_periods * 3
    assert df.shape[1] == expected_cols
    assert f"value_{2022}_q0.1" in df.columns  # Default prefix/year


@pytest.mark.parametrize("n_samples, n_models", [(60, 2), (20, 4)])
def test_make_taylor_data(n_samples, n_models):
    """Test make_taylor_data runs and returns Bunch with correct structure."""
    # Note: This function returns Bunch directly based on its signature now
    data_bunch = kdd.make_taylor_data(n_samples=n_samples, n_models=n_models, seed=101)
    assert isinstance(data_bunch, Bunch)
    assert hasattr(data_bunch, "reference")
    assert hasattr(data_bunch, "predictions")
    assert hasattr(data_bunch, "model_names")
    assert hasattr(data_bunch, "stats")
    assert hasattr(data_bunch, "ref_std")
    assert hasattr(data_bunch, "DESCR")
    assert isinstance(data_bunch.reference, np.ndarray)
    assert len(data_bunch.reference) == n_samples
    assert isinstance(data_bunch.predictions, list)
    assert len(data_bunch.predictions) == n_models
    assert len(data_bunch.model_names) == n_models
    assert isinstance(data_bunch.stats, pd.DataFrame)
    assert len(data_bunch.stats) == n_models
    assert isinstance(data_bunch.frame, pd.DataFrame)  # Check frame in Bunch
    assert len(data_bunch.frame) == n_samples
    assert data_bunch.frame.shape[1] == 1 + n_models  # ref + preds


@pytest.mark.parametrize("n_samples, n_models, num_q", [(70, 2, 3), (30, 1, 5)])
def test_make_multi_model_quantile_data(n_samples, n_models, num_q):
    """Test make_multi_model_quantile_data runs and returns DataFrame."""
    # Create quantiles list centered around 0.5
    quantiles = np.linspace(0.5 - 0.4 * num_q / 3, 0.5 + 0.4 * num_q / 3, num_q)
    quantiles = np.clip(np.round(quantiles, 2), 0.01, 0.99).tolist()
    if 0.5 not in quantiles:
        quantiles = sorted(quantiles + [0.5])

    df = kdd.make_multi_model_quantile_data(
        n_samples=n_samples,
        n_models=n_models,
        quantiles=quantiles,
        seed=202,
        as_frame=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples
    # 3 base features + n_models * len(quantiles) prediction columns
    expected_cols = 3 + n_models * len(quantiles)
    assert df.shape[1] == expected_cols
    try:
        assert f"pred_Model_A_q{quantiles[0]:.1f}" in df.columns
    except Exception:
        assert f"pred_Model_A_q{quantiles[0]:.2f}" in df.columns


@pytest.mark.parametrize("n_layers, n_features", [(4, 5), (2, 10)])
def test_make_fingerprint_data_as_bunch(n_layers, n_features):
    """Test make_fingerprint_data returns Bunch correctly."""
    data_bunch = kdd.make_fingerprint_data(
        n_layers=n_layers, n_features=n_features, seed=303, as_frame=False
    )
    assert isinstance(data_bunch, Bunch)
    assert hasattr(data_bunch, "importances")
    assert hasattr(data_bunch, "frame")
    assert hasattr(data_bunch, "layer_names")
    assert hasattr(data_bunch, "feature_names")
    assert hasattr(data_bunch, "DESCR")
    assert isinstance(data_bunch.importances, np.ndarray)
    assert data_bunch.importances.shape == (n_layers, n_features)
    assert isinstance(data_bunch.frame, pd.DataFrame)
    assert data_bunch.frame.shape == (n_layers, n_features)
    assert len(data_bunch.layer_names) == n_layers
    assert len(data_bunch.feature_names) == n_features


@pytest.mark.parametrize("n_samples, n_series", [(365, 2), (24, 2)])
def test_make_cyclical_data_as_bunch(n_samples, n_series):
    """Test make_cyclical_data returns Bunch correctly."""
    pred_bias = 0 if n_series == 1 else [0, 1.5]
    data_bunch = kdd.make_cyclical_data(
        n_samples=n_samples,
        n_series=n_series,
        seed=404,
        as_frame=False,
        pred_bias=pred_bias,
    )
    assert isinstance(data_bunch, Bunch)
    assert hasattr(data_bunch, "frame")
    assert hasattr(data_bunch, "target")
    assert hasattr(data_bunch, "target_names")
    assert hasattr(data_bunch, "feature_names")
    assert hasattr(data_bunch, "series_names")
    assert hasattr(data_bunch, "prediction_columns")
    assert hasattr(data_bunch, "DESCR")
    assert isinstance(data_bunch.frame, pd.DataFrame)
    assert len(data_bunch.frame) == n_samples
    assert len(data_bunch.target) == n_samples
    assert len(data_bunch.series_names) == n_series
    assert len(data_bunch.prediction_columns) == n_series
    # frame columns = target + features + predictions
    assert data_bunch.frame.shape[1] == 1 + 1 + n_series


def test_make_cyclical_data_as_frame():
    """Test make_cyclical_data returns DataFrame correctly."""
    df = kdd.make_cyclical_data(
        n_samples=50,
        n_series=1,
        as_frame=True,
        pred_bias=0,
        pred_noise_factor=1.5,
        pred_amplitude_factor=0.8,
        pred_phase_shift=np.pi / 6,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "y_true" in df.columns
    assert "time_step" in df.columns
    assert "model_A" in df.columns  # Default prefix and naming


def test_load_uncertainty_data_as_frame():
    """Test load_uncertainty_data returns DataFrame."""
    df = kdd.load_uncertainty_data(as_frame=True, n_samples=10, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "value_actual" in df.columns  # Default prefix


def test_load_uncertainty_data_as_bunch():
    """Test load_uncertainty_data returns Bunch with correct attributes."""
    n_samples = 20
    n_periods = 2
    bunch = kdd.load_uncertainty_data(
        as_frame=False, n_samples=n_samples, n_periods=n_periods, seed=2
    )
    assert isinstance(bunch, Bunch)
    assert hasattr(bunch, "frame")
    assert hasattr(bunch, "feature_names")
    assert hasattr(bunch, "target_names")
    assert hasattr(bunch, "target")
    assert hasattr(bunch, "quantile_cols")
    assert hasattr(bunch, "q10_cols")
    assert hasattr(bunch, "q50_cols")
    assert hasattr(bunch, "q90_cols")
    assert hasattr(bunch, "n_periods")
    assert hasattr(bunch, "prefix")
    assert hasattr(bunch, "start_year")
    assert hasattr(bunch, "DESCR")

    assert isinstance(bunch.frame, pd.DataFrame)
    assert len(bunch.frame) == n_samples
    assert isinstance(bunch.feature_names, list)
    assert isinstance(bunch.target_names, list)
    assert isinstance(bunch.target, np.ndarray)
    assert len(bunch.target) == n_samples
    assert isinstance(bunch.quantile_cols, dict)
    assert len(bunch.q10_cols) == n_periods
    assert len(bunch.q50_cols) == n_periods
    assert len(bunch.q90_cols) == n_periods
    assert bunch.n_periods == n_periods
    assert isinstance(bunch.DESCR, str)


# # --- Tests for load_zhongshan_subsidence ---

# Create a dummy DataFrame to be returned by mocked pd.read_csv
DUMMY_ZHONGSHAN_COLS = [
    "longitude",
    "latitude",
    "subsidence_2022",
    "subsidence_2023",
    "subsidence_2022_q0.1",
    "subsidence_2022_q0.5",
    "subsidence_2022_q0.9",
    "subsidence_2023_q0.1",
    "subsidence_2023_q0.5",
    "subsidence_2023_q0.9",
    "subsidence_2024_q0.1",
    "subsidence_2024_q0.5",
    "subsidence_2024_q0.9",
    "subsidence_2025_q0.1",
    "subsidence_2025_q0.5",
    "subsidence_2025_q0.9",
    "subsidence_2026_q0.1",
    "subsidence_2026_q0.5",
    "subsidence_2026_q0.9",
]
dummy_df = pd.DataFrame(
    np.arange(10 * len(DUMMY_ZHONGSHAN_COLS)).reshape(10, -1),
    columns=DUMMY_ZHONGSHAN_COLS,
)


@patch("kdiagram.datasets.load.pd.read_csv", return_value=dummy_df.copy())
@patch("kdiagram.datasets.load.os.path.exists")
@patch("kdiagram.datasets.load.download_file_if")
@patch("kdiagram.datasets.load.resources.is_resource")
def test_load_zhongshan_from_cache(
    mock_is_resource, mock_download, mock_exists, mock_read_csv, tmp_path
):
    """Test loading from cache when file exists."""
    # Simulate file existing in cache path
    mock_exists.return_value = True
    # Simulate resource *not* existing in package (to force cache check first)
    mock_is_resource.return_value = False
    # Define cache path for the mock
    cache_dir = str(tmp_path / "kdiagram_data")
    expected_path = os.path.join(cache_dir, "min_zhongshan.csv")

    with patch("kdiagram.datasets.load.get_data", return_value=cache_dir):
        data = kdd.load_zhongshan_subsidence(as_frame=True)

    # Assertions
    # mock_exists.assert_called_once_with(expected_path)
    mock_is_resource.assert_not_called()  # Should not check package
    mock_download.assert_not_called()  # Should not download
    mock_read_csv.assert_called_once_with(expected_path)  # Loaded from cache
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 10  # Length of dummy df


# Mock resources.path context manager
@pytest.fixture
def mock_resource_path(tmp_path):
    """Fixture to mock importlib.resources.path context manager."""
    dummy_pkg_path = tmp_path / "dummy_pkg_data"
    dummy_pkg_path.mkdir()
    dummy_file = dummy_pkg_path / "min_zhongshan.csv"
    dummy_file.touch()  # Create dummy file

    class MockPathManager:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self.path

        def __exit__(self, *args):
            pass  # No cleanup needed for test

    with patch(
        "kdiagram.datasets.load.resources.path",
        return_value=MockPathManager(dummy_file),
    ) as mock_cm:
        yield mock_cm


@pytest.mark.network
@patch("kdiagram.datasets.load.pd.read_csv", return_value=dummy_df.copy())
@patch("kdiagram.datasets.load.os.path.exists")
@patch("kdiagram.datasets.load.download_file_if")
@patch("kdiagram.datasets.load.resources.is_resource")
@patch("kdiagram.datasets.load.shutil.copyfile")
def test_load_zhongshan_from_package(
    mock_copy,
    mock_is_resource,
    mock_download,
    mock_exists,
    mock_read_csv,
    mock_resource_path,
    tmp_path,
):
    """Test loading from package resources when not in cache."""
    cache_dir = str(tmp_path / "kdiagram_data")
    expected_cache_path = os.path.join(cache_dir, "min_zhongshan.csv")
    package_file_path = str(mock_resource_path.return_value.path)

    # Make the mock return True only for the package path,
    # and False for the cache path.
    mock_exists.side_effect = lambda path: path == package_file_path
    
    mock_is_resource.return_value = True

    with patch("kdiagram.datasets.load.get_data", return_value=cache_dir):
        data = kdd.load_zhongshan_subsidence(
            as_frame=False,
            download_if_missing=True,
        )

    # Assertions
    mock_exists.assert_any_call(expected_cache_path) # Checked cache (returned False)
    mock_is_resource.assert_called_once()
    mock_resource_path.assert_called_once()
    mock_copy.assert_called_once()
    mock_download.assert_not_called()
    mock_read_csv.assert_called_once_with(package_file_path)
    assert isinstance(data, Bunch)
    

@pytest.mark.network
@patch("kdiagram.datasets.load.pd.read_csv", return_value=dummy_df.copy())
@patch("kdiagram.datasets.load.os.path.exists")
@patch("kdiagram.datasets.load.download_file_if")
@patch("kdiagram.datasets.load.resources.is_resource")
def test_load_zhongshan_from_download(
    mock_is_resource, mock_download, mock_exists, mock_read_csv, tmp_path
):
    """Test loading via download when not in cache or package."""
    cache_dir = str(tmp_path / "kdiagram_data")
    expected_path = os.path.join(cache_dir, "min_zhongshan.csv")

    # Create a stateful mock for os.path.exists.
    # It will return True only after the download has been simulated.
    _file_downloaded = False
    def exists_side_effect(path):
        # Only return True for the cache path if the download has "occurred"
        if path == expected_path:
            return _file_downloaded
        return False

    def download_side_effect(*args, **kwargs):
        nonlocal _file_downloaded
        # Simulate the download succeeding and the file now existing
        _file_downloaded = True
        return expected_path

    mock_exists.side_effect = exists_side_effect
    mock_download.side_effect = download_side_effect
    mock_is_resource.return_value = False

    with patch("kdiagram.datasets.load.get_data", return_value=cache_dir):
        data = kdd.load_zhongshan_subsidence(as_frame=True)

    # Assertions
    mock_is_resource.assert_called_once()
    mock_download.assert_called_once()
    mock_read_csv.assert_called_once_with(expected_path)
    assert isinstance(data, pd.DataFrame)

@patch("kdiagram.datasets.load.pd.read_csv", return_value=dummy_df.copy())
@patch("kdiagram.datasets.load.os.path.exists", return_value=True)
def test_load_zhongshan_subsetting(mock_exists, mock_read_csv, tmp_path):
    """Test subsetting options (years, quantiles, flags)."""
    cache_dir = str(tmp_path / "kdiagram_data")
    with patch("kdiagram.datasets.load.get_data", return_value=cache_dir):
        # Test year subsetting
        data_bunch = kdd.load_zhongshan_subsidence(
            years=[2023, 2025], quantiles=[0.1, 0.9], as_frame=False
        )

    assert isinstance(data_bunch, Bunch)
    assert "subsidence_2023" in data_bunch.frame.columns
    assert "subsidence_2022" not in data_bunch.frame.columns  # Check exclusion
    assert "subsidence_2023_q0.1" in data_bunch.frame.columns
    assert "subsidence_2025_q0.9" in data_bunch.frame.columns
    assert "subsidence_2024_q0.5" not in data_bunch.frame.columns  # Check exclusion
    assert (
        "subsidence_2023_q0.5" not in data_bunch.frame.columns
    )  # Check quantile exclusion
    assert "longitude" in data_bunch.frame.columns  # Default include coords

    # Test flag subsetting
    with patch("kdiagram.datasets.load.get_data", return_value=cache_dir):
        df_no_coords = kdd.load_zhongshan_subsidence(
            include_coords=False, include_target=False, as_frame=True
        )

    assert isinstance(df_no_coords, pd.DataFrame)
    assert "longitude" not in df_no_coords.columns
    assert "latitude" not in df_no_coords.columns
    assert "subsidence_2022" not in df_no_coords.columns
    assert "subsidence_2023" not in df_no_coords.columns
    assert "subsidence_2022_q0.1" in df_no_coords.columns  # Quantiles still present


if __name__ == "__main__":
    pytest.main([__file__])
