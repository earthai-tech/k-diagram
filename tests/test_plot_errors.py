import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from kdiagram.plot.errors import (
    plot_error_bands,
    plot_error_ellipses,
    plot_error_violins,
)

plt.switch_backend("Agg")


@pytest.fixture
def seasonal_error_data():
    """Fixture for plot_error_bands data."""
    np.random.seed(42)
    n_points = 500
    day_of_year = np.arange(n_points) % 365
    month = (day_of_year // 30) + 1
    errors = np.sin((day_of_year - 90) * np.pi / 180) * 5 + np.random.randn(
        n_points
    )
    df = pd.DataFrame({"month": month, "forecast_error": errors})
    # Add some NaNs to test handling
    df.loc[10:20, "forecast_error"] = np.nan
    return df


@pytest.fixture
def multi_model_error_data():
    """Fixture for plot_error_violins data."""
    np.random.seed(0)
    n_points = 200
    df = pd.DataFrame(
        {
            "Model_A_Error": np.random.normal(
                loc=0, scale=1.5, size=n_points
            ),
            "Model_B_Error": np.random.normal(
                loc=-3.0, scale=1.0, size=n_points
            ),
            "Model_C_Error": np.random.normal(
                loc=1.0, scale=3.0, size=n_points
            ),
        }
    )
    # Add NaNs to one column
    df.loc[5:15, "Model_B_Error"] = np.nan
    return df


@pytest.fixture
def positional_error_data():
    """Fixture for plot_polar_error_ellipses data."""
    np.random.seed(1)
    n_points = 10
    df = pd.DataFrame(
        {
            "angle_deg": np.linspace(0, 360, n_points, endpoint=False),
            "distance_km": np.random.uniform(20, 80, n_points),
            "distance_std": np.random.uniform(2, 7, n_points),
            "angle_std_deg": np.random.uniform(3, 10, n_points),
            "priority": np.random.randint(1, 5, n_points),
        }
    )
    # Add NaNs to test handling
    df.loc[2, "distance_std"] = np.nan
    return df


# --- Tests for plot_error_bands ---


def test_plot_error_bands_runs_successfully(seasonal_error_data):
    """Test that plot_error_bands runs without errors and returns an Axes object."""
    ax = plot_error_bands(
        df=seasonal_error_data,
        error_col="forecast_error",
        theta_col="month",
        theta_period=12,
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_plot_error_bands_with_masking(seasonal_error_data):
    """Test the mask_angle parameter in plot_error_bands."""
    ax = plot_error_bands(
        df=seasonal_error_data,
        error_col="forecast_error",
        theta_col="month",
        theta_period=12,
        mask_angle=True,
    )
    # Check that angular tick labels are empty
    assert all(label.get_text() == "" for label in ax.get_xticklabels())
    plt.close()


def test_plot_error_bands_empty_df():
    """Test that plot_error_bands handles an empty DataFrame gracefully."""
    with pytest.warns(UserWarning, match="DataFrame is empty"):
        ax = plot_error_bands(
            df=pd.DataFrame({"error": [], "theta": []}),
            error_col="error",
            theta_col="theta",
        )
        assert ax is None


# --- Tests for plot_error_violins ---


def test_plot_error_violins_runs_successfully(multi_model_error_data):
    """Test that plot_error_violins runs without errors and returns an Axes object."""
    ax = plot_error_violins(
        multi_model_error_data,
        "Model_A_Error",
        "Model_B_Error",
        "Model_C_Error",
        names=["Model A", "Model B", "Model C"],
    )
    assert isinstance(ax, Axes)
    # Check if labels are correctly set
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert "Model A" in xticklabels
    assert "Model B" in xticklabels
    assert "Model C" in xticklabels
    plt.close()


def test_plot_error_violins_no_cols_raises_error(multi_model_error_data):
    """Test that plot_error_violins raises a ValueError if no columns are provided."""
    with pytest.raises(
        ValueError, match="At least one error column must be provided"
    ):
        plot_error_violins(multi_model_error_data)


def test_plot_error_violins_mismatched_names_warns(multi_model_error_data):
    """Test that a warning is issued for mismatched names."""
    with pytest.warns(UserWarning, match="Names length does not"):
        ax = plot_error_violins(
            multi_model_error_data,
            "Model_A_Error",
            "Model_B_Error",
            names=["Only One Name"],
        )
    assert isinstance(ax, Axes)
    plt.close()


# --- Tests for plot_polar_error_ellipses ---


def test_plot_polar_error_ellipses_runs_successfully(positional_error_data):
    """Test that plot_polar_error_ellipses runs and returns an Axes object."""
    ax = plot_error_ellipses(
        df=positional_error_data,
        r_col="distance_km",
        theta_col="angle_deg",
        r_std_col="distance_std",
        theta_std_col="angle_std_deg",
    )
    assert isinstance(ax, Axes)
    plt.close()


@pytest.mark.parametrize("n_std", [1.0, 2.5])
def test_plot_polar_error_ellipses_n_std(positional_error_data, n_std):
    """Test the n_std parameter."""
    ax = plot_error_ellipses(
        df=positional_error_data,
        r_col="distance_km",
        theta_col="angle_deg",
        r_std_col="distance_std",
        theta_std_col="angle_std_deg",
        n_std=n_std,
        title=f"{n_std}-Sigma Ellipses",
    )
    assert isinstance(ax, Axes)
    assert f"{n_std:.1f}" in ax.get_title()
    plt.close()


def test_plot_polar_error_ellipses_masking(positional_error_data):
    """Test the mask_angle and mask_radius parameters."""
    ax = plot_error_ellipses(
        df=positional_error_data,
        r_col="distance_km",
        theta_col="angle_deg",
        r_std_col="distance_std",
        theta_std_col="angle_std_deg",
        mask_angle=True,
        mask_radius=True,
    )
    assert all(label.get_text() == "" for label in ax.get_xticklabels())
    assert all(label.get_text() == "" for label in ax.get_yticklabels())
    plt.close()


def test_plot_polar_error_ellipses_with_color_col(positional_error_data):
    """Test the custom color_col parameter."""
    ax = plot_error_ellipses(
        df=positional_error_data,
        r_col="distance_km",
        theta_col="angle_deg",
        r_std_col="distance_std",
        theta_std_col="angle_std_deg",
        color_col="priority",
    )
    assert isinstance(ax, Axes)
    # A simple check that a colorbar was created
    assert len(ax.figure.axes) > 1
    plt.close()
