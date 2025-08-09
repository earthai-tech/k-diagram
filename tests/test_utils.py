import pandas as pd
import pytest

from kdiagram.compat.numpy import NP_INT
from kdiagram.utils.q_utils import (
    melt_q_data,
    pivot_q_data,
    reshape_quantile_data,
)

# --- Fixtures ----------------------------------------------------------------

@pytest.fixture
def wide_df_full():
    # Two spatial points, two years, two quantiles each year
    return pd.DataFrame({
        "lon": [-118.25, -118.30],
        "lat": [34.05, 34.10],
        "subs_2022_q0.1": [1.2, 1.3],
        "subs_2022_q0.5": [1.5, 1.6],
        "subs_2023_q0.1": [1.7, 1.8],
        "subs_2023_q0.5": [2.0, 2.2],
    })

@pytest.fixture
def wide_df_partial():
    # One quantile missing in 2023 on purpose
    return pd.DataFrame({
        "lon": [-118.25, -118.30],
        "lat": [34.05, 34.10],
        "subs_2022_q0.1": [1.2, 1.3],
        "subs_2022_q0.5": [1.5, 1.6],
        "subs_2023_q0.1": [1.7, 1.8],
    })

# --- reshape_quantile_data ---------------------------------------------------

def test_reshape_basic(wide_df_full):
    out = reshape_quantile_data(
        wide_df_full, value_prefix="subs",
        spatial_cols=["lon", "lat"],
        dt_col="year",
        error="raise",
    )
    # Columns present and in expected naming scheme
    assert {"lon", "lat", "year", "subs_q0.1", "subs_q0.5"} <= set(out.columns)
    # Year is sorted ascending
    assert list(NP_INT(out["year"].unique())) == [2022, 2023]
    # Shape: 2 spatial points * 2 years = 4 rows
    assert out.shape[0] == 4

def test_reshape_missing_spatial_warn(wide_df_full):
    with pytest.warns(UserWarning, match="Missing spatial columns"):
        out = reshape_quantile_data(
            wide_df_full, value_prefix="subs",
            spatial_cols=["lon", "lat", "oops"],
            dt_col="year",
            error="warn",
        )
    assert "year" in out.columns
    # Dropped the nonexistent spatial col
    assert "oops" not in out.columns

def test_reshape_no_matching_cols_error():
    df = pd.DataFrame({"lon": [0], "lat": [0], "something_else": [1]})
    with pytest.raises(ValueError, match="No columns found with prefix 'subs'"):
        reshape_quantile_data(df, "subs", error="raise")

# --- melt_q_data -------------------------------------------------------------

def test_melt_basic_no_spatial(wide_df_partial):
    out = melt_q_data(
        wide_df_partial, value_prefix="subs",
        dt_name="year", spatial_cols=None, error="raise"
    )
    # Has year and quantile columns (converted to subs_qX)
    assert {"year", "subs_q0.1", "subs_q0.5"} <= set(out.columns)
    # One row per year since we didn’t carry spatial cols
    assert list(NP_INT(out["year"].unique())) == [2022, 2023]

def test_melt_with_spatial_and_filter_q(wide_df_full):
    out = melt_q_data(
        wide_df_full, value_prefix="subs",
        dt_name="year", spatial_cols=("lon", "lat"),
        q=[0.1],  # keep only 0.1
        error="raise"
    )
    # Only 0.1 column should be present + spatial + dt
    assert {"lon", "lat", "year", "subs_q0.1"} <= set(out.columns)
    assert "subs_q0.5" not in out.columns
    # 2 spatial * 2 years = 4 rows
    assert out.shape[0] == 4

def test_melt_no_matches_ignore_returns_empty():
    df = pd.DataFrame({"foo": [1, 2]})
    out = melt_q_data(df, value_prefix="subs", error="ignore")
    assert isinstance(out, pd.DataFrame)
    assert out.empty

# --- pivot_q_data ------------------------------------------------------------

def test_pivot_roundtrip(wide_df_full):
    # melt -> pivot should rebuild the wide names for both years/quantiles
    long_df = melt_q_data(
        wide_df_full, value_prefix="subs",
        dt_name="year", spatial_cols=("lon", "lat"), error="raise"
    )
    wide2 = pivot_q_data(
        long_df, value_prefix="subs",
        dt_col="year", spatial_cols=("lon", "lat"), error="raise"
    )
    cols = set(wide2.columns)
    # Expect all (year, quantile) combinations to be reconstructed
    expect = {
        "subs_2022_q0.1", "subs_2022_q0.5",
        "subs_2023_q0.1", "subs_2023_q0.5",
    }
    assert expect.issubset(cols)
    assert {"lon", "lat"}.issubset(cols)

def test_pivot_filter_q(wide_df_full):
    long_df = melt_q_data(
        wide_df_full, value_prefix="subs",
        dt_name="year", spatial_cols=("lon", "lat"), error="raise"
    )
    wide2 = pivot_q_data(
        long_df, value_prefix="subs",
        dt_col="year", spatial_cols=("lon", "lat"),
        q=[0.5], error="raise"
    )
    cols = set(wide2.columns)
    assert "subs_2022_q0.5" in cols and "subs_2023_q0.5" in cols
    assert "subs_2022_q0.1" not in cols and "subs_2023_q0.1" not in cols

def test_pivot_missing_dt_raises():
    df = pd.DataFrame({"subs_q0.1": [1.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        pivot_q_data(df, value_prefix="subs", dt_col="year")

# --- savefile decorator (smoke) ---------------------------------------------

def test_reshape_savefile_writes(tmp_path, wide_df_full):
    out_path = tmp_path / "reshaped.csv"
    out = reshape_quantile_data(
        wide_df_full, value_prefix="subs",
        spatial_cols=["lon", "lat"], dt_col="year",
        savefile=str(out_path), error="raise",
    )
    assert isinstance(out, pd.DataFrame)
    assert out_path.exists()
    # File shouldn't be empty
    assert out_path.stat().st_size > 0

if __name__=="__main__": 
    pytest.main( [__file__])