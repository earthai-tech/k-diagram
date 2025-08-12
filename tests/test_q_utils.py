
import re
import pandas as pd
import numpy as np
import pytest

from kdiagram.utils import q_utils


# ---------- Fixtures ----------
@pytest.fixture
def wide_df():
    # two points, several quantiles/years
    return pd.DataFrame(
        {
            "lon": [-118.25, -118.30],
            "lat": [34.05, 34.10],
            "subs_2022_q0.1": [1.2, 1.3],
            "subs_2022_q0.5": [1.5, 1.6],
            "subs_2023_q0.1": [1.7, 1.8],
            "subs_2023_q0.9": [np.nan, 1.9],
        }
    )


@pytest.fixture
def long_df():
    # long layout with two quantile columns
    return pd.DataFrame(
        {
            "lon": [-118.25, -118.25, -118.30],
            "lat": [34.05, 34.05, 34.10],
            "year": [2022, 2023, 2022],
            "subs_q0.1": [1.2, 1.7, 1.3],
            "subs_q0.5": [1.5, 2.0, 1.6],
        }
    )


# ---------- reshape_quantile_data ----------
def test_reshape_quantile_data_basic(wide_df):
    out = q_utils.reshape_quantile_data(
        wide_df, value_prefix="subs", 
        spatial_cols=["lon", "lat"], dt_col="year"
    )
    # expected columns exist
    assert set(["lon", "lat", "year"]).issubset(out.columns)
    assert "subs_q0.1" in out.columns
    assert "subs_q0.5" in out.columns
    # years sorted ascending
    assert list(out["year"].unique()) == sorted(out["year"].unique())


def test_reshape_quantile_data_missing_spatial_warns_and_continues(wide_df):
    with pytest.warns(UserWarning, match=r"Missing spatial columns"):
        out = q_utils.reshape_quantile_data(
            wide_df,
            value_prefix="subs",
            spatial_cols=["lon", "lat", "z_missing"],
            error="warn",
        )
    # function proceeds with existing spatial columns
    assert set(["lon", "lat"]).issubset(out.columns)


def test_reshape_quantile_data_no_matching_prefix(wide_df):
    with pytest.warns(
            UserWarning, match=r"No columns found with prefix 'other'"):
        out = q_utils.reshape_quantile_data(
            wide_df, value_prefix="other", 
            spatial_cols=["lon", "lat"], error="warn"
        )
    assert isinstance(out, pd.DataFrame) and out.empty


def test_reshape_quantile_data_invalid_pattern_returns_empty():
    # quant columns that do NOT match year regex (need 4 digits)
    df = pd.DataFrame(
        {
            "lon": [0],
            "lat": [0],
            "subs_22_q0.5": [1.0],  # wrong year format
        }
    )
    out = q_utils.reshape_quantile_data(
        df, value_prefix="subs", spatial_cols=["lon", "lat"])
    assert out.empty


# ---------- melt_q_data ----------
def test_melt_q_data_basic_with_spatial_and_year(wide_df):
    out = q_utils.melt_q_data(
        wide_df,
        value_prefix="subs",
        dt_name="year",
        spatial_cols=("lon", "lat"),
        error="raise",
    )
    # columns expected
    assert set(["lon", "lat", "year"]).issubset(out.columns)
    # renamed quantile columns
    assert any(c.startswith("subs_q") for c in out.columns)
    # check a couple values survived (presence, not exact layout)
    assert "subs_q0.1" in out.columns
    assert "subs_q0.5" in out.columns
    # sorting stable
    assert list(out["year"].unique()) == sorted(out["year"].unique())


def test_melt_q_data_filter_quantiles(wide_df):
    out = q_utils.melt_q_data(
        wide_df,
        value_prefix="subs",
        dt_name="year",
        q=[0.1, 0.9],
        spatial_cols=("lon", "lat"),
    )
    # Only 0.1 and 0.9 present
    assert "subs_q0.1" in out.columns
    assert "subs_q0.9" in out.columns
    assert "subs_q0.5" not in out.columns


def test_melt_q_data_no_match_prefix_warns_and_empty(wide_df):
    with pytest.warns(
            UserWarning, match=r"No columns found with prefix 'other'"):
        out = q_utils.melt_q_data(
            wide_df, value_prefix="other", dt_name="year", error="warn"
        )
    assert out.empty


def test_melt_q_data_invalid_spatial_cols_raises(wide_df):
    # spatial col 'y' does not exist
    with pytest.raises(ValueError, match=re.escape ( 
            "The following spatial_cols are"
            " not present in the dataframe: {'y', 'x'}"
            )):
        q_utils.melt_q_data(
            wide_df,
            value_prefix="subs",
            dt_name="year",
            spatial_cols=("x", "y"),
        )


def test_melt_q_data_sort_values_ok_and_bad(wide_df):
    # ok: sort by existing column
    out1 = q_utils.melt_q_data(
        wide_df,
        value_prefix="subs",
        dt_name="year",
        spatial_cols=("lon", "lat"),
        sort_values="year",
    )
    assert "year" in out1.columns

    # bad: non-existing sort column "
    # triggers internal fallback (no exception)
    out2 = q_utils.melt_q_data(
        wide_df,
        value_prefix="subs",
        dt_name="year",
        spatial_cols=("lon", "lat"),
        sort_values="not_there",
        verbose=2,  # prints warn message; no exception raised
    )
    assert isinstance(out2, pd.DataFrame)


# ---------- pivot_q_data ----------
def test_pivot_q_data_basic(long_df):
    out = q_utils.pivot_q_data(
        long_df,
        value_prefix="subs",
        dt_col="year",
        spatial_cols=("lon", "lat"),
        error="raise",
    )
    # Must have lon/lat and reconstructed quantile/year columns
    assert set(["lon", "lat"]).issubset(out.columns)
    assert any(
        c.startswith(
            "subs_2022_q") or c.startswith("subs_2023_q"
                                           ) for c in out.columns
    )
    # check a known value location (rough presence check)
    # there should be at least these two reconstructed columns
    cols = set(out.columns)
    assert {"subs_2022_q0.1", "subs_2022_q0.5"}.issubset(cols)


def test_pivot_q_data_missing_dt_col_warns_and_empty(long_df):
    df = long_df.drop(columns=["year"])
    with pytest.warns(UserWarning, match=r"Missing required columns"):
        out = q_utils.pivot_q_data(
            df,
            value_prefix="subs",
            dt_col="year",
            spatial_cols=("lon", "lat"),
            error="warn",
        )
    assert out.empty


def test_pivot_q_data_no_quantile_columns_warns_and_empty():
    df = pd.DataFrame({"lon": [0, 0], "lat": [0, 0], 
                       "year": [2022, 2022], "x": [1, 2]})
    with pytest.warns(
            UserWarning, match=r"No quantile columns found"):
        out = q_utils.pivot_q_data(
            df,
            value_prefix="subs",
            dt_col="year",
            spatial_cols=("lon", "lat"),
            error="warn",
        )
    assert out.empty


def test_pivot_q_data_filter_q_subset(long_df):
    out = q_utils.pivot_q_data(
        long_df,
        value_prefix="subs",
        dt_col="year",
        q=[0.1],  # keep only 0.1 column
        spatial_cols=("lon", "lat"),
        error="raise",
    )
    # only q0.1 columns remain
    assert any(c.endswith("_q0.1") for c in out.columns)
    assert not any(c.endswith("_q0.5") for c in out.columns)


def test_pivot_q_data_invalid_spatial_cols_raises(long_df):
    with pytest.raises(ValueError, match=re.escape ( 
            "The following spatial_cols are"
            " not present in the dataframe: {'y', 'x'}"
            )):
        q_utils.pivot_q_data(
            long_df,
            value_prefix="subs",
            dt_col="year",
            spatial_cols=("x", "y"),
        )


def test_pivot_q_data_input_type_check():
    with pytest.raises(
            TypeError, match="Input must be a pandas DataFrame"):
        q_utils.pivot_q_data(
            ["not", "a", "df"],
            value_prefix="subs",
            dt_col="year",
        )


if __name__ == "__main__": # pragma : no-cover
    pytest.main([__file__])