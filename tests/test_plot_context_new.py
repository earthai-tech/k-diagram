import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import pytest

from kdiagram.plot import context as ctx

# ------------------------------- test data -----------------------------------


def make_df(n=60, seed=0, with_nans=False):
    rng = np.random.default_rng(seed)
    t = pd.to_datetime(
        np.arange(n),
        unit="h",
        origin=pd.Timestamp("2024-01-01"),
    )

    y = 50 + np.linspace(0, 10, n) + 5 * np.sin(np.arange(n) * 2 * np.pi / 15)
    pred1 = y + rng.normal(0, 1.5, n)
    pred2 = 0.9 * y + 5 + rng.normal(0, 2.0, n)
    df = pd.DataFrame(
        {
            "time": t,
            "actual": y,
            "pred1": pred1,
            "pred2": pred2,
            "q10": y - 3,
            "q90": y + 3,
        }
    )
    if with_nans:
        df.loc[[3, 7, 11], "pred1"] = np.nan
    return df


# ------------------------------- plot_time_series ----------------------------


def test_time_series_errors_and_branches(tmp_path):
    df = make_df(40)

    # error when neither actual_col nor pred_cols provided
    with pytest.raises(ValueError):
        ctx.plot_time_series(df)

    # missing feature raises via exist_features
    with pytest.raises(ValueError):
        ctx.plot_time_series(df, x_col="time", pred_cols=["missing_model"])

    # names length mismatch -> warning; includes q-band; uses x_col
    with pytest.warns(UserWarning, match="Length of `names`"):
        ax = ctx.plot_time_series(
            df,
            x_col="time",
            actual_col="actual",
            pred_cols=["pred1", "pred2"],
            names=["only_one_name"],  # mismatch -> warn
            q_lower_col="q10",
            q_upper_col="q90",
            title="TS with Bands",
            cmap="viridis",
            savefig=tmp_path / "ts_with_bands.png",
        )
    assert ax is not None

    # use index for x (no x_col); only actual line; custom grid props
    ax2 = ctx.plot_time_series(
        df,
        actual_col="actual",
        figsize=(10, 4),
        grid_props={"linestyle": ":"},
        savefig=tmp_path / "ts_index.png",
    )
    assert ax2 is not None


# ------------------------------- plot_scatter_correlation --------------------


def test_scatter_correlation_paths(tmp_path):
    df = make_df(45, with_nans=True)

    # must provide at least one pred col
    with pytest.raises(ValueError):
        ctx.plot_scatter_correlation(df, actual_col="actual", pred_cols=[])

    # normal path + names mismatch warning + no identity line
    with pytest.warns(UserWarning, match="Length of `names`"):
        ax = ctx.plot_scatter_correlation(
            df,
            actual_col="actual",
            pred_cols=["pred1", "pred2"],
            names=["p1"],  # mismatch
            title="AVP Scatter",
            show_identity_line=False,
            savefig=tmp_path / "scatter.png",
        )
    assert ax is not None


# ------------------------------- plot_error_autocorrelation ------------------


def test_error_autocorrelation_short_and_normal(tmp_path):
    # too few points -> warn and return None
    df_short = make_df(1)
    with pytest.warns(UserWarning, match="Not enough data points"):
        out = ctx.plot_error_autocorrelation(
            df_short, actual_col="actual", pred_col="pred1"
        )
    assert out is None

    # normal case, pass an extra kw that should be filtered out safely
    df = make_df(80)
    ax = ctx.plot_error_autocorrelation(
        df,
        actual_col="actual",
        pred_col="pred1",
        bogus_kw_that_should_be_ignored=True,
        savefig=tmp_path / "acf.png",
    )
    assert ax is not None


# ----------------------------------- plot_qq ---------------------------------


def test_plot_qq_short_and_normal(tmp_path):
    # too few -> warn and return None
    df_short = make_df(1)
    with pytest.warns(UserWarning, match="Not enough data points"):
        out = ctx.plot_qq(df_short, actual_col="actual", pred_col="pred1")
    assert out is None

    # normal path
    df = make_df(70)
    ax = ctx.plot_qq(
        df, actual_col="actual", pred_col="pred1", savefig=tmp_path / "qq.png"
    )
    assert ax is not None


# ------------------------------- plot_error_distribution ---------------------


def test_error_distribution_short_and_normal():
    # too few -> warn and return None
    df_short = make_df(1)
    with pytest.warns(UserWarning, match="Not enough data points"):
        out = ctx.plot_error_distribution(
            df_short, actual_col="actual", pred_col="pred1"
        )
    assert out is None

    # normal path (delegates to plot_hist_kde)
    df = make_df(120)
    ax = ctx.plot_error_distribution(
        df, actual_col="actual", pred_col="pred1", bins=30
    )
    assert ax is not None


# -------------------------------- plot_error_pacf ----------------------------


def test_error_pacf_variants(tmp_path):
    df_short = make_df(1)

    # If statsmodels isn't installed, ensure decorator raises ImportError
    try:
        import statsmodels  # noqa: F401

        has_sm = True
    except Exception:
        has_sm = False

    if not has_sm:
        with pytest.raises(ImportError):
            ctx.plot_error_pacf(
                df_short, actual_col="actual", pred_col="pred1"
            )
        pytest.skip("statsmodels not installed; PACF runtime paths skipped.")

    # with statsmodels: too few -> warn and return None
    with pytest.warns(UserWarning, match="Not enough data points"):
        out = ctx.plot_error_pacf(
            df_short, actual_col="actual", pred_col="pred1"
        )
    assert out is None

    # normal path + force a too-large lags to trigger safe fallback
    df = make_df(60)
    ax = ctx.plot_error_pacf(
        df,
        actual_col="actual",
        pred_col="pred1",
        lags=10_000,  # intentionally huge -> fallback inside
        savefig=tmp_path / "pacf.png",
    )
    assert ax is not None
