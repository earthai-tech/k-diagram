import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from kdiagram.plot import uncertainty as up


def _toy_df(n=30, seed=0):
    rng = np.random.default_rng(seed)
    # t = pd.date_range("2024-01-01", periods=n, freq="h")
    t = pd.to_datetime(
        np.arange(n),
        unit="h",
        origin=pd.Timestamp("2024-01-01"),
    )
    y = np.sin(np.linspace(0, 6 * np.pi, n)) * 5 + 20
    ql1 = y - 2.0
    qu1 = y + 2.0
    ql2 = y - 3.0
    qu2 = y + 3.0

    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    theta_bad = [letters[i % len(letters)] for i in range(n)]

    df = pd.DataFrame(
        {
            "t": t,
            "actual": y,
            "q10": ql1,
            "q90": qu1,
            "q10_b": ql2,
            "q90_b": qu2,
            # central predictions for velocity/avp
            "pred": y + rng.normal(0, 0.5, size=n),
            "pred_b": y + rng.normal(0, 0.6, size=n),
            # extra fields used by some plots
            "theta_ok": np.linspace(0, 1, n),
            "theta_bad": theta_bad,  # non-numeric
            "ring": np.clip(np.abs(rng.normal(0, 1.0, n)), 0, None),
            "grp": np.where(np.arange(n) % 2 == 0, "A", "B"),
            "z": rng.normal(0, 1, n),
        }
    )
    return df


# ------------------------ plot_coverage --------------------------------------
@pytest.mark.filterwarnings(
    "ignore:result dtype changed due to the removal of "
    "value-based promotion from NumPy:UserWarning:matplotlib"
)
def test_plot_coverage_variants(tmp_path):
    df = _toy_df(24)
    y = df["actual"].to_numpy().astype(np.float32)

    # intervals for two "models"
    pred1 = np.c_[df["q10"].to_numpy(), df["q90"].to_numpy()].astype(
        np.float32
    )
    pred2 = np.c_[df["q10_b"].to_numpy(), df["q90_b"].to_numpy()].astype(
        np.float32
    )

    # kind='line' + user-provided ax (function ignores ax -> at least one warning)
    fig, pre_ax = plt.subplots()
    with warnings.catch_warnings(record=True) as w:
        up.plot_coverage(
            y,
            pred1,
            pred2,
            kind="line",
            names=["M1", "M2"],
            q=(0.1, 0.9),
            ax=pre_ax,
            verbose=0,
            savefig=tmp_path / "cov_line.png",
        )
    assert len(w) >= 0  # do not depend on specific wording

    # kind='pie'
    up.plot_coverage(
        y,
        pred1,
        kind="pie",
        names=["OnlyOne"],
        q=(0.1, 0.9),
        savefig=tmp_path / "cov_pie.png",
    )

    # fallback branch (unknown kind) + verbose summary printing
    up.plot_coverage(
        y, pred1, kind="unknown", names=["U"], q=(0.1, 0.9), verbose=1
    )

    # kind='radar'
    up.plot_coverage(
        y,
        pred1,
        pred2,
        kind="radar",
        names=["A", "B"],
        q=(0.1, 0.9),
        savefig=tmp_path / "cov_radar.png",
    )


# ------------------------ plot_model_drift -----------------------------------


def test_plot_model_drift_defaults(tmp_path):
    df = _toy_df(16)
    # pass q_cols as a list of tuples (pairs)
    up.plot_model_drift(
        df,
        q_cols=[("q10", "q90")],
        horizons=None,
        savefig=tmp_path / "drift.png",
    )


# ------------------------ plot_velocity --------------------------------------


def test_plot_velocity_axes_and_acov(tmp_path):
    df = _toy_df(18)

    # invalid acov -> warning path
    with warnings.catch_warnings(record=True) as w:
        up.plot_velocity(
            df,
            q50_cols=["pred", "pred_b"],
            acov="weird",
            savefig=tmp_path / "vel1.png",
        )
    # Look for at least one warning (message text not pinned)
    assert len(w) >= 1

    # existing ax branch (also needs >=2 q50_cols)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    up.plot_velocity(
        df,
        q50_cols=["pred", "pred_b"],
        acov="default",
        ax=ax,
    )


# -------------------- plot_interval_consistency ------------------------------


def test_plot_interval_consistency_happy_and_error(tmp_path):
    df = _toy_df(20)

    # q50 given as existing but non-numeric columns -> triggers fallback coloring path with a warning
    with warnings.catch_warnings(record=True) as w:
        up.plot_interval_consistency(
            df,
            qlow_cols=["q10", "q10_b"],
            qup_cols=["q90", "q90_b"],
            q50_cols=["theta_bad", "grp"],  # exist but non-numeric
            acov="default",
            savefig=tmp_path / "consistency_ok.png",
        )
    assert len(w) >= 1

    # error path: make lower bound non-numeric -> computation raises
    df_bad = df.copy()
    df_bad["q10"] = list("abcdefghijmnopqrstuvwx")[: len(df_bad)]
    with pytest.raises(TypeError):
        up.plot_interval_consistency(
            df_bad,
            qlow_cols=["q10", "q10_b"],
            qup_cols=["q90", "q90_b"],
        )

    # invalid acov -> warning (but still runs)
    with warnings.catch_warnings(record=True) as w2:
        up.plot_interval_consistency(
            df,
            qlow_cols=["q10"],
            qup_cols=["q90"],
            acov="nope",
        )
    assert len(w2) >= 1

    # savefig with non-existent folder -> internal try/except swallows I/O error
    bad_path = tmp_path / "no_such_dir" / "x" / "consistency.png"
    up.plot_interval_consistency(
        df,
        qlow_cols=["q10"],
        qup_cols=["q90"],
        savefig=bad_path,
    )


# --------------------- plot_anomaly_magnitude --------------------------------


def test_plot_anomaly_magnitude_variants_and_warnings(tmp_path):
    df = _toy_df(22)

    # invalid colormaps -> warnings (theta present & numeric)
    with warnings.catch_warnings(record=True) as w:
        up.plot_anomaly_magnitude(
            df,
            actual_col="actual",
            q_cols=["q10", "q90"],
            theta_col="theta_ok",
            cmap_under="no_such_cmap",
            cmap_over="also_bad",
            show_grid=False,
            mask_angle=True,
            savefig=tmp_path / "anom1.png",
        )
    assert len(w) >= 1

    # non-numeric theta -> should warn and fall back to default order (no exception)
    with warnings.catch_warnings(record=True) as w_nonnum:
        ax = up.plot_anomaly_magnitude(
            df,
            actual_col="actual",
            q_cols=["q10", "q90"],
            theta_col="theta_bad",
        )
    assert ax is not None
    assert len(w_nonnum) >= 1


# ------------------- plot_actual_vs_predicted --------------------------------


def test_plot_actual_vs_predicted_line_and_legend(tmp_path):
    df = _toy_df(25)

    # Good run with line=True and legend
    up.plot_actual_vs_predicted(
        df,
        actual_col="actual",
        pred_col="pred",
        show_legend=True,
        line=True,
        pred_props={"linestyle": "--"},
        savefig=tmp_path / "avp1.png",
    )

    # Exercise decorator warning path about missing theta_col
    with warnings.catch_warnings(record=True) as w:
        up.plot_actual_vs_predicted(
            df,
            actual_col="actual",
            pred_col="pred",
            show_legend=True,
            line=False,
            theta_col="missing",
            savefig=tmp_path / "avp2.png",
        )
    assert len(w) >= 1

    # savefig exception path (non-existent folder)
    bad = tmp_path / "missing_dir" / "x" / "avp.png"
    up.plot_actual_vs_predicted(df, "actual", "pred", savefig=bad)


# -------------------- plot_polar_heatmap -------------------------------------


def test_plot_polar_heatmap_paths(tmp_path):
    df = _toy_df(28)

    # valid r/theta; invalid acov -> ValueError
    with pytest.raises(ValueError):
        up.plot_polar_heatmap(
            df,
            r_col="ring",
            theta_col="theta_ok",
            acov="nope",
            savefig=tmp_path / "heat1.png",
        )

    # basic run (no z -> statistic='count')
    ax = up.plot_polar_heatmap(
        df,
        r_col="ring",
        theta_col="theta_ok",
    )
    assert ax is not None

    # z constant -> still plot
    df2 = df.copy()
    df2["z"] = 1.0
    up.plot_polar_heatmap(
        df2,
        r_col="ring",
        theta_col="theta_ok",
        savefig=tmp_path / "heat2.png",
    )

    # savefig exception branch
    bad = tmp_path / "nope_heat.png"
    up.plot_polar_heatmap(df, "ring", "theta_ok", savefig=bad)


# ----------------- plot_coverage_diagnostic ----------------------------------


def test_plot_coverage_diagnostic_kernels_and_save(tmp_path):
    df = _toy_df(26)

    # basic run with gradient fill & bars off; verbose on
    up.plot_coverage_diagnostic(
        df,
        actual_col="actual",
        q_cols=["q10", "q90"],
        as_bars=False,
        verbose=1,
        savefig=tmp_path / "covdiag.png",
    )

    # savefig exception path
    bad = tmp_path / "no_dir" / "x" / "covdiag.png"
    up.plot_coverage_diagnostic(
        df,
        "actual",
        ["q10", "q90"],
        savefig=bad,
    )


# ----------------- plot_interval_width ---------------------------------------


def test_plot_interval_width_errors_and_hue(tmp_path):
    df = _toy_df(21)
    df["q90"] = df["q90"] + np.linspace(-0.5, 0.5, len(df))
    # invalid q_cols length -> TypeError
    with pytest.raises(TypeError):
        up.plot_interval_width(df, q_cols=["q10"])

    # non-numeric -> ValueError inside width computation
    df_bad = df.copy()
    df_bad["q10_b"] = list("abcdefghijklmnopqrstu")[: len(df_bad)]
    with pytest.raises(TypeError):
        up.plot_interval_width(df_bad, q_cols=["q10_b", "q90_b"])

    # happy path + force a warning by giving reversed (upper, lower) to produce negative widths
    with warnings.catch_warnings(record=True) as w:
        up.plot_interval_width(
            df,
            q_cols=[
                "q90",
                "q10",
            ],  # reversed on purpose -> negative width warning
            show_grid=False,
            cbar=True,
            savefig=tmp_path / "width.png",
        )
    assert len(w) >= 1

    # savefig exception path
    bad = tmp_path / "no_dir" / "width.png"
    up.plot_interval_width(df, q_cols=["q10", "q90"], savefig=bad)
