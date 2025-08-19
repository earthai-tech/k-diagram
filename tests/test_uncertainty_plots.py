import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from kdiagram.plot.uncertainty import (
    plot_actual_vs_predicted,
    plot_anomaly_magnitude,
    plot_coverage,
    plot_interval_consistency,
    plot_interval_width,
    plot_model_drift,
    plot_uncertainty_drift,
    plot_velocity,
)

def _cleanup():
    plt.close("all")

# ---------- plot_coverage ----------
def test_plot_coverage_line_bar_pie_radar(tmp_path):
    y_true = np.linspace(0, 1, 20)
    # one point forecast equals truth -> coverage 1
    pred_point = y_true.copy()
    # quantiles around truth -> also coverage 1 (unsorted q to hit sorting)
    lower = y_true - 0.1
    mid = y_true
    upper = y_true + 0.1
    pred_q = np.vstack([upper, lower, mid]).T  # intentionally shuffled columns
    q_levels = [0.9, 0.1, 0.5]

    # line
    out1 = tmp_path / "cov_line.png"
    plot_coverage(
        y_true,
        pred_point,
        pred_q,
        names=["P1"],
        q=q_levels,
        kind="line",
        title="line",
        savefig=str(out1),
    )
    assert out1.exists()

    # bar (names padded)
    out2 = tmp_path / "cov_bar.png"
    plot_coverage(
        y_true,
        pred_point,
        pred_q,
        names=["OnlyOne"],
        q=q_levels,
        kind="bar",
        title="bar",
        savefig=str(out2),
    )
    assert out2.exists()

    # pie with zero total coverage (use totally wrong preds)
    out3 = tmp_path / "cov_pie.png"
    preds_zero = np.ones_like(y_true) * 42
    plot_coverage(
        y_true, preds_zero, names=["Zero"], kind="pie", title="pie", savefig=str(out3)
    )
    assert out3.exists()

    # radar with cov_fill True single-model path (gradient branch)
    out4 = tmp_path / "cov_radar.png"
    plot_coverage(
        y_true,
        pred_point,
        names=["Solo"],
        kind="radar",
        cov_fill=True,
        title="radar",
        savefig=str(out4),
    )
    assert out4.exists()

    _cleanup()


def test_plot_coverage_bad_q_raises():
    y_true = np.arange(5)
    pred_q = np.column_stack([y_true - 1, y_true, y_true + 1])
    with pytest.raises(ValueError, match="Quantile levels must be between 0 and 1"):
        plot_coverage(y_true, pred_q, q=[-0.1, 0.5, 0.9])


# ---------- plot_model_drift ----------
def test_plot_model_drift_half_circle_and_colors(tmp_path):
    # 3 horizons, widths positive
    n = 30
    df = pd.DataFrame(
        {
            "q10_h1": np.random.rand(n) * 5,
            "q90_h1": np.random.rand(n) * 5 + 6,
            "q10_h2": np.random.rand(n) * 6 + 1,
            "q90_h2": np.random.rand(n) * 6 + 8,
            "q10_h3": np.random.rand(n) * 4 + 2,
            "q90_h3": np.random.rand(n) * 4 + 7,
            # add a generic metric to drive color metric path
            "rmse_h1": np.random.rand(n),
            "rmse_h2": np.random.rand(n),
            "rmse_h3": np.random.rand(n),
        }
    )
    out = tmp_path / "drift.png"
    ax = plot_model_drift(
        df,
        q10_cols=["q10_h1", "q10_h2", "q10_h3"],
        q90_cols=["q90_h1", "q90_h2", "q90_h3"],
        horizons=["H1", "H2", "H3"],
        color_metric_cols=["rmse_h1", "rmse_h2", "rmse_h3"],
        acov="half_circle",
        show_grid=False,
        savefig=str(out),
    )
    assert out.exists()
    # half circle -> 180 deg span
    assert pytest.approx(ax.get_thetamax(), rel=1e-3) == 180.0
    # grid should be off via public API
    gx = any(gl.get_visible() for gl in ax.get_xgridlines())
    gy = any(gl.get_visible() for gl in ax.get_ygridlines())
    assert not gx and not gy
    _cleanup()


# ---------- plot_velocity ----------
def test_plot_velocity_paths_and_warnings(tmp_path):
    # Build df with 3 time steps; include theta_col to trigger warning
    n = 40
    df = pd.DataFrame(
        {
            "t1": np.linspace(0, 1, n) + 2,
            "t2": np.linspace(0, 1, n) + 3,
            "t3": np.linspace(0, 1, n) + 4,
            "theta_like": np.linspace(10, 20, n),
        }
    )

    out = tmp_path / "vel.png"
    with pytest.warns(UserWarning):  # theta_col provided but ignored warning
        ax = plot_velocity(
            df=df,
            q50_cols=["t1", "t2", "t3"],
            theta_col="theta_like",
            acov="eighth_circle",  # 45deg span
            cmap="not_a_cmap",  # invalid colormap -> fallback warning
            use_abs_color=False,  # color by velocity path
            cbar=True,
            savefig=str(out),
        )
    assert out.exists()
    assert ax.get_thetamax() == pytest.approx(45.0, rel=1e-3)
    _cleanup()


def test_plot_velocity_constant_velocity_warns(tmp_path):
    n = 20
    # make all columns equal so diffs are zero -> velocity range zero warning
    base = np.ones(n) * 5
    df = pd.DataFrame({"a": base, "b": base, "c": base})
    out = tmp_path / "vel_const.png"
    with pytest.warns(UserWarning, match="Velocity range is zero"):
        plot_velocity(df=df, q50_cols=["a", "b", "c"], savefig=str(out))
    assert out.exists()
    _cleanup()


def test_plot_velocity_missing_cols_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(
        ValueError,
    ):
        plot_velocity(df=df, q50_cols=["a", "b"])  # b missing
    with pytest.raises(ValueError):
        # match=(
        # 'At least two Q50 columns (representing two time points)'
        # ' are required to compute velocity.')
        # ):
        plot_velocity(df=df, q50_cols=["a"])  # not enough columns


# ---------- plot_interval_consistency ----------


def test_plot_interval_consistency_cv_and_std(tmp_path):
    n = 35
    df = pd.DataFrame(
        {
            "q10_1": np.random.rand(n) * 2,
            "q90_1": np.random.rand(n) * 2 + 3,
            "q10_2": np.random.rand(n) * 2 + 0.5,
            "q90_2": np.random.rand(n) * 2 + 3.5,
            "q10_3": np.random.rand(n) * 2 + 1.0,
            "q90_3": np.random.rand(n) * 2 + 4.0,
            "q50_1": np.random.rand(n) * 4 + 5,
            "q50_2": np.random.rand(n) * 4 + 6,
            "q50_3": np.random.rand(n) * 4 + 7,
            "theta_like": np.linspace(0, 1, n),
        }
    )

    # CV path + warning for theta_col ignored + invalid cmap fallback
    out1 = tmp_path / "ic_cv.png"
    with pytest.warns(UserWarning):
        ax1 = plot_interval_consistency(
            df=df,
            qlow_cols=["q10_1", "q10_2", "q10_3"],
            qup_cols=["q90_1", "q90_2", "q90_3"],
            q50_cols=["q50_1", "q50_2", "q50_3"],
            theta_col="theta_like",
            cmap="def_nope",
            acov="quarter_circle",
            savefig=str(out1),
        )
    assert out1.exists()
    assert ax1.get_thetamax() == pytest.approx(90.0, rel=1e-3)

    # Std-dev path (use_cv=False)
    out2 = tmp_path / "ic_std.png"
    plot_interval_consistency(
        df=df,
        qlow_cols=["q10_1", "q10_2", "q10_3"],
        qup_cols=["q90_1", "q90_2", "q90_3"],
        use_cv=False,
        savefig=str(out2),
    )
    assert out2.exists()
    _cleanup()


def test_plot_interval_consistency_length_mismatch_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Mismatch in length"):
        plot_interval_consistency(df, ["a"], ["b", "b2"])

# ---------- plot_anomaly_magnitude ----------
def test_plot_anomaly_magnitude_under_over_and_cbar(tmp_path):
    n = 60
    df = pd.DataFrame(
        {
            "actual": np.linspace(0, 10, n),
            "q10": np.linspace(-1, 2, n),
            "q90": np.linspace(8, 12, n),
            "order": np.linspace(100, 200, n),
        }
    )

    # Force clear anomalies at both ends
    df.loc[:5, "actual"] = -5  # under
    # FIX: Use .loc for safe assignment to avoid pandas warnings
    df.loc[df.index[-6:], "actual"] = 20  # over

    out = tmp_path / "anom.png"

    # FIX: Wrap the call in pytest.warns to catch the expected warning
    with pytest.warns(UserWarning, match="Colormap 'nope_over' not found"):
        ax = plot_anomaly_magnitude(
            df=df,
            actual_col="actual",
            q_cols=["q10", "q90"],
            theta_col="order",
            acov="eighth_circle",
            cbar=True,
            cmap_over="nope_over",  # invalid -> fallback warning
            savefig=str(out),
        )
        
    assert out.exists()
    assert ax.get_thetamax() == pytest.approx(45.0, rel=1e-3)
    _cleanup()

def test_plot_anomaly_magnitude_no_anomalies_warning(tmp_path):
    n = 20
    df = pd.DataFrame(
        {
            "actual": np.ones(n) * 10,
            "q10": np.ones(n) * 9,
            "q90": np.ones(n) * 11,
        }
    )
    out = tmp_path / "anom_none.png"
    with pytest.warns(UserWarning, match="No anomalies detected"):
        ax = plot_anomaly_magnitude(
            df=df,
            actual_col="actual",
            q_cols=["q10", "q90"],
            savefig=str(out),
        )
    assert out.exists()
    assert ax is not None
    _cleanup()


def test_plot_anomaly_magnitude_bad_q_cols_raises():
    df = pd.DataFrame({"actual": [1, 2, 3], "a": [0, 0, 0]})
    with pytest.raises(ValueError, match="Validation of `q_cols` failed"):
        plot_anomaly_magnitude(df=df, actual_col="actual", q_cols=["a"])  # not 2 cols


# ---------- plot_uncertainty_drift ----------


def test_plot_uncertainty_drift_default_and_warnings(tmp_path):
    n = 50
    df = pd.DataFrame(
        {
            "q10_1": np.random.rand(n) * 2,
            "q90_1": np.random.rand(n) * 2 + 3,
            "q10_2": np.random.rand(n) * 2 + 0.5,
            "q90_2": np.random.rand(n) * 2 + 3.5,
        }
    )
    out = tmp_path / "drift_unc.png"
    with pytest.warns(UserWarning):  # theta_col ignored warning
        ax = plot_uncertainty_drift(
            df=df,
            qlow_cols=["q10_1", "q10_2"],
            qup_cols=["q90_1", "q90_2"],
            theta_col="maybe_missing",  # warned/ignored
            acov="garbage",  # invalid -> fallback warning
            savefig=str(out),
        )
    assert out.exists()
    # fallback to default -> full circle
    assert ax.get_thetamax() == pytest.approx(360.0, rel=1e-3)
    _cleanup()


def test_plot_uncertainty_drift_empty_after_dropna_returns_none(tmp_path):
    df = pd.DataFrame(
        {
            "q10_1": [np.nan, np.nan],
            "q90_1": [np.nan, np.nan],
        }
    )
    with pytest.warns(UserWarning, match="empty after dropping NaN"):
        res = plot_uncertainty_drift(df=df, qlow_cols=["q10_1"], qup_cols=["q90_1"])
    assert res is None
    _cleanup()


# ---------- plot_actual_vs_predicted ----------


def test_plot_actual_vs_predicted_line_and_dots(tmp_path):
    n = 80
    df = pd.DataFrame(
        {
            "act": 5 + np.sin(np.linspace(0, 2 * np.pi, n)),
            "pred": 5 + np.cos(np.linspace(0, 2 * np.pi, n)),
            "theta_like": np.linspace(0, 1, n),
        }
    )

    # line=True path with legend, mask_angle True, grid off
    out1 = tmp_path / "avp_line.png"
    with pytest.warns(UserWarning):  # theta_col ignored warning
        ax1 = plot_actual_vs_predicted(
            df=df,
            actual_col="act",
            pred_col="pred",
            theta_col="theta_like",
            acov="default",
            title="Line",
            line=True,
            r_label="Value",
            actual_props={"color": "black"},
            pred_props={"color": "red"},
            show_grid=False,
            mask_angle=True,
            savefig=str(out1),
        )
    assert out1.exists()
    # dots path
    out2 = tmp_path / "avp_dots.png"
    ax2 = plot_actual_vs_predicted(
        df=df,
        actual_col="act",
        pred_col="pred",
        line=False,
        alpha=0.5,
        savefig=str(out2),
    )
    assert out2.exists()
    # legends should be possible if labels present
    assert ax1 is not None and ax2 is not None
    _cleanup()


def test_plot_actual_vs_predicted_missing_cols_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises((KeyError, ValueError)):
        plot_actual_vs_predicted(df=df, actual_col="a", pred_col="not_here")


# ---------- plot_interval_width ----------


def test_plot_interval_width_with_z_and_masks(tmp_path):
    n = 45
    df = pd.DataFrame(
        {
            "q10": np.random.rand(n) * 2,
            "q90": np.random.rand(n) * 2 + 3,
            "z": np.random.rand(n) * 10,
            "theta_like": np.linspace(0, 1, n),
        }
    )
    out = tmp_path / "iw.png"
    # theta_col triggers warning (ignored), colorbar on
    with pytest.warns(UserWarning):
        ax = plot_interval_width(
            df=df,
            q_cols=["q10", "q90"],
            z_col="z",
            theta_col="theta_like",
            acov="eighth_circle",
            cbar=True,
            mask_angle=True,
            savefig=str(out),
        )
    assert out.exists()
    assert ax.get_thetamax() == pytest.approx(45.0, rel=1e-3)
    _cleanup()


def test_plot_interval_width_bad_qcols_and_missing_z_raises():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    with pytest.raises(TypeError, match="expects exactly two"):
        plot_interval_width(df=df, q_cols=["a"])  # not 2 cols
    with pytest.raises(ValueError, match="`z_col`"):
        plot_interval_width(df=df, q_cols=["a", "b"], z_col="nope")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
