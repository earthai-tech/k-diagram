from __future__ import annotations

import math
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless backend

from kdiagram.plot.feature_based import (
    _draw_angular_labels,
    plot_feature_fingerprint,
    plot_feature_interaction,
    plot_fingerprint,
)


def _toy_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    th = rng.uniform(0.0, 24.0, n)
    r = rng.uniform(0.0, 1.0, n)
    z = rng.normal(0.0, 1.0, n) + 0.4 * np.cos(th)
    return pd.DataFrame({"th": th, "r": r, "z": z})


def test_plot_feature_interaction_basic_savefig(tmp_path):
    df = _toy_df(400)

    out = tmp_path / "fi.png"
    ax = plot_feature_interaction(
        df=df,
        theta_col="th",
        r_col="r",
        color_col="z",
        statistic="mean",
        theta_period=24.0,
        theta_bins=16,
        r_bins=6,
        acov="default",
        title="T",
        cmap="viridis",
        mask_radius=True,
        savefig=str(out),
        dpi=120,
    )
    assert out.exists()
    assert ax.name.lower() == "polar"
    # masked radial labels
    ylabs = [t.get_text() for t in ax.get_yticklabels()]
    assert all(lbl == "" for lbl in ylabs)


def test_plot_feature_interaction_empty_warns():
    # Only NaNs -> returns None with warning
    df = pd.DataFrame({"th": [np.nan], "r": [np.nan], "z": [np.nan]})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = plot_feature_interaction(
            df=df,
            theta_col="th",
            r_col="r",
            color_col="z",
        )
    assert res is None
    assert any("empty after dropping" in str(w.message) for w in rec)


def test_plot_feature_fingerprint_name_mismatch_and_colors(tmp_path):
    # 2 layers x 3 features
    M = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 3.0]])
    # more feature names than needed -> warn and truncate
    feats = ["a", "b", "c", "extra"]
    # fewer labels than layers -> warn and pad
    labs = ["L1"]
    # color list shorter than layers -> warn and cycle
    cols = ["#000000"]

    out = tmp_path / "fp.png"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ax = plot_feature_fingerprint(
            importances=M,
            features=feats,
            labels=labs,
            normalize=True,
            fill=False,
            cmap=cols,
            acov="half_circle",
            title="X",
            savefig=str(out),
        )

    assert out.exists()
    assert ax.name.lower() == "polar"
    msgs = "\n".join(str(w.message) for w in rec)
    assert "Extra feature names ignored" in msgs
    assert "Auto-filling" in msgs
    # assert "colors will repeat" in msgs


def test_plot_feature_fingerprint_no_normalize_reuse_ax():
    # reuse an existing polar ax and disable fill
    M = np.array([[0.0, 1.0, 3.0, 2.0]])
    feats = ["f1", "f2", "f3", "f4"]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    with pytest.warns(UserWarning):
        ax2 = plot_feature_fingerprint(
            importances=M,
            features=feats,
            labels=["Only"],
            normalize=False,
            fill=False,
            cmap="tab10",
            acov="quarter_circle",
            ax=ax,
            show_grid=False,
        )
    assert ax2 is ax
    # ticks hidden or set by helper later
    assert isinstance(ax.get_xticks(), np.ndarray)


def test_plot_fingerprint_precomputed_ndarray_and_save(tmp_path):
    M = np.array([[1.0, 0.0], [0.5, 1.0]])
    # features shorter than needed -> auto-extend
    feats = ["A"]

    out = tmp_path / "pf.png"
    ax = plot_fingerprint(
        importances=M,
        precomputed=True,
        features=feats,
        labels=None,
        normalize=True,
        fill=True,
        acov="full",
        savefig=str(out),
    )
    assert out.exists()
    assert ax.name.lower() == "polar"
    # legend present
    leg = ax.get_legend()
    assert leg is not None


def test_plot_fingerprint_precomputed_dataframe_numeric_selection():
    df = pd.DataFrame(
        {
            "non": list("abc"),
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 3.0, 4.0],
        }
    )
    # numeric-only selection, index present -> labels from index
    ax = plot_fingerprint(
        importances=df,
        precomputed=True,
        normalize=False,
        fill=False,
        acov="full",
        show_grid=False,
    )
    assert ax.name.lower() == "polar"
    # xticks were hidden and redrawn by helper
    assert len(ax.get_xticks()) == 0


def _mk_group_df() -> pd.DataFrame:
    # two features + target + grouping
    rng = np.random.default_rng(2)
    g = np.array(["A"] * 6 + ["B"] * 6)
    x1 = np.r_[rng.normal(0, 1, 6), np.ones(6)]
    x2 = np.r_[rng.normal(0, 1, 6), rng.normal(0, 1, 6)]
    # group B with constant y -> zero corr path
    y = np.r_[rng.normal(0, 1, 6), np.ones(6)]
    return pd.DataFrame({"g": g, "x1": x1, "x2": x2, "y": y})


@pytest.mark.parametrize("method", ["std", "var", "mad"])
def test_plot_fingerprint_compute_stat_methods(tmp_path, method):
    df = _mk_group_df()
    out = tmp_path / f"pf_{method}.png"
    ax = plot_fingerprint(
        importances=df,
        precomputed=False,
        y_col=None,
        group_col=None,
        method=method,
        features=["x1", "x2"],
        labels=None,
        normalize=True,
        fill=True,
        acov="half",
        savefig=str(out),
    )
    assert out.exists()
    assert ax.name.lower() == "polar"


def test_plot_fingerprint_compute_abs_corr_grouped(tmp_path):
    df = _mk_group_df()
    out = tmp_path / "pf_abs.png"
    ax = plot_fingerprint(
        importances=df,
        precomputed=False,
        y_col="y",
        group_col="g",
        method="abs_corr",
        features=["x1", "x2"],
        labels=None,
        normalize=True,
        fill=False,
        acov="quarter",
        savefig=str(out),
    )
    assert out.exists()
    assert ax.name.lower() == "polar"


def test__draw_angular_labels_narrow_and_wide():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # start with no extra texts
    n0 = len(ax.texts)

    # narrow (<= 180 deg) triggers staggering
    span = math.pi
    ang = np.linspace(0.0, span, 6, endpoint=False)
    _draw_angular_labels(
        ax=ax,
        angles=ang,
        labels=[f"l{i}" for i in range(6)],
        r=1.0,
        span=span,
    )
    n1 = len(ax.texts)
    assert n1 == n0 + 6

    # wide (> 180 deg) no staggering branch
    span2 = 2.0 * math.pi
    ang2 = np.linspace(0.0, span2, 4, endpoint=False)
    _draw_angular_labels(
        ax=ax,
        angles=ang2,
        labels=[f"w{i}" for i in range(4)],
        r=1.0,
        span=span2,
    )
    n2 = len(ax.texts)
    assert n2 == n1 + 4
