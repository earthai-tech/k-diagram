import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from kdiagram.plot.feature_based import plot_fingerprint

matplotlib.use("Agg")


def _df_precomputed(layers=3, feats=6, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.random((layers, feats))
    cols = [f"feat_{i+1}" for i in range(feats)]
    idx = [f"Layer {i+1}" for i in range(layers)]
    return pd.DataFrame(M, index=idx, columns=cols)


def _df_raw(n=300, groups=("2022", "2023", "2024"), seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(groups, size=n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    # make y correlated (for abs_corr method)
    y = 0.8 * x1 + 0.2 * x2 + rng.normal(0, 0.2, n)
    return pd.DataFrame({"grp": g, "y": y, "x1": x1, "x2": x2, "x3": x3})


def test_precomputed_df_half_circle_basic(tmp_path):
    df = _df_precomputed(layers=3, feats=6, seed=1)
    ax = plot_fingerprint(
        df,
        precomputed=True,
        acov="half_circle",
        title="t",
        savefig=str(tmp_path / "a.png"),
    )
    assert ax.name == "polar"
    # half circle ~ 180°
    assert math.isclose(ax.get_thetamax(), 180.0, rel_tol=1e-3)
    # legend labels match layer count
    _, labels = ax.get_legend_handles_labels()
    assert len(labels) == 3
    # file saved
    p = tmp_path / "a.png"
    assert p.exists() and p.stat().st_size > 0


def test_precomputed_ndarray_default_warns():
    M = np.array([[0.2, 0.4, 0.1, 0.9, 0.5], [0.1, 0.6, 0.2, 0.7, 0.3]])
    # shorter feature list on purpose; function should handle it
    feats = ["A", "B", "C"]

    ax = plot_fingerprint(
        M, precomputed=True, features=feats, acov="default", title="t"
    )
    assert ax.name == "polar"
    # full / default ~ 360°
    assert math.isclose(ax.get_thetamax(), 360.0, rel_tol=1e-3)


@pytest.mark.parametrize(
    "acov,deg",
    [
        ("quarter_circle", 90.0),
        ("quarter", 90.0),  # alias
        ("eighth_circle", 45.0),
        ("full", 360.0),  # alias
    ],
)
def test_compute_grouped_abs_corr_acov_variants(acov, deg):
    df = _df_raw()

    ax = plot_fingerprint(
        df,
        precomputed=False,
        y_col="y",
        group_col="grp",
        method="abs_corr",
        acov=acov,
        title="t",
    )

    assert ax.name == "polar"
    assert math.isclose(ax.get_thetamax(), deg, rel_tol=1e-3)
    # legend entries == number of groups
    _, labels = ax.get_legend_handles_labels()
    assert len(labels) == df["grp"].nunique()


def test_ax_injection_and_quarter_circle():
    fig, ax_in = plt.subplots(subplot_kw={"projection": "polar"})
    df = _df_precomputed(layers=2, feats=5)

    ax_out = plot_fingerprint(
        df, precomputed=True, acov="quarter_circle", ax=ax_in, title="t"
    )
    # same axes object reused
    assert ax_out is ax_in
    assert math.isclose(ax_out.get_thetamax(), 90.0, rel_tol=1e-3)


def test_normalize_ticks_present_when_normalize_true():
    df = _df_precomputed(layers=2, feats=6)
    ax = plot_fingerprint(df, precomputed=True, normalize=True, title="t")
    # expect radial tick "1.00" among yticks
    tick_texts = [t.get_text() for t in ax.get_yticklabels()]
    assert any(txt.strip() == "1.00" for txt in tick_texts)


def test_raw_std_method_single_layer(tmp_path):
    df = _df_raw(groups=("A",), n=120)
    ax = plot_fingerprint(
        df,
        precomputed=False,
        group_col=None,
        method="std",
        title="t",
        savefig=str(tmp_path / "b.png"),
    )
    assert ax.name == "polar"
    # only one legend label
    _, labels = ax.get_legend_handles_labels()
    assert len(labels) == 1
    assert (tmp_path / "b.png").exists()
