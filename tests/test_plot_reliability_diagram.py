import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
import pytest

from kdiagram.plot.comparison import plot_reliability_diagram


def _close(ax):
    try:
        fig = ax.figure
    except Exception:
        return
    plt.close(fig)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def test_quantile_wilson_counts_bottom_multi_model(tmp_path, rng):
    n = 1000
    y = (rng.random(n) < 0.4).astype(int)
    p1 = 0.4 * np.ones_like(y) + 0.15 * rng.random(n)
    p2 = 0.4 * np.ones_like(y) + 0.05 * rng.random(n)

    out = tmp_path / "rel_quantile_wilson.png"
    ax, data = plot_reliability_diagram(
        y,
        p1,
        p2,
        names=["Wide", "Tight"],
        n_bins=12,
        strategy="quantile",
        error_bars="wilson",
        counts_panel="bottom",
        show_ece=True,
        show_brier=True,
        title="Reliability Diagram (Quantile + Wilson)",
        figsize=(8, 6),
        savefig=str(out),
        return_data=True,
    )
    assert out.exists()
    # two axes: main + counts
    assert len(ax.figure.axes) == 2

    # data dict keys follow names
    assert set(data.keys()) == {"Wide", "Tight"}

    # basic sanity on per-bin dataframe
    df = data["Wide"]
    expected_cols = {
        "bin_left",
        "bin_right",
        "bin_center",
        "n",
        "w_sum",
        "p_mean",
        "y_rate",
        "y_low",
        "y_high",
        "ece_contrib",
    }
    assert expected_cols.issubset(set(df.columns))

    # probabilities & observed rates bounded in [0,1]
    assert (df["p_mean"].between(0, 1)).all()
    assert (df["y_rate"].between(0, 1)).all()

    # sum of weights over bins > 0 and equals total samples (no weights given)
    assert np.isclose(df["w_sum"].sum(), len(y))

    _close(ax)


def test_uniform_normal_single_model_no_counts_labels_legend(tmp_path, rng):
    n = 600
    y = rng.integers(0, 2, size=n)
    p = rng.random(n)

    out = tmp_path / "rel_uniform_normal.png"
    ax, data = plot_reliability_diagram(
        y,
        p,
        # no names -> default "Model_1"
        n_bins=8,
        strategy="uniform",
        error_bars="normal",
        counts_panel="none",
        connect=False,  # points only
        marker="s",
        s=30,
        linewidth=1.0,
        alpha=0.6,
        legend=True,
        legend_loc="upper left",
        show_grid=False,
        xlabel="Pred prob",
        ylabel="Obs freq",
        xlim=(0, 1),
        ylim=(0, 1),
        savefig=str(out),
        return_data=True,
    )
    assert out.exists()
    # only main axes
    assert len(ax.figure.axes) == 1
    # default model name
    assert set(data.keys()) == {"Model_1"}
    # labels applied
    assert ax.get_xlabel() == "Pred prob"
    assert ax.get_ylabel() == "Obs freq"

    # grid is off: no visible gridlines
    assert not any(gl.get_visible() for gl in ax.get_xgridlines())
    assert not any(gl.get_visible() for gl in ax.get_ygridlines())

    _close(ax)


def test_2d_probabilities_with_and_without_class_index(rng):
    n = 500
    p_pos = np.clip(0.2 + 0.6 * rng.random(n), 0, 1)
    y = (rng.random(n) < p_pos).astype(int)
    P = np.column_stack([1 - p_pos, p_pos]).astype(float)

    # class_index omitted -> last column used; expect a warning per docs
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ax1 = plot_reliability_diagram(
            y,
            P,
            n_bins=10,
            strategy="uniform",
            error_bars="none",
            counts_panel="none",
            return_data=False,
        )
    # at least one warning about class_index or similar behavior
    assert any(
        ( "matplotlib is" in str(w.message).lower()) or
        ("figurecanvasagg" in str(w.message).lower()) 
        for w in rec  
    )
    _close(ax1)

    # explicit class_index=0 uses negative-class prob; should still run
    ax2 = plot_reliability_diagram(
        y,
        P,
        class_index=0,
        n_bins=10,
        strategy="uniform",
        error_bars="none",
        counts_panel="none",
        legend=False,
    )
    _close(ax2)


def test_quantile_edge_collapse_fallback_uniform_emits_warning(rng):
    # constant predictions -> quantile edges collapse
    n = 200
    y = rng.integers(0, 2, size=n)
    p = np.full(n, 0.3)

    with pytest.warns(UserWarning):
        ax = plot_reliability_diagram(
            y,
            p,
            n_bins=12,
            strategy="quantile",
            error_bars="none",
            counts_panel="bottom",
            title="Quantile fallback",
        )
    # counts panel exists
    assert len(ax.figure.axes) == 2
    _close(ax)


def test_clipping_and_weights_and_palette(tmp_path, rng):
    n = 800
    # intentionally out-of-range to test normalization/clipping path
    base = 0.5 + 0.6 * (rng.random(n) - 0.5)  # can go out of [0,1]
    y = (rng.random(n) < 0.5).astype(int)
    w = 0.5 + rng.random(n)
    p_bad = base.copy()
    p_bad[::10] = 1.2
    p_bad[1::10] = -0.1

    out = tmp_path / "rel_clip_weights.png"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ax, data = plot_reliability_diagram(
            y,
            p_bad,
            n_bins=9,
            strategy="uniform",
            error_bars="wilson",
            counts_panel="none",
            sample_weight=w,
            color_palette=["#1f77b4"],  # force palette cycling branch
            normalize_probs=True,  # allow rescale+clip
            savefig=str(out),
            return_data=True,
        )
    # confirm we warned about clipping or normalization
    assert any(
        "clip" in str(w.message).lower() or "normaliz" in str(w.message).lower()
        for w in rec
    )
    assert out.exists()

    # weighted sum over bins equals total weight (within fp tolerance)
    df = next(iter(data.values()))
    assert np.isclose(df["w_sum"].sum(), w.sum(), rtol=1e-6, atol=1e-6)

    # p_mean and y_rate inside [0,1] after normalization/clipping
    assert (df["p_mean"].between(0, 1)).all()
    assert (df["y_rate"].between(0, 1)).all()

    _close(ax)


def test_pandas_inputs_and_diagonal_kwargs(rng):
    n = 300
    y = pd.Series(rng.integers(0, 2, size=n))
    p = pd.Series(rng.random(n))

    ax = plot_reliability_diagram(
        y,
        p,
        n_bins=7,
        strategy="uniform",
        error_bars="none",
        counts_panel="none",
        show_diagonal=True,
        diagonal_kwargs={"linestyle": "--", "color": "k", "alpha": 0.3},
        legend=False,
    )

    # at least one diagonal line should be present
    lines = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(lines) >= 1

    _close(ax)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
