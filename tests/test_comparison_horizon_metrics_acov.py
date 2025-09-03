from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Headless backend for CI
matplotlib.use("Agg")

from kdiagram.plot.comparison import plot_horizon_metrics

ACOV_DEG = {
    "default": 360.0,
    "half_circle": 180.0,
    "quarter_circle": 90.0,
    "eighth_circle": 45.0,
}


def _is_polar(ax) -> bool:
    return getattr(ax, "name", None) == "polar"


def _get_thetamax(ax) -> float:
    if hasattr(ax, "get_thetamax"):
        return float(ax.get_thetamax())
    return float(getattr(ax, "thetamax", 360.0))


def _close(ax):
    plt.close(ax.figure)


def _synth_df(n_rows: int = 12, n_cols: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # build matched qlow/qup pairs across multiple columns
    qlow_cols = [f"q10_c{i+1}" for i in range(n_cols)]
    qup_cols = [f"q90_c{i+1}" for i in range(n_cols)]
    q50_cols = [f"q50_c{i+1}" for i in range(n_cols)]

    base = rng.normal(10.0, 2.0, size=(n_rows, n_cols))
    width = np.abs(rng.normal(2.0, 0.7, size=(n_rows, n_cols)))
    q10 = base - 0.5 * width
    q90 = base + 0.5 * width
    q50 = base + rng.normal(0.0, 0.2, size=(n_rows, n_cols))

    df = pd.DataFrame(
        {
            **{c: q10[:, i] for i, c in enumerate(qlow_cols)},
            **{c: q90[:, i] for i, c in enumerate(qup_cols)},
            **{c: q50[:, i] for i, c in enumerate(q50_cols)},
        }
    )
    return df, qlow_cols, qup_cols, q50_cols


def test_horizon_metrics_respects_acov_minimal():
    df, ql, qu, _ = _synth_df(n_rows=10, n_cols=3)

    for acov, deg in ACOV_DEG.items():
        ax = plot_horizon_metrics(
            df=df,
            qlow_cols=ql,
            qup_cols=qu,
            acov=acov,
            cbar=False,  # avoid extra axes in test
            show_grid=False,
            show_value_labels=False,
            figsize=(6, 6),
        )
        assert ax is not None
        assert _is_polar(ax)
        # thetamax matches requested coverage
        got = _get_thetamax(ax)
        assert np.isclose(got, deg, atol=0.5)
        # bars rendered (at least as many patches as rows)
        assert len(ax.patches) >= len(df)
        _close(ax)


def test_horizon_metrics_with_q50_and_xticks_and_cbar():
    df, ql, qu, q50 = _synth_df(n_rows=8, n_cols=4)
    labels = [f"H{i+1}" for i in range(len(df))]

    ax = plot_horizon_metrics(
        df=df,
        qlow_cols=ql,
        qup_cols=qu,
        q50_cols=q50,
        acov="quarter_circle",
        xtick_labels=labels,
        cbar=True,
        show_grid=True,
        figsize=(7, 7),
    )
    assert ax is not None
    assert _is_polar(ax)
    got = _get_thetamax(ax)
    assert np.isclose(got, ACOV_DEG["quarter_circle"], atol=0.5)
    # xticks count matches number of bars
    assert len(ax.get_xticks()) == len(df)
    # there should be at least one colorbar mappable attached
    # (colorbar adds an axes to the figure)
    assert len(ax.figure.axes) >= 2
    _close(ax)
