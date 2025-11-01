# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kdiagram.plot.feature_based import plot_feature_interaction


def _grid_df(theta_bins=6, r_bins=4,
             tmin=0.0, tmax=24.0, rmin=0.0, rmax=1.0):
    """Make one sample exactly in the interior of each (theta, r) bin."""
    rows = []
    dt = (tmax - tmin) / theta_bins
    dr = (rmax - rmin) / r_bins
    for i in range(theta_bins):
        t = tmin + (i + 0.5) * dt
        for j in range(r_bins):
            r = rmin + (j + 0.5) * dr
            rows.append((t, r, i + j))   # any deterministic value
    return pd.DataFrame(rows, columns=["theta", "radius", "z"])


def test_annular_draws_one_wedge_per_bin():
    """Annular rendering should create r_bins * theta_bins bar patches
    on the target polar Axes (one per populated cell)."""
    theta_bins, r_bins = 7, 5
    df = _grid_df(theta_bins=theta_bins, r_bins=r_bins)

    ax = plot_feature_interaction(
        df=df,
        theta_col="theta",
        r_col="radius",
        color_col="z",
        statistic="mean",
        theta_period=24,     # period mapping (independent of min/max)
        theta_bins=theta_bins,
        r_bins=r_bins,
        mode="annular",
        show_grid=False,
    )

    # Count only the rectangle patches on THIS axes (each bar is one Rectangle).
    rects = [p for p in ax.patches if isinstance(p, mpl.patches.Rectangle)]
    assert len(rects) == theta_bins * r_bins

    plt.close(ax.figure)


def test_annular_respects_generic_tick_specifications():
    """Custom ticks/labels (callables, mappings, sequences) should be
    applied to both theta (angular) and r (radial) axes."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "hour": rng.uniform(0, 24, 2000),
        "sent": rng.uniform(-1, 1, 2000),
        "val":  rng.normal(size=2000)
    })

    theta_ticks = [0.0, 12.0, 24.0]
    theta_labels_map = {0.0: "Start", 12.0: "Mid", 24.0: "End"}

    r_ticks = [-1.0, 0.0, 1.0]
    r_labels_fn = lambda x: { -1.0: "Bearish", 0.0: "Neutral", 1.0: "Bullish" }[x]

    ax = plot_feature_interaction(
        df=df,
        theta_col="hour",
        r_col="sent",
        color_col="val",
        statistic="mean",
        theta_period=24,
        theta_bins=12,
        r_bins=8,
        mode="annular",
        theta_ticks=theta_ticks,
        theta_ticklabels=theta_labels_map,   # Mapping
        r_ticks=r_ticks,
        r_ticklabels=r_labels_fn,            # Callable
        show_grid=False,
    )

    xtlbls = [t.get_text() for t in ax.get_xticklabels()]
    ytlbls = [t.get_text() for t in ax.get_yticklabels()]

    assert xtlbls == [theta_labels_map[v] for v in theta_ticks]
    assert ytlbls == [r_labels_fn(v) for v in r_ticks]

    plt.close(ax.figure)
