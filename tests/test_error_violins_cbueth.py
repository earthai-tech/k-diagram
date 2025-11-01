# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

import matplotlib
matplotlib.use("Agg")  
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kdiagram.plot.errors import plot_error_violins


def _df_errors(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "A": rng.normal(loc=0.5, scale=1.5, size=n),
        "B": rng.normal(loc=-4.0, scale=1.5, size=n),
        "C": rng.normal(loc=0.0, scale=4.0, size=n),
    })


def test_cbueth_overlay_two_models_has_two_polygons_and_stats_in_legend():
    """With k<=2 and overlay='auto', cbueth should overlay both models
    on one spoke, draw two filled polygons, hide xticks, and put stats
    (median, skew) in the legend along with the zero-error marker."""
    df = _df_errors()

    ax = plot_error_violins(
        df,
        "A", "B",
        names=["A (Balanced)", "B (Biased)"],
        mode="cbueth",
        overlay="auto",
        show_stats=True,
        cmap="plasma",
        show_grid=False,
        title="Two-Model Overlay",
    )

    # Two polygons (one per model)
    polys = [p for p in ax.patches if isinstance(p, mpl.patches.Polygon)]
    assert len(polys) == 2

    # Overlay mode hides xticks
    assert len(ax.get_xticks()) == 0

    # Legend contains the stats and the zero-error entry
    _, labels = ax.get_legend_handles_labels()
    assert any("Zero Error (center)" in s for s in labels)
    assert any(("med=" in s and "skew=" in s) for s in labels)  # at least one
    # both model labels should include stats
    assert sum(("med=" in s and "skew=" in s) for s in labels) >= 2

    plt.close(ax.figure)


def test_cbueth_split_spokes_three_models_labels_outside_and_stats_in_legend():
    """With overlay=False and k=3, cbueth should draw one polygon per
    model on separate spokes, *no* xticks, outside text labels equal to
    provided names, and stats in the legend."""
    df = _df_errors()

    names = ["A (Balanced)", "B (Biased)", "C (Inconsistent)"]
    ax = plot_error_violins(
        df,
        "A", "B", "C",
        names=names,
        mode="cbueth",
        overlay=False,         # split spokes → outside labels
        show_stats=True,
        cmap="viridis",
        show_grid=False,
        title="Three-Model Split",
    )

    # One polygon per model
    polys = [p for p in ax.patches if isinstance(p, mpl.patches.Polygon)]
    assert len(polys) == 3

    # Split-spokes mode still hides xticks (labels are drawn as free texts)
    assert len(ax.get_xticks()) == 0

    # Outside labels: each provided name should appear as a Text on the axes
    plot_texts = [t.get_text() for t in ax.texts]
    for nm in names:
        assert any(txt == nm for txt in plot_texts)

    # Legend shows stats and zero-error marker
    _, labels = ax.get_legend_handles_labels()
    assert any("Zero Error (center)" in s for s in labels)
    assert sum(("med=" in s and "skew=" in s) for s in labels) >= 3

    plt.close(ax.figure)
