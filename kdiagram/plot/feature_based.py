# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ..api.typing import Acov
from ..compat.matplotlib import get_cmap
from ..decorators import check_non_emptiness, isdf
from ..utils.handlers import columns_manager
from ..utils.plot import (
    acov_to_span,
    map_theta_to_span,
    resolve_polar_axes,
    set_axis_grid,
    set_polar_angular_span,
    setup_polar_axes,
    warn_acov_preference,
)
from ..utils.validator import ensure_2d, exist_features

__all__ = [
    "plot_feature_fingerprint",
    "plot_feature_interaction",
    "plot_fingerprint",
]


@check_non_emptiness(params=["df"])
@isdf
def plot_feature_interaction(
    df: pd.DataFrame,
    theta_col: str,
    r_col: str,
    color_col: str,
    *,
    statistic: str = "mean",
    theta_period: float | None = None,
    theta_bins: int = 24,
    r_bins: int = 10,
    acov: Acov = "default",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
):

    warn_acov_preference(
        acov,
        preferred="default",
        plot_name="feature_interaction",
        advice=(
            "A full 360° span usually produces the most readable comparison; "
            "proceeding with the requested span."
        ),
    )
    # ---- validate & subset
    required = [theta_col, r_col, color_col]
    exist_features(df, features=required)

    data = df[required].dropna().copy()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # axes setup (acov-aware)
    # - returns (fig, ax, span_radians)
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
        zero_at="N",  # 0° at north by default
        clockwise=True,  # CW increasing angles (common in polar charts)
    )

    # map theta data to [0, span]
    th_raw = data[theta_col].to_numpy()
    if theta_period is not None:
        th_scaled = map_theta_to_span(
            th_raw,
            span=span,
            theta_period=theta_period,
        )
    else:
        # min-max scale to the selected span
        tmin, tmax = float(data[theta_col].min()), float(
            data[theta_col].max()
        )
        th_scaled = map_theta_to_span(
            th_raw,
            span=span,
            data_min=tmin,
            data_max=tmax,
        )
    data["theta_mapped"] = th_scaled

    #  build bin edges
    theta_edges = np.linspace(0.0, span, theta_bins + 1)
    r_min, r_max = float(data[r_col].min()), float(data[r_col].max())
    r_edges = np.linspace(r_min, r_max, r_bins + 1)

    # bin assignments (include_lowest=True keeps left edge)
    data["theta_bin"] = pd.cut(
        data["theta_mapped"],
        bins=theta_edges,
        include_lowest=True,
    )
    data["r_bin"] = pd.cut(
        data[r_col],
        bins=r_edges,
        include_lowest=True,
    )

    #  aggregate to a 2D grid (r x theta)
    agg = (
        data.groupby(["r_bin", "theta_bin"], observed=False)[color_col]
        .agg(statistic)
        .reset_index()
    )
    # pivot so rows=r, cols=theta -> shape (r_bins, theta_bins)
    grid = agg.pivot(
        index="r_bin",
        columns="theta_bin",
        values=color_col,
    )

    # ensure grid has the correct shape even if some bins are missing
    # (fill absent bin categories)
    if grid.shape != (r_bins, theta_bins):
        # reindex rows/cols to complete the grid
        grid = grid.reindex(
            index=pd.IntervalIndex.from_breaks(r_edges, closed="right"),
            columns=pd.IntervalIndex.from_breaks(theta_edges, closed="right"),
        )

    Z = grid.to_numpy()  # may contain NaNs (rendered as gaps)

    #  draw heatmap
    T, R = np.meshgrid(theta_edges, r_edges)
    cmap_obj = get_cmap(cmap, default="viridis")
    ax.grid(False)  # background grid off; we'll add styled grid next

    pcm = ax.pcolormesh(
        T,
        R,
        Z,
        cmap=cmap_obj,
        shading="auto",
    )

    # colorbar
    cb = fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.75)
    cb.set_label(
        f"{statistic.capitalize()} of {color_col}",
        fontsize=10,
    )

    #  cosmetics
    ax.set_title(
        title or f"Interaction between {theta_col} and {r_col}",
        fontsize=14,
        y=1.08,
    )
    ax.set_xlabel(theta_col)
    ax.set_ylabel(r_col, labelpad=22)

    # styled grid from shared utility
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    # output
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_feature_interaction.__doc__ = r"""
Plots a polar heatmap of feature interactions.

This function visualizes how a target variable (``color_col``)
changes based on the interaction between two features, one
mapped to the angle and one to the radius. It is a powerful
tool for discovering non-linear relationships and conditional
patterns in the data.

- The **angular position (θ)** represents the binned values
  of the first feature (``theta_col``).
- The **radial distance (r)** represents the binned values
  of the second feature (``r_col``).
- The **color** of each polar sector represents the aggregated
  value (e.g., mean) of the target variable (``color_col``)
  for all data points that fall into that specific bin.

This plot is useful for identifying "hot spots" where a
particular combination of feature values leads to a specific
outcome, revealing complex interactions that are not visible
from one-dimensional feature importance plots.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the feature and target data.
theta_col : str
    The name of the feature to be mapped to the angular axis.
    This is often a cyclical feature like "month" or "hour".
r_col : str
    The name of the feature to be mapped to the radial axis.
color_col : str
    The name of the target column whose value will be
    represented by the color in each bin.
statistic : str, default='mean'
    The aggregation function to apply to ``color_col`` within
    each bin (e.g., 'mean', 'median', 'std').
theta_period : float, optional
    The period of the cyclical data in ``theta_col`` (e.g., 24
    for hours, 12 for months). This ensures the data wraps
    correctly around the polar plot.
theta_bins : int, default=24
    The number of bins to create for the angular feature.
r_bins : int, default=10
    The number of bins to create for the radial feature.
acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'},
    default='default'
    Angular coverage (span) of the plot:

    - ``'default'``: :math:`2\pi` (full circle)
    - ``'half_circle'``: :math:`\pi`
    - ``'quarter_circle'``: :math:`\tfrac{\pi}{2}`
    - ``'eighth_circle'``: :math:`\tfrac{\pi}{4}`
    
title : str, optional
    The title for the plot. If ``None``, a default is generated.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap for the heatmap.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

Raises
------
ValueError
    If any of the specified columns are not found in the DataFrame.

See Also
--------
pandas.cut : Bin values into discrete intervals.
pandas.DataFrame.groupby : 
    Group DataFrame using a mapper or by a Series of columns.
matplotlib.pyplot.pcolormesh : 
    Create a pseudocolor plot with a non-regular rectangular grid.

Notes
-----
This plot is a novel visualization method developed as part of
the analytics framework in :footcite:p:`kouadiob2025`.

The heatmap is constructed by first binning the 2D polar space
defined by ``theta_col`` and ``r_col``. For each resulting polar
sector, the specified ``statistic`` (e.g., mean) is calculated
for all data points whose feature values fall within that
sector. The resulting aggregate value is then mapped to a color,
creating the heatmap effect.

**Coordinate Mapping and Binning**:
    
1.  The angular data from ``theta_col``, :math:`\theta_{data}`, is
    converted to radians :math:`[0, 2\pi]`. If a period :math:`P`
    is given, the mapping is:

    .. math::
       :label: eq:theta_mapping

       \theta_{rad} = \left( \frac{\theta_{data} \pmod P}{P} \right) \cdot 2\pi

2.  The data space is then divided into a grid of :math:`K_r \times K_{\theta}`
    bins, where :math:`K_r` is ``r_bins`` and :math:`K_{\theta}` is
    ``theta_bins``.

3.  For each bin :math:`B_{ij}`, the aggregate value :math:`C_{ij}`
    is computed from the target column ``color_col`` (:math:`z`):

    .. math::
       :label: eq:bin_aggregation

       C_{ij} = \text{statistic}(\{z_k \mid (r_k, \theta_k) \in B_{ij}\})

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.feature_based import plot_feature_interaction
>>>
>>> # Simulate solar panel output data
>>> np.random.seed(0)
>>> n_points = 5000
>>> hour = np.random.uniform(0, 24, n_points)
>>> cloud = np.random.rand(n_points)
>>>
>>> # Output depends on the interaction of daylight and cloud cover
>>> daylight = np.sin(hour * np.pi / 24)**2
>>> cloud_factor = (1 - cloud**0.5)
>>> output = 100 * daylight * cloud_factor + np.random.rand(n_points) * 5
>>> output[(hour < 6) | (hour > 18)] = 0 # No output at night
>>>
>>> df_solar = pd.DataFrame({
...     'hour_of_day': hour,
...     'cloud_cover': cloud,
...     'panel_output': output
... })
>>>
>>> # Generate the plot
>>> ax = plot_feature_interaction(
...     df=df_solar,
...     theta_col='hour_of_day',
...     r_col='cloud_cover',
...     color_col='panel_output',
...     theta_period=24,
...     theta_bins=24,
...     r_bins=8,
...     cmap='inferno',
...     title='Solar Panel Output by Hour and Cloud Cover'
... )

References
----------
.. footbibliography::
"""


@check_non_emptiness(params=["importances"])
def plot_feature_fingerprint(
    importances,
    features: list[str] | None = None,
    labels: list[str] | None = None,
    normalize: bool = True,
    fill: bool = True,
    cmap: str | list[Any] = "tab10",
    title: str = "Feature Impact Fingerprint",
    figsize: tuple[float, float] | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    acov: Acov = "half_circle",
    ax: Axes | None = None,
    savefig: str | None = None,
    dpi: int = 300,
):

    warn_acov_preference(
        acov,
        preferred="half_circle",
        plot_name="fingerprint",
        reason="best shows symmetry across prediction levels",
        advice="we'll still render with your chosen span.",
    )

    # -- shape: (n_layers, n_features)
    imp_mat = ensure_2d(importances)
    n_layers, n_feat_data = imp_mat.shape

    # -- feature names
    if features is None:
        feat_list = [f"feature {i+1}" for i in range(n_feat_data)]
    else:
        feat_list = columns_manager(features, empty_as_none=False)

    if len(feat_list) < n_feat_data:
        feat_list.extend(
            [f"feature {i+1}" for i in range(len(feat_list), n_feat_data)]
        )
    # Truncate if user provided more names than needed
    elif len(feat_list) > n_feat_data:
        warnings.warn(
            "Extra feature names ignored.", UserWarning, stacklevel=2
        )
        feat_list = feat_list[:n_feat_data]

    n_features = len(feat_list)

    # -- layer labels
    if labels is None:
        # Generate default layer labels
        lab_list = [f"Layer {i+1}" for i in range(n_layers)]
    else:
        lab_list = list(labels)
        if len(lab_list) < n_layers:
            warnings.warn(
                "Fewer labels than layers. Auto-filling.",
                UserWarning,
                stacklevel=2,
            )
            lab_list.extend(
                [f"Layer {i+1}" for i in range(len(lab_list), n_layers)]
            )
        elif len(lab_list) > n_layers:
            warnings.warn("Extra labels ignored.", UserWarning, stacklevel=2)
            lab_list = lab_list[:n_layers]

    # -- normalization per layer (row-wise)
    if normalize:
        # allow DataFrame or ndarray
        imp_arr = (
            imp_mat.values if isinstance(imp_mat, pd.DataFrame) else imp_mat
        )
        max_row = imp_arr.max(axis=1, keepdims=True)
        valid = max_row > 1e-9
        norm = np.zeros_like(imp_arr, dtype=float)
        if np.any(valid[:, 0]):
            rows = imp_arr[valid[:, 0]]
            mx = max_row[valid[:, 0]]
            norm[valid[:, 0]] = rows / mx
        imp_arr = norm
    else:
        imp_arr = (
            imp_mat.values if isinstance(imp_mat, pd.DataFrame) else imp_mat
        )

    # -- angular span per `acov`
    span = acov_to_span(acov)  # 2pi, pi, pi/2, or pi/4
    # spread features across the chosen span
    angles = np.linspace(0.0, span, n_features, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # -- axes construction / reuse
    ax = resolve_polar_axes(
        ax=ax, acov=acov, figsize=figsize, clockwise=False, zero_at="E"
    )
    # set [thetamin, thetamax] for the chosen coverage
    set_polar_angular_span(ax, acov)

    # -- color handling (cmap or list)
    try:
        cmap_obj = get_cmap(cmap, default="tab10", failsafe="discrete")
        cols = [cmap_obj(i / max(1, n_layers - 1)) for i in range(n_layers)]
    except ValueError:
        if isinstance(cmap, list):
            cols = list(cmap)
            if len(cols) < n_layers:
                warnings.warn(
                    "Provided color list shorter than layers; "
                    "colors will repeat.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Invalid cmap. Falling back to 'tab10'.",
                UserWarning,
                stacklevel=2,
            )
            cmap_obj = get_cmap("tab10", default="tab10", failsafe="discrete")
            cols = [
                cmap_obj(i / max(1, n_layers - 1)) for i in range(n_layers)
            ]

    # -- draw each layer polygon -
    for idx, row in enumerate(imp_arr):
        vals = row.tolist()
        vals_closed = vals + vals[:1]
        color = cols[idx % len(cols)]
        label = lab_list[idx]

        # outline
        ax.plot(
            angles_closed,
            vals_closed,
            color=color,
            linewidth=2,
            label=label,
        )
        # fill
        if fill:
            ax.fill(angles_closed, vals_closed, color=color, alpha=0.25)

    # -- labels, ticks, limits
    ax.set_title(title, size=16, y=1.1)
    ax.set_xticks(angles)
    ax.set_xticklabels(feat_list, fontsize=11)

    ax.set_ylim(bottom=0.0)
    if normalize:
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(
            ["0.25", "0.50", "0.75", "1.00"], fontsize=9, color="gray"
        )
    else:
        # let Matplotlib pick; still hide crowded labels if desired
        pass

    # -- grid styling via helper
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    # -- legend outside the plot
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # -- layout + save/show
    plt.tight_layout(pad=2.0)
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=dpi)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")

        plt.close()
    else:
        plt.show()

    return ax


plot_feature_fingerprint.__doc__ = r"""
Create a radar chart visualizing feature-importance profiles.

This function draws a polar (radar) chart that compares how the
importance of a common set of features varies across multiple
groups/layers (e.g., different models, years, or spatial zones).
Each group is drawn as a closed polygon, producing an interpretable
"fingerprint" of relative influence across features (see also the
dataset helper :func:`~kdiagram.datasets.make_fingerprint_data`;
concept introduced in :footcite:t:`kouadiob2025`.

The angular position encodes the feature index, and the radius encodes
its (optionally normalized) importance value. Normalization allows
shape-only comparison across layers, independent of absolute scale.

Parameters
----------
importances : array-like of shape (n_layers, n_features)
    The importance matrix. Each row corresponds to one layer/group
    and each column to a feature. Accepts a list of lists, a NumPy
    array, or a pandas DataFrame.

features : list of str, optional
    Names of the features (length must match the number of columns
    in ``importances``). If ``None``, generic names
    ``['feature 1', ..., 'feature N']`` are generated.

labels : list of str, optional
    Display names for layers (length should match ``n_layers``).
    If ``None``, generic names ``['Layer 1', ..., 'Layer M']`` are
    generated. When counts mismatch, the function pads/truncates and
    issues a warning.

normalize : bool, default=True
    If ``True``, normalize each row to the unit interval via
    :math:`r'_{ij} = r_{ij}/\max_k r_{ik}` (safe-dividing by zero
    yields zeros). This highlights *shape* differences across layers.
    If ``False``, raw magnitudes are plotted.

fill : bool, default=True
    If ``True``, fill each polygon with a translucent color; otherwise
    draw outlines only.

cmap : str or list, default='tab10'
    Either a Matplotlib colormap name (e.g., ``'viridis'``,
    ``'plasma'``, ``'tab10'``) or an explicit list of colors. Lists
    shorter than the number of layers will cycle with a warning.

title : str, default='Feature Impact Fingerprint'
    Figure title.

figsize : tuple of (float, float), optional
    Figure size in inches. If ``None``, a sensible default is used.

show_grid : bool, default=True
    Whether to show polar grid lines.

savefig : str, optional
    Path to save the figure (e.g., ``'fingerprint.png'``). If
    omitted, the plot is shown interactively.

Returns
-------
ax : matplotlib.axes.Axes
    The polar axes containing the radar chart (useful for further
    customization).

Notes
-----
**Angular encoding.** With :math:`N` features, angular positions are
equally spaced as:

.. math::

   \theta_j \;=\; \frac{2\pi j}{N}, \qquad j = 0, \dots, N-1.

**Closing polygons.** To draw closed fingerprints, the first vertex
:math:`(\theta_0, r_{i0})` is appended again at :math:`2\pi` for each
layer :math:`i`.

**Row-wise normalization (default).** If ``normalize=True``, each row
:math:`\mathbf r_i=(r_{i0},\dots,r_{i,N-1})` is scaled to its maximum:

.. math::

   r'_{ij} \;=\;
   \begin{cases}
     \dfrac{r_{ij}}{\max_k r_{ik}}, & \max_k r_{ik} > 0,\\[6pt]
     0, & \text{otherwise,}
   \end{cases}

which emphasizes *shape* differences between layers but removes absolute
magnitude information. Set ``normalize=False`` to compare magnitudes.

**Alternative min–max scaling (pre-processing).** If you prefer values
distributed over :math:`[0,1]` using the local range, apply this
transformation per row before calling the function as:

.. math::

   r''_{ij} \;=\;
   \frac{r_{ij} - \min_k r_{ik}}
        {\max_k r_{ik} - \min_k r_{ik} + \varepsilon},

with a small :math:`\varepsilon>0` to avoid division by zero.

**Data assumptions.** Importance values are expected to be non-negative.
Rows with a non-positive maximum (all zeros or all negative) become
zeros under the default normalization. If your data can be negative,
either:
(1) set ``normalize=False`` and choose appropriate radial limits, or
(2) shift/scale to non-negative values (e.g., min–max per row).

**Missing/invalid values.** ``NaN`` or ``inf`` entries propagate to the
plot and may render gaps. Clean data beforehand, e.g.:

.. code-block:: python

   import numpy as np
   X = np.asarray(importances, float)
   X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

**Radial limits and ticks.** The plot enforces a non-negative radius
(``ax.set_ylim(bottom=0)``). For unnormalized data, you may set a
custom maximum:

.. code-block:: python

   ax.set_rmax( np.nanmax(importances) )

Optionally add/readjust radial ticks for readability:

.. code-block:: python

   ax.set_yticks([0.25, 0.5, 0.75, 1.0])
   ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])

**Feature order matters.** The perceived shape depends on feature
ordering around the circle. Keep a consistent, meaningful order across
comparisons (e.g., domain grouping or sorted by average importance).

**Many features or layers.** With large :math:`N`, tick labels can
overlap. Consider thinning labels or rotating them:

.. code-block:: python

   angles = ax.get_xticks()
   ax.set_xticks(angles[::2])
   ax.set_xticklabels([lbl for i, lbl in enumerate(features) if i % 2 == 0],
                      rotation=25, ha="right")

For many layers, prefer a discrete colormap and a multi-column legend
or move it outside:

.. code-block:: python

   ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=2)

**Color and accessibility.** Use colorblind-friendly palettes (e.g.,
``'tab10'``, ``'tab20'``) or pass an explicit color list. Avoid relying
on color alone when printing in grayscale—consider distinct linestyles.

**Complexity.** Runtime and memory scale as
:math:`\mathcal O(MN)` for :math:`M` layers and :math:`N` features.
For very large inputs, down-select features or layers for clarity.

**Utilities.** Inputs are coerced to a numeric 2D array and feature
names managed via lightweight helpers (e.g., ``ensure_2d``,
``columns_manager``). Name count mismatches are padded/truncated with a
warning rather than raising.

See Also
--------
kdiagram.datasets.make_fingerprint_data :
    Generate a synthetic importance matrix suitable for this plot.
kdiagram.plot.relationship.plot_relationship :
    Polar scatter for true–predicted relationships.
matplotlib.pyplot.polar :
    Underlying polar plotting primitives.

Examples
--------
Generate random importances and plot with normalization and fills.

>>> import numpy as np
>>> from kdiagram.plot.feature_based import plot_feature_fingerprint
>>> rng = np.random.default_rng(42)
>>> imp = rng.random((3, 6))   # 3 layers, 6 features
>>> feats = [f'Feature {i+1}' for i in range(6)]
>>> labels = ['Model A', 'Model B', 'Model C']
>>> ax = plot_feature_fingerprint(
...     importances=imp,
...     features=feats,
...     labels=labels,
...     title='Random Feature Importance Comparison',
...     cmap='Set3',
...     normalize=True,
...     fill=True
... )

Year-over-year weights without normalization.

>>> features = ['rainfall', 'GWL', 'seismic', 'density', 'geo']
>>> weights = [
...     [0.2, 0.4, 0.1, 0.6, 0.3],  # 2023
...     [0.3, 0.5, 0.2, 0.4, 0.4],  # 2024
...     [0.1, 0.6, 0.2, 0.5, 0.3],  # 2025
... ]
>>> years = ['2023', '2024', '2025']
>>> ax = plot_feature_fingerprint(
...     importances=weights,
...     features=features,
...     labels=years,
...     title='Feature Influence Over Years',
...     cmap='tab10',
...     normalize=False
... )

References
----------
.. footbibliography::
"""


@check_non_emptiness(params=["importances"])
def plot_fingerprint(
    importances: pd.DataFrame | np.ndarray,
    *,
    precomputed: bool = True,
    y_col: str | None = None,
    group_col: str | None = None,
    method: str = "abs_corr",
    features: list[str] | None = None,
    labels: list[str] | None = None,
    normalize: bool = True,
    fill: bool = True,
    cmap: str | list[Any] = "tab10",
    title: str = "Feature Impact Fingerprint",
    figsize: tuple[float, float] | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    acov: str = "full",
    ax: Axes | None = None,
    dpi: int = 300,
):
    # 1) Prepare the importance matrix M (layers x features)
    if precomputed:
        if isinstance(importances, pd.DataFrame):
            # numeric columns only (order preserved)
            num = importances.select_dtypes(include=[np.number])
            if num.empty:
                raise ValueError(
                    "No numeric columns found in precomputed DataFrame."
                )
            M = num.to_numpy()
            feat_names = list(num.columns)
            layer_labels = [
                str(x)
                for x in (num.index if len(num.index) else range(len(M)))
            ]
        else:
            M = ensure_2d(importances)
            feat_names = features or [
                f"feature {i+1}" for i in range(M.shape[1])
            ]
            layer_labels = labels or [
                f"Layer {i+1}" for i in range(M.shape[0])
            ]
    else:
        # must be a DataFrame to compute importance
        if not isinstance(importances, pd.DataFrame):
            raise ValueError(
                "When precomputed=False, `importances` must be a DataFrame."
            )

        df = importances.copy()
        if y_col is not None and y_col not in df.columns:
            raise ValueError(f"y_col '{y_col}' not in DataFrame.")

        # choose numeric feature columns (exclude y/group)
        drop_cols = {c for c in [y_col, group_col] if c is not None}
        feat_candidates = df.drop(columns=list(drop_cols), errors="ignore")
        feat_candidates = feat_candidates.select_dtypes(include=[np.number])
        if feat_candidates.empty:
            raise ValueError("No numeric feature columns to compute from.")

        feat_names = features or list(feat_candidates.columns)

        def _compute_one(block: pd.DataFrame) -> pd.Series:
            X = block[feat_names]
            if method == "abs_corr" and y_col is not None:
                # abs Pearson correlation; NaNs->0
                vals = []
                yv = block[y_col].to_numpy()
                for c in feat_names:
                    xc = block[c].to_numpy()
                    if np.std(xc) < 1e-12 or np.std(yv) < 1e-12:
                        vals.append(0.0)
                    else:
                        corr = np.corrcoef(xc, yv)[0, 1]
                        vals.append(abs(corr))
                return pd.Series(vals, index=feat_names)
            elif method in {"std", "var", "mad"}:
                if method == "std":
                    s = X.std(axis=0, ddof=1)
                elif method == "var":
                    s = X.var(axis=0, ddof=1)
                else:
                    s = (X - X.median()).abs().median(axis=0)
                return s.fillna(0.0)
            else:
                raise ValueError(
                    "Unknown method. Use 'abs_corr', 'std', 'var', or 'mad'."
                )

        if group_col is not None and group_col in df.columns:
            groups = df[group_col].dropna().unique().tolist()
            groups = sorted(groups, key=lambda x: str(x))
            rows = []
            for g in groups:
                part = df[df[group_col] == g]
                if len(part) == 0:
                    rows.append(np.zeros(len(feat_names)))
                else:
                    rows.append(
                        _compute_one(part).reindex(feat_names).to_numpy()
                    )
            M = np.vstack(rows)
            layer_labels = [str(g) for g in groups]
        else:
            s = _compute_one(df).reindex(feat_names).to_numpy()
            M = s.reshape(1, -1)
            layer_labels = ["Layer 1"]

    n_layers, n_feats = M.shape

    # 2) Normalize row-wise (optional)
    if normalize:
        rmax = M.max(axis=1, keepdims=True)
        ok = rmax > 1e-9
        M_norm = np.zeros_like(M, dtype=float)
        if np.any(ok[:, 0]):
            M_norm[ok[:, 0]] = M[ok[:, 0]] / rmax[ok[:, 0]]
        M = M_norm

    # ensure features/labels list lengths
    if len(feat_names) < n_feats:
        feat_names += [
            f"feature {i+1}" for i in range(len(feat_names), n_feats)
        ]
    elif len(feat_names) > n_feats:
        warnings.warn(
            "More feature names than columns; extra ignored.",
            UserWarning,
            stacklevel=2,
        )
        feat_names = feat_names[:n_feats]

    if len(layer_labels) < n_layers:
        layer_labels += [
            f"Layer {i+1}" for i in range(len(layer_labels), n_layers)
        ]
    elif len(layer_labels) > n_layers:
        warnings.warn(
            "More labels than layers; extra ignored.",
            UserWarning,
            stacklevel=2,
        )
        layer_labels = layer_labels[:n_layers]

    # 3) Axes set-up (polar, acov)
    fig, ax, span = setup_polar_axes(
        ax, acov=acov, figsize=figsize or (9, 6), zero_at="N", clockwise=True
    )

    # angles along the visible span
    ang = np.linspace(0.0, span, n_feats, endpoint=False)
    ang_closed = np.r_[ang, ang[:1]]

    # 4) Colors
    try:
        cmap_obj = get_cmap(cmap, default="tab10", failsafe="discrete")
        cols = [cmap_obj(i / max(1, n_layers - 1)) for i in range(n_layers)]
    except ValueError:
        if isinstance(cmap, list):
            cols = [cmap[i % len(cmap)] for i in range(n_layers)]
        else:  # fallback
            cmap_obj = get_cmap("tab10", default="tab10", failsafe="discrete")
            cols = [
                cmap_obj(i / max(1, n_layers - 1)) for i in range(n_layers)
            ]

    # 5) Plot layers
    for i in range(n_layers):
        vals = M[i].tolist()
        vals_closed = vals + vals[:1]
        col = cols[i % len(cols)]

        ax.plot(
            ang_closed, vals_closed, color=col, lw=2, label=layer_labels[i]
        )
        if fill:
            ax.fill(ang_closed, vals_closed, color=col, alpha=0.25)

    # 6) Grid, ticks, and custom angular labels
    set_axis_grid(
        ax, show_grid=show_grid, grid_props=grid_props or {"alpha": 0.45}
    )
    if normalize:
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="0.45")
    else:
        ax.set_ylim(bottom=0)

    # Hide default xticklabels and draw our own to reduce overlap
    ax.set_xticks([])
    _draw_angular_labels(
        ax=ax, angles=ang, labels=feat_names, r=ax.get_ylim()[1], span=span
    )

    # 7) Title / legend / layout
    ax.set_title(title, fontsize=16, pad=14, y=1.06)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.02),
        frameon=False,
        borderaxespad=0.0,
    )
    plt.subplots_adjust(right=0.78, top=0.90, bottom=0.08, left=0.08)

    if savefig:
        try:
            fig.savefig(savefig, bbox_inches="tight", dpi=dpi)
        except Exception as e:  # pragma: no cover
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


# ---------- helpers ----------
def _draw_angular_labels(
    *,
    ax: plt.Axes,
    angles: np.ndarray,
    labels: list[str],
    r: float,
    span: float,
) -> None:
    """Readable, rotated labels, with slight staggering on smaller spans."""
    n = len(labels)
    fs = max(8, 13 - int(0.25 * n))  # shrink font as labels grow
    r_base = r * 1.06
    narrow = np.degrees(span) <= 180 + 1e-6

    for k, (theta, lab) in enumerate(zip(angles, labels)):
        deg = np.degrees(theta)
        rot = deg - 90.0
        ha = "left" if 0.0 <= deg <= 180.0 else "right"
        r_off = r_base * (1.0 + (0.03 if (narrow and (k % 2)) else 0.0))
        ax.text(
            theta,
            r_off,
            lab,
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=fs,
        )
