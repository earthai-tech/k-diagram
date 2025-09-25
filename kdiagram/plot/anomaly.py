# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

from __future__ import annotations

import warnings
from typing import Any, Literal

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from ..compat.matplotlib import get_cmap, get_colors
from ..decorators import check_non_emptiness, isdf
from ..metrics import clustered_anomaly_severity
from ..utils.generic_utils import get_valid_kwargs
from ..utils.plot import (
    Acov,
    map_theta_to_span,
    resolve_polar_span,
    set_axis_grid,
    setup_polar_axes,
)
from ..utils.validator import exist_features

__all__ = [
    "plot_anomaly_severity",
    "plot_anomaly_profile",
    "plot_anomaly_glyphs",
    "plot_cas_profile",
    "plot_glyphs",
    "plot_cas_layers",
]


@check_non_emptiness
@isdf
def plot_anomaly_severity(
    df: pd.DataFrame,
    actual_col: str,
    q_low_col: str,
    q_up_col: str,
    *,
    window_size: int = 21,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 9.0),
    cmap: str = "plasma",
    s: int = 40,
    alpha: float = 0.8,
    acov: str = "default",
    mask_angle: bool = True,
    mask_radius: bool = False,
    show_grid: bool = True,
    grid_props: dict = None,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    required_cols = [actual_col, q_low_col, q_up_col]
    exist_features(df, features=required_cols)

    data = df[required_cols].dropna().copy()
    if data.empty:
        warnings.warn("DataFrame is empty after dropping NaNs.", stacklevel=2)
        return None

    # 1. Calculate the score and get detailed data
    cas_score, details = clustered_anomaly_severity(
        data[actual_col].to_numpy(),
        data[q_low_col].to_numpy(),
        data[q_up_col].to_numpy(),
        window_size=window_size,
        return_details=True,
    )

    anomalies = details[details["is_anomaly"]].copy()
    if anomalies.empty:
        warnings.warn("No anomalies detected in the data.", stacklevel=2)
        return None

    # 2. Set up polar axes and angular mapping
    fig, ax, span = setup_polar_axes(ax, acov=acov, figsize=figsize)

    theta = (anomalies.index.to_numpy() / len(data)) * float(span)

    # 3. Prepare visual mappings
    radii = anomalies["magnitude"].to_numpy()
    density = anomalies["local_density"].to_numpy()

    cmap_obj = get_cmap(cmap, default="plasma")
    norm = Normalize(vmin=density.min(), vmax=density.max())
    colors = cmap_obj(norm(density))

    # 4. Plot the anomalies
    under_mask = anomalies["type"] == "under"
    over_mask = anomalies["type"] == "over"

    if np.any(over_mask):
        ax.scatter(
            theta[over_mask],
            radii[over_mask],
            c=colors[over_mask],
            s=s,
            alpha=alpha,
            marker="o",
            label="Over-prediction (Risk Underestimated)",
            edgecolor="k",
            linewidth=0.5,
        )

    if np.any(under_mask):
        ax.scatter(
            theta[under_mask],
            radii[under_mask],
            c=colors[under_mask],
            s=s * 1.5,
            alpha=alpha,
            marker="X",
            label="Under-prediction (Risk Overestimated)",
            edgecolor="w",
            linewidth=0.5,
        )

    # 5. Add formatting and interpretation aids
    title_str = title or (
        f"Anomaly Severity Analysis\nCAS Score: {cas_score:.4f}"
    )
    ax.set_title(title_str, fontsize=16, y=1.1)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Local Anomaly Density", fontsize=10)

    ax.set_ylabel("Anomaly Magnitude", labelpad=25)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.05))

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_angle:
        ax.set_xticklabels([])

    if mask_radius:
        ax.set_yticklabels([])

    # fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_anomaly_severity.__doc__ = r"""
Visualizes clustered anomaly severity using a polar scatter plot.

This function creates a diagnostic plot to analyze forecast
failures. It identifies anomalies where the true value falls
outside the predicted interval and visualizes their location,
magnitude, type, and clustering density in a single,
compact polar view.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted
    quantile values.

actual_col : str
    The name of the column containing the true observed values.

q_low_col : str
    The name of the column for the lower bound of the
    prediction interval.

q_up_col : str
    The name of the column for the upper bound of the
    prediction interval.

window_size : int, default=21
    The size of the moving window used to calculate the local
    density of anomalies, which defines a "cluster".

title : str, optional
    A custom title for the plot. If ``None``, a default title
    including the CAS score is generated.

figsize : tuple of (float, float), default=(9.0, 9.0)
    The figure size in inches.

cmap : str, default='plasma'
    The colormap for coloring points based on local density.

s : int, default=40
    The marker size for the scatter points.

alpha : float, default=0.8
    The transparency of the scatter points.

acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'},\
    default='default'
    
    Specifies the angular coverage of the polar plot.

mask_angle : bool, default=True
    If ``True``, hides the angular tick labels (e.g., degrees).

mask_radius : bool, default=False
    If ``True``, hides the radial tick labels.

ax : matplotlib.axes.Axes, optional
    An existing polar axes to draw the plot on. If ``None``, a
    new figure and axes are created.

savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if no anomalies are detected in the data.

See Also
--------
plot_anomaly_profile : A stylized "fiery ring" version of this plot.
plot_anomaly_glyphs : A version using informative glyphs instead of dots.
clustered_anomaly_severity_score : The underlying metric function.

Notes
-----
This plot visualizes the four key dimensions of forecast
failures as described in :footcite:t:`kouadioc2025`.

**Visual Mapping:**

- **Angle (:math:`varepsilon`)**: The sample index, showing *where* in the
  dataset the failure occurred.
- **Radius (`r`)**: The **Anomaly Magnitude**—the distance from the
  true value to the nearest violated interval bound. Larger
  radii indicate more severe failures.
- **Color**: The **Local Anomaly Density**. Hotter colors indicate
  the anomaly is part of a dense cluster of other failures.
- **Marker Shape**: The **Type** of anomaly.

    - 'o' (circle): Over-prediction (risk was underestimated).
    - 'X': Under-prediction (risk was overestimated).

The title of the plot automatically includes the overall
Clustered Anomaly Severity (CAS) score for a quantitative summary.

References
----------
.. footbibliography::

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.anomaly import plot_anomaly_severity
>>>
>>> # Simulate data with a cluster of severe failures
>>> np.random.seed(0)
>>> n_samples = 400
>>> y_true = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, n_samples))
>>> y_qlow = y_true - 10
>>> y_qup = y_true + 10
>>> y_true[100:140] = y_qup[100:140] + np.random.uniform(10, 25, 40)
>>>
>>> df = pd.DataFrame({
...     "actual": y_true, "q10": y_qlow, "q90": y_qup
... })
>>>
>>> ax = plot_anomaly_severity(
...     df,
...     actual_col="actual",
...     q_low_col="q10",
...     q_up_col="q90",
...     window_size=31,
...     title="Severity of Clustered Anomalies"
... )

"""


@check_non_emptiness
@isdf
def plot_anomaly_profile(
    df: pd.DataFrame,
    actual_col: str,
    q_low_col: str,
    q_up_col: str,
    *,
    window_size: int = 21,
    theta_bins: int = 72,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 9.0),
    cmap: str = "plasma",
    colors: list[str] = None,
    alpha: float = 0.8,
    acov: str = "default",
    show_grid: bool = True,
    grid_props: dict = None,
    ax: Axes | None = None,
    savefig: str | None = None,
    jitter: float = 0.85,
    max_flares_per_bin: int | None = None,
    flare_scale: str = "sqrt",
    flare_clip: float | None = None,
    flare_linewidth: float = 1.4,
    ring_height: float = 0.06,
    ring_alpha: float = 0.95,
    legend_anchor: tuple[float, float] = (1.35, 1.04),
    **kwargs: Any,
) -> Axes | None:
    # ---- data & metric
    req = [actual_col, q_low_col, q_up_col]
    data = df[req].dropna().copy()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs.",
            stacklevel=2,
        )
        return None

    cas_score, details = clustered_anomaly_severity(
        data[actual_col].to_numpy(),
        data[q_low_col].to_numpy(),
        data[q_up_col].to_numpy(),
        window_size=window_size,
        return_details=True,
    )

    anomalies = details[details["is_anomaly"]].copy()
    if anomalies.empty:
        warnings.warn(
            "No anomalies detected in the data.",
            stacklevel=2,
        )
        return None

    # ---- axes
    fig, ax, span = setup_polar_axes(ax, acov=acov, figsize=figsize)

    # ---- angle mapping & binning
    # angle ~ index position in [0, span)
    anomalies["theta"] = (anomalies.index.to_numpy() / len(data)) * float(
        span
    )

    bin_edges = np.linspace(0, float(span), theta_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = float(span) / theta_bins

    # integer bin ids in [0, theta_bins)
    bin_id = (
        np.digitize(
            anomalies["theta"].to_numpy(),
            bin_edges,
            right=False,
        )
        - 1
    )
    bin_id = np.clip(bin_id, 0, theta_bins - 1)
    anomalies["theta_bin_id"] = bin_id

    # ---- ring (density heat)
    cmap_obj = get_cmap(cmap, default="plasma")
    vmin = details["local_density"].min()
    vmax = details["local_density"].max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    ring_density = anomalies.groupby("theta_bin_id", observed=False)[
        "local_density"
    ].mean()

    ax.bar(
        bin_centers[ring_density.index],
        height=ring_height,
        width=bin_width,
        bottom=1.0,
        color=cmap_obj(norm(ring_density.values)),
        alpha=ring_alpha,
        linewidth=0,
    )

    # ---- flare helpers
    def _scale_mag(x: np.ndarray) -> np.ndarray:
        # map magnitude -> radial length
        x = np.asarray(x)
        if flare_scale == "linear":
            out = x.copy()
        elif flare_scale == "sqrt":
            out = np.sqrt(np.maximum(x, 0.0))
        elif flare_scale == "log":
            out = np.log1p(np.maximum(x, 0.0))
        else:
            raise ValueError("flare_scale must be {'linear','sqrt','log'}")
        if flare_clip is not None:
            out = np.minimum(out, float(flare_clip))
        return out

    # safe jitter
    jitter = float(np.clip(jitter, 0.0, 1.0))

    # pick colors (over, under)
    over_color, under_color = get_colors(
        2,
        colors=colors or ["#E74C3C", "#3498DB"],
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # ---- flares with in-bin angular dodging
    for b in range(theta_bins):
        for typ, base_r, color in (
            ("over", 1.05, over_color),
            ("under", 1.00, under_color),
        ):
            sub = anomalies[
                (anomalies["theta_bin_id"] == b) & (anomalies["type"] == typ)
            ]
            if sub.empty:
                continue

            # limit count per bin (keep largest)
            if (
                max_flares_per_bin is not None
                and len(sub) > max_flares_per_bin
            ):
                sub = sub.nlargest(max_flares_per_bin, "magnitude")

            # small first, big last (nice layering)
            sub = sub.sort_values("magnitude", ascending=True)

            # angular offsets inside the bin
            n = len(sub)
            if n == 1:
                thetas = np.array([bin_centers[b]])
            else:
                spread = bin_width * jitter
                offsets = np.linspace(-spread / 2.0, spread / 2.0, n)
                thetas = bin_centers[b] + offsets

            mags = _scale_mag(sub["magnitude"].to_numpy())

            # draw flares
            if typ == "over":
                for th, m in zip(thetas, mags):
                    ax.plot(
                        [th, th],
                        [base_r, base_r + m],
                        color=color,
                        linewidth=flare_linewidth,
                        alpha=alpha,
                        solid_capstyle="round",
                        zorder=3,
                    )
            else:
                for th, m in zip(thetas, mags):
                    ax.plot(
                        [th, th],
                        [base_r, max(0.0, base_r - m)],
                        color=color,
                        linewidth=flare_linewidth,
                        alpha=alpha,
                        solid_capstyle="round",
                        zorder=3,
                    )

    # ---- formatting
    ttl = title or (f"Anomaly Severity Profile\nCAS: {cas_score:.4f}")
    ax.set_title(ttl, fontsize=16, y=1.04)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    # legend outside (no overlap)
    legend_elems = [
        Line2D(
            [0],
            [0],
            color=over_color,
            lw=2,
            label="Over-prediction Flare",
        ),
        Line2D(
            [0],
            [0],
            color=under_color,
            lw=2,
            label="Under-prediction Flare",
        ),
    ]
    ax.legend(
        handles=legend_elems,
        loc="upper right",
        bbox_to_anchor=legend_anchor,
    )

    # colorbar for ring density
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(
        "Local Anomaly Density (Ring Color)",
        fontsize=10,
    )

    # radial limit: ring + longest scaled flare
    max_len = _scale_mag(anomalies["magnitude"].to_numpy()).max()
    ax.set_ylim(0.0, 1.1 + max_len)

    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_anomaly_profile.__doc__ = r"""
Visualize anomaly severity as a polar profile or "fiery ring".

This figure emphasizes readability for papers.  It encodes
clustered anomaly density as a colored ring, and shows each
failed sample as a short "flare" growing inward or outward.
The design avoids overlap by angular binning and in–bin
dodging.

Parameters
----------
df : pandas.DataFrame
    Input table that holds the observed series and the two
    prediction bounds.  Missing rows are dropped.

actual_col : str
    Column name of the observed values.

q_low_col : str
    Column name of the lower prediction bound.

q_up_col : str
    Column name of the upper prediction bound.

window_size : int, default=21
    Window length used to compute the local anomaly density.
    Odd values are recommended for symmetric windows.

theta_bins : int, default=72
    Number of angular bins used for the density ring and to
    group flares for anti-overlap dodging.

title : str, optional
    Figure title.  If not given, a title including the CAS
    score is used.

figsize : tuple of float, default=(9.0, 9.0)
    Figure size in inches.

cmap : str, default='plasma'
    Colormap used for the density ring.

colors : list of str, optional
    Two colors for the flares (over, under).  If not given,
    defaults are chosen.

alpha : float, default=0.8
    Global alpha for flare lines.

acov : {'default','half_circle','quarter_circle',
      'eighth_circle'}, default='default'

    Angular coverage preset.  This controls the polar span.

show_grid : bool, default=True
    If True, show a light polar grid.

grid_props : dict, optional
    Keyword arguments forwarded to the grid styling helper.

ax : matplotlib.axes.Axes, optional
    Existing polar axes.  If None, a new figure is created.

savefig : str, optional
    Path to save the figure.  If None, the figure is shown.

jitter : float, default=0.85
    Fraction of bin width used to spread flares within each
    bin to reduce overlap.  Clipped to [0, 1].

max_flares_per_bin : int, optional
    If given, at most this many flares are drawn per bin.
    The largest magnitudes are kept.

flare_scale : {'linear','sqrt','log'}, default='sqrt'
    Transform applied to anomaly magnitude before mapping to
    flare length.  Use 'sqrt' or 'log' to tame outliers.

flare_clip : float, optional
    Maximum flare length after scaling.  If None, no clipping
    is applied.

flare_linewidth : float, default=1.4
    Line width of the flares.

ring_height : float, default=0.06
    Radial thickness of the density ring.

ring_alpha : float, default=0.95
    Alpha value for the density ring.

legend_anchor : tuple of float, default=(1.35, 1.04)
    Anchor for the legend box (axes coordinates).

Returns
-------
ax : matplotlib.axes.Axes or None
    The polar axes with the plot.  Returns None if no anomaly
    is detected after preprocessing.

Notes
-----
**Visual mapping.**

- Angle :math:`\varepsilon`: encodes sample position (index order).
- Ring color: mean local anomaly density within each angular
  bin.  Hot colors indicate clustered failures.
- Flares: one per failed sample.

  - Length: anomaly magnitude (scaled and optionally clipped).
  - Direction: type.  Outward = over-prediction.  Inward =
    under-prediction.

**Anomalies.**

A point is an anomaly when the observed value lies outside the
prediction interval.  Density is computed with a moving window
of length ``window_size`` and then averaged within bins.

**Styling and overlap control.**
Angular binning plus in-bin jitter reduce overlap.  Use
``theta_bins`` to raise angular resolution, ``jitter`` to
control spread, ``max_flares_per_bin`` to cap clutter, and
``flare_scale`` or ``flare_clip`` to keep lengths balanced.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.anomaly import plot_anomaly_profile
>>> rng = np.random.default_rng(30)
>>> n = 500
>>> base = np.sin(np.linspace(0, 6*np.pi, n)) * 10 + 20
>>> qlow = base - 5
>>> qup = base + 5
>>> y = base.copy()
>>> y[100:130] += rng.uniform(6, 12, 30)   # over
>>> y[300:330] -= rng.uniform(6, 12, 30)   # under
>>> df = pd.DataFrame({'actual': y, 'q10': qlow, 'q90': qup})
>>> ax = plot_anomaly_profile(
...     df,
...     actual_col='actual',
...     q_low_col='q10',
...     q_up_col='q90',
...     window_size=31,
...     theta_bins=96,
...     jitter=0.9,
... )
>>> _ = ax.figure  # keep handle for saving outside

See Also
--------
plot_anomaly_severity : Polar scatter of anomaly points.
plot_anomaly_glyphs : Glyph-based variant with richer marks.
clustered_anomaly_severity : Metric used by this plot.

References
----------
.. [1] Kouadio, K. L., et al. 2025.  CAS: Cluster-Aware Scoring 
       for Probabilistic Forecasts. in review.
.. [2] Gneiting, T., and Raftery, A. E. 2007.  Strictly
       proper scoring rules, prediction, and estimation.
       JASA, 102(477), 359–378.
"""


@check_non_emptiness
@isdf
def plot_anomaly_glyphs(
    df: pd.DataFrame,
    actual_col: str,
    q_low_col: str,
    q_up_col: str,
    *,
    window_size: int = 21,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 9.0),
    cmap: str = "inferno",
    s: int = 70,
    alpha: float = 0.85,
    acov: str = "default",
    mask_angle: bool = True,
    mask_radius: bool = False,
    show_grid: bool = True,
    grid_props: dict = None,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    required_cols = [actual_col, q_low_col, q_up_col]
    data = df[required_cols].dropna().copy()
    if data.empty:
        warnings.warn("DataFrame is empty after dropping NaNs.", stacklevel=2)
        return None

    cas_score, details = clustered_anomaly_severity(
        data[actual_col].to_numpy(),
        data[q_low_col].to_numpy(),
        data[q_up_col].to_numpy(),
        window_size=window_size,
        return_details=True,
    )

    anomalies = details[details["is_anomaly"]].copy()
    if anomalies.empty:
        warnings.warn("No anomalies detected in the data.", stacklevel=2)
        return None

    fig, ax, span = setup_polar_axes(ax, acov=acov, figsize=figsize)

    theta = (anomalies.index.to_numpy() / len(data)) * float(span)

    # Visual Mappings
    radii = anomalies["magnitude"].to_numpy()
    density = anomalies["local_density"].to_numpy()

    cmap_obj = get_cmap(cmap, default="inferno")
    norm = Normalize(vmin=density.min(), vmax=density.max())
    colors = cmap_obj(norm(density))

    # Plot Glyphs in two passes (for under- and over-predictions)
    under_mask = anomalies["type"] == "under"
    over_mask = anomalies["type"] == "over"

    # Over-prediction glyphs (outward triangles)
    if np.any(over_mask):
        ax.scatter(
            theta[over_mask],
            radii[over_mask],
            c=colors[over_mask],
            s=s,
            alpha=alpha,
            marker="^",  # Triangle pointing up (radially outward)
            label="Over-prediction (Risk Underestimated)",
            edgecolor="w",
            linewidth=0.5,
        )

    # Under-prediction glyphs (inward triangles)
    if np.any(under_mask):
        ax.scatter(
            theta[under_mask],
            radii[under_mask],
            c=colors[under_mask],
            s=s,
            alpha=alpha,
            marker="v",  # Triangle pointing down (radially inward)
            label="Under-prediction (Risk Overestimated)",
            edgecolor="w",
            linewidth=0.5,
        )

    # Formatting
    title_str = (
        title or f"Polar Anomaly Glyph Plot\nCAS Score: {cas_score:.4f}"
    )
    ax.set_title(title_str, fontsize=16, y=1.1)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Local Anomaly Density (Glyph Color)", fontsize=10)

    ax.set_ylabel("Anomaly Magnitude (Glyph Radius)", labelpad=25)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.05))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_angle:
        ax.set_xticklabels([])

    if mask_radius:
        ax.set_yticklabels([])

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_anomaly_glyphs.__doc__ = r"""
Visualizes anomaly severity using polar glyphs.

This function creates a highly informative diagnostic plot where
each forecast failure (anomaly) is represented by a glyph
(a custom symbol). The glyph's properties—location, size,
shape, and color—encode the key characteristics of the anomaly,
offering a clear and scientifically rigorous visualization.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted
    quantile values.

actual_col : str
    The name of the column containing the true observed values.

q_low_col : str
    The name of the column for the lower bound of the
    prediction interval.

q_up_col : str
    The name of the column for the upper bound of the
    prediction interval.

window_size : int, default=21
    The size of the moving window used to calculate the local
    anomaly density, which determines the glyph color.

title : str, optional
    A custom title for the plot. If ``None``, a default title
    including the CAS score is generated.

figsize : tuple of (float, float), default=(9.0, 9.0)
    The figure size in inches.

cmap : str, default='inferno'
    The colormap for coloring glyphs based on local density.

s : int, default=70
    The marker size for the glyphs.

alpha : float, default=0.85
    The transparency of the glyphs.

acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'}, default='default'
    Specifies the angular coverage of the polar plot.

mask_angle : bool, default=True
    If ``True``, hides the angular tick labels.

ax : matplotlib.axes.Axes, optional
    An existing polar axes to draw the plot on. If ``None``, a
    new figure and axes are created.

savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if no anomalies are detected in the data.

See Also
--------
plot_anomaly_severity : A version using simple scatter dots.
plot_anomaly_profile : A stylized "fiery ring" version of this plot.
clustered_anomaly_severity_score : The underlying metric function.

Notes
-----
This plot uses a glyph-based approach to encode multiple
dimensions of information for each forecast failure, as proposed
in the framework of :footcite:t:`kouadiob2025`.

**Visual Mapping (Glyph Properties):**

- **Angle (`θ`)**: The sample index, showing *where* in the
  dataset the failure occurred.
- **Radius (`r`)**: The **Anomaly Magnitude**. Glyphs farther from
  the center are more severe failures.
- **Color**: The **Local Anomaly Density**. Hotter colors indicate
  the anomaly is part of a dense cluster.
- **Shape**: The **Type** of anomaly, using an intuitive metaphor:
    
    - `▲` (up-triangle): Over-prediction (risk "escaping" upward).
    - `▼` (down-triangle): Under-prediction (risk "collapsing"
      inward).

References
----------
.. footbibliography::

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.anomaly import plot_anomaly_glyphs
>>>
>>> # Simulate data with a cluster of severe failures
>>> np.random.seed(0)
>>> n_samples = 400
>>> y_true = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, n_samples))
>>> y_qlow = y_true - 10
>>> y_qup = y_true + 10
>>> y_true[100:140] = y_qup[100:140] + np.random.uniform(10, 25, 40)
>>>
>>> df = pd.DataFrame({
...     "actual": y_true, "q10": y_qlow, "q90": y_qup
... })
>>>
>>> ax = plot_anomaly_glyphs(
...     df,
...     actual_col="actual",
...     q_low_col="q10",
...     q_up_col="q90",
...     window_size=31,
...     title="Glyph Plot of Anomaly Hotspot"
... )

"""


@check_non_emptiness
@isdf
def plot_cas_profile(
    df: pd.DataFrame,
    actual_col: str,
    q_low_col: str,
    q_up_col: str,
    *,
    window_size: int = 21,
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 6.0),
    cmap: str = "plasma",
    s: int = 60,
    alpha: float = 0.85,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    required_cols = [actual_col, q_low_col, q_up_col]
    data = df[required_cols].dropna().copy()
    if data.empty:
        warnings.warn("DataFrame is empty after dropping NaNs.", stacklevel=2)
        return None

    cas_score, details = clustered_anomaly_severity(
        data[actual_col].to_numpy(),
        data[q_low_col].to_numpy(),
        data[q_up_col].to_numpy(),
        window_size=window_size,
        return_details=True,
    )

    anomalies = details[details["is_anomaly"]].copy()
    if anomalies.empty:
        warnings.warn("No anomalies detected in the data.", stacklevel=2)
        return None

    # Use a standard Cartesian axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Visual Mappings
    x_coords = anomalies.index.to_numpy()
    y_coords = anomalies["magnitude"].to_numpy()
    density = anomalies["local_density"].to_numpy()

    cmap_obj = get_cmap(cmap, default="plasma")
    norm = Normalize(vmin=density.min(), vmax=density.max())
    colors = cmap_obj(norm(density))

    # Plot Glyphs in two passes for different markers
    under_mask = anomalies["type"] == "under"
    over_mask = anomalies["type"] == "over"

    if np.any(over_mask):
        ax.scatter(
            x_coords[over_mask],
            y_coords[over_mask],
            c=colors[over_mask],
            s=s,
            alpha=alpha,
            marker="^",
            label="Over-prediction (Risk Underestimated)",
            edgecolor="k",
            linewidth=0.5,
        )

    if np.any(under_mask):
        ax.scatter(
            x_coords[under_mask],
            y_coords[under_mask],
            c=colors[under_mask],
            s=s,
            alpha=alpha,
            marker="v",
            label="Under-prediction (Risk Overestimated)",
            edgecolor="k",
            linewidth=0.5,
        )

    # Formatting
    title_str = (
        title or f"Anomaly Severity Profile (CAS Score: {cas_score:.4f})"
    )
    ax.set_title(title_str, fontsize=16)
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Anomaly Magnitude", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.02)  # Add a little padding to x-axis

    # Add a color bar for the cluster density
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Local Anomaly Density", fontsize=10)

    ax.legend(loc="upper right")
    set_axis_grid(ax, show_grid=True)

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _ensure_array_like(x):
    if x is None:
        return None, None

    if isinstance(x, str):
        return x, None

    if isinstance(x, (np.ndarray, pd.Series, list, tuple)):
        arr = np.asarray(x)
        return None, arr

    return None, None


def _prepare_sort_values(df, sort_by):
    # Robustly resolve `sort_by` to a float array of length len(df).
    # Accepts: column name, array-like, or None.  Handles dtype
    # casting (datetime, timedelta, categorical, bool, numeric,
    # object) and returns np.ndarray[float].

    from pandas.api.types import (
        # is_categorical_dtype as _iscat,
        is_bool_dtype as _isbool,
    )
    from pandas.api.types import (
        is_datetime64_any_dtype as _isdt,
    )
    from pandas.api.types import (
        is_numeric_dtype as _isnum,
    )
    from pandas.api.types import (
        is_timedelta64_dtype as _istd,
    )

    key, arr = _ensure_array_like(sort_by)

    def _resolve_key(k):
        # match exact, case-insensitive, or stripped/underscored
        cols = list(df.columns)

        if k in df:
            return k
        low = {str(c).lower(): c for c in cols}
        k_low = str(k).lower()
        if k_low in low:
            return low[k_low]

        def norm(s):
            return str(s).strip().lower().replace(" ", "_")

        norm_map = {norm(c): c for c in cols}
        k_norm = norm(k)
        if k_norm in norm_map:
            return norm_map[k_norm]
        raise KeyError(f"`sort_by` '{k}' not in DataFrame.")

    if key is not None:
        col = _resolve_key(key)
        s = df[col]
    elif arr is not None:
        vals = np.asarray(arr)
        if vals.shape[0] != len(df):
            raise ValueError("`sort_by` length mismatch with DataFrame.")
        s = pd.Series(vals, index=df.index, copy=False)
    else:
        return np.arange(len(df), dtype=float)

    # ---- dtype-specific coercions to float array
    if _isdt(s) or _istd(s):
        # ns since epoch
        vals = s.astype("int64").to_numpy("int64", copy=False)
        return vals.astype(float, copy=False)

    if isinstance(s.dtype, pd.CategoricalDtype):
        codes = s.cat.codes.to_numpy()
        vals = codes.astype(float, copy=False)
        vals[codes == -1] = np.nan
        return vals

    if _isbool(s):
        return s.astype(float, copy=False).to_numpy()

    if _isnum(s):
        return s.astype(float, copy=False).to_numpy()

    # object or mixed: try datetime, then numeric, else factorize
    td = pd.to_datetime(s, errors="coerce")
    if _isdt(td) and getattr(td, "notna", lambda: np.array([]))().any():
        vals = td.view("int64").to_numpy()
        return vals.astype(float, copy=False)

    vals = pd.to_numeric(s, errors="coerce").to_numpy()
    if np.isnan(vals).all():
        codes, _ = pd.factorize(s, sort=True)
        vals = codes.astype(float, copy=False)
        vals[codes == -1] = np.nan
        return vals

    return vals.astype(float, copy=False)


plot_cas_profile.__doc__ = r"""
Visualizes clustered anomaly severity on a Cartesian plot.

This function creates a non-polar "profile" of forecast
failures. It is highly effective for sequential data (like
time series) where the x-axis represents the sample index or
time. It visualizes an anomaly's location, magnitude, type, and
clustering density.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted
    quantile values.

actual_col : str
    The name of the column containing the true observed values.

q_low_col : str
    The name of the column for the lower bound of the
    prediction interval.

q_up_col : str
    The name of the column for the upper bound of the
    prediction interval.

window_size : int, default=21
    The size of the moving window used to calculate the local
    anomaly density, which determines the point color.

title : str, optional
    A custom title for the plot. If ``None``, a default title
    including the CAS score is generated.

figsize : tuple of (float, float), default=(12.0, 6.0)
    The figure size in inches.

cmap : str, default='plasma'
    The colormap for coloring points based on local density.

s : int, default=60
    The marker size for the points.

alpha : float, default=0.85
    The transparency of the points.

ax : matplotlib.axes.Axes, optional
    An existing Cartesian axes to draw the plot on. If ``None``, a
    new figure and axes are created.

savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if no anomalies are detected in the data.

See Also
--------
plot_anomaly_glyphs : A polar version using informative glyphs.
clustered_anomaly_severity_score : The underlying metric function.

Notes
-----
This plot provides a direct, sequential view of forecast
failures, making it easy to spot trends or regime changes in
model performance over time :footcite:p:`kouadiob2025`.

**Visual Mapping:**

- **X-axis**: The sample index, showing *when* or *where* in the
  sequence the failure occurred.
- **Y-axis**: The **Anomaly Magnitude**. The height of a point
  shows its severity.
- **Color**: The **Local Anomaly Density**. Hotter colors show
  "hotspots" where failures are concentrated.
- **Shape**: The **Type** of anomaly.

    - `▲` (up-triangle): Over-prediction (risk underestimated).
    - `▼` (down-triangle): Under-prediction (risk overestimated).

References
----------
.. footbibliography::

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.anomaly import plot_anomaly_profile_cartesian
>>>
>>> # Simulate data with a failure hotspot in the middle
>>> np.random.seed(0)
>>> n_samples = 400
>>> y_true = np.zeros(n_samples)
>>> y_qlow = y_true - 10
>>> y_qup = y_true + 10
>>> y_true[180:220] = y_qup[180:220] + np.random.uniform(5, 15, 40)
>>>
>>> df = pd.DataFrame({
...     "actual": y_true, "q10": y_qlow, "q90": y_qup
... })
>>>
>>> ax = plot_anomaly_profile_cartesian(
...     df,
...     actual_col="actual",
...     q_low_col="q10",
...     q_up_col="q90",
...     title="Cartesian Anomaly Severity Profile"
... )

"""


def plot_glyphs(
    df: pd.DataFrame,
    actual_col: str,
    q_low_col: str,
    q_up_col: str,
    *,
    sort_by=None,
    window_size: int = 21,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 9.0),
    cmap: str = "inferno",
    s: int = 70,
    alpha: float = 0.85,
    acov: Acov = "default",
    mask_angle: bool = True,
    mask_radius: bool = False,
    show_grid: bool = True,
    grid_props: dict | None = None,
    radius: str = "severity",
    color_by: str = "local_density",
    vmin: float | None = None,
    vmax: float | None = None,
    zero_at: Literal["N", "E", "S", "W"] = "N",
    clockwise: bool = True,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs,
) -> Axes | None:
    # Compute CAS details
    _, details = clustered_anomaly_severity(
        actual_col,
        q_low_col,
        q_up_col,
        data=df,
        window_size=window_size,
        return_details=True,
    )

    # Prepare sort axis → theta
    if isinstance(sort_by, str):
        exist_features(df, features=sort_by)
        sort_by = df[sort_by].values

    sort_vals = _prepare_sort_values(
        df[[actual_col, q_low_col, q_up_col]].assign(
            __idx=np.arange(len(df))
        ),
        sort_by,
    )
    span = resolve_polar_span(acov)
    theta = map_theta_to_span(
        sort_vals,
        span=span,
        data_min=np.nanmin(sort_vals),
        data_max=np.nanmax(sort_vals),
    )

    # Choose radial field
    valid_r = ("magnitude", "local_density", "severity")
    if radius not in valid_r:
        raise ValueError(f"`radius` must be in {valid_r}. Got {radius!r}.")
    r = details[radius].to_numpy(dtype=float)

    # Optionally mask non-anomalies
    is_anom = details["is_anomaly"].to_numpy()
    if mask_angle:
        theta_plot = np.where(is_anom, theta, np.nan)
    else:
        theta_plot = theta.copy()

    if mask_radius:
        r_plot = np.where(is_anom, r, 0.0)
    else:
        r_plot = r.copy()

    # Normalize radius to [0, 1] for nicer scaling
    r_max = float(np.nanmax(r_plot)) if np.isfinite(r_plot).any() else 1.0
    if r_max <= 0:
        r_max = 1.0
    r_plot = r_plot / r_max

    # Color mapping
    if color_by not in details.columns:
        raise ValueError(
            f"`color_by` must be a details column. "
            f"Available: {list(details.columns)}"
        )
    cvals = details[color_by].to_numpy(dtype=float)
    if vmin is None:
        vmin = np.nanmin(cvals) if np.isfinite(cvals).any() else 0.0
    if vmax is None:
        vmax = np.nanmax(cvals) if np.isfinite(cvals).any() else 1.0

    # Setup polar axes
    fig, ax, _ = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
        zero_at=zero_at,
        clockwise=clockwise,
    )

    if title:
        ax.set_title(title, fontsize=13, pad=10.0)

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    # Markers by anomaly type
    typ = details["type"].astype(str).to_numpy()
    mk_over = "^"
    mk_under = "v"
    # mk_none = "o"

    # Base scatter for non-anomalies (if not masked)
    if not mask_angle or not mask_radius:
        sel = ~is_anom
        if sel.any():
            ax.scatter(
                theta_plot[sel],
                r_plot[sel],
                s=max(int(s * 0.45), 10),
                c=cvals[sel],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                alpha=max(min(alpha * 0.65, 1.0), 0.1),
                linewidths=0.0,
                **get_valid_kwargs(ax.scatter, kwargs),
            )

    # Over anomalies
    sel_over = is_anom & (typ == "over")
    if sel_over.any():
        ax.scatter(
            theta_plot[sel_over],
            r_plot[sel_over],
            s=s,
            c=cvals[sel_over],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            marker=mk_over,
            alpha=alpha,
            edgecolors="none",
            **get_valid_kwargs(ax.scatter, kwargs),
        )

    # Under anomalies
    sel_under = is_anom & (typ == "under")
    if sel_under.any():
        ax.scatter(
            theta_plot[sel_under],
            r_plot[sel_under],
            s=s,
            c=cvals[sel_under],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            marker=mk_under,
            alpha=alpha,
            edgecolors="none",
            **get_valid_kwargs(ax.scatter, kwargs),
        )

    # Optional thin path of selected radius metric
    if kwargs.get("show_path", True):
        order = np.argsort(sort_vals)
        ax.plot(
            theta[order],
            r_plot[order],
            lw=1.25,
            alpha=0.85,
            color="tab:gray",
        )

    # Colorbar
    mappable = plt.cm.ScalarMappable(
        cmap=get_cmap(cmap, default="inferno"),
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    cbar = plt.colorbar(mappable, ax=ax, pad=0.12, shrink=0.86)
    cbar.set_label(color_by.replace("_", " "))

    # Legend
    handles = [
        Line2D(
            [0],
            [0],
            marker=mk_over,
            linestyle="",
            label="over",
            markersize=max(np.sqrt(s) * 0.8, 4.0),
            color="black",
        ),
        Line2D(
            [0],
            [0],
            marker=mk_under,
            linestyle="",
            label="under",
            markersize=max(np.sqrt(s) * 0.8, 4.0),
            color="black",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.15, 1.15),
        frameon=False,
        title="Anomaly type",
    )

    if mask_angle:
        ax.set_xticklabels([])

    if mask_radius:
        ax.set_yticklabels([])

    if savefig:
        fig.savefig(savefig, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _as_array(x):
    import numpy as np
    import pandas as pd

    if x is None:
        return None
    if isinstance(x, str):
        return None
    if isinstance(x, (np.ndarray, pd.Series, list, tuple)):
        return np.asarray(x)
    return None


def _order_index(df, sort_by):
    import numpy as np

    if sort_by is None:
        return np.arange(len(df))
    if isinstance(sort_by, str):
        vals = df[sort_by].to_numpy()
    else:
        vals = _as_array(sort_by)
        if vals is None:
            raise ValueError("Invalid `sort_by`.")
        if vals.shape[0] != len(df):
            raise ValueError("`sort_by` length mismatch.")
    # datetime friendly
    if hasattr(vals, "dtype") and str(vals.dtype).startswith("datetime"):
        vals = vals.astype("datetime64[ns]").astype("int64")
    idx = np.argsort(vals)
    return idx, vals


plot_glyphs.__doc__ = r"""
Visualizes anomaly characteristics using a polar glyph plot.

This function creates a highly informative diagnostic plot where
each data point is represented by a glyph (a custom symbol) on
a polar axis. The glyph's properties—location, size, shape, and
color—encode multiple dimensions of the data, offering a clear
and scientifically rigorous visualization of forecast failures or
other phenomena.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted
    quantile values.
actual_col : str
    Name of the column containing the true observed values.
q_low_col : str
    Name of the column for the lower bound of the prediction
    interval.
q_up_col : str
    Name of the column for the upper bound of the prediction
    interval.
sort_by : str or array-like, optional
    The feature used to order points around the angular axis.
    Can be a column name or an external array. If ``None``, the
    DataFrame's index is used, which is suitable for time
    series.
window_size : int, default=21
    The size of the moving window used to calculate the local
    anomaly density, which can be used for coloring.
title : str, optional
    A custom title for the plot.
figsize : tuple of (float, float), default=(9.0, 9.0)
    The figure size in inches.
cmap : str, default='inferno'
    The colormap for coloring glyphs.
s : int, default=70
    The base marker size for the glyphs.
alpha : float, default=0.85
    The transparency of the glyphs.
acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'},\
    default='default'
    Specifies the angular coverage of the polar plot.
mask_angle : bool, default=True
    If ``True``, hides the angular tick labels.
mask_radius : bool, default=False
    If ``True``, hides the radial tick labels.
show_grid : bool, default=True
    If ``True``, displays the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments to style the grid.
radius : {'magnitude', 'local_density', 'severity'}, default='magnitude'
    The data field to map to the radial coordinate (distance
    from the center).
color_by : str, default='local_density'
    The data field to map to the glyph color. Must be a column
    in the details DataFrame returned by the CAS metric.
vmin, vmax : float, optional
    The minimum and maximum values for the color normalization.
    If ``None``, they are inferred from the `color_by` data.
zero_at : {'N', 'E', 'S', 'W'}, default='N'
    The direction for the 0° angle on the polar plot.
clockwise : bool, default=True
    If ``True``, angles increase in the clockwise direction.
ax : matplotlib.axes.Axes, optional
    An existing polar axes to draw the plot on.
savefig : str, optional
    The file path to save the plot.

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if no anomalies are detected.

See Also
--------
clustered_anomaly_severity : The underlying metric function.
plot_cas_layers : A Cartesian alternative showing anomaly layers.

Notes
-----
This plot uses a glyph-based approach to encode multiple
dimensions of information for each forecast failure.

**Visual Mapping (Glyph Properties):**

- **Angle** (:math:`\varepsilon`): The position in the sequence, ordered by
  the `sort_by` parameter.
- **Radius (`r`)**: The value of the column specified by `radius`,
  normalized to [0, 1]. A larger radius indicates a higher
  value for that metric.
- **Color**: The value of the column specified by `color_by`.
  By default, hotter colors indicate a denser cluster of
  anomalies.
  
- **Shape**: The **Type** of anomaly, using an intuitive metaphor:
    
    - `▲` (up-triangle): Over-prediction (risk "escaping" upward).
    - `▼` (down-triangle): Under-prediction (risk "collapsing"
      inward).

The plot also includes a thin gray line (`show_path=True`) that
traces the radial metric along the sorted angular path, helping
to visualize trends.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.anomaly import plot_glyphs
>>>
>>> # Simulate data with a failure hotspot
>>> np.random.seed(0)
>>> n_samples = 200
>>> time = pd.to_datetime(pd.date_range(
...     "2024-01-01", periods=n_samples)
... )
>>> y_true = 50 + 10 * np.sin(np.arange(n_samples) * np.pi / 50)
>>> y_qlow = y_true - 5
>>> y_qup = y_true + 5
>>> y_true[80:100] = y_qup[80:100] + np.random.uniform(5, 10, 20)
>>>
>>> df = pd.DataFrame({
...     "time": time, "actual": y_true,
...     "q10": y_qlow, "q90": y_qup
... })
>>>
>>> ax = plot_glyphs(
...     df,
...     actual_col="actual",
...     q_low_col="q10",
...     q_up_col="q90",
...     sort_by="time",
...     radius="magnitude",
...     color_by="local_density",
...     title="Glyph Plot of Anomaly Hotspot"
... )
"""


def plot_cas_layers(
    df: pd.DataFrame,
    actual_col: str,
    q_low_col: str,
    q_up_col: str,
    *,
    sort_by=None,
    window_size: int = 21,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 6.5),
    show_severity: bool = True,
    show_density: bool = True,
    band_alpha: float = 0.18,
    anom_alpha: float = 0.85,
    base_alpha: float = 0.35,
    cmap: str = "inferno",
    mark_size: int = 36,
    show_grid: bool = True,
    grid_props: dict = None,
    lw: float = 1.4,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs,
) -> Axes | tuple[Axes, Axes] | None:
    score, details = clustered_anomaly_severity(
        actual_col,
        q_low_col,
        q_up_col,
        data=df,
        window_size=window_size,
        return_details=True,
    )

    if isinstance(sort_by, str) or _as_array(sort_by) is not None:
        # order_idx, x_vals = _order_index(df, sort_by)
        # det = details.iloc[order_idx].reset_index(drop=True)
        # x = x_vals[order_idx]
        order_idx, x_vals = _order_index(df, sort_by)
        det = details.iloc[order_idx].reset_index(drop=True)

        # use ordinal x so the series spans 0..N-1
        x = np.arange(len(det))

        # keep the sorted categories for ticks later
        cats = np.asarray(x_vals)[order_idx]

    else:
        det = details.copy()
        x = np.arange(len(det))

    y = det["y_true"].to_numpy(dtype=float)
    lo = det["y_qlow"].to_numpy(dtype=float)
    up = det["y_qup"].to_numpy(dtype=float)

    is_anom = det["is_anomaly"].to_numpy()
    typ = det["type"].astype(str).to_numpy()
    sev = det["severity"].to_numpy(dtype=float)
    dens = det["local_density"].to_numpy(dtype=float)
    # mag = det["magnitude"].to_numpy(dtype=float)

    vmin = float(np.nanmin(sev)) if np.isfinite(sev).any() else 0.0
    vmax = float(np.nanmax(sev)) if np.isfinite(sev).any() else 1.0

    if ax is None:
        if show_severity:
            fig, (ax, ax2) = plt.subplots(
                2,
                1,
                figsize=figsize,
                sharex=True,
                gridspec_kw={"height_ratios": [2.6, 1.0]},
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax2 = None
    else:
        fig = ax.figure
        ax2 = None

    if isinstance(sort_by, str):
        s = pd.Series(cats)
        sizes = s.value_counts(sort=False)
        edges = np.r_[0, sizes.cumsum().to_numpy()]
        centers = (edges[:-1] + edges[1:]) / 2
        labels = sizes.index.astype(str).tolist()

        ax.set_xlim(0, len(det))
        ax.set_xticks(centers)
        ax.set_xticklabels(labels)

        for e in edges[1:-1]:
            ax.axvline(e, color="0.7", lw=0.8, ls="--", alpha=0.8)
            if ax2 is not None:
                ax2.axvline(e, color="0.7", lw=0.8, ls="--", alpha=0.8)

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    ax.fill_between(
        x,
        lo,
        up,
        color="0.2",
        alpha=band_alpha,
        linewidth=0.0,
    )

    ax.plot(
        x,
        (lo + up) * 0.5,
        lw=lw * 0.8,
        color="0.35",
        alpha=base_alpha,
    )

    ax.plot(x, y, lw=lw, color="0.05", alpha=0.95)

    cm = get_cmap(cmap, default="inferno")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    sel_over = is_anom & (typ == "over")
    sel_under = is_anom & (typ == "under")

    ax.scatter(
        x[sel_over],
        y[sel_over],
        s=mark_size,
        c=sev[sel_over],
        cmap=cm,
        norm=norm,
        marker="^",
        alpha=anom_alpha,
        edgecolors="none",
        **get_valid_kwargs(ax.scatter, kwargs),
    )
    ax.scatter(
        x[sel_under],
        y[sel_under],
        s=mark_size,
        c=sev[sel_under],
        cmap=cm,
        norm=norm,
        marker="v",
        alpha=anom_alpha,
        edgecolors="none",
        **get_valid_kwargs(ax.scatter, kwargs),
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cm),
        ax=ax,
        pad=0.02,
        fraction=0.04,
    )
    cbar.set_label("severity")

    if title:
        ax.set_title(title, fontsize=13, pad=10.0)

    if show_severity:
        set_axis_grid(ax2, show_grid=show_grid, grid_props=grid_props)

        bar_c = cm(norm(sev))
        ax2.vlines(
            x,
            0.0,
            sev,
            colors=bar_c,
            linewidth=lw,
            alpha=0.95,
        )
        if show_density:
            ax2.plot(
                x,
                dens,
                lw=lw * 1.1,
                color="0.1",
                alpha=0.65,
            )
        ax2.set_ylabel("sev / dens")
        ax2.set_xlabel(kwargs.get("xlabel", ""))

    if savefig:
        fig.savefig(savefig, bbox_inches="tight")

    return (ax, ax2) if show_severity else ax


plot_cas_layers.__doc__ = r"""
Visualizes anomaly severity in layered Cartesian coordinates.

This function creates a highly informative, non-polar diagnostic
plot that visualizes a forecast's prediction interval, the true
values, and the calculated anomaly characteristics in layered
panels. It is particularly effective for sequential data like
time series.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the data.
actual_col : str
    Name of the column containing the true observed values.
q_low_col : str
    Name of the column for the lower bound of the prediction
    interval.
q_up_col : str
    Name of the column for the upper bound of the prediction
    interval.
sort_by : str or array-like, optional
    The feature used to order points along the x-axis. Can be a
    column name (e.g., a datetime column) or an external array.
    If ``None``, the DataFrame's index is used.
window_size : int, default=21
    The size of the moving window used to calculate the local
    anomaly density and severity.
title : str, optional
    A custom title for the plot.
figsize : tuple of (float, float), default=(11.0, 6.5)
    The figure size in inches.
show_severity : bool, default=True
    If ``True``, a second panel is added below the main plot to
    visualize the per-sample severity scores.
show_density : bool, default=True
    If ``True`` and `show_severity` is also ``True``, the local
    density line is overlaid in the bottom panel.
band_alpha : float, default=0.18
    Transparency of the shaded prediction interval band.
anom_alpha : float, default=0.85
    Transparency of the anomaly markers.
cmap : str, default='inferno'
    The colormap for coloring anomaly markers and severity bars
    based on their severity score.
mark_size : int, default=36
    The size of the anomaly markers (`^` and `v`).
show_grid : bool, default=True
    If ``True``, displays grid lines on the axes.
grid_props : dict, optional
    Custom keyword arguments to style the grid.
lw : float, default=1.4
    The base line width for plotted lines.
ax : matplotlib.axes.Axes, optional
    An existing Cartesian axes for the main plot. Note that the
    second panel (`ax2`) will be created new.
savefig : str, optional
    The file path to save the plot.

Returns
-------
ax : matplotlib.axes.Axes
    The Axes object for the main plot.
(ax, ax2) : tuple of (Axes, Axes)
    A tuple of both axes is returned if `show_severity` is True.
None
    If the data is empty after handling NaNs.

See Also
--------
clustered_anomaly_severity : The underlying metric function.
plot_glyphs : A polar alternative for visualizing anomalies.

Notes
-----
This plot decomposes the CAS diagnostic into layers, providing
a clear, sequential view of model performance. 

**Top Panel (Forecast and Anomalies):**

- A shaded gray area shows the prediction interval
  (`q_low_col` to `q_up_col`).
- A dark line shows the true values (`actual_col`).
- Anomalies are marked with colored triangles (`▲` for
  over-predictions, `▼` for under-predictions). The color
  intensity of the marker corresponds to its **severity score**.

**Bottom Panel (Severity Breakdown):**

- Vertical bars show the per-sample **severity score**, colored
  consistently with the markers above.
- An optional black line shows the **local anomaly density**,
  highlighting the "hotspot" regions that contribute to high
  severity scores.

References
----------
.. footbibliography::

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.anomaly import plot_cas_layers
>>>
>>> # Simulate data with a failure hotspot in the middle
>>> np.random.seed(0)
>>> n_samples = 400
>>> x_axis = np.arange(n_samples)
>>> y_true = 20 * np.sin(x_axis * np.pi / 100)
>>> y_qlow = y_true - 10
>>> y_qup = y_true + 10
>>> # Introduce a cluster of severe failures
>>> y_true[180:220] += np.random.uniform(12, 20, 40)
>>>
>>> df = pd.DataFrame({
...     "x": x_axis, "actual": y_true,
...     "q10": y_qlow, "q90": y_qup
... })
>>>
>>> axes = plot_cas_layers(
...     df,
...     actual_col="actual",
...     q_low_col="q10",
...     q_up_col="q90",
...     sort_by="x",
...     title="Layered CAS Diagnostic Profile"
... )

"""
