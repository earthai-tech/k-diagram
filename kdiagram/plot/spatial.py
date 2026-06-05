# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Spatial diagnostic plots for geocoded forecast evaluation.

Functions here visualize prediction metrics, uncertainty intervals,
coverage rates, and multi-model comparisons mapped onto any (x, y)
or (longitude, latitude) coordinate space using pure Matplotlib.
No basemap dependency is required; an optional ``add_basemap`` hook
is accepted but silently ignored unless ``contextily`` is installed.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.interpolate import griddata

from ..compat.matplotlib import get_cmap
from ..decorators import check_non_emptiness, isdf
from ..utils.fs import savefig as safe_savefig
from ..utils.handlers import columns_manager
from ..utils.plot import set_axis_grid
from ..utils.validator import exist_features

__all__ = [
    "plot_spatial_scatter",
    "plot_spatial_heatmap",
    "plot_spatial_uncertainty",
    "plot_spatial_coverage",
    "plot_spatial_comparison",
    "plot_spatial_ordering",
    "plot_polar_from_spatial",
    "plot_paired_spatial_polar",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_fig_ax(
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _finish(
    fig: plt.Figure,
    ax: Axes | list[Axes],
    savefig: str | None,
    dpi: int,
) -> Axes | list[Axes] | None:
    out = safe_savefig(savefig, fig, dpi=dpi, bbox_inches="tight")
    if out is not None:
        plt.close(fig)
    else:
        if not fig.get_constrained_layout():
            fig.tight_layout()
        plt.show()
    return ax


def _colorbar(
    fig: plt.Figure,
    sc,
    ax: Axes,
    label: str | None,
) -> None:
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=10)


def _scale_sizes(
    values: np.ndarray,
    size_range: tuple[float, float],
) -> np.ndarray:
    lo, hi = size_range
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmax > vmin:
        return lo + (values - vmin) / (vmax - vmin) * (hi - lo)
    return np.full_like(values, (lo + hi) / 2.0, dtype=float)


def _try_basemap(ax: Axes, add_basemap: bool) -> None:
    if not add_basemap:
        return
    try:
        import contextily as ctx  # noqa: F401
        ctx.add_basemap(ax, crs="EPSG:4326", zoom="auto")
    except Exception:
        warnings.warn(
            "add_basemap=True requires 'contextily'. "
            "Install it with: pip install contextily",
            ImportWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@check_non_emptiness
@isdf
def plot_spatial_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_col: str,
    *,
    size_col: str | None = None,
    s: float = 80.0,
    size_range: tuple[float, float] = (20.0, 400.0),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.85,
    edgecolor: str = "none",
    linewidths: float = 0.5,
    marker: str = "o",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    annotate: bool = False,
    annotation_col: str | None = None,
    annotation_kwargs: dict[str, Any] | None = None,
    add_basemap: bool = False,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 300,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    r"""Color-coded scatter plot of any numeric metric at spatial locations.

    Each point represents one row in ``df``. The metric value controls the
    point color; an optional second column can control point size (bubble
    chart).  This is the primary function for mapping CAS scores, CRPS,
    interval widths, or any other per-location statistic onto a 2-D space.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Each row is one spatial observation or one aggregated
        location.
    x_col : str
        Column with the x-coordinate (easting, longitude, or any abscissa).
    y_col : str
        Column with the y-coordinate (northing, latitude, or any ordinate).
    metric_col : str
        Numeric column whose values are mapped to point color.
    size_col : str, optional
        Column whose values control point size.  Values are linearly rescaled
        to ``size_range``.  If ``None``, all points use the fixed size ``s``.
    s : float, default=80
        Fixed marker size (in points²) when ``size_col`` is ``None``.
    size_range : (float, float), default=(20, 400)
        (min, max) marker size when ``size_col`` is given.
    title : str, optional
        Plot title.  Defaults to ``"<metric_col> across space"``.
    xlabel, ylabel : str, optional
        Axis labels.  Default to the column names.
    cmap : str, default='viridis'
        Matplotlib colormap for the metric.
    vmin, vmax : float, optional
        Color scale limits.  Inferred from data if not given.
    alpha : float, default=0.85
        Marker transparency.
    edgecolor : str, default='none'
        Marker edge color.
    linewidths : float, default=0.5
        Marker edge line width.
    marker : str, default='o'
        Marker symbol.
    colorbar : bool, default=True
        Draw a colorbar.
    colorbar_label : str, optional
        Colorbar label.  Defaults to ``metric_col``.
    annotate : bool, default=False
        Annotate each point with the value in ``annotation_col`` (or the
        metric value if ``annotation_col`` is ``None``).
    annotation_col : str, optional
        Column to use for annotation text.  Only used when ``annotate=True``.
    annotation_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.annotate``.
    add_basemap : bool, default=False
        Overlay a web tile basemap using ``contextily`` (must be installed).
        Coordinates must be in geographic (lon/lat) or Web Mercator CRS.
    show_grid : bool, default=True
        Show grid lines.
    grid_props : dict, optional
        Grid styling passed to ``set_axis_grid``.
    figsize : (float, float), default=(8, 6)
        Figure size in inches.
    dpi : int, default=300
        Figure resolution when saving.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.  A new figure is created when ``None``.
    savefig : str, optional
        File path for saving the figure.  Displays interactively when ``None``.
    **kwargs
        Extra keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The axes with the plot, or ``None`` if the data is empty after
        dropping NaNs.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from kdiagram.plot.spatial import plot_spatial_scatter
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> df = pd.DataFrame({
    ...     "lon": rng.uniform(113.0, 113.5, n),
    ...     "lat": rng.uniform(22.3, 22.7, n),
    ...     "cas": rng.uniform(0.0, 1.0, n),
    ...     "interval_width": rng.uniform(0.1, 2.0, n),
    ... })
    >>> ax = plot_spatial_scatter(
    ...     df, "lon", "lat", "cas",
    ...     size_col="interval_width",
    ...     cmap="plasma",
    ...     title="CAS per monitoring station",
    ... )
    """
    required = [x_col, y_col, metric_col]
    if size_col:
        required.append(size_col)
    if annotation_col:
        required.append(annotation_col)
    exist_features(df, features=required)

    data = df[required].dropna().copy()
    if data.empty:
        warnings.warn(
            "plot_spatial_scatter: no data remains after dropping NaNs.",
            stacklevel=2,
        )
        return None

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    c = data[metric_col].to_numpy(dtype=float)

    sizes: float | np.ndarray
    if size_col:
        sizes = _scale_sizes(data[size_col].to_numpy(dtype=float), size_range)
    else:
        sizes = s

    fig, ax = _make_fig_ax(ax, figsize)

    sc = ax.scatter(
        x, y, c=c, s=sizes,
        cmap=get_cmap(cmap, default="viridis"),
        vmin=vmin, vmax=vmax,
        alpha=alpha, edgecolors=edgecolor,
        linewidths=linewidths, marker=marker,
        **kwargs,
    )

    if colorbar:
        _colorbar(fig, sc, ax, colorbar_label or metric_col)

    if annotate:
        ann_kw = dict(fontsize=7, ha="center", va="bottom")
        ann_kw.update(annotation_kwargs or {})
        labels = (
            data[annotation_col].astype(str).tolist()
            if annotation_col
            else [f"{v:.2g}" for v in c]
        )
        for xi, yi, lbl in zip(x, y, labels):
            ax.annotate(lbl, (xi, yi), **ann_kw)

    ax.set_title(title or f"{metric_col} across space", fontsize=13)
    ax.set_xlabel(xlabel or x_col, fontsize=11)
    ax.set_ylabel(ylabel or y_col, fontsize=11)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    _try_basemap(ax, add_basemap)

    return _finish(fig, ax, savefig, dpi)


@check_non_emptiness
@isdf
def plot_spatial_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_col: str,
    *,
    method: Literal["linear", "cubic", "nearest"] = "linear",
    resolution: int = 200,
    contour: bool = False,
    contour_levels: int = 8,
    contour_color: str = "white",
    contour_linewidth: float = 0.8,
    contour_kwargs: dict[str, Any] | None = None,
    scatter_overlay: bool = True,
    scatter_s: float = 30.0,
    scatter_color: str = "k",
    scatter_alpha: float = 0.6,
    scatter_kwargs: dict[str, Any] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    add_basemap: bool = False,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 300,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    r"""Interpolated 2-D heatmap of a scalar metric at scattered spatial points.

    Scattered (x, y, metric) data are interpolated onto a regular grid via
    ``scipy.interpolate.griddata`` and displayed as a continuous color surface.
    Points outside the convex hull of the data are masked automatically.
    Original scatter points can optionally be overlaid.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x_col : str
        Column with x-coordinates.
    y_col : str
        Column with y-coordinates.
    metric_col : str
        Column with the metric to interpolate and display.
    method : {'linear', 'cubic', 'nearest'}, default='linear'
        Interpolation method forwarded to ``scipy.interpolate.griddata``.
    resolution : int, default=200
        Number of grid points per axis for the interpolation target.
    contour : bool, default=False
        Overlay iso-lines on the heatmap.
    contour_levels : int, default=8
        Number of contour levels.
    contour_color : str, default='white'
        Contour line color.
    contour_linewidth : float, default=0.8
        Contour line width.
    contour_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.contour``.
    scatter_overlay : bool, default=True
        Draw the original data points on top of the heatmap.
    scatter_s : float, default=30
        Size of the overlay scatter markers.
    scatter_color : str, default='k'
        Color of the overlay scatter markers.
    scatter_alpha : float, default=0.6
        Transparency of the overlay scatter markers.
    scatter_kwargs : dict, optional
        Extra keyword arguments forwarded to the overlay ``ax.scatter`` call.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    cmap : str, default='viridis'
        Colormap.
    vmin, vmax : float, optional
        Color scale limits.
    colorbar : bool, default=True
        Draw a colorbar.
    colorbar_label : str, optional
        Colorbar label.
    add_basemap : bool, default=False
        Overlay a tile basemap via ``contextily``.
    show_grid, grid_props
        Grid visibility and styling.
    figsize : (float, float), default=(8, 6)
        Figure size in inches.
    dpi : int, default=300
        Resolution for saving.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    savefig : str, optional
        Save path.
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes or None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from kdiagram.plot.spatial import plot_spatial_heatmap
    >>> rng = np.random.default_rng(1)
    >>> n = 60
    >>> lon = rng.uniform(113.0, 113.5, n)
    >>> lat = rng.uniform(22.3, 22.7, n)
    >>> cas = np.exp(-((lon - 113.25) ** 2 + (lat - 22.5) ** 2) / 0.02)
    >>> df = pd.DataFrame({"lon": lon, "lat": lat, "cas": cas})
    >>> ax = plot_spatial_heatmap(df, "lon", "lat", "cas", contour=True)
    """
    exist_features(df, features=[x_col, y_col, metric_col])

    data = df[[x_col, y_col, metric_col]].dropna().copy()
    if data.empty:
        warnings.warn(
            "plot_spatial_heatmap: no data remains after dropping NaNs.",
            stacklevel=2,
        )
        return None

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    z = data[metric_col].to_numpy(dtype=float)

    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi_grid, yi_grid), method=method)

    fig, ax = _make_fig_ax(ax, figsize)

    im_kw = dict(
        aspect="auto", origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap=get_cmap(cmap, default="viridis"),
        vmin=vmin, vmax=vmax,
        interpolation="bilinear",
    )
    im_kw.update(kwargs)
    im = ax.imshow(zi, **im_kw)

    if contour:
        c_kw = dict(
            levels=contour_levels,
            colors=contour_color,
            linewidths=contour_linewidth,
        )
        c_kw.update(contour_kwargs or {})
        ax.contour(xi_grid, yi_grid, zi, **c_kw)

    if scatter_overlay:
        sc_kw = dict(
            c=scatter_color, s=scatter_s,
            alpha=scatter_alpha, edgecolors="none",
        )
        sc_kw.update(scatter_kwargs or {})
        ax.scatter(x, y, **sc_kw)

    if colorbar:
        _colorbar(fig, im, ax, colorbar_label or metric_col)

    ax.set_title(title or f"{metric_col} — interpolated surface", fontsize=13)
    ax.set_xlabel(xlabel or x_col, fontsize=11)
    ax.set_ylabel(ylabel or y_col, fontsize=11)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    _try_basemap(ax, add_basemap)

    return _finish(fig, ax, savefig, dpi)


@check_non_emptiness
@isdf
def plot_spatial_uncertainty(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    actual_col: str,
    q_low_col: str,
    q_up_col: str,
    *,
    nominal: float = 0.9,
    size_range: tuple[float, float] = (30.0, 500.0),
    cmap: str = "RdBu_r",
    alpha: float = 0.80,
    marker: str = "o",
    edgecolor: str = "k",
    linewidths: float = 0.4,
    colorbar: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    add_basemap: bool = False,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (9.0, 7.0),
    dpi: int = 300,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    r"""Bubble map of per-location interval width and coverage deviation.

    For every spatial location (unique ``(x_col, y_col)`` pair) the function
    computes:

    * **bubble size** — mean prediction-interval width
      ``mean(q_up - q_low)`` rescaled to ``size_range``.
    * **bubble color** — coverage rate minus ``nominal``
      (e.g. ``nominal=0.9`` means actual 90 % PI coverage).
      Diverging: **blue** = over-covered, **red** = under-covered, **white** = on-target.

    If each row has a unique (x, y) pair the aggregation is a no-op and each
    row is plotted individually.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  May contain multiple rows per spatial location (e.g. one
        row per time step), in which case statistics are averaged per location.
    x_col : str
        Column with x-coordinates.
    y_col : str
        Column with y-coordinates.
    actual_col : str
        Column with observed values.
    q_low_col : str
        Column with the lower quantile bound.
    q_up_col : str
        Column with the upper quantile bound.
    nominal : float, default=0.9
        Nominal coverage level (e.g. 0.9 for a 90 % PI).  Used as the
        diverging center of the colormap.
    size_range : (float, float), default=(30, 500)
        (min, max) bubble area in points² across all locations.
    cmap : str, default='RdBu_r'
        Diverging colormap.  RdBu_r maps negative deviations (under-coverage)
        to red and positive deviations (over-coverage) to blue.
    alpha : float, default=0.80
        Marker transparency.
    marker : str, default='o'
        Marker symbol.
    edgecolor : str, default='k'
        Marker edge color.
    linewidths : float, default=0.4
        Marker edge line width.
    colorbar : bool, default=True
        Draw a colorbar annotated with the deviation from ``nominal``.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    legend : bool, default=True
        Add a size legend showing example bubble diameters.
    add_basemap : bool, default=False
        Overlay a tile basemap via ``contextily``.
    show_grid, grid_props
        Grid visibility and styling.
    figsize : (float, float), default=(9, 7)
        Figure size in inches.
    dpi : int, default=300
        Resolution for saving.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    savefig : str, optional
        Save path.
    **kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    ax : matplotlib.axes.Axes or None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from kdiagram.plot.spatial import plot_spatial_uncertainty
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> stations = np.repeat(np.arange(20), 10)
    >>> df = pd.DataFrame({
    ...     "lon": np.repeat(rng.uniform(113.0, 113.5, 20), 10),
    ...     "lat": np.repeat(rng.uniform(22.3, 22.7, 20), 10),
    ...     "y":   rng.normal(0, 1, n),
    ...     "q10": rng.normal(-1.5, 0.3, n),
    ...     "q90": rng.normal( 1.5, 0.3, n),
    ... })
    >>> ax = plot_spatial_uncertainty(
    ...     df, "lon", "lat", "y", "q10", "q90", nominal=0.9,
    ...     title="Interval width and coverage deviation per station",
    ... )
    """
    required = [x_col, y_col, actual_col, q_low_col, q_up_col]
    exist_features(df, features=required)

    data = df[required].dropna().copy()
    if data.empty:
        warnings.warn(
            "plot_spatial_uncertainty: no data after dropping NaNs.",
            stacklevel=2,
        )
        return None

    data["_covered"] = (
        (data[actual_col] >= data[q_low_col]) &
        (data[actual_col] <= data[q_up_col])
    ).astype(float)
    data["_width"] = (data[q_up_col] - data[q_low_col]).clip(lower=0.0)

    agg = (
        data.groupby([x_col, y_col], sort=False)
        .agg(
            _mean_width=("_width", "mean"),
            _coverage=("_covered", "mean"),
        )
        .reset_index()
    )

    x = agg[x_col].to_numpy(dtype=float)
    y = agg[y_col].to_numpy(dtype=float)
    widths = agg["_mean_width"].to_numpy(dtype=float)
    coverage = agg["_coverage"].to_numpy(dtype=float)
    deviation = coverage - nominal  # centred at 0

    sizes = _scale_sizes(widths, size_range)

    abs_max = float(np.nanmax(np.abs(deviation)))
    if abs_max < 1e-9:
        abs_max = 0.1
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    fig, ax = _make_fig_ax(ax, figsize)

    sc = ax.scatter(
        x, y, c=deviation, s=sizes,
        cmap=get_cmap(cmap, default="RdBu_r"),
        norm=norm,
        alpha=alpha, edgecolors=edgecolor,
        linewidths=linewidths, marker=marker,
        **kwargs,
    )

    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(f"Coverage − {nominal:.0%}", fontsize=10)

    if legend:
        # Size legend: show 3 representative bubble diameters
        lo, hi = size_range
        size_ticks = [lo, (lo + hi) / 2.0, hi]
        w_lo = float(np.nanmin(widths))
        w_hi = float(np.nanmax(widths))
        w_mid = (w_lo + w_hi) / 2.0
        for st, wt in zip(size_ticks, [w_lo, w_mid, w_hi]):
            ax.scatter(
                [], [], s=st, c="grey", alpha=0.6, edgecolors="k",
                linewidths=0.4,
                label=f"width ≈ {wt:.2g}",
            )
        ax.legend(
            title="Interval width", fontsize=8, title_fontsize=9,
            loc="lower right",
        )

    ax.set_title(
        title or "Interval width and coverage deviation",
        fontsize=13,
    )
    ax.set_xlabel(xlabel or x_col, fontsize=11)
    ax.set_ylabel(ylabel or y_col, fontsize=11)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    _try_basemap(ax, add_basemap)

    return _finish(fig, ax, savefig, dpi)


@check_non_emptiness
@isdf
def plot_spatial_coverage(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    coverage_col: str,
    *,
    nominal: float = 0.9,
    tol: float | None = None,
    cmap: str = "RdBu",
    s: float = 90.0,
    alpha: float = 0.85,
    marker: str = "o",
    edgecolor: str = "k",
    linewidths: float = 0.4,
    colorbar: bool = True,
    annotate: bool = False,
    fmt: str = ".2f",
    annotation_kwargs: dict[str, Any] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    add_basemap: bool = False,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 300,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    r"""Spatial map of pre-computed coverage rates, centered on the nominal level.

    Uses a diverging colormap so that locations matching the nominal coverage
    appear neutral, under-covered locations appear in one end of the scale,
    and over-covered locations appear in the other.  Optionally marks stations
    that exceed a tolerance threshold with a distinct annotation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  One row per spatial location with a pre-aggregated
        coverage rate in ``coverage_col``.
    x_col : str
        Column with x-coordinates.
    y_col : str
        Column with y-coordinates.
    coverage_col : str
        Column with empirical coverage rates (values in [0, 1]).
    nominal : float, default=0.9
        Nominal (target) coverage level.  This becomes the center of the
        diverging colormap.
    tol : float, optional
        If given, locations where ``|coverage − nominal| > tol`` are
        annotated with a star ``★`` to flag problematic stations.
    cmap : str, default='RdBu'
        Diverging colormap (RdBu: red = low coverage, blue = high coverage).
    s : float, default=90
        Marker size in points².
    alpha : float, default=0.85
        Marker transparency.
    marker : str, default='o'
        Marker symbol.
    edgecolor : str, default='k'
        Marker edge color.
    linewidths : float, default=0.4
        Marker edge line width.
    colorbar : bool, default=True
        Draw a colorbar.
    annotate : bool, default=False
        Annotate each point with its coverage value (formatted with ``fmt``).
    fmt : str, default='.2f'
        Format string for annotation values.
    annotation_kwargs : dict, optional
        Extra keyword arguments for ``ax.annotate``.
    add_basemap : bool, default=False
        Overlay a tile basemap via ``contextily``.
    show_grid, grid_props
        Grid visibility and styling.
    figsize : (float, float), default=(8, 6)
        Figure size in inches.
    dpi : int, default=300
        Resolution for saving.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    savefig : str, optional
        Save path.
    **kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    ax : matplotlib.axes.Axes or None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from kdiagram.plot.spatial import plot_spatial_coverage
    >>> rng = np.random.default_rng(7)
    >>> n = 30
    >>> df = pd.DataFrame({
    ...     "lon": rng.uniform(113.0, 113.5, n),
    ...     "lat": rng.uniform(22.3, 22.7, n),
    ...     "coverage": rng.beta(9, 1, n),   # mostly near 0.9
    ... })
    >>> ax = plot_spatial_coverage(
    ...     df, "lon", "lat", "coverage",
    ...     nominal=0.9, tol=0.1,
    ...     title="Coverage deviation from 90% PI",
    ... )
    """
    exist_features(df, features=[x_col, y_col, coverage_col])

    data = df[[x_col, y_col, coverage_col]].dropna().copy()
    if data.empty:
        warnings.warn(
            "plot_spatial_coverage: no data after dropping NaNs.",
            stacklevel=2,
        )
        return None

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    cov = data[coverage_col].to_numpy(dtype=float)
    deviation = cov - nominal

    abs_max = float(np.nanmax(np.abs(deviation)))
    if abs_max < 1e-9:
        abs_max = 0.05
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    fig, ax = _make_fig_ax(ax, figsize)

    sc = ax.scatter(
        x, y, c=deviation, s=s,
        cmap=get_cmap(cmap, default="RdBu"),
        norm=norm,
        alpha=alpha, edgecolors=edgecolor,
        linewidths=linewidths, marker=marker,
        **kwargs,
    )

    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(f"Coverage − {nominal:.0%}", fontsize=10)

    ann_kw = dict(fontsize=7, ha="center", va="bottom", color="0.3")
    ann_kw.update(annotation_kwargs or {})

    for xi, yi, cv, dv in zip(x, y, cov, deviation):
        if annotate:
            ax.annotate(f"{cv:{fmt}}", (xi, yi), **ann_kw)
        if tol is not None and abs(dv) > tol:
            ax.annotate(
                "★", (xi, yi),
                fontsize=10, ha="center", va="top",
                color="crimson" if dv < 0 else "steelblue",
            )

    ax.set_title(
        title if title is not None else f"Coverage deviation from {nominal:.0%}",
        fontsize=13,
    )
    ax.set_xlabel(xlabel or x_col, fontsize=11)
    ax.set_ylabel(ylabel or y_col, fontsize=11)

    # set_axis_grid call
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
    _try_basemap(ax, add_basemap)

    return _finish(fig, ax, savefig, dpi)


@check_non_emptiness
@isdf
def plot_spatial_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_cols: list[str] | str,
    *,
    names: list[str] | None = None,
    ncols: int = 2,
    shared_scale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    s: float = 60.0,
    cmap: str = "viridis",
    alpha: float = 0.85,
    marker: str = "o",
    edgecolor: str = "none",
    linewidths: float = 0.5,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    add_basemap: bool = False,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    savefig: str | None = None,
    **kwargs: Any,
) -> list[Axes] | None:
    r"""Multi-panel spatial scatter comparison of the same metric across models.

    Produces an N-panel grid where every panel shows the same spatial
    scatter for a different column in ``metric_cols`` (e.g. one column per
    model).  When ``shared_scale=True`` all panels use the same colormap
    range, making cross-panel comparisons direct.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  One row per spatial location.
    x_col : str
        Column with x-coordinates.
    y_col : str
        Column with y-coordinates.
    metric_cols : list of str or str
        Columns to compare, one per panel.  Accepts a comma-separated string.
    names : list of str, optional
        Panel subtitles.  Defaults to the column names.
    ncols : int, default=2
        Number of columns in the panel grid.
    shared_scale : bool, default=True
        Use a single shared colormap range across all panels.
    vmin, vmax : float, optional
        Override the shared scale limits.  Ignored when ``shared_scale=False``.
    s : float, default=60
        Marker size in points².
    cmap : str, default='viridis'
        Colormap applied to all panels.
    alpha : float, default=0.85
        Marker transparency.
    marker : str, default='o'
        Marker symbol.
    edgecolor : str, default='none'
        Marker edge color.
    linewidths : float, default=0.5
        Marker edge line width.
    colorbar : bool, default=True
        Draw a single shared colorbar on the right of the figure
        (``shared_scale=True``) or one per panel (``shared_scale=False``).
    colorbar_label : str, optional
        Colorbar label.
    add_basemap : bool, default=False
        Overlay a tile basemap on every panel via ``contextily``.
    show_grid, grid_props
        Grid visibility and styling.
    figsize : (float, float), optional
        Figure size.  Auto-computed from ``ncols`` and the number of rows
        when ``None``.
    dpi : int, default=300
        Resolution for saving.
    savefig : str, optional
        Save path.
    **kwargs
        Forwarded to every ``ax.scatter`` call.

    Returns
    -------
    axes : list of matplotlib.axes.Axes or None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from kdiagram.plot.spatial import plot_spatial_comparison
    >>> rng = np.random.default_rng(3)
    >>> n = 40
    >>> df = pd.DataFrame({
    ...     "lon":  rng.uniform(113.0, 113.5, n),
    ...     "lat":  rng.uniform(22.3, 22.7, n),
    ...     "cas_qar":  rng.uniform(0.1, 0.9, n),
    ...     "cas_qgbm": rng.uniform(0.05, 0.7, n),
    ...     "cas_xtft": rng.uniform(0.02, 0.5, n),
    ... })
    >>> axes = plot_spatial_comparison(
    ...     df, "lon", "lat",
    ...     ["cas_qar", "cas_qgbm", "cas_xtft"],
    ...     names=["QAR", "QGBM", "XTFT"],
    ...     cmap="plasma",
    ...     title_prefix="CAS —",   # passed via **kwargs; ignored silently
    ... )
    """
    metric_cols = columns_manager(metric_cols, empty_as_none=False) or []
    if not metric_cols:
        raise ValueError("`metric_cols` must name at least one column.")

    exist_features(df, features=[x_col, y_col] + list(metric_cols))

    required = [x_col, y_col] + list(metric_cols)
    data = df[required].dropna().copy()
    if data.empty:
        warnings.warn(
            "plot_spatial_comparison: no data after dropping NaNs.",
            stacklevel=2,
        )
        return None

    n_panels = len(metric_cols)
    if names is None:
        names = list(metric_cols)
    elif len(names) != n_panels:
        warnings.warn(
            "Length of `names` does not match `metric_cols`. Using column names.",
            stacklevel=2,
        )
        names = list(metric_cols)

    nrows = math.ceil(n_panels / ncols)
    actual_ncols = min(ncols, n_panels)

    if figsize is None:
        figsize = (5.5 * actual_ncols, 4.5 * nrows)

    fig, axes_arr = plt.subplots(
        nrows, actual_ncols,
        figsize=figsize,
        squeeze=False,
        layout="constrained",
    )

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    cmap_obj = get_cmap(cmap, default="viridis")

    # Compute shared scale
    if shared_scale:
        all_vals = np.concatenate(
            [data[col].to_numpy(dtype=float) for col in metric_cols]
        )
        _vmin = vmin if vmin is not None else float(np.nanmin(all_vals))
        _vmax = vmax if vmax is not None else float(np.nanmax(all_vals))
    else:
        _vmin = _vmax = None

    scatter_handles = []
    axes_flat = axes_arr.flat

    for idx, (col, name) in enumerate(zip(metric_cols, names)):
        ax = next(axes_flat)
        c = data[col].to_numpy(dtype=float)

        panel_vmin = _vmin if shared_scale else (vmin or float(np.nanmin(c)))
        panel_vmax = _vmax if shared_scale else (vmax or float(np.nanmax(c)))

        sc = ax.scatter(
            x, y, c=c, s=s,
            cmap=cmap_obj,
            vmin=panel_vmin, vmax=panel_vmax,
            alpha=alpha, edgecolors=edgecolor,
            linewidths=linewidths, marker=marker,
            **kwargs,
        )
        scatter_handles.append(sc)

        if colorbar and not shared_scale:
            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            if colorbar_label:
                cb.set_label(colorbar_label, fontsize=9)

        ax.set_title(name, fontsize=12)
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.tick_params(labelsize=8)
        set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
        _try_basemap(ax, add_basemap)

    # Hide unused axes
    for ax in axes_flat:
        ax.set_visible(False)

    # Shared colorbar
    if colorbar and shared_scale and scatter_handles:
        cbar = fig.colorbar(
            scatter_handles[0],
            ax=axes_arr[:, -1].tolist(),
            shrink=0.8, pad=0.02,
        )
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=10)

    axes_list = [axes_arr[r, c] for r in range(nrows) for c in range(actual_ncols)
                 if axes_arr[r, c].get_visible()]

    return _finish(fig, axes_list, savefig, dpi)


# ---------------------------------------------------------------------------
# Polar-from-spatial helpers
# ---------------------------------------------------------------------------

def _compute_site_order(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    order_by: str,
    order_ascending: bool,
    order_col: str | None,
) -> pd.DataFrame:
    """Return a copy of *df* sorted by the chosen criterion.

    A new integer column ``'_site_order'`` (0-based rank) is appended.
    Supported values for *order_by*: ``'lat'`` / ``'latitude'``,
    ``'lon'`` / ``'longitude'``, or any column name present in *df*.
    """
    df = df.copy()
    if order_col is not None:
        exist_features(df, features=[order_col])
        df = df.sort_values(order_col, ascending=order_ascending)
    elif order_by in ("lat", "latitude"):
        df = df.sort_values(y_col, ascending=order_ascending)
    elif order_by in ("lon", "longitude"):
        df = df.sort_values(x_col, ascending=order_ascending)
    elif order_by in df.columns:
        df = df.sort_values(order_by, ascending=order_ascending)
    # else: keep current row order
    df = df.reset_index(drop=True)
    df["_site_order"] = np.arange(len(df))
    return df


def _polar_spikes(
    ax,
    thetas: np.ndarray,
    radii: np.ndarray,
    color_vals: np.ndarray,
    cmap_obj,
    norm,
    lw: float,
    alpha: float,
    r_base: float = 0.0,
) -> None:
    """Draw colored radial spikes on a polar Axes (vectorised per color bin)."""
    # Group by quantised color to batch ax.plot calls (10x faster than per-spike loop)
    N = len(thetas)
    n_bins = min(64, N)
    bin_idx = np.floor(
        np.clip((color_vals - norm.vmin) / max(norm.vmax - norm.vmin, 1e-12), 0, 1)
        * (n_bins - 1)
    ).astype(int)

    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        color = cmap_obj(norm(float(np.mean(color_vals[mask]))))
        t = np.empty(3 * mask.sum()); r = np.empty_like(t)
        t[0::3] = thetas[mask]; t[1::3] = thetas[mask]; t[2::3] = np.nan
        r[0::3] = r_base;        r[1::3] = radii[mask];   r[2::3] = np.nan
        ax.plot(t, r, "-", color=color, lw=lw, alpha=alpha, solid_capstyle="butt")


def _polar_reference_rings(
    ax,
    r_max: float,
    n_labels: int,
    label_angle: float,
    label_fmt: str = ".2g",
    color: str = "0.4",
    lw: float = 0.4,
) -> None:
    """Draw concentric reference circles with radial labels."""
    t_circle = np.linspace(0, 2 * np.pi, 360)
    for rv in np.linspace(0, r_max, n_labels + 1)[1:]:
        ax.plot(t_circle, np.full(360, rv), "-", color=color, lw=lw, alpha=0.45)
        ax.text(label_angle, rv, f"{rv:{label_fmt}}",
                ha="left", va="center", fontsize=7.5, color=color)


def _setup_polar_ax(ax, zero_loc: str = "N", clockwise: bool = True) -> None:
    """Configure a polar Axes for the spatial-polar diagnostic style."""
    ax.set_theta_zero_location(zero_loc)
    ax.set_theta_direction(-1 if clockwise else 1)
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("none")


# ---------------------------------------------------------------------------
# Public API — polar-from-spatial functions
# ---------------------------------------------------------------------------

@check_non_emptiness
@isdf
def plot_spatial_ordering(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    order_col: str | None = None,
    order_by: Literal["lat", "lon", "default"] = "lat",
    order_ascending: bool = True,
    label_sites: list[int] | None = None,
    show_arrows: bool = True,
    arrow_step: int | None = None,
    arrow_kwargs: dict[str, Any] | None = None,
    cmap: str = "viridis",
    s: float = 25.0,
    alpha: float = 0.85,
    edgecolor: str = "none",
    colorbar: bool = True,
    colorbar_label: str = "Site order",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (7.0, 6.0),
    dpi: int = 300,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    r"""Geographic scatter map showing the spatial ordering of sites.

    Each site is color-coded by its position in the ordering sequence
    (0 = first, N-1 = last).  Optional arrows connect consecutive sites
    to make the traversal path legible.  This plot is the companion to
    :func:`plot_polar_from_spatial`: it shows *why* sites appear at
    certain polar angles in the hedgehog diagnostic.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x_col : str
        Column with x-coordinates (longitude or easting).
    y_col : str
        Column with y-coordinates (latitude or northing).
    order_col : str, optional
        Pre-computed ordering column.  If given, sites are sorted by
        this column instead of by ``order_by``.
    order_by : {'lat', 'lon', 'default'}, default='lat'
        Spatial criterion for ordering when ``order_col`` is ``None``.
        ``'lat'`` sorts by latitude, ``'lon'`` by longitude.  Any column
        name present in ``df`` is also accepted.
    order_ascending : bool, default=True
        Sort direction.
    label_sites : list of int, optional
        0-based indices (in sorted order) to annotate with their 1-based
        site number.  Defaults to first, last, and a few evenly-spaced
        intermediates.
    show_arrows : bool, default=True
        Draw direction arrows between consecutive sites to visualise the
        traversal path.
    arrow_step : int, optional
        Draw one arrow every ``arrow_step`` consecutive sites.
        Defaults to ``max(1, N // 15)``.
    arrow_kwargs : dict, optional
        Extra keyword arguments for the ``arrowprops`` dict.
    cmap : str, default='viridis'
        Colormap for the site-order color scale.
    s : float, default=25
        Marker size in points².
    alpha : float, default=0.85
        Marker transparency.
    edgecolor : str, default='none'
        Marker edge color.
    colorbar : bool, default=True
        Draw a colorbar for the order index.
    colorbar_label : str, default='Site order'
        Colorbar label.
    title, xlabel, ylabel : str, optional
        Plot annotations.
    show_grid, grid_props
        Grid visibility and styling.
    figsize : (float, float), default=(7, 6)
        Figure size in inches.
    dpi : int, default=300
        Resolution for saving.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    savefig : str, optional
        Save path.
    **kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    ax : matplotlib.axes.Axes or None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from kdiagram.plot.spatial import plot_spatial_ordering
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> df = pd.DataFrame({
    ...     "lon": rng.uniform(113.1, 113.6, n),
    ...     "lat": rng.uniform(22.3, 22.8, n),
    ... })
    >>> ax = plot_spatial_ordering(df, "lon", "lat", order_by="lat",
    ...                            label_sites=[0, 29, 59])
    """
    _ord_cols = [x_col, y_col]
    if order_col is not None:
        _ord_cols.append(order_col)
    exist_features(df, features=_ord_cols)
    data = df[_ord_cols].dropna().copy()
    if data.empty:
        warnings.warn("plot_spatial_ordering: no data after dropping NaNs.",
                      stacklevel=2)
        return None

    ordered = _compute_site_order(data, x_col, y_col, order_by, order_ascending, order_col)
    N = len(ordered)
    order_vals = ordered["_site_order"].to_numpy(dtype=float)

    fig, ax = _make_fig_ax(ax, figsize)

    sc = ax.scatter(
        ordered[x_col], ordered[y_col],
        c=order_vals, s=s,
        cmap=get_cmap(cmap, default="viridis"),
        vmin=0, vmax=N - 1,
        alpha=alpha, edgecolors=edgecolor,
        **kwargs,
    )

    if colorbar:
        _colorbar(fig, sc, ax, colorbar_label)

    # Arrows
    if show_arrows:
        step = arrow_step if arrow_step is not None else max(1, N // 15)
        akw = dict(color="0.55", lw=0.6, mutation_scale=8)
        akw.update(arrow_kwargs or {})
        for i in range(0, N - 1, step):
            xi = float(ordered[x_col].iloc[i])
            yi = float(ordered[y_col].iloc[i])
            xj = float(ordered[x_col].iloc[i + 1])
            yj = float(ordered[y_col].iloc[i + 1])
            ax.annotate("", xy=(xj, yj), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="->", **akw))

    # Site labels
    if label_sites is None:
        n_lbl = min(6, N)
        label_sites = [int(v) for v in np.linspace(0, N - 1, n_lbl)]
    for idx in sorted(set(label_sites)):
        if 0 <= idx < N:
            ax.annotate(
                str(idx + 1),
                (float(ordered[x_col].iloc[idx]),
                 float(ordered[y_col].iloc[idx])),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8, fontweight="bold", color="0.2",
            )

    ax.set_title(title or f"Geographic domain and site ordering ({order_by})", fontsize=12)
    ax.set_xlabel(xlabel or x_col, fontsize=10)
    ax.set_ylabel(ylabel or y_col, fontsize=10)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    return _finish(fig, ax, savefig, dpi)


@check_non_emptiness
@isdf
def plot_polar_from_spatial(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_col: str,
    *,
    order_col: str | None = None,
    order_by: Literal["lat", "lon", "default"] = "lat",
    order_ascending: bool = True,
    horizon_cols: list[str] | str | None = None,
    horizon_labels: list[str] | None = None,
    horizon_colors: list[Any] | None = None,
    color_col: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    lw: float = 0.7,
    alpha: float = 0.75,
    n_ring_labels: int = 3,
    ring_label_angle: float = 0.25,
    label_n_sites: int = 4,
    zero_loc: str = "N",
    clockwise: bool = True,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 7.0),
    dpi: int = 300,
    ax: Axes | None = None,
    savefig: str | None = None,
    **kwargs: Any,
) -> Axes | None:
    r"""Polar diagnostic where **angle = spatially-ordered site index** and
    **radius = metric value**.

    This is the central visualization of the spatial-polar paradigm
    introduced in the k-diagram paper.  Sites are first ordered
    geographically (by latitude, longitude, or a custom key), their
    rank is mapped linearly to polar angle
    :math:`\theta_i = 2\pi\,i/N`, and the metric value is mapped to
    the radial coordinate.  Each site appears as a thin radial *spike*
    (needle) pointing outward from the center.

    The result is a *hedgehog diagram* that encodes the spatial
    distribution of a metric in a compact, rotation-invariant view.
    The companion geographic map (:func:`plot_spatial_scatter` or
    :func:`plot_spatial_ordering`) provides the spatial key to read
    which angle corresponds to which location.

    When ``horizon_cols`` is given the function switches to **ring
    mode**: one concentric ring per horizon/condition, allowing the
    evolution of the metric across forecast horizons (or any discrete
    grouping) to be read at every site simultaneously.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x_col : str
        Column with x-coordinates.
    y_col : str
        Column with y-coordinates.
    metric_col : str
        Column whose values become the radial coordinate (spike length).
        Used as the primary metric when ``horizon_cols`` is ``None``.
    order_col : str, optional
        Pre-computed ordering column (overrides ``order_by``).
    order_by : {'lat', 'lon', 'default'}, default='lat'
        Spatial ordering criterion.
    order_ascending : bool, default=True
        Sort direction.
    horizon_cols : list of str or str, optional
        Columns for additional horizons/conditions rendered as
        concentric rings.  The inner-most ring corresponds to the first
        element; subsequent elements add outer rings.  When given,
        ``metric_col`` is used as the *first* ring (horizon H1).
    horizon_labels : list of str, optional
        Labels for each ring (e.g. ``['H1', 'H3', 'H7']``).
        Defaults to ``['H1', 'H2', ...]``.
    horizon_colors : list, optional
        One color per ring.  Defaults to evenly-spaced samples from
        ``cmap``.
    color_col : str, optional
        Column whose values drive spike color (single-horizon mode
        only).  Defaults to ``metric_col``.
    cmap : str, default='viridis'
        Colormap.
    vmin, vmax : float, optional
        Color scale limits (single-horizon mode).
    lw : float, default=0.7
        Spike line width.
    alpha : float, default=0.75
        Spike transparency.
    n_ring_labels : int, default=3
        Number of concentric reference circle labels in single-horizon
        mode.
    ring_label_angle : float, default=0.25
        Angle (radians) at which to place the radial reference labels.
    label_n_sites : int, default=4
        Number of angular site-index labels to display.  Set to ``0``
        to suppress.
    zero_loc : str, default='N'
        Where the first site (angle = 0) appears on the circle:
        ``'N'`` = top (north), ``'E'`` = right, etc.
    clockwise : bool, default=True
        If ``True``, site ordering increases clockwise (matches the
        paper convention).
    colorbar : bool, default=True
        Draw a colorbar (single-horizon mode only).
    colorbar_label : str, optional
        Colorbar label.
    title : str, optional
        Plot title.
    figsize : (float, float), default=(7, 7)
        Figure size in inches.
    dpi : int, default=300
        Resolution for saving.
    ax : matplotlib.axes.Axes, optional
        An existing **polar** Axes to draw on.
    savefig : str, optional
        Save path.
    **kwargs
        Reserved for future use.

    Returns
    -------
    ax : matplotlib.axes.Axes or None

    Notes
    -----
    **Angle mapping**: :math:`\theta_i = 2\pi\,i/N` where :math:`i` is
    the 0-based rank after ordering.  Sites are therefore distributed
    uniformly around the full circle.

    **Ring mode normalisation**: within each ring :math:`k`, spike
    heights are normalised to the *global* maximum across all rings so
    that inter-ring comparisons are valid.

    Examples
    --------
    Single-horizon hedgehog (color = interval width):

    >>> import numpy as np
    >>> import pandas as pd
    >>> from kdiagram.plot.spatial import plot_polar_from_spatial
    >>> rng = np.random.default_rng(1)
    >>> n = 80
    >>> df = pd.DataFrame({
    ...     "lon": rng.uniform(113.1, 113.6, n),
    ...     "lat": rng.uniform(22.3, 22.8, n),
    ...     "width": rng.uniform(0.5, 3.0, n),
    ... })
    >>> ax = plot_polar_from_spatial(df, "lon", "lat", "width",
    ...                              order_by="lat", cmap="plasma")

    Multi-horizon ring encoding (H1, H3, H7):

    >>> df["width_h3"] = df["width"] * 1.4
    >>> df["width_h7"] = df["width"] * 2.1
    >>> ax = plot_polar_from_spatial(
    ...     df, "lon", "lat", "width",
    ...     horizon_cols=["width_h3", "width_h7"],
    ...     horizon_labels=["H1", "H3", "H7"],
    ... )
    """
    # ---- validate ----
    hcols = columns_manager(horizon_cols, empty_as_none=False) or []
    required = [x_col, y_col, metric_col] + hcols
    if color_col:
        required.append(color_col)
    if order_col is not None:
        required.append(order_col)
    exist_features(df, features=required)

    data = df[list(dict.fromkeys(required))].dropna().copy()
    if data.empty:
        warnings.warn("plot_polar_from_spatial: no data after dropping NaNs.",
                      stacklevel=2)
        return None

    ordered = _compute_site_order(data, x_col, y_col, order_by, order_ascending, order_col)
    N = len(ordered)
    thetas = 2 * np.pi * np.arange(N) / N

    # ---- polar axes ----
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure
    _setup_polar_ax(ax, zero_loc=zero_loc, clockwise=clockwise)

    cmap_obj = get_cmap(cmap, default="viridis")

    # ============================  single-horizon  ===========================
    if not hcols:
        radii = ordered[metric_col].to_numpy(dtype=float)
        color_vals = (
            ordered[color_col].to_numpy(dtype=float) if color_col else radii
        )
        _vmin = vmin if vmin is not None else float(np.nanmin(color_vals))
        _vmax = vmax if vmax is not None else float(np.nanmax(color_vals))
        norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

        _polar_spikes(ax, thetas, radii, color_vals, cmap_obj, norm, lw, alpha)

        r_max = float(np.nanmax(radii))
        ax.set_ylim(0, r_max * 1.08)
        _polar_reference_rings(ax, r_max, n_ring_labels, ring_label_angle)

        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.10, shrink=0.72, aspect=20)
            cbar.set_label(colorbar_label or metric_col, fontsize=9)

    # ================================  ring mode  ============================
    else:
        all_cols = [metric_col] + hcols
        K = len(all_cols)

        if horizon_labels is None:
            horizon_labels = [f"H{k + 1}" for k in range(K)]
        if horizon_colors is None:
            horizon_colors = [cmap_obj(k / max(1, K - 1)) for k in range(K)]

        all_vals = np.concatenate(
            [ordered[c].to_numpy(dtype=float) for c in all_cols]
        )
        global_max = float(np.nanmax(all_vals)) or 1.0
        ring_step = global_max / K

        t_circle = np.linspace(0, 2 * np.pi, 360)

        for k, col in enumerate(all_cols):
            r_base = k * ring_step
            metric_k = ordered[col].to_numpy(dtype=float)
            r_top = r_base + (metric_k / global_max) * ring_step

            # batch-color within ring by value
            norm_k = mcolors.Normalize(
                vmin=float(np.nanmin(metric_k)),
                vmax=float(np.nanmax(metric_k)),
            )
            hcolor = horizon_colors[k]
            t = np.empty(3 * N); r = np.empty(3 * N)
            t[0::3] = thetas; t[1::3] = thetas; t[2::3] = np.nan
            r[0::3] = r_base;  r[1::3] = r_top;   r[2::3] = np.nan
            ax.plot(t, r, "-", color=hcolor, lw=lw, alpha=alpha,
                    solid_capstyle="butt")

            # ring boundary + label
            ax.plot(t_circle, np.full(360, r_base), "-",
                    color="0.55", lw=0.4, alpha=0.5)
            ax.text(ring_label_angle, r_base + ring_step * 0.55,
                    horizon_labels[k],
                    ha="left", va="center", fontsize=8, fontweight="bold",
                    color=hcolor if isinstance(hcolor, str) else "k")

        # outer boundary
        ax.plot(t_circle, np.full(360, K * ring_step), "-",
                color="0.45", lw=0.7, alpha=0.6)
        ax.set_ylim(0, K * ring_step * 1.05)

    # ---- angular site labels ----
    if label_n_sites > 0:
        r_lim = ax.get_ylim()[1]
        for idx in np.linspace(0, N - 1, label_n_sites, dtype=int):
            ax.text(thetas[idx], r_lim * 1.09, str(idx + 1),
                    ha="center", va="center", fontsize=8, color="0.4")

    ax.set_title(
        title or f"Polar view: angle = ordered sites;\nradius = {metric_col}",
        fontsize=10, pad=14,
    )

    return _finish(fig, ax, savefig, dpi)


@check_non_emptiness
@isdf
def plot_paired_spatial_polar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_col: str,
    *,
    order_col: str | None = None,
    order_by: Literal["lat", "lon", "default"] = "lat",
    order_ascending: bool = True,
    horizon_cols: list[str] | str | None = None,
    horizon_labels: list[str] | None = None,
    horizon_colors: list[Any] | None = None,
    color_col: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    map_s: float = 15.0,
    polar_lw: float = 0.7,
    alpha: float = 0.80,
    map_label_sites: dict[int, str] | None = None,
    show_ordering_arrows: bool = False,
    arrow_step: int | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    map_title: str | None = None,
    polar_title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    n_ring_labels: int = 3,
    ring_label_angle: float = 0.25,
    zero_loc: str = "N",
    clockwise: bool = True,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (13.0, 6.0),
    dpi: int = 300,
    savefig: str | None = None,
    **kwargs: Any,
) -> list[Axes] | None:
    r"""Paired geographic map + polar hedgehog diagnostic.

    Creates a two-panel figure that ties together the spatial and polar
    representations of the same metric:

    * **Left panel** — geographic scatter map (color = metric value).
    * **Right panel** — polar hedgehog diagnostic where angle = spatially
      ordered site index and radius = metric value
      (see :func:`plot_polar_from_spatial`).

    Both panels share the same colormap and, optionally, the same color
    scale.  The left panel can optionally display ordering arrows and
    site labels (``map_label_sites``) so the reader can trace which
    geographic locations correspond to which polar angles.

    This layout directly reproduces the paired (a)+(b) and (c)+(d)
    panels of the k-diagram paper figures.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x_col : str
        Column with x-coordinates (longitude or easting).
    y_col : str
        Column with y-coordinates (latitude or northing).
    metric_col : str
        Column whose values are displayed in both panels.
    order_col : str, optional
        Pre-computed ordering column (overrides ``order_by``).
    order_by : {'lat', 'lon', 'default'}, default='lat'
        Spatial ordering criterion.
    order_ascending : bool, default=True
        Sort direction.
    horizon_cols : list of str or str, optional
        Additional horizon columns for multi-ring mode in the polar
        panel (see :func:`plot_polar_from_spatial`).
    horizon_labels : list of str, optional
        Ring labels for multi-ring mode.
    horizon_colors : list, optional
        One color per ring.
    color_col : str, optional
        Column for spike color in single-horizon polar mode.
    cmap : str, default='viridis'
        Colormap applied to both panels.
    vmin, vmax : float, optional
        Shared color scale limits.
    map_s : float, default=15
        Marker size for the geographic scatter.
    polar_lw : float, default=0.7
        Spike line width in the polar panel.
    alpha : float, default=0.80
        Transparency for both panels.
    map_label_sites : dict {int: str}, optional
        Site annotations for the map: keys are 0-based order indices,
        values are the label strings (e.g. ``{0: 'S1', 62: 'S2'}``).
    show_ordering_arrows : bool, default=False
        Draw direction arrows on the geographic map to visualise the
        traversal path.
    arrow_step : int, optional
        Arrow frequency; defaults to ``max(1, N // 15)``.
    colorbar : bool, default=True
        Draw a shared colorbar on the map panel.
    colorbar_label : str, optional
        Colorbar label.
    title : str, optional
        Super-title for the whole figure.
    map_title, polar_title : str, optional
        Per-panel titles (override defaults).
    xlabel, ylabel : str, optional
        Map axis labels.
    n_ring_labels : int, default=3
        Reference circle labels in the polar panel.
    ring_label_angle : float, default=0.25
        Angle (radians) for radial reference labels.
    zero_loc : str, default='N'
        Angle-zero location on the polar circle (``'N'`` = top).
    clockwise : bool, default=True
        Clockwise site ordering on the polar panel.
    show_grid : bool, default=True
        Grid on the map panel.
    grid_props : dict, optional
        Grid styling.
    figsize : (float, float), default=(13, 6)
        Figure size in inches.
    dpi : int, default=300
        Resolution for saving.
    savefig : str, optional
        Save path.
    **kwargs
        Forwarded to ``ax_map.scatter``.

    Returns
    -------
    axes : [map_ax, polar_ax] or None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from kdiagram.plot.spatial import plot_paired_spatial_polar
    >>> rng = np.random.default_rng(7)
    >>> n = 120
    >>> df = pd.DataFrame({
    ...     "lon":   rng.uniform(113.1, 113.6, n),
    ...     "lat":   rng.uniform(22.3,  22.8,  n),
    ...     "width": rng.uniform(0.5, 3.5, n),
    ... })
    >>> axes = plot_paired_spatial_polar(
    ...     df, "lon", "lat", "width",
    ...     order_by="lat",
    ...     cmap="YlOrRd",
    ...     map_label_sites={0: "S1", 59: "S2", 119: "S3"},
    ...     title="XTFT: Paired maps and polar diagnostics",
    ... )
    """
    hcols = columns_manager(horizon_cols, empty_as_none=False) or []
    required = [x_col, y_col, metric_col] + hcols
    if color_col:
        required.append(color_col)
    if order_col is not None:
        required.append(order_col)
    exist_features(df, features=required)

    data = df[list(dict.fromkeys(required))].dropna().copy()
    if data.empty:
        warnings.warn("plot_paired_spatial_polar: no data after dropping NaNs.",
                      stacklevel=2)
        return None

    ordered = _compute_site_order(data, x_col, y_col, order_by, order_ascending, order_col)
    N = len(ordered)

    # ---- figure layout ----
    fig = plt.figure(figsize=figsize)
    ax_map = fig.add_subplot(1, 2, 1)
    ax_pol = fig.add_subplot(1, 2, 2, projection="polar")

    metric_vals = ordered[metric_col].to_numpy(dtype=float)
    _vmin = vmin if vmin is not None else float(np.nanmin(metric_vals))
    _vmax = vmax if vmax is not None else float(np.nanmax(metric_vals))
    norm_shared = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
    cmap_obj = get_cmap(cmap, default="viridis")

    # ========================== left: geographic map ==========================
    sc = ax_map.scatter(
        ordered[x_col], ordered[y_col],
        c=metric_vals, s=map_s,
        cmap=cmap_obj, norm=norm_shared,
        alpha=alpha, edgecolors="none",
        **kwargs,
    )

    if colorbar:
        cbar = fig.colorbar(sc, ax=ax_map, pad=0.03, shrink=0.85)
        cbar.set_label(colorbar_label or metric_col, fontsize=9)

    if show_ordering_arrows:
        step = arrow_step if arrow_step is not None else max(1, N // 15)
        for i in range(0, N - 1, step):
            ax_map.annotate(
                "", xy=(float(ordered[x_col].iloc[i + 1]),
                         float(ordered[y_col].iloc[i + 1])),
                xytext=(float(ordered[x_col].iloc[i]),
                        float(ordered[y_col].iloc[i])),
                arrowprops=dict(arrowstyle="->", color="0.55",
                                lw=0.6, mutation_scale=7),
            )

    if map_label_sites:
        for idx, lbl in map_label_sites.items():
            if 0 <= idx < N:
                ax_map.annotate(
                    lbl,
                    (float(ordered[x_col].iloc[idx]),
                     float(ordered[y_col].iloc[idx])),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.72, lw=0),
                )

    ax_map.set_title(map_title or f"Map view: {metric_col}", fontsize=11)
    ax_map.set_xlabel(xlabel or x_col, fontsize=9)
    ax_map.set_ylabel(ylabel or y_col, fontsize=9)
    set_axis_grid(ax_map, show_grid=show_grid, grid_props=grid_props)

    # ========================== right: polar diagnostic =======================
    thetas = 2 * np.pi * np.arange(N) / N
    _setup_polar_ax(ax_pol, zero_loc=zero_loc, clockwise=clockwise)

    if not hcols:
        # single-horizon spikes
        color_vals = (
            ordered[color_col].to_numpy(dtype=float) if color_col
            else metric_vals
        )
        _polar_spikes(ax_pol, thetas, metric_vals, color_vals,
                      cmap_obj, norm_shared, polar_lw, alpha)
        r_max = float(np.nanmax(metric_vals))
        ax_pol.set_ylim(0, r_max * 1.08)
        _polar_reference_rings(ax_pol, r_max, n_ring_labels, ring_label_angle)

    else:
        # ring mode
        all_cols = [metric_col] + hcols
        K = len(all_cols)
        if horizon_labels is None:
            horizon_labels = [f"H{k + 1}" for k in range(K)]
        if horizon_colors is None:
            horizon_colors = [cmap_obj(k / max(1, K - 1)) for k in range(K)]

        all_ring_vals = np.concatenate(
            [ordered[c].to_numpy(dtype=float) for c in all_cols]
        )
        global_max = float(np.nanmax(all_ring_vals)) or 1.0
        ring_step = global_max / K
        t_circle = np.linspace(0, 2 * np.pi, 360)

        for k, col in enumerate(all_cols):
            r_base = k * ring_step
            r_top = r_base + (ordered[col].to_numpy(dtype=float) / global_max) * ring_step
            hcolor = horizon_colors[k]
            t = np.empty(3 * N); r = np.empty(3 * N)
            t[0::3] = thetas; t[1::3] = thetas; t[2::3] = np.nan
            r[0::3] = r_base;  r[1::3] = r_top;   r[2::3] = np.nan
            ax_pol.plot(t, r, "-", color=hcolor, lw=polar_lw, alpha=alpha,
                        solid_capstyle="butt")
            ax_pol.plot(t_circle, np.full(360, r_base), "-",
                        color="0.55", lw=0.4, alpha=0.5)
            ax_pol.text(ring_label_angle, r_base + ring_step * 0.55,
                        horizon_labels[k], ha="left", va="center",
                        fontsize=8, fontweight="bold",
                        color=hcolor if isinstance(hcolor, str) else "k")

        ax_pol.plot(t_circle, np.full(360, K * ring_step), "-",
                    color="0.45", lw=0.7, alpha=0.6)
        ax_pol.set_ylim(0, K * ring_step * 1.05)

    ax_pol.set_title(
        polar_title or (
            f"Polar view: angle = ordered sites;\nradius = {metric_col}"
        ),
        fontsize=10, pad=14,
    )

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    return _finish(fig, [ax_map, ax_pol], savefig, dpi)
