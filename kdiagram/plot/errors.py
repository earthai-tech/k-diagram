# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde, skew

from ..api.typing import Acov
from ..compat.matplotlib import get_cmap, get_colors
from ..core._io_utils import _get_valid_kwargs
from ..decorators import check_non_emptiness, isdf
from ..utils.fs import savefig as safe_savefig
from ..utils.plot import (
    map_theta_to_span,
    set_axis_grid,
    setup_polar_axes,
)
from ..utils.validator import exist_features

__all__ = ["plot_error_ellipses", "plot_error_bands", "plot_error_violins"]

def _get_mode (mode): 
    """ Find mode. Fallback to optimized if mode is "cbueth", 
    This will be deprecated in future version. 
    
    """
    mode = str(mode).lower() 
    if mode in {"optimized", "cbueth"}: 
        return "cbueth" 
    return mode 

def _plot_error_violins(
    df: pd.DataFrame,
    *error_cols: str,
    names: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 9.0),
    cmap: str = "viridis",
    colors: list[str] = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    acov: Acov = "default",
    ax: Axes | None = None,
    **violin_kws,
):
    # --- validate
    if not error_cols:
        raise ValueError("At least one error column must be provided.")
    exist_features(df, features=list(error_cols))

    if names and len(names) != len(error_cols):
        warnings.warn(
            "Names length does not match error columns. Using defaults.",
            UserWarning,
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i + 1}" for i in range(len(error_cols))]

    # --- gather all errors & common grid
    arrays = [df[c].dropna().to_numpy() for c in error_cols]
    all_err = np.concatenate(arrays) if arrays else np.array([0.0])
    r_min, r_max = float(np.min(all_err)), float(np.max(all_err))
    grid = np.linspace(r_min, r_max, 200)

    # KDE per series, normalized
    violin_data: list[np.ndarray | None] = []
    for arr in arrays:
        if arr.size < 2:
            violin_data.append(None)
            continue
        kde = gaussian_kde(arr)
        dens = kde(grid)
        m = float(np.max(dens)) if dens.size else 1.0
        dens = dens / (m if m > 0 else 1.0)
        violin_data.append(dens)

    # --- axes via utility (handles offset/dir/thetamax)
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
    )

    # angular slots across requested span
    k = len(error_cols)
    angles = np.linspace(0.0, float(span), k, endpoint=False)

    # colors
    colors = get_colors(
        k,
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # width of each violin in radians, scaled by span
    width = (float(span) / max(1, k)) * 0.8

    # draw violins (two-lobed polygon around the angle line)
    violin_kws = _get_valid_kwargs(ax.fill, violin_kws, error="ignore")

    for i, (ang, dens) in enumerate(zip(angles, violin_data)):
        if dens is None:
            continue

        # symmetric angular deviation around `ang`
        x = np.concatenate(
            [-dens * (width / 2.0), np.flip(dens * (width / 2.0))]
        )
        y = np.concatenate([grid, np.flip(grid)])

        theta = x + float(ang)
        r = y

        ax.fill(
            theta,
            r,
            color=colors[i],
            label=names[i],
            alpha=violin_kws.pop("alpha", 0.6),
            **violin_kws,
        )

    # zero-error reference (thin circle at r=0 over current span)
    ax.plot(
        np.linspace(0.0, float(span), 100),
        np.zeros(100),
        color="black",
        linestyle="--",
        lw=1.2,
        label="Zero Error",
    )

    # labels / grid / title
    ax.set_title(
        title or "Comparison of Error Distributions",
        fontsize=14,
    )
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(names)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    # saving
    final = safe_savefig(
        savefig,
        fig,
        dpi=dpi,
        bbox_inches="tight",
    )
    if final is not None:
        plt.close(fig)
    else:
        fig.tight_layout()
        plt.show()

    return ax


@check_non_emptiness
@isdf
def plot_error_violins(
    df: pd.DataFrame,
    *error_cols: str,
    names: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 9.0),
    cmap: str = "viridis",
    colors: list[str] = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    acov: Acov = "default",
    ax: Axes | None = None,
    mode: Literal["basic", "cbueth", "optimized"] = "optimized",
    bw_method: float | str | None = None,
    overlay: bool = False,
    overlay_angle: float | None = None,
    show_stats: bool = False,
    **violin_kws,
):
    mode = str(mode).lower()
    mode = _get_mode (mode ) 
    
    if mode == "basic":
        return _plot_error_violins(
            df,
            *error_cols,
            names=names,
            title=title,
            figsize=figsize,
            cmap=cmap,
            show_grid=show_grid,
            grid_props=grid_props,
            savefig=savefig,
            dpi=dpi,
            acov=acov,
            ax=ax,
            **violin_kws,
        )

    # --- validate
    if not error_cols:
        raise ValueError("At least one error column must be provided.")
    exist_features(df, features=list(error_cols))

    if names and len(names) != len(error_cols):
        warnings.warn(
            "Names length does not match error columns. Using defaults.",
            UserWarning,
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i + 1}" for i in range(len(error_cols))]

    # --- gather arrays
    arrays = [df[c].dropna().to_numpy() for c in error_cols]
    if not arrays:
        warnings.warn("No data after NaN removal.", stacklevel=2)
        return None

    # Build a nonnegative radial grid based on absolute error magnitude
    all_abs = np.abs(np.concatenate(arrays))
    r_max = float(np.max(all_abs)) if all_abs.size else 1.0
    r_grid = np.linspace(0.0, r_max, 200)

    # KDE for positive/negative sides (mapped to |error|); normalize to [0, 1]
    violins = []
    for arr in arrays:
        pos = arr[arr >= 0.0]
        neg = -arr[arr < 0.0]  # reflect

        dens_pos = None
        dens_neg = None

        if pos.size > 1:
            dp = gaussian_kde(pos, bw_method=bw_method)(r_grid)
            dp = dp / (np.max(dp) if np.max(dp) > 0 else 1.0)
            dens_pos = dp

        if neg.size > 1:
            dn = gaussian_kde(neg, bw_method=bw_method)(r_grid)
            dn = dn / (np.max(dn) if np.max(dn) > 0 else 1.0)
            dens_neg = dn

        violins.append((dens_pos, dens_neg))

    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
    )

    k = len(error_cols)

    # Auto-overlay: with 1–2 models, overlay improves visual comparison
    # (you can change default to "auto" in the signature later if you like).
    if overlay is None or (
        isinstance(overlay, str) and overlay.lower() == "auto"
    ):
        overlay = k <= 2

    # Headroom so outside text has space beyond the largest radius
    headroom = 1.26 if (not overlay and k <= 2) else 1.20
    ax.set_ylim(0, r_max * headroom)

    if overlay:
        base = np.pi / 2 if overlay_angle is None else float(overlay_angle)
        angles = np.full(k, base, dtype=float)
        width = float(span) * 0.32  # tighter for overlays
    else:
        angles = np.linspace(0.0, float(span), k, endpoint=False)
        width = (float(span) / max(1, k)) * 0.78  # slightly slimmer

    # colors
    colors = get_colors(
        k,
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # ---- draw two-lobe polygon per model ----
    alpha_default = float(violin_kws.pop("alpha", 0.45))
    edgecolor = violin_kws.pop("edgecolor", "black")
    lw = float(violin_kws.pop("linewidth", 0.8))
    violin_kws = _get_valid_kwargs(ax.fill, violin_kws, error="ignore")

    # order small → large so big shapes don’t hide small ones
    areas = []
    for dpos, dneg in violins:
        a = (np.trapz(dpos, r_grid) if dpos is not None else 0.0) + (
            np.trapz(dneg, r_grid) if dneg is not None else 0.0
        )
        areas.append(a)
    order = np.argsort(areas)

    for rank, i in enumerate(order):
        (dpos, dneg), ang, col = violins[i], float(angles[i]), colors[i]

        # fallback to zeros array if one side is missing
        z = np.zeros_like(r_grid)
        dpos = dpos if dpos is not None else z
        dneg = dneg if dneg is not None else z

        # left lobe = negatives, right lobe = positives
        left = ang - (0.5 * width) * dneg
        right = ang + (0.5 * width) * dpos

        theta = np.concatenate([left, right[::-1]])
        r = np.concatenate([r_grid, r_grid[::-1]])

        # legend label (stats only in legend)
        med = float(np.median(arrays[i]))
        skv = float(skew(arrays[i], bias=False))
        label_i = (
            names[i]
            if not show_stats
            else f"{names[i]}  (med={med:.2f}; skew={skv:.2f})"
        )

        ax.fill(
            theta,
            r,
            color=col,
            label=label_i,
            alpha=alpha_default,
            edgecolor=edgecolor,
            linewidth=lw,
            zorder=2 + rank,
            **violin_kws,
        )

    # ----- zero-error reference for |error| radial scale -----
    # r=0 collapses to the center; draw a tiny marker instead.
    zero_sz = max(12, 10000.0 / max(1.0, ax.figure.bbox.width))  # scale a bit
    ax.scatter(
        [0.0],
        [0.0],
        s=zero_sz,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        marker="o",
        label="Zero Error (center)",
    )

    # ----- labels / grid / title -----
    ax.set_title(title or "Comparison of Error Distributions", fontsize=14)
    ax.set_yticklabels([])

    # ----- labels / grid / title -----
    ax.set_title(title or "Comparison of Error Distributions", fontsize=14)
    ax.set_yticklabels([])

    if overlay:
        ax.set_xticks([])  # legend only; no outside labels in overlay mode
    else:
        ax.set_xticks([])
        r_out = r_max * (headroom + 0.04)  # just beyond the rim
        for name, ang in zip(names, angles):
            _label_outside(
                ax, ang, r_out, name
            )  # clip_on=False inside helper

    # single, clean legend (stats here, never on the plot)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.33, 1.06),
        frameon=False,
    )

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    # saving
    final = safe_savefig(
        savefig,
        fig,
        dpi=dpi,
        bbox_inches="tight",
    )
    if final is not None:
        plt.close(fig)
    else:
        fig.tight_layout()
        plt.show()

    return ax


def _label_outside(ax, angle, r_out, text, *, fontsize=11):
    rot = np.degrees(angle) - 90
    # keep upright text
    if -90 <= rot <= 90:
        ha = "left"
    else:
        rot += 180
        ha = "right"
    ax.text(
        angle,
        r_out,
        text,
        rotation=rot,
        rotation_mode="anchor",
        ha=ha,
        va="center",
        clip_on=False,
        fontsize=fontsize,
    )


plot_error_violins.__doc__ = r"""
Plot polar violin plots to compare multiple error distributions.

This function creates a polar plot where each angular sector
contains a violin plot representing the error distribution of a
different model or dataset. It is a powerful tool for visually
comparing bias, variance, and the overall shape of error
distributions [1]_.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the error data.

*error_cols : str
    One or more column names from ``df``, each containing the error
    values (e.g., ``actual - predicted``) for a model to be plotted.

names : list of str, optional
    Display names for each of the models corresponding to
    ``error_cols``. If not provided, generic names like
    ``'Model 1'`` will be generated. The list length must match
    the number of error columns.

title : str, optional
    The title for the plot. If ``None``, a default is generated.

figsize : tuple of (float, float), default=(9, 9)
    Figure size in inches.

cmap : str, default='viridis'
    Matplotlib colormap used to assign a unique color to each
    violin plot.
    
colors : list of str, optional
    An explicit list of colors to use for the violins. If
    provided, this overrides ``cmap``. The list will cycle if
    it is shorter than the number of error columns.

acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'},
    default='default'
    Angular coverage (span) of the plot:

    - ``'default'``: :math:`2\pi` (full circle)
    - ``'half_circle'``: :math:`\pi`
    - ``'quarter_circle'``: :math:`\tfrac{\pi}{2}`
    - ``'eighth_circle'``: :math:`\tfrac{\pi}{4}`
    
mode : {'optimized', 'basic'}, default='cbueth'
    The plotting mode to use.

    - ``'optimized'`` or ``'cbueth'`` (Default) A mode inspired by reviewer feedback
      (see Notes). It maps error magnitude to the radius, splits
      violins into positive/negative lobes, and uses a central
      dot for the zero reference. This mode is optimized for
      detecting bias and skew. 
    - ``'basic'``: The original implementation where the radial
      axis directly represents the error value (positive and
      negative), and violins are centered on their assigned angle.
      The zero reference is a dashed circle.

bw_method : float, str, or None, default=None
    The method used to calculate the estimator bandwidth for the
    KDE. This is passed directly to ``scipy.stats.gaussian_kde``.
    If ``None``, it uses the default "scott".

overlay : bool or 'auto', default='auto'
    *Applies to 'cbueth' mode only.*
    If ``True``, all violins are overlaid on a single shared spoke
    for direct comparison (best for k=1 or k=2).
    If ``False``, each violin gets its own spoke.
    If ``'auto'`` (default), overlay is enabled if k <= 2 and
    disabled otherwise.

overlay_angle : float, optional
    *Applies to 'cbueth' mode when ``overlay=True``.*
    The angle (in radians) for the shared spoke. If ``None``,
    defaults to :math:`\pi/2` (vertical North).

show_stats : bool, default=False
    *Applies to 'cbueth' mode only.*
    If ``True``, appends the median and skew of each distribution
    to its legend entry (e.g., "Model A (med=0.45; skew=-0.12)").
    
show_grid : bool, default=True
    Toggle gridlines via the package helper ``set_axis_grid``.

grid_props : dict, optional
    Keyword arguments passed to ``set_axis_grid`` for grid
    customization.

savefig : str, optional
    If provided, save the figure to this path; otherwise the
    plot is shown interactively.

dpi : int, default=300
    Resolution for the saved figure.

**violin_kws : dict, optional
    Additional keyword arguments passed to the ``ax.fill`` call
    for each violin (e.g., ``alpha``, ``edgecolor``).

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if the plot could not be generated.

Notes
-----
The plot visualizes and compares several one-dimensional error
distributions. It adapts the standard violin plot [1]_ to a polar
coordinate system for multi-model comparison.


1.  **Kernel Density Estimation (KDE)**: For each model's error
    data :math:`\mathbf{x} = \{x_1, x_2, ..., x_n\}`, the
    probability density function (PDF), :math:`\hat{f}_h(x)`, is
    estimated using a Gaussian kernel. This creates a smooth curve
    representing the distribution's shape.

    .. math::

       \hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)

    where :math:`K` is the Gaussian kernel and :math:`h` is the
    bandwidth, a smoothing parameter.

2.  **Violin Construction**: The violin shape is created by plotting
    the density curve :math:`\hat{f}_h(x)` symmetrically around a
    central axis. The width of the violin at any given error value
    :math:`x` is proportional to its estimated density.

3.  **Polar Arrangement**: Each model's violin is assigned a unique
    angular sector on the polar plot. The radial axis represents
    the error value, with a reference circle at :math:`r=0`
    indicating a perfect forecast. The violin is drawn radially
    within its assigned sector.

**The 'cbueth' Mode (Default)**

The default ``mode='cbueth'`` was developed in response to
insightful feedback during the JOSS paper review process. A
reviewer noted that the original "basic" mode could sometimes
make skewed distributions difficult to interpret, especially when
comparing only two models.

> *"...wouldn't it be easier to just plot them on top of each other
> with transparency? ... Maybe this is just a problem when just
> comparing two and not more models; with three it is already
> prettier."*

To honor this contribution, the new mode was named after the
reviewer's GitHub handle. It addresses this feedback with key
design changes:

1.  **Radial Axis:** The radius maps to **absolute error**
    :math:`|E|`, so all data starts from the center. The zero-error
    reference is a single point at the origin.
2.  **Two-Lobe Design:** Each violin is split into two lobes around
    its central spoke:
        
    * **Right Lobe:** Distribution of positive errors (:math:`E > 0`).
    * **Left Lobe:** Distribution of negative errors (:math:`E < 0`).
    
3.  **Interpretation:** This design makes key metrics instantly visible:
    
    * **Bias:** An imbalance in the *size* of the two lobes
        (e.g., a larger right lobe means a positive bias).
    * **Skew:** Asymmetry *within* a single lobe.
    * **Variance:** The overall radial extent of the lobes.
    
4.  **Auto-Overlay:** As suggested, when only two models are
    plotted (``overlay='auto'``), they are drawn on top of
    each other with transparency for a direct, "face-off"
    style comparison. For 3+ models, they are given separate spokes.
    
Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.errors import plot_polar_error_violins
>>>
>>> # Simulate errors from three different models
>>> np.random.seed(0)
>>> n_points = 1000
>>> df_errors = pd.DataFrame({
...     'Model A (Good)': np.random.normal(
...           loc=0.5, scale=1.5, size=n_points),
...     'Model B (Biased)': np.random.normal(
...           loc=-4.0, scale=1.5, size=n_points),
...     'Model C (Inconsistent)': np.random.normal(
...           loc=0, scale=4.0, size=n_points),
... })
>>>
>>> # Generate the polar violin plot
>>> ax = plot_polar_error_violins(
...     df_errors,
...     'Model A (Good)',
...     'Model B (Biased)',
...     'Model C (Inconsistent)',
...     title='Comparison of Model Error Distributions',
...     cmap='plasma',
...     alpha=0.7
... )

References
----------
.. [1] Hintze, J. L., & Nelson, R. D. (1998). Violin Plots: A Box
   Plot-Density Trace Synergism. The American Statistician, 52(2),
   181-184.

"""


@check_non_emptiness
@isdf
def plot_error_bands(
    df: pd.DataFrame,
    error_col: str,
    theta_col: str,
    *,
    theta_period: float | None = None,
    theta_bins: int = 24,
    n_std: float = 1.0,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_angle: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    acov: Acov = "default",
    ax: Axes | None = None,
    **fill_kws,
):
    # --- validate
    exist_features(df, features=[error_col, theta_col])

    data = df[[error_col, theta_col]].dropna()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs in required columns.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # --- axes via utility
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
    )

    # map theta data into [0, span]
    t_raw = data[theta_col].to_numpy()

    if theta_period is not None:
        data["theta_map"] = map_theta_to_span(
            t_raw,
            span=span,
            theta_period=float(theta_period),
        )
    else:
        tmin = float(np.min(t_raw))
        tmax = float(np.max(t_raw))
        if tmax > tmin + 1e-12:
            data["theta_map"] = map_theta_to_span(
                t_raw,
                span=span,
                data_min=tmin,
                data_max=tmax,
            )
        else:
            data["theta_map"] = 0.0

    # --- binning over span
    theta_edges = np.linspace(0.0, float(span), theta_bins + 1)
    theta_labels = (theta_edges[:-1] + theta_edges[1:]) / 2.0

    data["theta_bin"] = pd.cut(
        data["theta_map"],
        bins=theta_edges,
        labels=theta_labels,
        include_lowest=True,
    )

    # stats per bin (mean ± n_std*std)
    stats = (
        data.groupby("theta_bin", observed=False)[error_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)

    # --- draw mean + band
    ang = stats["theta_bin"].astype(float).to_numpy()
    mu = stats["mean"].to_numpy()
    sd = stats["std"].to_numpy()

    ax.plot(
        ang,
        mu,
        color="black",
        lw=2.0,
        label="Mean Error",
    )
    ax.fill_between(
        ang,
        mu - float(n_std) * sd,
        mu + float(n_std) * sd,
        alpha=fill_kws.pop("alpha", 0.3),
        label=f"{n_std} Std. Dev. Band",
        **fill_kws,
    )

    # zero-error reference as a thin circle over the current span
    ax.plot(
        np.linspace(0.0, float(span), 180),
        np.zeros(180),
        color="red",
        linestyle="--",
        lw=1.2,
        label="Zero Error",
    )

    # titles / labels / grid
    ax.set_title(
        title or f"Error Distribution vs. {theta_col}",
        fontsize=14,
    )
    ax.set_ylabel(f"Forecast Error ({error_col})")

    if mask_angle:
        ax.set_xticklabels([])

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_error_bands.__doc__ = r"""
Plot polar error bands to visualize systemic vs random error.

This function aggregates forecast errors across bins of a cyclical
or ordered feature (like month or hour) and plots the mean error
and its standard deviation. It is a powerful diagnostic tool for
identifying systemic biases and variations in model performance.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the error and feature data.

error_col : str
    Name of the column containing the forecast error values,
    typically calculated as ``actual - predicted``.

theta_col : str
    Name of the column representing the feature to bin against,
    which will be mapped to the angular axis.

theta_period : float, optional
    The period of the cyclical data in ``theta_col``. For example,
    if ``theta_col`` is the month of the year, the period is 12.
    This ensures the data wraps around the circle correctly.

theta_bins : int, default=24
    The number of angular bins to group the data into for
    calculating statistics.

n_std : float, default=1.0
    The number of standard deviations to display in the shaded
    error band around the mean error line.

title : str, optional
    The title for the plot. If ``None``, a default is generated.

figsize : tuple of (float, float), default=(8, 8)
    Figure size in inches.

cmap : str, default='viridis'
    *Note: This parameter is currently not used in this function
    as colors are fixed for clarity (black, red, and a fill color).*

show_grid : bool, default=True
    Toggle gridlines via the package helper ``set_axis_grid``.

grid_props : dict, optional
    Keyword arguments passed to ``set_axis_grid`` for grid
    customization.

mask_angle : bool, default=False
    If ``True``, hide the angular tick labels.

savefig : str, optional
    If provided, save the figure to this path; otherwise the
    plot is shown interactively.

dpi : int, default=300
    Resolution for the saved figure.

**fill_kws : dict, optional
    Additional keyword arguments passed to the ``ax.fill_between``
    call for the shaded error band (e.g., ``color``, ``alpha``).

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if the plot could not be generated.

Notes
-----
The plot visualizes the first two moments (mean and standard
deviation) of the error distribution conditioned on the angular
variable :math:`\theta`.

1.  **Binning**: The data is first partitioned into :math:`K` bins
    based on the values in ``theta_col``. Let :math:`B_k` be the set
    of indices of data points belonging to the :math:`k`-th bin.

2.  **Mean Error Calculation**: For each bin :math:`B_k`, the mean
    error :math:`\mu_{e,k}` is calculated. This value is plotted as a
    point on the central black line.

    .. math::

       \mu_{e,k} = \frac{1}{|B_k|} \sum_{i \in B_k} e_i

    where :math:`e_i` is the error for data point :math:`i`. A
    consistent deviation of this line from the zero-error circle
    indicates a **systemic bias**.

3.  **Error Variance Calculation**: For each bin, the standard
    deviation of the error, :math:`\sigma_{e,k}`, is also calculated.

    .. math::

       \sigma_{e,k} = \sqrt{\frac{1}{|B_k|-1}\\
                            \sum_{i \in B_k} (e_i - \mu_{e,k})^2}

4.  **Band Construction**: A shaded band is drawn between the lower
    and upper bounds, defined by the mean plus or minus a multiple
    of the standard deviation.

    .. math::

       \text{Upper Bound}_k &= \mu_{e,k} + n_{std} \cdot \sigma_{e,k} \\
       \text{Lower Bound}_k &= \mu_{e,k} - n_{std} \cdot \sigma_{e,k}

    The width of this band indicates the **random error** or
    inconsistency of the model within that bin.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.errors import plot_error_bands
>>>
>>> # Simulate a model with seasonal error patterns
>>> np.random.seed(42)
>>> n_points = 2000
>>> day_of_year = np.arange(n_points) % 365
>>> month = (day_of_year // 30) + 1
>>>
>>> # Create a bias (positive error) in summer and more noise in winter
>>> seasonal_bias = np.sin((day_of_year - 90) * np.pi / 180) * 5
>>> seasonal_noise = 2 + 2 * np.cos(day_of_year * np.pi / 180)**2
>>> errors = seasonal_bias + np.random.normal(0, seasonal_noise, n_points)
>>>
>>> df_seasonal = pd.DataFrame({'month': month, 'forecast_error': errors})
>>>
>>> # Generate the plot
>>> ax = plot_error_bands(
...     df=df_seasonal,
...     error_col='forecast_error',
...     theta_col='month',
...     theta_period=12,
...     theta_bins=12,
...     n_std=1.5,
...     title='Seasonal Forecast Error Analysis',
...     color='#2980B9',
...     alpha=0.3
... )
"""


@check_non_emptiness
@isdf
def plot_error_ellipses(
    df: pd.DataFrame,
    r_col: str,
    theta_col: str,
    r_std_col: str,
    theta_std_col: str,
    *,
    color_col: str | None = None,
    n_std: float = 2.0,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "viridis",
    mask_angle: bool = False,
    mask_radius: bool = False,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    acov: Acov = "default",
    ax: Axes | None = None,
    **ellipse_kws,
):
    required = [r_col, theta_col, r_std_col, theta_std_col]
    if color_col:
        required.append(color_col)
    exist_features(df, features=required)

    data = df[required].dropna()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs in required "
            "columns. Cannot plot.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Color metric (defaults to radial uncertainty)
    if color_col:
        color_vals = data[color_col].to_numpy()
        cbar_label = color_col
    else:
        color_vals = data[r_std_col].to_numpy()
        cbar_label = f"Uncertainty ({r_std_col})"

    # Normalize colors
    vmin = float(np.min(color_vals))
    vmax = float(np.max(color_vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(norm(color_vals))

    # Axes & angular coverage (sets offset/dir/thetamax)
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
    )

    # Map theta (degrees) into [0, span] radians
    # Use a 360° period to preserve circularity
    theta_deg = data[theta_col].to_numpy(dtype=float)
    theta_map = map_theta_to_span(
        theta_deg,
        span=span,
        theta_period=360.0,
    )

    # Scale angular std to the chosen span (assumed radians input)
    # If your theta_std is in degrees, convert before scaling.
    theta_std_raw = data[theta_std_col].to_numpy(dtype=float)
    angle_scale = float(span) / (2.0 * np.pi)
    theta_std_eff = theta_std_raw * angle_scale

    # Draw each ellipse as a filled path in polar coordinates
    r_mean = data[r_col].to_numpy(dtype=float)
    r_std = data[r_std_col].to_numpy(dtype=float)

    for i in range(len(data)):
        th_path, r_path = _get_ellipse_path(
            r_mean=r_mean[i],
            theta_mean=theta_map[i],  # already in radians
            r_std=r_std[i],
            theta_std=theta_std_eff[i],  # scaled for acov
            n_std=n_std,
        )
        ax.fill(
            th_path,
            r_path,
            color=colors[i],
            **ellipse_kws,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label(cbar_label, fontsize=10)

    # Formatting
    ax.set_title(
        title or f"Error Ellipses ({n_std:.1f} std. dev.)",
        fontsize=14,
    )
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_angle:
        ax.set_xticklabels([])
    if mask_radius:
        ax.set_yticklabels([])

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _get_ellipse_path(
    r_mean: float,
    theta_mean: float,
    r_std: float,
    theta_std: float,
    n_std: float = 2.0,
):
    """
    Helper to calculate the path of an ellipse in polar coordinates.
    The ellipse is defined in a local Cartesian frame and then
    transformed.
    """
    # width along radial, height along tangential direction
    width = n_std * r_std
    height = n_std * (r_mean * np.sin(theta_std))

    # ellipse center in Cartesian
    x_c = r_mean * np.cos(theta_mean)
    y_c = r_mean * np.sin(theta_mean)

    # parametric ellipse
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    x_loc = (width / 2.0) * np.cos(t)
    y_loc = (height / 2.0) * np.sin(t)

    # rotate ellipse to align with radial direction
    R = np.array(
        [
            [np.cos(theta_mean), -np.sin(theta_mean)],
            [np.sin(theta_mean), np.cos(theta_mean)],
        ]
    )
    x_rot, y_rot = np.dot(R, [x_loc, y_loc])
    x_fin = x_rot + x_c
    y_fin = y_rot + y_c

    # back to polar
    r_path = np.sqrt(x_fin**2 + y_fin**2)
    theta_path = np.arctan2(y_fin, x_fin)
    return theta_path, r_path


plot_error_ellipses.__doc__ = r"""
Plot polar error ellipses to visualize two-dimensional uncertainty.

This function draws ellipses on a polar plot to represent the
uncertainty of data points where both the radial and angular
components have associated errors (standard deviations).

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the data for the plot.

r_col : str
    Name of the column for the mean radial position (e.g., distance).

theta_col : str
    Name of the column for the mean angular position. **Must be in
    degrees.**

r_std_col : str
    Name of the column for the standard deviation of the radial
    position.

theta_std_col : str
    Name of the column for the standard deviation of the angular
    position. **Must be in degrees.**

color_col : str, optional
    Name of a column to use for coloring the ellipses. If ``None``,
    ellipses are colored by their radial uncertainty (``r_std_col``).

n_std : float, default=2.0
    The number of standard deviations to use for the ellipse size.
    For example, ``n_std=2.0`` corresponds to approximately a 95%
    confidence region for a normal distribution.

title : str, optional
    The title for the plot. If ``None``, a default is generated.

figsize : tuple of (float, float), default=(8, 8)
    Figure size in inches.

cmap : str, default='viridis'
    Matplotlib colormap for coloring the ellipses.

show_grid : bool, default=True
    Toggle gridlines via the package helper ``set_axis_grid``.

grid_props : dict, optional
    Keyword arguments passed to ``set_axis_grid`` for grid
    customization.

mask_angle : bool, default=False
    If ``True``, hide the angular tick labels (degrees).

mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.

savefig : str, optional
    If provided, save the figure to this path; otherwise the
    plot is shown interactively.

dpi : int, default=300
    Resolution for the saved figure.

**ellipse_kws : dict, optional
    Additional keyword arguments passed to the ``ax.fill`` call
    for each ellipse (e.g., ``alpha``, ``edgecolor``).

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if the plot could not be generated.

Notes
-----
The visualization for each data point :math:`i` is constructed
from its mean radial position :math:`\mu_{r,i}`, mean angular
position :math:`\mu_{\theta,i}`, and their respective standard
deviations :math:`\sigma_{r,i}` and :math:`\sigma_{\theta,i}`.

1.  **Ellipse Dimensions**: The ellipse is first defined in a local
    Cartesian coordinate system at the origin. Its half-width (along
    the radial direction) and half-height (along the tangential
    direction) are determined by the standard deviations:

    .. math::

        \text{width} &= n_{std} \cdot \sigma_{r,i} \\
        \text{height} &= n_{std} \cdot (\mu_{r,i} \cdot \sin(\sigma_{\theta,i}))

    Note that the tangential height depends on the radial distance
    :math:`\mu_{r,i}`.

2.  **Transformation**: This local ellipse is then transformed to the
    correct position on the polar plot. This involves two steps:
    
    a. **Rotation**: The ellipse is rotated by the mean angle
       :math:`\mu_{\theta,i}` to align its primary axis with the
       radial direction from the origin.
    b. **Translation**: The rotated ellipse is translated to the
       mean position, which in Cartesian coordinates is
       :math:`(x_c, y_c) = (\mu_{r,i} \cos(\mu_{\theta,i}), \mu_{r,i} \sin(\mu_{\theta,i}))`.

3.  **Plotting**: The final transformed ellipse is drawn as a filled
    path on the polar axes.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.errors import plot_polar_error_ellipses
>>>
>>> # Simulate tracking data for 15 objects
>>> np.random.seed(1)
>>> n_points = 15
>>> df_tracking = pd.DataFrame({
...     'angle_deg': np.linspace(0, 360, n_points, endpoint=False),
...     'distance_km': np.random.uniform(20, 80, n_points),
...     'distance_std': np.random.uniform(2, 7, n_points),
...     'angle_std_deg': np.random.uniform(3, 10, n_points),
...     'object_priority': np.random.randint(1, 5, n_points)
... })
>>>
>>> # Generate the plot
>>> ax = plot_polar_error_ellipses(
...     df=df_tracking,
...     r_col='distance_km',
...     theta_col='angle_deg',
...     r_std_col='distance_std',
...     theta_std_col='angle_std_deg',
...     color_col='object_priority',
...     n_std=1.5,
...     title='1.5-Sigma Positional Uncertainty',
...     cmap='cividis',
...     alpha=0.7,
...     edgecolor='black',
...     linewidth=0.5
... )
"""
