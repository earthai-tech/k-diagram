# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Probabilistic Forecast Evaluation Plots
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import kstest, uniform

from ..api.typing import Acov
from ..compat.matplotlib import get_cmap
from ..compat.sklearn import validate_params
from ..decorators import check_non_emptiness, isdf
from ..utils.plot import map_theta_to_span, set_axis_grid, setup_polar_axes
from ..utils.validator import exist_features, validate_yy

__all__ = [
    "plot_crps_comparison",
    "plot_pit_histogram",
    "plot_polar_sharpness",
    "plot_credibility_bands",
    "plot_calibration_sharpness",
]


@validate_params(
    {
        "y_true": ["array-like"],
        "y_preds_quantiles": ["array-like"],
        "quantiles": ["array-like"],
    }
)
@check_non_emptiness(params=["y_true", "y_preds_quantiles"])
def plot_pit_histogram(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
    *,
    acov: Acov = "default",
    n_bins: int = 10,
    title: str = "PIT Histogram",
    figsize: tuple[float, float] = (8, 8),
    color: str = "#3498DB",
    edgecolor: str = "black",
    alpha: float = 0.7,
    show_uniform_line: bool = True,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    # --- validation
    y_true, y_preds_quantiles = validate_yy(
        y_true,
        y_preds_quantiles,
        expected_type=None,
        allow_2d_pred=True,
    )
    quantiles = np.asarray(quantiles)
    if y_preds_quantiles.shape[1] != len(quantiles):
        raise ValueError(
            "Shape mismatch: y_preds_quantiles columns "
            f"({y_preds_quantiles.shape[1]}) must match "
            f"len(quantiles) ({len(quantiles)})."
        )

    # --- PIT values
    sort_idx = np.argsort(quantiles)
    sorted_preds = y_preds_quantiles[:, sort_idx]
    pit_values = np.mean(sorted_preds <= y_true[:, None], axis=1)

    # --- histogram
    hist, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # --- axes & angular span
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
    )

    # angles on [0, span], constant bar width = span / n_bins
    angles = bin_centers * span
    radii = hist
    width = float(span) / float(n_bins)

    # --- bars
    ax.bar(
        angles,
        radii,
        width=width,
        color=color,
        edgecolor=edgecolor,
        alpha=alpha,
        label="PIT Frequency",
    )

    # --- uniform reference
    if show_uniform_line:
        expected = len(y_true) / float(n_bins)
        ax.plot(
            np.linspace(0.0, float(span), 100),
            np.full(100, expected),
            color="red",
            linestyle="--",
            lw=2.0,
            label=f"Uniform ({expected:.1f})",
        )

    # --- ticks & labels (show PIT bin left-edges as labels)
    ax.set_xticks(np.linspace(0.0, float(span), n_bins, endpoint=False))
    ax.set_xticklabels([f"{e:.1f}" for e in bin_edges[:-1]])
    ax.set_xlabel("PIT Value Bins")
    ax.set_ylabel("Frequency")

    # --- title, legend, grid
    ax.set_title(title, fontsize=14, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    # --- output
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_pit_histogram.__doc__ = r"""
Plots a Polar Probability Integral Transform (PIT) Histogram.

This function creates a polar bar chart of PIT values to
diagnose the calibration of a probabilistic forecast. For a
perfectly calibrated forecast, the PIT histogram is uniform,
which results in a perfect circle on the polar plot. Deviations
from this shape indicate specific model biases.

Parameters
----------
y_true : np.ndarray
    1D array of observed (true) values.
y_preds_quantiles : np.ndarray
    2D array of quantile forecasts. Each row corresponds to an
    observation in ``y_true``, and each column is a specific
    quantile forecast.
quantiles : np.ndarray
    1D array of the quantile levels corresponding to the columns
    of ``y_preds_quantiles`` (e.g., ``[0.05, 0.1, ..., 0.95]``).
acov : {'default', 'half_circle', 'quarter_circle',
    'eighth_circle'}, default='default'
    Angular coverage of the polar sector.

    - ``'default'``        : full circle, :math:`2\pi` (360°)
    - ``'half_circle'``    : :math:`\pi` (180°)
    - ``'quarter_circle'`` : :math:`\pi/2` (90°)
    - ``'eighth_circle'``  : :math:`\pi/4` (45°)
    
n_bins : int, default=10
    Number of bins for the histogram, which will correspond to
    the angular sectors in the polar plot.
title : str, default="PIT Histogram"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
color : str, default="#3498DB"
    The fill color for the histogram bars.
edgecolor : str, default="black"
    The edge color for the histogram bars.
alpha : float, default=0.7
    The transparency of the histogram bars.
show_uniform_line : bool, default=True
    If ``True``, draws a reference circle indicating the expected
    frequency for a perfectly uniform (calibrated) distribution.
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

Notes
-----
The Probability Integral Transform (PIT) is a fundamental tool
for evaluating the calibration of probabilistic forecasts
:footcite:p:`Gneiting2007b`. For a continuous predictive
distribution with CDF :math:`F`, the PIT value for an
observation :math:`y` is :math:`F(y)`. If the forecast is
perfectly calibrated, the PIT values are uniformly distributed
on :math:`[0, 1]`.

When the predictive CDF is represented by a finite set of
:math:`M` quantiles, the PIT value for each observation
:math:`y_i` is approximated as the fraction of forecast
quantiles that are less than or equal to the observation:

.. math::

   \text{PIT}_i = \frac{1}{M} \sum_{j=1}^{M}
   \mathbf{1}\{q_{i,j} \le y_i\}

where :math:`q_{i,j}` is the :math:`j`-th quantile forecast
for observation :math:`i`, and :math:`\mathbf{1}` is the
indicator function.

Deviations from a uniform (flat) histogram indicate
miscalibration:
- **U-shaped**: The forecast is overconfident (too narrow).
- **Hump-shaped**: The forecast is underconfident (too wide).
- **Sloped**: The forecast is biased.

Examples
--------
>>> import numpy as np
>>> from scipy.stats import norm
>>> from kdiagram.plot.probabilistic import plot_pit_histogram
>>>
>>> # Generate synthetic data
>>> np.random.seed(42)
>>> n_samples = 1000
>>> y_true = np.random.normal(loc=10, scale=5, size=n_samples)
>>> quantiles = np.linspace(0.05, 0.95, 19)
>>>
>>> # A well-calibrated forecast
>>> calibrated_preds = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=5
... )
>>>
>>> # Generate the plot
>>> ax = plot_pit_histogram(
...     y_true,
...     calibrated_preds,
...     quantiles,
...     title="PIT Histogram (Well-Calibrated Model)"
... )

References
----------
.. footbibliography::
"""


@validate_params(
    {
        "quantiles": ["array-like"],
    }
)
@check_non_emptiness(params=["y_preds_quantiles"])
def plot_polar_sharpness(
    *y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
    names: list[str] | None = None,
    title: str = "Forecast Sharpness Comparison",
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "viridis",
    marker: str = "o",
    s: int = 100,
    acov: Acov = "default",
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    # ---- validate inputs
    if not y_preds_quantiles:
        raise ValueError("At least one prediction array must be provided.")
    quantiles = np.asarray(quantiles)
    if quantiles.ndim != 1:
        raise ValueError("`quantiles` must be a 1D array.")

    if names and len(names) != len(y_preds_quantiles):
        warnings.warn(
            "Number of names does not match number of models. "
            "Using defaults.",
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i + 1}" for i in range(len(y_preds_quantiles))]

    # ---- compute sharpness per model (widest interval width)
    sharpness_scores: list[float] = []
    for preds in y_preds_quantiles:
        preds = np.asarray(preds)
        if preds.shape[1] != len(quantiles):
            raise ValueError(
                "Prediction array shape mismatch with quantiles."
            )
        lo = preds[:, np.argmin(quantiles)]
        hi = preds[:, np.argmax(quantiles)]
        sharpness_scores.append(float(np.mean(hi - lo)))

    # ---- axes + span
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
    )

    # ---- angles and radii
    n = len(y_preds_quantiles)
    angles = np.linspace(0.0, float(span), n, endpoint=False)
    radii = np.asarray(sharpness_scores, dtype=float)

    # ---- draw
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0.0, 1.0, n))
    ax.scatter(
        angles,
        radii,
        c=colors,
        s=s,
        marker=marker,
        zorder=3,
    )

    # labels near points
    for i, name in enumerate(names):
        ax.text(
            angles[i],
            radii[i],
            f"  {name}\n  ({radii[i]:.2f})",
            ha="left",
            va="center",
            fontsize=9,
        )

    # ---- formatting
    ax.set_title(title, fontsize=14, y=1.1)
    ax.set_xticks([])  # no angular ticks
    ax.set_ylabel("Average Interval Width (Sharpness)")
    ax.set_ylim(bottom=0.0)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
    if mask_radius:
        ax.set_yticklabels([])

    # ---- output
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _calculate_crps(y_true, y_preds_quantiles, quantiles):
    """
    Approximates the CRPS using the pinball loss averaged over quantiles.
    """
    y_true = y_true[:, np.newaxis]  # Reshape for broadcasting
    pinball_loss = np.where(
        y_true >= y_preds_quantiles,
        (y_true - y_preds_quantiles) * quantiles,
        (y_preds_quantiles - y_true) * (1 - quantiles),
    )
    # Average over quantiles for each observation, then over all observations
    return np.mean(np.mean(pinball_loss, axis=1))


plot_polar_sharpness.__doc__ = r"""
Plots a Polar Sharpness Diagram to compare forecast precision.

This function creates a polar plot to visually compare the
sharpness of one or more probabilistic forecasts. Sharpness is
a measure of the concentration of the predictive distribution,
typically quantified by the average width of the prediction
intervals. Sharper (more precise) forecasts are represented by
points closer to the center of the plot.

Parameters
----------
*y_preds_quantiles : np.ndarray
    One or more 2D arrays of quantile forecasts. Each array
    corresponds to a different model, with shape
    ``(n_samples, n_quantiles)``.
quantiles : np.ndarray
    1D array of the quantile levels corresponding to the columns
    of the prediction arrays.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
title : str, default="Forecast Sharpness Comparison"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    marker.
marker : str, default='o'
    The marker style for the points representing each model.
s : int, default=100
    The size of the markers.
acov : {'default', 'half_circle', 'quarter_circle',
    'eighth_circle'}, default='default'
    Angular coverage of the polar sector.

    - ``'default'``        : full circle, :math:`2\pi` (360°)
    - ``'half_circle'``    : :math:`\pi` (180°)
    - ``'quarter_circle'`` : :math:`\pi/2` (90°)
    - ``'eighth_circle'``  : :math:`\pi/4` (45°)
    
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

Notes
-----
A good probabilistic forecast should be both calibrated
(reliable) and as sharp as possible :footcite:p:`Gneiting2007b`.
This diagram focuses on sharpness, which is independent of the
observed outcomes.

1.  **Interval Width**: For each model and each observation
    :math:`i`, the width of the central prediction interval is
    calculated using the lowest and highest provided quantiles
    (:math:`q_{min}` and :math:`q_{max}`).

    .. math::

       w_i = y_{i, q_{max}} - y_{i, q_{min}}

2.  **Sharpness Score**: The sharpness score :math:`S` for each
    model is the average of these interval widths over all
    :math:`N` observations. This score is used as the radial
    coordinate in the plot. A lower score is better.

    .. math::

       S = \frac{1}{N} \sum_{i=1}^{N} w_i

Examples
--------
>>> import numpy as np
>>> from scipy.stats import norm
>>> from kdiagram.plot.probabilistic import plot_polar_sharpness
>>>
>>> # Generate synthetic data for two models
>>> np.random.seed(0)
>>> n_samples = 500
>>> y_true = np.random.normal(loc=20, scale=5, size=n_samples)
>>> quantiles = np.linspace(0.1, 0.9, 9) # 80% interval
>>>
>>> # A sharp (precise) forecast
>>> sharp_preds = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=2
... )
>>> # A wide (less precise) forecast
>>> wide_preds = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=5
... )
>>>
>>> # Generate the plot
>>> ax = plot_polar_sharpness(
...     sharp_preds,
...     wide_preds,
...     quantiles=quantiles,
...     names=["Sharp Model", "Wide Model"]
... )

References
----------
.. footbibliography::

"""


@validate_params(
    {
        "y_true": ["array-like"],
        "quantiles": ["array-like"],
    }
)
@check_non_emptiness(params=["y_true", "y_preds_quantiles"])
def plot_crps_comparison(
    y_true: np.ndarray,
    *y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
    names: list[str] | None = None,
    title: str = "Probabilistic Forecast Performance (CRPS)",
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "viridis",
    marker: str = "o",
    s: int = 100,
    acov: Acov = "default",
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    # ---- validate inputs
    if not y_preds_quantiles:
        raise ValueError("At least one prediction array must be provided.")
    quantiles = np.asarray(quantiles)

    if names and len(names) != len(y_preds_quantiles):
        warnings.warn(
            "Number of names does not match number of models. "
            "Using defaults.",
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i + 1}" for i in range(len(y_preds_quantiles))]

    # ---- compute CRPS per model
    crps_scores: list[float] = []
    for preds in y_preds_quantiles:
        y_t, q_pred = validate_yy(
            y_true,
            preds,
            expected_type=None,
            allow_2d_pred=True,
        )
        crps = _calculate_crps(y_t, q_pred, quantiles)
        crps_scores.append(float(crps))

    # ---- axes + span
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
    )

    # ---- angles and radii
    n = len(y_preds_quantiles)
    angles = np.linspace(0.0, float(span), n, endpoint=False)
    radii = np.asarray(crps_scores, dtype=float)

    # ---- draw
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0.0, 1.0, n))
    ax.scatter(
        angles,
        radii,
        c=colors,
        s=s,
        marker=marker,
        zorder=3,
    )

    # labels near points
    for i, name in enumerate(names):
        ax.text(
            angles[i],
            radii[i],
            f"  {name}\n  ({radii[i]:.3f})",
            ha="left",
            va="center",
            fontsize=9,
        )

    # ---- formatting
    ax.set_title(title, fontsize=14, y=1.1)
    ax.set_xticks([])  # no angular ticks
    ax.set_ylabel("Average CRPS (Lower is Better)")
    ax.set_ylim(bottom=0.0)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
    if mask_radius:
        ax.set_yticklabels([])

    # ---- output
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_crps_comparison.__doc__ = r"""
Plots a Polar CRPS Comparison Diagram.

This function visualizes the overall performance of one or more
probabilistic forecasts using the Continuous Ranked Probability
Score (CRPS). The CRPS is a proper scoring rule that assesses
both calibration and sharpness simultaneously. A lower CRPS value
indicates a better forecast. In this plot, models closer to the
center are superior.

Parameters
----------
y_true : np.ndarray
    1D array of observed (true) values.
*y_preds_quantiles : np.ndarray
    One or more 2D arrays of quantile forecasts. Each array
    corresponds to a different model, with shape
    ``(n_samples, n_quantiles)``.
quantiles : np.ndarray
    1D array of the quantile levels corresponding to the columns
    of the prediction arrays.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
title : str, default="Probabilistic Forecast Performance (CRPS)"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    marker.
marker : str, default='o'
    The marker style for the points representing each model.
s : int, default=100
    The size of the markers.
acov : {'default', 'half_circle', 'quarter_circle',
    'eighth_circle'}, default='default'
    Angular coverage of the polar sector.

    - ``'default'``        : full circle, :math:`2\pi` (360°)
    - ``'half_circle'``    : :math:`\pi` (180°)
    - ``'quarter_circle'`` : :math:`\pi/2` (90°)
    - ``'eighth_circle'``  : :math:`\pi/4` (45°)
    
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

Notes
-----
The Continuous Ranked Probability Score (CRPS) is a widely
used metric for evaluating probabilistic forecasts
:footcite:p:`Gneiting2007b`. For a single observation :math:`y`
and a predictive CDF :math:`F`, it is defined as:

.. math::

   \text{CRPS}(F, y) = \int_{-\infty}^{\infty}\\
       (F(x) - \mathbf{1}\{x \ge y\})^2 dx

where :math:`\mathbf{1}` is the Heaviside step function.

When the forecast is given as a set of :math:`M` quantiles
:math:`\{q_1, ..., q_M\}`, the CRPS can be approximated by
averaging the pinball loss :math:`\mathcal{L}_{\tau}` over the
quantile levels :math:`\tau \in \{ \tau_1, ..., \tau_M \}`:

.. math::

   \text{CRPS}(F, y) \approx \frac{1}{M} \sum_{j=1}^{M} 2\\
       \mathcal{L}_{\tau_j}(q_j, y)

The pinball loss for a quantile :math:`\tau` is:

.. math::

   \mathcal{L}_{\tau}(q, y) =
   \begin{cases}
     (y - q) \tau & \text{if } y \ge q \\
     (q - y) (1 - \tau) & \text{if } y < q
   \end{cases}

This function calculates the average CRPS over all observations
for each model and plots it as the radial coordinate.

Examples
--------
>>> import numpy as np
>>> from scipy.stats import norm
>>> from kdiagram.plot.probabilistic import plot_crps_comparison
>>>
>>> # Generate synthetic data
>>> np.random.seed(42)
>>> n_samples = 1000
>>> y_true = np.random.normal(loc=10, scale=5, size=n_samples)
>>> quantiles = np.linspace(0.05, 0.95, 19)
>>>
>>> # Create forecasts for three models
>>> good_preds = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=5
... )
>>> sharp_biased_preds = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis] - 2, scale=3
... )
>>> wide_preds = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=8
... )
>>>
>>> # Generate the plot
>>> ax = plot_crps_comparison(
...     y_true,
...     good_preds,
...     sharp_biased_preds,
...     wide_preds,
...     quantiles=quantiles,
...     names=["Good", "Sharp/Biased", "Wide"]
... )

References
----------
.. footbibliography::
"""


@check_non_emptiness(params=["df"])
@isdf
def plot_credibility_bands(
    df: pd.DataFrame,
    q_cols: tuple[str, str, str],
    theta_col: str,
    *,
    theta_period: float | None = None,
    theta_bins: int = 24,
    acov: Acov = "default",
    title: str = "Forecast Credibility Bands",
    figsize: tuple[float, float] = (8.0, 8.0),
    color: str = "#3498DB",
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    theta_ticks: Sequence[float] | None = None,
    theta_ticklabels: Sequence[str] | Mapping[int, str] | None = None,
    zero_at: Literal["N", "E", "S", "W"] = "N",
    clockwise: bool = True,
    ax: Axes | None = None,
    **fill_kws,
) -> Axes | None:
    # --- input checks
    if len(q_cols) != 3:
        raise ValueError(
            "`q_cols` must be a 3-tuple: (lower, median, upper)."
        )
    q_low_col, q_med_col, q_up_col = q_cols
    req = [q_low_col, q_med_col, q_up_col, theta_col]
    exist_features(df, features=req)

    data = df[req].dropna().copy()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # --- axes & span (in radians)
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
        zero_at=zero_at,
        clockwise=clockwise,
    )

    # --- map theta to [0, span]
    t_raw = data[theta_col].to_numpy()
    if theta_period is not None:
        data["theta_map"] = map_theta_to_span(
            t_raw,
            span=span,
            theta_period=theta_period,
        )
    else:
        tmin = float(t_raw.min())
        tmax = float(t_raw.max())
        if tmax > tmin + 1e-12:
            data["theta_map"] = map_theta_to_span(
                t_raw,
                span=span,
                data_min=tmin,
                data_max=tmax,
            )
        else:
            data["theta_map"] = 0.0

    # --- binning along theta and mean stats per bin

    theta_edges = np.linspace(0.0, float(span), int(theta_bins) + 1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
    data["theta_bin"] = pd.cut(
        data["theta_map"],
        bins=theta_edges,
        labels=theta_centers,
        include_lowest=True,
    )

    stats = (
        data.groupby("theta_bin", observed=False)
        .agg(
            {
                q_low_col: "mean",
                q_med_col: "mean",
                q_up_col: "mean",
            }
        )
        .reset_index()
    )

    # --- plotting
    ang = stats["theta_bin"].astype(float).to_numpy()
    med = stats[q_med_col].to_numpy()
    lo = stats[q_low_col].to_numpy()
    hi = stats[q_up_col].to_numpy()

    # median line
    ax.plot(
        ang,
        med,
        color="black",
        lw=2.0,
        label="Mean Median Forecast",
    )

    # shaded band
    alpha_val = float(fill_kws.pop("alpha", 0.3))
    ax.fill_between(
        ang,
        lo,
        hi,
        color=color,
        alpha=alpha_val,
        label="Credibility Band",
        **fill_kws,
    )

    tick_pos = None
    tick_lbls = None

    if theta_ticks is not None:
        tick_pos = np.asarray(theta_ticks, dtype=float)

    if theta_ticklabels is not None:
        if isinstance(theta_ticklabels, dict):
            # map by bin index (0..theta_bins-1)
            tick_pos = theta_centers if tick_pos is None else tick_pos
            # build labels from mapping; fallback to str(index)
            idx = np.arange(len(theta_centers))
            tick_lbls = [theta_ticklabels.get(i, str(i)) for i in idx]
        else:
            tick_lbls = list(theta_ticklabels)

    if tick_pos is None:
        tick_pos = theta_centers

    ax.set_xticks(tick_pos)

    if tick_lbls is not None:
        ax.set_xticklabels(tick_lbls)

    # --- formatting
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_xlabel(f"Binned by {theta_col}")
    ax.set_ylabel("Forecast Value", labelpad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
    if mask_radius:
        ax.set_yticklabels([])

    # --- output
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _calculate_ks_statistic(pit_values):
    """
    Calculates the Kolmogorov-Smirnov statistic to measure deviation
    of PIT values from a perfect uniform distribution.
    """
    if len(pit_values) < 2:
        return 1.0  # Max penalty for insufficient data
    # Compare the empirical distribution of PIT values to a uniform distribution
    ks_statistic, _ = kstest(pit_values, uniform.cdf)
    return ks_statistic


plot_credibility_bands.__doc__ = r"""
Plots Polar Credibility Bands to visualize forecast uncertainty.

This function creates a polar plot that shows how the median
forecast and the prediction interval bounds change as a function
of another binned variable (e.g., month, hour). It is a
descriptive tool for understanding the structure of a model's
predictions and its uncertainty estimates.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the forecast data.
q_cols : tuple of (str, str, str)
    A tuple of three column names for the lower quantile, the
    median (Q50), and the upper quantile, in that order.
theta_col : str
    The name of the column to bin against for the angular axis.
theta_period : float, optional
    The period of the cyclical data in ``theta_col`` (e.g., 24
    for hours, 12 for months). This ensures the data wraps
    correctly around the polar plot.
theta_bins : int, default=24
    The number of angular bins to group the data into.
acov : {'default', 'half_circle', 'quarter_circle',
    'eighth_circle'}, default='default'
    Angular coverage of the polar sector.

    - ``'default'``        : full circle, :math:`2\pi` (360°)
    - ``'half_circle'``    : :math:`\pi` (180°)
    - ``'quarter_circle'`` : :math:`\pi/2` (90°)
    - ``'eighth_circle'``  : :math:`\pi/4` (45°)
    
title : str, default="Forecast Credibility Bands"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
color : str, default="#3498DB"
    The color for the shaded credibility band.
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
**fill_kws
    Additional keyword arguments passed to the ``ax.fill_between``
    call for the shaded band (e.g., ``alpha``).

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

Notes
-----
This plot visualizes the conditional expectation of the
forecast quantiles. It is a novel visualization developed as
part of the analytics framework in :footcite:t:`kouadiob2025`.

1.  **Binning**: The data is first partitioned into :math:`K` bins,
    :math:`B_k`, based on the values in ``theta_col``.

2.  **Conditional Means**: For each bin :math:`B_k`, the mean
    of the lower quantile (:math:`\bar{q}_{low,k}`), median
    quantile (:math:`\bar{q}_{med,k}`), and upper quantile
    (:math:`\bar{q}_{up,k}`) are calculated.

    .. math::

       \bar{q}_{j,k} = \frac{1}{|B_k|} \sum_{i \in B_k} q_{j,i}

    where :math:`j \in \{\text{low, med, up}\}`.

3.  **Visualization**: The plot displays:
    
    - A central line representing the mean median forecast
      (:math:`\bar{q}_{med,k}`).
    - A shaded band between the mean lower and upper bounds
      (:math:`\bar{q}_{low,k}` and :math:`\bar{q}_{up,k}`). The
      width of this band represents the average forecast
      sharpness for that bin.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.probabilistic import plot_credibility_bands
>>>
>>> # Simulate a forecast with seasonal uncertainty
>>> np.random.seed(0)
>>> n_points = 500
>>> month = np.random.randint(1, 13, n_points)
>>> median = 50 + 20 * np.sin((month - 3) * np.pi / 6)
>>> width = 10 + 8 * np.cos(month * np.pi / 6)**2
>>>
>>> df = pd.DataFrame({
...     'month': month,
...     'q50': median + np.random.randn(n_points),
...     'q10': median - width / 2,
...     'q90': median + width / 2,
... })
>>>
>>> # Generate the plot
>>> ax = plot_credibility_bands(
...     df=df,
...     q_cols=('q10', 'q50', 'q90'),
...     theta_col='month',
...     theta_period=12,
...     theta_bins=12,
...     title="Seasonal Forecast Credibility"
... )

References
----------
.. footbibliography::
    
"""


@check_non_emptiness(params=["y_true", "y_preds_quantiles"])
def plot_calibration_sharpness(
    y_true: np.ndarray,
    *y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
    names: list[str] | None = None,
    title: str = "Calibration vs. Sharpness Trade-off",
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "viridis",
    marker: str = "o",
    s: int = 150,
    acov: Acov = "default",
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    # --- input checks
    if not y_preds_quantiles:
        raise ValueError("At least one prediction array must be provided.")
    if not names:
        names = [f"Model {i + 1}" for i in range(len(y_preds_quantiles))]

    # --- scores
    sharpness_scores: list[float] = []
    calibration_scores: list[float] = []

    for preds in y_preds_quantiles:
        y_t, q_pred = validate_yy(
            y_true,
            preds,
            allow_2d_pred=True,
        )

        # sharpness: widest interval mean width
        lo = q_pred[:, np.argmin(quantiles)]
        hi = q_pred[:, np.argmax(quantiles)]
        sharpness_scores.append(float(np.mean(hi - lo)))

        # calibration: KS statistic on PIT
        order = np.argsort(quantiles)
        sorted_pred = q_pred[:, order]
        pit = np.mean(sorted_pred <= y_t[:, None], axis=1)
        ks_val = _calculate_ks_statistic(pit)
        calibration_scores.append(float(ks_val))

    # --- axes + span
    fig, ax, span = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
    )

    # --- angles/radii
    # map KS in [0,1] -> angle in [0, span]
    ks = np.asarray(calibration_scores, dtype=float)
    angles = ks * float(span)
    radii = np.asarray(sharpness_scores, dtype=float)

    # --- draw
    n = len(y_preds_quantiles)
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0.0, 1.0, n))

    ax.scatter(
        angles,
        radii,
        c=colors,
        s=s,
        marker=marker,
        zorder=3,
        alpha=0.8,
    )

    # labels
    for i, name in enumerate(names):
        ax.text(
            float(angles[i]),
            float(radii[i]),
            f"  {name}",
            ha="left",
            va="bottom",
            fontsize=9,
        )

    # --- formatting
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_ylim(bottom=0.0)

    # angular ticks reflect KS in [0,1]
    xt = np.linspace(0.0, float(span), 5)
    xl = [f"{v:.2f}" for v in np.linspace(0.0, 1.0, 5)]
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)

    ax.set_xlabel("Calibration Error (Lower is Better)")
    ax.set_ylabel("Sharpness (Lower is Better)", labelpad=25)

    # legend showing color->model mapping
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color=colors[i],
            label=names[i],
            linestyle="None",
            markersize=10,
        )
        for i in range(n)
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.1),
    )

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
    if mask_radius:
        ax.set_yticklabels([])

    # --- output
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_calibration_sharpness.__doc__ = r"""
Plots a Polar Calibration-Sharpness Diagram.

This function creates a polar plot to visualize the fundamental
trade-off between forecast **calibration** (reliability) and
**sharpness** (precision) for one or more models. Each model is
represented by a single point, allowing for a direct and
intuitive comparison of their overall probabilistic performance.

The ideal forecast is located at the center of the plot,
representing perfect calibration and perfect sharpness.

Parameters
----------
y_true : np.ndarray
    1D array of observed (true) values.
*y_preds_quantiles : np.ndarray
    One or more 2D arrays of quantile forecasts. Each array
    corresponds to a different model, with shape
    ``(n_samples, n_quantiles)``.
quantiles : np.ndarray
    1D array of the quantile levels corresponding to the columns
    of the prediction arrays.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
title : str, default="Calibration vs. Sharpness Trade-off"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    marker.
marker : str, default='o'
    The marker style for the points representing each model.
s : int, default=150
    The size of the markers.
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

Notes
-----
This plot synthesizes two key aspects of a probabilistic
forecast into a single point for each model. It is a novel
visualization developed as part of the analytics framework in
:footcite:t:`kouadiob2025`.

1.  **Sharpness (Radius)**: The radial coordinate represents the
    forecast's sharpness, calculated as the average width of the
    prediction interval between the lowest and highest provided
    quantiles. A smaller radius is better (sharper).

    .. math::

       S = \frac{1}{N} \sum_{i=1}^{N} (y_{i, q_{max}} - y_{i, q_{min}})

2.  **Calibration Error (Angle)**: The angular coordinate
    represents the forecast's calibration error. This is
    quantified by first calculating the Probability Integral
    Transform (PIT) values for each observation. The
    Kolmogorov-Smirnov (KS) statistic is then used to measure
    the maximum distance between the empirical CDF of these PIT
    values and the CDF of a perfect uniform distribution.

    .. math::

       E_{calib} = \sup_{x} | F_{PIT}(x) - U(x) |

    An error of 0 indicates perfect calibration. The angle is
    mapped such that :math:`\theta = E_{calib} \cdot \frac{\pi}{2}`,
    so 0° is perfect and 90° is the worst possible calibration.

Examples
--------
>>> import numpy as np
>>> from scipy.stats import norm
>>> from kdiagram.plot.probabilistic import plot_calibration_sharpness
>>>
>>> # Generate synthetic data
>>> np.random.seed(42)
>>> n_samples = 1000
>>> y_true = np.random.normal(loc=10, scale=5, size=n_samples)
>>> quantiles = np.linspace(0.05, 0.95, 19)
>>>
>>> # Create forecasts for three models with different trade-offs
>>> model_A = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=5
... ) # Balanced
>>> model_B = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis] - 2, scale=3
... ) # Sharp but biased
>>> model_C = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=8
... ) # Calibrated but wide
>>>
>>> # Generate the plot
>>> ax = plot_calibration_sharpness(
...     y_true,
...     model_A, model_B, model_C,
...     quantiles=quantiles,
...     names=["Balanced", "Sharp/Biased", "Calibrated/Wide"]
... )

References
----------
.. footbibliography::
"""
