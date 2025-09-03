# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>
from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..api.typing import Acov
from ..compat.matplotlib import get_cmap
from ..compat.sklearn import StrOptions, validate_params
from ..utils.generic_utils import drop_nan_in
from ..utils.plot import (
    map_theta_to_span,
    set_axis_grid,
    setup_polar_axes,
)
from ..utils.validator import validate_yy

__all__ = [
    "plot_relationship",
    "plot_conditional_quantiles",
    "plot_residual_relationship",
    "plot_error_relationship",
]


@validate_params({"y_true": ["array-like"]})
def plot_residual_relationship(
    y_true: np.ndarray,
    *y_preds: np.ndarray,
    names: list[str] | None = None,
    title: str = "Residual vs. Predicted Relationship",
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "viridis",
    s: int = 50,
    alpha: float = 0.7,
    show_zero_line: bool = True,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    mask_angle: bool = False,
    mask_radius: bool = False,
    acov: Acov = "default",
    ax: Axes | None = None,
):
    # --- validate / prepare
    if not y_preds:
        raise ValueError("At least one prediction array is required.")

    y_true, *y_preds = drop_nan_in(y_true, *y_preds, error="raise")
    y_true_val, _ = validate_yy(y_true, y_preds[0])

    if not names:
        names = [f"Model {i + 1}" for i in range(len(y_preds))]

    # errors and r-offset (to allow negatives on polar radius)
    errs_list = [y_true_val - np.asarray(yp) for yp in y_preds]
    all_errs = np.concatenate(errs_list)
    r_offset = np.abs(np.min(all_errs)) if np.min(all_errs) < 0 else 0.0

    # --- axes & colors
    fig, ax, span = setup_polar_axes(ax, acov=acov, figsize=figsize)
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0.0, 1.0, len(y_preds)))

    # zero-error circle over current span
    if show_zero_line:
        ax.plot(
            np.linspace(0.0, float(span), 120),
            np.full(120, r_offset),
            color="black",
            linestyle="--",
            lw=1.4,
            label="Zero Error",
        )

    # --- plot each model
    for i, (yp, errs) in enumerate(zip(y_preds, errs_list)):
        y_pred_val = np.asarray(yp)
        idx = np.argsort(y_pred_val)
        y_pred_sorted = y_pred_val[idx]
        errs_sorted = errs[idx]

        # map predicted values -> [0, span] (radians)
        if y_pred_sorted.max() > y_pred_sorted.min():
            theta = map_theta_to_span(
                y_pred_sorted,
                span=span,
                data_min=float(y_pred_sorted.min()),
                data_max=float(y_pred_sorted.max()),
            )
        else:
            theta = np.zeros_like(y_pred_sorted, dtype=float)

        radii = errs_sorted + r_offset
        ax.scatter(
            theta, radii, color=colors[i], s=s, alpha=alpha, label=names[i]
        )

    # --- formatting
    ax.set_title(title, fontsize=16, y=1.08)
    ax.set_xlabel("Based on Predicted Value")
    ax.set_ylabel("Forecast Error (Actual - Predicted)", labelpad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1))
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


plot_residual_relationship.__doc__ = r"""
Plots the relationship between forecast error and predicted value.

This function creates a polar scatter plot, a polar version of a
classic residual plot, to diagnose model performance. The angle is
proportional to the **predicted value**, and the radius represents
the **forecast error**. It is a powerful tool for identifying
conditional biases and heteroscedasticity related to the model's
own output magnitude.

Parameters
----------
y_true : np.ndarray
    1D array of true observed values.
*y_preds : np.ndarray
    One or more 1D arrays of predicted values from different
    models.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
title : str, default="Residual vs. Predicted Relationship"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    markers.
s : int, default=50
    The size of the scatter plot markers.
alpha : float, default=0.7
    The transparency of the markers.
show_zero_line : bool, default=True
    If ``True``, draws a reference circle representing zero error.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.
mask_angle : bool, default=False
    If ``True``, hide the angular tick labels.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'},
    default='default'
    Angular coverage (span) of the plot:

    - ``'default'``: :math:`2\pi` (full circle)
    - ``'half_circle'``: :math:`\pi`
    - ``'quarter_circle'``: :math:`\tfrac{\pi}{2}`
    - ``'eighth_circle'``: :math:`\tfrac{\pi}{4}`
    
Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

See Also
--------
plot_error_relationship : Plot error vs. the true value.
plot_conditional_quantiles : Visualize full conditional quantile bands.

Notes
-----
This plot is a novel visualization developed as part of the
analytics framework in :footcite:p:`kouadiob2025`. It helps
diagnose if the model's error is correlated with its own
predictions.

1.  **Error (Residual) Calculation**: For each observation
    :math:`i`, the error is the difference between the true and
    predicted value.

    .. math::
       :label: eq:error_calc

       e_i = y_{true,i} - y_{pred,i}

2.  **Angular Mapping**: The angle :math:`\theta_i` is made
    proportional to the predicted value :math:`y_{pred,i}`,
    after sorting, to create a continuous spiral.

    .. math::

       \theta_i \propto y_{pred,i}

3.  **Radial Mapping**: The radius :math:`r_i` represents the
    error :math:`e_i`. To handle negative error values on a
    polar plot, an offset is added to all radii so that the
    zero-error line becomes a reference circle.

Examples
--------
>>> import numpy as np
>>> from kdiagram.plot.relationship import plot_residual_relationship
>>>
>>> # Generate synthetic data with known flaws
>>> np.random.seed(0)
>>> n_samples = 200
>>> y_true = np.linspace(0, 20, n_samples)**1.5
>>> # Model has errors that increase with the prediction magnitude
>>> noise = np.random.normal(0, 1, n_samples) * (y_true / 20)
>>> y_pred = y_true + noise
>>>
>>> # Generate the plot
>>> ax = plot_residual_relationship(
...     y_true,
...     y_pred,
...     names=["My Model"],
...     title="Residual vs. Predicted Value (Heteroscedasticity)"
... )

References
----------
.. footbibliography::
"""


@validate_params({"y_true": ["array-like"]})
def plot_error_relationship(
    y_true: np.ndarray,
    *y_preds: np.ndarray,
    names: list[str] | None = None,
    title: str = "Error vs. True Value Relationship",
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "viridis",
    s: int = 50,
    alpha: float = 0.7,
    show_zero_line: bool = True,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_angle: bool = False,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    acov: Acov = "default",
    ax: Axes | None = None,
):
    # --- validate / prepare
    if not y_preds:
        raise ValueError("At least one prediction array is required.")

    y_true, *y_preds = drop_nan_in(y_true, *y_preds, error="raise")
    y_true, _ = validate_yy(y_true, y_preds[0])

    if not names:
        names = [f"Model {i + 1}" for i in range(len(y_preds))]

    errs_list = [y_true - np.asarray(yp) for yp in y_preds]
    all_errs = np.concatenate(errs_list)
    # To handle negative errors on a polar plot, we shift the origin.
    # The zero-error line will be a circle.
    r_offset = np.abs(np.min(all_errs)) if np.min(all_errs) < 0 else 0.0

    # sort true values for a smooth angle mapping
    idx = np.argsort(y_true)
    y_true_sorted = y_true[idx]

    # map true values -> [0, span] (radians)
    fig, ax, span = setup_polar_axes(ax, acov=acov, figsize=figsize)
    if y_true_sorted.max() > y_true_sorted.min():
        theta = map_theta_to_span(
            y_true_sorted,
            span=span,
            data_min=float(y_true_sorted.min()),
            data_max=float(y_true_sorted.max()),
        )
    else:
        theta = np.zeros_like(y_true_sorted, dtype=float)

    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0.0, 1.0, len(y_preds)))

    # zero-error circle
    if show_zero_line:
        ax.plot(
            np.linspace(0.0, float(span), 120),
            np.full(120, r_offset),
            color="black",
            linestyle="--",
            lw=1.4,
            label="Zero Error",
        )

    # plot each model’s errors
    for i, errs in enumerate(errs_list):
        errs_sorted = errs[idx]
        radii = errs_sorted + r_offset
        ax.scatter(
            theta, radii, color=colors[i], s=s, alpha=alpha, label=names[i]
        )

    # --- formatting
    ax.set_title(title, fontsize=16, y=1.08)
    nm = getattr(y_true, "name", "True Value")
    ax.set_xlabel(f"Based on {nm}")
    ax.set_ylabel("Forecast Error", labelpad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1))
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


plot_error_relationship.__doc__ = r"""
Plots the relationship between forecast error and the true value.

This function creates a polar scatter plot to diagnose model
performance by visualizing the structure of its errors. The
angle is proportional to the **true value**, and the radius
represents the **forecast error**. It is a powerful tool for
identifying conditional biases and heteroscedasticity.

Parameters
----------
y_true : np.ndarray
    1D array of true observed values.
*y_preds : np.ndarray
    One or more 1D arrays of predicted values from different
    models.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
title : str, default="Error vs. True Value Relationship"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    markers.
s : int, default=50
    The size of the scatter plot markers.
alpha : float, default=0.7
    The transparency of the markers.
show_zero_line : bool, default=True
    If ``True``, draws a reference circle representing zero error.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
mask_angle : bool, default=False
    If ``True``, hide the angular tick labels.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.
acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'},
    default='default'
    Angular coverage (span) of the plot:

    - ``'default'``: :math:`2\pi` (full circle)
    - ``'half_circle'``: :math:`\pi`
    - ``'quarter_circle'``: :math:`\tfrac{\pi}{2}`
    - ``'eighth_circle'``: :math:`\tfrac{\pi}{4}`
    
Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

See Also
--------
plot_residual_relationship : Plot error vs. the predicted value.
plot_conditional_quantiles : Visualize full conditional quantile bands.

Notes
-----
This plot is a novel visualization developed as part of the
analytics framework in :footcite:p:`kouadiob2025`. It helps
diagnose if the model's error is correlated with the true
value, a key assumption in many statistical models.

1.  **Error (Residual) Calculation**: For each observation
    :math:`i`, the error is the difference between the true and
    predicted value.

    .. math::

       e_i = y_{true,i} - y_{pred,i}

2.  **Angular Mapping**: The angle :math:`\theta_i` is made
    proportional to the true value :math:`y_{true,i}`,
    after sorting, to create a continuous spiral.

    .. math::

       \theta_i \propto y_{true,i}

3.  **Radial Mapping**: The radius :math:`r_i` represents the
    error :math:`e_i`. To handle negative error values on a
    polar plot, an offset is added to all radii so that the
    zero-error line becomes a reference circle.

Examples
--------
>>> import numpy as np
>>> from kdiagram.plot.relationship import plot_error_relationship
>>>
>>> # Generate synthetic data with known flaws
>>> np.random.seed(0)
>>> n_samples = 200
>>> y_true = np.linspace(0, 20, n_samples)**1.5
>>> # Model has a bias that depends on the true value
>>> bias = -0.1 * y_true
>>> y_pred = y_true + bias + np.random.normal(0, 2, n_samples)
>>>
>>> # Generate the plot
>>> ax = plot_error_relationship(
...     y_true,
...     y_pred,
...     names=["My Model"],
...     title="Error vs. True Value (Conditional Bias)"
... )

References
----------
.. footbibliography::
    
"""


@validate_params(
    {
        "y_true": ["array-like"],
        "y_preds_quantiles": ["array-like"],
        "quantiles": ["array-like"],
    }
)
def plot_conditional_quantiles(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
    *,
    bands: list[int] | None = None,
    title: str = "Conditional Quantile Plot",
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "viridis",
    alpha_min: float = 0.2,
    alpha_max: float = 0.5,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_angle: bool = False,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    acov: Acov = "default",
    ax: Axes | None = None,
):
    # --- validate inputs
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )
    quantiles = np.asarray(quantiles)
    if y_preds_quantiles.shape[1] != len(quantiles):
        raise ValueError("Shape mismatch between predictions and quantiles.")

    # sort by y_true for a smooth angular sweep
    idx = np.argsort(y_true)
    y_true_sorted = y_true[idx]
    y_preds_sorted = y_preds_quantiles[idx, :]

    # --- axes with requested angular coverage
    fig, ax, span = setup_polar_axes(ax, acov=acov, figsize=figsize)

    # map y_true -> [0, span] radians (handles constant case)
    if y_true_sorted.max() > y_true_sorted.min():
        theta = map_theta_to_span(
            y_true_sorted,
            span=span,
            data_min=float(y_true_sorted.min()),
            data_max=float(y_true_sorted.max()),
        )
    else:
        theta = np.zeros_like(y_true_sorted, dtype=float)
        warnings.warn(
            "y_true has zero range; mapping all angles to 0.",
            UserWarning,
            stacklevel=2,
        )

    # --- median + bands setup
    med_q = 0.5
    if med_q in quantiles:
        med_idx = int(np.where(np.isclose(quantiles, med_q))[0][0])
    else:
        med_idx = -1
        warnings.warn(
            "Median (0.5) not found in quantiles."
            " No central line will be plotted.",
            UserWarning,
            stacklevel=2,
        )

    if bands is None:
        qmin, qmax = float(np.min(quantiles)), float(np.max(quantiles))
        bands = [int((qmax - qmin) * 100)]

    bands = sorted(bands, reverse=True)
    cmap_obj = get_cmap(cmap, default="viridis")
    alphas = np.linspace(alpha_min, alpha_max, len(bands))
    colors = cmap_obj(np.linspace(0.3, 0.9, len(bands)))

    # --- draw quantile bands
    for i, pct in enumerate(bands):
        lo_q = (100 - pct) / 200.0
        hi_q = 1.0 - lo_q
        try:
            lo_idx = int(np.where(np.isclose(quantiles, lo_q))[0][0])
            hi_idx = int(np.where(np.isclose(quantiles, hi_q))[0][0])
        except IndexError:
            warnings.warn(
                f"Quantiles for {pct}% interval not found; skip.",
                UserWarning,
                stacklevel=2,
            )
            continue

        ax.fill_between(
            theta,
            y_preds_sorted[:, lo_idx],
            y_preds_sorted[:, hi_idx],
            color=colors[i],
            alpha=float(alphas[i]),
            label=f"{pct}% Interval",
        )

    # --- median curve
    if med_idx != -1:
        ax.plot(
            theta,
            y_preds_sorted[:, med_idx],
            color="black",
            lw=1.5,
            label="Median (Q50)",
        )

    # --- formatting
    nm = getattr(y_true, "name", "True Value")
    ax.set_title(title, fontsize=16, y=1.08)
    ax.set_xlabel(f"Based on {nm}")
    ax.set_ylabel("Predicted Value", labelpad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1))
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


plot_conditional_quantiles.__doc__ = r"""
Plots polar conditional quantile bands.

This function visualizes how the predicted conditional
distribution (represented by quantiles) changes as a function
of the true observed value. It is a powerful tool for
diagnosing heteroscedasticity, i.e., whether the forecast
uncertainty is constant or changes with the magnitude of the
target variable.

Parameters
----------
y_true : np.ndarray
    1D array of true observed values, which will be mapped
    to the angular coordinate.
y_preds_quantiles : np.ndarray
    2D array of quantile forecasts, with shape
    ``(n_samples, n_quantiles)``.
quantiles : np.ndarray
    1D array of the quantile levels corresponding to the columns
    of ``y_preds_quantiles``.
bands : list of int, optional
    A list of the desired interval percentages to plot as
    shaded bands (e.g., ``[90, 50]`` for the 90% and 50%
    prediction intervals). Defaults to the widest interval
    available from the provided quantiles.
title : str, default="Conditional Quantile Plot"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap for the shaded uncertainty bands.
alpha_min : float, default=0.2
    The minimum alpha (transparency) for the outermost band.
alpha_max : float, default=0.5
    The maximum alpha for the innermost band.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
mask_angle : bool, default=False
    If ``True``, hide the angular tick labels.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.
acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'},
    default='default'
    Angular coverage (span) of the plot:

    - ``'default'``: :math:`2\pi` (full circle)
    - ``'half_circle'``: :math:`\pi`
    - ``'quarter_circle'``: :math:`\tfrac{\pi}{2}`
    - ``'eighth_circle'``: :math:`\tfrac{\pi}{4}`
    
Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

Notes
-----
This plot is a novel visualization developed as part of the
analytics framework in :footcite:p:`kouadiob2025`. It provides
an intuitive view of the conditional predictive distribution.

1.  **Coordinate Mapping**: The plot first sorts the data based
    on the true values :math:`y_{true}` to ensure a continuous
    spiral. The sorted true values are then mapped to the
    angular coordinate :math:`\theta` in the range :math:`[0, 2\pi]`.

    .. math::

       \theta_i \propto y_{true,i}^{\text{(sorted)}}

    The predicted quantiles :math:`q_{i, \tau}` for each
    observation :math:`i` and quantile level :math:`\tau` are
    mapped directly to the radial coordinate :math:`r`.

2.  **Band Construction**: For a given prediction interval, for
    example 80%, the corresponding lower (:math:`\tau=0.1`) and
    upper (:math:`\tau=0.9`) quantile forecasts are used to
    define the boundaries of a shaded band. The function can
    plot multiple, nested bands (e.g., 80% and 50%) to give a
    more complete picture of the distribution's shape. The
    median forecast (:math:`\tau=0.5`) is drawn as a solid
    central line.

Examples
--------
>>> import numpy as np
>>> from kdiagram.plot.relationship import plot_conditional_quantiles
>>>
>>> # Generate synthetic data with heteroscedasticity
>>> np.random.seed(0)
>>> n_samples = 200
>>> y_true = np.linspace(0, 20, n_samples)**1.5
>>> quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
>>>
>>> # Uncertainty (interval width) increases with the true value
>>> interval_width = 5 + (y_true / y_true.max()) * 15
>>> y_preds = np.zeros((n_samples, len(quantiles)))
>>> y_preds[:, 2] = y_true # Median
>>> y_preds[:, 1] = y_true - interval_width * 0.25 # Q25
>>> y_preds[:, 3] = y_true + interval_width * 0.25 # Q75
>>> y_preds[:, 0] = y_true - interval_width * 0.5  # Q10
>>> y_preds[:, 4] = y_true + interval_width * 0.5  # Q90
>>>
>>> # Generate the plot
>>> ax = plot_conditional_quantiles(
...     y_true,
...     y_preds,
...     quantiles,
...     bands=[80, 50], # Show 80% and 50% intervals
...     title="Conditional Uncertainty (Heteroscedasticity)"
... )

References
----------
.. footbibliography::
"""


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "theta_scale": [StrOptions({"proportional", "uniform"})],
        "acov": [
            StrOptions(
                {"default", "half_circle", "quarter_circle", "eighth_circle"}
            )
        ],
    }
)
def plot_relationship(
    y_true,
    *y_preds,
    names=None,
    title=None,
    theta_offset=0.0,
    theta_scale="proportional",
    acov: Acov = "default",
    figsize=None,
    cmap="tab10",
    s=50,
    alpha=0.7,
    legend=True,
    show_grid=True,
    grid_props=None,
    color_palette=None,
    xlabel=None,
    ylabel=None,
    z_values=None,
    z_label=None,
    savefig=None,
    ax: Axes | None = None,
):
    # --- validate / clean
    y_true, *y_preds = drop_nan_in(y_true, *y_preds, error="raise")
    try:
        y_preds = [
            validate_yy(
                y_true, pred, expected_type="continuous", flatten=True
            )[1]
            for pred in y_preds
        ]
    except Exception as err:
        raise ValueError(
            "Validation failed. Please check your y_pred."
        ) from err

    # names
    n = len(y_preds)
    if names is None:
        names = [f"Model_{i + 1}" for i in range(n)]
    else:
        names = list(names)
        if len(names) < n:
            names += [f"Model_{i + 1}" for i in range(len(names), n)]
        elif len(names) > n:
            warnings.warn(
                f"Received {len(names)} names for {n} preds; "
                "extra names ignored.",
                UserWarning,
                stacklevel=2,
            )
            names = names[:n]

    # colors
    if color_palette is None:
        try:
            cmap_obj = get_cmap(cmap, default="tab10", failsafe="discrete")
            if hasattr(cmap_obj, "colors") and len(cmap_obj.colors) >= n:
                color_palette = cmap_obj.colors[:n]
            else:
                color_palette = [
                    (cmap_obj(i / max(1, n - 1)) if n > 1 else cmap_obj(0.5))
                    for i in range(n)
                ]
        except ValueError:
            warnings.warn(
                f"Invalid cmap '{cmap}'; using 'tab10'.",
                UserWarning,
                stacklevel=2,
            )
            color_palette = plt.cm.tab10.colors
    final_colors = [color_palette[i % len(color_palette)] for i in range(n)]

    # --- axes with requested angular coverage
    fig, ax, span = setup_polar_axes(ax, acov=acov, figsize=figsize or (8, 8))

    # map y_true -> theta
    y_true = np.asarray(y_true, dtype=float)
    if theta_scale == "proportional":
        if np.ptp(y_true) > 1e-9:
            theta = map_theta_to_span(
                y_true,
                span=span,
                data_min=float(np.min(y_true)),
                data_max=float(np.max(y_true)),
            )
        else:
            theta = np.zeros_like(y_true, dtype=float)
            warnings.warn(
                "y_true has zero range; mapping all angles to 0.",
                UserWarning,
                stacklevel=2,
            )
    elif theta_scale == "uniform":
        theta = np.linspace(0.0, float(span), len(y_true), endpoint=False)
    else:
        raise ValueError("`theta_scale` must be 'proportional' or 'uniform'.")

    # optional user offset (in radians)
    theta = theta + float(theta_offset)

    # plot each prediction series
    for i, y_pred in enumerate(y_preds):
        y_pred = np.asarray(y_pred, dtype=float)
        # normalize radius to [0, 1] per series
        if np.ptp(y_pred) > 1e-9:
            r = (y_pred - np.min(y_pred)) / np.ptp(y_pred)
        else:
            r = np.full_like(y_pred, 0.5)
            warnings.warn(
                f"Series '{names[i]}' has zero range; " "radius set to 0.5.",
                UserWarning,
                stacklevel=2,
            )

        ax.scatter(
            theta,
            r,
            label=names[i],
            color=final_colors[i],
            s=s,
            alpha=alpha,
            edgecolor="black",
        )

    # replace angle ticks with z_values if provided
    if z_values is not None:
        z_values = np.asarray(z_values)
        if len(z_values) != len(y_true):
            raise ValueError(
                "Length of `z_values` must match length of `y_true`."
            )
        k = min(len(z_values), 8)
        idx = np.linspace(0, len(z_values) - 1, k, dtype=int)
        ax.set_xticks(theta[idx])
        ax.set_xticklabels([f"{z_values[j]:.2g}" for j in idx])
        if z_label:
            ax.text(
                1.1,
                0.5,
                z_label,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="left",
            )

    # labels / grid / legend
    if z_values is None:
        ax.set_ylabel(ylabel or "Angular Mapping (θ)", labelpad=15)
    ax.set_xlabel(xlabel or "Normalized Predictions (r)", labelpad=15)
    ax.set_rlabel_position(22.5)
    ax.set_title(title or "Relationship Visualization", va="bottom", pad=20)
    set_axis_grid(ax, show_grid, grid_props=grid_props)
    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08))

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return ax


plot_relationship.__doc__ = r"""
Visualize the relationship between true values and one or more
prediction series on a polar (circular) scatter plot.

Each point uses an angular position derived from ``y_true`` and a
radial position derived from the corresponding prediction. This
compact view lets you compare multiple prediction series against the
same truth—useful for spotting systematic deviations and patterns
over a cyclic or ordered domain (e.g., phase, time-of-year).

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground-truth (observed) values. Must be numeric, 1D, and the
    same length as every array in ``y_preds``.

*y_preds : array-like(s)
    One or more prediction arrays, each with shape ``(n_samples,)``
    and aligned to ``y_true``.

names : list of str, optional
    Labels for each prediction series. If fewer names than series
    are provided, placeholders like ``'Model_3'`` are appended.

title : str, optional
    Figure title. If ``None``, uses ``'Relationship Visualization'``.

theta_offset : float, default=0
    Constant angular shift (radians) applied after the angle mapping.

theta_scale : {'proportional', 'uniform'}, default='proportional'
    Strategy for mapping ``y_true`` to angles:

    - ``'proportional'``: angle proportional to the scaled value of
      ``y_true`` within its range over the selected angular span.
    - ``'uniform'``: angles evenly spaced over the selected span,
      ignoring the numerical spacing in ``y_true``.

acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'},
    default='default'
    Angular coverage (span) of the plot:

    - ``'default'``: :math:`2\pi` (full circle)
    - ``'half_circle'``: :math:`\pi`
    - ``'quarter_circle'``: :math:`\tfrac{\pi}{2}`
    - ``'eighth_circle'``: :math:`\tfrac{\pi}{4}`

figsize : tuple of (float, float), optional
    Figure size in inches. If ``None``, a sensible default is used.

cmap : str, default='tab10'
    Matplotlib colormap name used to generate distinct series colors.

s : float, default=50
    Marker size for scatter points.

alpha : float, default=0.7
    Alpha (transparency) for scatter points in ``[0, 1]``.

legend : bool, default=True
    If ``True``, show a legend for the prediction series.

show_grid : bool, default=True
    Toggle polar grid lines (delegated to ``set_axis_grid``).

grid_props : dict, optional
    Keyword arguments forwarded to the grid helper (e.g., ``linestyle``,
    ``alpha``).

color_palette : list of color-like, optional
    Explicit list of colors. If omitted, colors are derived from
    ``cmap``. If provided with fewer colors than series, they repeat.

xlabel : str, optional
    Label for the radial axis. Defaults to
    ``'Normalized Predictions (r)'``.

ylabel : str, optional
    Label for the angular axis. Defaults to
    ``'Angular Mapping (θ)'`` when ``z_values`` is not used.

z_values : array-like of shape (n_samples,), optional
    Optional values used to label angular ticks (e.g., time, phase).
    If provided, a subset of positions is selected and tick labels
    are replaced by formatted entries from ``z_values``.

z_label : str, optional
    Axis/legend label describing ``z_values`` (shown as text next to
    the angular tick labels region).

savefig : str, optional
    Path to save the figure (with extension). If ``None``, the figure
    is shown instead.

Returns
-------
ax : matplotlib.axes.Axes
    The polar axes containing the visualization.

Notes
-----
**Angular span.** Let :math:`\Delta\theta` be the selected span:
:math:`2\pi` (full), :math:`\pi`, :math:`\pi/2`, or :math:`\pi/4`
depending on ``acov``. Angles are then limited to
:math:`[0,\,\Delta\theta]` and shifted by ``theta_offset``.

**Angle mapping.** For :math:`N=\text{len}(y_{\text{true}})` and
:math:`i=0,\dots,N-1`:

- Proportional mapping (range-aware):

  .. math::

     \theta_i \;=\;
     \begin{cases}
       \dfrac{y_i - y_{\min}}{y_{\max}-y_{\min}}\,\Delta\theta,
         & \text{if } y_{\max}>y_{\min},\\[6pt]
       0, & \text{otherwise,}
     \end{cases}

  where :math:`y_{\min}=\min_i y_i` and :math:`y_{\max}=\max_i y_i`.

- Uniform mapping (index-based):

  .. math::

     \theta_i \;=\; \frac{i}{N}\,\Delta\theta.

**Radial normalization.** Each prediction series :math:`p` is scaled
to :math:`[0,1]` by

.. math::

   r_i \;=\;
   \begin{cases}
     \dfrac{p_i - p_{\min}}{p_{\max}-p_{\min}}, & p_{\max}>p_{\min},\\[6pt]
     0.5, & \text{otherwise,}
   \end{cases}

to give comparable radii across heterogeneous series :footcite:p:`Hunter:2007`.

**Data preparation.** The function first removes joint NaNs via
``drop_nan_in`` and validates each pair ``(y_true, y_pred)`` through
``validate_yy`` (continuous expectations, 1D arrays). Colors are
drawn from ``cmap`` unless ``color_palette`` is supplied. Grid
appearance is managed by ``set_axis_grid``.

**Interpretation.** When ``theta_scale='proportional'``, nearby angles
reflect similar truth values; with ``'uniform'``, angles reflect order
only. Clustering by color (series) indicates systematic agreement or
disagreement versus truth across the domain :footcite:p:`kouadiob2025`.

Examples
--------
Basic comparison over a full circle:

>>> import numpy as np
>>> from kdiagram.plot.relationship import plot_relationship
>>> rng = np.random.default_rng(0)
>>> y = rng.random(200)
>>> p1 = y + rng.normal(0, 0.10, size=len(y))
>>> p2 = y + rng.normal(0, 0.20, size=len(y))
>>> ax = plot_relationship(
...     y, p1, p2,
...     names=["Model A", "Model B"],
...     acov="default",
...     title="Truth–Prediction (Full Circle)"
... )

Half-circle with custom angular tick labels (e.g., months):

>>> months = np.linspace(1, 12, len(y))
>>> ax = plot_relationship(
...     y, p1,
...     names=["Model A"],
...     theta_scale="uniform",
...     acov="half_circle",
...     z_values=months,
...     z_label="Month",
...     xlabel="Normalized Predictions (r)"
... )

See Also
--------
kdiagram.plot.uncertainty.plot_temporal_uncertainty :
    General polar series visualization (e.g., quantiles).
kdiagram.plot.uncertainty.plot_actual_vs_predicted :
    Side-by-side truth vs. point prediction comparison.

References
----------

.. footbibliography::
"""
