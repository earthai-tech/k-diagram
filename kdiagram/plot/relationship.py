# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..compat.matplotlib import get_cmap
from ..compat.sklearn import StrOptions, validate_params
from ..utils.generic_utils import drop_nan_in
from ..utils.plot import set_axis_grid
from ..utils.validator import validate_yy

__all__ = ["plot_relationship"]


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "theta_scale": [StrOptions({"proportional", "uniform"})],
        "acov": [
            StrOptions({"default", "half_circle", "quarter_circle", "eighth_circle"})
        ],
    }
)
def plot_relationship(
    y_true,
    *y_preds,
    names=None,
    title=None,
    theta_offset=0,
    theta_scale="proportional",
    acov="default",
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
):
    # Remove NaN values from y_true and all y_pred arrays
    y_true, *y_preds = drop_nan_in(y_true, *y_preds, error="raise")

    # Validate y_true and each y_pred to ensure consistency and continuity
    try:
        y_preds = [
            validate_yy(y_true, pred, expected_type="continuous", flatten=True)[1]
            for pred in y_preds
        ]
    except Exception as err:
        raise ValueError("Validation failed. Please check your y_pred") from err

    # Generate default model names if none are provided
    num_preds = len(y_preds)
    if names is None:
        names = [f"Model_{i+1}" for i in range(num_preds)]
    else:
        # Ensure names is a list
        names = list(names)
        # Ensure the length of names matches y_preds
        if len(names) < num_preds:
            names += [f"Model_{i+1}" for i in range(len(names), num_preds)]
        elif len(names) > num_preds:
            warnings.warn(
                f"Received {len(names)} names for {num_preds}"
                f" predictions. Extra names ignored.",
                UserWarning,
                stacklevel=2,
            )
            names = names[:num_preds]

    # --- Color Handling ---
    if color_palette is None:
        # Generate colors from cmap if palette not given
        try:
            cmap_obj = get_cmap(cmap, default="tab10", failsafe="discrete")
            # Sample enough distinct colors
            if hasattr(cmap_obj, "colors") and len(cmap_obj.colors) >= num_preds:
                # Use colors directly from discrete map if enough
                color_palette = cmap_obj.colors[:num_preds]
            else:  # Sample from continuous map or discrete map with fewer colors
                color_palette = [
                    (
                        cmap_obj(i / max(1, num_preds - 1))
                        if num_preds > 1
                        else cmap_obj(0.5)
                    )
                    for i in range(num_preds)
                ]
        except ValueError:
            warnings.warn(
                f"Invalid cmap '{cmap}'. Falling back to 'tab10'.", stacklevel=2
            )
            color_palette = plt.cm.tab10.colors  # Default palette
    # Ensure palette has enough colors, repeat if necessary
    final_colors = [color_palette[i % len(color_palette)] for i in range(num_preds)]

    # Determine the angular range based on `acov`
    if acov == "default":
        angular_range = 2 * np.pi
    elif acov == "half_circle":
        angular_range = np.pi
    elif acov == "quarter_circle":
        angular_range = np.pi / 2
    elif acov == "eighth_circle":
        angular_range = np.pi / 4
    else:
        # This case should be caught by @validate_params, but keep as safeguard
        raise ValueError(
            "Invalid value for `acov`. Choose from 'default',"
            " 'half_circle', 'quarter_circle', or 'eighth_circle'."
        )

    # Create the polar plot
    fig, ax = plt.subplots(
        figsize=figsize or (8, 8),  # Provide default here
        subplot_kw={"projection": "polar"},
    )

    # Limit the visible angular range
    ax.set_thetamin(0)  # Start angle (in degrees)
    ax.set_thetamax(np.degrees(angular_range))  # End angle (in degrees)

    # Map `y_true` to angular coordinates (theta)
    # Handle potential division by zero if y_true is constant
    y_true_range = np.ptp(y_true)  # Peak-to-peak range
    if theta_scale == "proportional":
        if y_true_range > 1e-9:  # Avoid division by zero
            theta = angular_range * (y_true - np.min(y_true)) / y_true_range
        else:  # Handle constant y_true case - map all to start angle?
            theta = np.zeros_like(y_true)
            warnings.warn(
                "y_true has zero range. Mapping all points to angle 0"
                " with 'proportional' scaling.",
                UserWarning,
                stacklevel=2,
            )
    elif theta_scale == "uniform":
        # linspace handles len=1 case correctly
        theta = np.linspace(0, angular_range, len(y_true), endpoint=False)
    else:
        # This case should be caught by @validate_params
        raise ValueError("`theta_scale` must be either 'proportional' or 'uniform'.")

    # Apply theta offset
    theta += theta_offset

    # Plot each model's predictions
    for i, y_pred in enumerate(y_preds):
        # Ensure `y_pred` is a numpy array
        y_pred = np.asarray(y_pred, dtype=float)  # Convert early

        # Normalize `y_pred` for radial coordinates
        # Handle potential division by zero if y_pred is constant
        y_pred_range = np.ptp(y_pred)
        if y_pred_range > 1e-9:
            r = (y_pred - np.min(y_pred)) / y_pred_range
        else:
            # If constant, map all to 0.5 radius (midpoint)? Or 0? Let's use 0.5
            r = np.full_like(y_pred, 0.5)
            warnings.warn(
                f"Prediction series '{names[i]}' has zero range."
                f" Plotting all its points at normalized radius 0.5.",
                UserWarning,
                stacklevel=2,
            )

        # Plot on the polar axis
        ax.scatter(
            theta,
            r,
            label=names[i],
            color=final_colors[i],
            s=s,
            alpha=alpha,
            edgecolor="black",
        )

    # If z_values are provided, replace angle labels with z_values
    if z_values is not None:
        z_values = np.asarray(z_values)  # Ensure numpy array
        if len(z_values) != len(y_true):
            raise ValueError("Length of `z_values` must match the length of `y_true`.")

        # Decide number of ticks, e.g., 5-10 depending on range/preference
        num_z_ticks = min(len(z_values), 8)  # Example: max 8 ticks
        tick_indices = np.linspace(
            0, len(z_values) - 1, num_z_ticks, dtype=int, endpoint=True
        )

        # Get theta values corresponding to these indices
        theta_ticks = theta[tick_indices]  # Use theta calculated earlier
        z_tick_labels = [f"{z_values[ix]:.2g}" for ix in tick_indices]  # Format labels

        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(z_tick_labels)
        # Optional: Set label for z-axis if z_label is provided
        if z_label:
            ax.text(
                1.1,
                0.5,
                z_label,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="left",
            )  # Adjust position as needed

    # Add labels for radial and angular axes (only if z_values are not used for angles)
    if z_values is None:
        ax.set_ylabel(ylabel or "Angular Mapping (θ)", labelpad=15)  # Use labelpad
    # Radial label
    ax.set_xlabel(xlabel or "Normalized Predictions (r)", labelpad=15)
    # Position radial labels better
    ax.set_rlabel_position(22.5)  # Adjust angle for radial labels

    # Add title
    ax.set_title(
        title or "Relationship Visualization", va="bottom", pad=20
    )  # Add padding

    # Add grid using helper or directly
    set_axis_grid(ax, show_grid, grid_props=grid_props)

    # Add legend
    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))  # Adjust position

    plt.tight_layout()  # Adjust layout to prevent overlap

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        # Warning for non-GUI backend is expected here in test envs
        plt.show()


plot_relationship.__doc__=r"""
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
