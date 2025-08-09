# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..compat.sklearn import StrOptions, validate_params
from ..utils.generic_utils import drop_nan_in
from ..utils.plot import set_axis_grid
from ..utils.validator import validate_yy

__all__=['plot_relationship']

@validate_params({
    "y_true": ['array-like'],
    "y_pred": ['array-like'],
    "theta_scale": [StrOptions({ 'proportional', 'uniform' })],
    'acov': [StrOptions({
        'default', 'half_circle', 'quarter_circle', 'eighth_circle'})],
    })
def plot_relationship(
    y_true, *y_preds,
    names=None,
    title=None,
    theta_offset=0,
    theta_scale='proportional',
    acov='default',
    figsize=None,
    cmap='tab10', 
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
    r"""
    Visualize the relationship between `y_true` and multiple `y_preds`
    using a circular or polar plot. The function allows flexible
    configurations such as angular coverage, z-values for replacing
    angle labels, and customizable axis labels.

    Parameters
    -------------
    y_true : array-like
        The true values. Must be numeric, one-dimensional, and of the
        same length as the values in `y_preds`.

    y_preds : array-like (one or more)
        Predicted values from one or more models. Each `y_pred` must
        have the same length as `y_true`.

    names : list of str, optional
        A list of model names corresponding to each `y_pred`. If not
        provided or if fewer names than predictions are given, the
        function assigns default names as ``"Model_1"``, ``"Model_2"``,
        etc. For instance, if `y_preds` has three predictions and
        `names` is `["SVC", "RF"]`, the names will be updated to
        `["SVC", "RF", "Model_3"]`.

    title : str, optional
        The title of the plot. If `None`, the title defaults to
        `"Relationship Visualization"`.

    theta_offset : float, default=0
        Angular offset in radians to rotate the plot. This allows
        customization of the orientation of the plot.

    theta_scale : {'proportional', 'uniform'}, default='proportional'
        Determines how `y_true` values are mapped to angular
        coordinates (`theta`):

        - ``'proportional'``: Maps `y_true` proportionally to the
          angular range (e.g., 0 to 360° or a subset defined by
          `acov`).
        - ``'uniform'``: Distributes `y_true` values uniformly around
          the angular range.

    acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'}, 
        default='default'

        Specifies the angular coverage of the plot:

        - ``'default'``: Full circle (360°).
        - ``'half_circle'``: Half circle (180°).
        - ``'quarter_circle'``: Quarter circle (90°).
        - ``'eighth_circle'``: Eighth circle (45°).
        The angular span is automatically restricted to the selected
        portion of the circle.

    figsize : tuple of float, default=(8, 8)
        The dimensions of the figure in inches.

    cmap : str, default='viridis'
        Colormap for the scatter points. Refer to Matplotlib
        documentation for a list of supported colormaps.

    s : float, default=50
        Size of scatter points representing predictions.

    alpha : float, default=0.7
        Transparency level for scatter points. Valid values range
        from 0 (completely transparent) to 1 (fully opaque).

    legend : bool, default=True
        Whether to display a legend indicating the model names.

    show_grid : bool, default=True
        Whether to display a grid on the polar plot.

    color_palette : list of str, optional
        A list of colors to use for the scatter points. If not
        provided, the default Matplotlib color palette (`tab10`) is
        used.

    xlabel : str, optional
        Label for the radial axis (distance from the center). Defaults
        to `"Normalized Predictions (r)"`.

    ylabel : str, optional
        Label for the angular axis (theta values). Defaults to
        `"Angular Mapping (θ)"`.

    z_values : array-like, optional
        Optional values to replace the angular labels. The length of
        `z_values` must match the length of `y_true`. If provided, the
        angular labels are replaced by the scaled `z_values`.

    z_label : str, optional
        Label for the `z_values`, if provided. Defaults to `None`.

    Notes
    -------
    The function dynamically maps `y_true` to angular coordinates
    based on the `theta_scale` and `acov` parameters [1]_. The `y_preds`
    are normalized to radial coordinates between 0 and 1 [2]_. Optionally,
    `z_values` can replace angular labels with custom values [3]_.

    .. math::
        \theta = 
        \begin{cases} 
        \text{Proportional mapping: } \theta_i = 
        \frac{y_{\text{true}_i} - \min(y_{\text{true}})}
        {\max(y_{\text{true}}) - \min(y_{\text{true}})} 
        \cdot \text{angular_range} \\
        \text{Uniform mapping: } \theta_i = 
        \frac{i}{N-1} \cdot \text{angular_range}
        \end{cases}

    Radial normalization:

    .. math::
        r_i = \frac{y_{\text{pred}_i} - \min(y_{\text{pred}})}
        {\max(y_{\text{pred}}) - \min(y_{\text{pred}})}

    Examples
    ----------

    >>> from kdiagram.plot.relationship import plot_relationship
    >>> import numpy as np
    >>> # Create sample data
    >>> y_true = np.random.rand(100)
    >>> y_pred1 = y_true + np.random.normal(0, 0.1, size=100)
    >>> y_pred2 = y_true + np.random.normal(0, 0.2, size=100)
    >>> # Full circle visualization
    >>> plot_relationship(
    ...     y_true, y_pred1, y_pred2,
    ...     names=["Model A", "Model B"],
    ...     acov="default",
    ...     title="Full Circle Visualization"
    ... )
    >>> # Half-circle visualization with z-values
    >>> z_values = np.linspace(0, 100, len(y_true))
    >>> plot_relationship(
    ...     y_true, y_pred1, 
    ...     names=["Model A"],
    ...     acov="half_circle",
    ...     z_values=z_values,
    ...     xlabel="Predicted Values",
    ...     ylabel="Custom Angles"
    ... )

    See Also
    ----------
    matplotlib.pyplot.polar : Polar plotting in Matplotlib.
    numpy.linspace : Uniformly spaced numbers.

    References
    ------------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
           Computing in Science & Engineering, 9(3), 90-95.
    .. [2] NumPy Documentation: https://numpy.org/doc/stable/
    .. [3] Matplotlib Documentation: https://matplotlib.org/stable/
    """

    # Remove NaN values from y_true and all y_pred arrays
    y_true, *y_preds = drop_nan_in(y_true, *y_preds, error='raise')

    # Validate y_true and each y_pred to ensure consistency and continuity
    try: 
        y_preds = [
            validate_yy(y_true, pred, expected_type="continuous", flatten=True)[1]
            for pred in y_preds
        ]
    except Exception as e: 
        raise ValueError (f"Validation failed, due to {e}. Please check your y_pred")

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
            warnings.warn(f"Received {len(names)} names for {num_preds}"
                          f" predictions. Extra names ignored.", UserWarning)
            names = names[:num_preds]

    # --- Color Handling ---
    if color_palette is None:
        # Generate colors from cmap if palette not given
        try:
            cmap_obj = plt.get_cmap(cmap)
            # Sample enough distinct colors
            if hasattr(cmap_obj, 'colors') and len(cmap_obj.colors) >= num_preds:
                 # Use colors directly from discrete map if enough
                 color_palette = cmap_obj.colors[:num_preds]
            else: # Sample from continuous map or discrete map with fewer colors
                 color_palette = [cmap_obj(i / max(1, num_preds-1)) if num_preds > 1
                                  else cmap_obj(0.5) for i in range(num_preds)]
        except ValueError:
            warnings.warn(f"Invalid cmap '{cmap}'. Falling back to 'tab10'.")
            color_palette = plt.cm.tab10.colors # Default palette
    # Ensure palette has enough colors, repeat if necessary
    final_colors = [color_palette[i % len(color_palette)] for i in range(num_preds)]


    # Determine the angular range based on `acov`
    if acov == 'default':
        angular_range = 2 * np.pi
    elif acov == 'half_circle':
        angular_range = np.pi
    elif acov == 'quarter_circle':
        angular_range = np.pi / 2
    elif acov == 'eighth_circle':
        angular_range = np.pi / 4
    else:
        # This case should be caught by @validate_params, but keep as safeguard
        raise ValueError(
            "Invalid value for `acov`. Choose from 'default',"
            " 'half_circle', 'quarter_circle', or 'eighth_circle'.")

    # Create the polar plot
    fig, ax = plt.subplots(figsize=figsize or (8, 8), # Provide default here
                           subplot_kw={'projection': 'polar'})

    # Limit the visible angular range
    ax.set_thetamin(0)  # Start angle (in degrees)
    ax.set_thetamax(np.degrees(angular_range))  # End angle (in degrees)

    # Map `y_true` to angular coordinates (theta)
    # Handle potential division by zero if y_true is constant
    y_true_range = np.ptp(y_true) # Peak-to-peak range
    if theta_scale == 'proportional':
        if y_true_range > 1e-9: # Avoid division by zero
             theta = angular_range * (y_true - np.min(y_true)) / y_true_range
        else: # Handle constant y_true case - map all to start angle?
             theta = np.zeros_like(y_true)
             warnings.warn("y_true has zero range. Mapping all points to angle 0"
                           " with 'proportional' scaling.", UserWarning)
    elif theta_scale == 'uniform':
        # linspace handles len=1 case correctly
        theta = np.linspace(0, angular_range, len(y_true), endpoint=False) 
    else:
        # This case should be caught by @validate_params
        raise ValueError(
            "`theta_scale` must be either 'proportional' or 'uniform'.")

    # Apply theta offset
    theta += theta_offset

    # Plot each model's predictions
    for i, y_pred in enumerate(y_preds):
        # Ensure `y_pred` is a numpy array
        y_pred = np.asarray(y_pred, dtype=float) # Convert early

        # Normalize `y_pred` for radial coordinates
        # Handle potential division by zero if y_pred is constant
        y_pred_range = np.ptp(y_pred)
        if y_pred_range > 1e-9:
            r = (y_pred - np.min(y_pred)) / y_pred_range
        else:
            # If constant, map all to 0.5 radius (midpoint)? Or 0? Let's use 0.5
            r = np.full_like(y_pred, 0.5)
            warnings.warn(f"Prediction series '{names[i]}' has zero range."
                          f" Plotting all its points at normalized radius 0.5.",
                          UserWarning)

        # Plot on the polar axis
        ax.scatter(
            theta, r,
            label=names[i],
            color=final_colors[i], # FIX: Use 'color' instead of 'c'
            s=s, alpha=alpha, edgecolor='black'
        )

    # If z_values are provided, replace angle labels with z_values
    if z_values is not None:
        z_values = np.asarray(z_values) # Ensure numpy array
        if len(z_values) != len(y_true):
            raise ValueError("Length of `z_values` must match the length of `y_true`.")

        # Decide number of ticks, e.g., 5-10 depending on range/preference
        num_z_ticks = min(len(z_values), 8) # Example: max 8 ticks
        tick_indices = np.linspace(0, len(z_values) - 1, num_z_ticks,
                                   dtype=int, endpoint=True)

        # Get theta values corresponding to these indices
        theta_ticks = theta[tick_indices] # Use theta calculated earlier
        z_tick_labels = [f"{z_values[ix]:.2g}" for ix in tick_indices] # Format labels

        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(z_tick_labels)
        # Optional: Set label for z-axis if z_label is provided
        if z_label:
             ax.text(1.1, 0.5, z_label, transform=ax.transAxes, rotation=90,
                     va='center', ha='left') # Adjust position as needed


    # Add labels for radial and angular axes (only if z_values are not used for angles)
    if z_values is None:
        ax.set_ylabel(ylabel or "Angular Mapping (θ)", labelpad=15) # Use labelpad
    # Radial label
    ax.set_xlabel(xlabel or "Normalized Predictions (r)", labelpad=15)
    # Position radial labels better
    ax.set_rlabel_position(22.5) # Adjust angle for radial labels

    # Add title
    ax.set_title(title or "Relationship Visualization", va='bottom', pad=20) # Add padding

    # Add grid using helper or directly
    set_axis_grid(ax, show_grid, grid_props=grid_props)

    # Add legend
    if legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1)) # Adjust position

    plt.tight_layout() # Adjust layout to prevent overlap

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        # Warning for non-GUI backend is expected here in test envs
        plt.show()

