# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

import warnings
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from ..compat.matplotlib import get_cmap
from ..decorators import check_non_emptiness, isdf
from ..utils.plot import set_axis_grid
from ..utils.validator import exist_features

__all__ = ["plot_error_ellipses", "plot_error_bands", "plot_error_violins"]


@check_non_emptiness
@isdf
def plot_error_violins(
    df: pd.DataFrame,
    *error_cols: str,
    names: Optional[list[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (9, 9),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
    **violin_kws,
):
    if not error_cols:
        raise ValueError("At least one error column must be provided.")
    exist_features(df, features=list(error_cols))

    if names and len(names) != len(error_cols):
        warnings.warn(
            f"Number of names ({len(names)}) does not match number of "
            f"error columns ({len(error_cols)}). Using default names.",
            UserWarning,
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i+1}" for i in range(len(error_cols))]

    # Prepare data and KDEs for each model
    violin_data = []
    all_errors = np.concatenate([df[col].dropna().to_numpy() for col in error_cols])
    r_min, r_max = all_errors.min(), all_errors.max()
    grid = np.linspace(r_min, r_max, 200)

    for col in error_cols:
        errors = df[col].dropna().to_numpy()
        if len(errors) < 2:
            violin_data.append(None)  # Cannot compute KDE
            continue

        kde = gaussian_kde(errors)
        density = kde(grid)
        violin_data.append(density / density.max())  # Normalize density

    # Plot setup
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    num_violins = len(error_cols)
    angles = np.linspace(0, 2 * np.pi, num_violins, endpoint=False)
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0, 1, num_violins))

    # Draw violins
    for i, (angle, density) in enumerate(zip(angles, violin_data)):
        if density is None:
            continue

        # Width of the violin slice
        width = (2 * np.pi / num_violins) * 0.8

        # Create the path for the violin polygon
        x = np.concatenate([-density * width / 2, np.flip(density * width / 2)])
        y = np.concatenate([grid, np.flip(grid)])

        # Rotate and translate path to the correct angle
        theta = x + angle
        r = y

        ax.fill(
            theta,
            r,
            color=colors[i],
            label=names[i],
            alpha=violin_kws.pop("alpha", 0.6),
            **violin_kws,
        )

    # Add zero-error reference line
    ax.plot(
        np.linspace(0, 2 * np.pi, 100),
        np.zeros(100),
        color="black",
        linestyle="--",
        lw=1.5,
        label="Zero Error",
    )

    ax.set_title(title or "Comparison of Error Distributions")
    ax.set_yticklabels([])  # Hide radial ticks for clarity
    ax.set_xticks(angles)
    ax.set_xticklabels(names)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


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
    theta_period: Optional[float] = None,
    theta_bins: int = 24,
    n_std: float = 1.0,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    mask_angle: bool = False,
    savefig: Optional[str] = None,
    dpi: int = 300,
    **fill_kws,
):
    exist_features(df, features=[error_col, theta_col])

    data = df[[error_col, theta_col]].dropna()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs in required columns.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if theta_period:
        data["theta_rad"] = (
            ((data[theta_col] % theta_period) / theta_period) * 2 * np.pi
        )
    else:
        min_theta, max_theta = data[theta_col].min(), data[theta_col].max()
        if (max_theta - min_theta) > 1e-9:
            data["theta_rad"] = (
                ((data[theta_col] - min_theta) / (max_theta - min_theta)) * 2 * np.pi
            )
        else:
            data["theta_rad"] = 0

    # Bin the data by angle
    theta_edges = np.linspace(0, 2 * np.pi, theta_bins + 1)
    theta_labels = (theta_edges[:-1] + theta_edges[1:]) / 2
    data["theta_bin"] = pd.cut(
        data["theta_rad"], bins=theta_edges, labels=theta_labels, include_lowest=True
    )

    # Calculate stats per bin
    stats = data.groupby("theta_bin")[error_col].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0)  # Handle bins with one sample

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    # Plot the mean error line
    ax.plot(stats["theta_bin"], stats["mean"], color="black", lw=2, label="Mean Error")

    # Create and plot the uncertainty band
    ax.fill_between(
        stats["theta_bin"],
        stats["mean"] - n_std * stats["std"],
        stats["mean"] + n_std * stats["std"],
        alpha=fill_kws.pop("alpha", 0.3),
        label=f"{n_std} Std. Dev. Band",
        **fill_kws,
    )

    # Add a zero-error reference line
    ax.axhline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")

    ax.set_title(title or f"Error Distribution vs. {theta_col}")
    ax.set_ylabel(f"Forecast Error ({error_col})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_angle:
        ax.set_xticklabels([])

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return ax


plot_error_bands.__doc__ = r"""
Plot polar error bands to visualize systemic vs. random error.

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
    color_col: Optional[str] = None,
    n_std: float = 2.0,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    mask_angle: bool = False,
    mask_radius: bool = False,
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
    **ellipse_kws,
):
    required = [r_col, theta_col, r_std_col, theta_std_col]
    if color_col:
        required.append(color_col)
    exist_features(df, features=required)

    data = df[required].dropna()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs in "
            "required columns. Cannot plot.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if color_col:
        color_data = data[color_col].to_numpy()
        cbar_label = color_col
    else:
        # Default color to radial uncertainty
        color_data = data[r_std_col].to_numpy()
        cbar_label = f"Uncertainty ({r_std_col})"

    norm = Normalize(vmin=np.min(color_data), vmax=np.max(color_data))
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(norm(color_data))

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    # Plot each ellipse as a filled path
    for i, row in data.iterrows():
        theta_path, r_path = _get_ellipse_path(
            r_mean=row[r_col],
            theta_mean=row[theta_col],
            r_std=row[r_std_col],
            theta_std=row[theta_std_col],
            n_std=n_std,
        )
        ax.fill(theta_path, r_path, color=colors[i], **ellipse_kws)

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, pad=0.1, shrink=0.75
    )
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_title(title or f"Error Ellipses ({n_std:.1f} std. dev.)")
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
    if mask_angle:
        ax.set_xticklabels([])

    if mask_radius:
        ax.set_yticklabels([])

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _get_ellipse_path(r_mean, theta_mean, r_std, theta_std, n_std=2.0):
    """
    Helper to calculate the path of an ellipse in polar coordinates.
    The ellipse is defined in a local Cartesian frame and then
    transformed.
    """
    # Width (radial) and height (tangential) of the ellipse
    width = n_std * r_std
    height = n_std * (r_mean * np.sin(theta_std))

    # Center of the ellipse in Cartesian coordinates
    x_c = r_mean * np.cos(theta_mean)
    y_c = r_mean * np.sin(theta_mean)

    # Generate points on a standard ellipse
    t = np.linspace(0, 2 * np.pi, 100)
    x_local = (width / 2) * np.cos(t)
    y_local = (height / 2) * np.sin(t)

    # Rotation matrix to align ellipse with the radial direction
    R = np.array(
        [
            [np.cos(theta_mean), -np.sin(theta_mean)],
            [np.sin(theta_mean), np.cos(theta_mean)],
        ]
    )

    # Rotate and translate local points
    x_rotated, y_rotated = np.dot(R, [x_local, y_local])
    x_final = x_rotated + x_c
    y_final = y_rotated + y_c

    # Convert final Cartesian points back to polar
    r_path = np.sqrt(x_final**2 + y_final**2)
    theta_path = np.arctan2(y_final, x_final)

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
