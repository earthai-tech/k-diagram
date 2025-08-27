import warnings
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from scipy.stats import probplot

from ..compat.matplotlib import get_cmap
from ..decorators import check_non_emptiness, isdf
from ..utils.deps import ensure_pkg
from ..utils.generic_utils import get_valid_kwargs
from ..utils.handlers import columns_manager
from ..utils.plot import set_axis_grid
from ..utils.validator import exist_features

__all__ = [
    "plot_time_series",
    "plot_scatter_correlation",
    "plot_error_autocorrelation",
    "plot_qq",
    "plot_error_pacf",
    "plot_error_distribution",
]


@isdf
@check_non_emptiness(params=["df"])
def plot_time_series(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    actual_col: Optional[str] = None,
    pred_cols: Optional[list[str]] = None,
    names: Optional[list[str]] = None,
    q_lower_col: Optional[str] = None,
    q_upper_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple[float, float] = (12, 6),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    pred_cols = columns_manager(pred_cols, empty_as_none=False)

    if not actual_col and not pred_cols:
        raise ValueError(
            "At least one of `actual_col`" " or `pred_cols` must be provided."
        )

    required_cols = []
    if x_col:
        required_cols.append(x_col)
    if actual_col:
        required_cols.append(actual_col)
    if q_lower_col:
        required_cols.append(q_lower_col)
    if q_upper_col:
        required_cols.append(q_upper_col)
    if pred_cols:
        required_cols.extend(pred_cols)

    exist_features(df, features=required_cols)

    # Use index if x_col is not provided
    x_data = df.index if x_col is None else df[x_col]

    # Handle names for the legend
    num_preds = len(pred_cols)
    if names and len(names) != num_preds:
        warnings.warn(
            "Length of `names` does not match"
            " `pred_cols`. Using defaults.",
            stacklevel=2,
        )
        names = None
    if not names:
        names = [col for col in pred_cols]

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0, 1, num_preds)) if num_preds > 0 else []

    # --- Plot Uncertainty Band (if provided) ---
    if q_lower_col and q_upper_col:
        ax.fill_between(
            x_data,
            df[q_lower_col],
            df[q_upper_col],
            color="gray",
            alpha=0.2,
            label="Uncertainty Interval",
        )

    # --- Plot Actual Values ---
    if actual_col:
        ax.plot(
            x_data, df[actual_col], color="black", linewidth=2, label="Actual"
        )

    # --- Plot Predicted Values ---
    for i, pred_col in enumerate(pred_cols):
        ax.plot(
            x_data,
            df[pred_col],
            color=colors[i],
            linestyle="--",
            linewidth=1.5,
            label=names[i],
        )

    # --- Formatting ---
    ax.set_title(title or "Time Series Forecast", fontsize=16)
    ax.set_xlabel(xlabel or (x_col if x_col else "Index"), fontsize=12)
    ax.set_ylabel(ylabel or "Value", fontsize=12)
    ax.legend()
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    fig.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_time_series.__doc__ = r"""
Plots one or more time series from a DataFrame.

This function creates a standard time series plot, which is a
fundamental tool for visualizing and comparing actual observed
values against one or more model forecasts over time. It serves
as an essential first-look diagnostic in any forecasting workflow.

More details in :ref:`Time Series Plot User Guide <ug_plot_time_series>`

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the time series data.
x_col : str, optional
    The name of the column to use for the x-axis (e.g., a
    datetime column). If ``None``, the DataFrame's index is used.
actual_col : str, optional
    The name of the column containing the true observed values.
    This is typically plotted as a solid reference line.
pred_cols : list of str, optional
    A list of one or more column names containing the point
    forecasts from different models.
names : list of str, optional
    Display names for each of the prediction series, to be used
    in the legend.
q_lower_col : str, optional
    The name of the column for the lower bound of a prediction
    interval. If provided with ``q_upper_col``, a shaded
    uncertainty band will be drawn.
q_upper_col : str, optional
    The name of the column for the upper bound of a prediction
    interval.
title : str, optional
    The title for the plot.
xlabel : str, optional
    The label for the x-axis.
ylabel : str, optional
    The label for the y-axis.
figsize : tuple of (float, float), default=(12, 6)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign unique colors to the different
    prediction series.
show_grid : bool, default=True
    Toggle the visibility of the plot's grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

See Also
--------
plot_scatter_correlation : A Cartesian plot for correlation.
plot_actual_vs_predicted : A polar plot for comparing true vs. predicted.
:ref:`userguide_context` : The user guide for contextual plots.

Notes
-----
This function provides a direct visualization of time-dependent
variables by mapping a time-like variable to the x-axis and the
series values to the y-axis. It is a foundational plot for
assessing a model's ability to track trends and seasonality.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.context import plot_time_series
>>>
>>> # Generate synthetic time series data
>>> np.random.seed(0)
>>> n_samples = 100
>>> time = pd.date_range("2024-01-01", periods=n_samples, freq='D')
>>> y_true = 50 + np.linspace(0, 10, n_samples) + \
...          5 * np.sin(np.arange(n_samples) * 2 * np.pi / 15)
>>>
>>> y_pred = y_true + np.random.normal(0, 1.5, n_samples)
>>> df = pd.DataFrame({
...     'time': time,
...     'actual': y_true,
...     'forecast': y_pred,
...     'q10': y_pred - 3,
...     'q90': y_pred + 3,
... })
>>>
>>> # Generate the plot
>>> ax = plot_time_series(
...     df,
...     x_col='time',
...     actual_col='actual',
...     pred_cols=['forecast'],
...     q_lower_col='q10',
...     q_upper_col='q90',
...     title="Forecast vs. Actuals with 80% Uncertainty"
... )
"""


@isdf
@check_non_emptiness(params=["df"])
def plot_scatter_correlation(
    df: pd.DataFrame,
    actual_col: str,
    pred_cols: list[str],
    names: Optional[list[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    s: int = 50,
    alpha: float = 0.7,
    show_identity_line: bool = True,
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    # --- Input Validation and Preparation ---
    pred_cols = columns_manager(pred_cols, empty_as_none=False)
    if not pred_cols:
        raise ValueError(
            "At least one prediction column"
            " (`pred_cols`) must be provided."
        )

    required_cols = [actual_col] + list(pred_cols)
    exist_features(df, features=required_cols)

    data_to_plot = df[required_cols].dropna()

    actual_data = data_to_plot[actual_col]

    num_preds = len(pred_cols)
    if names and len(names) != num_preds:
        warnings.warn(
            "Length of `names` does not match"
            " `pred_cols`. Using defaults.",
            stacklevel=2,
        )
        names = None
    if not names:
        names = [col for col in pred_cols]

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0, 1, num_preds)) if num_preds > 0 else []

    # --- Plot Identity Line (y=x) ---
    if show_identity_line:
        min_val = min(
            actual_data.min(), data_to_plot[list(pred_cols)].min().min()
        )
        max_val = max(
            actual_data.max(), data_to_plot[list(pred_cols)].max().max()
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.7,
            label="Identity Line",
        )

    # --- Plot Scatter for Each Prediction ---
    for i, pred_col in enumerate(pred_cols):
        ax.scatter(
            actual_data,
            data_to_plot[pred_col],
            color=colors[i],
            s=s,
            alpha=alpha,
            label=names[i],
        )

    # --- Formatting ---
    ax.set_title(title or "Actual vs. Predicted", fontsize=16)
    ax.set_xlabel(xlabel or f"True Values ({actual_col})", fontsize=12)
    ax.set_ylabel(ylabel or "Predicted Values", fontsize=12)
    ax.legend()
    ax.axis("equal")  # Ensure a square aspect ratio
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    fig.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_scatter_correlation.__doc__ = r"""
Plots a scatter plot of true vs. predicted values.

This function creates a classic Cartesian scatter plot to
visualize the relationship between true observed values and model
predictions. It is an essential tool for assessing linear
correlation, identifying systemic bias, and spotting outliers.

For more details, refer to
:ref:`Scatter Correlation Plot User Guide <ug_plot_scatter_correlation>`

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted values.
actual_col : str
    The name of the column containing the true observed values,
    which will be plotted on the x-axis.
pred_cols : list of str
    A list of one or more column names containing the point
    forecasts from different models.
names : list of str, optional
    Display names for each of the prediction series, to be used
    in the legend.
title : str, optional
    The title for the plot.
xlabel : str, optional
    The label for the x-axis.
ylabel : str, optional
    The label for the y-axis.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign unique colors to the different
    prediction series.
s : int, default=50
    The size of the scatter plot markers.
alpha : float, default=0.7
    The transparency of the markers.
show_identity_line : bool, default=True
    If ``True``, draws a dashed y=x line, which represents a
    perfect forecast.
show_grid : bool, default=True
    Toggle the visibility of the plot's grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

See Also
--------
plot_relationship : A polar version of this plot.
plot_error_relationship : A plot to diagnose error patterns.

Notes
-----
This plot directly visualizes the relationship between two
variables by plotting each observation :math:`i` as a point
:math:`(y_{true,i}, y_{pred,i})`.

The primary reference is the **identity line**, defined by the
equation:

.. math::

   y = x

For a perfect forecast, every predicted value would equal its
corresponding true value, and all points would fall exactly on
this line. Deviations from this line represent prediction errors.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.context import plot_scatter_correlation
>>>
>>> # Generate synthetic data
>>> np.random.seed(0)
>>> n_samples = 100
>>> y_true = np.linspace(0, 50, n_samples)
>>> y_pred_good = y_true + np.random.normal(0, 3, n_samples)
>>> y_pred_biased = y_true * 0.8 + 5
>>>
>>> df = pd.DataFrame({
...     'actual': y_true,
...     'good_model': y_pred_good,
...     'biased_model': y_pred_biased,
... })
>>>
>>> # Generate the plot
>>> ax = plot_scatter_correlation(
...     df,
...     actual_col='actual',
...     pred_cols=['good_model', 'biased_model'],
...     names=['Good Model', 'Biased Model'],
...     title="Actual vs. Predicted Correlation"
... )
"""


@isdf
@check_non_emptiness(params=["df"])
def plot_error_autocorrelation(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
    **acf_kwargs,
):
    # --- Input Validation and Preparation ---
    required_cols = [actual_col, pred_col]
    exist_features(df, features=required_cols)

    data_to_plot = df[required_cols].dropna()
    errors = data_to_plot[actual_col] - data_to_plot[pred_col]

    if len(errors) < 2:
        warnings.warn(
            "Not enough data points to plot autocorrelation.", stacklevel=2
        )
        return None

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- Generate ACF Plot ---
    autocorrelation_plot(errors, ax=ax, **acf_kwargs)

    # --- Formatting ---
    ax.set_title(title or "Autocorrelation of Forecast Errors", fontsize=16)
    ax.set_xlabel(xlabel or "Lag", fontsize=12)
    ax.set_ylabel(ylabel or "Autocorrelation", fontsize=12)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    fig.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_error_autocorrelation.__doc__ = r"""
Plots the autocorrelation of forecast errors.

This function creates an Autocorrelation Function (ACF) plot of
the forecast errors (residuals). It is a critical diagnostic
for time series models, used to check if there is any remaining
temporal structure (i.e., patterns) in the residuals. A well-
specified model should have errors that are uncorrelated over
time, behaving like random noise.

Additional details can be found in 
:ref:`Error Distribution Plot User Guide <ug_plot_error_distribution>`

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted values.
actual_col : str
    The name of the column containing the true observed values.
pred_col : str
    The name of the column containing the point forecast values.
title : str, optional
    The title for the plot.
xlabel : str, optional
    The label for the x-axis.
ylabel : str, optional
    The label for the y-axis.
figsize : tuple of (float, float), default=(10, 5)
    The figure size in inches.
show_grid : bool, default=True
    Toggle the visibility of the plot's grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.
**acf_kwargs
    Additional keyword arguments passed directly to the underlying
    ``pandas.plotting.autocorrelation_plot`` function.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

See Also
--------
plot_error_pacf : The companion plot for partial autocorrelation.

Notes
-----
The Autocorrelation Function (ACF) at lag :math:`k` measures the
correlation between a time series and its own past values. For a
series of errors :math:`e_t`, the ACF is defined as:

.. math::

   \rho_k = \frac{\text{Cov}(e_t, e_{t-k})}{\text{Var}(e_t)}

This plot displays the values of :math:`\rho_k` for a range of
different lags :math:`k`. The plot also includes significance
bands (typically at 95% and 99% confidence), which provide a
threshold for determining if a correlation is statistically
significant or likely due to random chance.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.context import plot_error_autocorrelation
>>>
>>> # Generate synthetic data with autocorrelated errors
>>> np.random.seed(0)
>>> n_samples = 200
>>> y_true = np.linspace(0, 50, n_samples)
>>> # Errors have an AR(1) structure
>>> errors = [0]
>>> for _ in range(n_samples - 1):
...     errors.append(0.7 * errors[-1] + np.random.normal(0, 1))
>>> y_pred = y_true + np.array(errors)
>>>
>>> df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
>>>
>>> # Generate the plot
>>> ax = plot_error_autocorrelation(
...     df,
...     actual_col='actual',
...     pred_col='predicted',
...     title="Autocorrelation of Dependent Errors"
... )
"""


@isdf
@check_non_emptiness(params=["df"])
def plot_qq(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple[float, float] = (7, 7),
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
    **scatter_kwargs,
):
    # --- Input Validation and Preparation ---
    required_cols = [actual_col, pred_col]
    exist_features(df, features=required_cols)

    data_to_plot = df[required_cols].dropna()
    errors = data_to_plot[actual_col] - data_to_plot[pred_col]

    if len(errors) < 2:
        warnings.warn(
            "Not enough data points to generate a Q-Q plot.", stacklevel=2
        )
        return None

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- Generate Q-Q Plot Data and Plot ---
    (osm, osr), (slope, intercept, r) = probplot(errors, dist="norm", plot=ax)

    # --- Formatting ---
    ax.get_lines()[0].set_markerfacecolor("#E74C3C")  # Change marker color
    ax.get_lines()[0].set_markeredgecolor("#E74C3C")
    ax.get_lines()[1].set_color("#2980B9")  # Change line color

    ax.set_title(title or "Q-Q Plot of Forecast Errors", fontsize=16)
    ax.set_xlabel(xlabel or "Theoretical Quantiles (Normal)", fontsize=12)
    ax.set_ylabel(ylabel or "Ordered Error Values", fontsize=12)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    fig.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_qq.__doc__ = r"""
Generates a Quantile-Quantile (Q-Q) plot of forecast errors.

This function creates a Q-Q plot, a standard graphical method
for comparing a dataset's distribution to a theoretical
distribution (in this case, the normal distribution). It is an
essential tool for visually checking if the forecast errors are
normally distributed, a key assumption for many statistical
methods.

More details in :ref:`Q-Q Plot User Guide <ug_plot_qq>`. 

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted values.
actual_col : str
    The name of the column containing the true observed values.
pred_col : str
    The name of the column containing the point forecast values.
title : str, optional
    The title for the plot.
xlabel : str, optional
    The label for the x-axis.
ylabel : str, optional
    The label for the y-axis.
figsize : tuple of (float, float), default=(7, 7)
    The figure size in inches.
show_grid : bool, default=True
    Toggle the visibility of the plot's grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.
**scatter_kwargs
    Additional keyword arguments passed directly to the underlying
    scatter plot for the data points.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

See Also
--------
plot_error_distribution : A histogram/KDE plot of the same errors.
scipy.stats.probplot : The underlying SciPy function used.

Notes
-----
A Q-Q plot is constructed by plotting the quantiles of two
distributions against each other. This function compares the
quantiles of the empirical distribution of the forecast errors,
:math:`e_i = y_{true,i} - y_{pred,i}`, against the theoretical
quantiles of a standard normal distribution,
:math:`\mathcal{N}(0, 1)`.

If the two distributions are identical, the resulting points will
fall perfectly along the identity line :math:`y=x`. Systematic
deviations from this line indicate a departure from normality.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.context import plot_qq
>>>
>>> # Generate synthetic data with normally distributed errors
>>> np.random.seed(0)
>>> n_samples = 200
>>> y_true = np.linspace(0, 50, n_samples)
>>> errors = np.random.normal(0, 5, n_samples) # Normal errors
>>> y_pred = y_true + errors
>>>
>>> df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
>>>
>>> # Generate the Q-Q plot
>>> ax = plot_qq(
...     df,
...     actual_col='actual',
...     pred_col='predicted',
...     title="Q-Q Plot of Normally Distributed Errors"
... )
"""


@isdf
@check_non_emptiness(params=["df"])
def plot_error_distribution(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    **hist_kwargs,
):
    """
    Plots a histogram and KDE of the forecast errors.
    (Full docstring to be added later)
    """
    from ..utils.hist import plot_hist_kde

    # --- Input Validation and Preparation ---
    required_cols = [actual_col, pred_col]
    exist_features(df, features=required_cols)

    data_to_plot = df[required_cols].dropna()
    errors = data_to_plot[actual_col] - data_to_plot[pred_col]
    errors.name = "Forecast Error"  # Give the series a name for the plot

    if len(errors) < 2:
        warnings.warn(
            "Not enough data points to plot a distribution.", stacklevel=2
        )
        return None

    # --- Plotting ---
    # This function acts as a wrapper around the more general plot_hist_kde
    # We pass through any extra histogram-related keyword arguments.
    ax = plot_hist_kde(
        data=errors,
        title=title or "Distribution of Forecast Errors",
        x_label=xlabel or "Error (Actual - Predicted)",
        return_ax=True,
        **hist_kwargs,
    )

    return ax


plot_error_distribution.__doc__ = r"""
Plots a histogram and KDE of the forecast errors.

This function creates a distribution plot of the forecast errors
(residuals), combining a histogram with a smooth Kernel Density
Estimate (KDE) curve. It is a fundamental diagnostic for checking
if a model's errors are unbiased (centered at zero) and normally
distributed.

For more details, refer to 
:ref:`Error Autocorrelation (ACF) Plot User Guide <ug_plot_error_autocorrelation>`

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted values.
actual_col : str
    The name of the column containing the true observed values.
pred_col : str
    The name of the column containing the point forecast values.
title : str, optional
    The title for the plot. If ``None``, a default is generated.
xlabel : str, optional
    The label for the x-axis. If ``None``, a default is generated.
**hist_kwargs
    Additional keyword arguments passed directly to the underlying
    :func:`~kdiagram.utils.hist.plot_hist_kde` function (e.g.,
    `bins`, `kde_color`, `figsize`).

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

See Also
--------
plot_qq : A complementary plot for checking error normality.
plot_hist_kde : The general-purpose histogram utility this function wraps.
:ref:`userguide_context` : The user guide for contextual plots.

Notes
-----
This function first calculates the forecast errors (or residuals),
:math:`e_i = y_{true,i} - y_{pred,i}`. It then visualizes the
distribution of these errors using two standard non-parametric
methods:

1.  **Histogram**: The range of errors is divided into bins, and
    the height of each bar represents the frequency (or density)
    of errors in that bin.
2.  **Kernel Density Estimate (KDE)**: This provides a smooth,
    continuous estimate of the error's probability density
    function, based on foundational work in density estimation
    :footcite:p:`Silverman1986`.

A well-behaved model should ideally produce errors that are
normally distributed and centered around zero.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.context import plot_error_distribution
>>>
>>> # Generate synthetic data with normally distributed errors
>>> np.random.seed(0)
>>> n_samples = 500
>>> y_true = np.linspace(0, 50, n_samples)
>>> errors = np.random.normal(0, 5, n_samples) # Normal errors
>>> y_pred = y_true + errors
>>>
>>> df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
>>>
>>> # Generate the plot
>>> ax = plot_error_distribution(
...     df,
...     actual_col='actual',
...     pred_col='predicted',
...     title="Distribution of Normally-Distributed Errors",
...     bins=40
... )

References
----------
.. footbibliography::
    
"""


@isdf
@check_non_emptiness(params=["df"])
@ensure_pkg(
    "statsmodels",
    extra=(
        "To use PACF plots, please install"
        " statsmodels (`pip install statsmodels`)"
    ),
)
def plot_error_pacf(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
    **pacf_kwargs,
):
    from statsmodels.graphics.tsaplots import plot_pacf

    # --- Input Validation and Preparation ---
    required_cols = [actual_col, pred_col]
    exist_features(df, features=required_cols)

    data_to_plot = df[required_cols].dropna()
    errors = data_to_plot[actual_col] - data_to_plot[pred_col]

    n = len(errors)

    if n < 2:
        warnings.warn(
            "Not enough data points to plot partial autocorrelation.",
            stacklevel=2,
        )
        return None

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize)

    # # --- Generate PACF Plot ---
    # pacf_kwargs = get_valid_kwargs(plot_pacf, pacf_kwargs)
    # # set a stable default method unless the user provided one
    # pacf_kwargs.setdefault("method",  "ywm")

    # plot_pacf(errors, ax=ax, **pacf_kwargs)

    pacf_kwargs = get_valid_kwargs(plot_pacf, pacf_kwargs)
    pacf_kwargs.setdefault("method", "ywm")

    # Ensure lags respects statsmodels constraint: lags < n//2
    max_lags = max(1, n // 2 - 1)

    if (
        "lags" not in pacf_kwargs
        or pacf_kwargs["lags"] is None
        or pacf_kwargs["lags"] >= n // 2
    ):
        pacf_kwargs["lags"] = max_lags

    try:
        plot_pacf(errors, ax=ax, **pacf_kwargs)

    except ValueError:
        # Fallback once with a safer lags if a user forced something too large
        pacf_kwargs["lags"] = max_lags
        plot_pacf(errors, ax=ax, **pacf_kwargs)

    # --- Formatting ---
    ax.set_title(
        title or "Partial Autocorrelation of Forecast Errors", fontsize=16
    )
    ax.set_xlabel(xlabel or "Lag", fontsize=12)
    ax.set_ylabel(ylabel or "Partial Autocorrelation", fontsize=12)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    fig.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_error_pacf.__doc__ = r"""
Plots the partial autocorrelation of forecast errors.

This function creates a Partial Autocorrelation Function (PACF)
plot of the forecast errors. It is a critical companion to the
ACF plot, used to identify the *direct* relationship between an
error and its past values, after removing the effects of the
intervening lags. This plot requires the ``statsmodels`` package.

Additional details can  be found in
:ref:`Error Partial Autocorrelation (PACF) Plot User Guide <ug_plot_error_pacf>`

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted values.
actual_col : str
    The name of the column containing the true observed values.
pred_col : str
    The name of the column containing the point forecast values.
title : str, optional
    The title for the plot.
xlabel : str, optional
    The label for the x-axis.
ylabel : str, optional
    The label for the y-axis.
figsize : tuple of (float, float), default=(10, 5)
    The figure size in inches.
show_grid : bool, default=True
    Toggle the visibility of the plot's grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
savefig : str, optional
    The file path to save the plot. If ``None``, the plot is
    displayed interactively.
dpi : int, default=300
    The resolution (dots per inch) for the saved figure.
**pacf_kwargs
    Additional keyword arguments passed directly to the underlying
    ``statsmodels.graphics.tsaplots.plot_pacf`` function.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the plot.

See Also
--------
plot_error_autocorrelation : The companion plot for autocorrelation.

Notes
-----
While the ACF at lag :math:`k` shows the total correlation between
:math:`e_t` and :math:`e_{t-k}`, the PACF shows the **partial
correlation**. It measures the correlation between :math:`e_t` and
:math:`e_{t-k}` after removing the linear dependence on the
intermediate observations :math:`e_{t-1}, e_{t-2}, ..., e_{t-k+1}`.

This helps to isolate the direct relationship at a specific lag,
making it a key tool for identifying the order of autoregressive
(AR) processes in the residuals.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.context import plot_error_pacf
>>>
>>> # Generate synthetic data where errors have an AR(2) structure
>>> np.random.seed(0)
>>> n_samples = 200
>>> y_true = np.linspace(0, 50, n_samples)
>>> errors = np.zeros(n_samples)
>>> errors[0] = np.random.normal(0, 1)
>>> errors[1] = 0.6 * errors[0] + np.random.normal(0, 1)
>>> for t in range(2, n_samples):
...     errors[t] = 0.6 * errors[t-1] - 0.3 * errors[t-2] + np.random.normal(0, 1)
>>> y_pred = y_true - errors # Subtracting so error = actual - pred
>>>
>>> df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
>>>
>>> # Generate the PACF plot
>>> # The plot should show significant spikes at lags 1 and 2
>>> try:
...     ax = plot_error_pacf(
...         df,
...         actual_col='actual',
...         pred_col='predicted',
...         title="PACF of AR(2) Errors"
...     )
... except ImportError:
...     print("Skipping PACF plot: statsmodels is not installed.")
"""
