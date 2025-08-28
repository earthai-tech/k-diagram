#   License: Apache-2.0
#   Author: LKouadio <etanoyau@gmail.com>

"""
Specialized diagnostic polar plots ('kdiagrams', named after author
Kouadio) designed for comprehensive model evaluation and forecast
analysis. Provides functions to visualize
prediction uncertainty, model drift, interval coverage, anomaly
magnitude, actual vs. predicted performance, feature influence, and
related diagnostics using polar coordinates.
"""
from __future__ import annotations

import warnings
from typing import Any, Literal

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from ..api.summary import ResultSummary
from ..compat.matplotlib import get_cmap
from ..compat.sklearn import validate_params
from ..decorators import check_non_emptiness, isdf
from ..utils.diagnose_q import (
    build_qcols_multiple,
    detect_quantiles_in,
    validate_qcols,
)
from ..utils.handlers import columns_manager
from ..utils.plot import _sample_colors, set_axis_grid
from ..utils.validator import (
    _assert_all_types,
    exist_features,
)

__all__ = [
    "plot_actual_vs_predicted",
    "plot_anomaly_magnitude",
    "plot_coverage",
    "plot_coverage_diagnostic",
    "plot_interval_consistency",
    "plot_interval_width",
    "plot_model_drift",
    "plot_temporal_uncertainty",
    "plot_uncertainty_drift",
    "plot_velocity",
    "plot_radial_density_ring",
    "plot_polar_heatmap",
    "plot_polar_quiver",
]


@validate_params({"y_true": ["array-like"]})
def plot_coverage(
    y_true,
    *y_preds,
    names: str | list[str] | None = None,
    q: str | list[str] | None = None,
    kind: str = "line",
    cmap: str = "viridis",
    pie_startangle: int | float = 140,
    pie_autopct: str = "%1.1f%%",
    radar_color: str = "tab:blue",
    radar_fill_alpha: float = 0.25,
    radar_line_style: str = "o-",
    cov_fill: bool = False,
    figsize: tuple[str, str] = None,
    title: str | None = None,
    savefig: str | None = None,
    verbose: int = 1,
):
    # Convert the true values to a numpy array for consistency
    y_true = np.array(y_true)

    # Count how many model predictions were passed via *y_preds.
    num_models = len(y_preds)

    # Handle model names: create or extend to match the number of models.
    names = columns_manager(names, to_string=True)
    if names is None:
        names = [f"Model_{i + 1}" for i in range(num_models)]
    else:
        if len(names) < num_models:
            extra = num_models - len(names)
            for _i in range(extra):
                names.append(f"Model_{len(names) + 1}")

    coverage_scores = []

    q = columns_manager(q)
    # Handle quantiles
    if q is not None:
        q = np.array(q)
        if q.ndim != 1:
            raise ValueError(
                "Parameter 'q' must be a 1D list or array of quantile levels."
            )

        if not np.all((0 < q) & (q < 1)):
            raise ValueError("Quantile levels must be between 0 and 1.")
        # Sort q and get the sorted indices
        sorted_indices = np.argsort(q)
        q_sorted = q[sorted_indices]
    else:
        q_sorted = None

    # Compute coverage for each model in *y_preds.
    #   - If pred has shape (n_samples, n_quantiles), we compute coverage
    #     between min and max quantile per sample.
    #   - If pred is 1D, treat as a point forecast and check exact match
    #     (illustrative; typically coverage would be 0 unless data match).
    for _i, pred in enumerate(y_preds):
        pred = np.array(pred)

        # if (q is not None) and (pred.ndim == 2):
        if pred.ndim == 2:
            if q_sorted is not None:
                # No need since we used the first and last for
                # computed coverage.
                # --------------------
                # if pred.shape[1] != len(q_sorted):
                #     raise ValueError(
                #         f"Model {i+1} predictions have"
                #         f"{pred.shape[1]} quantiles, "
                #         f"but 'q' has {len(q_sorted)} levels."
                #     )
                # ---------------------
                # Align predictions with sorted quantiles
                pred_sorted = pred[:, sorted_indices]
            else:
                pred_sorted = np.sort(pred, axis=1)

            # Sort columns to ensure ascending order of quantiles.
            # pred_sorted = np.sort(pred, axis=1)
            lower_q = pred_sorted[:, 0]
            upper_q = pred_sorted[:, -1]
            in_interval = ((y_true >= lower_q) & (y_true <= upper_q)).astype(
                int
            )
            coverage = np.mean(in_interval)

        elif pred.ndim == 1:
            # Point forecast coverage as fraction of exact matches
            matches = (y_true == pred).astype(int)
            coverage = np.mean(matches)

        else:
            # If neither scenario applies, store None.
            coverage = None

        coverage_scores.append(coverage)

    # Prepare data for plotting. Replace None with 0 for convenience.
    valid_cov = [c if c is not None else 0 for c in coverage_scores]
    x_idx = np.arange(num_models)

    if kind in {"bar", "line", "pipe"}:
        # Initialize the figure.
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
    # Plot according to the chosen 'kind'.
    if kind == "bar":
        plt.bar(x_idx, valid_cov, color="blue", alpha=0.7)
        for idx, val in enumerate(coverage_scores):
            if val is not None:
                plt.text(
                    x=idx,
                    y=val + 0.01,
                    s=f"{val:.2f}",
                    ha="center",
                    va="bottom",
                )
        plt.xticks(x_idx, names)
        plt.ylim([0, 1])
        plt.ylabel("Coverage")
        plt.xlabel("Models")

    elif kind == "line":
        plt.plot(x_idx, valid_cov, marker="o")
        for idx, val in enumerate(coverage_scores):
            if val is not None:
                plt.text(
                    x=idx,
                    y=val + 0.01,
                    s=f"{val:.2f}",
                    ha="center",
                    va="bottom",
                )
        plt.xticks(x_idx, names)
        plt.ylim([0, 1])
        plt.ylabel("Coverage")
        plt.xlabel("Models")

    elif kind == "pie":
        # Pie chart: each slice represents a model's coverage. By default,
        # the slice size is coverage[i] out of the sum of coverage.
        total_cov = sum(valid_cov)
        if total_cov == 0:
            # Avoid a zero-coverage pie chart.
            plt.text(
                0.5, 0.5, "No coverage to plot", ha="center", va="center"
            )
        else:
            plt.pie(
                valid_cov,
                labels=names,
                autopct=pie_autopct,
                startangle=pie_startangle,
                colors=get_cmap(cmap)(np.linspace(0, 1, num_models)),
            )
            plt.axis("equal")  # Make the pie chart a perfect circle.

    elif kind == "radar":
        # #Radar chart: place each model's coverage as a radial axis.

        N = num_models
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        coverage_radar = np.concatenate((valid_cov, [valid_cov[0]]))

        ax = plt.subplot(111, polar=True)

        # Plot main coverage line
        ax.plot(
            angles,
            coverage_radar,
            radar_line_style,
            color=radar_color,
            label="Coverage",
        )

        # Handle fill based on number of models
        if cov_fill:
            if num_models == 1:
                # Single model: radial gradient fill up to coverage value
                coverage_value = valid_cov[0]
                theta = np.linspace(0, 2 * np.pi, 100)
                r = np.linspace(0, coverage_value, 100)
                R, Theta = np.meshgrid(r, theta)

                # Create gradient using specified colormap
                ax.grid(False)
                ax.pcolormesh(
                    Theta,
                    R,
                    R,
                    cmap=cmap,
                    shading="auto",
                    alpha=radar_fill_alpha,
                    zorder=0,  # Place behind main plot
                )
                ax.grid(True, which="both")
                # Add red circle at coverage value
                ax.plot(
                    theta,
                    [coverage_value] * len(theta),  # Constant radius
                    color="red",
                    linewidth=2,
                    linestyle="-",
                    # label=f'Coverage Value ({coverage_value:.2f})'
                )

                # Add concentric grid circles at 0.2, 0.4, 0.6, 0.8
                # with correct properties
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])
                ax.yaxis.grid(
                    True,
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.7,
                )

            else:
                # Multiple models: transparent fill between center and line
                ax.fill(
                    angles,
                    coverage_radar,
                    color=radar_color,
                    alpha=radar_fill_alpha,
                    zorder=0,
                )
        # Final formatting
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels=names)
        ax.set_ylim(0, 1)
        plt.legend(loc="upper right")

    else:
        # Fallback: print coverage scores to the console for each model.
        for idx, val in enumerate(coverage_scores):
            print(f"{names[idx]} coverage: {val}")

    if verbose:
        cov_dict = {
            names[idx]: cov for idx, cov in enumerate(coverage_scores)
        }

        summary = ResultSummary("CoverageScores").add_results(cov_dict)
        print(summary)

    # Add title if provided.
    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")

    plt.show()


plot_coverage.__doc__ = r"""
Plot overall coverage scores for forecast intervals or points.

Computes and visualizes the empirical coverage rate, which is
the fraction of times the true values fall within predicted
quantile intervals (or match point forecasts). Supports comparing
multiple models or prediction sets using various chart types.

This plot provides a high-level summary of whether prediction
intervals are well-calibrated on average across the entire
dataset. It's useful for quickly comparing the overall
reliability of uncertainty estimates from different models or
parameter settings before diving into finer diagnostics 
:footcite:p:`kouadiob2025`.

Parameters
----------
y_true : array-like of shape (n_samples,)
    The ground truth target values. Must be a 1D array-like
    object (list, NumPy array, pandas Series).

*y_preds : array-like
    Variable number of prediction arrays. Each positional
    argument should be an array-like object corresponding to a
    model or prediction set. The shape determines interpretation:

    - shape (n_samples, n_quantiles): Represents quantile
      forecasts (e.g., lower, median, upper bounds). Coverage
      is calculated based on the interval defined by the minimum
      and maximum quantile columns provided (after sorting based
      on `q` if given, or by value otherwise). Requires `q`
      parameter for proper interval definition if quantiles
      are not ordered.
    - shape (n_samples,): Represents point forecasts. Coverage
      is calculated as the proportion of exact matches to
      `y_true` (typically near zero for continuous data).

names : list of str, optional
    Labels for each prediction set provided in `*y_preds`. Used
    in legends and axis labels. If `None` or shorter than the
    number of prediction sets, default names like "Model_1",
    "Model_2", etc., are generated for the missing ones.
    Default is ``None``.

q : list of float, optional
    Quantile levels corresponding to the columns in 2D `y_preds`
    arrays. Values must be between 0 and 1. Used to identify
    and sort quantile columns correctly before determining the
    interval bounds (minimum and maximum specified quantile) for
    coverage calculation. Required if `y_preds` contains 2D
    arrays representing unordered quantiles. Default is ``None``.

kind : {'line', 'bar', 'pie', 'radar'}, optional
    The type of plot to generate for visualizing coverage scores.
    Default is ``'line'``.

    - ``'line'``: Line chart connecting coverage scores for
      each prediction set.
    - ``'bar'``: Bar chart showing the coverage score for each
      prediction set.
    - ``'pie'``: Pie chart where each slice represents a model's
      coverage score as a fraction of the sum of all scores.
    - ``'radar'``: Radar (spider) chart where each axis
      represents a prediction set, and the distance along the
      axis indicates its coverage score.

cmap : str, optional
    Matplotlib colormap name used for coloring slices in the
    `'pie'` chart or the gradient fill in single-model radar
    plots when ``cov_fill=True``. Default is ``'viridis'``.

pie_startangle : float, optional
    Starting angle in degrees for the first slice in the `'pie'`
    chart. Rotates the chart orientation. Default is 140.

pie_autopct : str or None, optional
    String formatting pattern (e.g., ``'%1.1f%%'``) used to label
    pie slices with their numeric value (percentage). If `None`,
    no numeric labels are shown. Default is ``'%1.1f%%'``.

radar_color : str, optional
    Color specification for the main line and optional fill in
    the `'radar'` chart. Accepts any valid Matplotlib color.
    Default is ``'tab:blue'``.

radar_fill_alpha : float, optional
    Transparency level (alpha) for the filled area in the
    `'radar'` chart when ``cov_fill=True``. Value between 0
    (transparent) and 1 (opaque). Default is 0.25.

radar_line_style : str, optional
    Matplotlib line and marker style string for the radar chart
    outline (e.g., ``'o-'``, ``'--'``, ``':'``).
    Default is ``'o-'``.

cov_fill : bool, optional
    If ``True`` and ``kind='radar'``, fills the area under the
    radar plot lines. For a single model, creates a radial
    gradient; for multiple models, fills polygons.
    Default is ``False``.

figsize : tuple of (float, float), optional
    Figure size ``(width, height)`` in inches. If ``None``, uses
    Matplotlib's default figure size.

title : str, optional
    Optional title for the plot. Default is ``None``.

savefig : str, optional
    If not ``None``, specifies the full path (including filename
    and extension, e.g., 'coverage.png') to save the figure.
    If ``None``, the plot is displayed interactively.
    Default is ``None``.

verbose : {0, 1}, optional
    Controls printing of coverage scores to the console.
    - ``0``: Silent.
    - ``1``: Prints a formatted summary of coverage scores.
    Default is ``1``.

Returns
-------
None
    This function generates and displays or saves a Matplotlib plot
    but does not return any object.

Raises
------
ValueError
    If `q` contains values outside (0, 1), or if `kind` is not
    one of the allowed options, or if dimensions of `y_preds`
    and `q` are incompatible.
TypeError
    If inputs `y_true` or `y_preds` contain non-numeric data.

See Also
--------
plot_coverage_diagnostic : Plot point-wise coverage status.
numpy.mean : Used to calculate the average coverage rate.

Notes
-----
The empirical coverage rate is a key metric for assessing the
calibration of probabilistic forecasts, particularly prediction
intervals :footcite:p:`kouadiob2025`.

**Calculation:**

- For quantile intervals (2D `y_preds`), the interval for each
  sample :math:`i` is defined by the minimum and maximum
  predicted values across the specified quantiles for that sample,

  :math:`[\hat{y}_{i}^{(\ell)}, \hat{y}_{i}^{(u)}]`. Coverage is:

  .. math::
     \text{Coverage} = \frac{1}{N}\sum_{i=1}^{N}
     \mathbf{1}\{\hat{y}_{i}^{(\ell)} \leq y_i
     \leq \hat{y}_{i}^{(u)}\}
     
  where :math:`\mathbf{1}\{\cdot\}` is 1 if true, 0 otherwise.

- For point forecasts (1D `y_preds`), coverage is the proportion
  of exact matches:

  .. math::
     \text{Coverage} = \frac{1}{N}\sum_{i=1}^{N}
     \mathbf{1}\{ \hat{y}_i = y_i \}
     
  This is typically very low for continuous :math:`y_true`.

**Interpretation:** Compare the calculated coverage score(s) to
the nominal coverage rate implied by the quantiles (e.g., 80%
for a Q10-Q90 interval). Scores far from the nominal rate suggest
miscalibration (intervals too wide or too narrow on average)
:footcite:p:`Gneiting2007b`.

**Radar Plot Fill (`cov_fill=True`):**

- Single model: Shows a circular gradient fill from the center
  out to the calculated coverage score radius, with a solid
  reference line at the coverage score.
- Multiple models: Fills the polygon defined by the coverage
  scores for all models with a semi-transparent color.

Examples
--------
>>> import numpy as np
>>> from kdiagram.plot.uncertainty import plot_coverage
>>> # True values
>>> y_true = np.random.rand(100) * 10
>>> # 3-quantile predictions (Q10, Q50, Q90) for two models
>>> y_pred_A = np.sort(np.random.normal(
...    loc=9, scale=2, size=(100, 3)), axis=1)
>>> y_pred_B = np.sort(np.random.normal(
...    loc=11, scale=3, size=(100, 3)), axis=1)
>>> q_levels = [0.1, 0.5, 0.9]

>>> # Bar chart comparison
>>> plot_coverage(y_true, y_pred_A, y_pred_B, q=q_levels,
...               names=['Model A', 'Model B'], kind='bar',
...               title='Coverage Comparison (Bar)')

>>> # Radar chart comparison with fill
>>> plot_coverage(y_true, y_pred_A, y_pred_B, q=q_levels,
...               names=['Model A', 'Model B'], kind='radar',
...               title='Coverage Comparison (Radar)',
...               cov_fill=True, cmap='Set2')

References
----------
.. footbibliography::
    
"""


@check_non_emptiness
@isdf
def plot_model_drift(
    df: pd.DataFrame,
    q_cols: list | None = None,
    q10_cols: list[str] | None = None,
    q90_cols: list[str] | None = None,
    horizons: list[str | int] | None = None,
    color_metric_cols: list[str] | None = None,
    acov: str = "quarter_circle",
    value_label: str = "Uncertainty Width (Q90 - Q10)",
    cmap: str = "coolwarm",
    figsize: tuple[int, int] = (8, 8),
    title: str = "Model Forecast Drift Over Time",
    show_grid: bool = True,
    annotate: bool = True,
    grid_props: dict | None = None,
    savefig: str | None = None,
):
    r"""Visualize forecast drift across prediction horizons.

    Renders a polar bar chart depicting how average model
    uncertainty (or another metric) evolves as the forecast horizon
    increases. Each bar corresponds to a specific forecast horizon,
    arranged angularly.

    This plot is crucial for diagnosing model degradation over
    longer lead times, often termed *concept drift* or *model
    aging* [1]_. A distinct increase in bar height (radius) for
    later horizons signals inflating uncertainty or error, potentially
    indicating a need for model retraining or adjustments to account
    for changing dynamics. Use this visualization to assess if your
    model's reliability holds as you forecast further into the
    future.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the necessary quantile columns (or
        columns specified in ``color_metric_cols``) for each forecast
        horizon.

    q_cols : list[tuple[str, str]], optional
        A list where each element is a tuple containing the column
        names for the lower and upper quantiles for a specific
        horizon, e.g., ``[('q10_h1', 'q90_h1'), ('q10_h2', 'q90_h2')]``.
        If ``None``, ``q10_cols`` and ``q90_cols`` must be provided
        instead. Default is ``None``.

    q10_cols : list[str], optional
        List of column names representing the lower quantile (e.g.,
        10th percentile) for each successive horizon. Must be provided
        if ``q_cols`` is ``None``. Must have the same length as
        ``q90_cols`` and ``horizons`` (if provided). Default is ``None``.

    q90_cols : list[str], optional
        List of column names representing the upper quantile (e.g.,
        90th percentile) for each successive horizon. Must be provided
        if ``q_cols`` is ``None``. Must have the same length as
        ``q10_cols`` and ``horizons`` (if provided). Default is ``None``.

    horizons : list of str or int, optional
        Labels corresponding to each forecast horizon (plotted on the
        angular axis). The order must match the order in ``q_cols`` or
        ``q10_cols``/``q90_cols``. If ``None``, generic labels like
        "Horizon 1", "Horizon 2", ... are generated.
        Default is ``None``.

    color_metric_cols : list of str, optional
        If provided, the bars are colored based on the mean value of
        these columns for each horizon (e.g., provide RMSE columns
        like ``['rmse_h1', 'rmse_h2', ...]``). If ``None``, bars are
        colored based on the calculated mean interval width (Q90-Q10).
        Default is ``None``.

    acov : {'default', 'half_circle', 'quarter_circle', \
        'eighth_circle'}, optional

        Specifies the angular coverage (span) of the plot. Use narrower
        sectors for fewer horizons. Default is ``'quarter_circle'``.

        - ``'default'``: Full circle (:math:`2\pi`, 360°).
        - ``'half_circle'``: Half circle (:math:`\pi`, 180°).
        - ``'quarter_circle'``: Quarter circle (:math:`\pi/2`, 90°).
        - ``'eighth_circle'``: Eighth circle (:math:`\pi/4`, 45°).

    value_label : str, optional
        Label displayed for the radial axis, describing the metric
        represented by the bar height. Default is
        ``'Uncertainty Width (Q90 - Q10)'``.

    cmap : str, optional
        Name of the Matplotlib colormap used to color the bars based
        on their radial value (or the `color_metric_cols` value).
        Default is ``'coolwarm'``.

    figsize : tuple of (float, float), optional
        Figure dimensions ``(width, height)`` in inches. If ``None``,
        uses the default ``(8, 8)``.

    title : str, optional
        Headline text displayed above the plot. Default is
        ``'Model Forecast Drift Over Time'``.

    show_grid : bool, optional
        If ``True``, displays radial and angular grid lines to aid
        interpretation. Default is ``True``.

    annotate : bool, optional
        If ``True``, displays the numeric value (mean width or mean
        color metric) on top of each bar. Default is ``True``.

    grid_props : dict, optional
        Dictionary of keyword arguments to customize the appearance
        of the grid lines (passed to ``ax.grid()``). E.g.,
        ``{'linestyle': ':', 'linewidth': 0.5}``. Default is ``None``.

    savefig : str, optional
        Full path and filename to save the plot (e.g., 'drift.pdf').
        If ``None``, the plot is displayed interactively.
        Default is ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The polar axes object containing the bar chart, allowing
        for further customization if desired.

    Raises
    ------
    ValueError
        If required quantile columns (`q_cols` or both `q10_cols`
        and `q90_cols`) are missing from `df`, or if the lengths of
        provided column lists/horizons mismatch.
    TypeError
        If data in the specified columns cannot be processed
        numerically.

    See Also
    --------
    kdiagram.plot.uncertainty.plot_uncertainty_drift : Visualize drift
        of the uncertainty pattern using rings.
    kdiagram.utils.plot.set_axis_grid : Helper for grid styling (if used).

    Notes
    -----
    The primary radial value plotted for each horizon :math:`h` is the
    mean interval width calculated across all :math:`N` samples:

    .. math::
       \bar{w}_h = \frac{1}{N}\sum_{j=1}^{N} \left( Q_{up, j, h} -
       Q_{low, j, h} \right)

    If ``color_metric_cols`` is provided, a similar average is
    calculated for those columns to determine bar color.

    Radii may be scaled relative to the maximum radius if a
    restricted angular coverage (``acov`` is not 'default') is used,
    to better fit the visual sector [2]_.

    References
    ----------

    .. [1] Kouadio, K. L., Liu, R., Loukou, K. G. H., Liu, J., & Liu, W. (2025).
       Analytics Framework for Interpreting Spatiotemporal Probabilistic 
       Forecasts. *International Journal of Forecasting*. Manuscript submitted.
       
    .. [2] Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., &
           Bouchachia, A. (2014). A survey on concept drift
           adaptation. *ACM Computing Surveys (CSUR)*, 46(4), 1-37.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from kdiagram.plot.uncertainty import plot_model_drift
    >>> # Example with synthetic data
    >>> years = [2023, 2024, 2025, 2026]
    >>> n_samples=50
    >>> df_synth = pd.DataFrame()
    >>> q10_cols, q90_cols = [], []
    >>> for i, year in enumerate(years):
    ...     ql, qu = f'val_{year}_q10', f'val_{year}_q90'
    ...     q10_cols.append(ql); q90_cols.append(qu)
    ...     q10 = np.random.rand(n_samples)*5 + i*0.5
    ...     q90 = q10 + np.random.rand(n_samples)*2 + 1 + i*0.8
    ...     df_synth[ql]=q10; df_synth[qu]=q90
    >>> ax = plot_model_drift(
    ...     df=df_synth,
    ...     q10_cols=q10_cols,
    ...     q90_cols=q90_cols,
    ...     horizons=years,
    ...     acov='quarter_circle', # Use 90 degree span
    ...     title='Synthetic Model Drift Example'
    ... )
    """
    #
    # 1. Pair quantile columns
    q_cols = build_qcols_multiple(
        q_cols,
        qlow_cols=q10_cols,
        qup_cols=q90_cols,
    )

    n_horizons = len(q_cols)

    # Default angular labels
    if horizons is None:
        horizons = [f"Horizon {idx + 1}" for idx in range(n_horizons)]

    #
    # 2. Compute average inter-quantile width per horizon
    widths = np.array([(df[q90] - df[q10]).mean() for q10, q90 in q_cols])

    # Secondary colouring metric
    if color_metric_cols is not None:
        colour_vals = np.array([df[col].mean() for col in color_metric_cols])
    else:
        colour_vals = widths

    #
    # 3. Angular span selection
    #
    span = {
        "default": 2 * np.pi,
        "half_circle": np.pi,
        "quarter_circle": np.pi / 2,
        "eighth_circle": np.pi / 4,
    }.get(acov, 2 * np.pi)

    theta = np.linspace(0.0, span, n_horizons, endpoint=False)

    # Scale radii when angular coverage < full circle
    radii = widths / widths.max() if span < 2 * np.pi else widths

    #
    # 4. Figure setup
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": "polar"},
    )

    # Orient polar chart: 0° at the top, clockwise direction
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(span))

    # Colormap normalisation
    norm = Normalize(vmin=colour_vals.min(), vmax=colour_vals.max())
    colours = get_cmap(cmap)(norm(colour_vals))

    #
    # 5. Draw bars
    bar_width = (span / n_horizons) * 0.9  # slight gap between bars
    ax.bar(
        theta,
        radii,
        width=bar_width,
        color=colours,
        edgecolor="k",
        alpha=0.85,
        linewidth=0.8,
    )

    # Annotation
    if annotate:
        for ang, rad, raw in zip(theta, radii, widths):
            label = f"{raw:.2f}"
            ax.text(
                ang,
                rad + 0.03 * radii.max(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Ticks & labels
    ax.set_xticks(theta)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_yticklabels([])
    ax.set_ylabel(value_label)
    ax.set_title(title, fontsize=14, pad=20)

    # Optional grid
    set_axis_grid(ax, show_grid, grid_props=grid_props)

    # 6. Output handling
    if savefig is not None:
        fig.savefig(savefig, bbox_inches="tight")
    else:
        plt.show()

    return ax


@check_non_emptiness
@isdf
def plot_velocity(
    df: pd.DataFrame,
    q50_cols: list[str],
    theta_col: str | None = None,
    cmap: str = "viridis",
    acov: str = "default",
    normalize: bool = True,
    use_abs_color: bool = True,
    figsize: tuple[float, float] = (9, 9),
    title: str | None = None,
    s: float | int = 30,
    alpha: float = 0.85,
    show_grid: bool = True,
    savefig: str | None = None,
    cbar: bool = True,
    mask_angle: bool = False,
):
    # --- Input Validation ---
    # Check if required q50_cols exist in the DataFrame
    missing_cols = [col for col in q50_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following Q50 columns are missing from the "
            f"DataFrame: {', '.join(missing_cols)}"
        )

    if len(q50_cols) < 2:
        raise ValueError(
            "At least two Q50 columns (representing two time points)"
            " are required to compute velocity."
        )

    # Check theta_col status and warn if provided but unused
    if theta_col is not None:
        if theta_col not in df.columns:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found in "
                f"DataFrame columns. Using index for angular position.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but the current"
                f" implementation uses the DataFrame index for angular "
                f"positioning ('theta'). The column '{theta_col}' is "
                f"currently ignored for positioning.",
                UserWarning,
                stacklevel=2,
            )

    # --- Data Processing ---
    # Extract Q50 data into a NumPy array (locations x time)
    q50_array = df[q50_cols].values  # Shape (N, M)

    # Compute yearly differences along the time axis (axis=1)
    # Result shape (N, M-1)
    yearly_diff = np.diff(q50_array, axis=1)

    # Compute average velocity per location (mean across time diffs)
    # Result shape (N,)
    r = np.mean(yearly_diff, axis=1)

    # Normalize radial values (velocity) if requested
    r_normalized = r.copy()  # Use a copy for potential normalization
    if normalize:
        r_range = np.ptp(r)  # Peak-to-peak (max - min)
        if r_range > 1e-9:  # Avoid division by zero or near-zero
            r_min = r.min()
            r_normalized = (r - r_min) / r_range
        else:
            # Handle case where all velocities are the same
            r_normalized = np.zeros_like(r)  # Set all to 0 if range is zero
            warnings.warn(
                "Velocity range is zero or near-zero. Normalized radial "
                "values ('r') are set to 0.",
                UserWarning,
                stacklevel=2,
            )

    # Determine values used for coloring the points
    if use_abs_color:
        # Use average absolute Q50 magnitude across all years
        color_vals = np.mean(np.abs(q50_array), axis=1)
        cbar_label = "Average Abs Q50 Magnitude"
    else:
        # Use the calculated average velocity for color
        color_vals = r  # Use original velocity for color scale
        cbar_label = "Average Velocity"

    # --- Angular Coordinate Calculation ---
    N = len(df)  # Number of locations/points
    # Generate linear space from 0 to 1 for N points
    theta_normalized = np.linspace(
        0, 1, N, endpoint=True
    )  # Includes endpoint 1

    # Map normalized theta to the desired angular coverage
    angular_range_map = {
        "default": 2 * np.pi,
        "half_circle": np.pi,
        "quarter_circle": np.pi / 2,
        "eighth_circle": np.pi / 4,
    }
    # Get the angular span in radians, default to full circle if invalid
    angle_span = angular_range_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in angular_range_map:
        warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning,
            stacklevel=2,
        )

    # Calculate final theta values
    theta = theta_normalized * angle_span

    # --- Color Normalization for Plotting ---
    # try:
    cmap_used = get_cmap(cmap, default="viridis")
    # except (TypeError, ValueError, KeyError):
    #     warnings.warn(
    #         f"Invalid `cmap` name '{cmap}'. Falling back to 'viridis'.",
    #         UserWarning,
    #         stacklevel=2,
    #     )
    #     cmap = "viridis"
    #     cmap_used = get_cmap(cmap)

    # Normalize color values to the range [0, 1] for the colormap
    color_norm = Normalize(vmin=np.min(color_vals), vmax=np.max(color_vals))
    # Map normalized color values to actual colors using the colormap
    colors = cmap_used(color_norm(color_vals))

    # --- Plotting ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )

    # Set the angular limits based on angular coverage
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(angle_span))  # set_thetamax expects degrees

    # Create the polar scatter plot
    ax.scatter(
        theta,
        r_normalized if normalize else r,  # Use normalized or raw r
        c=colors,  # Point colors
        s=s,  # Point size
        edgecolor="k",  # Point edge color (optional, for visibility)
        linewidth=0.5,  # Point edge width (optional)
        alpha=alpha,  # Point transparency
    )

    # Set plot title
    ax.set_title(
        title or "Average Velocity Polar Plot", fontsize=14, y=1.08
    )  # Adjust title position

    # Add color bar if requested
    if cbar:
        # Create a ScalarMappable for the colorbar
        sm = cm.ScalarMappable(norm=color_norm, cmap=cmap_used)
        sm.set_array([])  # Necessary for ScalarMappable

        # Add the colorbar to the figure
        cbar_obj = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
        cbar_obj.set_label(cbar_label, fontsize=10)

    # Customize grid and labels
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)

    # Optionally mask angular tick labels
    if mask_angle:
        ax.set_xticklabels([])

    # Set radial label based on normalization
    if normalize:
        ax.set_ylabel("Normalized Average Velocity", labelpad=15, fontsize=10)
        # Ensure radial limits are appropriate for normalized data
        # ax.set_ylim(bottom=0, top=1.05) # Give slight padding
        # ax.set_yticks(np.linspace(0, 1, 5)) # Example radial ticks
    else:
        ax.set_ylabel("Average Velocity", labelpad=15, fontsize=10)
        # Radial limits might need auto-scaling or manual setting

    plt.tight_layout()  # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


plot_velocity.__doc__ = r"""
Polar plot visualizing average velocity across locations.

Generates a polar scatter plot where each point represents a
unique location or observation from the input DataFrame. The
radial distance (`r`) of each point corresponds to the average
rate of change (velocity) of the median prediction (Q50) over
consecutive time periods (e.g., years), optionally normalized
to [0, 1]. The angular position (`theta`) represents the location,
currently determined by its index in the DataFrame, mapped onto a
specified angular coverage. The color of each point provides an
additional dimension, representing either the calculated velocity
itself or the average absolute magnitude of the Q50 predictions
over the considered time periods [1]_.

This visualization is useful for identifying spatial patterns in
the dynamics of a phenomenon, such as locating areas of rapid or
slow change (high/low velocity) in land subsidence predictions.
Coloring by magnitude helps to contextualize the velocity (e.g.,
is high velocity occurring in areas of already high subsidence?).

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the data. Must include the columns
    specified in `q50_cols`. Decorator `@isdf` ensures this is a
    pandas DataFrame. Decorator `@check_non_emptiness` ensures
    it's not empty.

q50_cols : list of str
    An ordered list of column names representing the Q50 (median)
    predictions for consecutive time steps (e.g., years). The list
    must contain at least two column names to compute velocity.
    Example: ``['subsidence_2022_q50', 'subsidence_2023_q50',
    'subsidence_2024_q50']``.

theta_col : str, optional
    *Intended* column name to determine the angular position (`theta`)
    for each location (e.g., 'latitude', 'longitude', or a spatial
    index). If ``None``, the DataFrame index is conceptually used.
    *Note: The current implementation maps the DataFrame row index
    to the angular range specified by `acov`, regardless of whether
    `theta_col` is provided. Providing `theta_col` will currently
    trigger a warning but will not affect the plot's angular axis.*
    Default is ``None``.

cmap : str, default='viridis'
    The name of the Matplotlib colormap used to color the scatter
    points based on `color_vals` (determined by `use_abs_color`).

acov : str, default='default'
    Angular coverage defining the span of the polar plot's theta
    axis. Options are:

    - ``'default'``: Full circle (2p radians or 360 degrees).
    - ``'half_circle'``: Half circle (p radians or 180 degrees).
    - ``'quarter_circle'``: Quarter circle (p/2 radians or 90
      degrees).
    - ``'eighth_circle'``: Eighth circle (p/4 radians or 45
      degrees).
    
    Invalid options default to ``'default'``.

normalize : bool, default=True
    If ``True``, the calculated average velocity values (`r`) are
    min-max normalized to the range [0, 1] before plotting radially.
    This emphasizes relative velocity patterns. If ``False``, the
    raw average velocity values are used for the radial coordinate.

use_abs_color : bool, default=True
    Determines the variable used for coloring the points:

    - If ``True``, points are colored based on the average absolute
      magnitude of the Q50 values across the specified `q50_cols`.
      This highlights areas with high overall prediction values.
    - If ``False``, points are colored based on the calculated
      average velocity (`r`) itself. This highlights areas of high
      or low rate of change.

figsize : tuple of (float, float), default=(9, 9)
    The width and height of the figure in inches.

title : str, optional
    The title displayed above the polar plot. If ``None``, a default
    title "Normalized Subsidence Velocity" (or similar, depending
    on context, though not dynamically changed here) is used.
    Default is ``None``.

s : float or int, default=30
    The marker size for the scatter points.

alpha : float, default=0.85
    The transparency level of the scatter points (0=transparent,
    1=opaque). Useful for visualizing dense data.

show_grid : bool, default=True
    If ``True``, display the polar grid lines (radial and angular)
    on the plot.

savefig : str, optional
    The file path (including extension, e.g., 'velocity_plot.pdf')
    where the plot image should be saved. If ``None``, the plot is
    displayed interactively using `plt.show()`.
    Default is ``None``.

cbar : bool, default=True
    If ``True``, display a color bar alongside the plot indicating
    the mapping between colors and the values defined by
    `use_abs_color`.

mask_angle : bool, default=False
    If ``True``, hide the angular tick labels (the degrees/radians
    around the circumference). This can be useful if the angular
    position based on index is not inherently meaningful.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the polar scatter plot.
    Can be used for further customization.

Raises
------
ValueError
    If `q50_cols` contains fewer than two column names.

See Also
--------
numpy.diff : Computes the difference between consecutive elements.
numpy.mean : Computes the arithmetic mean.
matplotlib.pyplot.scatter : Creates scatter plots.
matplotlib.pyplot.polar : Creates polar plots.
kdiagram.plot.uncertainty.plot_uncertainty_drift : Visualizes
    uncertainty width changes over time.

Notes
-----

- The function assumes the columns in `q50_cols` represent equally
  spaced time steps for the velocity calculation to be meaningful
  as an average *yearly* (or per-step) velocity.
- The average velocity (`r`) is calculated as the mean of the
  first-order differences between consecutive columns in `q50_cols`.
- Normalization of `r` uses min-max scaling:
  :math:`r' = (r - \min(r)) / (\max(r) - \min(r))`.
- The angular coordinate `theta` is currently derived from the
  DataFrame index, mapped linearly onto the angular range defined
  by `acov`. The `theta_col` parameter is not used for positioning
  in the current implementation, which might be revised in future
  versions. A warning is issued if `theta_col` is provided [2]_.

Let :math:`\mathbf{Q}` be the data matrix extracted from `df` using
columns `q50_cols`, with shape :math:`(N, M)`, where :math:`N` is
the number of locations (rows) and :math:`M` is the number of time
points (columns). Note the transpose compared to the description
in `plot_feature_fingerprint`.

1. **Velocity Calculation**: The differences between consecutive time
   points for each location :math:`j` are computed:

   :math:`\Delta Q_{j,i} = Q_{j, i+1} - Q_{j, i}` for
   :math:`i = 0, \dots, M-2`.

   The average velocity for location :math:`j` is:

   .. math::
       r_j = \frac{1}{M-1} \sum_{i=0}^{M-2} \Delta Q_{j,i}

2. **Radial Normalization** (if `normalize=True`):

   Let :math:`\mathbf{r} = (r_0, \dots, r_{N-1})`.

   .. math::
       r'_j = \frac{r_j - \min(\mathbf{r})}{\max(\mathbf{r}) - \min(\mathbf{r})}

   If :math:`\max(\mathbf{r}) = \min(\mathbf{r})`, :math:`r'_j = 0`.

3. **Color Value Calculation**:

   - If `use_abs_color=True`: Average absolute magnitude.

     .. math::
         c_j = \frac{1}{M} \sum_{i=0}^{M-1} |Q_{j,i}|

   - If `use_abs_color=False`: Use average velocity.
     :math:`c_j = r_j`

4. **Angular Coordinate Calculation**:

   Let :math:`S` be the angular span in radians determined by `acov`
   (e.g., :math:`2\pi` for ``'default'``). The angle for location
   :math:`j` (where :math:`j` is the row index from :math:`0` to
   :math:`N-1`) is:

   .. math::
        \theta_j = \frac{j}{N} \times S

   The code uses `np.linspace(0, 1, N)` which generates N
   points from 0 to 1 inclusive, so the formula might be slightly
   different depending on endpoint handling, effectively
   :math:`\theta_j = \frac{j}{N-1} \times S` for the N points if
   endpoint=True, or spacing relates to N intervals if
   ``endpoint=False``. The code uses `np.linspace(0, 1, N)` and
   multiplies by `angle_span`, suggesting the angles might span
   from 0 up to `angle_span`.

References
----------

.. [1] Kouadio, K. L., Liu, R., Loukou, K. G. H., Liu, J., & Liu, W. (2025).
   Analytics Framework for Interpreting Spatiotemporal Probabilistic Forecasts.
   *International Journal of Forecasting*. Manuscript submitted.

.. [2] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
       Computing in Science & Engineering, 9(3), 90-95.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.uncertainty import plot_velocity

**1. Random Example:**

>>> np.random.seed(0)
>>> N_points = 100
>>> df_random = pd.DataFrame({
...     'location_id': range(N_points),
...     'value_2020_q50': np.random.rand(N_points) * 10,
...     'value_2021_q50': (np.random.rand(N_points) * 10 +
...                        np.linspace(0, 5, N_points)),
...     'value_2022_q50': (np.random.rand(N_points) * 10 +
...                        np.linspace(0, 10, N_points)),
...     'latitude': np.linspace(22, 23, N_points)
... })
>>> q50_cols_random = ['value_2020_q50', 'value_2021_q50',
...                    'value_2022_q50']
>>> ax_random = plot_velocity(
...     df=df_random,
...     q50_cols=q50_cols_random,
...     theta_col='latitude', # Note: currently ignored for pos
...     acov='default',
...     normalize=True,
...     use_abs_color=False, # Color by velocity
...     title='Random Data Velocity Profile',
...     cmap='coolwarm',
...     s=40,
...     cbar=True
... )
>>> # plt.show() is called internally if savefig is None

**2. Concrete Example (Subsidence Data - adapted from docstring):**

>>> # Assume zhongshan_pred_2023_2026 is a loaded DataFrame like:
>>> # zhongshan_pred_2023_2026 = pd.DataFrame({
>>> #     'subsidence_2022_q50': np.random.rand(50)*5 + 5,
>>> #     'subsidence_2023_q50': np.random.rand(50)*6 + 6,
>>> #     'subsidence_2024_q50': np.random.rand(50)*7 + 7,
>>> #     'subsidence_2025_q50': np.random.rand(50)*8 + 8,
>>> #     'subsidence_2026_q50': np.random.rand(50)*9 + 9,
>>> #     'latitude': np.linspace(22.2, 22.8, 50)
>>> # }) # Dummy data for example execution
>>> # Create dummy data if zhongshan_pred_2023_2026 doesn't exist
>>> try:
...    zhongshan_pred_2023_2026
... except NameError:
...    print("Creating dummy subsidence data for example...")
...    zhongshan_pred_2023_2026 = pd.DataFrame({
...       'subsidence_2022_q50': np.random.rand(150)*5 + 5,
...       'subsidence_2023_q50': np.random.rand(150)*6 + 6 + np.linspace(0, 2, 150),
...       'subsidence_2024_q50': np.random.rand(150)*7 + 7 + np.linspace(0, 4, 150),
...       'subsidence_2025_q50': np.random.rand(150)*8 + 8 + np.linspace(0, 6, 150),
...       'subsidence_2026_q50': np.random.rand(150)*9 + 9 + np.linspace(0, 8, 150),
...       'latitude': np.linspace(22.2, 22.8, 150)
...     })
>>> subsidence_q50_cols = [
...     'subsidence_2022_q50', 'subsidence_2023_q50',
...     'subsidence_2024_q50', 'subsidence_2025_q50',
...     'subsidence_2026_q50',
... ]
>>> ax_subsidence = plot_velocity(
...     df=zhongshan_pred_2023_2026,
...     q50_cols=subsidence_q50_cols,
...     theta_col='latitude',       # Ignored for pos, triggers warning
...     acov='quarter_circle',      # Focus angular range
...     normalize=True,
...     use_abs_color=True,         # Color by Q50 magnitude
...     title='Subsidence Velocity Across Zhongshan (20222026)',
...     cmap='plasma',
...     s=25,
...     cbar=True,
...     mask_angle=True             # Hide angle labels
... )
>>> # plt.show() called internally

"""


@check_non_emptiness
@isdf
def plot_interval_consistency(
    df: pd.DataFrame,
    qlow_cols: list[str],
    qup_cols: list[str],
    q50_cols: list[str] | None = None,
    theta_col: str | None = None,
    use_cv: bool = True,
    cmap: str = "coolwarm",
    acov: str = "default",
    title: str | None = None,
    figsize: tuple[float, float] = (9, 9),
    s: float | int = 30,
    alpha: float = 0.85,
    show_grid: bool = True,
    mask_angle: bool = False,
    savefig: str | None = None,
):
    # --- Input Validation ---
    # Basic DataFrame checks handled by decorators @isdf @check_non_emptiness
    if len(qlow_cols) != len(qup_cols):
        raise ValueError(
            "Mismatch in length between `qlow_cols` "
            f"({len(qlow_cols)}) and `qup_cols` ({len(qup_cols)})."
        )
    if q50_cols is not None and len(qlow_cols) != len(q50_cols):
        raise ValueError(
            "Mismatch in length between quantile columns: "
            f"qlow/qup ({len(qlow_cols)}) and q50 ({len(q50_cols)})."
        )

    # Check if all specified columns exist in the DataFrame
    all_cols = qlow_cols + qup_cols + (q50_cols if q50_cols else [])
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following columns are missing from the DataFrame: "
            f"{', '.join(missing_cols)}"
        )

    # Check theta_col status and warn if provided but unused
    if theta_col is not None:
        if theta_col not in df.columns:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found in "
                f"DataFrame columns. Using index for angular position.",
                UserWarning,
                stacklevel=2,
            )
        else:
            # Issue warning as current implementation uses index
            warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but the current"
                f" implementation uses the DataFrame index for angular "
                f"positioning ('theta'). The column '{theta_col}' is "
                f"currently ignored for positioning.",
                UserWarning,
                stacklevel=2,
            )

    # --- Data Calculation ---
    # Calculate interval widths for each year/timepoint for all locations
    # widths shape: (M, N) where M=num_time_steps, N=num_locations
    try:
        widths = np.array(
            [
                df[qup].values - df[qlo].values
                for qlo, qup in zip(qlow_cols, qup_cols)
            ]
        )
    except Exception as e:
        raise TypeError(
            "Could not compute widths. Ensure quantile columns contain "
            "numeric data."
        ) from e

    # Calculate radial value 'r' (std dev or CV of widths over time)
    # Result shape: (N,)
    mean_widths = np.mean(widths, axis=0)
    std_widths = np.std(widths, axis=0)

    if use_cv:
        # Calculate Coefficient of Variation (CV)
        # Handle division by zero or near-zero mean width
        # Use np.divide for safe division, setting result to 0 where mean is ~0
        r = np.divide(
            std_widths,
            mean_widths,
            out=np.zeros_like(mean_widths, dtype=float),  # Output array
            where=np.abs(mean_widths) > 1e-9,
        )  # Condition for division
        # Optionally issue warning if division by zero occurred
        if np.any(np.abs(mean_widths) <= 1e-9):
            num_zeros = np.sum(np.abs(mean_widths) <= 1e-9)
            warnings.warn(
                f"Mean interval width was zero or near-zero for {num_zeros}"
                f" locations. CV is set to 0 for these locations.",
                RuntimeWarning,
                stacklevel=2,
            )
        radial_label = "CV of Interval Width (Q90-Q10)"
    else:
        # Use Standard Deviation
        r = std_widths
        radial_label = "Std Dev of Interval Width (Q90-Q10)"

    # Calculate color values
    if q50_cols:
        try:
            # Average Q50 across time for each location
            q50_values = np.array([df[q].values for q in q50_cols])
            color_vals_source = np.mean(q50_values, axis=0)
            cbar_label = "Average Q50 Prediction"
        except Exception as e:
            warnings.warn(
                f"Could not compute average Q50. Ensure Q50 columns contain"
                f" numeric data. Falling back to coloring by 'r'. Error: {e}",
                UserWarning,
                stacklevel=2,
            )
            # Fallback: color by the radial value itself
            color_vals_source = r
            cbar_label = radial_label  # Label reflects 'r'
    else:
        # Fallback: color by the radial value itself if no q50_cols given
        color_vals_source = r
        cbar_label = radial_label  # Label reflects 'r'

    # --- Angular Coordinate Calculation ---
    N = len(df)  # Number of locations
    # Generate linear space [0, 1] for N points
    theta_normalized = np.linspace(0, 1, N, endpoint=True)

    # Map normalized theta to the desired angular coverage
    angular_range_map = {
        "default": 2 * np.pi,
        "half_circle": np.pi,
        "quarter_circle": np.pi / 2,
        "eighth_circle": np.pi / 4,
    }
    angle_span = angular_range_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in angular_range_map:
        warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning,
            stacklevel=2,
        )
    # Calculate final theta values
    theta = theta_normalized * angle_span

    # --- Color Normalization ---
    # Ensure cmap is valid for fallback
    cmap_used = get_cmap(cmap, default="coolwarm")

    # Normalize color values for the colormap
    color_norm = Normalize(
        vmin=np.min(color_vals_source), vmax=np.max(color_vals_source)
    )
    # Get actual colors
    plot_colors = cmap_used(color_norm(color_vals_source))

    # --- Plotting ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )

    # Set angular limits
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(angle_span))

    # Create the polar scatter plot
    ax.scatter(
        theta,
        r,  # Radial value (CV or Std Dev)
        c=plot_colors,  # Point colors
        s=s,  # Point size
        edgecolor="k",  # Point edge color
        linewidth=0.5,  # Point edge width
        alpha=alpha,  # Point transparency
    )

    # Set plot title
    ax.set_title(
        title or "Prediction Interval Consistency", fontsize=14, y=1.08
    )

    # Add color bar
    sm = cm.ScalarMappable(norm=color_norm, cmap=cmap_used)
    sm.set_array([])  # Necessary for ScalarMappable
    cbar_obj = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar_obj.set_label(cbar_label, fontsize=10)

    # Customize grid and labels
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)

    # Optionally mask angular tick labels
    if mask_angle:
        ax.set_xticklabels([])

    # Set radial axis label
    ax.set_ylabel(radial_label, labelpad=15, fontsize=10)
    # Optional: adjust radial limits if needed, e.g., start at 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()  # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


plot_interval_consistency.__doc__ = r"""
Polar plot showing consistency of prediction interval widths.

This function generates a polar scatter plot to visualize the
temporal consistency (or variability) of prediction interval
widths (e.g., Q90 - Q10) across different locations over multiple
time steps or forecast horizons:

- The **angular position (`theta`)** represents each location,
  currently derived from the DataFrame index and mapped onto the
  specified angular coverage (`acov`).
- The **radial distance (`r`)** quantifies the inconsistency or
  variability of the interval width over time for each location. It
  is calculated as either the standard deviation (absolute
  variability) or the coefficient of variation (CV, relative
  variability) of the interval widths (Upper Quantile - Lower
  Quantile) across the specified time steps. Higher `r` values
  indicate locations where the predicted uncertainty range
  fluctuates more significantly over time.
- The **color** of each point typically represents the average
  median prediction (Q50) across the time steps (if `q50_cols`
  are provided). This adds context, helping to identify if interval
  inconsistency occurs in regions of high or low average predictions.
  If `q50_cols` are not provided, color defaults to representing the
  inconsistency measure `r`.

This plot is useful for diagnosing model reliability, identifying
locations or conditions where the model's uncertainty estimates
are unstable or vary considerably across different forecast
horizons.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the data. Must include columns
    specified in `qlow_cols` and `qup_cols`. Decorator `@isdf`
    ensures this is a pandas DataFrame. Decorator
    `@check_non_emptiness` ensures it's not empty.

qlow_cols : list of str
    List of column names representing the lower quantile (e.g., Q10)
    predictions for consecutive time steps (e.g., years). Order
    should correspond to the time steps. Example:
    ``['subsidence_2023_q10', 'subsidence_2024_q10', ...]``.

qup_cols : list of str
    List of column names representing the upper quantile (e.g., Q90)
    predictions for the same consecutive time steps as `qlow_cols`.
    Must be the same length as `qlow_cols`. Example:
    ``['subsidence_2023_q90', 'subsidence_2024_q90', ...]``.

q50_cols : list of str, optional
    List of column names representing the median quantile (Q50)
    predictions for the same time steps. If provided, the average
    Q50 value across these columns will be used to color the points.
    Must be the same length as `qlow_cols` if provided. If ``None``,
    the color will represent the radial value `r` (the inconsistency
    measure). Default is ``None``.

theta_col : str, optional
    *Intended* column name to determine the angular position (`theta`)
    for each location (e.g., 'latitude', 'longitude', or a spatial
    index). If ``None``, the DataFrame index is conceptually used.
    *Note: The current implementation maps the DataFrame row index
    to the angular range specified by `acov`, regardless of whether
    `theta_col` is provided. Providing `theta_col` will currently
    trigger a warning but will not affect the plot's angular axis.*
    Default is ``None``.

use_cv : bool, default=True
    Determines the measure of interval width variability used for the
    radial coordinate `r`:

    - If ``True``, `r` is the Coefficient of Variation (CV) of the
      interval widths (Std Dev / Mean). CV measures relative
      variability, useful when mean widths differ substantially.
    - If ``False``, `r` is the Standard Deviation (Std Dev) of the
      interval widths. Std Dev measures absolute variability.

cmap : str, default='coolwarm'
    The name of the Matplotlib colormap used to color the scatter
    points based on the average Q50 value (or `r` if `q50_cols` is
    ``None``).

acov : str, default='default'
    Angular coverage defining the span of the polar plot's theta
    axis. Options: ``'default'`` (360°), ``'half_circle'`` (180°),
    ``'quarter_circle'`` (90°), ``'eighth_circle'`` (45°). Invalid
    options default to ``'default'``.

title : str, optional
    The title displayed above the polar plot. If ``None``, a default
    title like "Prediction Interval Consistency (Q90Q10)" is used.
    Default is ``None``.

figsize : tuple of (float, float), default=(9, 9)
    The width and height of the figure in inches.

s : float or int, default=30
    The marker size for the scatter points.

alpha : float, default=0.85
    The transparency level of the scatter points (0=transparent,
    1=opaque).

show_grid : bool, default=True
    If ``True``, display the polar grid lines.

mask_angle : bool, default=False
    If ``True``, hide the angular tick labels. Useful if the index-
    based angle is not directly interpretable.

savefig : str, optional
    File path to save the plot image. If ``None``, displays the plot
    interactively. Default is ``None``.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the polar scatter plot.

Raises
------
AssertionError
    If `qlow_cols` and `qup_cols` have different lengths.
ValueError
    If specified columns in `qlow_cols`, `qup_cols`, or `q50_cols`
    are not found in the DataFrame.

See Also
--------
plot_velocity : Plot average velocity in polar coordinates.
numpy.std : Compute the standard deviation.
numpy.mean : Compute the arithmetic mean.
matplotlib.pyplot.scatter : Create scatter plots.

Notes
-------

Interval-width consistency is assessed from paired lower/upper quantiles
for multiple time steps. For each location and time step, the width is
computed as upper minus lower. The radial value encodes either the
standard deviation of these widths (absolute variability) or their
coefficient of variation (relative variability), with safe handling of
zero means by setting the CV to zero when the average width is
numerically indistinguishable from zero. Angles are derived from the row
index and mapped linearly across the angular span determined by
:py:data:`acov`; the current implementation does not use
:py:data:`theta_col` for positioning. Rows containing missing values in
any required column are dropped prior to computation. These diagnostics
relate to standard notions of predictive-interval calibration and
stability; see :footcite:t:`Gneiting2007b, Jolliffe2012`.

**Interval widths.** Let :math:`\mathbf L` and :math:`\mathbf U` be
matrices extracted from :py:data:`df` using :py:data:`qlow_cols` and
:py:data:`qup_cols`, respectively, both of shape :math:`(N,M)`, with
:math:`N` locations and :math:`M` time steps. Define the width matrix
:math:`\mathbf W` by

.. math::
    W_{j,i} = U_{j,i} - L_{j,i}
    
where :math:`j` indexes locations (:math:`0` to :math:`N-1`) and
:math:`i` indexes time steps (:math:`0` to :math:`M-1`).

**Radial Coordinate Calculation (`r`)**: Let :math:`\mathbf{w}_j =
(W_{j,0}, \dots, W_{j,M-1})` be the vector of widths over time
for location :math:`j`. Let :math:`\bar{w}_j = \text{mean}(\mathbf{w}_j)`
and :math:`\sigma_{w_j} = \text{std}(\mathbf{w}_j)`.

- If `use_cv=False` (Standard Deviation):

  .. math::
      r_j = \sigma_{w_j}

- If `use_cv=True` (Coefficient of Variation):

  .. math::
      r_j = \begin{cases} \frac{\sigma_{w_j}}{\bar{w}_j} &\\
          \text{if } |\bar{w}_j| > \epsilon \\ 0 & \text{if }\\
              |\bar{w}_j| \le \epsilon \end{cases}
              
  where :math:`\epsilon` is a small threshold to prevent division
  by zero.

**Color Value Calculation (`c`)**: Let :math:`\mathbf{Q50}` be the
data matrix (shape :math:`(N, M)`) from `q50_cols`.

- If `q50_cols` is provided: Let :math:`\mathbf{q50}_j =
  (Q50_{j,0}, \dots, Q50_{j,M-1})`.

  .. math::
      c_j = \text{mean}(\mathbf{q50}_j) = \frac{1}{M} \sum_{i=0}^{M-1} Q50_{j,i}
      
- If `q50_cols` is ``None``:
    
  :math:`c_j = r_j`

**Angular Coordinate Calculation (`theta`)**:
Same index-based calculation as `plot_velocity`. Let :math:`S` be
the angular span from `acov`.

.. math::
     \theta_j = \frac{j}{N} \times S

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.uncertainty import plot_interval_consistency

**1. Random Example:**

>>> np.random.seed(1)
>>> N_points = 120
>>> df_rand_interval = pd.DataFrame({
...     'id': range(N_points),
...     'lat': np.linspace(30, 31, N_points),
...     'val_2021_q10': np.random.rand(N_points) * 5,
...     'val_2021_q50': np.random.rand(N_points) * 5 + 5,
...     'val_2021_q90': np.random.rand(N_points) * 5 + 10,
...     'val_2022_q10': np.random.rand(N_points) * 6, # Slightly wider
...     'val_2022_q50': np.random.rand(N_points) * 6 + 6,
...     'val_2022_q90': np.random.rand(N_points) * 6 + 12,
...     'val_2023_q10': np.random.rand(N_points) * 4, # Narrower
...     'val_2023_q50': np.random.rand(N_points) * 4 + 7,
...     'val_2023_q90': np.random.rand(N_points) * 4 + 11,
... })
>>> q10_cols_rand = ['val_2021_q10', 'val_2022_q10', 'val_2023_q10']
>>> q90_cols_rand = ['val_2021_q90', 'val_2022_q90', 'val_2023_q90']
>>> q50_cols_rand = ['val_2021_q50', 'val_2022_q50', 'val_2023_q50']
>>> ax_rand_ic = plot_interval_consistency(
...     df=df_rand_interval,
...     qlow_cols=q10_cols_rand,
...     qup_cols=q90_cols_rand,
...     q50_cols=q50_cols_rand,
...     theta_col='lat',      # Note: Ignored for positioning
...     use_cv=True,          # Use CV for radial axis
...     cmap='viridis',
...     acov='half_circle',
...     title='Random Interval Width Consistency (CV)',
...     s=35
... )
>>> # plt.show() called internally

**2. Concrete Example (Subsidence Data - adapted from docstring):**

>>> # Assume zhongshan_pred_2023_2026 is loaded DataFrame like:
>>> # Create dummy data if it doesn't exist
>>> try:
...    zhongshan_pred_2023_2026
... except NameError:
...    print("Creating dummy subsidence data for example...")
...    N_sub = 150
...    zhongshan_pred_2023_2026 = pd.DataFrame({
...       'latitude': np.linspace(22.2, 22.8, N_sub),
...       **{f'subsidence_{yr}_q10': np.random.rand(N_sub)*(yr-2020)+1
...          for yr in range(2023, 2027)},
...       **{f'subsidence_{yr}_q50': np.random.rand(N_sub)*(yr-2019)+5
...          + np.linspace(0, (yr-2022)*2, N_sub)
...          for yr in range(2023, 2027)},
...       **{f'subsidence_{yr}_q90': np.random.rand(N_sub)*(yr-2018)+10
...          + np.linspace(0, (yr-2022)*4, N_sub)
...          for yr in range(2023, 2027)},
...     })
>>> qlow_sub = [f'subsidence_{yr}_q10' for yr in range(2023, 2027)]
>>> qup_sub = [f'subsidence_{yr}_q90' for yr in range(2023, 2027)]
>>> q50_sub = [f'subsidence_{yr}_q50' for yr in range(2023, 2027)]
>>> ax_sub_ic = plot_interval_consistency(
...     df=zhongshan_pred_2023_2026,
...     qlow_cols=qlow_sub,
...     qup_cols=qup_sub,
...     q50_cols=q50_sub,
...     theta_col='latitude',    # Ignored for pos, triggers warning
...     acov='default',
...     title='Subsidence Uncertainty Consistency (20232026)',
...     use_cv=False,            # Use Std Dev for radius
...     cmap='coolwarm',
...     s=28,
...     alpha=0.8,
...     mask_angle=True
... )
>>> 

References
----------
.. footbibliography::
    
"""


@check_non_emptiness
@isdf
def plot_anomaly_magnitude(
    df: pd.DataFrame,
    actual_col: str,
    q_cols: list[str] | tuple[str, str],
    theta_col: str | None = None,
    acov: str = "default",
    title: str = "Anomaly Magnitude Polar Plot",
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap_under: str = "Blues",
    cmap_over: str = "Reds",
    s: int = 30,
    alpha: float = 0.8,
    show_grid: bool = True,
    verbose: int = 1,
    cbar: bool = False,
    savefig: str | None = None,
    mask_angle: bool = False,
):
    try:
        qlow_col, qup_col = validate_qcols(
            q_cols=q_cols,
            ncols_exp="==2",  # Expect exactly two columns
            err_msg=(
                "Expected `q_cols` to contain exactly two column names "
                f"[lower_bound, upper_bound], but got: {q_cols}"
            ),
        )
    except Exception as e:
        # Catch potential errors from validate_qcols if it raises them
        raise ValueError(f"Validation of `q_cols` failed: {e}") from e

    # Consolidate list of essential columns
    cols_needed = [actual_col, qlow_col, qup_col]
    if theta_col:
        # Only add if specified, check existence later if needed
        cols_needed.append(theta_col)

    # Check existence of essential columns
    # missing_cols = [col for col in cols_needed if col not in df.columns]
    # Allow theta_col to be missing if specified, handled later
    missing_essential = [
        col
        for col in [actual_col, qlow_col, qup_col]
        if col not in df.columns
    ]
    if missing_essential:
        raise ValueError(
            "The following essential columns are missing from the "
            f"DataFrame: {', '.join(missing_essential)}"
        )

    # Drop rows with NaN in essential columns before proceeding
    # Use only the definitely required columns for dropna
    essential_cols_for_na = [actual_col, qlow_col, qup_col]
    data = df[cols_needed].dropna(subset=essential_cols_for_na).copy()
    if len(data) == 0:
        warnings.warn(
            "DataFrame is empty after dropping NaN values"
            " in essential columns. Cannot generate plot.",
            UserWarning,
            stacklevel=2,
        )
        return None  # Cannot proceed

    # --- Anomaly Calculation ---
    # Extract data as numpy arrays for efficiency
    try:
        y = data[actual_col].to_numpy(dtype=float)
        y_lo = data[qlow_col].to_numpy(dtype=float)
        y_hi = data[qup_col].to_numpy(dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Failed to convert essential columns to numeric arrays."
            f" Check data types. Original error: {e}"
        ) from e

    # Identify under- and over-predictions
    under_mask = y < y_lo
    over_mask = y > y_hi

    # Calculate anomaly magnitude (distance from the violated bound)
    anomaly_mag = np.zeros_like(y, dtype=float)
    # Magnitude is positive: bound - actual for under, actual - bound for over
    anomaly_mag[under_mask] = y_lo[under_mask] - y[under_mask]
    anomaly_mag[over_mask] = y[over_mask] - y_hi[over_mask]

    # Filter out non-anomalies for plotting (only plot r > 0)
    is_anomaly = under_mask | over_mask
    if not np.any(is_anomaly):
        warnings.warn(
            "No anomalies detected (all actual values are within "
            "the specified quantile bounds). Plot will be empty.",
            UserWarning,
            stacklevel=2,
        )
        # Still create plot structure, but it will be empty
    # Filter data to only include anomalies
    anomaly_mag = anomaly_mag[is_anomaly]
    under_mask_filtered = under_mask[is_anomaly]
    over_mask_filtered = over_mask[is_anomaly]
    data_filtered = data[is_anomaly]  # Filter DataFrame rows too
    N_anomalies = len(data_filtered)  # Number of anomalies

    # --- Theta Coordinate and Ordering ---
    # Determine ordering index
    if theta_col and theta_col in data_filtered.columns:
        try:
            # Sort based on the theta_col values of the anomalies
            ordered_idx = np.argsort(
                data_filtered[theta_col].to_numpy(dtype=float)
            )
        except (ValueError, TypeError):
            warnings.warn(
                f"Could not sort by `theta_col` ('{theta_col}') as it "
                f"contains non-numeric data after NaN removal. "
                f"Using default DataFrame order.",
                UserWarning,
                stacklevel=2,
            )
            ordered_idx = np.arange(N_anomalies)  # Fallback to original order
    else:
        # Use default order if theta_col not specified or not found
        if theta_col and theta_col not in data_filtered.columns:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found in the"
                f" (filtered) DataFrame. Using default DataFrame order.",
                UserWarning,
                stacklevel=2,
            )
        ordered_idx = np.arange(N_anomalies)

    # Generate base theta values (linear spacing)
    theta_norm = np.linspace(0.0, 1.0, N_anomalies, endpoint=True)

    # Define angular coverage range
    coverage_map = {
        "default": 2 * np.pi,
        "half_circle": np.pi,
        "quarter_circle": np.pi / 2,
        "eighth_circle": np.pi / 4,
    }
    coverage = coverage_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in coverage_map:
        warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning,
            stacklevel=2,
        )

    # Calculate final theta values based on coverage
    theta = theta_norm * coverage

    # Apply ordering to all relevant arrays
    theta = theta[ordered_idx]
    anomaly_mag_ordered = anomaly_mag[ordered_idx]
    under_mask_ordered = under_mask_filtered[ordered_idx]
    over_mask_ordered = over_mask_filtered[ordered_idx]

    # --- Plotting ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(coverage))  # Expects degrees
    ax.set_title(title, fontsize=14, y=1.08)  # Adjust position

    # Setup grid
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)

    # Optionally mask angle labels
    if mask_angle:
        ax.set_xticklabels([])

    # --- Color Normalization (shared scale based on max magnitude) ---
    # Ensure vmax is at least a small positive number for normalization
    vmax = max(
        float(anomaly_mag_ordered.max()) if N_anomalies > 0 else 0.0, 1e-5
    )
    norm = Normalize(vmin=0.0, vmax=vmax)

    # Retrieve colormaps safely
    try:
        cmap_under_obj = get_cmap(cmap_under)
    except ValueError:
        warnings.warn(
            f"Invalid `cmap_under` ('{cmap_under}'). "
            f"Using default 'Blues'.",
            UserWarning,
            stacklevel=2,
        )
        cmap_under_obj = get_cmap("Blues")
    try:
        cmap_over_obj = get_cmap(cmap_over)
    except ValueError:
        warnings.warn(
            f"Invalid `cmap_over` ('{cmap_over}'). " f"Using default 'Reds'.",
            UserWarning,
            stacklevel=2,
        )
        cmap_over_obj = get_cmap("Reds")

    # --- Scatter Plot Anomalies (separate calls for color/label) ---
    if np.any(under_mask_ordered):
        ax.scatter(
            theta[under_mask_ordered],  # Angles for under-preds
            anomaly_mag_ordered[
                under_mask_ordered
            ],  # Magnitudes for under-preds
            c=anomaly_mag_ordered[
                under_mask_ordered
            ],  # Color value is magnitude
            cmap=cmap_under_obj,  # Blues colormap
            norm=norm,  # Shared normalization
            s=s,  # Marker size
            alpha=alpha,  # Transparency
            edgecolor="grey",  # Edge color
            linewidth=0.5,
            label="Under-prediction",  # Legend label
        )

    if np.any(over_mask_ordered):
        ax.scatter(
            theta[over_mask_ordered],  # Angles for over-preds
            anomaly_mag_ordered[
                over_mask_ordered
            ],  # Magnitudes for over-preds
            c=anomaly_mag_ordered[
                over_mask_ordered
            ],  # Color value is magnitude
            cmap=cmap_over_obj,  # Reds colormap
            norm=norm,  # Shared normalization
            s=s,  # Marker size
            alpha=alpha,  # Transparency
            edgecolor="grey",  # Edge color
            linewidth=0.5,
            label="Over-prediction",  # Legend label
        )

    # Add legend if any anomalies were plotted
    if np.any(under_mask_ordered) or np.any(over_mask_ordered):
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    else:
        # Add text if plot is empty
        ax.text(
            0,
            0,
            "No anomalies detected",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # --- Add Colorbar (optional) ---
    # Note: Visually uses cmap_over, but scale (norm) is correct
    if cbar and N_anomalies > 0:
        # Create a mappable object linked to the normalization and cmap_over
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_over_obj)
        sm.set_array([])  # Needed for ScalarMappable
        cbar_obj = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
        cbar_obj.set_label("Anomaly magnitude |Actual - Bound|", fontsize=10)

    # Set radial axis label
    ax.set_ylabel("Anomaly Magnitude", labelpad=15, fontsize=10)
    ax.set_ylim(bottom=0)  # Ensure radius starts at 0

    # --- Logging ---
    if verbose > 0:
        # Use original masks on NaN-dropped data before filtering for plot
        n_total_checked = len(data)  # Total valid points checked
        n_under_total = np.sum(y < y_lo)
        n_over_total = np.sum(y > y_hi)
        n_anomalies_total = n_under_total + n_over_total
        print("-" * 50)
        print("Anomaly Detection Summary:")
        print(f"  Total valid points checked: {n_total_checked}")
        print(
            f"  Anomalies detected (outside {qlow_col}-{qup_col}): "
            f"{n_anomalies_total}"
        )
        print(
            f"  ? Under-predictions ({actual_col} < {qlow_col}): {n_under_total}"
        )
        print(
            f"  ? Over-predictions  ({actual_col} > {qup_col}): {n_over_total}"
        )
        print("-" * 50)

    # --- Output ---
    plt.tight_layout()
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


plot_anomaly_magnitude.__doc__ = r"""
Visualize magnitude and type of prediction-interval anomalies.

This function draws a polar scatter plot to highlight prediction-interval
failures—cases where the observed value lies outside a user-specified
interval (e.g., :math:`[Q_{0.10}, Q_{0.90}]`). It encodes the anomaly’s
**order/location** (angle), **magnitude** (radius), and **type** (color).
See :footcite:t:`kouadiob2025`; background on coverage and calibration in
:footcite:p:`Gneiting2007b, Jolliffe2012`.

- **Angular position (:math:`\theta`)**: maps each row to an angle over the
  chosen span (``acov``). If ``theta_col`` is provided, rows are *ordered*
  by that column before angles are assigned; otherwise the original row
  order is used. Angles are spaced linearly across the selected coverage.

- **Radial distance (:math:`r`)**: the anomaly magnitude for interval
  violations, computed as :math:`r=\lvert y - B\rvert`, where
  :math:`B\in\{L,U\}` is the violated bound (lower :math:`L` or upper
  :math:`U`). Points satisfying :math:`L\le y\le U` are omitted.

- **Color**: indicates anomaly type and scales with magnitude on a shared
  normalization. ``cmap_under`` colors under-predictions
  (:math:`y<L`); ``cmap_over`` colors over-predictions (:math:`y>U`).

This diagnostic helps to (i) localize clusters of failures along the
ordering induced by ``theta_col`` (or row index), (ii) assess how severe
misses are via larger radii, and (iii) support calibration checks and
targeted model refinement, alongside coverage and reliability analyses,
see :footcite:p:`Gneiting2007b, Jolliffe2012`.

Parameters
----------
df : pandas.DataFrame
    Input table containing the actual values and the two quantile
    (interval) columns. Must be non-empty after NaN removal in the
    required columns (see below).

actual_col : str
    Name of the column with the observed/ground-truth values used to
    check interval coverage.

q_cols : list[str] or tuple[str, str]
    Two-element sequence with the **lower** and **upper** quantile
    column names, in that order: ``[q_low, q_up]``. Each referenced
    column must exist in ``df`` and be numeric. Semantically, rows are
    expected to satisfy :math:`q_\text{low} \le q_\text{up}`.

theta_col : str, optional
    Column used **only to order points** angularly before mapping them
    into the selected coverage span. Useful for spatial or categorical
    ordering (e.g., ``'latitude'``, ``'station_id'``). If omitted or
    non-numeric after NaN filtering, the original row order is used.

acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'}, \
    default='default'
    Angular **coverage** of the plot:
        
    - ``'default'``: full circle, :math:`2\pi`
    - ``'half_circle'``: :math:`\pi`
    - ``'quarter_circle'``: :math:`\pi/2`
    - ``'eighth_circle'``: :math:`\pi/4`
    
    (Value is case-insensitive; invalid values fall back to full circle
    with a warning.)

title : str, default='Anomaly Magnitude Polar Plot'
    Figure title.

figsize : tuple[float, float], default=(8.0, 8.0)
    Figure size in inches; each dimension must be positive.

cmap_under : str, default='Blues'
    Matplotlib colormap **name** used for *under-predictions*
    (:math:`y<q_\text{low}`). If invalid, a warning is issued and a
    default is used.

cmap_over : str, default='Reds'
    Matplotlib colormap **name** used for *over-predictions*
    (:math:`y>q_\text{up}`). If invalid, a warning is issued and a
    default is used.

s : int, default=30
    Marker size for anomaly points (points with interval failure);
    must be positive.

alpha : float, default=0.8
    Point transparency in :math:`[0,1]`.

show_grid : bool, default=True
    Whether to draw polar grid lines.

verbose : int, default=1
    Verbosity level. If :math:`>0`, prints a short anomaly summary
    (counts of under-/over-predictions).

cbar : bool, default=False
    If ``True``, draw a colorbar for the **shared** magnitude
    normalization. The bar uses the ``cmap_over`` colormap for display,
    but its scale matches both under/over magnitudes.

savefig : str, optional
    Path to save the figure (e.g., ``'anomaly_plot.png'``). If omitted,
    the figure is shown interactively. Errors during saving are reported
    via a printed message.

mask_angle : bool, default=False
    If ``True``, hide angular tick labels (useful when the angle order
    is arbitrary or index-based).

Returns
-------
ax : matplotlib.axes.Axes or None
    The polar ``Axes`` containing the scatter plot. Returns ``None`` if
    the DataFrame becomes empty **after** dropping NaNs in the required
    columns and no plot can be produced. If no anomalies are found, an
    empty polar frame is returned with a notice text, not ``None``.

Raises
------
ValueError
    - ``q_cols`` is not a 2-item sequence.
    - Any of ``actual_col``, the two ``q_cols``, or the provided
      ``theta_col`` (when used) is missing from ``df``.

TypeError
    Required columns exist but cannot be coerced to numeric dtype after
    NaN handling.

See Also
--------
kdiagram.plot.uncertainty.plot_coverage :
    Aggregate empirical coverage comparison.
kdiagram.plot.uncertainty.plot_coverage_diagnostic :
    Point-wise coverage successes/failures.
kdiagram.plot.uncertainty.plot_interval_width :
    Magnitude of prediction-interval widths.
kdiagram.plot.uncertainty.plot_interval_consistency :
    Stability of interval widths over time/steps.
validate_qcols :
    Helper for validating the lower/upper quantile columns.

Notes
-----
- Anomalies are detected by comparing ``actual_col`` to the bounds
  in ``q_cols`` for each row.

- Rows with NaNs in any required column (``actual_col``, the two
  quantile columns, and ``theta_col`` if provided) are dropped
  before analysis.

- The anomaly magnitude :math:`r` is non-negative and measures the
  distance to the **exceeded** bound.

- ``theta_col`` controls **ordering only**, not spacing. Angles are
  always linearly spaced within the selected coverage (:py:data:`acov`)
  and then assigned according to the sort order of ``theta_col``.

- Color intensity for both under- and over-predictions reflects the
  shared normalization of magnitudes, based on the maximum observed
  :math:`r`.

- The helper ``validate_qcols`` (from your utilities, e.g. *gofast*)
  performs initial structure checks on ``q_cols``.

Let :math:`y_j` be the actual value, :math:`L_j` the lower bound,
and :math:`U_j` the upper bound for location :math:`j`.

1. **Anomaly masks**

   - Under-prediction:
     :math:`M_{\text{under},j} = (y_j < L_j)`

   - Over-prediction:
     :math:`M_{\text{over},j} = (y_j > U_j)`

2. **Anomaly magnitude (radial coordinate :math:`r`)**

   .. math::

      r_j =
      \begin{cases}
      L_j - y_j, & \text{if } y_j < L_j,\\
      y_j - U_j, & \text{if } y_j > U_j,\\
      0,         & \text{otherwise.}
      \end{cases}

   Only points with :math:`r_j > 0` are plotted.

3. **Angular coordinate (:math:`\theta`)**

   - For :math:`N` retained rows, define base angles

     .. math::

        \theta'_k \;=\; \frac{k}{N}\,S, \qquad k=0,\ldots,N-1,

     where :math:`S \in \{2\pi, \pi, \pi/2, \pi/4\}` is the span
     implied by :py:data:`acov`.

   - If ``theta_col`` exists, sort by that column to obtain a
     permutation :math:`\pi`. The plotted angle for original row
     :math:`j` is :math:`\theta_{\pi^{-1}(j)}=\theta'_{\pi^{-1}(j)}`.
     If ``theta_col`` is absent, use the original row order:
     :math:`\theta_j=\theta'_j`.

4. **Color mapping**

   - Normalize magnitudes

     .. math::

        \text{norm}(r_j) \;=\;
        \frac{r_j}{\max(r_1,\ldots,r_N) + \varepsilon},

     with a small :math:`\varepsilon>0` for numerical safety.

   - Apply colormaps: ``cmap_under(norm(r_j))`` for under-predictions
     and ``cmap_over(norm(r_j))`` for over-predictions.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.uncertainty import plot_anomaly_magnitude

**1. Random Example:**

>>> np.random.seed(42)
>>> N_points = 150
>>> df_anomaly_rand = pd.DataFrame({
...     'id': range(N_points),
...     'actual': np.random.randn(N_points) * 5 + 10,
...     'pred_q10': np.random.randn(N_points) * 1 + 7, # Interval around 10
...     'pred_q90': np.random.randn(N_points) * 1 + 13,
...     'feature_order': np.random.rand(N_points) * 100 # For ordering
... })
>>> # Introduce some anomalies
>>> df_anomaly_rand.loc[5:15, 'actual'] = 0 # Under-predictions
>>> df_anomaly_rand.loc[100:110, 'actual'] = 25 # Over-predictions
>>>
>>> ax_rand_anomaly = plot_anomaly_magnitude(
...     df=df_anomaly_rand,
...     actual_col='actual',
...     q_cols=['pred_q10', 'pred_q90'],
...     theta_col='feature_order', # Order by this feature
...     acov='default',
...     title='Random Anomaly Distribution',
...     cmap_under='GnBu',
...     cmap_over='OrRd',
...     s=40,
...     cbar=True,
...     verbose=1
... )
>>> # Output will show anomaly counts...
>>> # plt.show() called internally

**2. Concrete Example (Subsidence Data - adapted from docstring):**

>>> # Assume small_sample_pred is a loaded DataFrame like:
>>> # Create dummy data if it doesn't exist
>>> try:
...    small_sample_pred
... except NameError:
...    print("Creating dummy small sample prediction data...")
...    N_small = 200
...    small_sample_pred = pd.DataFrame({
...        'subsidence_2023': np.random.rand(N_small)*15 + np.linspace(0, 5, N_small),
...        'subsidence_2023_q10': np.random.rand(N_small)*10,
...        'subsidence_2023_q90': np.random.rand(N_small)*10 + 10,
...        'latitude': np.linspace(22.3, 22.7, N_small) + np.random.randn(N_small)*0.01
...     })
...     # Ensure some anomalies exist in dummy data
...     anom_indices_under = np.random.choice(N_small, 15, replace=False)
...     anom_indices_over = np.random.choice(
...         list(set(range(N_small)) - set(anom_indices_under)), 20, replace=False
...     )
...     small_sample_pred.loc[anom_indices_under, 'subsidence_2023'] = (
...         small_sample_pred.loc[anom_indices_under, 'subsidence_2023_q10']
...         - np.random.rand(15)*5 - 1
...         )
...     small_sample_pred.loc[anom_indices_over, 'subsidence_2023'] = (
...         small_sample_pred.loc[anom_indices_over, 'subsidence_2023_q90']
...         + np.random.rand(20)*5 + 1
...         )

>>> ax_sub_anomaly = plot_anomaly_magnitude(
...     df=small_sample_pred,
...     actual_col='subsidence_2023',
...     q_cols=['subsidence_2023_q10', 'subsidence_2023_q90'],
...     theta_col='latitude',      # Order points by latitude
...     acov='quarter_circle',   # Use only 90 degrees
...     title='Anomaly Magnitude (2023)  Zhongshan',
...     figsize=(9, 9),
...     s=35,
...     cbar=True,               # Show colorbar
...     mask_angle=True,         # Hide angle labels
...     verbose=1                # Print anomaly counts
... )
>>> # Output will show anomaly counts...
>>> # plt.show() called internally

References
----------
.. footbibliography::
"""


@check_non_emptiness
@isdf
def plot_uncertainty_drift(
    df: pd.DataFrame,
    qlow_cols: list[str],
    qup_cols: list[str],
    dt_labels: list[str] | None = None,
    theta_col: str | None = None,
    acov: str = "default",
    base_radius: float = 0.15,
    band_height: float = 0.15,
    cmap: str = "tab10",
    label: str = "Year",
    alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 9),
    title: str | None = None,
    show_grid: bool = True,
    show_legend: bool = True,
    mask_angle: bool = True,
    savefig: str | None = None,
):
    # --- Input Validation ---
    if len(qlow_cols) != len(qup_cols):
        raise ValueError(
            "Mismatched lengths for `qlow_cols` "
            f"({len(qlow_cols)}) and `qup_cols` ({len(qup_cols)})."
        )
    num_time_steps = len(qlow_cols)
    if num_time_steps == 0:
        raise ValueError("Quantile column lists cannot be empty.")

    # Generate default labels if none provided
    if dt_labels is None:
        # Use the 'label' param as base for default labels
        time_labels = [f"{label}_{i+1}" for i in range(num_time_steps)]
    else:
        if len(dt_labels) != num_time_steps:
            raise ValueError(
                f"Length of `dt_labels` ({len(dt_labels)}) must match "
                f"the number of time steps ({num_time_steps})."
            )
        time_labels = list(dt_labels)  # Ensure list type

    # Consolidate required columns and check existence
    all_cols = qlow_cols + qup_cols
    # Do not check theta_col here, handle warning later
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following quantile columns are missing from the "
            f"DataFrame: {', '.join(missing_cols)}"
        )

    # Handle theta_col warning
    if theta_col:
        if theta_col not in df.columns:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found. "
                f"Using index for angular position.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but currently "
                f"ignored for positioning. Using index for angular position.",
                UserWarning,
                stacklevel=2,
            )

    # Prepare data: Drop rows with NaNs in relevant columns
    data = df[all_cols].dropna()
    if len(data) == 0:
        warnings.warn(
            "DataFrame is empty after dropping NaN values in quantile "
            "columns. Cannot generate plot.",
            UserWarning,
            stacklevel=2,
        )
        return None
    N = len(data)  # Number of valid data points (locations)

    # --- Calculate Interval Widths and Normalize Globally ---
    widths = []
    try:
        for ql, qu in zip(qlow_cols, qup_cols):
            width_values = (data[qu] - data[ql]).to_numpy(dtype=float)
            if np.any(width_values < 0):
                warnings.warn(
                    f"Negative interval widths detected for columns "
                    f"'{qu}' and '{ql}'. Check if upper < lower bound.",
                    UserWarning,
                    stacklevel=2,
                )
            # Ensure non-negative widths, clamp if necessary? Or just proceed.
            # width_values[width_values < 0] = 0 # Option to clamp
            widths.append(width_values)
    except Exception as e:
        raise TypeError(
            f"Could not compute widths. Ensure quantile columns "
            f"({ql}, {qu}) contain numeric data. Original error: {e}"
        ) from e

    # Find global maximum width across all years and locations
    if not widths:  # Should not happen due to earlier checks, but safe
        return None
    # Calculate max, handle case where all widths might be zero or negative
    all_width_values = np.concatenate(widths) if widths else np.array([0])
    max_width = np.max(all_width_values) if len(all_width_values) > 0 else 0.0

    # Normalize widths using the global maximum
    normalized_widths = []
    if max_width > 1e-9:  # Avoid division by zero/near-zero
        normalized_widths = [w / max_width for w in widths]
    else:
        # If max width is zero, all normalized widths are zero
        normalized_widths = [np.zeros_like(w) for w in widths]
        warnings.warn(
            "Maximum interval width across all data is zero or near-zero. "
            "Normalized widths are all set to 0.",
            UserWarning,
            stacklevel=2,
        )

    # --- Angular Coordinate Calculation ---
    acov_map = {  # Map coverage name to (min_angle, max_angle) in radians
        "default": (0, 2 * np.pi),
        "half_circle": (0, np.pi),
        "quarter_circle": (0, np.pi / 2),
        "eighth_circle": (0, np.pi / 4),
    }
    theta_min_rad, theta_max_rad = acov_map.get(acov.lower(), (0, 2 * np.pi))
    if acov.lower() not in acov_map:
        warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (0 to 2*pi).",
            UserWarning,
            stacklevel=2,
        )
    angular_range_rad = theta_max_rad - theta_min_rad

    # Generate theta values based on index, mapped to the specified range
    theta = (
        np.linspace(0.0, 1.0, N, endpoint=True)  # Linear space [0, 1]
        * angular_range_rad  # Scale to range width
        + theta_min_rad
    )

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    ax.set_thetamin(np.degrees(theta_min_rad))  # Expects degrees
    ax.set_thetamax(np.degrees(theta_max_rad))  # Expects degrees

    # Hide angular tick labels if requested
    if mask_angle:
        ax.set_xticklabels([])
    # Configure grid
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)
    # Hide radial ticks as they primarily separate years visually
    ax.set_yticks([])

    # Get color palette
    color_palette = _sample_colors(
        cmap, num_time_steps, trim=0.08, default="tab10"
    )

    # --- Draw Rings for Each Time Step ---
    for i, (w_norm, step_label) in enumerate(
        zip(normalized_widths, time_labels)
    ):
        # XXX TODO
        # Calculate base radius for this ring (increases for later years)
        # base_r = base_radius + i * some_increment # Alternative logic
        base_r = base_radius * (
            i + 1
        )  # Base offset increases multiplicatively

        # Calculate final radius: base + scaled normalized width
        # Ensure w_norm has the same length as theta (N)
        if len(w_norm) != N:
            # This should not happen if dropna was done correctly, but safeguard
            raise InternalError(
                "Mismatch between width data and theta length."
            )  # Or handle gracefully

        r = base_r + band_height * w_norm

        # Determine color for this ring, cycling if needed
        color = color_palette[i % len(color_palette)]

        # Plot the line for this time step
        # Ensure data wraps around by appending first point? Not needed
        # for line plot if endpoint=True in linspace? Check.
        # For visual continuity if range isn't full 2*pi, might not need wrap.
        # Let's omit wrap for now.
        ax.plot(
            theta,
            r,
            label=step_label,  # Label for the legend
            color=color,  # Color for this ring
            linewidth=1.8,  # Line thickness
            alpha=alpha,  # Transparency
        )

    # --- Final Touches ---
    # Set plot title
    ax.set_title(
        title or "Multi-time Uncertainty Drift (Interval Width)",
        fontsize=14,
        y=1.08,
    )

    # Add legend if requested
    if show_legend:
        # Use the provided 'label' parameter as the intended legend title
        # The example hardcodes 'Year'. Let's compromise:
        # use 'label' if given, else 'Time Step'.
        legend_title = label if label else "Time Step"
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.25, 1.1),  # Position outside plot
            title=legend_title,  # Use dynamic title
            fontsize=9,
        )

    plt.tight_layout()  # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


plot_uncertainty_drift.__doc__ = r"""
Polar plot visualizing temporal drift of uncertainty width.

This function creates a polar line plot showing how the width of
the prediction interval (e.g., Q90 - Q10), representing model
uncertainty, evolves over multiple time steps (e.g., years) across
different locations. Each time step is depicted as a distinct
concentric ring [1]_.

- **Angular Position (`theta`)**: Represents each location or data
  point. Currently derived from the DataFrame index, mapped
  linearly onto the angular range specified by `acov`. The optional
  `theta_col` parameter is intended for future use in ordering but
  is currently ignored for positioning.

- **Radial Rings (`r`)**: Each ring corresponds to a specific time
  step provided via `qlow_cols`/`qup_cols`. The position of the
  ring (distance from the center) indicates the time step (later
  times are further out). The radius of the line at a specific angle
  (location) on a given ring is determined by a base offset for that
  year plus a component proportional to the *globally normalized*
  interval width at that location and time. Thus, the 'thickness' or
  deviation of a ring from a perfect circle reflects the magnitude
  of uncertainty (interval width) relative to the maximum width
  observed across all locations and times.

- **Color**: Each ring (time step) is assigned a unique color based
  on the specified `cmap`, aiding in distinguishing and tracking
  changes across time steps.

This visualization is particularly useful for:

- Identifying locations where prediction uncertainty grows or shrinks
  significantly over the forecast horizon.
- Monitoring the overall trend (drift) of uncertainty as forecasts
  extend further into the future.
- Highlighting areas with consistently high or low uncertainty across
  all time steps.
- Comparing the spatial patterns of uncertainty at different forecast
  lead times.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the quantile prediction columns.
    Decorators ensure it's a valid, non-empty pandas DataFrame.

qlow_cols : list of str
    List of column names for the lower quantile bound (e.g., Q10)
    for consecutive time steps. Example: ``['pred_2023_q10',
    'pred_2024_q10', ...]``.

qup_cols : list of str
    List of column names for the upper quantile bound (e.g., Q90)
    for the same time steps as `qlow_cols`. Must be the same length.
    Example: ``['pred_2023_q90', 'pred_2024_q90', ...]``.

dt_labels : list of str, optional
    List of labels for each time step, used in the legend to identify
    the rings. Must match the length of `qlow_cols`. If ``None``,
    generic labels like 'Time_1', 'Time_2', ... are generated based
    on the `label` parameter. Default is ``None``.

theta_col : str, optional
    *Intended* column name for ordering points angularly. *Note: This
    parameter is currently ignored in the positioning logic; angles
    are based on the DataFrame index.* A warning is issued if provided.
    Default is ``None``.

acov : {'default', 'half_circle', 'quarter_circle', \
    'eighth_circle'}, default='default'

    Specifies the angular coverage (span) of the plot: ``'default'``
    (360°), ``'half_circle'`` (180°), ``'quarter_circle'`` (90°),
    ``'eighth_circle'`` (45°).

base_radius : float, default=0.15
    Determines the spacing between the base circles of consecutive
    time step rings. The base radial offset for the ring representing
    time step `i` (0-indexed) is calculated as ``base_radius * (i+1)``.
    A larger value increases the separation between rings.

band_height : float, default=0.15
    Scaling factor applied to the *normalized* interval width. This
    controls how much the radius deviates from the base circle for
    each ring, visually representing the uncertainty magnitude.
    Effectively, it's the maximum radial 'height' allocated to
    represent uncertainty on each ring.

cmap : str, default='tab10'
    Name of the Matplotlib colormap used to assign distinct colors
    to the rings representing different time steps.

label : str, default='Year'
    Base name used for generating default `dt_labels` if they are
    not provided (e.g., 'Year_1', 'Year_2', ...). *Note: This
    parameter is currently *not* used for the legend title itself.*

alpha : float, default=0.85
    Transparency level for the plotted lines (rings).

figsize : tuple of (float, float), default=(9, 9)
    Width and height of the figure in inches.

title : str, optional
    Title displayed above the polar plot. If ``None``, a default
    title is used. Default is ``None``.

show_grid : bool, default=True
    If ``True``, display polar grid lines.

show_legend : bool, default=True
    If ``True``, display a legend identifying the time step for
    each colored ring, using `dt_labels`.

mask_angle : bool, default=True
    If ``True``, hide the angular tick labels (degrees). Recommended
    if the angular position is based on index.

savefig : str, optional
    File path to save the plot image. If ``None``, displays the plot
    interactively. Default is ``None``.

Returns
-------
ax : matplotlib.axes.Axes
    The Matplotlib Axes object containing the polar line plot.

Raises
------
AssertionError
    If `qlow_cols` and `qup_cols` lists have different lengths.
ValueError
    If specified quantile columns are not found in the DataFrame.
    If data in quantile columns is not numeric.

See Also
--------
plot_interval_consistency : Plot consistency of interval widths using
                            scatter points.
numpy.linspace : Generate evenly spaced numbers.
matplotlib.pyplot.plot : Plot lines.

Notes
-------

- Interval widths (:math:`W = Q_{upper} - Q_{lower}`) are calculated
  for each location and time step.
- These widths are then *globally normalized* by dividing by the
  maximum width observed across all locations and all time steps.
  This ensures that the radial deviations are comparable across rings.
- The radius for a point on ring `i` (0-indexed time step) is
  :math:`r = (\text{base}_{radius} \times (i+1)) + (\text{band}_{height}
  \times w_{normalized})`.
- The angular coordinate `theta` is currently based on the DataFrame
  index, not influenced by `theta_col`.
- Radial axis ticks and labels are hidden by default (`set_yticks([])`)
  as the radial dimension primarily separates the time steps.

Let :math:`\mathbf{L}_i` and :math:`\mathbf{U}_i` be the lower and
upper quantile bound vectors (length :math:`N`, number of locations)
for time step :math:`i` (:math:`i = 0, \dots, M-1`).

1. **Interval Width Calculation**: For each time step :math:`i`,
   calculate the width vector:

   :math:`\mathbf{W}_i = \mathbf{U}_i - \mathbf{L}_i`.

2. **Global Normalization**: Find the maximum width across all
   locations and time steps:

   :math:`W_{max} = \max_{i,j} (\mathbf{W}_i)_j`.

   Normalize each width vector :math:`\mathbf{W}_i`:

   .. math::
       \mathbf{w}_i = \frac{\mathbf{W}_i}{W_{max}} \quad (\text{element-wise})
       
   If :math:`W_{max} = 0`, :math:`\mathbf{w}_i` will be all zeros.

3. **Angular Coordinate (`theta`)**: Let :math:`S` be the angular span
   and :math:`\theta_{min}` the start angle from `acov`. For location
   index :math:`j` (:math:`j=0, ..., N-1`):

   .. math::
       \theta_j = \left( \frac{j}{N} \times S \right) + \theta_{min}

4. **Radial Coordinate (`r`)**: For time step :math:`i` and location
   :math:`j`:

   .. math::
       r_{i,j} = (\text{base}_{radius} \times (i+1)) + \\
           (\text{band}_{height} \times (\mathbf{w}_i)_j)

5. **Plotting**: For each time step :math:`i`, plot a line
   connecting points :math:`(r_{i,j}, \theta_j)` for
   :math:`j = 0, \dots, N-1`.

References
----------

.. [1] Kouadio, K. L., Liu, R., Loukou, K. G. H., Liu, J., & Liu, W. (2025).
   Analytics Framework for Interpreting Spatiotemporal Probabilistic Forecasts.
   *International Journal of Forecasting*. Manuscript submitted.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.uncertainty import plot_uncertainty_drift

**1. Random Example:**

>>> np.random.seed(2)
>>> N_points = 100
>>> df_drift_rand = pd.DataFrame({'location_id': range(N_points)})
>>> years = range(2020, 2024)
>>> q10_drift_cols = []
>>> q90_drift_cols = []
>>> for i, year in enumerate(years):
...     q10_col = f'q10_{year}'
...     q90_col = f'q90_{year}'
...     base_val = np.random.rand(N_points) * 10
...     width = (np.random.rand(N_points) + 0.5) * (2 + i) # Increasing width
...     df_drift_rand[q10_col] = base_val - width / 2
...     df_drift_rand[q90_col] = base_val + width / 2
...     q10_drift_cols.append(q10_col)
...     q90_drift_cols.append(q90_col)
>>> ax_drift_rand = plot_uncertainty_drift(
...     df=df_drift_rand,
...     qlow_cols=q10_drift_cols,
...     qup_cols=q90_drift_cols,
...     dt_labels=[str(y) for y in years],
...     theta_col='location_id', # Ignored for positioning
...     acov='default',
...     base_radius=0.1,      # Smaller spacing
...     band_height=0.1,      # Smaller uncertainty scaling
...     cmap='viridis',
...     title='Random Uncertainty Drift Example',
...     mute_angle=False      # Show angle labels
... )
>>> # plt.show() called internally

**2. Concrete Example (Subsidence Data - adapted):**

>>> # Assume zhongshan_pred_2023_2026 is a loaded DataFrame like:
>>> # Create dummy data if it doesn't exist
>>> try:
...    zhongshan_pred_2023_2026
... except NameError:
...    print("Creating dummy subsidence data for example...")
...    N_sub = 150
...    zhongshan_pred_2023_2026 = pd.DataFrame({
...       'latitude': np.linspace(22.2, 22.8, N_sub),
...       **{f'subsidence_{yr}_q10': np.random.rand(N_sub)*(yr-2022)*2 + 1
...          for yr in range(2023, 2027)},
...       **{f'subsidence_{yr}_q90': np.random.rand(N_sub)*(yr-2022)*2 + 5
...          + np.linspace(0, (yr-2022)*3, N_sub) # Increasing width trend
...          for yr in range(2023, 2027)},
...     })
>>> qlow_sub_drift = [f'subsidence_{yr}_q10' for yr in range(2023, 2027)]
>>> qup_sub_drift = [f'subsidence_{yr}_q90' for yr in range(2023, 2027)]
>>> year_labels_sub = [str(yr) for yr in range(2023, 2027)]
>>> ax_sub_drift = plot_uncertainty_drift(
...     df=zhongshan_pred_2023_2026,
...     qlow_cols=qlow_sub_drift,
...     qup_cols=qup_sub_drift,
...     dt_labels=year_labels_sub,
...     theta_col='latitude',     # Ignored for positioning
...     acov='half_circle',     # Use 180 degrees
...     title='Uncertainty Drift Over Time (Zhongshan)',
...     cmap='tab10',
...     band_height=0.1,        # Controls visual width effect
...     base_radius=0.2,        # Controls spacing between years
...     show_legend=True,
...     mute_degree=True
... )
>>> # plt.show() called internally

"""


@check_non_emptiness
@isdf
def plot_actual_vs_predicted(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    theta_col: str | None = None,
    acov: str = "default",
    figsize: tuple[float, float] = (8.0, 8.0),
    title: str | None = None,
    line: bool = True,
    r_label: str | None = None,
    cmap: str | None = None,  # Note: Currently unused
    alpha: float = 0.3,
    actual_props: dict[str, Any] | None = None,
    pred_props: dict[str, Any] | None = None,
    show_grid: bool = True,
    grid_props: dict | None = None,
    show_legend: bool = True,
    mask_angle: bool = False,
    savefig: str | None = None,
):
    r"""Polar plot comparing actual observed vs. predicted values.

    This function generates a polar plot to visually compare actual
    ground truth values against model predictions (typically a central
    estimate like the median, Q50) for multiple data points or
    locations arranged circularly [1]_.

    - **Angular Position (`theta`)**: Represents each data point or
      location. Points are currently plotted in their DataFrame index
      order, mapped linearly onto the specified angular coverage
      (`acov`). The `theta_col` parameter is intended for future use
      in ordering points based on a specific feature (like latitude)
      but is currently ignored for positioning.
    - **Radial Distance (`r`)**: Represents the magnitude of the values.
      Both the actual value (`actual_col`) and the predicted value
      (`pred_col`) are plotted at the corresponding angle `theta`.

    - **Visual Comparison**:

      - Actual and predicted values are shown as either continuous
        lines or individual dots based on the `line` parameter [2]_.
      - Gray vertical lines connect the actual and predicted values
        at each angle, visually highlighting the magnitude and
        direction (over- or under-prediction) of the difference
        at each point.

    This plot facilitates:

    - Quick visual assessment of prediction accuracy and bias across
      samples.
    - Identification of regions or conditions (if angle relates to a
      feature) where the model performs well or poorly.
    - Communication of model performance to stakeholders.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing actual and predicted value columns.
        Decorators ensure it's a valid, non-empty pandas DataFrame.

    actual_col : str
        Name of the column holding the actual observed (ground truth)
        values.

    pred_col : str
        Name of the column holding the corresponding predicted values
        (e.g., the Q50 median prediction).

    theta_col : str, optional
        *Intended* column name for ordering points angularly based on
        its values (e.g., 'latitude'). *Note: This parameter is
        currently ignored for positioning/ordering in this implementation;
        points use DataFrame index order.* A warning is issued if
        provided. Default is ``None``.

    acov : {'default', 'half_circle', 'quarter_circle', \
        'eighth_circle'}, default='default'
        Specifies the angular coverage (span) of the polar plot:
        ``'default'`` (360°), ``'half_circle'`` (180°),
        ``'quarter_circle'`` (90°), ``'eighth_circle'`` (45°).

    figsize : tuple of (float, float), default=(8.0, 8.0)
        Width and height of the figure in inches.

    title : str, optional
        Custom title for the plot. If ``None``, a default title is used.
        Default is ``None``.

    line : bool, default=True
        Determines the plotting style:

        - If ``True``, actual and predicted values are plotted as lines
          connecting consecutive points.
        - If ``False``, values are plotted as individual scatter dots.

    r_label : str, optional
        Custom label for the radial axis (representing value magnitude).
        If ``None``, no label is set. Default is ``None``.

    cmap : str, optional
        *Note: This parameter is currently unused in the function.*
        It might be intended for future use, perhaps coloring the
        difference lines. Default is ``None``.

    alpha : float, default=0.3
        Transparency level applied to the gray difference lines drawn
        between actual and predicted values, and also to the predicted
        dots if ``line=False``.

    actual_props : dict, optional
        Dictionary of keyword arguments passed directly to the Matplotlib
        `plot` or `scatter` function for the 'Actual' data series. Allows
        customization (e.g., ``{'color': 'blue', 'linestyle': '--'}``).
        Defaults to basic black line/dots if ``None``.

    pred_props : dict, optional
        Dictionary of keyword arguments passed directly to the Matplotlib
        `plot` or `scatter` function for the 'Predicted' data series.
        Allows customization (e.g., ``{'color': 'orange', 'marker': 'x'}``).
        Defaults to basic red line/dots if ``None``.

    show_grid : bool, default=True
        If ``True``, display the polar grid lines.

    show_legend : bool, default=True
        If ``True``, display a legend labeling the 'Actual' and
        'Predicted' series.

    mute_degree : bool, default=False
        If ``True``, hide the angular tick labels (degrees).

    savefig : str, optional
        File path to save the plot image. If ``None``, displays the plot
        interactively. Default is ``None``.

    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        The Matplotlib Axes object containing the polar plot. Note that
        due to subplot_kw, it's specifically a PolarAxesSubplot.

    Raises
    ------
    ValueError
        If `actual_col` or `pred_col` are not found in `df`.
    TypeError
        If data in actual or predicted columns is not numeric.

    See Also
    --------
    plot_anomaly_magnitude : Visualize only points outside prediction
                             intervals.
    matplotlib.pyplot.plot : Function for line plots.
    matplotlib.pyplot.scatter : Function for scatter plots.

    Notes
    --------

    - Rows with NaN values in `actual_col` or `pred_col` (or `theta_col`
      if specified, though currently unused for position) are dropped.
    - The gray lines indicating the difference are drawn individually
      for each point using a loop. *Warning: This approach can be very
      slow for large datasets (many thousands of points).* An alternative
      like `fill_between` might be more efficient for showing shaded areas
      but would require sorting by theta.
    - The `theta_col` parameter is currently ignored for positioning;
      angles are always based on the DataFrame index order after NaN
      removal.
    - The `cmap` parameter is currently unused. The difference lines are
      hardcoded to 'gray'.
    - Default plotting styles are black for actual and red for predicted,
      but can be overridden using `actual_props` and `pred_props` [1]_.

    Let :math:`y_j` be the actual value and :math:`\hat{y}_j` the
    predicted value for data point (location) :math:`j` (:math:`j=0,
    \dots, N-1` after NaN removal).

    1. **Angular Coordinate (`theta`)**: Let :math:`S` be the angular span
       and :math:`\theta_{min}` the start angle from `acov`.

       .. math::
           \theta_j = \left( \frac{j}{N} \times S \right) + \theta_{min}

    2. **Radial Coordinates**: The radial coordinates are directly the
       values: :math:`r_{actual, j} = y_j` and :math:`r_{pred, j} =
       \hat{y}_j`.

    3. **Plotting**:

       - Plot points/lines connecting :math:`(r_{actual, j}, \theta_j)`
         and :math:`(r_{pred, j}, \theta_j)`.
       - For each :math:`j`, draw a gray line segment connecting the points
         :math:`(\min(y_j, \hat{y}_j), \theta_j)` and
         :math:`(\max(y_j, \hat{y}_j), \theta_j)`.

    References
    ----------
    
    .. [1] Kouadio, K. L., Liu, R., Loukou, K. G. H., Liu, J., & Liu, W. (2025).
       Analytics Framework for Interpreting Spatiotemporal Probabilistic Forecasts.
       *International Journal of Forecasting*. Manuscript submitted.
       
    .. [2] Matplotlib documentation: https://matplotlib.org/stable/

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from kdiagram.plot.uncertainty import plot_actual_vs_predicted

    **1. Random Example:**

    >>> np.random.seed(0)
    >>> N = 100
    >>> df_avp_rand = pd.DataFrame({
    ...     'Time': pd.date_range('2023-01-01', periods=N, freq='D'),
    ...     'ActualTemp': 15 + 10 * np.sin(np.linspace(0, 4 * np.pi, N)) + np.random.randn(N) * 2,
    ...     'PredictedTemp': 16 + 9 * np.sin(np.linspace(0, 4 * np.pi, N) + 0.1) + np.random.randn(N) * 1.5
    ... })
    >>> ax_avp_rand = plot_actual_vs_predicted(
    ...     df=df_avp_rand,
    ...     actual_col='ActualTemp',
    ...     pred_col='PredictedTemp',
    ...     theta_col='Time', # Note: Ignored for positioning
    ...     acov='default',
    ...     title='Temperature: Actual vs. Predicted',
    ...     line=True, # Use lines
    ...     r_label='Temperature (°C)',
    ...     actual_props={'color': 'navy', 'linestyle': '-'},
    ...     pred_props={'color': 'crimson', 'linestyle': '--'}
    ... )
    >>> # plt.show() called internally

    **2. Concrete Example (Subsidence Data - using dots):**

    >>> # Assume zhongshan_pred_2023_2026 is a loaded DataFrame
    >>> # Create dummy data if it doesn't exist
    >>> try:
    ...    zhongshan_pred_2023_2026
    ... except NameError:
    ...    print("Creating dummy subsidence data for example...")
    ...    N_sub = 150
    ...    zhongshan_pred_2023_2026 = pd.DataFrame({
    ...       'latitude': np.linspace(22.2, 22.8, N_sub),
    ...       'subsidence_2023': np.random.rand(N_sub)*15 + np.linspace(0, 5, N_sub),
    ...       'subsidence_2023_q50': np.random.rand(N_sub)*14 + np.linspace(0.5, 5.5, N_sub),
    ...       # Add other columns if needed by other examples
    ...       **{f'subsidence_{yr}_q10': np.random.rand(N_sub)*(yr-2022)*2 + 1
    ...          for yr in range(2023, 2027)},
    ...       **{f'subsidence_{yr}_q90': np.random.rand(N_sub)*(yr-2022)*2 + 5
    ...          + np.linspace(0, (yr-2022)*3, N_sub)
    ...          for yr in range(2023, 2027)},
    ...     })

    >>> ax_avp_sub = plot_actual_vs_predicted(
    ...     df=zhongshan_pred_2023_2026.head(100), # Use subset for speed
    ...     actual_col='subsidence_2023',
    ...     pred_col='subsidence_2023_q50',
    ...     theta_col='latitude',      # Note: Ignored for positioning
    ...     acov='half_circle',      # Use 180 degrees
    ...     title='Actual vs Predicted Subsidence (2023)',
    ...     line=False,              # Use dots instead of lines
    ...     r_label="Subsidence (mm)",
    ...     mute_degree=True,
    ...     pred_props={'marker': 'x', 'color': 'purple'} # Customize predicted dots
    ... )
    >>> # plt.show() called internally

    """
    # If theta_col is provided but currently unused, warn:
    if theta_col is not None:
        warnings.warn(
            "`theta_col` is currently ignored"
            " by plot_actual_vs_predicted.",
            UserWarning,
            stacklevel=2,
        )
    # --- Input Validation and Preparation ---
    # Basic checks handled by decorators
    # Check existence of primary columns
    exist_features(
        df,
        features=[actual_col, pred_col],
        error="raise",  # Raise error if missing
        name="Actual and Predicted columns",
    )

    # Consolidate columns needed, check theta_col existence only if specified
    cols_to_select = [actual_col, pred_col]
    if theta_col:
        if theta_col not in df.columns:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found. "
                f"Using index for angular position.",
                UserWarning,
                stacklevel=2,
            )
            # Proceed without theta_col for ordering
        else:
            warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but currently "
                f"ignored for positioning/ordering. Using index.",
                UserWarning,
                stacklevel=2,
            )
        # Although ignored, keep it for potential future use if needed?
        # For now, just select it if present, even if unused later.
        # cols_to_select.append(theta_col) # Decided against adding if unused

    # Drop rows with NaNs in essential columns
    data = df[cols_to_select].dropna().copy()
    if len(data) == 0:
        warnings.warn(
            "DataFrame is empty after dropping NaN values in actual"
            " and predicted columns. Cannot generate plot.",
            UserWarning,
            stacklevel=2,
        )
        return None
    N = len(data)

    if N > 2_000:
        warnings.warn(
            "Large number of samples; consider "
            "saving the figure instead of showing.",
            UserWarning,
            stacklevel=2,
        )

    # --- Angular Coordinate Calculation ---
    acov_map = {  # Map name to angular range in radians
        "default": 2 * np.pi,
        "half_circle": np.pi,
        "quarter_circle": np.pi / 2,
        "eighth_circle": np.pi / 4,
    }
    angular_range = acov_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in acov_map:
        warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning,
            stacklevel=2,
        )
    # Calculate theta based on index, mapped to the angular range
    # Use endpoint=False if using lines to avoid overlap at 2pi?
    # If using dots, endpoint=True or False matters less visually.
    # Let's use endpoint=False for lines, True for dots for potentially
    # better spacing.
    use_endpoint = not line
    theta = np.linspace(0.0, angular_range, N, endpoint=use_endpoint)

    # --- Extract Data ---
    try:
        actual = data[actual_col].to_numpy(dtype=float)
        pred = data[pred_col].to_numpy(dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Failed to convert actual or predicted columns to numeric."
            f" Check data types. Original error: {e}"
        ) from e

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(angular_range))  # Expects degrees

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_angle:
        ax.set_xticklabels([])

    # --- Plot Difference Lines ---
    # Warning: This loop can be very slow for large N
    # Consider alternatives like fill_between if performance is critical
    # and data can be meaningfully sorted by theta.
    if N > 5_000:  # Add warning for potentially slow loop
        warnings.warn(
            f"Plotting difference lines for {N} points individually."
            f" This may be slow. Consider using `line=False` or sampling data.",
            PerformanceWarning,
            stacklevel=2,
        )
    for t, a, p in zip(theta, actual, pred):
        # Plot a vertical line segment at angle t between actual and pred
        ax.plot(
            [t, t],  # Start and end angle (same)
            [min(a, p), max(a, p)],  # Start and end radius
            color="gray",  # Hardcoded color for difference
            alpha=alpha,  # Use specified transparency
            linewidth=1,  # Fixed linewidth for diff lines
        )

    # --- Plot Actual and Predicted Data (Lines or Dots) ---
    # Define default properties, merge with user-provided props
    default_actual_props_line = {
        "color": "black",
        "linewidth": 1.5,
        "label": "Actual",
    }
    default_pred_props_line = {
        "color": "red",
        "linewidth": 1.5,
        "label": "Predicted (Q50)",
    }
    default_actual_props_scatter = {
        "color": "black",
        "s": 20,
        "label": "Actual",
    }
    default_pred_props_scatter = {
        "color": "red",
        "s": 20,
        "alpha": alpha,
        "label": "Predicted (Q50)",
    }

    # Ensure user props are dictionaries if provided
    actual_props = actual_props or {}
    pred_props = pred_props or {}
    _assert_all_types(actual_props, dict, objname="'actual_props'")
    _assert_all_types(pred_props, dict, objname="'pred_props'")

    if line:
        # Merge user props with defaults for line plot
        current_actual_props = {**default_actual_props_line, **actual_props}
        current_pred_props = {**default_pred_props_line, **pred_props}
        # Ensure theta wraps around if using full circle for line plot
        theta_plot = (
            np.append(theta, theta[0])
            if np.isclose(angular_range, 2 * np.pi)
            else theta
        )
        actual_plot = (
            np.append(actual, actual[0])
            if np.isclose(angular_range, 2 * np.pi)
            else actual
        )
        pred_plot = (
            np.append(pred, pred[0])
            if np.isclose(angular_range, 2 * np.pi)
            else pred
        )

        ax.plot(theta_plot, actual_plot, **current_actual_props)
        ax.plot(theta_plot, pred_plot, **current_pred_props)
    else:
        # Merge user props with defaults for scatter plot
        current_actual_props = {
            **default_actual_props_scatter,
            **actual_props,
        }
        current_pred_props = {**default_pred_props_scatter, **pred_props}

        ax.scatter(theta, actual, **current_actual_props)
        ax.scatter(theta, pred, **current_pred_props)

    # --- Final Touches ---
    ax.set_title(
        title or "Actual vs Predicted Polar Plot", fontsize=14, y=1.08
    )
    if r_label:
        # Use set_ylabel for radial axis label, adjust padding
        ax.set_ylabel(r_label, labelpad=15, fontsize=10)

    # Add legend if requested and labels were provided in props
    if show_legend:
        # Check if labels exist before showing legend
        handles, labels = ax.get_legend_handles_labels()
        if labels:  # Only show legend if there's something to label
            ax.legend(
                loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9
            )
        else:
            warnings.warn(
                "Legend requested but no labels found for plot elements.",
                UserWarning,
                stacklevel=2,
            )

    plt.tight_layout()  # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


@check_non_emptiness
@isdf
def plot_interval_width(
    df: pd.DataFrame,
    q_cols: list[str] | tuple[str, str],
    theta_col: str | None = None,
    z_col: str | None = None,
    acov: str = "default",
    figsize: tuple[float, float] = (8.0, 8.0),
    title: str | None = None,
    cmap: str = "viridis",
    s: int = 30,
    alpha: float = 0.8,
    show_grid: bool = True,
    grid_props: dict | None = None,
    cbar: bool = True,
    mask_angle: bool = True,
    savefig: str | None = None,
):
    # --- Input Validation ---
    # Basic df checks handled by decorators
    q_cols_processed = columns_manager(q_cols, empty_as_none=False)
    if len(q_cols_processed) != 2:
        raise TypeError(
            "`q_cols` expects exactly two column names: "
            "[lower_quantile_column, upper_quantile_column]. "
            f"Received {len(q_cols_processed)} columns: {q_cols_processed}"
        )
    # Assign validated lower and upper quantile columns
    qlow_col, qup_col = q_cols_processed[0], q_cols_processed[1]

    # Consolidate list of columns needed for processing and NaN checks
    cols_needed = [qlow_col, qup_col]
    if theta_col:
        # Although unused for positioning, check if exists if provided
        if theta_col not in df.columns:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found. "
                f"It will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        else:
            # Add to list for potential future use or NaN check if desired
            # cols_needed.append(theta_col) # Currently ignored, don't add
            warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but currently "
                f"ignored for positioning/ordering. Using index.",
                UserWarning,
                stacklevel=2,
            )
    if z_col:
        if z_col not in df.columns:
            raise ValueError(
                f"Specified `z_col` ('{z_col}') not found in DataFrame."
            )
        cols_needed.append(z_col)

    # Check existence of essential quantile columns (redundant if exist_features used)
    missing_essential = [
        col for col in [qlow_col, qup_col] if col not in df.columns
    ]
    if missing_essential:
        raise ValueError(
            f"Essential quantile columns missing: {', '.join(missing_essential)}"
        )
    # Also check z_col again if it was added
    if z_col and z_col not in df.columns:
        raise ValueError(f"`z_col` ('{z_col}') not found.")

    # Drop rows with NaNs in the essential columns used for plotting (q, z)
    data = df[cols_needed].dropna().copy()
    if len(data) == 0:
        warnings.warn(
            "DataFrame is empty after dropping NaN values."
            " Cannot generate plot.",
            UserWarning,
            stacklevel=2,
        )
        return None
    N = len(data)  # Number of valid data points

    # --- Calculate Radial Coordinate (Interval Width) ---
    try:
        r = data[qup_col].to_numpy(dtype=float) - data[qlow_col].to_numpy(
            dtype=float
        )
    except Exception as e:
        raise TypeError(
            f"Could not compute interval width. Ensure columns '{qup_col}'"
            f" and '{qlow_col}' contain numeric data. Original error: {e}"
        ) from e

    # Check for negative widths
    if np.any(r < 0):
        num_negative = np.sum(r < 0)
        warnings.warn(
            f"{num_negative} out of {N} locations have negative interval "
            f"width ({qup_col} < {qlow_col}). These will be plotted with "
            f"negative radius, potentially causing visual artifacts.",
            UserWarning,
            stacklevel=2,
        )
        # Option: clamp negative radius to 0?
        # r = np.maximum(r, 0)

    # --- Calculate Color Coordinate (Z-value) ---
    if z_col:
        try:
            z = data[z_col].to_numpy(dtype=float)
            cbar_label = z_col  # Label colorbar with column name
        except Exception as e:
            raise TypeError(
                f"Could not use `z_col` ('{z_col}') for color. Ensure it "
                f"contains numeric data. Original error: {e}"
            ) from e
    else:
        # Default: color by interval width `r`
        z = r
        cbar_label = "Interval Width"  # Label colorbar appropriately

    # --- Angular Coordinate Calculation ---
    acov_map = {  # Map name to angular range in radians
        "default": 2 * np.pi,
        "half_circle": np.pi,
        "quarter_circle": np.pi / 2,
        "eighth_circle": np.pi / 4,
    }
    angular_range = acov_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in acov_map:
        warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning,
            stacklevel=2,
        )
    # Calculate theta based on index, mapped to the angular range
    # Using endpoint=False might give slightly better visual spacing for scatter
    theta = np.linspace(0.0, angular_range, N, endpoint=False)

    # --- Color Normalization ---
    # Check if z has variance for normalization
    if np.ptp(z) > 1e-9:  # Check peak-to-peak range
        norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    else:
        # Handle constant z case: map all to middle color
        norm = Normalize(vmin=np.min(z) - 0.5, vmax=np.max(z) + 0.5)
        warnings.warn(
            f"Color values ('{cbar_label}') have zero range."
            f" All points will have the same color.",
            UserWarning,
            stacklevel=2,
        )
    try:
        cmap_ref = get_cmap(cmap)
        colors = cmap_ref(norm(z))
    except ValueError:
        warnings.warn(
            f"Invalid `cmap` name '{cmap}'. Falling back to 'viridis'.",
            stacklevel=2,
        )
        cmap = "viridis"  # Ensure cmap is valid for fallback
        cmap_ref = get_cmap(cmap)
        colors = cmap_ref(norm(z))

    # --- Plotting ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(angular_range))  # Expects degrees

    # Apply grid and angle label settings
    set_axis_grid(ax, show_grid, grid_props=grid_props)
    if mask_angle:
        ax.set_xticklabels([])

    # Create the scatter plot
    ax.scatter(
        theta,
        r,  # Radius is interval width
        c=colors,  # Point color based on z
        s=s,  # Point size
        alpha=alpha,  # Point transparency
        edgecolor="k",  # Point edge color (optional)
        linewidth=0.5,  # Point edge width (optional)
        # cmap=cmap_ref is implicitly used via 'colors', no need to pass here
    )

    # Set plot title
    ax.set_title(
        title or f"Prediction Interval Width ({qup_col} - {qlow_col})",
        fontsize=14,
        y=1.08,
    )

    # Add color bar if requested
    if cbar:
        # Create a ScalarMappable for the colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_ref)
        sm.set_array([])  # Necessary for ScalarMappable

        # Add the colorbar to the figure
        cbar_obj = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
        # Set label dynamically based on whether z_col was used
        cbar_obj.set_label(cbar_label, rotation=270, labelpad=15, fontsize=10)

    # Set radial axis label (useful to know radius means width)
    ax.set_ylabel("Interval Width", labelpad=15, fontsize=10)
    # Set radial limits, ensure 0 is included if widths are non-negative
    if np.all(r >= 0):
        ax.set_ylim(bottom=0)
    # else: let matplotlib auto-scale if negative widths exist

    plt.tight_layout()  # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


plot_interval_width.__doc__ = r"""
Polar scatter plot visualizing prediction interval width.

This function generates a polar scatter plot to visualize the
magnitude of prediction uncertainty, represented by the width of the
prediction interval (Upper Quantile - Lower Quantile), across
different locations or samples.

- **Angular Position (`theta`)**: Represents each location or data
  point. Currently derived from the DataFrame index, mapped
  linearly onto the specified angular coverage (`acov`). The optional
  `theta_col` parameter is intended for future use in ordering but
  is currently ignored for positioning.
- **Radial Distance (`r`)**: Directly represents the width of the
  prediction interval (:math:`Q_{upper} - Q_{lower}`). A larger
  radius indicates greater predicted uncertainty for that point.
- **Color (`z`)**: Optionally represents a third variable, specified
  by `z_col` (e.g., the median prediction Q50, or the actual value).
  This allows for correlating the uncertainty width with another
  metric. If `z_col` is not provided, the color defaults to
  representing the interval width (`r`) itself [1]_.

This plot helps to:

- Identify locations with high or low prediction uncertainty.
- Visualize the spatial distribution or sample distribution of
  uncertainty magnitude.
- Explore potential correlations between uncertainty width and other
  variables (like the central prediction) when using `z_col`.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the quantile prediction columns and
    optionally columns for theta ordering and color (`z_col`).
    Decorators ensure it's a valid, non-empty pandas DataFrame.

q_cols : list or tuple of str
    A sequence containing exactly two string elements: the column
    name for the lower quantile bound (e.g., 'prediction_q10') and
    the column name for the upper quantile bound (e.g.,
    'prediction_q90'). The order must be ``[lower_col, upper_col]``.

theta_col : str, optional
    *Intended* column name for ordering points angularly based on
    its values (e.g., 'latitude'). *Note: This parameter is
    currently ignored for positioning/ordering in this implementation;
    points use DataFrame index order.* A warning is issued if
    provided. Default is ``None``.

z_col : str, optional
    Name of the column whose values will be used to color the scatter
    points. Common choices include the median prediction column (e.g.,
    'prediction_q50') or an actual value column to see if high/low
    uncertainty correlates with high/low values. If ``None``, the
    color will represent the interval width (`r`) itself.
    Default is ``None``.

acov : {'default', 'half_circle', 'quarter_circle', \
    'eighth_circle'}, default='default'

    Specifies the angular coverage (span) of the polar plot:
    ``'default'`` (360°), ``'half_circle'`` (180°),
    ``'quarter_circle'`` (90°), ``'eighth_circle'`` (45°).

figsize : tuple of (float, float), default=(8.0, 8.0)
    Width and height of the figure in inches.

title : str, optional
    Custom title for the plot. If ``None``, a default title like
    "Prediction Interval Width (Q_upper - Q_lower)" is used.
    Default is ``None``.

cmap : str, default='viridis'
    Name of the Matplotlib colormap used to color the points based
    on the `z` value (from `z_col` or defaulting to interval width `r`).

s : int, default=30
    Marker size for the scatter points.

alpha : float, default=0.8
    Transparency level for the scatter points (0=transparent,
    1=opaque).

show_grid : bool, default=True
    If ``True``, display the polar grid lines.

cbar : bool, default=True
    If ``True``, display a color bar indicating the mapping between
    colors and the `z` values.

mask_angle : bool, default=True
    If ``True``, hide the angular tick labels (degrees). Recommended
    if the angle is based on index.

savefig : str, optional
    File path to save the plot image. If ``None``, displays the plot
    interactively. Default is ``None``.

Returns
-------
ax : matplotlib.axes._axes.Axes
    The Matplotlib Axes object containing the polar scatter plot,
    specifically a PolarAxesSubplot.

Raises
------
TypeError
    If `q_cols` does not contain exactly two elements.
ValueError
    If specified columns in `q_cols`, `theta_col` (if provided), or
    `z_col` (if provided) are not found in the DataFrame `df`.
    If data in the required columns is not numeric.

See Also
--------
plot_interval_consistency : Visualize the temporal consistency of
                            interval widths.
plot_uncertainty_drift : Visualize the temporal drift of interval
                         widths as rings.
matplotlib.pyplot.scatter : Function used for plotting points.
matplotlib.colors.Normalize : Used for scaling color values.

Notes
------

- Rows with NaN values in any of the required columns (`q_cols`,
  `theta_col` if used, `z_col` if used) are dropped before plotting.
- The interval width `r` is calculated as `Upper Quantile - Lower
  Quantile`. If the lower quantile is greater than the upper
  quantile for any point, this will result in a negative width,
  which might lead to unexpected plotting behavior as radius is
  typically non-negative. A warning is issued if negative widths
  are detected.
- The angular coordinate `theta` is derived from the DataFrame index
  after NaN removal, not influenced by `theta_col` currently.
- Color is determined by the `z_col` values if provided, otherwise
  it defaults to representing the interval width `r`.

Let :math:`L_j` and :math:`U_j` be the lower and upper quantile bound
values for data point (location) :math:`j` (:math:`j=0, \dots, N-1`
after NaN removal). Let :math:`z'_j` be the value from `z_col` for
point :math:`j`, if `z_col` is provided.

1. **Radial Coordinate (`r`)**: Interval Width.

   .. math::
       r_j = U_j - L_j

2. **Color Value (`z`)**:

   .. math::
       z_j = \begin{cases} z'_j & \text{if } z\_col \text{ is provided}\\
           \\ r_j & \text{if } z\_col \text{ is None} \end{cases}

3. **Angular Coordinate (`theta`)**: Let :math:`S` be the angular span
   and :math:`\theta_{min}` the start angle from `acov`.

   .. math::
       \theta_j = \left( \frac{j}{N} \times S \right) + \theta_{min}

4. **Plotting**: Plot points :math:`(r_j, \theta_j)` using `scatter`,
   where the color of each point is determined by applying `cmap` to
   the normalized value of :math:`z_j`.

References
----------

.. [1] Kouadio, K. L., Liu, R., Loukou, K. G. H., Liu, J., & Liu, W. (2025).
   Analytics Framework for Interpreting Spatiotemporal Probabilistic Forecasts.
   *International Journal of Forecasting*. Manuscript submitted.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.uncertainty import plot_interval_width

**1. Random Example:**

>>> np.random.seed(1)
>>> N = 120
>>> df_iw_rand = pd.DataFrame({
...     'sample_id': range(N),
...     'latitude': np.linspace(40, 42, N) + np.random.randn(N)*0.05,
...     'q10_pred': np.random.rand(N) * 10,
...     'q50_pred': np.random.rand(N) * 10 + 5,
...     'q90_pred': np.random.rand(N) * 10 + 10, # Width varies
... })
>>> # Ensure Q90 > Q10
>>> df_iw_rand['q90_pred'] = df_iw_rand['q10_pred'] + np.abs(
...     df_iw_rand['q90_pred'] - df_iw_rand['q10_pred'])
>>>
>>> ax_iw_rand = plot_interval_width(
...     df=df_iw_rand,
...     q_cols=['q10_pred', 'q90_pred'], # Pass as list [lower, upper]
...     z_col='q50_pred',           # Color by median prediction
...     theta_col='latitude',       # Ignored for positioning
...     acov='default',
...     title='Interval Width vs. Median Prediction',
...     cmap='plasma',
...     s=40,
...     cbar=True
... )
>>> # plt.show() called internally

**2. Concrete Example (Subsidence Data):**

>>> # Assume zhongshan_pred_2023_2026 is a loaded DataFrame
>>> # Create dummy data if it doesn't exist
>>> try:
...    zhongshan_pred_2023_2026
... except NameError:
...    print("Creating dummy subsidence data for example...")
...    N_sub = 150
...    zhongshan_pred_2023_2026 = pd.DataFrame({
...       'latitude': np.linspace(22.2, 22.8, N_sub),
...       'subsidence_2023_q10': np.random.rand(N_sub)*5 + 1,
...       'subsidence_2023_q50': np.random.rand(N_sub)*10 + 3,
...       'subsidence_2023_q90': np.random.rand(N_sub)*5 + 6 + np.linspace(0, 10, N_sub),
...       # Ensure q90 > q10
...       **{f'subsidence_{yr}_q10': np.random.rand(N_sub)*(yr-2022)*2 + 1
...          for yr in range(2024, 2027)}, # Add other cols if needed
...       **{f'subsidence_{yr}_q90': np.random.rand(N_sub)*(yr-2022)*2 + 5
...          + np.linspace(0, (yr-2022)*3, N_sub)
...          for yr in range(2024, 2027)},
...     })
>>> # Ensure Q90 > Q10 for the primary year
>>> zhongshan_pred_2023_2026['subsidence_2023_q90'] = (
...      zhongshan_pred_2023_2026['subsidence_2023_q10'] +
...      np.abs(zhongshan_pred_2023_2026['subsidence_2023_q90'] -
...             zhongshan_pred_2023_2026['subsidence_2023_q10']) + 0.1
...      )
>>> ax_iw_sub = plot_interval_width(
...     df=zhongshan_pred_2023_2026.head(100), # Use subset
...     q_cols=['subsidence_2023_q10', 'subsidence_2023_q90'], # Use list
...     z_col='subsidence_2023_q50',   # Color by Q50
...     theta_col='latitude',          # Ignored for positioning
...     acov='quarter_circle',       # Use 90 degrees
...     title='Spatial Spread of Uncertainty (2023)',
...     cmap='YlGnBu',
...     s=25,
...     cbar=True,                   # Show colorbar for Q50
...     mask_angle=True
... )
>>> # plt.show() called internally

"""


@check_non_emptiness
@isdf
def plot_coverage_diagnostic(
    df: pd.DataFrame,
    actual_col: str,
    q_cols: list[str] | tuple[str, str],
    theta_col: str | None = None,
    acov: str = "default",
    figsize: tuple[float, float] = (8.0, 8.0),
    title: str | None = None,
    show_grid: bool = True,
    grid_props: dict | None = None,
    cmap: str = "RdYlGn",
    alpha: float = 0.85,
    s: int = 35,
    as_bars: bool = False,
    coverage_line_color: str = "r",
    buffer_pts: int = 500,
    fill_gradient: bool = True,
    gradient_size: int = 300,
    gradient_cmap: str = "Greens",
    gradient_levels: list[float] | None = None,
    gradient_props: dict[str, Any] | None = None,
    mask_angle: bool = True,
    savefig: str | None = None,
    verbose: int = 0,
):
    q_cols_processed = columns_manager(q_cols, empty_as_none=False)
    if len(q_cols_processed) != 2:
        raise TypeError(
            "`q_cols` expects exactly two column names: "
            "[lower_quantile_column, upper_quantile_column]. "
            f"Received {len(q_cols_processed)} columns: {q_cols_processed}"
        )
    qlow_col, qup_col = q_cols_processed[0], q_cols_processed[1]

    # Consolidate needed columns and check existence
    cols_needed = [actual_col, qlow_col, qup_col]
    # Handle theta_col warning (existence checked implicitly by df[cols])
    if theta_col:
        if theta_col not in df.columns:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found. "
                f"Using index order.",
                UserWarning,
                stacklevel=2,
            )
        # Don't add to cols_needed if missing
        else:
            warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but currently "
                f"ignored for positioning/ordering. Using index order.",
                UserWarning,
                stacklevel=2,
            )

    missing_essential = [
        col
        for col in [actual_col, qlow_col, qup_col]
        if col not in df.columns
    ]
    if missing_essential:
        raise ValueError(
            f"Essential columns missing: {', '.join(missing_essential)}"
        )

    # Drop rows with NaNs in essential columns
    data = df[cols_needed].dropna().copy()
    if len(data) == 0:
        warnings.warn(
            "DataFrame is empty after dropping NaN values."
            " Cannot generate plot.",
            UserWarning,
            stacklevel=2,
        )
        return None
    N = len(data)  # Number of valid points

    # --- Calculate Coverage ---
    try:
        y = data[actual_col].to_numpy(dtype=float)
        y_lo = data[qlow_col].to_numpy(dtype=float)
        y_hi = data[qup_col].to_numpy(dtype=float)
    except Exception as e:
        raise TypeError(
            f"Could not convert actual or quantile columns to numeric."
            f" Check data types. Original error: {e}"
        ) from e

    # Binary coverage indicator (1 if covered, 0 otherwise)
    covered = ((y >= y_lo) & (y <= y_hi)).astype(int)
    # Overall coverage rate
    total_coverage = covered.mean() if N > 0 else 0.0

    # --- Angular Coordinate Calculation ---
    acov_map = {  # Map name to (min_angle, max_angle) in radians
        "default": (0, 2 * np.pi),
        "half_circle": (0, np.pi),
        "quarter_circle": (0, np.pi / 2),
        "eighth_circle": (0, np.pi / 4),
    }
    if acov.lower() not in acov_map:
        # Use .get() for default or raise error explicitly? Let's raise.
        raise ValueError(
            f"Invalid `acov` value '{acov}'. Choose from: "
            f"{', '.join(acov_map.keys())}"
        )
    theta_min_rad, theta_max_rad = acov_map[acov.lower()]
    angular_range_rad = theta_max_rad - theta_min_rad

    # Calculate theta based on index, mapped to the angular range
    theta = (
        np.linspace(
            0.0, 1.0, N, endpoint=False
        )  # Use endpoint=False for bars/scatter
        * angular_range_rad
        + theta_min_rad
    )

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    ax.set_thetamin(np.degrees(theta_min_rad))
    ax.set_thetamax(np.degrees(theta_max_rad))
    # Set radial limits strictly to [0, 1] for coverage plot
    ax.set_ylim(0, 1.05)  # Slight padding at top
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Set explicit radial ticks

    # Apply grid and angle label settings
    set_axis_grid(ax, show_grid, grid_props)
    if mask_angle:
        ax.set_xticklabels([])

    # --- Optional Background Gradient ---
    if fill_gradient:
        try:
            grad_cmap_obj = get_cmap(gradient_cmap)
            # Create radial and angular meshgrid
            # R goes from 0 up to the total_coverage value
            R, T = np.meshgrid(
                np.linspace(0, total_coverage, gradient_size),
                np.linspace(theta_min_rad, theta_max_rad, gradient_size),
            )
            # Z represents the value to map color to (e.g., radius itself)
            # Tile radius values across angles for pcolormesh
            Z = np.tile(
                np.linspace(0, total_coverage, gradient_size)[:, np.newaxis],
                (1, gradient_size),
            )
            # Normalize Z based on [0, 1] range for colormap
            norm_gradient = Normalize(vmin=0, vmax=1.0)
            # Plot the gradient mesh
            ax.grid(False)
            ax.pcolormesh(
                T,
                R,
                Z,
                shading="auto",
                cmap=grad_cmap_obj,
                alpha=0.20,  # Make gradient subtle
                norm=norm_gradient,  # Ensure consistent color mapping
            )
            ax.grid(True, which="both")
        except ValueError:
            warnings.warn(
                f"Invalid `gradient_cmap` ('{gradient_cmap}')."
                f" Skipping background gradient.",
                UserWarning,
                stacklevel=2,
            )
        except Exception as e:
            warnings.warn(
                f"Failed to plot background gradient: {e}",
                UserWarning,
                stacklevel=2,
            )

    # --- Optional Concentric Reference Lines ---
    # Define default levels if None
    gradient_levels = (
        gradient_levels
        if gradient_levels is not None
        else [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    # Filter levels to be within [0, 1]
    valid_levels = [lv for lv in gradient_levels if 0 <= lv <= 1]
    # Define default properties, merge with user props
    default_gradient_props = {
        "linestyle": ":",
        "color": "gray",
        "linewidth": 0.8,
        "alpha": 0.8,
    }
    current_gradient_props = {
        **default_gradient_props,
        **(gradient_props or {}),
    }

    # Plot each reference line and its label
    theta_line = np.linspace(theta_min_rad, theta_max_rad, buffer_pts)
    for lv in valid_levels:
        ax.plot(theta_line, [lv] * buffer_pts, **current_gradient_props)
        # Add text label near the end of the line
        ax.text(
            theta_max_rad * 0.98,  # Position slightly inside max angle
            lv,
            f"{lv:.1f}",  # Format label
            color=current_gradient_props.get("color", "gray"),
            fontsize=8,
            ha="right",  # Horizontal alignment
            va="center",  # Vertical alignment
        )

    # --- Average Coverage Line ---
    ax.plot(
        theta_line,
        [total_coverage] * buffer_pts,
        color=coverage_line_color,
        linewidth=2.0,  # Make it prominent
        label=f"Avg Coverage ({total_coverage:.2f})",  # Add rate to label
    )

    # --- Plot Individual Coverage (Bars or Scatter) ---
    # Radius 'r' is the 'covered' array (0 or 1)
    r_plot = covered.astype(float)
    # Setup colormap and normalization for points/bars
    try:
        cmap_main_obj = get_cmap(cmap)
    except ValueError:
        warnings.warn(
            f"Invalid `cmap` ('{cmap}'). Using 'RdYlGn'.",
            UserWarning,
            stacklevel=2,
        )
        cmap_main_obj = get_cmap("RdYlGn")
    # Normalize 0 to low end, 1 to high end of cmap
    norm_main = Normalize(vmin=0, vmax=1)
    point_colors = cmap_main_obj(norm_main(r_plot))

    if as_bars:
        # Calculate bar width to fill space (approximate)
        # Use difference between angles, handle wrap around if full circle
        if N > 1:
            # Estimate width based on average angular spacing
            # Could also use angular_range_rad / N
            bar_widths = np.diff(theta, append=theta[0] + angular_range_rad)
            # Ensure positive widths if theta wraps around
            bar_widths = (
                np.abs(bar_widths) * 0.9
            )  # Make slightly narrower than gap
        elif N == 1:
            bar_widths = (
                angular_range_rad * 0.8
            )  # Default width for single bar
        else:  # N=0
            bar_widths = 0.1  # Placeholder

        ax.bar(
            theta,  # Angle for each bar
            r_plot,  # Height (0 or 1)
            width=bar_widths,  # Width of bars
            bottom=0.0,  # Bars start from center
            color=point_colors,
            alpha=alpha,
            edgecolor="gray",
            linewidth=0.5,
        )
    else:  # Plot as scatter points
        ax.scatter(
            theta,
            r_plot,  # Radius (0 or 1)
            c=point_colors,  # Color based on coverage
            s=s,
            alpha=alpha,
            edgecolor="gray",
            linewidth=0.5,
            # No label needed here, color indicates status
        )

    # --- Final Touches ---
    ax.set_title(
        title or "Prediction Interval Coverage Diagnostic",
        fontsize=14,
        y=1.08,
    )
    # Add legend only for the average coverage line
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # If the coverage line was plotted and labeled
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper right",
            bbox_to_anchor=(1.25, 1.1),
            fontsize=9,
        )

    # Set radial axis label (Y-axis in polar)
    ax.set_ylabel("Coverage (1=In, 0=Out)", labelpad=15, fontsize=10)

    # --- Logging ---
    if verbose > 0:
        print("-" * 50)
        print("Coverage Rate Calculation:")
        print(f"  Interval: [{qlow_col}, {qup_col}]")
        print(f"  Points checked (after NaN removal): {N}")
        print(f"  Overall Coverage Rate: {total_coverage * 100:.2f}%")
        print("-" * 50)

    # --- Output ---
    plt.tight_layout()
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


plot_coverage_diagnostic.__doc__ = r"""
Diagnose prediction-interval coverage on a polar plot.

This visualization checks whether observed values fall within their
predicted intervals and summarizes the **empirical coverage rate**
against the nominal level. It is a compact diagnostic for calibration
of quantile/interval forecasts; foundational background on forecast
verification and calibration appears in :footcite:p:`Gneiting2007b,
Jolliffe2012`. See :footcite:t:`Gneiting2007b` for a discussion of
calibration vs. sharpness.

The plot maps samples around a circle (angular coordinate) and encodes
**covered** vs **not covered** on the radial axis. A solid reference
ring marks the overall coverage rate, and optional concentric guides
and background gradients aid interpretation.

Parameters
----------
df : pandas.DataFrame
    Input table containing the observed target and the two quantile
    bounds. Decorators ensure a valid, non-empty DataFrame.

actual_col : str
    Column name of the observed (ground-truth) values.

q_cols : list of str or tuple of str
    Exactly two names ``[lower_quantile_col, upper_quantile_col]``
    that define the prediction interval. The order must be
    *[lower, upper]*.

theta_col : str, optional
    Intended column for angular ordering. **Currently ignored**; the
    plot uses the DataFrame index order. A warning is issued if
    provided. Default is ``None``.

acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'}, \
default='default'
    Angular coverage (span) of the plot:

    - ``'default'``: full circle :math:`2\pi`
    - ``'half_circle'``: :math:`\pi`
    - ``'quarter_circle'``: :math:`\pi/2`
    - ``'eighth_circle'``: :math:`\pi/4`

figsize : tuple of (float, float), default=(8.0, 8.0)
    Figure size in inches.

title : str, optional
    Custom title. If ``None``, a default title is used.

show_grid : bool, default=True
    If ``True``, show polar grid lines.

grid_props : dict, optional
    Keyword arguments passed to your grid helper for customizing the
    grid (e.g., ``{'linestyle': '--', 'alpha': 0.6}``).

cmap : str, default='RdYlGn'
    Colormap for per-point coverage (0 or 1). Diverging maps work well
    (e.g., red for uncovered, green for covered).

alpha : float, default=0.85
    Transparency for scatter points or bars.

s : int, default=35
    Marker size for scatter points (ignored when ``as_bars=True``).

as_bars : bool, default=False
    If ``True``, draw radial bars (height 0/1) instead of points.

coverage_line_color : str, default='r'
    Color of the solid ring at the average coverage rate.

buffer_pts : int, default=500
    Number of samples used to draw smooth circular lines (average rate
    and guide rings).

fill_gradient : bool, default=True
    If ``True``, fill the background radially up to the average
    coverage with a subtle gradient.

gradient_size : int, default=300
    Resolution of the background gradient mesh.

gradient_cmap : str, default='Greens'
    Colormap used for the optional background gradient.

gradient_levels : list of float, optional
    Radii in :math:`[0,1]` for dashed concentric reference rings.
    Defaults to ``[0.2, 0.4, 0.6, 0.8, 1.0]``.

gradient_props : dict, optional
    Style for the concentric guide rings (e.g., ``{'linestyle': ':',
    'color': 'gray', 'linewidth': 0.8}``).

mask_angle : bool, default=True
    If ``True``, hide angular tick labels.

savefig : str, optional
    File path to save the figure. If ``None``, the figure is shown.

verbose : int, default=0
    If ``> 0``, print the computed overall coverage rate.

Returns
-------
ax : matplotlib.axes.Axes
    The polar Axes containing the coverage diagnostic.

Raises
------
TypeError
    If ``q_cols`` does not contain exactly two names.

ValueError
    If required columns are missing or cannot be coerced to numeric,
    or if ``acov`` is invalid.

See Also
--------
plot_anomaly_magnitude:
    Polar diagnostic for the *magnitude* and *type* of interval
    failures (under/over).
plot_reliability_diagram:
    Calibration (reliability) curves for probabilistic classifiers.

Notes
-----
Coverage for row :math:`j` is defined by the closed interval test
:math:`L_j \le y_j \le U_j`. Rows with missing values in essential
columns are removed prior to computation, so all symbols below refer
to the filtered data. The radial axis is fixed to :math:`[0,1]` and
encodes a binary outcome per sample, while a solid reference ring at
radius :math:`\bar{C}` summarizes the empirical coverage rate. Compare
:math:`\bar{C}` to the nominal level implied by your interval (e.g.,
:math:`0.8` for Q10–Q90, :math:`0.9` for Q5–Q95) to assess
calibration. Angular positions follow index order over the chosen span;
``theta_col`` is currently ignored for positioning.

Let :math:`y_j` denote the actual value, :math:`L_j` the lower bound,
and :math:`U_j` the upper bound for sample :math:`j`, with
:math:`j=0,\dots,N-1` after NaN removal. The per-sample coverage
indicator (radial coordinate :math:`r_j`) is

.. math::

   r_j \;=\;
   \begin{cases}
     1, & L_j \le y_j \le U_j, \\\\
     0, & \text{otherwise.}
   \end{cases}

The overall coverage rate drawn as a ring is

.. math::

   \bar{C} \;=\; \frac{1}{N} \sum_{j=0}^{N-1} r_j.

Let :math:`S \in \{2\pi,\;\pi,\;\pi/2,\;\pi/4\}` be the angular span
set by ``acov`` and let :math:`\theta_{\min}` be the start angle. The
angular coordinate for sample :math:`j` is

.. math::

   \theta_j \;=\; \frac{j}{N}\,S \;+\; \theta_{\min}.

Plotting semantics: each sample is placed at
:math:`(\theta_j, r_j)` and colored via ``cmap`` according to
:math:`r_j`; a solid ring at :math:`\bar{C}` is overlaid as a global
summary; optional concentric guides at user-specified radii and an
optional radial background gradient up to :math:`\bar{C}` provide
additional visual context.
Rows with NaNs in essential columns are dropped before computation. 
``theta_col`` is currently ignored (index order is used).

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.plot.uncertainty import plot_coverage_diagnostic

**1. Random Example (Well-calibrated 80% interval):**

>>> np.random.seed(0)
>>> N = 200
>>> df_cov_rand = pd.DataFrame({'id': range(N)})
>>> df_cov_rand['actual'] = np.random.normal(loc=10, scale=2, size=N)
>>> # Simulate an ~80% interval (e.g., +/- 1.28 std devs for Normal)
>>> std_dev_pred = 2.0
>>> df_cov_rand['q10_pred'] = 10 - 1.28 * std_dev_pred
>>> df_cov_rand['q90_pred'] = 10 + 1.28 * std_dev_pred
>>> # Add some noise to interval bounds
>>> df_cov_rand['q10_pred'] += np.random.randn(N) * 0.2
>>> df_cov_rand['q90_pred'] += np.random.randn(N) * 0.2
>>> ax_cov_rand = plot_coverage_diagnostic(
...     df=df_cov_rand,
...     actual_col='actual',
...     q_cols=['q10_pred', 'q90_pred'], # [lower, upper]
...     theta_col='id',           # Ignored for positioning
...     acov='default',
...     title='Coverage Diagnostic (Simulated 80% Interval)',
...     as_bars=False,           # Use scatter points
...     coverage_line_color='blue', # Color for avg coverage line
...     gradient_cmap='Blues',    # Background gradient color
...     verbose=1                # Print coverage rate
... )
>>> # Expected coverage rate near 80%
>>> # plt.show() called internally

**2. Concrete Example (Subsidence Data):**

>>> # Assume small_sample_pred is a loaded DataFrame
>>> # Create dummy data if it doesn't exist
>>> try:
...    small_sample_pred
... except NameError:
...    print("Creating dummy small sample prediction data...")
...    N_small = 200
...    small_sample_pred = pd.DataFrame({
...        'subsidence_2023': np.random.rand(N_small)*15 + np.linspace(0, 5, N_small),
...        'subsidence_2023_q10': np.random.rand(N_small)*10,
...        'subsidence_2023_q90': np.random.rand(N_small)*10 + 10,
...        'latitude': np.linspace(22.3, 22.7, N_small) + np.random.randn(N_small)*0.01
...     })
>>> # Ensure Q90 > Q10
>>> small_sample_pred['subsidence_2023_q90'] = (
...     small_sample_pred['subsidence_2023_q10'] +
...     np.abs(small_sample_pred['subsidence_2023_q90'] -
...            small_sample_pred['subsidence_2023_q10']) + 0.1
...     )
>>> ax_cov_sub = plot_coverage_diagnostic(
...     df=small_sample_pred,
...     actual_col='subsidence_2023',
...     q_cols=['subsidence_2023_q10', 'subsidence_2023_q90'],
...     theta_col=None,            # Use index order
...     acov='half_circle',      # Use 180 degrees
...     as_bars=True,            # Use bars instead of scatter
...     coverage_line_color='darkgreen',
...     title='Coverage Evaluation for 2023 (Q10Q90)',
...     mask_angle=False,         # Show angle labels if meaningful
...     fill_gradient=False,     # Turn off background gradient
...     gradient_levels=[0.5, 0.8, 0.9], # Custom reference lines
...     verbose=1
... )
>>> 

References
----------
.. footbibliography::
"""


@check_non_emptiness
@isdf
def plot_temporal_uncertainty(
    df: pd.DataFrame,
    q_cols: str | list[str] = "auto",
    theta_col: str | None = None,
    names: list[str] | None = None,
    acov: str = "default",
    figsize: tuple[float, float] = (8.0, 8.0),
    title: str | None = None,
    cmap: str = "tab10",
    normalize: bool = True,
    show_grid: bool = True,
    grid_props: dict | None = None,
    alpha: float = 0.7,
    s: int = 25,
    dot_style: str = "o",
    legend_loc: str = "upper right",
    mask_label: bool = False,
    mask_angle: bool = True,
    savefig: str | None = None,
):
    r"""Visualize multiple data series using polar scatter plots.

    This function creates a general-purpose polar scatter plot to
    visualize and compare one or more data series (columns) from a
    DataFrame in a circular layout. Each series is plotted with a
    distinct color [1]_.

    - **Angular Position (`theta`)**: Represents each data point or
      sample, ordered by the DataFrame index after removing rows with
      NaNs in the selected `q_cols` (and `theta_col` if used for NaN
      alignment). The points are mapped linearly onto the specified
      angular coverage (`acov`). The `theta_col` parameter is currently
      ignored for sorting/positioning but helps align data if it has NaNs.
    - **Radial Distance (`r`)**: Represents the magnitude of the values
      from each column specified in `q_cols`. Values can be optionally
      normalized independently for each series using min-max scaling
      (`normalize=True`).
    - **Color**: Each data series (column in `q_cols`) is assigned a
      unique color based on the specified `cmap` [2]_.

    This plot is flexible and can be used for various purposes, such as:

    - Comparing different model predictions for the same target.
    - Visualizing different quantile predictions (e.g., Q10, Q50, Q90)
      for a single time step to show uncertainty spread.
    - Plotting related variables against each other in a polar context.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data columns to be plotted.
        Decorators ensure it's a valid, non-empty pandas DataFrame.

    q_cols : str or list of str, default='auto'
        Specifies the columns to plot.

        - If ``'auto'``, attempts to automatically detect quantile columns
          (e.g., names containing '_q' followed by numbers) using the
          helper function `detect_quantiles_in`. Raises error if detection
          fails or no such columns are found.
        - If a list of strings, these column names are used directly.
          Must contain at least one valid column name from `df`.

    theta_col : str, optional
        *Intended* for ordering points angularly based on this column's
        values. *Note: Currently ignored for sorting/positioning; uses
        index order. However, if provided and present in `df`, it is
        included when checking for and dropping NaN values to ensure data
        alignment between angles and plotted values.* A warning is issued
        regarding its limited use. Default is ``None``.

    names : list of str, optional
        Custom labels for each data series specified in `q_cols`, used
        in the plot legend. Must match the number of columns in `q_cols`.
        If ``None``, generic labels like 'Q1', 'Q2', ... (or based on
        detected quantile names) are generated. Default is ``None``.

    acov : {'default', 'half_circle', 'quarter_circle', \
        'eighth_circle'}, default='default'

        Specifies the angular coverage (span) of the polar plot:
        ``'default'`` (360°), ``'half_circle'`` (180°),
        ``'quarter_circle'`` (90°), ``'eighth_circle'`` (45°).

    figsize : tuple of (float, float), default=(8.0, 8.0)
        Width and height of the figure in inches.

    title : str, optional
        Custom title for the plot. If ``None``, a default title is used.

    cmap : str, default='tab10'
        Name of the Matplotlib colormap used to assign distinct colors
        to each data series specified in `q_cols`.

    normalize : bool, default=True
        If ``True``, normalize the values within *each column* specified
        in `q_cols` independently to the range [0, 1] using min-max
        scaling before plotting. Useful for comparing shapes of series
        with different scales. If ``False``, raw values are plotted.

    show_grid : bool, default=True
        If ``True``, display the polar grid lines.

    alpha : float, default=0.7
        Transparency level for the scatter points (0=transparent, 1=opaque).

    s : int, default=25
        Marker size for the scatter points.

    dot_style : str, default='o'
        Marker style used for the scatter points (e.g., 'o', '.', 'x', '+').
        Passed to `matplotlib.pyplot.scatter`.

    legend_loc : str, default='upper right'
        Location of the legend identifying the data series. Common values
        include 'best', 'upper right', 'lower left', etc.

    mask_angle : bool, default=True
        If ``True``, hide the angular tick labels (degrees). Recommended
        if the angle is based on index.

    savefig : str, optional
        File path to save the plot image. If ``None``, displays interactively.

    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        The Matplotlib Axes object (PolarAxesSubplot) containing the plot.

    Raises
    ------
    ValueError
        If `q_cols='auto'` fails to find columns or `detect_quantiles_in` fails.
        If `q_cols` (explicitly provided or auto-detected) is empty or
        contains columns not found in `df`.
        If `names` is provided but length doesn't match `q_cols`.
        If `acov` value is invalid.
    TypeError
        If data in `q_cols` is not numeric.

    See Also
    --------
    plot_polar_uncertainty_spread : Previous function focused on plotting
                                   quantiles for uncertainty (potentially similar).
    detect_quantiles_in : Helper function for automatic column detection.
    matplotlib.pyplot.scatter : Underlying function for plotting points.

    Notes
    ------

    - Normalization :math:`\aleph` (`normalize=True`) is applied independently to each
      column in `q_cols`, scaling each series to its own [0, 1] range.
    - Rows containing NaN values in *any* of the selected `q_cols` (and
      `theta_col` if present) are dropped *before* plotting to ensure
      alignment. `theta` angles are based on the index of this cleaned data.
    - The `theta_col` parameter currently does not affect the order or
      position of points but is used for consistent NaN handling.
    - Radial axis ticks and labels are hidden by default (`set_yticklabels([])`)
      as the radius represents potentially normalized values from multiple
      series, making a single scale less meaningful.


    Let :math:`\mathbf{V}_i` be the data vector for column `i` in `q_cols`
    (length :math:`N`, after NaN removal based on all selected columns).

    1. **Normalization** (if `normalize=True`, applied per column `i`):

       .. math::

           \mathbf{v}'_i = \aleph ( \mathbf{V}_i)

       where

       .. math::

           \aleph(x)_j = \frac{x_j - \min(\mathbf{x})}{\max(\mathbf{x}) - \min(\mathbf{x})}

       (Handles zero range by returning the original vector).

    2. **Angular Coordinate (`theta`)**: Let :math:`S` be the angular span
       and :math:`\theta_{min}` the start angle from `acov`. For index
       :math:`j` (:math:`j=0, \dots, N-1` of cleaned data):

       .. math::
           \theta_j = \left( \frac{j}{N} \times S \right) + \theta_{min}

    3. **Radial Coordinate (`r`)**: For series `i` and point `j`:

       :math:`r_{i,j} = v'_{i,j}` (if normalized) or
       :math:`r_{i,j} = v_{i,j}` (if not normalized).

    4. **Plotting**: For each series `i`, plot points :math:`(r_{i,j}, \theta_j)`
       using `scatter` with a distinct color derived from `cmap`.

    References
    ----------
    .. [1] Kouadio, K. L., Liu, R., Loukou, K. G. H., Liu, J., & Liu, W. (2025).
       Analytics Framework for Interpreting Spatiotemporal Probabilistic Forecasts.
       *International Journal of Forecasting*. Manuscript submitted.
       
    .. [2] Matplotlib documentation: https://matplotlib.org/stable/

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from kdiagram.plot.uncertainty import plot_temporal_uncertainty

    **1. Random Example (Comparing two series):**

    >>> np.random.seed(42)
    >>> N = 100
    >>> df_comp_rand = pd.DataFrame({
    ...     'Index': range(N),
    ...     'ModelA_Pred': 50 + 10 * np.sin(np.linspace(0, 3 * np.pi, N)) + np.random.randn(N)*5,
    ...     'ModelB_Pred': 55 + 12 * np.sin(np.linspace(0, 3 * np.pi, N) - 0.5) + np.random.randn(N)*4,
    ... })
    >>> ax_comp_rand = plot_temporal_uncertainty(
    ...     df=df_comp_rand,
    ...     q_cols=['ModelA_Pred', 'ModelB_Pred'],
    ...     names=['Model A', 'Model B'],
    ...     theta_col=None,           # Use index order
    ...     acov='default',
    ...     title='Comparison of Model A vs Model B',
    ...     normalize=True,           # Normalize for shape comparison
    ...     cmap='Set1',
    ...     dot_style='x',            # Use 'x' markers
    ...     mask_angle=False          # Show angle ticks
    ... )
    >>> # plt.show() called internally

    **2. Concrete Example (Subsidence Quantiles for 2023):**

    >>> # Assume zhongshan_pred_2023_2026 is a loaded DataFrame
    >>> # Create dummy data if it doesn't exist
    >>> try:
    ...    zhongshan_pred_2023_2026
    ... except NameError:
    ...    print("Creating dummy subsidence data for example...")
    ...    N_sub = 150
    ...    zhongshan_pred_2023_2026 = pd.DataFrame({
    ...       'latitude': np.linspace(22.2, 22.8, N_sub),
    ...       'subsidence_2023_q10': np.random.rand(N_sub)*5 + 1 + np.linspace(0,2, N_sub),
    ...       'subsidence_2023_q50': np.random.rand(N_sub)*5 + 3 + np.linspace(1,3, N_sub),
    ...       'subsidence_2023_q90': np.random.rand(N_sub)*5 + 5 + np.linspace(2,4, N_sub),
    ...     })
    >>> # Ensure Q90 > Q50 > Q10 roughly
    >>> zhongshan_pred_2023_2026['subsidence_2023_q50'] = np.maximum(
    ...     zhongshan_pred_2023_2026['subsidence_2023_q50'],
    ...     zhongshan_pred_2023_2026['subsidence_2023_q10'] + 0.1)
    >>> zhongshan_pred_2023_2026['subsidence_2023_q90'] = np.maximum(
    ...     zhongshan_pred_2023_2026['subsidence_2023_q90'],
    ...     zhongshan_pred_2023_2026['subsidence_2023_q50'] + 0.1)
    >>> ax_tu_sub = plot_temporal_uncertainty(
    ...     df=zhongshan_pred_2023_2026.head(100), # Use subset
    ...     # Explicitly list quantile columns for 2023
    ...     q_cols=['subsidence_2023_q10', 'subsidence_2023_q50',
    ...             'subsidence_2023_q90'],
    ...     theta_col='latitude',       # Used for NaN alignment, not order
    ...     names=["Lower Bound (Q10)", "Median (Q50)", "Upper Bound (Q90)"],
    ...     acov='eighth_circle',    # Use smaller angle span
    ...     title='Uncertainty Spread for 2023 (Zhongshan)',
    ...     normalize=False,          # Plot raw values
    ...     cmap='coolwarm',          # Use diverging map for bounds
    ...     mask_angle=True,
    ...     s=30
    ... )
    >>> # plt.show() called internally

    """
    # --- Input Validation and Column Setup ---
    # Handle 'auto' detection or process explicit list for q_cols
    if isinstance(q_cols, str) and q_cols.lower() == "auto":
        try:
            # Assume detect_quantiles_in returns list of column names
            detected_cols = detect_quantiles_in(df)
            if not detected_cols:
                raise ValueError("Auto-detection found no quantile columns.")
            # Check if detected columns actually exist (redundant?)
            exist_features(
                df,
                features=detected_cols,
                error="raise",
                name="Auto-detected quantile columns",
            )
            q_cols_list = detected_cols
        except (
            NameError
        ) as err:  # If detect_quantiles_in is not defined/imported
            raise ImportError(
                "Helper function 'detect_quantiles_in' is needed for "
                "`q_cols='auto'` but seems unavailable."
            ) from err
        except Exception as e:
            # Catch errors from detect_quantiles_in or exist_features
            raise ValueError(
                f"Automatic detection or validation of `q_cols` failed: {e}. "
                "Please provide `q_cols` explicitly as a list of column names."
            ) from e
    else:
        # Process explicit list using columns_manager if available
        if "columns_manager" in globals():
            q_cols_list = columns_manager(q_cols, empty_as_none=False)
        else:
            # Basic list validation if helper not present
            if not isinstance(q_cols, (list, tuple)):
                raise TypeError("`q_cols` must be 'auto' or a list/tuple.")
            q_cols_list = list(q_cols)
        # Check if list is empty
        if not q_cols_list:
            raise ValueError(
                "`q_cols` list cannot be empty. Please provide column names."
            )
        # Check existence of explicitly provided columns
        exist_features(
            df,
            features=q_cols_list,
            error="raise",
            name="Specified plot columns (`q_cols`)",
        )

    # Determine columns needed for NaN handling
    cols_for_na_check = list(q_cols_list)  # Start with plot columns
    # theta_col_valid = False
    if theta_col:
        if theta_col in df.columns:
            cols_for_na_check.append(theta_col)
            # theta_col_valid = True
            warnings.warn(
                f"`theta_col` ('{theta_col}') is currently used for NaN "
                f"alignment but ignored for positioning/ordering. Using index order.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found. "
                f"It will be ignored for NaN alignment and positioning.",
                UserWarning,
                stacklevel=2,
            )

    # Drop rows with NaNs in ANY of the selected columns
    data = df[cols_for_na_check].dropna().copy()
    if len(data) == 0:
        warnings.warn(
            "DataFrame is empty after dropping NaN values. Cannot generate plot.",
            UserWarning,
            stacklevel=2,
        )
        return None
    N = len(data)  # Number of valid data points

    # --- Angular Coordinate Calculation ---
    angular_range_map = {  # Map name to (min_angle, max_angle) in radians
        "default": (0, 2 * np.pi),
        "half_circle": (0, np.pi),
        "quarter_circle": (0, np.pi / 2),
        "eighth_circle": (0, np.pi / 4),
    }
    if acov.lower() not in angular_range_map:
        raise ValueError(
            f"Invalid `acov` value '{acov}'. Choose from: "
            f"{', '.join(angular_range_map.keys())}"
        )
    theta_min_rad, theta_max_rad = angular_range_map[acov.lower()]
    angular_range_rad = theta_max_rad - theta_min_rad

    # Calculate theta based on index of cleaned data
    # Use endpoint=False for scatter for potentially better spacing visual
    theta = (
        np.linspace(0.0, 1.0, N, endpoint=False) * angular_range_rad
        + theta_min_rad
    )

    # --- Prepare Radial Values (Normalization) ---
    # Internal helper for min-max scaling
    def _normalize(x):
        """Min-max scales array x to [0, 1], handling zero range."""
        x = np.asarray(x, dtype=float)  # Ensure float array
        range_x = np.ptp(x)  # Peak-to-peak range
        if range_x > 1e-9:  # Check if range is non-negligible
            min_x = np.min(x)
            return (x - min_x) / range_x

        # XXX TODO
        # If range is zero or near-zero, return array of 0.5 or 0?
        # Returning original might be safer if scale is important even if constant
        # Let's return 0.5 for constant values after normalization request.
        # Or return 0 to keep origin clear? Let's use 0.5 as neutral midpoint.
        return np.full_like(x, 0.5)

    # Extract values from the cleaned data, apply normalization if needed
    values_list = []
    for col in q_cols_list:
        try:
            vals = data[col].to_numpy(dtype=float)
            if normalize:
                vals = _normalize(vals)
            values_list.append(vals)
        except Exception as e:
            raise TypeError(
                f"Could not process column '{col}'. Ensure it contains numeric"
                f" data. Original error: {e}"
            ) from e

    # --- Setup Labels and Colors ---
    num_series = len(q_cols_list)
    # Generate default names if needed
    if names is None:
        # Try to make default names slightly more informative if auto-detected
        if isinstance(q_cols, str) and q_cols.lower() == "auto":
            # Use the detected names if possible
            plot_labels = [c.replace("_", " ").title() for c in q_cols_list]
        else:
            # Generic default if explicit list or auto-detection failed context
            plot_labels = [f"Series {i+1}" for i in range(num_series)]
    else:
        # Use provided names, check length
        if len(names) != num_series:
            raise ValueError(
                f"Length of `names` ({len(names)}) must match the number "
                f"of plotted columns ({num_series})."
            )
        plot_labels = list(names)

    # Get color palette
    try:
        cmap_obj = get_cmap(cmap)
        # Sample colors - handle discrete vs continuous cmaps
        if hasattr(cmap_obj, "colors") and len(cmap_obj.colors) >= num_series:
            # Use colors directly from discrete map if enough available
            color_palette = cmap_obj.colors
        else:  # Sample from continuous map or discrete map with fewer colors
            color_palette = cmap_obj(np.linspace(0, 1, num_series))
    except ValueError:
        warnings.warn(
            f"Invalid `cmap` name '{cmap}'. Falling back to 'tab10'.",
            stacklevel=2,
        )
        cmap_obj = get_cmap("tab10")  # Fallback cmap
        color_palette = cmap_obj(np.linspace(0, 1, num_series))

    # --- Plotting ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )

    # Set angular limits
    ax.set_thetamin(np.degrees(theta_min_rad))
    ax.set_thetamax(np.degrees(theta_max_rad))

    # Plot each data series
    for i, (vals, label) in enumerate(zip(values_list, plot_labels)):
        if len(vals) != N:  # Should not happen if NaN handling is correct
            raise InternalError(
                "Mismatch between data length and theta length."
            )
        color = color_palette[
            i % len(color_palette)
        ]  # Cycle colors if needed
        ax.scatter(
            theta,
            vals,
            label=label,
            alpha=alpha,
            s=s,
            marker=dot_style,
            color=color,
            edgecolor="k",  # Add edge color for visibility
            linewidth=0.5,
        )

    # --- Final Touches ---
    ax.set_title(title or "Polar Data Series Plot", fontsize=14, y=1.08)
    set_axis_grid(ax, show_grid, grid_props)

    if mask_angle:
        ax.set_xticklabels([])  # Hide angular tick labels
    if mask_label:
        ax.set_yticklabels([])  # Hide radial tick labels
    else:
        ax.set_ylabel(f"{'Normalized ' if normalize else ''}Q_pred values")
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if labels:  # Only show legend if labels were generated/provided
        ax.legend(
            handles=handles,
            labels=labels,
            loc=legend_loc,
            bbox_to_anchor=(1.25, 1.0) if "right" in legend_loc else None,
            fontsize=9,
        )

    # Set radial axis label if normalization is off? Maybe not useful.
    if (
        not normalize and len(q_cols_list) == 1
    ):  # Only label if one series, unnormalized
        ax.set_ylabel(q_cols_list[0])

    plt.tight_layout()

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax


@check_non_emptiness
@isdf
def plot_radial_density_ring(
    df: pd.DataFrame,
    *,
    kind: Literal["width", "velocity", "direct"] = "direct",
    target_cols: str | list[str],
    title: str | None = None,
    r_label: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    alpha: float = 0.8,
    cbar: bool = True,
    show_grid: bool = True,
    grid_props: dict[str, any] | None = None,
    mask_angle: bool = True,
    bandwidth: float | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    show_yticklabels: bool = False,
):
    if show_yticklabels:
        warnings.warn(
            'Parameter "show_yticklabels" is deprecated as of v1.2.0'
            " and will be removed in a future release. "
            'Use "mask_angle" instead '
            "(set mask_angle=False to show angular labels, "
            "mask_angle=True to hide).",
            DeprecationWarning,
            stacklevel=2,
        )
        # delegate to mask angle
        mask_angle = False

    cols = columns_manager(target_cols)
    exist_features(df, features=cols)

    if kind == "direct":
        if len(cols) != 1:
            raise ValueError(
                "`kind='direct'` requires one column in "
                f"`target_cols`, but got {len(cols)}."
            )
        data_1d = df[cols[0]]
        r_label = r_label or cols[0]
        title = title or f"Distribution of {cols[0]}"

    elif kind in ("width", "velocity"):
        if len(cols) != 2:
            raise ValueError(
                f"`kind='{kind}'` requires two columns in "
                f"`target_cols`, but got {len(cols)}."
            )
        data_1d = df[cols[1]] - df[cols[0]]
        r_label = r_label or f"{cols[1]} $-$ {cols[0]}"
        title = title or f"Distribution of {kind.title()}"
    else:
        raise ValueError(
            f"Invalid `kind`: '{kind}'. Use 'width', "
            "'velocity', or 'direct'."
        )

    data_1d = data_1d.dropna()
    if data_1d.empty:
        warnings.warn(
            "Derived data is empty after dropping NaNs. "
            "Cannot generate plot.",
            UserWarning,
            stacklevel=2,
        )
        return None

    grid, pdf = _calculate_kde_for_ring(
        data_1d.to_numpy(), bandwidth=bandwidth
    )
    if grid.size == 0:
        warnings.warn(
            "Could not compute KDE. Cannot generate plot.",
            UserWarning,
            stacklevel=2,
        )
        return None

    pdf_norm = pdf.copy()
    pdf_max = np.max(pdf_norm)
    if pdf_max > 0:
        pdf_norm /= pdf_max

    thetas = np.linspace(0, 2 * np.pi, 128)
    T, R = np.meshgrid(thetas, grid)
    C = np.tile(pdf_norm, (len(thetas), 1)).T

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    cmap_obj = get_cmap(cmap, default="viridis")

    ax.grid(False)
    pcm = ax.pcolormesh(
        T,
        R,
        C,
        shading="auto",
        cmap=cmap_obj,
        alpha=alpha,
        vmin=0,
        vmax=1,
    )

    ax.set_title(title, va="bottom", fontsize=14)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
    ax.set_thetagrids([])
    ax.set_rlabel_position(0)

    if r_label:
        ax.set_ylabel(r_label, fontsize=12, labelpad=20)

    if mask_angle:
        ax.set_yticklabels([])

    if cbar:
        cbar_obj = plt.colorbar(pcm, ax=ax, pad=0.1, shrink=0.7)
        cbar_obj.set_label(
            "Normalized Density",
            rotation=270,
            labelpad=15,
            fontsize=10,
        )

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_radial_density_ring.__doc__ = r"""
Plot a radial density ring to visualize a 1D distribution.

This function computes the probability density function (PDF) of a
one-dimensional variable using Kernel Density Estimation (KDE) and
visualizes it as a colored ring on a polar plot. It's a
powerful way to inspect the shape, peaks, and spread of a
distribution, such as forecast errors or interval widths.

- **Radial Distance (`r`)**: Represents the domain of the
  variable being analyzed (e.g., interval width in mm).
- **Color**: Represents the **probability density** at each
  radial position. More intense colors indicate more common
  values (peaks in the distribution).
- **Angular Position (`theta`)**: Is purely for aesthetics and
  carries no information; the density is uniform around the
  circle to form the ring.


Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the data.

kind : {'width', 'velocity', 'direct'}, default='direct'
    Specifies how to derive the 1D data for the distribution
    plot from the ``target_cols``.
    
    - ``'direct'``: Directly uses the data from a single
      column specified in ``target_cols``.
    - ``'width'``: Calculates the interval width by
      subtracting the first column from the second in
      ``target_cols`` (e.g., ``q90 - q10``).
    - ``'velocity'``: Calculates the difference between two
      columns, representing a change or velocity (e.g.,
      ``value_t2 - value_t1``).

target_cols : str or list of str
    The column(s) to use for deriving the data.
    
    - For ``kind='direct'``, this must be a single column name
      (``str``).
    - For ``kind='width'`` or ``kind='velocity'``, this must be a
      list of exactly two column names (``[col1, col2]``).

title : str, optional
    The title for the plot. If ``None``, a default is generated.

r_label : str, optional
    Custom label for the radial axis. If ``None``, a default
    is generated from the column names.

figsize : tuple of (float, float), default=(8, 8)
    Figure size in inches.

cmap : str, default='viridis'
    Matplotlib colormap name for the density ring.

alpha : float, default=0.8
    Transparency level for the density ring.

cbar : bool, default=True
    If ``True``, display a color bar representing the
    normalized density.

show_grid : bool, default=True
    Toggle gridlines via the package helper ``set_axis_grid``.

grid_props : dict, optional
    Keyword arguments passed to ``set_axis_grid`` for grid
    customization.

mask_angle : bool, default=True
    If ``False``, show the tick labels on the radial axis.
    Defaults to ``True`` as the ring is primarily a
    qualitative visual.

bandwidth : float, optional
    The bandwidth for the Kernel Density Estimation. If ``None``,
    it is estimated using Silverman's rule of thumb.

savefig : str, optional
    If provided, save the figure to this path; otherwise the
    plot is shown interactively.

dpi : int, default=300
    Resolution for the saved figure.

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if the plot could not be generated (e.g., due to empty data).

Notes
-----
The plot is generated through the following steps:

1.  **Data Derivation**: A one-dimensional data vector,
    :math:`\mathbf{x}`, is created from the ``target_cols``
    based on the specified ``kind``.

2.  **Kernel Density Estimation**: The probability density
    function (PDF), :math:`\hat{f}_h(x)`, is estimated from
    the data vector :math:`\mathbf{x}` using a Gaussian kernel.
    This step is handled by ``scipy.stats.gaussian_kde``.

3.  **Normalization for Coloring**: The estimated PDF is
    normalized to the range ``[0, 1]`` to map its values to the
    color map.

    .. math::
        \text{PDF}_{\text{norm}}(x) = \frac{\hat{f}_h(x)}{\max(\hat{f}_h)}

4.  **Visualization**: The plot is rendered on polar axes using
    ``pcolormesh``. The radial axis corresponds to the value
    domain :math:`x`, and the color at each radius corresponds
    to :math:`\text{PDF}_{\text{norm}}(x)`. This color value is
    repeated across all angles to form a ring.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.uncertainty import plot_radial_density_ring
>>>
>>> # Create synthetic data
>>> np.random.seed(42)
>>> n_samples = 500
>>> df_test = pd.DataFrame({
...     'q10_pred': np.random.normal(10, 2, n_samples),
...     'q90_pred': np.random.normal(30, 3, n_samples),
... })
>>> # Ensure upper bound is greater than lower bound
>>> df_test['q90_pred'] = df_test[['q10_pred', 'q90_pred']].max(axis=1) \
...     + np.random.rand(n_samples)
>>>
>>> # Generate the plot for the distribution of interval widths
>>> ax = plot_radial_density_ring(
...     df=df_test,
...     kind="width",
...     target_cols=["q10_pred", "q90_pred"],
...     title="Distribution of Prediction Interval Width",
...     r_label="Interval Width (mm)",
...     cmap="magma",
...     show_yticklabels=True
... )
"""


@check_non_emptiness
@isdf
def plot_polar_quiver(
    df: pd.DataFrame,
    r_col: str,
    theta_col: str,
    u_col: str,
    v_col: str,
    *,
    color_col: str | None = None,
    theta_period: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    mask_angle: bool = False,
    mask_radius: bool = False,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    **quiver_kws,
):

    required_cols = [r_col, theta_col, u_col, v_col]
    if color_col:
        required_cols.append(color_col)
    exist_features(df, features=required_cols)

    data = df[required_cols].dropna()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs in "
            "required columns. Cannot plot.",
            UserWarning,
            stacklevel=2,
        )
        return None

    r_data = data[r_col].to_numpy()
    theta_data = data[theta_col].to_numpy()
    u_data = data[u_col].to_numpy()  # Radial component
    v_data = data[v_col].to_numpy()  # Angular component

    if theta_period:
        theta_rad = (theta_data % theta_period) / theta_period * (2 * np.pi)
    else:
        min_theta, max_theta = theta_data.min(), theta_data.max()
        if (max_theta - min_theta) > 1e-9:
            theta_rad = (
                ((theta_data - min_theta) / (max_theta - min_theta))
                * 2
                * np.pi
            )
        else:
            theta_rad = np.zeros_like(theta_data)

    if color_col:
        color_data = data[color_col].to_numpy()
        cbar_label = color_col
    else:
        # Default to coloring by vector magnitude
        color_data = np.sqrt(u_data**2 + v_data**2)
        cbar_label = "Vector Magnitude"

    norm = Normalize(vmin=np.min(color_data), vmax=np.max(color_data))
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(norm(color_data))

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )

    # Matplotlib's quiver on a polar axis expects:
    # (theta, r, angular_change, radial_change)
    ax.quiver(theta_rad, r_data, v_data, u_data, color=colors, **quiver_kws)

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        ax=ax,
        pad=0.1,
        shrink=0.75,
    )
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_title(title or f"Quiver Plot of ({u_col}, {v_col})")
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


plot_polar_quiver.__doc__ = r"""
Plot a polar quiver plot to visualize vector data.

This function draws arrows (vectors) on a polar grid. Each arrow's
origin, direction, and magnitude are determined by the data. This is
highly effective for visualizing dynamic phenomena like forecast
revisions, error vectors, or flow fields footcite:p:`Brewer2003`.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the vector data.

r_col : str
    Name of the column for the **radial position** of the vector's
    origin (tail).

theta_col : str
    Name of the column for the **angular position** of the vector's
    origin.

u_col : str
    Name of the column for the **radial component** of the vector
    (the change in radius).

v_col : str
    Name of the column for the **tangential component** of the vector
    (the change in angle).

color_col : str, optional
    Name of a column to use for coloring the arrows. If ``None``,
    arrows are colored by their total vector magnitude.

theta_period : float, optional
    The period of the cyclical data in ``theta_col``. For example,
    if ``theta_col`` represents degrees in a circle, the period is
    360. If provided, the angle is mapped to ``[0, 2*pi]`` based
    on this cycle.

title : str, optional
    The title for the plot. If ``None``, a default is generated.

figsize : tuple of (float, float), default=(8, 8)
    Figure size in inches.

cmap : str, default='viridis'
    Matplotlib colormap for coloring the arrows.

show_grid : bool, default=True
    Toggle gridlines via the package helper ``set_axis_grid``.

grid_props : dict, optional
    Keyword arguments passed to ``set_axis_grid`` for grid
    customization.

mask_angle : bool, default=False
    If ``True``, hide the angular tick labels (e.g., degrees).

mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.

savefig : str, optional
    If provided, save the figure to this path; otherwise the
    plot is shown interactively.

dpi : int, default=300
    Resolution for the saved figure.

**quiver_kws : dict, optional
    Additional keyword arguments passed to the ``ax.quiver`` call
    (e.g., ``scale``, ``width``, ``headwidth``).

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if the plot could not be generated.

Notes
-----
The plot visualizes a vector field on a polar coordinate system. For
each data point :math:`i`, a vector is drawn footcite:p:`Hunter2007`.

1.  **Vector Origin**: The tail of the vector is positioned at the
    polar coordinates :math:`(r_i, \theta_i)` specified by ``r_col``
    and ``theta_col``.

2.  **Vector Components**: The vector itself is defined by its
    components in the radial and tangential directions:
        
    - :math:`u_i` (from ``u_col``) is the component in the radial
      direction (pointing away from the center).
    - :math:`v_i` (from ``v_col``) is the component in the tangential
      direction (perpendicular to the radial line).

3.  **Visualization**: Matplotlib's polar ``quiver`` function places an
    arrow at each origin point :math:`(r_i, \theta_i)`. The arrow's
    length and orientation are determined by the vector
    :math:`(u_i, v_i)`.

4.  **Coloring**: The color of each arrow can be determined by a
    separate metric from ``color_col``. If not specified, it defaults
    to the vector's Euclidean magnitude, :math:`M_i`.

    .. math::

        M_i = \sqrt{u_i^2 + v_i^2}

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.uncertainty import plot_polar_quiver
>>>
>>> # Simulate forecast revisions for 50 spatial locations
>>> np.random.seed(0)
>>> n_points = 50
>>> locations = np.linspace(0, 360, n_points, endpoint=False)
>>> initial_forecast = 10 + 5 * np.sin(np.deg2rad(locations) * 3)
>>>
>>> # Simulate revisions (changes in value)
>>> radial_change = np.random.normal(0, 1.5, n_points)
>>> tangential_change = np.random.normal(0, 0.1, n_points)
>>>
>>> df_forecasts = pd.DataFrame({
...     'location_angle': locations,
...     'initial_value': initial_forecast,
...     'update_radial': radial_change,
...     'update_tangential': tangential_change,
... })
>>>
>>> # Generate the polar quiver plot
>>> ax = plot_polar_quiver(
...     df=df_forecasts,
...     r_col='initial_value',
...     theta_col='location_angle',
...     u_col='update_radial',
...     v_col='update_tangential',
...     theta_period=360,
...     title='Forecast Revisions for Spatial Locations',
...     cmap='coolwarm',
...     scale=25  # Adjusts arrow size for better visibility
... )

References
----------

.. footbibliography::
    
"""


@check_non_emptiness
@isdf
def plot_polar_heatmap(
    df: pd.DataFrame,
    r_col: str,
    theta_col: str,
    *,
    theta_period: float | None = None,
    r_bins: int = 20,
    theta_bins: int = 36,
    statistic: str = "count",
    cbar_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    mask_angle: bool = False,
    mask_radius: bool = False,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
):
    exist_features(df, features=[r_col, theta_col])

    data = df[[r_col, theta_col]].dropna()
    if data.empty:
        warnings.warn(
            "DataFrame is empty after dropping NaNs in "
            f"'{r_col}' or '{theta_col}'. Cannot plot.",
            UserWarning,
            stacklevel=2,
        )
        return None

    r_data = data[r_col].to_numpy()
    theta_data = data[theta_col].to_numpy()

    if theta_period:
        # Map to [0, 2*pi] based on a known cycle
        theta_rad = (theta_data % theta_period) / theta_period * (2 * np.pi)
    else:
        # Min-max scale to [0, 2*pi]
        min_theta, max_theta = theta_data.min(), theta_data.max()
        if (max_theta - min_theta) > 1e-9:
            theta_rad = (
                ((theta_data - min_theta) / (max_theta - min_theta))
                * 2
                * np.pi
            )
        else:
            theta_rad = np.zeros_like(theta_data)

    # Define bins for the 2D histogram
    r_edges = np.linspace(r_data.min(), r_data.max(), r_bins + 1)
    theta_edges = np.linspace(0, 2 * np.pi, theta_bins + 1)

    # Use np.histogram2d to get counts in each polar bin
    counts, _, _ = np.histogram2d(
        x=r_data, y=theta_rad, bins=[r_edges, theta_edges]
    )

    # Transpose counts to match pcolormesh expectations
    # counts = counts.T

    # Create the polar plot

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    cmap_obj = get_cmap(cmap, default="viridis")

    ax.grid(False)
    # Use pcolormesh to draw the heatmap
    T, R = np.meshgrid(theta_edges, r_edges)
    pcm = ax.pcolormesh(T, R, counts, cmap=cmap_obj, shading="auto")

    # Add a color bar
    cbar = plt.colorbar(pcm, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label(cbar_label or statistic.capitalize(), fontsize=10)

    ax.set_title(title or f"Density of {r_col} vs. {theta_col}")
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


def _calculate_kde_for_ring(
    data: np.ndarray, bandwidth: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal helper to compute KDE. In a package, this
    should be a shared utility.
    """
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([]), np.array([])

    lo, hi = np.min(data), np.max(data)
    pad = 0.05 * (hi - lo) if (hi - lo) > 0 else 0.5
    grid = np.linspace(lo - pad, hi + pad, 512)

    if bandwidth is None:
        std = np.std(data, ddof=1) if data.size > 1 else 1.0
        # Silverman's rule of thumb
        bw_val = 1.06 * std * (data.size ** (-1 / 5)) if std > 0 else 1.0
    else:
        bw_val = bandwidth

    std_dev = np.std(data, ddof=1)
    bw_method = bw_val / std_dev if std_dev > 0 else "silverman"
    kde = gaussian_kde(data, bw_method=bw_method)
    pdf = kde(grid)
    return grid, pdf


plot_polar_heatmap.__doc__ = r"""
Plot a polar heatmap to visualize a 2D density distribution.

This function creates a heatmap on a polar grid to show the
concentration of data points based on two variables: one mapped to
the **radius** and another to the **angle**. It is highly effective
for identifying patterns and correlations between a linear metric and
a cyclical or ordered feature.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the data.

r_col : str
    Name of the column to be used for the radial coordinate.

theta_col : str
    Name of the column to be used for the angular coordinate.

theta_period : float, optional
    The period of the cyclical data in ``theta_col``. For example,
    if ``theta_col`` represents the hour of the day, ``theta_period``
    should be 24. If provided, the angle is mapped to ``[0, 2*pi]``
    based on this cycle. If ``None``, the angle is scaled from the
    data's min/max values.

r_bins : int, default=20
    The number of bins to create along the radial axis.

theta_bins : int, default=36
    The number of bins to create around the angular axis.

statistic : str, default='count'
    The statistic to compute in each bin. Currently, only 'count'
    is supported, which corresponds to a 2D histogram.

cbar_label : str, optional
    Custom label for the color bar. If ``None``, it defaults to
    the value of ``statistic``.

title : str, optional
    The title for the plot. If ``None``, a default is generated.

figsize : tuple of (float, float), default=(8, 8)
    Figure size in inches.

cmap : str, default='viridis'
    Matplotlib colormap for the heatmap.

mask_angle : bool, default=False
    If ``True``, hide the angular tick labels (e.g., degrees).

mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.

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

Returns
-------
ax : matplotlib.axes.Axes or None
    The Matplotlib Axes object containing the plot, or ``None``
    if the plot could not be generated.

Notes
-----
The heatmap is constructed by binning the 2D polar space and
counting the number of data points that fall into each bin.

1.  **Coordinate Mapping**:
    
    - The radial data from ``r_col`` is used directly.
    - The angular data from ``theta_col``, :math:`\theta_{data}`, is
      converted to radians in the range :math:`[0, 2\pi]`. If
      ``theta_period`` (:math:`P`) is given, the mapping is:

      .. math::
          \theta_{rad} = \left( \frac{\theta_{data} \pmod P}{P} \right) \cdot 2\pi

      Otherwise, it is linearly scaled from the data's range.

2.  **Binning**: The data space is divided into a grid of polar
    bins defined by ``r_bins`` and ``theta_bins``. A 2D histogram
    is then computed, where the value of each cell :math:`C_{ij}`
    is the number of data points :math:`(r, \theta)` that fall
    within the radial and angular bounds of that cell.

3.  **Visualization**: The resulting count matrix :math:`\mathbf{C}`
    is visualized using ``pcolormesh``, where the color of each
    cell corresponds to its count, creating the heatmap effect.
    See :footcite:p:`Hunter:2007` for Matplotlib details.

See Also
--------
matplotlib.pyplot.pcolormesh :
    Draw a pseudocolor plot with a nonregular rectangular grid.
numpy.histogram2d :
    Compute the bi-dimensional histogram of two data samples.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from kdiagram.plot.uncertainty import plot_polar_heatmap
>>>
>>> # Simulate weather data with a daily cycle
>>> np.random.seed(42)
>>> n_points = 5000
>>> # Simulate hour of day with more events in the afternoon
>>> hour = np.concatenate([
...     np.random.normal(15, 2, int(n_points * 0.7)),
...     np.random.normal(5, 2, int(n_points * 0.3))
... ]) % 24
>>> # Simulate rainfall, correlated with afternoon hours
>>> rainfall = np.random.gamma(2, 5, n_points) + \
...     (hour > 12) * np.random.gamma(3, 5, n_points)
>>>
>>> df_weather = pd.DataFrame({'hour': hour, 'rainfall_mm': rainfall})
>>>
>>> # Generate the polar heatmap
>>> ax = plot_polar_heatmap(
...     df=df_weather,
...     r_col='rainfall_mm',
...     theta_col='hour',
...     theta_period=24,
...     r_bins=25,
...     theta_bins=24,
...     cmap='plasma',
...     title='Rainfall Intensity vs. Hour of Day',
...     cbar_label='Event Count'
... )

References
----------

.. footbibliography::
    
"""


class PerformanceWarning(Warning):
    pass


# Define InternalError for consistency if needed
class InternalError(Exception):
    pass
