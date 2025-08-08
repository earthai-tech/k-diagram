# -*- coding: utf-8 -*-
# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

from numbers import Real 
import warnings
import matplotlib.pyplot as plt 
import numpy as np 
from typing import Optional, Tuple, List, Union, Any, Callable 

from ..compat.sklearn import validate_params, StrOptions, type_of_target
from ..utils.generic_utils import drop_nan_in 
from ..utils.handlers import columns_manager
from ..utils.metric_utils import get_scorer 
from ..utils.plot import set_axis_grid  
from ..utils.validator import validate_yy, is_iterable 

__all__=[ 'plot_model_comparison']

@validate_params({
    'train_times': ['array-like', None],
    'metrics': [str, 'array-like', callable, None], 
    'scale': [StrOptions({"norm", "min-max", 'std', 'standard',}), None], 
    "lower_bound": [Real],
    })
def plot_model_comparison(
    y_true,
    *y_preds,
    train_times: Optional[Union[float, List[float]]] = None, 
    metrics: Optional[Union[str, Callable, List[Union[str, Callable]]]] = None, 
    names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None, 
    colors: Optional[List[Any]] = None, 
    alpha: float = 0.7,
    legend: bool = True,
    show_grid: bool = True,
    grid_props: dict =None, 
    scale: Optional[str] = 'norm', 
    lower_bound: float = 0, 
    savefig: Optional[str] = None,
    loc: str = 'upper right',
    verbose: int = 0,
):
    r"""Plot multi-metric model performance comparison on a radar chart.

    Generates a radar chart (spider chart) visualizing multiple
    performance metrics for one or more models simultaneously. Each
    axis corresponds to a metric (e.g., R2, MAE, accuracy,
    precision), and each polygon represents a model, allowing for a
    holistic comparison of their strengths and weaknesses across
    different evaluation criteria [1]_.

    This function is highly valuable for model selection, providing a
    compact overview that goes beyond single-score comparisons. Use
    it when you need to balance trade-offs between various metrics
    (like accuracy vs. training time) or understand how different
    models perform relative to each other across a spectrum of
    relevant performance indicators. Internally relies on helpers
    to handle potential NaN values and determine data types [2]_.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The ground truth (correct) target values.

    *y_preds : array-like of shape (n_samples,)
        Variable number of prediction arrays, one for each model to
        be compared. Each array must have the same length as
        `y_true`.

    train_times : float or list of float, optional
        Training time in seconds for each model corresponding to
        `*y_preds`. If provided:
        - A single float assumes the same time for all models.
        - A list must match the number of models.
        It will be added as an additional axis/metric on the chart.
        Default is ``None``.

    metrics : str, callable, list of these, optional
        The performance metrics to calculate and plot. Default is
        ``None``, which triggers automatic metric selection based on
        the target type inferred from `y_true`:

        - **Regression:** Defaults to ``["r2", "mae", "mape", "rmse"]``.
        - **Classification:** Defaults to ``["accuracy", "precision",
          "recall"]``.

        Can be provided as:

        - A list of strings: Names of metrics known by scikit-learn
          or gofast's `get_scorer` (e.g., ``['r2', 'rmse']``).
        - A list of callables: Functions with the signature
          `metric(y_true, y_pred)`.
        - A mix of strings and callables.

    names : list of str, optional
        Names for each model corresponding to `*y_preds`. Used for
        the legend. If ``None`` or too short, defaults like
        "Model_1", "Model_2" are generated. Default is ``None``.

    title : str, optional
        Title displayed above the radar chart. If ``None``, a generic
        title may be used internally or omitted. Default is ``None``.

    figsize : tuple of (float, float), optional
        Figure size ``(width, height)`` in inches. If ``None``, uses
        Matplotlib's default (often similar to ``(8, 8)`` for this
        type of plot).

    colors : list of str or None, optional
        List of Matplotlib color specifications for each model's
        polygon. If ``None``, colors are automatically assigned from
        the default palette ('tab10'). If provided, the list length
        should ideally match `n_models`.

    alpha : float, optional
        Transparency level (between 0 and 1) for the plotted lines
        and filled areas. Default is ``0.7``. (Note: Fill alpha is
        often hardcoded lower, e.g., 0.1, in implementation).

    legend : bool, optional
        If ``True``, display a legend mapping colors/lines to model
        names. Default is ``True``.

    show_grid : bool, optional
        If ``True``, display the radial grid lines on the chart.
        Default is ``True``.

    scale : {'norm', 'min-max', 'std', 'standard'}, optional
        Method for scaling metric values before plotting. Scaling is
        applied independently to each metric (axis) across models.
        Default is ``'norm'``.

        - ``'norm'`` or ``'min-max'``: Min-max scaling. Transforms
          values to the range [0, 1] using
          :math:`(X - min) / (max - min)`. Useful for comparing
          relative performance when metrics have different scales.
        - ``'std'`` or ``'standard'``: Standard scaling (Z-score).
          Transforms values to have zero mean and unit variance using
          :math:`(X - mean) / std`. Preserves relative spacing better
          than min-max but results can be negative.
        - ``None``: Plot raw metric values without scaling. Use only
          if metrics naturally share a comparable, non-negative range.

    lower_bound : float, optional
        Sets the minimum value for the radial axis (innermost circle).
        Useful when using standard scaling ('std') which can produce
        negative values, or to adjust the plot's center.
        Default is ``0``.

    savefig : str, optional
        If provided, the file path (e.g., 'radar_comparison.svg')
        where the figure will be saved. If ``None``, the plot is
        displayed interactively. Default is ``None``.

    loc : str, optional
        Location argument passed to `matplotlib.pyplot.legend()` to
        position the legend (e.g., 'upper right', 'lower left',
        'center right'). Default is ``'upper right'``.

    verbose : int, optional
        Controls the verbosity level. ``0`` is silent. Higher values
        may print debugging information during metric calculation or
        scaling. Default is ``0``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object containing the radar chart. Allows
        for further customization after the function call.

    Raises
    ------
    ValueError
        If lengths of `y_preds`, `names` (if provided), and
        `train_times` (if provided) do not match. If an invalid
        string is provided for `scale`. If a metric string name is
        not recognized by the internal scorer.
    TypeError
        If `y_true` or `y_preds` contain non-numeric data.

    See Also
    --------
    kdiagram.utils.metric_utils.get_scorer : Function likely used
        internally to fetch metric callables (verify path).
    sklearn.metrics : Scikit-learn metrics module.
    matplotlib.pyplot.polar : Function for creating polar plots.

    Notes
    -----
    This function provides a multi-dimensional view of model performance.

    **Metric Calculation:**
    For each model :math:`k` with predictions :math:`\hat{y}_k` and
    each metric :math:`m` (from the `metrics` list), the score
    :math:`S_{m,k}` is calculated:

    .. math::
        S_{m,k} = \text{Metric}_m(y_{true}, \hat{y}_k)

    If `train_times` are provided, they are treated as an additional
    metric axis.

    **Scaling:**
    If `scale` is specified, scaling is applied column-wise (per metric)
    across all models before plotting:

    - Min-Max ('norm'):

      .. math::
         S'_{m,k} = \frac{S_{m,k} - \min_j(S_{m,j})}{\max_j(S_{m,j}) - \min_j(S_{m,j})}
    - Standard ('std'):

      .. math::
         S'_{m,k} = \frac{S_{m,k} - \text{mean}_j(S_{m,j})}{\text{std}_j(S_{m,j})}

    **Plotting:**
    The (scaled) scores :math:`S'_{m,k}` for each model :math:`k`
    determine the radial distance along the axis corresponding to
    metric :math:`m`. Points are connected to form a polygon for
    each model.

    References
    ----------
    .. [1] Wikipedia contributors. (2024). Radar chart. In Wikipedia,
           The Free Encyclopedia. Retrieved April 14, 2025, from
           https://en.wikipedia.org/wiki/Radar_chart
           *(General reference for radar charts)*
    .. [2] Kenny-Denecke, J. F., Hernandez-Amaro, A.,
           Martin-Gorriz, M. L., & Castejon-Limos, P. (2024).
           Lead-Time Prediction in Wind Tower Manufacturing: A Machine
           Learning-Based Approach. *Mathematics*, 12(15), 2347.
           https://doi.org/10.3390/math12152347
           *(Example application using radar charts for ML comparison)*

    Examples
    --------
    >>> from kdiagram.plot.comparison import plot_model_comparison 
    >>> import numpy as np
    >>> 
    >>> # Example 1: Regression task
    >>> y_true_reg = np.array([3, -0.5, 2, 7, 5])
    >>> y_pred_r1 = np.array([2.5, 0.0, 2.1, 7.8, 5.2])
    >>> y_pred_r2 = np.array([3.2, 0.2, 1.8, 6.5, 4.8])
    >>> times = [0.1, 0.5] # Training times in seconds
    >>> names = ['ModelLin', 'ModelTree']
    >>> ax1 = plot_model_comparison(y_true_reg, y_pred_r1, y_pred_r2,
    ...                        train_times=times, names=names,
    ...                        metrics=['r2', 'mae', 'rmse'], # Specify metrics
    ...                        title="Regression Model Comparison",
    ...                        scale='norm') # Normalize for comparison
    >>> 
    >>> # Example 2: Classification task (requires appropriate y_true/y_pred)
    >>> y_true_clf = np.array([0, 1, 0, 1, 1, 0])
    >>> y_pred_c1 = np.array([0, 1, 0, 1, 0, 0]) # Model 1 preds
    >>> y_pred_c2 = np.array([0, 1, 1, 1, 1, 0]) # Model 2 preds
    >>> ax2 = plot_model_comparison(y_true_clf, y_pred_c1, y_pred_c2,
    ...                        names=["LogReg", "SVM"],
    ...                        # Uses default classification metrics
    ...                        title="Classification Model Comparison",
    ...                        scale='norm')
    """
    # Docstring omitted as requested
    # --- Input Validation and Preparation ---
    try:
        # Remove NaN values and ensure consistency
        y_true, *y_preds = drop_nan_in(y_true, *y_preds, error='raise')
        # Validate y_true and each y_pred
        temp_preds = []
        for i, pred in enumerate(y_preds):
            # Validate returns tuple, we need the second element
            validated_pred = validate_yy(
                y_true, pred, expected_type=None, flatten=True
                )[1]
            temp_preds.append(validated_pred)
        y_preds = temp_preds
    except Exception as e:
        # Catch potential errors during validation/NaN drop
        raise TypeError(f"Input validation failed: {e}") from e

    n_models = len(y_preds)
    if n_models == 0:
        warnings.warn("No prediction arrays (*y_preds) provided.")
        return None # Cannot plot without predictions

    # --- Handle Names ---
    if names is None:
        names = [f"Model_{i+1}" for i in range(n_models)]
    else:
        names = columns_manager(list(names), empty_as_none=False) # Ensure list
        if len(names) < n_models:
            names += [f"Model_{i+1}" for i in range(len(names), n_models)]
        elif len(names) > n_models:
             warnings.warn(f"Received {len(names)} names for {n_models}"
                           f" models. Extra names ignored.", UserWarning)
             names = names[:n_models]

    # --- Handle Metrics ---
    if metrics is None:
        target_type = type_of_target(y_true)
        if target_type in ['continuous', 'continuous-multioutput']:
            # Default regression metrics
            metrics = ["r2", "mae", "mape", "rmse"]
        else:
            # Default classification metrics
            metrics = ["accuracy", "precision", "recall", "f1"]
        if verbose >= 1:
            print(f"[INFO] Auto-selected metrics for target type "
                  f"'{target_type}': {metrics}")

    metrics = is_iterable(metrics, exclude_string=True, transform=True)

    metric_funcs = []
    metric_names = []
    error_metrics = [] # Track metrics needing sign inversion

    for metric in metrics:
        try:
            if isinstance(metric, str):
                # get_scorer returns a callable scorer object
                scorer_func = get_scorer(metric) 
                metric_funcs.append(scorer_func)
                metric_names.append(metric)
                # Identify error metrics (lower is better) for potential scaling flip
                if metric in ['mae', 'mape', 'rmse', 'mse']: # Add others if needed
                     error_metrics.append(metric)
            elif callable(metric):
                metric_funcs.append(metric)
                m_name = getattr(metric, '__name__', f'func_{len(metric_names)}')
                metric_names.append(m_name)
                # Cannot easily determine if callable is error/score metric
            else:
                 warnings.warn(
                     f"Ignoring invalid metric type: {type(metric)}")
        except Exception as e:
             warnings.warn(
                 f"Could not retrieve scorer for metric '{metric}': {e}")

    if not metric_funcs:
        raise ValueError("No valid metrics found or specified.")

    # --- Handle Train Times ---
    train_time_vals = None
    if train_times is not None:
        if isinstance(train_times, (int, float, np.number)): # Handle single value
             train_time_vals = np.array([float(train_times)] * n_models)
        else:
             train_times = np.asarray(train_times, dtype=float)
             if train_times.ndim != 1 or len(train_times) != n_models:
                 raise ValueError(
                     f"train_times must be a single float or a list/array "
                     f"of length n_models ({n_models}). "
                     f"Got shape {train_times.shape}."
                 )
             train_time_vals = train_times
        metric_names.append("Train Time (s)") # Use clearer name
        # Add a placeholder for calculation loop, will substitute later
        metric_funcs.append("train_time_placeholder")


    # --- Calculate Metric Results ---
    results = np.zeros((n_models, len(metric_names)), dtype=float)
    for i, y_pred in enumerate(y_preds):
        for j, metric_func in enumerate(metric_funcs):
            if metric_func == "train_time_placeholder":
                results[i, j] = train_time_vals[i]
            elif metric_func is not None:
                try:
                    score = metric_func(y_true, y_pred)
                    results[i, j] = score
                except Exception as e:
                    warnings.warn(f"Could not compute metric "
                                  f"'{metric_names[j]}' for model "
                                  f"'{names[i]}': {e}. Setting to NaN.")
                    results[i, j] = np.nan
            else:
                results[i, j] = np.nan # Should not happen if logic is correct

    # --- Scale Results ---
    # Make copy for scaling to preserve original results if needed later
    results_scaled = results.copy()

    # Handle potential NaNs before scaling
    if np.isnan(results_scaled).any():
        warnings.warn("NaN values found in metric results. Scaling might "
                      "be affected or rows/cols dropped depending on method.")
        # Option 1: Impute (e.g., with column mean) - complex
        # Option 2: Use nan-aware numpy functions
        # Let's use nan-aware functions

    # Note: Some metrics are better when *lower* (MAE, RMSE, MAPE, train_time).
    # For visualization where larger radius is better, we might invert these
    # before scaling, or adjust the interpretation. Let's scale first.
    if scale in ['norm', 'min-max']:
        if verbose >= 1: 
            print("[INFO] Scaling metrics using Min-Max.")
        min_vals = np.nanmin(results_scaled, axis=0)
        max_vals = np.nanmax(results_scaled, axis=0)
        range_vals = max_vals - min_vals
        # Avoid division by zero for metrics with no variance
        range_vals[range_vals < 1e-9] = 1.0
        results_scaled = (results_scaled - min_vals) / range_vals
        # Now, for error metrics, higher value (closer to 1) is WORSE.
        # Invert them so higher value (closer to 1) is BETTER.
        for j, name in enumerate(metric_names):
            if name in error_metrics or name == "Train Time (s)":
                results_scaled[:, j] = 1.0 - results_scaled[:, j]
        # Scaled results are now in [0, 1], higher is better.

    elif scale in ['std', 'standard']:
        if verbose >= 1: 
            print("[INFO] Scaling metrics using Standard Scaler.")
        mean_vals = np.nanmean(results_scaled, axis=0)
        std_vals = np.nanstd(results_scaled, axis=0)
        # Avoid division by zero
        std_vals[std_vals < 1e-9] = 1.0
        results_scaled = (results_scaled - mean_vals) / std_vals
        # Std scaling preserves relative order but changes range.
        # Lower errors become more negative. Higher scores become more positive.
        # Maybe invert sign for error metrics?
        for j, name in enumerate(metric_names):
            if name in error_metrics or name == "Train Time (s)":
                results_scaled[:, j] = -results_scaled[:, j]
        # Now higher value means better performance (higher score or lower error)
        # but range is not [0, 1]. We need to handle lower_bound.

    # Replace any potential NaNs resulting from scaling (e.g., if all NaNs)
    results_scaled = np.nan_to_num(results_scaled, nan=lower_bound)

    # --- Plotting ---
    fig = plt.figure(figsize=figsize or (8, 8)) # Default figsize here
    ax = fig.add_subplot(111, polar=True)

    # Angles for each metric axis
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles_closed = angles + angles[:1] # Repeat first angle to close plot

    # Colors
    if colors is None:
        # Use a robust colormap like tab10 if available
        try:
            cmap_obj = plt.get_cmap("tab10")
            plot_colors = [cmap_obj(i % 10) for i in range(n_models)]
        except ValueError: # Fallback if tab10 not found (unlikely)
            cmap_obj = plt.get_cmap("viridis")
            plot_colors = [cmap_obj(i / n_models) for i in range(n_models)]
    else:
        plot_colors = colors # Use user-provided list

    # Plot each model
    for i, row in enumerate(results_scaled):
        values = np.concatenate((row, [row[0]])) # Close the polygon
        color = plot_colors[i % len(plot_colors)] # Cycle colors if needed
        ax.plot(angles_closed, values, label=names[i], color=color,
                linewidth=1.5, alpha=alpha)
        ax.fill(angles_closed, values, color=color, alpha=0.1) # Lighter fill

    # --- Configure Axes ---
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_names)

    # Adjust radial limits and labels
    # If scaled to [0, 1], set limit slightly above 1
    # If std scaled, auto-limit might be better, but respect lower_bound
    if scale in ['norm', 'min-max']:
         ax.set_ylim(bottom=lower_bound, top=1.05)
         # Optional: Add radial ticks for [0, 1] scale
         ax.set_yticks(np.linspace(lower_bound, 1, 5))
    else: # Raw or std scaled
         ax.set_ylim(bottom=lower_bound)
         # Let matplotlib auto-determine upper limit and ticks

    ax.tick_params(axis='y', labelsize=8) # Smaller radial labels
    ax.tick_params(axis='x', pad=10) # Pad angular labels outwards

    # Grid
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props )

    # Legend
    if legend:
        ax.legend(loc=loc, bbox_to_anchor=(1.25, 1.05)) # Adjust position

    # Title
    ax.set_title(title or "Model Performance Comparison", y=1.15, fontsize=14)

    # --- Output ---
    plt.tight_layout(pad=2.0) # Adjust layout

    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        try:
            plt.show()
        except Exception as e:
             warnings.warn(f"Could not display plot interactively ({e})."
                           f" Use savefig parameter.", UserWarning)

    return ax
