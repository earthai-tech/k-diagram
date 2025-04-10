# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
#   License: Apache 2.0
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd 

from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Optional,
)
from ..api.summary import ResultSummary
from ..utils.validator import (
    _assert_all_types,
    exist_features,
    )
from ..utils.diagnose_q import (
    detect_quantiles_in,
    validate_qcols,
    build_qcols_multiple
)
from ..utils.handlers import columns_manager
from ..utils.plot import set_axis_grid
from ..decorators import isdf, check_non_emptiness

__all__=[
     'plot_actual_vs_predicted',
     'plot_anomaly_magnitude',
     'plot_coverage_diagnostic',
     'plot_interval_consistency',
     'plot_interval_width',
     'plot_model_drift',
     'plot_temporal_uncertainty',
     'plot_uncertainty_drift',
     'plot_velocity', 
     'plot_coverage', 
    ]

def plot_coverage(
    y_true,
    *y_preds,
    names=None,
    q=None,
    kind='line',
    cmap='viridis',
    pie_startangle=140,
    pie_autopct='%1.1f%%',
    radar_color='tab:blue',
    radar_fill_alpha=0.25,
    radar_line_style='o-',
    cov_fill=False, 
    figsize=None,
    title=None,
    savefig=None,
    verbose=1 
):
    """
    Plot coverage scores for quantile or point forecasts and allow
    multiple visualization styles (line, bar, pie, and radar).

    This function computes and visualizes the fraction of times
    the true values :math:`y_i` lie within predicted quantile
    intervals or match point forecasts, for one or more models.
    If multiple prediction arrays are passed (e.g. from different
    models), this function compares their coverage on the same
    figure through different plot types.

    .. math::
        \\text{coverage} = \\frac{1}{N}\\sum_{i=1}^{N}
        1\\{\\hat{y}_{i}^{(\\ell)} \\leq y_i
        \\leq \\hat{y}_{i}^{(u)}\\}

    where :math:`\\hat{y}_{i}^{(\\ell)}` is the lower quantile
    prediction for the :math:`i`th sample and :math:`\\hat{y}_{i}^{
    (u)}` is the upper quantile prediction. The indicator function
    :math:`1\\{\\cdot\\}` counts how many times the true value
    :math:`y_i` lies within or on the boundaries of the predicted
    interval.

    
    Examples
    --------
    >>> import numpy as np
    >>> from kdiagram.plot.uncertainty import plot_coverage
    >>> # True values
    >>> y_true = np.random.rand(100)
    >>> y_pred = np.random.rand(100, 3)
    >>> # 3-quantile predictions for a single model
    >>> y_pred_q = np.random.rand(100, 3)
    >>> q = [0.1, 0.5, 0.9]
    >>> # Bar chart coverage
    >>> plot_coverage(y_true, y_pred_q, q=q,
    ...               names=['QuantModel'],
    ...               kind='bar',
    ...               title='Coverage (Bar)')
    # Single model quantile coverage
    >>> y_pred = np.random.rand(200, 3)
    >>> plot_coverage(y_true, y_pred, q=[0.1, 0.5, 0.9],
    ...               kind='radar', names=['QModel'],
    ...               cov_fill=True, cmap='plasma')
    >>> # Multiple models with radar plot
    >>> y_pred_q2 = np.random.rand(100, 3)
    >>> plot_coverage(y_true, y_pred_q, y_pred_q2,
    ...               q=q,
    ...               names=['Model1','Model2'],
    ...               kind='radar',
    ...               title='Coverage (Radar)')
    """

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
            for i in range(extra):
                names.append(f"Model_{len(names) + 1}")

    coverage_scores = []

    q= columns_manager(q)
    # Handle quantiles
    if q is not None:
        q = np.array(q)
        if q.ndim != 1:
            raise ValueError(
                "Parameter 'q' must be a 1D list or"
                " array of quantile levels."
                )
            
        if not np.all((0 < q) & (q < 1)):
            raise ValueError(
                "Quantile levels must be between 0 and 1."
            )
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
    for i, pred in enumerate(y_preds):
        pred = np.array(pred)

        #if (q is not None) and (pred.ndim == 2):
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
            in_interval = (
                (y_true >= lower_q) & (y_true <= upper_q)
            ).astype(int)
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
    valid_cov = [
        c if c is not None else 0 for c in coverage_scores
    ]
    x_idx = np.arange(num_models)
    
    if kind in {'bar', 'line', 'pipe'}: 
        # Initialize the figure.
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
    # Plot according to the chosen 'kind'.
    if kind == 'bar':
        plt.bar(x_idx, valid_cov, color='blue', alpha=0.7)
        for idx, val in enumerate(coverage_scores):
            if val is not None:
                plt.text(
                    x=idx,
                    y=val + 0.01,
                    s=f"{val:.2f}",
                    ha='center',
                    va='bottom'
                )
        plt.xticks(x_idx, names)
        plt.ylim([0, 1])
        plt.ylabel("Coverage")
        plt.xlabel("Models")

    elif kind == 'line':
        plt.plot(x_idx, valid_cov, marker='o')
        for idx, val in enumerate(coverage_scores):
            if val is not None:
                plt.text(
                    x=idx,
                    y=val + 0.01,
                    s=f"{val:.2f}",
                    ha='center',
                    va='bottom'
                )
        plt.xticks(x_idx, names)
        plt.ylim([0, 1])
        plt.ylabel("Coverage")
        plt.xlabel("Models")

    elif kind == 'pie':
        # Pie chart: each slice represents a model's coverage. By default,
        # the slice size is coverage[i] out of the sum of coverage.
        total_cov = sum(valid_cov)
        if total_cov == 0:
            # Avoid a zero-coverage pie chart.
            plt.text(
                0.5, 0.5,
                "No coverage to plot",
                ha='center',
                va='center'
            )
        else:
            plt.pie(
                valid_cov,
                labels=names,
                autopct=pie_autopct,
                startangle=pie_startangle,
                colors=plt.cm.get_cmap(cmap)(
                    np.linspace(0, 1, num_models)
                )
            )
            plt.axis('equal')  # Make the pie chart a perfect circle.

    elif kind == 'radar':
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
            label='Coverage'
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
                ax.pcolormesh(
                    Theta, R, R, 
                    cmap=cmap, 
                    shading='auto', 
                    alpha=radar_fill_alpha,
                    zorder=0  # Place behind main plot
                )
                # Add red circle at coverage value
                ax.plot(
                    theta, 
                    [coverage_value] * len(theta),  # Constant radius
                    color='red', 
                    linewidth=2, 
                    linestyle='-',
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
                    alpha=0.7
                )
            
            else:
                # Multiple models: transparent fill between center and line
                ax.fill(
                    angles,
                    coverage_radar,
                    color=radar_color,
                    alpha=radar_fill_alpha,
                    zorder=0
                )
        # Final formatting
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels=names)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right')

    else:
        # Fallback: print coverage scores to the console for each model.
        for idx, val in enumerate(coverage_scores):
            print(f"{names[idx]} coverage: {val}")

    if verbose:
       cov_dict = {
           names[idx]: cov 
           for idx, cov in enumerate(coverage_scores)
           }
       
       summary = ResultSummary(
           "CoverageScores").add_results (cov_dict)
       print(summary)
       
    # Add title if provided.
    if title is not None:
        plt.title(title)
        
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')

    plt.show()

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
    """Visualise forecast drift across prediction horizons.

    This utility renders a polar bar chart to depict how model
    reliability *evolves* as the forecast horizon increases.
    The radial coordinate encodes the *average* predictive
    uncertainty  computed as the mean inter-quantile width
    :math:`w_i = \mathbb{E}[q_{0.90} - q_{0.10}]`  while the
    angular coordinate corresponds to successive horizons
    (e.g. *2023 ? 2026*).

    The function is particularly helpful for identifying
    *concept drift* or *model aging* [1]_. A steep radial growth
    signals that uncertainty (or any supplied error metric)
    inflates with lead-time, suggesting the need for model
    retraining or data augmentation.

   
    Examples
    --------
    >>> from kdiagram.plot.uncertainty import plot_model_drift
    >>> ax = plot_model_drift(
    ...     df=zhongshan_pred_2023_2026,
    ...     q10_cols=[
    ...         'subsidence_2023_q10', 'subsidence_2024_q10',
    ...         'subsidence_2025_q10', 'subsidence_2026_q10'],
    ...     q90_cols=[
    ...         'subsidence_2023_q90', 'subsidence_2024_q90',
    ...         'subsidence_2025_q90', 'subsidence_2026_q90'],
    ...     horizons=[2023, 2024, 2025, 2026],
    ...     acov='quarter_circle',
    ...     title='Forecast Horizon Drift  Zhongshan')

    >>> # Random demo with synthetic data
    >>> rng = np.random.default_rng(seed=42)
    >>> years = np.arange(1, 9)
    >>> synth = pd.DataFrame({
    ...     f'q10_{y}': rng.normal(loc=0, scale=1, size=100)
    ...     for y in years})
    >>> synth.update({
    ...     f'q90_{y}': synth[f'q10_{y}'] +
    ...     rng.uniform(0.5, 1.5, size=100)})
    >>> plot_model_drift(synth,
    ...                 q10_cols=[f'q10_{y}' for y in years],
    ...                 q90_cols=[f'q90_{y}' for y in years])

    See Also
    --------
    gofast.plot.utils.build_qcols_multiple : Helper to pair
        quantile columns.
    gofast.plot.utils.set_axis_grid : Convenience wrapper for
        grid styling.

    References
    ----------
    .. [1] Gama, J., liobaite, I., Bifet, A., Pechenizkiy, M.,
       & Bouchachia, A. (2014). *A survey on concept drift
       adaptation*. ACM Computing Surveys (CSUR), 46(4), 1-37.
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
    widths = np.array([
        (df[q90] - df[q10]).mean() for q10, q90 in q_cols
    ])

    # Secondary colouring metric 
    if color_metric_cols is not None:
        colour_vals = np.array([df[col].mean() for col in
                                color_metric_cols])
    else:
        colour_vals = widths

    # 
    # 3. Angular span selection 
    # 
    span = {
        'default': 2 * np.pi,
        'half_circle': np.pi,
        'quarter_circle': np.pi / 2,
        'eighth_circle': np.pi / 4,
    }.get(acov, 2 * np.pi)

    theta = np.linspace(0.0, span, n_horizons, endpoint=False)

    # Scale radii when angular coverage < full circle 
    radii = widths / widths.max() if span < 2 * np.pi else widths

    # 
    # 4. Figure setup 
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': 'polar'},
    )

    # Orient polar chart: 0° at the top, clockwise direction 
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(span))

    # Colormap normalisation 
    norm = Normalize(vmin=colour_vals.min(), vmax=colour_vals.max())
    colours = cm.get_cmap(cmap)(norm(colour_vals))

    #
    # 5. Draw bars 
    bar_width = (span / n_horizons) * 0.9  # slight gap between bars
    ax.bar(theta, radii,
           width=bar_width,
           color=colours,
           edgecolor='k',
           alpha=0.85,
           linewidth=0.8)

    # Annotation 
    if annotate:
        for ang, rad, raw in zip(theta, radii, widths):
            label = f"{raw:.2f}"
            ax.text(ang, rad + 0.03 * radii.max(), label,
                    ha='center', va='bottom', fontsize=9)

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
    q50_cols: List[str],
    theta_col: Optional[str] = None,
    cmap: str = 'viridis',
    acov: str = 'default',
    normalize: bool = True,
    use_abs_color: bool = True,
    figsize: Tuple[float, float] = (9, 9),
    title: Optional[str] = None,
    s: Union[float, int] = 30,
    alpha: float = 0.85,
    show_grid: bool = True,
    savefig: Optional[str] = None,
    cbar: bool = True,
    mask_angle: bool = False,
):
    """Polar plot visualizing average velocity across locations.

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
    over the considered time periods.

    This visualization is useful for identifying spatial patterns in
    the dynamics of a phenomenon, such as locating areas of rapid or
    slow change (high/low velocity) in land subsidence predictions.
    Coloring by magnitude helps to contextualize the velocity (e.g.,
    is high velocity occurring in areas of already high subsidence?).


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
                UserWarning
            )
        else:
             warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but the current"
                f" implementation uses the DataFrame index for angular "
                f"positioning ('theta'). The column '{theta_col}' is "
                f"currently ignored for positioning.",
                UserWarning
            )

    # --- Data Processing ---
    # Extract Q50 data into a NumPy array (locations x time)
    q50_array = df[q50_cols].values # Shape (N, M)

    # Compute yearly differences along the time axis (axis=1)
    # Result shape (N, M-1)
    yearly_diff = np.diff(q50_array, axis=1)

    # Compute average velocity per location (mean across time diffs)
    # Result shape (N,)
    r = np.mean(yearly_diff, axis=1)

    # Normalize radial values (velocity) if requested
    r_normalized = r.copy() # Use a copy for potential normalization
    if normalize:
        r_range = np.ptp(r) # Peak-to-peak (max - min)
        if r_range > 1e-9: # Avoid division by zero or near-zero
            r_min = r.min()
            r_normalized = (r - r_min) / r_range
        else:
            # Handle case where all velocities are the same
            r_normalized = np.zeros_like(r) # Set all to 0 if range is zero
            warnings.warn(
                "Velocity range is zero or near-zero. Normalized radial "
                "values ('r') are set to 0.", UserWarning
                )

    # Determine values used for coloring the points
    if use_abs_color:
        # Use average absolute Q50 magnitude across all years
        color_vals = np.mean(np.abs(q50_array), axis=1)
        cbar_label = "Average Abs Q50 Magnitude"
    else:
        # Use the calculated average velocity for color
        color_vals = r # Use original velocity for color scale
        cbar_label = "Average Velocity"

    # --- Angular Coordinate Calculation ---
    N = len(df) # Number of locations/points
    # Generate linear space from 0 to 1 for N points
    theta_normalized = np.linspace(0, 1, N, endpoint=True) # Includes endpoint 1

    # Map normalized theta to the desired angular coverage
    angular_range_map = {
        'default': 2 * np.pi,
        'half_circle': np.pi,
        'quarter_circle': np.pi / 2,
        'eighth_circle': np.pi / 4
    }
    # Get the angular span in radians, default to full circle if invalid
    angle_span = angular_range_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in angular_range_map:
        warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning
        )

    # Calculate final theta values
    theta = theta_normalized * angle_span

    # --- Color Normalization for Plotting ---
    try:
        cmap_used = plt.get_cmap(cmap)
    except ValueError:
         warnings.warn(
            f"Invalid `cmap` name '{cmap}'. Falling back to 'viridis'.",
            UserWarning
        )
         cmap = 'viridis'
         cmap_used = plt.get_cmap(cmap)

    # Normalize color values to the range [0, 1] for the colormap
    color_norm = Normalize(vmin=np.min(color_vals),
                           vmax=np.max(color_vals))
    # Map normalized color values to actual colors using the colormap
    colors = cmap_used(color_norm(color_vals))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': 'polar'})

    # Set the angular limits based on angular coverage
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(angle_span)) # set_thetamax expects degrees

    # Create the polar scatter plot
    ax.scatter(
        theta,
        r_normalized if normalize else r, # Use normalized or raw r
        c=colors,            # Point colors
        s=s,                 # Point size
        edgecolor='k',       # Point edge color (optional, for visibility)
        linewidth=0.5,       # Point edge width (optional)
        alpha=alpha          # Point transparency
    )

    # Set plot title
    ax.set_title(title or "Average Velocity Polar Plot",
                 fontsize=14, y=1.08) # Adjust title position

    # Add color bar if requested
    if cbar:
        # Create a ScalarMappable for the colorbar
        sm = cm.ScalarMappable(norm=color_norm, cmap=cmap_used)
        sm.set_array([]) # Necessary for ScalarMappable

        # Add the colorbar to the figure
        cbar_obj = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
        cbar_obj.set_label(cbar_label, fontsize=10)

    # Customize grid and labels
    if show_grid:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)

    # Optionally mask angular tick labels
    if mask_angle:
        ax.set_xticklabels([])

    # Set radial label based on normalization
    if normalize:
        ax.set_ylabel("Normalized Average Velocity",
                      labelpad=15, fontsize=10)
        # Ensure radial limits are appropriate for normalized data
        # ax.set_ylim(bottom=0, top=1.05) # Give slight padding
        # ax.set_yticks(np.linspace(0, 1, 5)) # Example radial ticks
    else:
        ax.set_ylabel("Average Velocity", labelpad=15, fontsize=10)
        # Radial limits might need auto-scaling or manual setting

    plt.tight_layout() # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax

@check_non_emptiness 
@isdf 
def plot_interval_consistency(
    df: pd.DataFrame,
    qlow_cols: List[str],
    qup_cols: List[str],
    q50_cols: Optional[List[str]] = None,
    theta_col: Optional[str] = None,
    use_cv: bool = True,
    cmap: str = 'coolwarm',
    acov: str = 'default',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 9),
    s: Union[float, int] = 30,
    alpha: float = 0.85,
    show_grid: bool = True,
    mask_angle: bool = False,
    savefig: Optional[str] = None
):
    """Polar plot showing consistency of prediction interval widths.

    This function generates a polar scatter plot to visualize the
    temporal consistency (or variability) of prediction interval
    widths (e.g., Q90 - Q10) across different locations over multiple
    time steps or forecast horizons.

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

    
    References
    ----------
    .. [1] Matplotlib documentation: https://matplotlib.org/

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
    >>> # plt.show() called internally

    """
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
                UserWarning
            )
        else:
            # Issue warning as current implementation uses index
             warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but the current"
                f" implementation uses the DataFrame index for angular "
                f"positioning ('theta'). The column '{theta_col}' is "
                f"currently ignored for positioning.",
                UserWarning
            )

    # --- Data Calculation ---
    # Calculate interval widths for each year/timepoint for all locations
    # widths shape: (M, N) where M=num_time_steps, N=num_locations
    try:
        widths = np.array(
            [df[qup].values - df[qlo].values
             for qlo, qup in zip(qlow_cols, qup_cols)]
        )
    except Exception as e:
        raise TypeError(
            f"Could not compute widths. Ensure quantile columns contain "
            f"numeric data. Original error: {e}"
        )

    # Calculate radial value 'r' (std dev or CV of widths over time)
    # Result shape: (N,)
    mean_widths = np.mean(widths, axis=0)
    std_widths = np.std(widths, axis=0)

    if use_cv:
        # Calculate Coefficient of Variation (CV)
        # Handle division by zero or near-zero mean width
        # Use np.divide for safe division, setting result to 0 where mean is ~0
        r = np.divide(std_widths, mean_widths,
                      out=np.zeros_like(mean_widths, dtype=float), # Output array
                      where=np.abs(mean_widths) > 1e-9) # Condition for division
        # Optionally issue warning if division by zero occurred
        if np.any(np.abs(mean_widths) <= 1e-9):
             num_zeros = np.sum(np.abs(mean_widths) <= 1e-9)
             warnings.warn(
                f"Mean interval width was zero or near-zero for {num_zeros}"
                f" locations. CV is set to 0 for these locations.",
                RuntimeWarning
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
                UserWarning
            )
             # Fallback: color by the radial value itself
             color_vals_source = r
             cbar_label = radial_label # Label reflects 'r'
    else:
        # Fallback: color by the radial value itself if no q50_cols given
        color_vals_source = r
        cbar_label = radial_label # Label reflects 'r'

    # --- Angular Coordinate Calculation ---
    N = len(df) # Number of locations
    # Generate linear space [0, 1] for N points
    theta_normalized = np.linspace(0, 1, N, endpoint=True)

    # Map normalized theta to the desired angular coverage
    angular_range_map = {
        'default': 2 * np.pi,
        'half_circle': np.pi,
        'quarter_circle': np.pi / 2,
        'eighth_circle': np.pi / 4
    }
    angle_span = angular_range_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in angular_range_map:
         warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning
        )
    # Calculate final theta values
    theta = theta_normalized * angle_span

    # --- Color Normalization ---
    try:
        cmap_used = plt.get_cmap(cmap)
    except ValueError:
         warnings.warn(
            f"Invalid `cmap` name '{cmap}'. Falling back to 'coolwarm'.",
            UserWarning
        )
         cmap = 'coolwarm' # Ensure cmap is valid for fallback
         cmap_used = plt.get_cmap(cmap)

    # Normalize color values for the colormap
    color_norm = Normalize(vmin=np.min(color_vals_source),
                           vmax=np.max(color_vals_source))
    # Get actual colors
    plot_colors = cmap_used(color_norm(color_vals_source))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': 'polar'})

    # Set angular limits
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(angle_span))

    # Create the polar scatter plot
    ax.scatter(
        theta,
        r,                   # Radial value (CV or Std Dev)
        c=plot_colors,       # Point colors
        s=s,                 # Point size
        edgecolor='k',       # Point edge color
        linewidth=0.5,       # Point edge width
        alpha=alpha          # Point transparency
    )

    # Set plot title
    ax.set_title(
        title or "Prediction Interval Consistency",
        fontsize=14, y=1.08
    )

    # Add color bar
    sm = cm.ScalarMappable(norm=color_norm, cmap=cmap_used)
    sm.set_array([]) # Necessary for ScalarMappable
    cbar_obj = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar_obj.set_label(cbar_label, fontsize=10)

    # Customize grid and labels
    if show_grid:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)

    # Optionally mask angular tick labels
    if mask_angle:
        ax.set_xticklabels([])

    # Set radial axis label
    ax.set_ylabel(radial_label, labelpad=15, fontsize=10)
    # Optional: adjust radial limits if needed, e.g., start at 0
    ax.set_ylim(bottom=0)

    plt.tight_layout() # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax

@check_non_emptiness 
@isdf
def plot_anomaly_magnitude(
    df: pd.DataFrame,
    actual_col: str,
    q_cols: Union[List[str], Tuple[str, str]],
    theta_col: Optional[str] = None,
    acov: str = 'default',
    title: str = "Anomaly Magnitude Polar Plot",
    figsize: Tuple[float, float] = (8.0, 8.0),
    cmap_under: str = 'Blues',
    cmap_over: str = 'Reds',
    s: int = 30,
    alpha: float = 0.8,
    show_grid: bool = True,
    verbose: int = 1,
    cbar: bool = False,
    savefig: Optional[str] = None,
    mask_angle: bool = False,
):
    """Visualize magnitude and type of prediction anomalies polar plot.

    This function generates a polar scatter plot designed to highlight
    prediction anomalies  instances where the actual ground truth value
    falls outside a specified prediction interval (defined by a lower
    and an upper quantile, e.g., Q10 and Q90). It visually maps the
    location, magnitude, and type of these anomalies.

    - **Angular Position (`theta`)**: Represents each data point
      (location). If `theta_col` is provided and valid, points are
      ordered angularly based on the values in that column (e.g.,
      latitude, longitude, station index). Otherwise, points are
      plotted in their original DataFrame order. The angles are mapped
      linearly onto the specified angular coverage (`acov`).
    - **Radial Distance (`r`)**: Represents the *magnitude* of the
      anomaly for points falling outside the prediction interval. It's
      calculated as the absolute difference between the actual value and
      the nearest violated interval bound (:math:`|y_{actual} - y_{bound}|`).
      Points *within* the interval are not plotted.
    - **Color**: Distinguishes the *type* of anomaly and indicates its
      magnitude. Separate colormaps are used:
        - `cmap_under` (default: Blues) for under-predictions
          (:math:`y_{actual} < y_{lower\_bound}`).
        - `cmap_over` (default: Reds) for over-predictions
          (:math:`y_{actual} > y_{upper\_bound}`).
      The color intensity within each map corresponds to the anomaly
      magnitude `r`, based on a shared normalization scale.

    This plot serves as a powerful diagnostic tool for evaluating
    prediction models, especially those providing uncertainty estimates.
    It helps to:
    - Identify specific locations or regions where the model
      significantly misestimates outcomes (under or over).
    - Assess the severity (magnitude) of these prediction errors.
    - Guide post-hoc analysis, model calibration checks, or targeted
      field validation efforts.

    
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

    """
    # --- Input Validation ---
    # Decorators handle basic df checks
    # Validate quantile columns using helper function
    try:
        qlow_col, qup_col = validate_qcols(
            q_cols=q_cols,
            ncols_exp='==2', # Expect exactly two columns
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
        col for col in [actual_col, qlow_col, qup_col]
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
        warnings.warn("DataFrame is empty after dropping NaN values"
                      " in essential columns. Cannot generate plot.", UserWarning)
        return None # Cannot proceed

    # --- Anomaly Calculation ---
    # Extract data as numpy arrays for efficiency
    try:
        y    = data[actual_col].to_numpy(dtype=float)
        y_lo = data[qlow_col].to_numpy(dtype=float)
        y_hi = data[qup_col].to_numpy(dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Failed to convert essential columns to numeric arrays."
            f" Check data types. Original error: {e}"
        ) from e

    # Identify under- and over-predictions
    under_mask = y < y_lo
    over_mask  = y > y_hi

    # Calculate anomaly magnitude (distance from the violated bound)
    anomaly_mag = np.zeros_like(y, dtype=float)
    # Magnitude is positive: bound - actual for under, actual - bound for over
    anomaly_mag[under_mask] = y_lo[under_mask] - y[under_mask]
    anomaly_mag[over_mask]  = y[over_mask]  - y_hi[over_mask]

    # Filter out non-anomalies for plotting (only plot r > 0)
    is_anomaly = (under_mask | over_mask)
    if not np.any(is_anomaly):
         warnings.warn(
            "No anomalies detected (all actual values are within "
            "the specified quantile bounds). Plot will be empty.", UserWarning
            )
         # Still create plot structure, but it will be empty
    # Filter data to only include anomalies
    anomaly_mag = anomaly_mag[is_anomaly]
    under_mask_filtered = under_mask[is_anomaly]
    over_mask_filtered = over_mask[is_anomaly]
    data_filtered = data[is_anomaly] # Filter DataFrame rows too
    N_anomalies = len(data_filtered) # Number of anomalies

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
                f"Using default DataFrame order.", UserWarning
            )
             ordered_idx = np.arange(N_anomalies) # Fallback to original order
    else:
        # Use default order if theta_col not specified or not found
        if theta_col and theta_col not in data_filtered.columns:
              warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found in the"
                f" (filtered) DataFrame. Using default DataFrame order.",
                 UserWarning
            )
        ordered_idx = np.arange(N_anomalies)

    # Generate base theta values (linear spacing)
    theta_norm = np.linspace(0.0, 1.0, N_anomalies, endpoint=True)

    # Define angular coverage range
    coverage_map = {
        'default':        2 * np.pi,
        'half_circle':    np.pi,
        'quarter_circle': np.pi / 2,
        'eighth_circle':  np.pi / 4,
    }
    coverage = coverage_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in coverage_map:
         warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning
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
        figsize=figsize,
        subplot_kw={'projection': 'polar'}
    )
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(coverage)) # Expects degrees
    ax.set_title(title, fontsize=14, y=1.08) # Adjust position

    # Setup grid
    if show_grid:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)

    # Optionally mask angle labels
    if mask_angle:
        ax.set_xticklabels([])

    # --- Color Normalization (shared scale based on max magnitude) ---
    # Ensure vmax is at least a small positive number for normalization
    vmax = max(float(anomaly_mag_ordered.max()) if N_anomalies > 0 else 0.0, 1e-5)
    norm = Normalize(vmin=0.0, vmax=vmax)

    # Retrieve colormaps safely
    try:
        cmap_under_obj = plt.get_cmap(cmap_under)
    except ValueError:
        warnings.warn(f"Invalid `cmap_under` ('{cmap_under}'). "
                      f"Using default 'Blues'.", UserWarning)
        cmap_under_obj = plt.get_cmap('Blues')
    try:
        cmap_over_obj = plt.get_cmap(cmap_over)
    except ValueError:
         warnings.warn(f"Invalid `cmap_over` ('{cmap_over}'). "
                       f"Using default 'Reds'.", UserWarning)
         cmap_over_obj = plt.get_cmap('Reds')


    # --- Scatter Plot Anomalies (separate calls for color/label) ---
    if np.any(under_mask_ordered):
        ax.scatter(
            theta[under_mask_ordered],           # Angles for under-preds
            anomaly_mag_ordered[under_mask_ordered], # Magnitudes for under-preds
            c=anomaly_mag_ordered[under_mask_ordered], # Color value is magnitude
            cmap=cmap_under_obj,                 # Blues colormap
            norm=norm,                           # Shared normalization
            s=s,                                 # Marker size
            alpha=alpha,                         # Transparency
            edgecolor='grey',                    # Edge color
            linewidth=0.5,
            label="Under-prediction",            # Legend label
        )

    if np.any(over_mask_ordered):
        ax.scatter(
            theta[over_mask_ordered],            # Angles for over-preds
            anomaly_mag_ordered[over_mask_ordered],  # Magnitudes for over-preds
            c=anomaly_mag_ordered[over_mask_ordered],  # Color value is magnitude
            cmap=cmap_over_obj,                  # Reds colormap
            norm=norm,                           # Shared normalization
            s=s,                                 # Marker size
            alpha=alpha,                         # Transparency
            edgecolor='grey',                    # Edge color
            linewidth=0.5,
            label="Over-prediction",             # Legend label
        )

    # Add legend if any anomalies were plotted
    if np.any(under_mask_ordered) or np.any(over_mask_ordered):
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
    else:
        # Add text if plot is empty
         ax.text(0, 0, "No anomalies detected",
                 horizontalalignment='center', verticalalignment='center')


    # --- Add Colorbar (optional) ---
    # Note: Visually uses cmap_over, but scale (norm) is correct
    if cbar and N_anomalies > 0:
        # Create a mappable object linked to the normalization and cmap_over
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_over_obj)
        sm.set_array([]) # Needed for ScalarMappable
        cbar_obj = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
        cbar_obj.set_label("Anomaly magnitude |Actual - Bound|", fontsize=10)

    # Set radial axis label
    ax.set_ylabel("Anomaly Magnitude", labelpad=15, fontsize=10)
    ax.set_ylim(bottom=0) # Ensure radius starts at 0

    # --- Logging ---
    if verbose > 0:
        # Use original masks on NaN-dropped data before filtering for plot
        n_total_checked = len(data) # Total valid points checked
        n_under_total = np.sum(y < y_lo)
        n_over_total = np.sum(y > y_hi)
        n_anomalies_total = n_under_total + n_over_total
        print("-" * 50)
        print("Anomaly Detection Summary:")
        print(f"  Total valid points checked: {n_total_checked}")
        print(f"  Anomalies detected (outside {qlow_col}-{qup_col}): "
              f"{n_anomalies_total}")
        print(f"  ? Under-predictions ({actual_col} < {qlow_col}): {n_under_total}")
        print(f"  ? Over-predictions  ({actual_col} > {qup_col}): {n_over_total}")
        print("-" * 50)


    # --- Output ---
    plt.tight_layout()
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax

@check_non_emptiness 
@isdf
def plot_uncertainty_drift(
    df: pd.DataFrame,
    qlow_cols: List[str],
    qup_cols: List[str],
    dt_labels: Optional[List[str]] = None,
    theta_col: Optional[str] = None,
    acov: str = 'default',
    base_radius: float = 0.15,
    band_height: float = 0.15,
    cmap: str = 'tab10',
    label: str = 'Year',
    alpha: float = 0.85,
    figsize: Tuple[float, float] = (9, 9),
    title: Optional[str] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    mask_degree: bool = True,
    savefig: Optional[str] = None
):
    """Polar plot visualizing temporal drift of uncertainty width.

    This function creates a polar line plot showing how the width of
    the prediction interval (e.g., Q90 - Q10), representing model
    uncertainty, evolves over multiple time steps (e.g., years) across
    different locations. Each time step is depicted as a distinct
    concentric ring.

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
    >>>
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
    ...     mute_degree=False      # Show angle labels
    ... )
    >>> # plt.show() called internally

    **2. Concrete Example (Subsidence Data - adapted from docstring):**

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
        time_labels = list(dt_labels) # Ensure list type

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
                f"Using index for angular position.", UserWarning
            )
        else:
            warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but currently "
                f"ignored for positioning. Using index for angular position.",
                UserWarning
            )

    # Prepare data: Drop rows with NaNs in relevant columns
    data = df[all_cols].dropna()
    if len(data) == 0:
        warnings.warn(
            "DataFrame is empty after dropping NaN values in quantile "
            "columns. Cannot generate plot.", UserWarning
            )
        return None
    N = len(data) # Number of valid data points (locations)

    # --- Calculate Interval Widths and Normalize Globally ---
    widths = []
    try:
        for ql, qu in zip(qlow_cols, qup_cols):
            width_values = (data[qu] - data[ql]).to_numpy(dtype=float)
            if np.any(width_values < 0):
                 warnings.warn(
                    f"Negative interval widths detected for columns "
                    f"'{qu}' and '{ql}'. Check if upper < lower bound.",
                    UserWarning
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
    if not widths: # Should not happen due to earlier checks, but safe
        return None
    # Calculate max, handle case where all widths might be zero or negative
    all_width_values = np.concatenate(widths) if widths else np.array([0])
    max_width = np.max(all_width_values) if len(all_width_values) > 0 else 0.0

    # Normalize widths using the global maximum
    normalized_widths = []
    if max_width > 1e-9: # Avoid division by zero/near-zero
        normalized_widths = [w / max_width for w in widths]
    else:
        # If max width is zero, all normalized widths are zero
        normalized_widths = [np.zeros_like(w) for w in widths]
        warnings.warn(
            "Maximum interval width across all data is zero or near-zero. "
            "Normalized widths are all set to 0.", UserWarning
        )

    # --- Angular Coordinate Calculation ---
    acov_map = { # Map coverage name to (min_angle, max_angle) in radians
        'default':        (0, 2 * np.pi),
        'half_circle':    (0, np.pi),
        'quarter_circle': (0, np.pi / 2),
        'eighth_circle':  (0, np.pi / 4)
    }
    theta_min_rad, theta_max_rad = acov_map.get(acov.lower(), (0, 2 * np.pi))
    if acov.lower() not in acov_map:
        warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (0 to 2*pi).",
            UserWarning
        )
    angular_range_rad = theta_max_rad - theta_min_rad

    # Generate theta values based on index, mapped to the specified range
    theta = (np.linspace(0., 1., N, endpoint=True) # Linear space [0, 1]
             * angular_range_rad                   # Scale to range width
             + theta_min_rad)                      # Add start angle offset

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': 'polar'})
    ax.set_thetamin(np.degrees(theta_min_rad)) # Expects degrees
    ax.set_thetamax(np.degrees(theta_max_rad)) # Expects degrees

    # Hide angular tick labels if requested
    if mask_degree:
        ax.set_xticklabels([])
    # Configure grid
    if show_grid:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)
    # Hide radial ticks as they primarily separate years visually
    ax.set_yticks([])

    # Get color palette
    try:
        cmap_obj = cm.get_cmap(cmap)
        # Sample colors - handle discrete vs continuous cmaps
        if hasattr(cmap_obj, 'colors'): # Discrete colormap
            color_palette = cmap_obj.colors
        else: # Continuous colormap
            color_palette = cmap_obj(np.linspace(0, 1, num_time_steps))
    except ValueError:
        warnings.warn(f"Invalid `cmap` name '{cmap}'. Falling back to 'tab10'.")
        cmap_obj = cm.get_cmap('tab10') # Fallback cmap
        color_palette = cmap_obj.colors

    # --- Draw Rings for Each Time Step ---
    for i, (w_norm, step_label) in enumerate(
            zip(normalized_widths, time_labels)):
        
        #XXX TODO 
        # Calculate base radius for this ring (increases for later years)
        # base_r = base_radius + i * some_increment # Alternative logic
        base_r = base_radius * (i + 1) # Base offset increases multiplicatively

        # Calculate final radius: base + scaled normalized width
        # Ensure w_norm has the same length as theta (N)
        if len(w_norm) != N:
             # This should not happen if dropna was done correctly, but safeguard
             raise InternalError(
                 "Mismatch between width data and theta length.") # Or handle gracefully

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
            label=step_label,    # Label for the legend
            color=color,         # Color for this ring
            linewidth=1.8,       # Line thickness
            alpha=alpha          # Transparency
        )

    # --- Final Touches ---
    # Set plot title
    ax.set_title(
        title or "Multi-time Uncertainty Drift (Interval Width)",
        fontsize=14, y=1.08
    )

    # Add legend if requested
    if show_legend:
        # Use the provided 'label' parameter as the intended legend title
        # The example hardcodes 'Year'. Let's compromise:
            # use 'label' if given, else 'Time Step'.
        legend_title = label if label else "Time Step"
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.25, 1.1), # Position outside plot
            title=legend_title,        # Use dynamic title
            fontsize=9
        )

    plt.tight_layout() # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax

@check_non_emptiness
@isdf 
def plot_actual_vs_predicted(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    theta_col: Optional[str] = None,
    acov: str = 'default',
    figsize: Tuple[float, float] = (8.0, 8.0),
    title: Optional[str] = None,
    line: bool = True,
    r_label: Optional[str] = None,
    cmap: Optional[str] = None, # Note: Currently unused
    alpha: float = 0.3,
    actual_props: Optional[Dict[str, Any]] = None,
    pred_props: Optional[Dict[str, Any]] = None,
    show_grid: bool = True,
    grid_props: Optional[dict] =None, 
    show_legend: bool = True,
    mask_angle: bool = False,
    savefig: Optional[str] = None
):
    """Polar plot comparing actual observed vs. predicted values.

    This function generates a polar plot to visually compare actual
    ground truth values against model predictions (typically a central
    estimate like the median, Q50) for multiple data points or
    locations arranged circularly.

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
          lines or individual dots based on the `line` parameter.
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
    # --- Input Validation and Preparation ---
    # Basic checks handled by decorators
    # Check existence of primary columns
    exist_features(
        df,
        features=[actual_col, pred_col],
        error='raise', # Raise error if missing
        name='Actual and Predicted columns'
        )

    # Consolidate columns needed, check theta_col existence only if specified
    cols_to_select = [actual_col, pred_col]
    if theta_col:
        if theta_col not in df.columns:
            warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found. "
                f"Using index for angular position.", UserWarning
                )
            # Proceed without theta_col for ordering
        else:
             warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but currently "
                f"ignored for positioning/ordering. Using index.",
                UserWarning
            )
            # Although ignored, keep it for potential future use if needed?
            # For now, just select it if present, even if unused later.
            # cols_to_select.append(theta_col) # Decided against adding if unused

    # Drop rows with NaNs in essential columns
    data = df[cols_to_select].dropna().copy()
    if len(data) == 0:
        warnings.warn("DataFrame is empty after dropping NaN values in actual"
                      " and predicted columns. Cannot generate plot.", 
                      UserWarning)
        return None
    N = len(data)

    # --- Angular Coordinate Calculation ---
    acov_map = { # Map name to angular range in radians
        'default':        2 * np.pi,
        'half_circle':    np.pi,
        'quarter_circle': np.pi / 2,
        'eighth_circle':  np.pi / 4
    }
    angular_range = acov_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in acov_map:
         warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning
        )
    # Calculate theta based on index, mapped to the angular range
    # Use endpoint=False if using lines to avoid overlap at 2pi?
    # If using dots, endpoint=True or False matters less visually.
    # Let's use endpoint=False for lines, True for dots for potentially
    # better spacing.
    use_endpoint = not line
    theta = np.linspace(0., angular_range, N, endpoint=use_endpoint)

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
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': 'polar'})
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(angular_range)) # Expects degrees
    
    set_axis_grid(
        ax, show_grid=show_grid, 
        grid_props =grid_props
    )

    if mask_angle:
        ax.set_xticklabels([])

    # --- Plot Difference Lines ---
    # Warning: This loop can be very slow for large N
    # Consider alternatives like fill_between if performance is critical
    # and data can be meaningfully sorted by theta.
    if N > 5000: # Add warning for potentially slow loop
         warnings.warn(
             f"Plotting difference lines for {N} points individually."
             f" This may be slow. Consider using `line=False` or sampling data.",
             PerformanceWarning
         )
    for t, a, p in zip(theta, actual, pred):
        # Plot a vertical line segment at angle t between actual and pred
        ax.plot(
            [t, t],                  # Start and end angle (same)
            [min(a, p), max(a, p)],  # Start and end radius
            color='gray',            # Hardcoded color for difference
            alpha=alpha,             # Use specified transparency
            linewidth=1              # Fixed linewidth for diff lines
            )

    # --- Plot Actual and Predicted Data (Lines or Dots) ---
    # Define default properties, merge with user-provided props
    default_actual_props_line = {
        'color': 'black', 'linewidth': 1.5, 'label':'Actual'}
    default_pred_props_line = {
        'color': 'red', 'linewidth': 1.5, 'label':'Predicted (Q50)'
        }
    default_actual_props_scatter = {
        'color': 'black', 's': 20, 'label':'Actual'
        }
    default_pred_props_scatter = {
        'color': 'red', 's': 20, 'alpha': alpha, 
        'label':'Predicted (Q50)'}

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
        theta_plot = np.append(theta, theta[0]) if np.isclose(
            angular_range, 2*np.pi) else theta
        actual_plot = np.append(actual, actual[0]) if np.isclose(
            angular_range, 2*np.pi) else actual
        pred_plot = np.append(pred, pred[0]) if np.isclose(
            angular_range, 2*np.pi) else pred

        ax.plot(theta_plot, actual_plot, **current_actual_props)
        ax.plot(theta_plot, pred_plot, **current_pred_props)
    else:
        # Merge user props with defaults for scatter plot
        current_actual_props = {**default_actual_props_scatter, **actual_props}
        current_pred_props = {**default_pred_props_scatter, **pred_props}

        ax.scatter(theta, actual, **current_actual_props)
        ax.scatter(theta, pred, **current_pred_props)

    # --- Final Touches ---
    ax.set_title(title or "Actual vs Predicted Polar Plot",
                 fontsize=14, y=1.08)
    if r_label:
        # Use set_ylabel for radial axis label, adjust padding
        ax.set_ylabel(r_label, labelpad=15, fontsize=10)

    # Add legend if requested and labels were provided in props
    if show_legend:
        # Check if labels exist before showing legend
        handles, labels = ax.get_legend_handles_labels()
        if labels: # Only show legend if there's something to label
            ax.legend(loc='upper right', 
                      bbox_to_anchor=(1.25, 1.1), fontsize=9)
        else:
            warnings.warn(
                "Legend requested but no labels found for plot elements.",
                UserWarning)

    plt.tight_layout() # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
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
    q_cols: Union[List[str], Tuple[str, str]],
    theta_col: Optional[str] = None,
    z_col: Optional[str] = None,
    acov: str = 'default',
    figsize: Tuple[float, float] = (8.0, 8.0),
    title: Optional[str] = None,
    cmap: str = 'viridis', 
    s: int = 30,
    alpha: float = 0.8,
    show_grid: bool = True,
    grid_props: Optional [dict]=None, 
    cbar: bool = True,
    mask_angle: bool = True,
    savefig: Optional[str] = None
):
    """Polar scatter plot visualizing prediction interval width.

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
      representing the interval width (`r`) itself.

    This plot helps to:
    - Identify locations with high or low prediction uncertainty.
    - Visualize the spatial distribution or sample distribution of
      uncertainty magnitude.
    - Explore potential correlations between uncertainty width and other
      variables (like the central prediction) when using `z_col`.

    
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
                f"It will be ignored.", UserWarning
            )
        else:
            # Add to list for potential future use or NaN check if desired
            # cols_needed.append(theta_col) # Currently ignored, don't add
             warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but currently "
                f"ignored for positioning/ordering. Using index.",
                UserWarning
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
        warnings.warn("DataFrame is empty after dropping NaN values."
                      " Cannot generate plot.", UserWarning)
        return None
    N = len(data) # Number of valid data points

    # --- Calculate Radial Coordinate (Interval Width) ---
    try:
        r = data[qup_col].to_numpy(dtype=float) - data[qlow_col].to_numpy(
            dtype=float)
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
            UserWarning
        )
        # Option: clamp negative radius to 0?
        # r = np.maximum(r, 0)

    # --- Calculate Color Coordinate (Z-value) ---
    if z_col:
        try:
            z = data[z_col].to_numpy(dtype=float)
            cbar_label = z_col # Label colorbar with column name
        except Exception as e:
             raise TypeError(
                f"Could not use `z_col` ('{z_col}') for color. Ensure it "
                f"contains numeric data. Original error: {e}"
            ) from e
    else:
        # Default: color by interval width `r`
        z = r
        cbar_label = "Interval Width" # Label colorbar appropriately

    # --- Angular Coordinate Calculation ---
    acov_map = { # Map name to angular range in radians
        'default':        2 * np.pi,
        'half_circle':    np.pi,
        'quarter_circle': np.pi / 2,
        'eighth_circle':  np.pi / 4
    }
    angular_range = acov_map.get(acov.lower(), 2 * np.pi)
    if acov.lower() not in acov_map:
         warnings.warn(
            f"Invalid `acov` value '{acov}'. Using 'default' (2*pi).",
            UserWarning
        )
    # Calculate theta based on index, mapped to the angular range
    # Using endpoint=False might give slightly better visual spacing for scatter
    theta = np.linspace(0., angular_range, N, endpoint=False)

    # --- Color Normalization ---
    # Check if z has variance for normalization
    if np.ptp(z) > 1e-9: # Check peak-to-peak range
         norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    else:
         # Handle constant z case: map all to middle color
         norm = Normalize(vmin=np.min(z) - 0.5, vmax=np.max(z) + 0.5)
         warnings.warn(
             f"Color values ('{cbar_label}') have zero range."
             f" All points will have the same color.", UserWarning
             )
    try:
        cmap_ref = cm.get_cmap(cmap)
        colors = cmap_ref(norm(z))
    except ValueError:
        warnings.warn(f"Invalid `cmap` name '{cmap}'. Falling back to 'viridis'.")
        cmap = 'viridis' # Ensure cmap is valid for fallback
        cmap_ref = cm.get_cmap(cmap)
        colors = cmap_ref(norm(z))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': 'polar'})
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(angular_range)) # Expects degrees

    # Apply grid and angle label settings
    set_axis_grid(ax, show_grid, grid_props =grid_props)
    if mask_angle:
        ax.set_xticklabels([])

    # Create the scatter plot
    ax.scatter(
        theta,
        r,                   # Radius is interval width
        c=colors,            # Point color based on z
        s=s,                 # Point size
        alpha=alpha,         # Point transparency
        edgecolor='k',       # Point edge color (optional)
        linewidth=0.5        # Point edge width (optional)
        # cmap=cmap_ref is implicitly used via 'colors', no need to pass here
    )

    # Set plot title
    ax.set_title(
        title or f"Prediction Interval Width ({qup_col} - {qlow_col})",
        fontsize=14, y=1.08
    )

    # Add color bar if requested
    if cbar:
        # Create a ScalarMappable for the colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_ref)
        sm.set_array([]) # Necessary for ScalarMappable

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

    plt.tight_layout() # Adjust layout

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax

@check_non_emptiness 
@isdf
def plot_coverage_diagnostic(
    df: pd.DataFrame,
    actual_col: str,
    q_cols: Union[List[str], Tuple[str, str]],
    theta_col: Optional[str] = None,
    acov: str = 'default',
    figsize: Tuple[float, float] = (8.0, 8.0),
    title: Optional[str] = None,
    show_grid: bool = True,
    grid_props: Optional[dict]=None, 
    cmap: str = 'RdYlGn',
    alpha: float = 0.85,
    s: int = 35,
    as_bars: bool = False,
    coverage_line_color: str = 'r',
    buffer_pts: int = 500,
    fill_gradient: bool = True,
    gradient_size: int = 300,
    gradient_cmap: str = 'Greens',
    gradient_levels: Optional[List[float]] = None,
    gradient_props: Optional[Dict[str, Any]] = None,
    mask_angle: bool = True,
    savefig: Optional[str] = None,
    verbose: int = 0,
):
    """Diagnose prediction interval coverage using a polar plot.

    This function generates a polar plot to visually assess whether
    actual observed values fall within their corresponding prediction
    intervals (defined by a lower and upper quantile). It helps diagnose
    the calibration of uncertainty estimates.

    - **Angular Position (`theta`)**: Represents each data point or
      location, ordered by DataFrame index and mapped linearly onto the
      specified angular coverage (`acov`). `theta_col` is currently ignored.
    - **Radial Position (`r`)**: Binary indicator of coverage. Points are
      plotted at radius 1 if the actual value is within the interval
      (:math:`Q_{lower} \le y_{actual} \le Q_{upper}`), and at radius 0
      otherwise.
    - **Color (Points/Bars)**: Indicates coverage status using `cmap`
      (default 'RdYlGn'), typically green for covered (1) and red for
      uncovered (0).
    - **Reference Lines**: Concentric dashed lines can be drawn at
      specified `gradient_levels` (e.g., 0.2, 0.4, ...) for reference.
    - **Average Coverage Line**: A prominent solid line is drawn at a
      radius equal to the overall coverage rate (proportion of points
      covered), providing a benchmark against the expected coverage level
      (e.g., for a 90% interval [Q5-Q95], the line should ideally be near 0.9).
    - **Background Gradient (Optional)**: A radial gradient fills the
      background from the center up to the average coverage rate, using
      `gradient_cmap`. This visually emphasizes the overall coverage level.

    This plot is essential for evaluating if the model's uncertainty
    quantification is reliable (i.e., if a 90% prediction interval truly
    covers about 90% of the actual outcomes).

    

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
    >>> # plt.show() called internally

    """
    # --- Input Validation ---
    # Basic checks by decorators
    # Validate q_cols format
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
                f"Using index order.", UserWarning
            )
            # Don't add to cols_needed if missing
        else:
             warnings.warn(
                f"`theta_col` ('{theta_col}') is provided but currently "
                f"ignored for positioning/ordering. Using index order.",
                UserWarning
            )
            # Add only if present, for potential NaN check (though unused)
            # cols_needed.append(theta_col) # Decided against adding if unused

    missing_essential = [
        col for col in [actual_col, qlow_col, qup_col] if col not in df.columns
        ]
    if missing_essential:
         raise ValueError(
             f"Essential columns missing: {', '.join(missing_essential)}"
             )

    # Drop rows with NaNs in essential columns
    data = df[cols_needed].dropna().copy()
    if len(data) == 0:
        warnings.warn("DataFrame is empty after dropping NaN values."
                      " Cannot generate plot.", UserWarning)
        return None
    N = len(data) # Number of valid points

    # --- Calculate Coverage ---
    try:
        y    = data[actual_col].to_numpy(dtype=float)
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
    acov_map = { # Map name to (min_angle, max_angle) in radians
        'default':        (0, 2 * np.pi),
        'half_circle':    (0, np.pi),
        'quarter_circle': (0, np.pi / 2),
        'eighth_circle':  (0, np.pi / 4)
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
    theta = (np.linspace(0., 1., N, endpoint=False) # Use endpoint=False for bars/scatter
             * angular_range_rad
             + theta_min_rad)

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': 'polar'})
    ax.set_thetamin(np.degrees(theta_min_rad))
    ax.set_thetamax(np.degrees(theta_max_rad))
    # Set radial limits strictly to [0, 1] for coverage plot
    ax.set_ylim(0, 1.05) # Slight padding at top
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]) # Set explicit radial ticks

    # Apply grid and angle label settings
    set_axis_grid(ax, show_grid, grid_props)
    if mask_angle:
        ax.set_xticklabels([])

    # --- Optional Background Gradient ---
    if fill_gradient:
        try:
            grad_cmap_obj = cm.get_cmap(gradient_cmap)
            # Create radial and angular meshgrid
            # R goes from 0 up to the total_coverage value
            R, T = np.meshgrid(
                np.linspace(0, total_coverage, gradient_size),
                np.linspace(theta_min_rad, theta_max_rad, gradient_size)
            )
            # Z represents the value to map color to (e.g., radius itself)
            # Tile radius values across angles for pcolormesh
            Z = np.tile(
                np.linspace(0, total_coverage, gradient_size)[:, np.newaxis],
                (1, gradient_size)
                )
            # Normalize Z based on [0, 1] range for colormap
            norm_gradient = Normalize(vmin=0, vmax=1.0)
            # Plot the gradient mesh
            ax.pcolormesh(
                T, R, Z, shading='auto', cmap=grad_cmap_obj,
                alpha=0.20, # Make gradient subtle
                norm=norm_gradient # Ensure consistent color mapping
            )
        except ValueError:
             warnings.warn(f"Invalid `gradient_cmap` ('{gradient_cmap}')."
                           f" Skipping background gradient.", UserWarning)
        except Exception as e:
            warnings.warn(f"Failed to plot background gradient: {e}", UserWarning)


    # --- Optional Concentric Reference Lines ---
    # Define default levels if None
    gradient_levels = gradient_levels if gradient_levels is not None else [
        0.2, 0.4, 0.6, 0.8, 1.0
        ]
    # Filter levels to be within [0, 1]
    valid_levels = [lv for lv in gradient_levels if 0 <= lv <= 1]
    # Define default properties, merge with user props
    default_gradient_props = {
        'linestyle': ':', 'color': 'gray', 'linewidth': 0.8, 'alpha': 0.8
    }
    current_gradient_props = {
        **default_gradient_props,
        **(gradient_props or {})
        }

    # Plot each reference line and its label
    theta_line = np.linspace(theta_min_rad, theta_max_rad, buffer_pts)
    for lv in valid_levels:
        ax.plot(theta_line, [lv] * buffer_pts, **current_gradient_props)
        # Add text label near the end of the line
        ax.text(
            theta_max_rad * 0.98, # Position slightly inside max angle
            lv,
            f"{lv:.1f}", # Format label
            color=current_gradient_props.get('color', 'gray'),
            fontsize=8,
            ha='right', # Horizontal alignment
            va='center' # Vertical alignment
        )

    # --- Average Coverage Line ---
    ax.plot(
        theta_line,
        [total_coverage] * buffer_pts,
        color=coverage_line_color,
        linewidth=2.0, # Make it prominent
        label=f'Avg Coverage ({total_coverage:.2f})' # Add rate to label
    )

    # --- Plot Individual Coverage (Bars or Scatter) ---
    # Radius 'r' is the 'covered' array (0 or 1)
    r_plot = covered.astype(float)
    # Setup colormap and normalization for points/bars
    try:
         cmap_main_obj = cm.get_cmap(cmap)
    except ValueError:
         warnings.warn(f"Invalid `cmap` ('{cmap}'). Using 'RdYlGn'.", UserWarning)
         cmap_main_obj = cm.get_cmap('RdYlGn')
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
             bar_widths = np.abs(bar_widths) * 0.9 # Make slightly narrower than gap
        elif N == 1:
             bar_widths = angular_range_rad * 0.8 # Default width for single bar
        else: # N=0
            bar_widths = 0.1 # Placeholder

        ax.bar(
            theta,          # Angle for each bar
            r_plot,         # Height (0 or 1)
            width=bar_widths, # Width of bars
            bottom=0.0,     # Bars start from center
            color=point_colors,
            alpha=alpha,
            edgecolor='gray',
            linewidth=0.5
        )
    else: # Plot as scatter points
        ax.scatter(
            theta,
            r_plot,         # Radius (0 or 1)
            c=point_colors, # Color based on coverage
            s=s,
            alpha=alpha,
            edgecolor='gray',
            linewidth=0.5
            # No label needed here, color indicates status
        )

    # --- Final Touches ---
    ax.set_title(
        title or "Prediction Interval Coverage Diagnostic",
        fontsize=14, y=1.08
    )
    # Add legend only for the average coverage line
    handles, labels = ax.get_legend_handles_labels()
    if handles: # If the coverage line was plotted and labeled
       ax.legend(handles=handles, labels=labels,
                 loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)

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
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax

@check_non_emptiness 
@isdf
def plot_temporal_uncertainty(
    df: pd.DataFrame,
    q_cols: Union[str, List[str]] = 'auto',
    theta_col: Optional[str] = None,
    names: Optional[List[str]] = None,
    acov: str = 'default',
    figsize: Tuple[float, float] = (8.0, 8.0),
    title: Optional[str] = None,
    cmap: str = 'tab10',
    normalize: bool = True,
    show_grid: bool = True,
    grid_props: Optional[dict]=None, 
    alpha: float = 0.7,
    s: int = 25,
    dot_style: str = 'o',
    legend_loc: str = 'upper right',
    mask_label: bool=False, 
    mask_angle: bool = True, 
    savefig: Optional[str] = None
):
    """Visualize multiple data series using polar scatter plots.

    This function creates a general-purpose polar scatter plot to
    visualize and compare one or more data series (columns) from a
    DataFrame in a circular layout. Each series is plotted with a
    distinct color.

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
      unique color based on the specified `cmap`.

    This plot is flexible and can be used for various purposes, such as:
    - Comparing different model predictions for the same target.
    - Visualizing different quantile predictions (e.g., Q10, Q50, Q90)
      for a single time step to show uncertainty spread.
    - Plotting related variables against each other in a polar context.

    
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
    if isinstance(q_cols, str) and q_cols.lower() == 'auto':
        try:
            # Assume detect_quantiles_in returns list of column names
            detected_cols = detect_quantiles_in(df)
            if not detected_cols:
                 raise ValueError("Auto-detection found no quantile columns.")
            # Check if detected columns actually exist (redundant?)
            exist_features(
                df, features=detected_cols,
                error='raise', name='Auto-detected quantile columns'
            )
            q_cols_list = detected_cols
        except NameError: # If detect_quantiles_in is not defined/imported
             raise ImportError(
                "Helper function 'detect_quantiles_in' is needed for "
                "`q_cols='auto'` but seems unavailable."
                )
        except Exception as e:
            # Catch errors from detect_quantiles_in or exist_features
            raise ValueError(
                f"Automatic detection or validation of `q_cols` failed: {e}. "
                "Please provide `q_cols` explicitly as a list of column names."
            ) from e
    else:
        # Process explicit list using columns_manager if available
        if 'columns_manager' in globals():
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
            df, features=q_cols_list,
            error='raise', name='Specified plot columns (`q_cols`)'
            )

    # Determine columns needed for NaN handling
    cols_for_na_check = list(q_cols_list) # Start with plot columns
    # theta_col_valid = False
    if theta_col:
        if theta_col in df.columns:
            cols_for_na_check.append(theta_col)
            # theta_col_valid = True
            warnings.warn(
                f"`theta_col` ('{theta_col}') is currently used for NaN "
                f"alignment but ignored for positioning/ordering. Using index order.",
                UserWarning
            )
        else:
             warnings.warn(
                f"Specified `theta_col` ('{theta_col}') not found. "
                f"It will be ignored for NaN alignment and positioning.", UserWarning
            )

    # Drop rows with NaNs in ANY of the selected columns
    data = df[cols_for_na_check].dropna().copy()
    if len(data) == 0:
        warnings.warn(
            "DataFrame is empty after dropping NaN values. Cannot generate plot.",
             UserWarning
             )
        return None
    N = len(data) # Number of valid data points

    # --- Angular Coordinate Calculation ---
    angular_range_map = { # Map name to (min_angle, max_angle) in radians
        'default':        (0, 2 * np.pi),
        'half_circle':    (0, np.pi),
        'quarter_circle': (0, np.pi / 2),
        'eighth_circle':  (0, np.pi / 4)
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
    theta = (np.linspace(0., 1., N, endpoint=False)
             * angular_range_rad
             + theta_min_rad)

    # --- Prepare Radial Values (Normalization) ---
    # Internal helper for min-max scaling
    def _normalize(x):
        """Min-max scales array x to [0, 1], handling zero range."""
        x = np.asarray(x, dtype=float) # Ensure float array
        range_x = np.ptp(x) # Peak-to-peak range
        if range_x > 1e-9: # Check if range is non-negligible
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
        if isinstance(q_cols, str) and q_cols.lower() == 'auto':
             # Use the detected names if possible
             plot_labels = [c.replace('_', ' ').title() for c in q_cols_list]
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
        cmap_obj = plt.get_cmap(cmap)
        # Sample colors - handle discrete vs continuous cmaps
        if hasattr(cmap_obj, 'colors') and len(cmap_obj.colors) >= num_series:
             # Use colors directly from discrete map if enough available
             color_palette = cmap_obj.colors
        else: # Sample from continuous map or discrete map with fewer colors
            color_palette = cmap_obj(np.linspace(0, 1, num_series))
    except ValueError:
         warnings.warn(f"Invalid `cmap` name '{cmap}'. Falling back to 'tab10'.")
         cmap_obj = cm.get_cmap('tab10') # Fallback cmap
         color_palette = cmap_obj(np.linspace(0, 1, num_series))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': 'polar'})

    # Set angular limits
    ax.set_thetamin(np.degrees(theta_min_rad))
    ax.set_thetamax(np.degrees(theta_max_rad))

    # Plot each data series
    for i, (vals, label) in enumerate(zip(values_list, plot_labels)):
        if len(vals) != N: # Should not happen if NaN handling is correct
            raise InternalError(
                "Mismatch between data length and theta length."
                )
        color = color_palette[i % len(color_palette)] # Cycle colors if needed
        ax.scatter(
            theta,
            vals,
            label=label,
            alpha=alpha,
            s=s,
            marker=dot_style,
            color=color,
            edgecolor='k', # Add edge color for visibility
            linewidth=0.5
        )

    # --- Final Touches ---
    ax.set_title(
        title or "Polar Data Series Plot", fontsize=14,
        y=1.08)
    set_axis_grid (ax, show_grid, grid_props )
    
    if mask_angle:
        ax.set_xticklabels([]) # Hide angular tick labels
    if mask_label: 
       ax.set_yticklabels([]) # Hide radial tick labels
    else: 
        ax.set_ylabel (f"{'Normalized ' if normalize else ''}Q_pred values")
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if labels: # Only show legend if labels were generated/provided
        ax.legend(handles=handles, labels=labels, loc=legend_loc,
                  bbox_to_anchor=(1.25, 1.0) if 'right' in legend_loc else None,
                  fontsize=9)

    # Set radial axis label if normalization is off? Maybe not useful.
    if not normalize and len(q_cols_list)==1: # Only label if one series, unnormalized
        ax.set_ylabel(q_cols_list[0])

    plt.tight_layout()

    # --- Save or Show ---
    if savefig:
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    return ax

class PerformanceWarning(Warning):
    pass

# Define InternalError for consistency if needed
class InternalError(Exception):
    pass