# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

import warnings
from typing import Any, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..decorators import check_non_emptiness
from ..utils.handlers import columns_manager
from ..utils.validator import ensure_2d

__all__=['plot_feature_fingerprint']

@check_non_emptiness (params =["importances"])
def plot_feature_fingerprint(
    importances,
    features: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    normalize: bool = True,
    fill: bool = True,
    cmap: Union[str, List[Any]] = 'tab10', 
    title: str = "Feature Impact Fingerprint",
    figsize: Optional[Tuple[float, float]] = None, 
    show_grid: bool = True,
    savefig: Optional[str] = None
):
    r"""Create a radar chart visualizing feature importance profiles.

    This function generates a polar (radar) chart to visually
    compare the importance or contribution profiles of a set of
    features across different groups, conditions, or time periods
    (e.g., geographical zones, yearly data, different models). Each
    group is represented by a distinct polygon (layer) on the chart,
    making it easy to identify patterns, dominant features, and
    shifts in feature influence across the groups, often referred
    to as a 'fingerprint'.

    It is particularly useful for model interpretation, allowing a
    quick comparison of how feature rankings change under different
    circumstances.

    Parameters
    -------------
    importances : array-like of shape (n_layers, n_features)
        The core data containing feature importance values. Each row
        represents a different layer (e.g., a zone, a year, a model)
        and each column corresponds to a feature. Can be a list of
        lists or a NumPy array.

    features : list of str, optional
        Names of the features corresponding to the columns in
        ``importances``. The order must match the columns. If ``None``,
        generic names like 'feature 1', 'feature 2', etc., will be
        generated.
        Default is ``None``.

    labels : list of str, optional
        Names for each layer (row) in ``importances``. These labels
        will appear in the legend. If ``None``, generic names like
        'Layer 1', 'Layer 2', etc., will be generated.
        Default is ``None``.

    normalize : bool, default=True
        If ``True``, normalize the importance values within each layer
        (row) to a range of [0, 1] by dividing by the maximum
        importance value in that layer. This is useful for comparing
        the *shape* of the importance profiles independent of their
        absolute magnitudes. If ``False``, the raw importance values
        are plotted.

    fill : bool, default=True
        If ``True``, the area enclosed by each layer's polygon on the
        radar chart will be filled with a semi-transparent color,
        enhancing visual distinction between layers. If ``False``, only
        the outlines are plotted.

    cmap : str or list, default='tab10'
        Matplotlib colormap name (e.g., 'viridis', 'plasma', 'tab10')
        or a list of valid color specifications (e.g., ['red',
        '#00FF00', 'blue']) to color the different layers. If a
        colormap name is provided, colors will be sampled from it. If
        a list is provided, it should ideally have at least as many
        colors as there are layers.

    title : str, default="Feature Impact Fingerprint"
        The title displayed above the radar chart.

    figsize : tuple of (float, float), default=(8, 8)
        The width and height of the figure in inches.

    show_grid : bool, default=True
        If ``True``, display the polar grid lines (both radial and
        angular) on the plot, which can aid in reading values. If
        ``False``, the grid is hidden.

    savefig : str, optional
        The file path (including extension, e.g., 'fingerprint.png')
        where the plot should be saved. If ``None``, the plot is
        displayed interactively using ``plt.show()`` instead of being
        saved.
        Default is ``None``.

    Returns
    --------
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object containing the radar chart. This
        can be used for further customization if needed.

    See Also
    ---------
    matplotlib.pyplot.polar : Underlying function for polar plots.
    numpy.linspace : Used for calculating angles.

    Notes
    ------
    - The function uses helper utilities like `ensure_2d` and
      `columns_manager` (assumed available) for input validation
      and preprocessing.
    - To create closed polygons, the function appends the first data
      point and the first angle to the end of their respective lists
      before plotting each layer.
    - Normalization (`normalize=True`) scales each layer independently:
      :math:`r'_{ij} = r_{ij} / \max_{j}(r_{ij})`, where :math:`r_{ij}`
      is the importance of feature :math:`j` for layer :math:`i`. This
      can highlight relative importance patterns but obscures absolute
      magnitude differences between layers.
    - The angular positions of features are evenly spaced around the
      circle: :math:`\theta_j = 2 \pi j / N` for :math:`j=0, ..., N-1`,
      where :math:`N` is the number of features.

    Let :math:`\mathbf{R}` be the input `importances` matrix of shape
    :math:`(M, N)`, where :math:`M` is the number of layers (labels)
    and :math:`N` is the number of features.

    1. **Angle Calculation**: Angles for each feature axis are
       calculated as:
           
       .. math::
           \theta_j = \frac{2 \pi j}{N}, \quad j = 0, 1, \dots, N-1

    2. **Normalization** (if `normalize=True`): Each row
       :math:`\mathbf{r}_i = (r_{i0}, r_{i1}, \dots, r_{i,N-1})` is
       normalized:

       .. math::
           r'_{ij} = \frac{r_{ij}}{\max_{k}(r_{ik})}
       If :math:`\max_{k}(r_{ik}) = 0`, :math:`r'_{ij}` is set to 0.
       Let :math:`\mathbf{R}'` be the matrix of normalized values.

    3. **Plotting**: For each layer :math:`i`, the function plots points
       in polar coordinates :math:`(r'_{ij}, \theta_j)` (or
       :math:`(r_{ij}, \theta_j)` if not normalized). To close the shape,
       the first point :math:`(r'_{i0}, \theta_0)` is repeated at angle
       :math:`2\pi`. The points are connected by lines, and optionally,
       the enclosed area is filled.

    Examples
    ---------
    >>> import numpy as np
    >>> from kdiagram.plot.feature_based import plot_feature_fingerprint

    **1. Random Example:**

    >>> np.random.seed(42) # for reproducibility
    >>> random_importances = np.random.rand(3, 6) # 3 layers, 6 features
    >>> feature_names = [f'Feature {i+1}' for i in range(6)]
    >>> layer_labels = ['Model A', 'Model B', 'Model C']
    >>> ax = plot_feature_fingerprint(
    ...     importances=random_importances,
    ...     features=feature_names,
    ...     labels=layer_labels,
    ...     title="Random Feature Importance Comparison",
    ...     cmap='Set3',
    ...     normalize=True,
    ...     fill=True
    ... )
    >>> # plt.show() is called internally if savefig is None

    **2. Concrete Example (Yearly Weights):**

    >>> features = ['rainfall', 'GWL', 'seismic', 'density', 'geo']
    >>> weights_per_year = [
    ...     [0.2, 0.4, 0.1, 0.6, 0.3],  # 2023
    ...     [0.3, 0.5, 0.2, 0.4, 0.4],  # 2024
    ...     [0.1, 0.6, 0.2, 0.5, 0.3],  # 2025
    ... ]
    >>> years = ['2023', '2024', '2025']
    >>> ax_yearly = plot_feature_fingerprint(
    ...     importances=weights_per_year,
    ...     features=features,
    ...     labels=years,
    ...     title="Feature Influence Over Years",
    ...     cmap='tab10',
    ...     normalize=False # Show raw weights
    ... )
    >>> # plt.show() is called internally

    """
    # --- Input Validation and Preparation ---
    # Ensure importances is a 2D NumPy array
    importance_matrix = ensure_2d(importances)

    n_layers, n_features_data = importance_matrix.shape

    # Manage feature names
    if features is None:
        # Generate default feature names if none provided
        features_list = [f'feature {i+1}'
                         for i in range(n_features_data)]
    else:
        # Ensure features is a list and handle potential discrepancies
        features_list = columns_manager(features, empty_as_none=False)

    # If user provided fewer feature names than data columns, append
    # generic names
    if len(features_list) < n_features_data:
        features_list.extend(
            [f'feature {ix + 1}'
             for ix in range(len(features_list), n_features_data)]
        )
    # Truncate if user provided more names than needed (optional,
    # could also raise error)
    elif len(features_list) > n_features_data:
         warnings.warn(
            f"More feature names ({len(features_list)}) provided "
            f"than data columns ({n_features_data}). "
            "Extra names ignored."
            )
         features_list = features_list[:n_features_data]

    n_features = len(features_list) # Final number of features used

    # Manage labels
    if labels is None:
        # Generate default layer labels if none provided
        labels_list = [f"Layer {idx+1}" for idx in range(n_layers)]
    else:
        labels_list = list(labels) # Ensure it's a list
        # Check label count consistency
        if len(labels_list) < n_layers:
            warnings.warn(
                f"Fewer labels ({len(labels_list)}) provided than "
                f"layers ({n_layers}). Using generic names for the rest."
                )
            labels_list.extend(
                [f'Layer {ix + 1}'
                 for ix in range(len(labels_list), n_layers)]
            )
        elif len(labels_list) > n_layers:
            warnings.warn(
                f"More labels ({len(labels_list)}) provided than "
                f"layers ({n_layers}). Extra labels ignored."
                )
            labels_list = labels_list[:n_layers]


    # --- Normalization (if requested) ---
    if normalize:
        # Calculate max per row (layer), keep dimensions for broadcasting
        # max_per_row shape: (n_layers, 1), e.g., (3, 1)
        importance_matrix = importance_matrix.values if isinstance (
            importance_matrix, pd.DataFrame) else importance_matrix 
        
        max_per_row = importance_matrix.max(axis=1, keepdims=True)

        # Create a mask for rows with max_val > 0 (where normalization is safe)
        # valid_max_mask shape: (n_layers, 1), e.g., (3, 1)
        valid_max_mask = max_per_row > 1e-9

        # Initialize normalized matrix
        normalized_matrix = np.zeros_like(importance_matrix, dtype=float)

        # --- FIX START ---
        # Get boolean index for valid rows, shape (n_layers,) e.g., (3,)
        valid_rows_indices = valid_max_mask[:, 0]

        # Proceed only if there are any rows to normalize
        if np.any(valid_rows_indices):
            # Select the rows from the original matrix that need normalization
            # Shape: (n_valid_rows, n_features), e.g., (3, 6)
            rows_to_normalize = importance_matrix[valid_rows_indices]

            # Select the corresponding max values for these rows
            # Since max_per_row is (n_layers, 1) and valid_rows_indices is (n_layers,),
            # this indexing correctly results in shape (n_valid_rows, 1), e.g., (3, 1)
            max_values_for_valid_rows = max_per_row[valid_rows_indices]

            # Perform the division using broadcasting: (MxN / Mx1 works)
            normalized_rows = rows_to_normalize / max_values_for_valid_rows

            # Place the normalized rows back into the result matrix
            normalized_matrix[valid_rows_indices] = normalized_rows
        # --- FIX END ---

        # Rows where max_val <= 0 remain zero (already initialized)
        # Update importance_matrix with normalized values
        importance_matrix = normalized_matrix

    # --- Angle Calculation for Radar Axes ---
    # Calculate evenly spaced angles for each feature axis
    angles = np.linspace(0, 2 * np.pi, n_features,
                         endpoint=False).tolist()
    # Add the first angle to the end to close the loop for plotting
    angles_closed = angles + angles[:1]

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw=dict(polar=True))

    # Get colors from specified colormap or list
    try:
        cmap_obj = cm.get_cmap(cmap)
        # Sample colors if it's a standard Matplotlib cmap
        colors = [cmap_obj(i / n_layers) for i in range(n_layers)]
    except ValueError: # Handle case where cmap might be a list of colors
        if isinstance(cmap, list):
            colors = cmap
            if len(colors) < n_layers:
                 warnings.warn(
                    f"Provided color list has fewer colors "
                    f"({len(colors)}) than layers ({n_layers}). "
                    f"Colors will repeat."
                )
        else: # Fallback if cmap is invalid string or list
            warnings.warn(
                f"Invalid cmap '{cmap}'. Falling back to 'tab10'.")
            cmap_obj = cm.get_cmap('tab10')
            colors = [cmap_obj(i / n_layers) for i in range(n_layers)]

    # --- Plot Each Layer ---
    for idx, row in enumerate(importance_matrix):
        # Get the importance values for the current layer
        values = row.tolist()
        # Add the first value to the end to close the loop
        values_closed = values + values[:1]

        # Determine the label for the legend
        label = labels_list[idx]
        # Determine the color, cycling if necessary
        color = colors[idx % len(colors)]

        # Plot the outline
        ax.plot(angles_closed, values_closed, label=label,
                color=color, linewidth=2)

        # Fill the area if requested
        if fill:
            ax.fill(angles_closed, values_closed, color=color,
                    alpha=0.25)

    # --- Customize Plot Appearance ---
    ax.set_title(title, size=16, y=1.1) # Adjust title position

    # Set feature labels on the angular axes
    ax.set_xticks(angles)
    ax.set_xticklabels(features_list, fontsize=11)

    # Hide radial tick labels (often preferred for normalized data)
    ax.set_yticklabels([])
    # Set radial limits (optional, e.g., enforce 0 start)
    ax.set_ylim(bottom=0)
    if normalize:
        # Optionally add a single radial label for the max value (1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                           fontsize=9, color='gray')

    # Show grid lines if requested
    if show_grid:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    else:
        ax.grid(False)

    # Add legend, positioned outside the plot area
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              fontsize=10)

    # Adjust layout to prevent labels/title overlapping
    plt.tight_layout(pad=2.0)

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
