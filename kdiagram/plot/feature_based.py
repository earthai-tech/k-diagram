# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

import warnings
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..compat.matplotlib import get_cmap
from ..decorators import check_non_emptiness
from ..utils.handlers import columns_manager
from ..utils.validator import ensure_2d

__all__ = ["plot_feature_fingerprint"]


@check_non_emptiness(params=["importances"])
def plot_feature_fingerprint(
    importances,
    features: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    normalize: bool = True,
    fill: bool = True,
    cmap: Union[str, list[Any]] = "tab10",
    title: str = "Feature Impact Fingerprint",
    figsize: Optional[tuple[float, float]] = None,
    show_grid: bool = True,
    savefig: Optional[str] = None,
):
    # --- Input Validation and Preparation ---
    # Ensure importances is a 2D NumPy array
    importance_matrix = ensure_2d(importances)

    n_layers, n_features_data = importance_matrix.shape

    # Manage feature names
    if features is None:
        # Generate default feature names if none provided
        features_list = [f"feature {i+1}" for i in range(n_features_data)]
    else:
        # Ensure features is a list and handle potential discrepancies
        features_list = columns_manager(features, empty_as_none=False)

    # If user provided fewer feature names than data columns, append
    # generic names
    if len(features_list) < n_features_data:
        features_list.extend(
            [
                f"feature {ix + 1}"
                for ix in range(len(features_list), n_features_data)
            ]
        )
    # Truncate if user provided more names than needed (optional,
    # could also raise error)
    elif len(features_list) > n_features_data:
        warnings.warn(
            f"More feature names ({len(features_list)}) provided "
            f"than data columns ({n_features_data}). "
            "Extra names ignored.",
            UserWarning,
            stacklevel=2,
        )
        features_list = features_list[:n_features_data]

    n_features = len(features_list)  # Final number of features used

    # Manage labels
    if labels is None:
        # Generate default layer labels if none provided
        labels_list = [f"Layer {idx+1}" for idx in range(n_layers)]
    else:
        labels_list = list(labels)  # Ensure it's a list
        # Check label count consistency
        if len(labels_list) < n_layers:
            warnings.warn(
                f"Fewer labels ({len(labels_list)}) provided than "
                f"layers ({n_layers}). Using generic names for the rest.",
                UserWarning,
                stacklevel=2,
            )
            labels_list.extend(
                [
                    f"Layer {ix + 1}"
                    for ix in range(len(labels_list), n_layers)
                ]
            )
        elif len(labels_list) > n_layers:
            warnings.warn(
                f"More labels ({len(labels_list)}) provided than "
                f"layers ({n_layers}). Extra labels ignored.",
                UserWarning,
                stacklevel=2,
            )
            labels_list = labels_list[:n_layers]

    # --- Normalization (if requested) ---
    if normalize:
        # Calculate max per row (layer), keep dimensions for broadcasting
        # max_per_row shape: (n_layers, 1), e.g., (3, 1)
        importance_matrix = (
            importance_matrix.values
            if isinstance(importance_matrix, pd.DataFrame)
            else importance_matrix
        )

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
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    # Add the first angle to the end to close the loop for plotting
    angles_closed = angles + angles[:1]

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Get colors from specified colormap or list
    try:
        cmap_obj = get_cmap(cmap, default="tab10", failsafe="discrete")
        # Sample colors if it's a standard Matplotlib cmap
        colors = [cmap_obj(i / n_layers) for i in range(n_layers)]
    except ValueError:  # Handle case where cmap might be a list of colors
        if isinstance(cmap, list):
            colors = cmap
            if len(colors) < n_layers:
                warnings.warn(
                    f"Provided color list has fewer colors "
                    f"({len(colors)}) than layers ({n_layers}). "
                    f"Colors will repeat.",
                    UserWarning,
                    stacklevel=2,
                )
        else:  # Fallback if cmap is invalid string or list
            warnings.warn(
                f"Invalid cmap '{cmap}'. Falling back to 'tab10'.",
                UserWarning,
                stacklevel=2,
            )
            cmap_obj = get_cmap("tab10", default="tab10", failsafe="discrete")
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
        ax.plot(
            angles_closed,
            values_closed,
            label=label,
            color=color,
            linewidth=2,
        )

        # Fill the area if requested
        if fill:
            ax.fill(angles_closed, values_closed, color=color, alpha=0.25)

    # --- Customize Plot Appearance ---
    ax.set_title(title, size=16, y=1.1)  # Adjust title position

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
        ax.set_yticklabels(
            ["0.25", "0.50", "0.75", "1.00"], fontsize=9, color="gray"
        )

    # Show grid lines if requested
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    else:
        ax.grid(False)

    # Add legend, positioned outside the plot area
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # Adjust layout to prevent labels/title overlapping
    plt.tight_layout(pad=2.0)

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


plot_feature_fingerprint.__doc__ = r"""
Create a radar chart visualizing feature-importance profiles.

This function draws a polar (radar) chart that compares how the
importance of a common set of features varies across multiple
groups/layers (e.g., different models, years, or spatial zones).
Each group is drawn as a closed polygon, producing an interpretable
"fingerprint" of relative influence across features (see also the
dataset helper :func:`~kdiagram.datasets.make_fingerprint_data`;
concept introduced in :footcite:t:`kouadiob2025`.

The angular position encodes the feature index, and the radius encodes
its (optionally normalized) importance value. Normalization allows
shape-only comparison across layers, independent of absolute scale.

Parameters
----------
importances : array-like of shape (n_layers, n_features)
    The importance matrix. Each row corresponds to one layer/group
    and each column to a feature. Accepts a list of lists, a NumPy
    array, or a pandas DataFrame.

features : list of str, optional
    Names of the features (length must match the number of columns
    in ``importances``). If ``None``, generic names
    ``['feature 1', ..., 'feature N']`` are generated.

labels : list of str, optional
    Display names for layers (length should match ``n_layers``).
    If ``None``, generic names ``['Layer 1', ..., 'Layer M']`` are
    generated. When counts mismatch, the function pads/truncates and
    issues a warning.

normalize : bool, default=True
    If ``True``, normalize each row to the unit interval via
    :math:`r'_{ij} = r_{ij}/\max_k r_{ik}` (safe-dividing by zero
    yields zeros). This highlights *shape* differences across layers.
    If ``False``, raw magnitudes are plotted.

fill : bool, default=True
    If ``True``, fill each polygon with a translucent color; otherwise
    draw outlines only.

cmap : str or list, default='tab10'
    Either a Matplotlib colormap name (e.g., ``'viridis'``,
    ``'plasma'``, ``'tab10'``) or an explicit list of colors. Lists
    shorter than the number of layers will cycle with a warning.

title : str, default='Feature Impact Fingerprint'
    Figure title.

figsize : tuple of (float, float), optional
    Figure size in inches. If ``None``, a sensible default is used.

show_grid : bool, default=True
    Whether to show polar grid lines.

savefig : str, optional
    Path to save the figure (e.g., ``'fingerprint.png'``). If
    omitted, the plot is shown interactively.

Returns
-------
ax : matplotlib.axes.Axes
    The polar axes containing the radar chart (useful for further
    customization).

Notes
-----
**Angular encoding.** With :math:`N` features, angular positions are
equally spaced:

.. math::

   \theta_j \;=\; \frac{2\pi j}{N}, \qquad j = 0, \dots, N-1.

**Closing polygons.** To draw closed fingerprints, the first vertex
:math:`(\theta_0, r_{i0})` is appended again at :math:`2\pi` for each
layer :math:`i`.

**Row-wise normalization (default).** If ``normalize=True``, each row
:math:`\mathbf r_i=(r_{i0},\dots,r_{i,N-1})` is scaled to its maximum:

.. math::

   r'_{ij} \;=\;
   \begin{cases}
     \dfrac{r_{ij}}{\max_k r_{ik}}, & \max_k r_{ik} > 0,\\[6pt]
     0, & \text{otherwise,}
   \end{cases}

which emphasizes *shape* differences between layers but removes absolute
magnitude information. Set ``normalize=False`` to compare magnitudes.

**Alternative min–max scaling (pre-processing).** If you prefer values
distributed over :math:`[0,1]` using the local range, apply this
transformation per row before calling the function:

.. math::

   r''_{ij} \;=\;
   \frac{r_{ij} - \min_k r_{ik}}
        {\max_k r_{ik} - \min_k r_{ik} + \varepsilon},

with a small :math:`\varepsilon>0` to avoid division by zero.

**Data assumptions.** Importance values are expected to be non-negative.
Rows with a non-positive maximum (all zeros or all negative) become
zeros under the default normalization. If your data can be negative,
either:
(1) set ``normalize=False`` and choose appropriate radial limits, or
(2) shift/scale to non-negative values (e.g., min–max per row).

**Missing/invalid values.** ``NaN`` or ``inf`` entries propagate to the
plot and may render gaps. Clean data beforehand, e.g.:

.. code-block:: python

   import numpy as np
   X = np.asarray(importances, float)
   X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

**Radial limits and ticks.** The plot enforces a non-negative radius
(``ax.set_ylim(bottom=0)``). For unnormalized data, you may set a
custom maximum:

.. code-block:: python

   ax.set_rmax( np.nanmax(importances) )

Optionally add/readjust radial ticks for readability:

.. code-block:: python

   ax.set_yticks([0.25, 0.5, 0.75, 1.0])
   ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])

**Feature order matters.** The perceived shape depends on feature
ordering around the circle. Keep a consistent, meaningful order across
comparisons (e.g., domain grouping or sorted by average importance).

**Many features or layers.** With large :math:`N`, tick labels can
overlap. Consider thinning labels or rotating them:

.. code-block:: python

   angles = ax.get_xticks()
   ax.set_xticks(angles[::2])
   ax.set_xticklabels([lbl for i, lbl in enumerate(features) if i % 2 == 0],
                      rotation=25, ha="right")

For many layers, prefer a discrete colormap and a multi-column legend
or move it outside:

.. code-block:: python

   ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=2)

**Color and accessibility.** Use colorblind-friendly palettes (e.g.,
``'tab10'``, ``'tab20'``) or pass an explicit color list. Avoid relying
on color alone when printing in grayscale—consider distinct linestyles.

**Complexity.** Runtime and memory scale as
:math:`\mathcal O(MN)` for :math:`M` layers and :math:`N` features.
For very large inputs, down-select features or layers for clarity.

**Utilities.** Inputs are coerced to a numeric 2D array and feature
names managed via lightweight helpers (e.g., ``ensure_2d``,
``columns_manager``). Name count mismatches are padded/truncated with a
warning rather than raising.

See Also
--------
kdiagram.datasets.make_fingerprint_data :
    Generate a synthetic importance matrix suitable for this plot.
kdiagram.plot.relationship.plot_relationship :
    Polar scatter for true–predicted relationships.
matplotlib.pyplot.polar :
    Underlying polar plotting primitives.

Examples
--------
Generate random importances and plot with normalization and fills.

>>> import numpy as np
>>> from kdiagram.plot.feature_based import plot_feature_fingerprint
>>> rng = np.random.default_rng(42)
>>> imp = rng.random((3, 6))   # 3 layers, 6 features
>>> feats = [f'Feature {i+1}' for i in range(6)]
>>> labels = ['Model A', 'Model B', 'Model C']
>>> ax = plot_feature_fingerprint(
...     importances=imp,
...     features=feats,
...     labels=labels,
...     title='Random Feature Importance Comparison',
...     cmap='Set3',
...     normalize=True,
...     fill=True
... )

Year-over-year weights without normalization.

>>> features = ['rainfall', 'GWL', 'seismic', 'density', 'geo']
>>> weights = [
...     [0.2, 0.4, 0.1, 0.6, 0.3],  # 2023
...     [0.3, 0.5, 0.2, 0.4, 0.4],  # 2024
...     [0.1, 0.6, 0.2, 0.5, 0.3],  # 2025
... ]
>>> years = ['2023', '2024', '2025']
>>> ax = plot_feature_fingerprint(
...     importances=weights,
...     features=features,
...     labels=years,
...     title='Feature Influence Over Years',
...     cmap='tab10',
...     normalize=False
... )

References
----------
.. footbibliography::
"""
