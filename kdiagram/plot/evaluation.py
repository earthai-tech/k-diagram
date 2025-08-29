#   License: Apache-2.0
#   Author: LKouadio <etanoyau@gmail.com>

import warnings
from typing import Any, Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
)

from ..compat.matplotlib import get_cmap
from ..compat.sklearn import root_mean_squared_error, type_of_target
from ..decorators import check_non_emptiness
from ..utils.handlers import columns_manager
from ..utils.mathext import compute_pinball_loss
from ..utils.plot import set_axis_grid
from ..utils.validator import validate_yy

__all__ = [
    "plot_polar_roc",
    "plot_polar_pr_curve",
    "plot_polar_confusion_matrix",
    "plot_polar_confusion_matrix_in",
    "plot_polar_confusion_multiclass",
    "plot_polar_classification_report",
    "plot_pinball_loss",
    "plot_regression_performance",
]


@check_non_emptiness(params=["y_true", "y_preds"])
def plot_polar_roc(
    y_true: np.ndarray,
    *y_preds: np.ndarray,
    names: Optional[list[str]] = None,
    title: str = "Polar ROC Curve",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    if not y_preds:
        raise ValueError(
            "At least one prediction array" " (*y_preds) must be provided."
        )

    if names and len(names) != len(y_preds):
        warnings.warn(
            "Number of names does not match models." " Using defaults.",
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i+1}" for i in range(len(y_preds))]

    y_true, _ = validate_yy(y_true, y_preds[0])  # Validate first pred

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0, 1, len(y_preds)))

    # --- Plot No-Skill Reference Spiral ---
    # In polar, the y=x line becomes an Archimedean spiral
    no_skill_theta = np.linspace(0, np.pi / 2, 100)
    no_skill_radius = np.linspace(0, 1, 100)
    ax.plot(
        no_skill_theta,
        no_skill_radius,
        color="gray",
        linestyle="--",
        lw=1.5,
        label="No-Skill (AUC = 0.5)",
    )

    # --- Calculate and Plot ROC for Each Model ---
    for i, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Map FPR to angle and TPR to radius
        model_theta = fpr * (np.pi / 2)
        model_radius = tpr

        # Plot the model's ROC spiral
        ax.plot(
            model_theta,
            model_radius,
            color=colors[i],
            lw=2.5,
            label=f"{names[i]} (AUC = {roc_auc:.2f})",
        )

        # Fill the area under the curve (AUC)
        ax.fill(model_theta, model_radius, color=colors[i], alpha=0.15)

    # --- Formatting ---
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim(0, 1.0)

    # Set angular tick labels to represent False Positive Rate
    ax.set_xticks(np.linspace(0, np.pi / 2, 6))
    ax.set_xticklabels([f"{val:.1f}" for val in np.linspace(0, 1, 6)])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate", labelpad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_polar_roc.__doc__ = r"""
Plots a Polar Receiver Operating Characteristic (ROC) Curve.

This function visualizes the performance of binary
classification models by mapping the standard ROC curve onto a
polar plot. It is a novel visualization developed as part of the
analytics framework in :footcite:p:`kouadiob2025`.

Parameters
----------
y_true : np.ndarray
    1D array of true binary labels (0 or 1).
*y_preds : np.ndarray
    One or more 1D arrays of predicted probabilities or scores
    for the positive class.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
title : str, default="Polar ROC Curve"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    curve.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
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
plot_polar_pr_curve : A companion plot for precision-recall.
sklearn.metrics.roc_curve : The underlying scikit-learn function.

Notes
-----
A Receiver Operating Characteristic (ROC) curve is a standard
tool for evaluating binary classifiers :footcite:p:`Powers2011`.
It plots the True Positive Rate (TPR) against the False
Positive Rate (FPR) at various threshold settings.

.. math::

   \text{TPR} = \frac{TP}{TP + FN} \quad , \quad
   \text{FPR} = \frac{FP}{FP + TN}

This function adapts the concept to a polar plot:
    
- The **angle (θ)** is mapped to the False Positive Rate,
  spanning from 0 at 0° to 1 at 90°.
- The **radius (r)** is mapped to the True Positive Rate,
  spanning from 0 at the center to 1 at the edge.

A model with no skill (random guessing) is represented by a
perfect Archimedean spiral. A good model will have a curve that
bows outwards, maximizing the area under the curve (AUC).

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_classification
>>> from kdiagram.plot.evaluation import plot_polar_roc
>>>
>>> # Generate synthetic binary classification data
>>> X, y_true = make_classification(
...     n_samples=500, n_classes=2, random_state=42
... )
>>>
>>> # Simulate predictions from two models
>>> y_pred_good = y_true * 0.7 + np.random.rand(500) * 0.3
>>> y_pred_bad = np.random.rand(500)
>>>
>>> # Generate the plot
>>> ax = plot_polar_roc(
...     y_true,
...     y_pred_good,
...     y_pred_bad,
...     names=["Good Model", "Random Model"]
... )

References
----------
.. footbibliography::
"""


@check_non_emptiness(params=["y_true", "y_preds"])
def plot_polar_confusion_matrix(
    y_true: np.ndarray,
    *y_preds: np.ndarray,
    names: Optional[list[str]] = None,
    normalize: bool = True,
    title: str = "Polar Confusion Matrix",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    mask_radius: bool = False,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    # --- Input Validation and Preparation ---
    if not y_preds:
        raise ValueError(
            "At least one prediction array (*y_preds) must be provided."
        )

    if names and len(names) != len(y_preds):
        warnings.warn(
            "Number of names does not match models. Using defaults.",
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i+1}" for i in range(len(y_preds))]

    y_true, _ = validate_yy(y_true, y_preds[0])

    # Check for target type and handle multiclass
    target_type = type_of_target(y_true)
    if target_type != "binary":
        raise NotImplementedError(
            f"Polar confusion matrix currently only supports binary "
            f"classification. Got target type '{target_type}'. A chord "
            f"diagram is planned for multiclass support in the future."
        )

    # --- Calculate Confusion Matrices ---
    matrices = []
    for y_pred in y_preds:
        # For binary classification, we assume predictions are probabilities
        # and use a 0.5 threshold.
        y_pred_class = (np.asarray(y_pred) > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        matrices.append([tp, fp, tn, fn])

    matrices = np.array(matrices)
    if normalize:
        totals = matrices.sum(axis=1, keepdims=True)
        matrices = matrices / totals

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0, 1, len(y_preds)))

    # --- Plot Bars for Each Model ---
    num_models = len(y_preds)
    bar_width = (2 * np.pi / 4) / (num_models + 1)  # Width of each bar

    categories = [
        "True Positive",
        "False Positive",
        "True Negative",
        "False Negative",
    ]
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)

    for i, (name, matrix) in enumerate(zip(names, matrices)):
        offsets = angles + (i - num_models / 2 + 0.5) * bar_width
        ax.bar(
            offsets,
            matrix,
            width=bar_width,
            color=colors[i],
            alpha=0.7,
            label=name,
        )

    # --- Formatting ---
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Proportion" if normalize else "Count", labelpad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_polar_confusion_matrix.__doc__ = r"""
Plots a Polar Confusion Matrix for binary classification.

This function creates a polar bar chart to visualize the four
key components of a binary confusion matrix: True Positives
(TP), False Positives (FP), True Negatives (TN), and False
Negatives (FN).

Parameters
----------
y_true : np.ndarray
    1D array of true binary labels (0 or 1).
*y_preds : np.ndarray
    One or more 1D arrays of predicted probabilities or scores
    for the positive class. A threshold of 0.5 is used to
    convert probabilities to class labels.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
normalize : bool, default=True
    If ``True``, the confusion matrix values are normalized to
    proportions (summing to 1.0 for each model). If ``False``,
    raw counts are shown.
title : str, default="Polar Confusion Matrix"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    set of bars.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
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
plot_polar_confusion_multiclass : 
    The companion plot for multiclass problems.
sklearn.metrics.confusion_matrix : 
    The underlying scikit-learn function.

Notes
-----
The confusion matrix is a fundamental tool for evaluating a
classifier's performance :footcite:p:`scikit-learn`. This function 
maps its four components to a polar bar chart for intuitive 
comparison.

- **True Positives (TP)**: 
  Correctly predicted positive cases.
- **False Positives (FP)**: 
  Negative cases incorrectly predicted as positive.
- **True Negatives (TN)**: 
  Correctly predicted negative cases.
- **False Negatives (FN)**: 
  Positive cases incorrectly predicted as negative.

Each of these four categories is assigned its own angular sector,
and the height (radius) of the bar in that sector represents the
count or proportion of samples in that category.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_classification
>>> from kdiagram.plot.evaluation import plot_polar_confusion_matrix
>>>
>>> # Generate synthetic binary classification data
>>> X, y_true = make_classification(
...     n_samples=500, n_classes=2, flip_y=0.2, random_state=42
... )
>>>
>>> # Simulate predictions from two models
>>> y_pred1 = y_true * 0.8 + np.random.rand(500) * 0.4 # Good model
>>> y_pred2 = np.random.rand(500) # Random model
>>>
>>> # Generate the plot
>>> ax = plot_polar_confusion_matrix(
...     y_true,
...     y_pred1,
...     y_pred2,
...     names=["Good Model", "Random Model"],
...     normalize=True
... )

References
----------
.. footbibliography::
"""


@check_non_emptiness(params=["y_true", "y_preds"])
def plot_polar_confusion_matrix_in(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[list[str]] = None,
    normalize: bool = True,
    title: str = "Polar Confusion Matrix",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    mask_radius: bool = False,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    # --- Input Validation and Preparation ---
    y_true, y_pred = validate_yy(y_true, y_pred)

    labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)

    if class_labels and len(class_labels) != n_classes:
        warnings.warn(
            "Length of class_labels does not match number of classes.",
            stacklevel=2,
        )
        class_labels = None
    if not class_labels:
        class_labels = [f"Class {lo}" for lo in labels]

    # --- Calculate Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        # Normalize across rows (true labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm.astype("float") / row_sums

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0, 1, n_classes))

    # --- Plot Grouped Bars ---
    bar_width = (2 * np.pi / n_classes) / (n_classes + 1)
    # Angles for each "True Label" group
    group_angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)

    for i in range(n_classes):  # For each true class
        # Calculate offsets for the predicted class bars within the group
        offsets = group_angles + (i - n_classes / 2 + 0.5) * bar_width
        # The values are the predictions for that true class
        values = cm[:, i]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            color=colors[i],
            alpha=0.7,
            label=f"Predicted {class_labels[i]}",
        )

    # --- Formatting ---
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_xticks(group_angles)
    ax.set_xticklabels([f"True\n{lo}" for lo in class_labels], fontsize=10)
    ax.set_ylabel("Proportion" if normalize else "Count", labelpad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return ax


# Create a more convenient alias for the function
plot_polar_confusion_multiclass = plot_polar_confusion_matrix_in

plot_polar_confusion_matrix_in.__doc__ = r"""
Plots a Polar Confusion Matrix for multiclass classification.

This function creates a grouped polar bar chart to visualize the
performance of a multiclass classifier. Each angular sector
represents a true class, and the bars within it show the
distribution of the model's predictions for that class. 

Parameters
----------
y_true : np.ndarray
    1D array of true class labels.
y_pred : np.ndarray
    1D array of predicted class labels from a model.
class_labels : list of str, optional
    Display names for each of the classes. If not provided,
    generic names like ``'Class 0'`` will be generated. The
    order must correspond to the sorted order of the labels in
    ``y_true`` and ``y_pred``.
normalize : bool, default=True
    If ``True``, the confusion matrix values are normalized across
    each true class (row) to show proportions. If ``False``,
    raw counts are shown.
title : str, default="Polar Confusion Matrix"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each
    predicted class bar.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
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
plot_polar_confusion_matrix : 
    The companion plot for binary problems.
sklearn.metrics.confusion_matrix : 
    The underlying scikit-learn function.

Notes
-----
The confusion matrix, :math:`\mathbf{C}`, is a fundamental tool
for evaluating a classifier. Each element :math:`C_{ij}` contains
the number of observations known to be in group :math:`i` but
predicted to be in group :math:`j`.

This function visualizes this matrix by dedicating an angular
sector to each true class :math:`i`. Within that sector, a set of
bars is drawn, where the height of the :math:`j`-th bar
corresponds to the value of :math:`C_{ij}`. This makes it easy to
see how samples from a single true class are distributed among the
predicted classes :footcite:p:`scikit-learn`.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_classification
>>> from kdiagram.plot.evaluation import plot_polar_confusion_matrix_in
>>>
>>> # Generate synthetic multiclass data
>>> X, y_true = make_classification(
...     n_samples=1000,
...     n_features=20,
...     n_informative=10,
...     n_classes=4,
...     n_clusters_per_class=1,
...     flip_y=0.15,
...     random_state=42
... )
>>> # Simulate predictions with some common confusions
>>> y_pred = y_true.copy()
>>> # Confuse some 2s as 3s
>>> y_pred[np.where((y_true == 2) & (np.random.rand(1000) < 0.3))] = 3
>>>
>>> # Generate the plot
>>> ax = plot_polar_confusion_matrix_in(
...     y_true,
...     y_pred,
...     class_labels=["Class A", "Class B", "Class C", "Class D"],
...     title="Multiclass Polar Confusion Matrix"
... )

References
----------
.. footbibliography::
"""


@check_non_emptiness(params=["y_true", "y_preds"])
def plot_polar_pr_curve(
    y_true: np.ndarray,
    *y_preds: np.ndarray,
    names: Optional[list[str]] = None,
    title: str = "Polar Precision-Recall Curve",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    # --- Input Validation ---
    if not y_preds:
        raise ValueError("Provide at least one prediction array (*y_preds).")

    if names and len(names) != len(y_preds):
        warnings.warn(
            "Number of names does not match models. Using defaults.",
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i+1}" for i in range(len(y_preds))]

    y_true, _ = validate_yy(y_true, y_preds[0])

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0, 1, len(y_preds)))

    # --- Plot No-Skill Reference Circle ---
    no_skill = np.mean(y_true)
    ax.plot(
        np.linspace(0, np.pi / 2, 100),
        [no_skill] * 100,
        color="gray",
        linestyle="--",
        lw=1.5,
        label=f"No-Skill (AP = {no_skill:.2f})",
    )

    # --- Calculate and Plot PR Curve for Each Model ---
    for i, y_pred in enumerate(y_preds):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap_score = average_precision_score(y_true, y_pred)

        # Map Recall to angle and Precision to radius
        model_theta = recall * (np.pi / 2)
        model_radius = precision

        # Plot the model's PR curve
        ax.plot(
            model_theta,
            model_radius,
            color=colors[i],
            lw=2.5,
            label=f"{names[i]} (AP = {ap_score:.2f})",
        )
        # Optional: Fill area
        ax.fill(model_theta, model_radius, color=colors[i], alpha=0.15)

    # --- Formatting ---
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim(0, 1.0)

    # Set angular tick labels to represent Recall
    ax.set_xticks(np.linspace(0, np.pi / 2, 6))
    ax.set_xticklabels([f"{val:.1f}" for val in np.linspace(0, 1, 6)])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision", labelpad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_polar_pr_curve.__doc__ = r"""
Plots a Polar Precision-Recall (PR) Curve.

This function visualizes the performance of binary
classification models by mapping the standard PR curve onto a
polar plot. It is particularly useful for evaluating models on
imbalanced datasets where ROC curves can be misleading.

Parameters
----------
y_true : np.ndarray
    1D array of true binary labels (0 or 1).
*y_preds : np.ndarray
    One or more 1D arrays of predicted probabilities or scores
    for the positive class.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
title : str, default="Polar Precision-Recall Curve"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    curve.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
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
plot_polar_roc : A companion plot for ROC analysis.
sklearn.metrics.precision_recall_curve : 
    The underlying scikit-learn function.

Notes
-----
A Precision-Recall (PR) curve is a standard tool for
evaluating binary classifiers, especially on imbalanced data
:footcite:p:`Powers2011`. It plots Precision against Recall at
various threshold settings.

.. math::

   \text{Precision} = \frac{TP}{TP + FP} \quad , \quad
   \text{Recall} = \frac{TP}{TP + FN}

This function adapts the concept to a polar plot:
    
- The **angle (θ)** is mapped to **Recall**, spanning from 0
  at 0° to 1 at 90°.
- The **radius (r)** is mapped to **Precision**, spanning from 0
  at the center to 1 at the edge.

A "no-skill" classifier, which predicts randomly based on the
class distribution, is represented by a horizontal line (a
circle in polar coordinates) at a radius equal to the
proportion of positive samples. A good model will have a curve
that bows outwards towards the top-right corner of the plot,
maximizing the area under the curve (Average Precision).

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_classification
>>> from kdiagram.plot.evaluation import plot_polar_pr_curve
>>>
>>> # Generate imbalanced binary classification data
>>> X, y_true = make_classification(
...     n_samples=1000,
...     n_classes=2,
...     weights=[0.9, 0.1], # 10% positive class
...     flip_y=0.1,
...     random_state=42
... )
>>>
>>> # Simulate predictions from two models
>>> y_pred_good = y_true * 0.6 + np.random.rand(1000) * 0.4
>>> y_pred_bad = np.random.rand(1000)
>>>
>>> # Generate the plot
>>> ax = plot_polar_pr_curve(
...     y_true,
...     y_pred_good,
...     y_pred_bad,
...     names=["Good Model", "Random Model"]
... )

References
----------
.. footbibliography::
"""


@check_non_emptiness(params=["y_true", "y_pred"])
def plot_polar_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[list[str]] = None,
    title: str = "Polar Classification Report",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    mask_radius: bool = False,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    # --- Input Validation ---
    y_true, y_pred = validate_yy(y_true, y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)

    if class_labels and len(class_labels) != n_classes:
        warnings.warn(
            "Length of class_labels does not match number of classes.",
            stacklevel=2,
        )
        class_labels = None
    if not class_labels:
        class_labels = [f"Class {lo}" for lo in labels]

    # --- Calculate Metrics ---
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True
    )
    metrics = {"Precision": [], "Recall": [], "F1-Score": []}
    for label in labels:
        metrics["Precision"].append(report[str(label)]["precision"])
        metrics["Recall"].append(report[str(label)]["recall"])
        metrics["F1-Score"].append(report[str(label)]["f1-score"])

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )
    cmap_obj = get_cmap(cmap, default="viridis")
    metric_colors = cmap_obj(np.linspace(0, 1, 3))

    # --- Plot Grouped Bars ---
    n_metrics = 3
    bar_width = (2 * np.pi / n_classes) / (n_metrics + 1)
    group_angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)

    for i, (metric_name, values) in enumerate(metrics.items()):
        offsets = group_angles + (i - n_metrics / 2 + 0.5) * bar_width
        ax.bar(
            offsets,
            values,
            width=bar_width,
            color=metric_colors[i],
            alpha=0.7,
            label=metric_name,
        )

    # --- Formatting ---
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_xticks(group_angles)
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_ylabel("Score", labelpad=25)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_polar_classification_report.__doc__ = r"""
Plots a Polar Classification Report.

This function creates a grouped polar bar chart to visualize the
key performance metrics (Precision, Recall, and F1-Score) for
each class in a multiclass classification problem. It provides a
detailed, per-class summary of a classifier's performance.

Parameters
----------
y_true : np.ndarray
    1D array of true class labels.
y_pred : np.ndarray
    1D array of predicted class labels from a model.
class_labels : list of str, optional
    Display names for each of the classes. If not provided,
    generic names like ``'Class 0'`` will be generated. The
    order must correspond to the sorted order of the labels in
    ``y_true`` and ``y_pred``.
title : str, default="Polar Classification Report"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each of the
    three metrics (Precision, Recall, F1-Score).
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
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
plot_polar_confusion_multiclass :
    A plot showing the raw counts of predictions.
sklearn.metrics.classification_report : 
    The underlying scikit-learn function.

Notes
-----
This plot visualizes the three most common metrics for evaluating
a multiclass classifier on a per-class basis
:footcite:p:`Powers2011`.

1.  **Precision**: The ability of the classifier not to label as
    positive a sample that is negative.

    .. math::

       \text{Precision} = \frac{TP}{TP + FP}

2.  **Recall (Sensitivity)**: The ability of the classifier to
    find all the positive samples.

    .. math::

       \text{Recall} = \frac{TP}{TP + FN}

3.  **F1-Score**: The harmonic mean of precision and recall,
    providing a single score that balances both.

    .. math::

       \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}\\
           {\text{Precision} + \text{Recall}}

Each class is assigned an angular sector, and within that sector,
three bars are drawn, with their heights (radii) corresponding
to the scores for these metrics.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_classification
>>> from kdiagram.plot.evaluation import plot_polar_classification_report
>>>
>>> # Generate synthetic multiclass data
>>> X, y_true = make_classification(
...     n_samples=1000,
...     n_classes=4,
...     n_informative=10,
...     flip_y=0.2,
...     random_state=42
... )
>>> # Simulate predictions
>>> y_pred = y_true.copy()
>>> # Add some errors
>>> y_pred[np.random.choice(1000, 150, replace=False)] = 0
>>>
>>> # Generate the plot
>>> ax = plot_polar_classification_report(
...     y_true,
...     y_pred,
...     class_labels=["Class A", "Class B", "Class C", "Class D"],
...     title="Per-Class Performance Report"
... )

References
----------
.. footbibliography::
"""


@check_non_emptiness(params=["y_true", "y_preds_quantiles"])
def plot_pinball_loss(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
    names: Optional[list[str]] = None,
    title: str = "Pinball Loss per Quantile",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    mask_radius: bool = False,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    # --- Input Validation ---
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )
    # Ensure quantiles are sorted for plotting
    sort_idx = np.argsort(quantiles)
    quantiles = np.asarray(quantiles)[sort_idx]
    y_preds_quantiles = y_preds_quantiles[:, sort_idx]

    # --- Calculate Pinball Loss for each quantile ---
    losses = []
    for i in range(len(quantiles)):
        loss = compute_pinball_loss(
            y_true, y_preds_quantiles[:, i], quantiles[i]
        )
        losses.append(loss)

    # --- Plotting Setup ---
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "polar"}
    )

    # Angle is the quantile level, radius is the loss
    angles = quantiles * 2 * np.pi
    radii = losses

    # --- Plotting ---
    ax.plot(angles, radii, "o-", label="Pinball Loss")
    ax.fill(angles, radii, alpha=0.25)

    # --- Formatting ---
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(
        [f"{q:.2f}" for q in np.linspace(0, 1, 8, endpoint=False)]
    )
    ax.set_xlabel("Quantile Level")
    ax.set_ylabel("Average Pinball Loss (Lower is Better)", labelpad=25)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return ax


plot_pinball_loss.__doc__ = r"""
Plots the Pinball Loss for each quantile of a forecast.

This function creates a polar plot to visualize the performance
of a probabilistic forecast at each individual quantile level.
The radius of the plot at a given angle (quantile) represents
the average Pinball Loss, providing a granular view of the
model's accuracy across its entire predictive distribution.

Parameters
----------
y_true : np.ndarray
    1D array of the true observed values.
y_preds_quantiles : np.ndarray
    2D array of quantile forecasts, with shape
    ``(n_samples, n_quantiles)``.
quantiles : np.ndarray
    1D array of the quantile levels corresponding to the columns
    of the prediction array.
names : list of str, optional
    Display names for each of the models. *Note: This function
    currently supports plotting one model at a time.*
title : str, default="Pinball Loss per Quantile"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used for the plot's fill and line.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
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
compute_pinball_loss : The underlying mathematical utility.
compute_crps : A score calculated by averaging the pinball loss.
:ref:`userguide_probabilistic` : The user guide for probabilistic plots.

Notes
-----
The Pinball Loss, :math:`\mathcal{L}_{\tau}`, is a proper scoring
rule for evaluating a single quantile forecast :math:`q` at level
:math:`\tau` against an observation :math:`y`. It asymmetrically
penalizes errors, giving a different weight to over- and under-
predictions :footcite:p:`Gneiting2007b`.

.. math::

   \mathcal{L}_{\tau}(q, y) =
   \begin{cases}
     (y - q) \tau & \text{if } y \ge q \\
     (q - y) (1 - \tau) & \text{if } y < q
   \end{cases}

This plot calculates the average Pinball Loss for each provided
quantile and visualizes these scores on a polar axis, where the
angle represents the quantile level and the radius represents the
loss. A good forecast will have a small, symmetrical shape close
to the center.

Examples
--------
>>> import numpy as np
>>> from scipy.stats import norm
>>> from kdiagram.plot.evaluation import plot_pinball_loss
>>>
>>> # Generate synthetic data
>>> np.random.seed(0)
>>> n_samples = 1000
>>> y_true = np.random.normal(loc=50, scale=10, size=n_samples)
>>> quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
>>>
>>> # Simulate a model that is good at the median, worse at the tails
>>> scales = np.array([12, 10, 8, 10, 12]) # Different scales per quantile
>>> y_preds = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=scales
... )
>>>
>>> # Generate the plot
>>> ax = plot_pinball_loss(
...     y_true,
...     y_preds,
...     quantiles,
...     title="Pinball Loss per Quantile"
... )

References
----------
.. footbibliography::
"""


def _get_scores(
    y_true: np.ndarray,
    y_preds: list[np.ndarray],
    metrics: list[Union[str, Callable]],
    higher_is_better: Optional[dict[str, bool]] = None,
):
    """
    Internal helper to compute scores, ensuring higher is always better.
    """
    scores = {}
    higher_is_better = higher_is_better or {}

    METRIC_MAP = {
        "r2": (r2_score, True),
        "neg_mean_absolute_error": (mean_absolute_error, False),
        "neg_root_mean_squared_error": (
            root_mean_squared_error,
            # lambda yt, yp: mean_squared_error(yt, yp, squared=False),
            False,
        ),
    }

    for metric in metrics:
        metric_name = (
            metric
            if isinstance(metric, str)
            else getattr(metric, "__name__", "custom")
        )

        func = None
        # Default assumption: higher is better
        is_score = True

        # 1. Prioritize the user's explicit override.
        if metric_name in higher_is_better:
            is_score = higher_is_better[metric_name]
            # Now find the function to call
            if callable(metric):
                func = metric
            elif isinstance(metric, str) and metric in METRIC_MAP:
                func, _ = METRIC_MAP[metric]  # Ignore the default is_score
            else:
                warnings.warn(
                    f"Unknown metric '{metric}' provided in "
                    f"higher_is_better. Skipping.",
                    stacklevel=2,
                )
                continue

        # 2. If no override, use the default logic.
        elif callable(metric):
            func = metric
            # Infer from name if it's an error metric
            if "error" in metric_name or "loss" in metric_name:
                is_score = False
        elif isinstance(metric, str) and metric in METRIC_MAP:
            func, is_score = METRIC_MAP[metric]
        else:
            warnings.warn(
                f"Unknown metric '{metric}'. Skipping.", stacklevel=2
            )
            continue

        # Calculate the scores using the determined function
        calculated_scores = [func(y_true, yp) for yp in y_preds]

        # If lower is better (i.e., it's an error metric), negate the scores
        if not is_score:
            scores[metric_name] = [-s for s in calculated_scores]
        else:
            scores[metric_name] = calculated_scores

    return scores


def plot_regression_performance(
    y_true: Optional[np.ndarray] = None,
    *y_preds: np.ndarray,
    names: Optional[list[str]] = None,
    metrics: Optional[
        Union[str, Callable, list[Union[str, Callable]]]
    ] = None,
    metric_values: Optional[dict[str, list[float]]] = None,
    add_to_defaults: bool = False,
    metric_labels: Optional[Union[dict[str, str], bool, list]] = None,
    higher_is_better: Optional[dict[str, bool]] = None,
    norm: Literal["per_metric", "global", "none"] = "per_metric",
    global_bounds: Optional[dict[str, tuple[float, float]]] = None,
    min_radius: float = 0.02,
    clip_to_bounds: bool = True,
    title: str = "Regression Model Performance",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    show_grid: bool = True,
    grid_props: Optional[dict[str, Any]] = None,
    mask_radius: bool = False,
    savefig: Optional[str] = None,
    dpi: int = 300,
):
    # --- 1. Determine Mode and Calculate Scores ---
    # The function operates in two modes:
    # a) "Values Mode": Pre-computed scores are provided.
    # b) "Data Mode": Scores are computed from y_true and y_preds.

    if metric_values is not None:
        # --- a) Values Mode: Use pre-computed scores ---
        if (y_true is not None) or y_preds:
            raise ValueError(
                "If `metric_values` is provided, `y_true` and "
                "`y_preds` must be None."
            )
        scores = metric_values
        metric_names = list(scores.keys())
        # Infer number of models from the first metric's list
        n_models = len(next(iter(scores.values())))

    elif (y_true is not None) and y_preds:
        # --- b) Data Mode: Compute scores from data ---
        n_models = len(y_preds)
        default_metrics = [
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ]

        if metrics is None:
            metrics_to_use = default_metrics
        else:
            user_metrics = columns_manager(metrics)
            metrics_to_use = (
                default_metrics + user_metrics
                if add_to_defaults
                else user_metrics
            )
        # Helper function calculates scores, ensuring higher is better
        scores = _get_scores(
            y_true,
            list(y_preds),
            metrics_to_use,
            higher_is_better,
        )
        metric_names = list(scores.keys())
    else:
        raise ValueError(
            "Either `metric_values` or both `y_true` and "
            "`y_preds` must be provided."
        )

    # Generate default model names if not provided
    if not names:
        names = [f"Model {i + 1}" for i in range(n_models)]

    # --- 2. Normalize Scores to Determine Bar Radii ---
    # This section translates raw scores into radii for the bars,
    # based on the chosen normalization strategy.
    if norm not in {"per_metric", "global", "none"}:
        raise ValueError(
            "`norm` must be one of {'per_metric','global','none'}."
        )

    normalized: dict[str, np.ndarray] = {}

    if norm == "per_metric":
        # Scale each metric independently to the range [0, 1].
        # 'Best' is 1, 'Worst' is 0 for that specific metric.
        for m, values in scores.items():
            v = np.asarray(values, dtype=float)
            vmin, vmax = float(v.min()), float(v.max())
            if (vmax - vmin) > 1e-12:
                r = (v - vmin) / (vmax - vmin)
                # Ensure even the worst bar is slightly visible
                r = np.maximum(r, min_radius)
            else:
                r = np.ones_like(v)  # All scores are equal
            normalized[m] = r

        radial_min, radial_max = 0.0, 1.0
        tick_vals = [0, 0.25, 0.5, 0.75, 1.0]
        tick_lbls = ["Worst", "0.25", "0.5", "0.75", "Best"]

    elif norm == "global":
        # Scale each metric to [0, 1] based on fixed,
        # user-provided global bounds.
        gb = global_bounds or {}
        for m, values in scores.items():
            v = np.asarray(values, dtype=float)
            if m in gb:
                gmin, gmax = map(float, gb[m])
            else:
                # Fallback to per-metric bounds if not provided
                gmin, gmax = float(v.min()), float(v.max())
                warnings.warn(
                    f"`global_bounds` missing for metric '{m}'. "
                    "Using current data bounds instead.",
                    stacklevel=2,
                )

            if gmax <= gmin:
                r = np.ones_like(v)
            else:
                if clip_to_bounds:
                    v = np.clip(v, gmin, gmax)
                r = (v - gmin) / (gmax - gmin)
                r = np.maximum(r, min_radius)
            normalized[m] = r

        radial_min, radial_max = 0.0, 1.0
        tick_vals = [0, 0.25, 0.5, 0.75, 1.0]
        tick_lbls = ["Worst", "0.25", "0.5", "0.75", "Best"]

    else:  # norm == "none"
        # Plot the raw score values directly without scaling.
        for m, values in scores.items():
            normalized[m] = np.asarray(values, dtype=float)

        # Determine axis limits from all raw values
        all_vals = np.concatenate([normalized[m] for m in metric_names])
        radial_min = float(all_vals.min())
        radial_max = float(all_vals.max())
        if np.isclose(radial_max, radial_min):
            radial_min -= 0.5
            radial_max += 0.5

        tick_vals = np.linspace(radial_min, radial_max, 5).tolist()
        tick_lbls = [f"{t:.2g}" for t in tick_vals]

    # --- 3. Create the Polar Plot ---
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": "polar"},
    )

    # Prepare angles and widths for the grouped bars
    n_metrics = len(metric_names)
    group_angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    bar_width = (2 * np.pi / n_metrics) / (len(names) + 1)

    # Get a color for each model
    cmap_obj = get_cmap(cmap, default="viridis")
    colors = cmap_obj(np.linspace(0, 1, len(names)))

    # Draw the bars for each model
    for i, name in enumerate(names):
        radii = [normalized[m][i] for m in metric_names]
        # Calculate the angular offset for each bar in the group
        offsets = group_angles + ((i - len(names) / 2 + 0.5) * bar_width)
        ax.bar(
            offsets,
            radii,
            width=bar_width,
            color=colors[i],
            alpha=0.7,
            label=name,
        )

    # --- 4. Add Formatting and Rings ---
    # Draw the 'Best' and 'Worst' performance rings for reference
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        theta,
        np.full_like(theta, radial_max),
        color="green",
        linestyle="-",
        lw=1.5,
        label="Best Performance",
    )
    ax.plot(
        theta,
        np.full_like(theta, radial_min),
        color="red",
        linestyle="--",
        lw=1.5,
        label="Worst Performance",
    )

    # Set titles, ticks, and labels
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_xticks(group_angles)

    # Handle custom metric labels
    if metric_labels is False or (
        isinstance(metric_labels, list) and not metric_labels
    ):
        ax.set_xticklabels([])
    elif isinstance(metric_labels, dict):
        ax.set_xticklabels(
            [metric_labels.get(m, m) for m in metric_names],
            fontsize=10,
        )
    else:
        ax.set_xticklabels(metric_names, fontsize=10)

    # Set radial ticks and limits
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_lbls)
    ax.set_ylim(radial_min, radial_max)

    # Place legend outside the plot area for clarity
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.1),
    )

    # Apply grid styling and optional masking
    set_axis_grid(
        ax,
        show_grid=show_grid,
        grid_props=grid_props,
    )
    if mask_radius:
        ax.set_yticklabels([])

    # --- 5. Finalize and Show/Save ---
    plt.tight_layout()
    if savefig:
        plt.savefig(
            savefig,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()

    return ax


plot_regression_performance.__doc__ = r"""
Creates a Polar Performance Chart for regression models.

This function generates a grouped polar bar chart to visually
compare the performance of multiple regression models across
several evaluation metrics simultaneously. It provides a
holistic snapshot of model strengths and weaknesses.

Parameters
----------
y_true : np.ndarray, optional
    1D array of true observed values. Required unless
    ``metric_values`` is provided.
*y_preds : np.ndarray
    One or more 1D arrays of predicted values from different
    models.
names : list of str, optional
    Display names for each of the models. If not provided,
    generic names like ``'Model 1'`` will be generated.
metrics : str, callable, or list of such, optional
    The metric(s) to compute. If ``None``, defaults to
    ``['r2', 'neg_mean_absolute_error',
    'neg_root_mean_squared_error']``. Can be strings
    recognized by scikit-learn or custom callable functions.
metric_values : dict of {str: list of float}, optional
    A dictionary of pre-calculated metric scores. Keys are the
    metric names and values are lists of scores, one for each
    model. If provided, ``y_true`` and ``y_preds`` must be ``None``.
add_to_defaults : bool, default=False
    If ``True``, the user-provided ``metrics`` are added to the
    default set of metrics instead of replacing them.
metric_labels : dict, bool, or list, optional
    Controls the angular axis labels.
    
    - ``dict``: A mapping from original metric names to new
      display names (e.g., ``{'r2': 'R²'}``).
    - ``False`` or ``[]``: Hides all angular labels.
    - ``None`` (default): Shows the original metric names.
    
higher_is_better : dict of {str: bool}, optional
    A dictionary to explicitly specify whether a higher score is
    better for each metric. Keys should be metric names and
    values should be ``True`` (higher is better) or ``False``
    (lower is better). This overrides the default behavior for
    both string and callable metrics.    
norm : {'per_metric', 'global', 'none'}, default='per_metric'
    The strategy for normalizing raw metric scores into bar radii.

    - ``'per_metric'``: (Default) Normalizes scores for each
      metric independently to the range [0, 1]. The best-
      performing model on a given metric gets a radius of 1,
      and the worst gets 0. This is best for comparing the
      *relative* performance of models.
    - ``'global'``: Normalizes scores using fixed, absolute
      bounds defined in the ``global_bounds`` parameter. This is
      useful for comparing models against a consistent,
      predefined scale.
    - ``'none'``: Plots the raw, un-normalized metric scores
      directly. Use with caution, as metrics with different
      scales can make the plot difficult to interpret.

global_bounds : dict of {str: (float, float)}, optional
    A dictionary providing fixed `(min, max)` bounds for each
    metric when using ``norm='global'``. The dictionary keys
    should be the metric names (e.g., 'r2') and the values
    should be a tuple of the worst and best possible scores.
    For example, ``{'r2': (0.0, 1.0)}``.
min_radius : float, default=0.02
    A small minimum radius to ensure that even the worst-
    performing bars (with a normalized score of 0) remain
    slightly visible on the plot.
clip_to_bounds : bool, default=True
    If ``True`` and ``norm='global'``, any score that falls
    outside the range specified in ``global_bounds`` will be
    clipped to that range before normalization. If ``False``,
    scores can result in radii less than 0 or greater than 1.
title : str, default="Regression Model Performance"
    The title for the plot.
figsize : tuple of (float, float), default=(8, 8)
    The figure size in inches.
cmap : str, default='viridis'
    The colormap used to assign a unique color to each model's
    bars.
show_grid : bool, default=True
    Toggle the visibility of the polar grid lines.
grid_props : dict, optional
    Custom keyword arguments passed to the grid for styling.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels.
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
plot_model_comparison : A similar plot using a radar chart format.
:ref:`userguide_evaluation` : The user guide for evaluation plots.

Notes
-----
This plot provides a holistic, multi-metric view of model
performance, making it easy to identify trade-offs.

1.  **Score Calculation**: For each model and each metric, a
    score is calculated. Note that for error-based metrics
    (like MAE or RMSE), the function uses the negated version
    (e.g., ``neg_mean_absolute_error``) so that a **higher
    score is always better** :footcite:p:`scikit-learn`.

2.  **Normalization**: To make scores comparable, the scores for
    each metric are independently scaled to the range [0, 1]
    using Min-Max normalization. A score of 1 represents the
    best-performing model for that metric, and a score of 0
    represents the worst.

3.  **Polar Mapping**:
    
    - Each metric is assigned its own angular sector.
    - The normalized score of each model is mapped to the
      **radius** (height) of its bar within that sector.

Examples
--------
>>> import numpy as np
>>> from kdiagram.plot.evaluation import plot_regression_performance
>>>
>>> # Generate synthetic data for three models
>>> np.random.seed(0)
>>> n_samples = 200
>>> y_true = np.random.rand(n_samples) * 50
>>> y_pred_good = y_true + np.random.normal(0, 5, n_samples)
>>> y_pred_biased = y_true - 10 + np.random.normal(0, 2, n_samples)
>>>
>>> # Generate the plot with clean labels
>>> ax = plot_regression_performance(
...     y_true,
...     y_pred_good,
...     y_pred_biased,
...     names=["Good Model", "Biased Model"],
...     title="Model Performance Comparison",
...     metric_labels={
...         'r2': '$R$^2',
...         'neg_mean_absolute_error': 'MAE',
...         'neg_root_mean_squared_error': 'RMSE'
...     }
... )

References
----------
.. footbibliography::
"""
