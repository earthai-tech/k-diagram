#   License: Apache-2.0
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
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

from ..compat.matplotlib import get_colors
from ..compat.sklearn import root_mean_squared_error, type_of_target
from ..decorators import check_non_emptiness
from ..utils.handlers import columns_manager
from ..utils.mathext import compute_pinball_loss
from ..utils.plot import (
    canonical_acov,
    maybe_delegate_cartesian,
    set_axis_grid,
    setup_polar_axes,
    validate_kind,
    warn_acov_preference,
)
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
    names: list[str] | None = None,
    title: str = "Polar ROC Curve",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "tab10",
    colors: list[str] = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    acov: str = "quarter_circle",
    fill_alpha: float = 0.15,
    show_no_skill: bool = True,
    show_auc: bool = True,
    savefig: str | None = None,
    dpi: int = 300,
    kind: str = "polar",
    ax: Axes | None = None,
):
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
        names = [f"Model {i + 1}" for i in range(len(y_preds))]

    y_true, _ = validate_yy(y_true, y_preds[0])  # Validate first pred

    # Branch on rendering kind; keep existing warnings/messages for polar path.
    kind = validate_kind(kind)
    # --- Plotting Setup ---
    if kind == "polar":
        # Respect parameter but force quarter-circle (keep original warning text)
        canon = canonical_acov(acov, raise_on_invalid=False)
        if canon != "quarter_circle":
            warnings.warn(
                "plot_polar_roc currently renders best as a quarter circle. "
                f"Received acov='{acov}'. For ROC, θ=0 at the right (East) and a "
                "90° span give the clearest reading of FPR (angle) vs TPR (radius). "
                "Proceeding with acov='quarter_circle'.",
                UserWarning,
                stacklevel=2,
            )

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": "polar"}
            )
        else:
            fig = ax.figure
    else:  # cartesian
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

    # Colors (unchanged behavior / messages)
    colors = get_colors(
        len(y_preds),
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # --- No-skill reference ---
    if show_no_skill:
        if kind == "polar":
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
        else:
            # Cartesian: y = x diagonal
            ax.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                color="gray",
                linestyle="--",
                lw=1.5,
                label="No-Skill (AUC = 0.5)",
            )

    # --- ROC curves ---
    for i, (name, pred) in enumerate(zip(names, y_preds)):
        fpr, tpr, _ = roc_curve(y_true, pred)
        roc_auc = auc(fpr, tpr)
        label = f"{name} (AUC = {roc_auc:.2f})" if show_auc else name

        if kind == "polar":
            # Map FPR to angle and TPR to radius
            model_theta = fpr * (np.pi / 2)
            model_radius = tpr

            ax.plot(
                model_theta,
                model_radius,
                color=colors[i],
                lw=2.5,
                label=label,
            )
            ax.fill(
                model_theta,
                model_radius,
                0.0,
                color=colors[i],
                alpha=fill_alpha,
            )
        else:
            # Standard ROC in Cartesian coordinates
            ax.plot(
                fpr,
                tpr,
                color=colors[i],
                lw=2.5,
                label=label,
            )
            ax.fill_between(
                fpr, tpr, 0.0, color=colors[i], alpha=fill_alpha, step=None
            )

    # --- Formatting ---
    ax.set_title(title, fontsize=16, y=1.1)

    if kind == "polar":
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_ylim(0, 1.0)

        # Angular tick labels represent False Positive Rate
        ax.set_xticks(np.linspace(0, np.pi / 2, 6))
        ax.set_xticklabels([f"{val:.1f}" for val in np.linspace(0, 1, 6)])

        ax.set_xlabel("False Positive Rate", labelpad=25)
        ax.set_ylabel("True Positive Rate", labelpad=25)
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
        set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
    else:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        if show_grid:
            if grid_props:
                ax.grid(True, **grid_props)
            else:
                ax.grid(True)
        else:
            ax.grid(False)

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
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
y_true : array-like of shape (n_samples,)
    Ground-truth binary labels (0/1).  Values are validated and
    flattened.  If labels are not in {0, 1}, they will be cast
    to integers after validation.
*y_preds : array-like of shape (n_samples,), required
    One or more arrays of predicted *scores* or *probabilities*
    for the positive class.  Each array is validated against
    ``y_true`` and flattened.  At least one prediction vector
    must be provided.
names : list of str or None, default=None
    Display names for the prediction series.  When ``None``,
    generic names such as ``"Model 1"``, ``"Model 2"``, … are
    generated.  If provided but the length differs from the
    number of series, a warning is issued and generic names are
    used.
title : str, default="Polar ROC Curve"
    Figure title.
figsize : tuple of float, default=(8, 8)
    Figure size in inches.
cmap : str, default="viridis"
    Matplotlib colormap used to assign distinct colors to the
    curves.
show_grid : bool, default=True
    Whether to display polar grid lines.  Styling can be tuned
    through ``grid_props``.
grid_props : dict or None, default=None
    Keyword arguments forwarded to the internal grid helper to
    adjust grid line style (e.g., ``{"linestyle": "--",
    "alpha": 0.5}``).
acov : {"quarter_circle", ...}, default="quarter_circle"
    Angular coverage request.  **For this release it is accepted
    only for compatibility; any value other than
    ``"quarter_circle"`` causes a warning and the plot is reset
    to a 0–90° span with θ=0 at East.**
fill_alpha : float, default=0.15
    Opacity used to fill the area under each ROC curve.
show_no_skill : bool, default=True
    If ``True``, draws the baseline no-skill curve (TPR = FPR)
    as a dashed line.
show_auc : bool, default=True
    If ``True``, appends the numerical AUC value to each legend
    label.
savefig : str or None, default=None
    When a path is given, the figure is saved to that location
    (directory must exist).  If ``None``, the figure is shown.
dpi : int, default=300
    Resolution used when saving the figure.
kind : {'polar', 'cartesian'}, default='polar'
    Rendering mode selector. When set to ``'polar'`` (default), the
    plot uses a Matplotlib polar projection and applies polar-specific
    options (``acov``, ``zero_at``, ``clockwise``)
ax : matplotlib.axes.Axes or None, default=None
    Existing polar axes to draw on.  When ``None``, a new figure
    and polar axes are created.

Returns
-------
ax : matplotlib.axes.Axes
    The polar axes containing the ROC visualization.  This can
    be used for further customization.

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

the plot is **always** rendered as a *quarter circle* (0–90°) with
θ=0 placed at the **East** (right) of the plot.  Passing any
value for ``acov`` other than ``"quarter_circle"`` will emit a
warning and the setting will be reset to a quarter circle.  This
layout yields the clearest reading of FPR (angle) versus TPR
(radius) for ROC.

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
    names: list[str] | None = None,
    normalize: bool = True,
    title: str = "Polar Confusion Matrix",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    colors: list[str] = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    acov: str = "default",
    zero_at: Literal["N", "E", "S", "W"] = "N",
    clockwise: bool = True,
    categories: list[str] = None,
    kind: str = "polar",
    ax: Axes | None = None,
) -> Axes:
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
        names = [f"Model {i + 1}" for i in range(len(y_preds))]

    y_true, _ = validate_yy(y_true, y_preds[0])

    # ---- Early delegation ----

    kind = validate_kind(kind)
    cart_ax = maybe_delegate_cartesian(
        kind,
        _plot_confusion_matrix_cartesian,
        y_true,
        *y_preds,
        names=names,
        normalize=normalize,
        title=title,
        figsize=figsize,
        cmap=cmap,
        colors=colors,
        show_grid=show_grid,
        grid_props=grid_props,
        savefig=savefig,
        dpi=dpi,
        ax=ax,
        categories=categories,
    )
    if cart_ax is not None:
        return cart_ax

    target_type = type_of_target(y_true)
    if target_type != "binary":
        raise ValueError(
            "Polar confusion matrix currently supports only binary "
            f"classification. Got target type '{target_type}'."
            "For multiclass classification, use "
            "`plot_polar_confusion_multiclass` instead."
        )

    # --- Calculate Confusion Matrices ---
    matrices = []
    for y_pred in y_preds:
        y_pred_class = (np.asarray(y_pred) > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        # order: TP, FP, TN, FN to match your categories below
        matrices.append([tp, fp, tn, fn])

    matrices = np.asarray(matrices, dtype=float)
    if normalize:
        totals = matrices.sum(axis=1, keepdims=True)
        # guard against division by zero
        totals[totals == 0] = 1.0
        matrices = matrices / totals

    # AXES with ACOV (angular coverage)
    canon = canonical_acov(acov, raise_on_invalid=False, fallback="default")
    # Optional, but recommended: nudge users toward a full 360° for CM
    warn_acov_preference(canon, preferred="default")

    # Create/configure a polar Axes and get the span in radians
    fig, ax, span = setup_polar_axes(
        ax,
        acov=canon,
        figsize=figsize,
        zero_at=zero_at,  # where θ=0 points, default "N"
        clockwise=clockwise,  # True => clockwise
    )

    # cmap_obj = get_cmap(cmap, default="viridis")
    # colors = cmap_obj(np.linspace(0, 1, len(y_preds)))
    colors = get_colors(
        len(y_preds),
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # Bars computed from the SPAN (no hard-coded 2pi)
    categories = columns_manager(categories, empty_as_none=False)

    _CATEGORIES = [
        "True Positive",
        "False Positive",
        "True Negative",
        "False Negative",
    ]

    categories += _CATEGORIES
    categories = categories[:4]

    n_sectors = len(categories)

    # Centers of each sector over the chosen span
    angles = np.linspace(0.0, span, n_sectors, endpoint=False)

    # Each sector is span / n_sectors wide; share it across models
    sector_width = span / n_sectors
    # 80% of sector width total for bars; split among models
    bar_width = (sector_width * 0.80) / max(1, len(y_preds))

    for i, (name, row) in enumerate(zip(names, matrices)):
        # offset bars around the sector center for side-by-side display
        offset = (i - (len(y_preds) - 1) / 2.0) * bar_width
        centers = angles + offset
        ax.bar(
            centers,
            row,  # 4 values: TP, FP, TN, FN
            width=bar_width,
            color=colors[i],
            alpha=0.7,
            label=name,
            edgecolor="black",
            linewidth=0.5,
        )

    # Formatting
    ax.set_title(title, fontsize=16, y=1.10)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)

    ax.set_ylabel(
        "Proportion" if normalize else "Count",
        labelpad=30,
    )

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    # Legends can get cramped on small spans; keep outside on the right
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10))

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _plot_confusion_matrix_cartesian(
    y_true: np.ndarray,
    *y_preds: np.ndarray,
    names: list[str] | None = None,
    normalize: bool = True,
    title: str = "Polar Confusion Matrix",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    colors: list[str] | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
    categories: list[str] | None = None,
) -> Axes:
    # Mirror the same binary restriction and message for consistency
    target_type = type_of_target(y_true)
    if target_type != "binary":
        raise ValueError(
            "Polar confusion matrix currently supports only binary "
            f"classification. Got target type '{target_type}'."
            "For multiclass classification, use "
            "`plot_polar_confusion_multiclass` instead."
        )

    # Compute matrices like the polar path
    matrices = []
    for y_pred in y_preds:
        y_pred_class = (np.asarray(y_pred) > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        matrices.append([tp, fp, tn, fn])  # TP, FP, TN, FN
    matrices = np.asarray(matrices, dtype=float)
    if normalize:
        totals = matrices.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        matrices = matrices / totals

    # Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = get_colors(
        len(y_preds),
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    cats = columns_manager(categories, empty_as_none=False)
    _CATEGORIES = [
        "True Positive",
        "False Positive",
        "True Negative",
        "False Negative",
    ]
    cats += _CATEGORIES
    cats = cats[:4]

    x = np.arange(len(cats))
    width = 0.8 / max(1, len(y_preds))  # use 80% of the band, split by models

    for i, (name, row) in enumerate(zip(names or [], matrices)):
        offset = (i - (len(y_preds) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            row,
            width=width,
            label=name,
            color=colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Proportion" if normalize else "Count")

    if show_grid:
        ax.grid(True, **(grid_props or {}))
    else:
        ax.grid(False)

    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0.0, 1.0 if normalize else max(1.0, matrices.max() * 1.1))
    ax.legend(loc="upper right")

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
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
    1D array of true binary labels (0 or 1). Shape
    ``(n_samples,)``.
*y_preds : np.ndarray
    One or more 1D arrays of predicted probabilities or
    scores for the positive class. A fixed threshold of
    ``0.5`` is applied to derive class labels.
names : list of str, optional
    Display names for the models. If not provided, generic
    names like ``'Model 1'`` are generated.
normalize : bool, default=True
    If ``True``, values are converted to proportions that sum
    to ``1.0`` within each model. If ``False``, raw counts
    are shown.
title : str, default="Polar Confusion Matrix"
    Title to place above the figure.
figsize : tuple of (float, float), default=(8, 8)
    Figure size in inches.
cmap : str, default='viridis'
    Colormap used to assign distinct colors to the bars of
    each model.
show_grid : bool, default=True
    Toggle the polar grid. Set to ``False`` for a minimal
    look.
grid_props : dict, optional
    Keyword arguments forwarded to the grid styling helper,
    e.g. ``{'linestyle': '--', 'alpha': 0.5}``.
mask_radius : bool, default=False
    If ``True``, hide the radial tick labels to declutter
    the plot.
savefig : str, optional
    File path where the figure is saved. If ``None``, the
    plot is shown interactively.
dpi : int, default=300
    Resolution in dots per inch used when saving.

acov : {'default', 'half_circle', 'quarter_circle',
    'eighth_circle'}, default='default'
    Angular coverage (span) of the polar plot. ``'default'``
    covers 360°, ``'half_circle'`` 180°, ``'quarter_circle'``
    90°, and ``'eighth_circle'`` 45°. Fewer degrees compress
    the four sectors into a smaller sweep.
zero_at : {'N', 'E', 'S', 'W'}, default='N'
    Cardinal direction where ``θ = 0`` is placed. For
    example, ``'E'`` puts zero at the right-hand side.
clockwise : bool, default=True
    Direction of increasing angle. ``True`` draws clockwise;
    ``False`` draws counter-clockwise.
categories : list of str, optional
    Custom labels for the four sectors. Must contain exactly
    four items. If ``None``, uses::
        
        ['True Positive', 'False Positive',
         'True Negative', 'False Negative']
        
    The sectors are laid out starting at ``θ = 0`` and follow
    the chosen direction.
kind : {'polar', 'cartesian'}, default='polar'
    Rendering mode selector. When set to ``'polar'`` (default), the
    plot uses a Matplotlib polar projection and applies polar-specific
    options (``acov``, ``zero_at``, ``clockwise``) via internal helpers.
    When set to ``'cartesian'``, the function *delegates* to a
    Cartesian renderer (through ``maybe_delegate_cartesian``), keeping
    names/colors/figsize/grid behavior consistent while ignoring polar-
    only arguments (e.g., ``acov``, ``zero_at``, ``clockwise``). The
    return value is always the ``Axes`` actually used. The value is
    validated with ``validate_kind`` (case-insensitive); invalid values
    raise ``ValueError("kind must be 'polar' or 'cartesian'.")``.
ax : matplotlib.axes.Axes, optional
    Existing polar axes to draw on. If ``None``, a new
    figure and polar axes are created.
    
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
    class_labels: list[str] | None = None,
    normalize: bool = True,
    title: str = "Polar Confusion Matrix",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "tab10",
    colors: list[str] = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    acov: str = "default",
    zero_at: Literal["N", "E", "S", "W"] = "N",
    clockwise: bool = True,
    categories: list[str] = None,
    savefig: str | None = None,
    dpi: int = 300,
    kind: str = "polar",
    ax: Axes | None = None,
) -> Axes:
    if categories is not None:
        warnings.warn(
            "Categories is kept for API symmetry; unused", stacklevel=2
        )

    # ----- validate inputs -----
    y_true, y_pred = validate_yy(y_true, y_pred)

    labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)

    if class_labels and len(class_labels) != n_classes:
        warnings.warn(
            "Length of class_labels does not match number of classes. "
            "Falling back to generic labels.",
            stacklevel=2,
        )
        class_labels = None

    if not class_labels:
        class_labels = [f"Class {lo}" for lo in labels]

    # ----- early delegation if 'cartesian' -----
    kind = validate_kind(kind)
    cart_ax = maybe_delegate_cartesian(
        kind,
        _plot_confusion_matrix_in_cartesian,  # cartesian renderer
        y_true,
        y_pred,
        class_labels=class_labels,
        normalize=normalize,
        title=title,
        figsize=figsize,
        cmap=cmap,
        colors=colors,
        show_grid=show_grid,
        grid_props=grid_props,
        categories=categories,  # will warn (kept for symmetry; unused)
        savefig=savefig,
        dpi=dpi,
        ax=ax,
    )
    if cart_ax is not None:
        return cart_ax

    # ----- compute confusion matrix (row-normalize if asked) -----
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero
        cm = cm.astype(float) / row_sums

    # ----- configure polar axes with acov helpers -----
    canon = canonical_acov(acov, raise_on_invalid=False, fallback="default")
    # Heuristic: full 360° usually looks best for multiclass CM
    warn_acov_preference(canon, preferred="default")

    fig, ax, span = setup_polar_axes(
        ax,
        acov=canon,
        figsize=figsize,
        zero_at=zero_at,  # where θ=0 points (N/E/S/W)
        clockwise=clockwise,  # True => clockwise
    )
    # ----- colors -----
    colors = get_colors(
        n_classes,
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )
    # cmap_obj = get_cmap(cmap, default="viridis")
    # colors = cmap_obj(np.linspace(0, 1, n_classes))

    # ----- layout: sectors and bar widths driven by 'span' -----
    # One sector per TRUE class across the chosen angular span
    sector_width = span / max(1, n_classes)
    group_angles = np.linspace(0.0, span, n_classes, endpoint=False)

    # Inside each sector, render 'n_classes' bars (one per PRED class).
    # Reserve ~80% of the sector for bars, split evenly among them.
    bar_width = (sector_width * 0.80) / max(1, n_classes)

    # ----- draw grouped bars -----
    # Loop over predicted class index; draw its bar in every sector.
    for i in range(n_classes):
        # offset bars around the sector center for side-by-side display
        offset = (i - (n_classes - 1) / 2.0) * bar_width
        centers = group_angles + offset
        # values: column i => predicted==class i across all true classes
        values = cm[:, i]
        ax.bar(
            centers,
            values,
            width=bar_width,
            color=colors[i],
            alpha=0.7,
            label=f"Predicted {class_labels[i]}",
            edgecolor="black",
            linewidth=0.5,
        )

    # ----- cosmetics -----
    ax.set_title(title, fontsize=16, y=1.10)
    ax.set_xticks(group_angles)
    ax.set_xticklabels([f"True\n{lo}" for lo in class_labels], fontsize=10)

    # Slightly larger pad raises the radial label to avoid clashes
    ax.set_ylabel("Proportion" if normalize else "Count", labelpad=30)

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    # Legends can get cramped on narrow spans; keep it outside
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10))

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _plot_confusion_matrix_in_cartesian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str] | None = None,
    normalize: bool = True,
    title: str = "Cartesian Confusion Matrix",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "tab10",
    colors: list[str] | None = None,
    show_grid: bool = True,
    grid_props: dict | None = None,
    categories: list[str] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    if categories is not None:
        warnings.warn(
            "Categories is kept for API symmetry; unused", stacklevel=2
        )

    # Same validation / label logic as polar path
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)

    if class_labels and len(class_labels) != n_classes:
        warnings.warn(
            "Length of class_labels does not match number of classes. "
            "Falling back to generic labels.",
            stacklevel=2,
        )
        class_labels = None
    if not class_labels:
        class_labels = [f"Class {lo}" for lo in labels]

    # Confusion matrix (row-normalize if requested)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm.astype(float) / row_sums

    # Axes creation
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Colors consistent with polar path
    colors = get_colors(
        n_classes,
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # Grouped bars: x = true class, one bar per predicted class
    x = np.arange(n_classes)
    width = 0.8 / max(1, n_classes)

    for i in range(n_classes):
        offset = (i - (n_classes - 1) / 2.0) * width
        ax.bar(
            x + offset,
            cm[:, i],
            width=width,
            color=colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=f"Predicted {class_labels[i]}",
        )

    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"True\n{lo}" for lo in class_labels], fontsize=10)
    ax.set_ylabel("Proportion" if normalize else "Count")

    if show_grid:
        ax.grid(True, **(grid_props or {}))
    else:
        ax.grid(False)

    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ymax = 1.0 if normalize else max(1.0, cm.max() * 1.1)
    ax.set_ylim(0.0, ymax)
    ax.legend(loc="upper right")

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return ax


# Convenient alias
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
kind : {'polar', 'cartesian'}, default='polar'
    Rendering mode selector. When set to ``'polar'`` (default), the
    plot uses a Matplotlib polar projection and applies polar-specific
    options (``acov``, ``zero_at``, ``clockwise``) via internal helpers.
    When set to ``'cartesian'``, the function *delegates* to a
    Cartesian renderer (through ``maybe_delegate_cartesian``), keeping
    names/colors/figsize/grid behavior consistent while ignoring polar-
    only arguments (e.g., ``acov``, ``zero_at``, ``clockwise``). The
    return value is always the ``Axes`` actually used.
ax : Axes, optional
    The Matplotlib Axes object to use for plotting. If ``None``,
    a new figure and axes will be created.
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
    names: list[str] | None = None,
    title: str = "Polar Precision-Recall Curve",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    colors: list[str] = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    acov: str = "quarter_circle",
    fill_alpha: float = 0.15,
    show_no_skill: bool = True,
    show_ap: bool = True,
    savefig: str | None = None,
    dpi: int = 300,
    kind: str = "polar",
    ax: Axes | None = None,
) -> Axes:
    # ---------- validation ----------
    if not y_preds:
        raise ValueError("Provide at least one prediction array (*y_preds).")

    if names and len(names) != len(y_preds):
        warnings.warn(
            ("Number of names does not match models. Using defaults."),
            stacklevel=2,
        )
        names = None
    if not names:
        names = [f"Model {i + 1}" for i in range(len(y_preds))]

    # ---- early delegation if 'cartesian' (keep polar code unchanged below) ----
    kind = validate_kind(kind)
    cart_ax = maybe_delegate_cartesian(
        kind,
        _plot_pr_curve_cartesian,
        y_true,
        *y_preds,
        names=names,
        title=title,
        figsize=figsize,
        cmap=cmap,
        colors=colors,
        show_grid=show_grid,
        grid_props=grid_props,
        fill_alpha=fill_alpha,
        show_no_skill=show_no_skill,
        show_ap=show_ap,
        savefig=savefig,
        dpi=dpi,
        ax=ax,
    )
    if cart_ax is not None:
        return cart_ax

    # Canonicalize and enforce quarter-circle coverage
    canon = canonical_acov(acov, raise_on_invalid=False)
    if canon != "quarter_circle":
        warnings.warn(
            (
                "Non-default 'acov' received (acov="
                f"'{acov}'). The polar PR plot is fixed to a "
                "quarter circle (0–90°) with θ=0 at the right "
                "(East) so Recall maps cleanly to angle and "
                "Precision to radius. Proceeding with "
                "acov='quarter_circle'."
            ),
            UserWarning,
            stacklevel=2,
        )

    # Validate arrays and drop/align NaNs consistently
    y_true, _ = validate_yy(y_true, y_preds[0])

    #  figure / axis
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw={"projection": "polar"}
        )
    else:
        fig = ax.figure

    # Colors for each series
    colors = get_colors(
        len(y_preds),
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # reference: no-skill line
    # For PR, the no-skill level equals the positive prevalence.
    if show_no_skill:
        base = float(np.mean(y_true))
        ax.plot(
            np.linspace(0.0, np.pi / 2.0, 100),
            [base] * 100,
            color="gray",
            linestyle="--",
            lw=1.5,
            label=f"No-Skill (AP = {base:.2f})",
        )

    #  curves
    for i, y_pred in enumerate(y_preds):
        # Compute PR
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        # Map recall->angle, precision->radius
        theta = rec * (np.pi / 2.0)
        radius = prec

        # Draw line and an optional filled band
        lbl = f"{names[i]} (AP = {ap:.2f})" if show_ap else names[i]
        ax.plot(theta, radius, color=colors[i], lw=2.5, label=lbl)
        ax.fill(theta, radius, color=colors[i], alpha=fill_alpha)

    # formatting
    ax.set_title(title, fontsize=16, y=1.1)

    # Limit to quarter circle explicitly
    ax.set_thetamin(0.0)
    ax.set_thetamax(90.0)
    ax.set_ylim(0.0, 1.0)

    # Angle ticks show recall ∈ [0,1]
    ax.set_xticks(np.linspace(0.0, np.pi / 2.0, 6))
    ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0.0, 1.0, 6)])

    # Axis labels
    ax.set_xlabel("Recall", labelpad=25)
    ax.set_ylabel("Precision", labelpad=25)

    # Legend and grid
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    # I/O
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _plot_pr_curve_cartesian(
    y_true: np.ndarray,
    *y_preds: np.ndarray,
    names: list[str] | None = None,
    title: str = "Polar Precision-Recall Curve",  # keep same default title for parity
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    colors: list[str] | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    fill_alpha: float = 0.15,
    show_no_skill: bool = True,
    show_ap: bool = True,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Colors per model (consistent with polar path)
    colors = get_colors(
        len(y_preds),
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # No-skill baseline: positive prevalence
    if show_no_skill:
        base = float(np.mean(y_true))
        ax.plot(
            [0.0, 1.0],
            [base, base],
            color="gray",
            linestyle="--",
            lw=1.5,
            label=f"No-Skill (AP = {base:.2f})",
        )

    for i, y_pred in enumerate(y_preds):
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        lbl = f"{names[i]} (AP = {ap:.2f})" if show_ap else names[i]
        ax.plot(rec, prec, color=colors[i], lw=2.5, label=lbl)
        ax.fill_between(rec, prec, 0.0, color=colors[i], alpha=fill_alpha)

    ax.set_title(title, fontsize=16)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if show_grid:
        ax.grid(True, **(grid_props or {}))
    else:
        ax.grid(False)
    ax.legend(loc="lower left")

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
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
y_true : ndarray of shape (n_samples,)
    Ground-truth binary labels in {0, 1}. Values are cast to
    1D and validated against the first prediction.

*y_preds : array-like of shape (n_samples,), optional
    One or more score/probability arrays for the positive
    class. Each must be 1D, numeric, and the same length as
    `y_true`.

names : list of str, optional
    Display names for each model. If not given or length does
    not match `y_preds`, generic labels like "Model 1" are
    used.

title : str, default="Polar Precision-Recall Curve"
    Title displayed above the plot.

figsize : (float, float), default=(8, 8)
    Figure size in inches passed to Matplotlib.

cmap : str, default="viridis"
    Colormap name used to assign distinct colors to curves.
    Any valid Matplotlib colormap string is accepted.

show_grid : bool, default=True
    Whether to draw the polar grid (spokes and rings).

grid_props : dict, optional
    Styling for the grid. For example:
        
    `{"linestyle": "--", "linewidth": 0.6, "alpha": 0.6}`.
    
    Passed to the internal grid helper.

acov : {"quarter_circle", "half_circle", "default",
    "full", "full_circle", "eighth_circle"}, default="quarter_circle"

    Requested angular coverage. For PR, the plot is fixed to
    a quarter circle (0–90°) with θ=0 at the right. If a
    different value is provided, a warning is issued and the
    quarter-circle layout is used.

fill_alpha : float, default=0.15
    Opacity of the area fill under each PR curve. Must be in
    [0, 1].

show_no_skill : bool, default=True
    If True, draws a dashed reference at the positive class
    prevalence and labels it as the no-skill baseline.

show_ap : bool, default=True
    If True, appends "AP = ..." to each curve label using the
    average precision of that model.

savefig : str or path-like, optional
    Path to write the figure. If None, the figure is shown
    instead.

dpi : int, default=300
    Resolution (dots per inch) used when saving the figure.
kind : {'polar', 'cartesian'}, default='polar'
    Rendering mode selector. When set to ``'polar'`` (default), the
    plot uses a Matplotlib polar projection and applies polar-specific
    options (``acov``, ``zero_at``, ``clockwise``) via internal helpers.
    When set to ``'cartesian'``, the function *delegates* to a
    Cartesian renderer (through ``maybe_delegate_cartesian``), keeping
    names/colors/figsize/grid behavior consistent while ignoring polar-
    only arguments (e.g., ``acov``, ``zero_at``, ``clockwise``). The
    return value is always the ``Axes`` actually used. 
ax : matplotlib.axes.Axes, optional
    Existing polar Axes to draw into. If None, a new figure
    and polar Axes are created.

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
    class_labels: list[str] | None = None,
    title: str = "Polar Classification Report",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "tab10",
    colors: list[str] | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    acov: str = "full",
    zero_at: Literal["N", "E", "S", "W"] = "N",
    clockwise: bool = True,
    savefig: str | None = None,
    dpi: int = 300,
    kind: str = "polar",
    ax: Axes | None = None,
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

    kind = validate_kind(kind)
    cart_ax = maybe_delegate_cartesian(
        kind,
        _plot_classification_report_cartesian,
        y_true,
        y_pred,
        class_labels=class_labels,
        title=title,
        figsize=figsize,
        cmap=cmap,
        colors=colors,
        show_grid=show_grid,
        grid_props=grid_props,
        savefig=savefig,
        dpi=dpi,
        ax=ax,
    )
    if cart_ax is not None:
        return cart_ax

    # ----- Configure polar axes with acov helpers -----
    canon = canonical_acov(acov, raise_on_invalid=False, fallback="defaut")
    warn_acov_preference(canon, preferred="default")

    # Create/configure a polar Axes and get the span in radians
    fig, ax, span = setup_polar_axes(
        ax,
        acov=canon,
        figsize=figsize,
        zero_at=zero_at,  # Where θ=0 points (N/E/S/W)
        clockwise=clockwise,  # True => clockwise
    )

    # cmap_obj = get_cmap(cmap, default="viridis")
    # metric_colors = cmap_obj(np.linspace(0, 1, 3))
    # Get colors based on user input or defaults
    metric_colors = get_colors(
        3, colors=colors, cmap=cmap, default="tab10", failsafe="discrete"
    )

    # --- Plot Grouped Bars ---
    n_metrics = 3
    bar_width = (span / n_classes) / (n_metrics + 1)
    group_angles = np.linspace(0, span, n_classes, endpoint=False)

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

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _plot_classification_report_cartesian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str] | None = None,
    title: str = "Polar Classification Report",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "tab10",
    colors: list[str] | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    # Match polar path’s labeling behavior/messages
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

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True
    )
    metrics = {"Precision": [], "Recall": [], "F1-Score": []}
    for label in labels:
        metrics["Precision"].append(report[str(label)]["precision"])
        metrics["Recall"].append(report[str(label)]["recall"])
        metrics["F1-Score"].append(report[str(label)]["f1-score"])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    metric_colors = get_colors(
        3, colors=colors, cmap=cmap, default="tab10", failsafe="discrete"
    )

    x = np.arange(n_classes)
    n_metrics = 3
    width = 0.8 / n_metrics

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=metric_colors[i],
            alpha=0.7,
            label=metric_name,
        )

    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    if show_grid:
        ax.grid(True, **(grid_props or {}))
    else:
        ax.grid(False)
    ax.legend(loc="upper right")

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
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
kind : {'polar', 'cartesian'}, default='polar'
    Rendering mode selector. When set to ``'polar'`` (default), the
    plot uses a Matplotlib polar projection and applies polar-specific
    options (``acov``, ``zero_at``, ``clockwise``) via internal helpers.
    When set to ``'cartesian'``, the function *delegates* to a
    Cartesian renderer (through ``maybe_delegate_cartesian``), keeping
    names/colors/figsize/grid behavior consistent while ignoring polar-
    only arguments (e.g., ``acov``, ``zero_at``, ``clockwise``). The
    return value is always the ``Axes`` actually used. The value is
    validated with ``validate_kind`` (case-insensitive); invalid values
    raise ``ValueError("kind must be 'polar' or 'cartesian'.")``.
ax : Axes, optional
    The Matplotlib Axes object to use for plotting. If ``None``,
    a new figure and axes will be created.
    
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
    names: list[str] | None = None,
    title: str = "Pinball Loss per Quantile",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "tab10",
    colors: list[str] | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    acov: str = "default",
    zero_at: Literal["N", "E", "S", "W"] = "E",
    clockwise: bool = True,
    savefig: str | None = None,
    dpi: int = 300,
    kind: str = "polar",
    ax: Axes | None = None,
):
    # --- Input Validation ---
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )
    # Ensure quantiles are sorted for plotting
    sort_idx = np.argsort(quantiles)
    quantiles = np.asarray(quantiles)[sort_idx]
    y_preds_quantiles = y_preds_quantiles[:, sort_idx]

    kind = validate_kind(kind)
    cart_ax = maybe_delegate_cartesian(
        kind,
        _plot_pinball_loss_cartesian,
        y_true,
        y_preds_quantiles,
        quantiles,
        names=names,
        title=title,
        figsize=figsize,
        cmap=cmap,
        colors=colors,
        show_grid=show_grid,
        grid_props=grid_props,
        savefig=savefig,
        dpi=dpi,
        ax=ax,
    )
    if cart_ax is not None:
        return cart_ax

    # --- Calculate Pinball Loss for each quantile ---
    losses = []
    for i in range(len(quantiles)):
        loss = compute_pinball_loss(
            y_true, y_preds_quantiles[:, i], quantiles[i]
        )
        losses.append(loss)

    # ----- Configure polar axes with acov helpers -----
    canon = canonical_acov(acov, raise_on_invalid=False, fallback="defaut")
    warn_acov_preference(canon, preferred="default")

    # Create/configure a polar Axes and get the span in radians
    fig, ax, span = setup_polar_axes(
        ax,
        acov=canon,
        figsize=figsize,
        zero_at=zero_at,  # Where θ=0 points (N/E/S/W)
        clockwise=clockwise,  # True => clockwise
    )

    # --- Get Colors ---
    colors_list = get_colors(
        len(quantiles),
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # --- Plotting Setup ---
    angles = quantiles * span  # Quantile level as angle
    radii = losses  # Pinball loss as radius

    # Plotting the loss per quantile as a line
    ax.plot(angles, radii, "o-", label="Pinball Loss", color=colors_list[0])
    ax.fill(angles, radii, alpha=0.25, color=colors_list[0])

    # --- Formatting ---
    ax.set_title(title, fontsize=16, y=1.1)
    ax.set_xticks(np.linspace(0, span, 8, endpoint=False))
    ax.set_xticklabels(
        [f"{q:.2f}" for q in np.linspace(0, 1, 8, endpoint=False)]
    )

    ax.set_xlabel("Quantile Level")
    ax.set_ylabel("Average Pinball Loss (Lower is Better)", labelpad=25)
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if mask_radius:
        ax.set_yticklabels([])

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ax


def _plot_pinball_loss_cartesian(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
    names: list[str] | None = None,
    title: str = "Pinball Loss per Quantile",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "tab10",
    colors: list[str] | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    # Match polar path’s validation and sorting
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )

    sort_idx = np.argsort(quantiles)
    quantiles = np.asarray(quantiles)[sort_idx]
    y_preds_quantiles = y_preds_quantiles[:, sort_idx]

    # Compute pinball loss per quantile (same as polar path)
    losses = []
    for i in range(len(quantiles)):
        loss = compute_pinball_loss(
            y_true, y_preds_quantiles[:, i], quantiles[i]
        )
        losses.append(loss)

    # Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Colors (first color mirrors polar behavior)
    colors_list = get_colors(
        len(quantiles),
        colors=colors,
        cmap=cmap,
        default="tab10",
        failsafe="discrete",
    )

    # Plot: x = quantile in [0,1], y = average pinball loss
    ax.plot(
        quantiles, losses, "o-", label="Pinball Loss", color=colors_list[0]
    )
    ax.fill_between(quantiles, losses, 0.0, alpha=0.25, color=colors_list[0])

    # Formatting
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Quantile Level")
    ax.set_ylabel("Average Pinball Loss (Lower is Better)")
    if show_grid:
        ax.grid(True, **(grid_props or {}))
    else:
        ax.grid(False)

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
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
ax : Axes, optional
    The Matplotlib Axes object to use for plotting. If ``None``,
    a new figure and axes will be created.
kind : {'polar', 'cartesian'}, default='polar'
    Rendering mode selector. When set to ``'polar'`` (default), the
    plot uses a Matplotlib polar projection and applies polar-specific
    options (``acov``, ``zero_at``, ``clockwise``) via internal helpers.
    When set to ``'cartesian'``, the function *delegates* to a
    Cartesian renderer (through ``maybe_delegate_cartesian``), keeping
    names/colors/figsize/grid behavior consistent while ignoring polar-
    only arguments (e.g., ``acov``, ``zero_at``, ``clockwise``). The
    return value is always the ``Axes`` actually used. The value is
    validated with ``validate_kind`` (case-insensitive); invalid values
    raise ``ValueError("kind must be 'polar' or 'cartesian'.")``.

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
    metrics: list[str | Callable],
    higher_is_better: dict[str, bool] | None = None,
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
    y_true: np.ndarray | None = None,
    *y_preds: np.ndarray,
    names: list[str] | None = None,
    metrics: str | Callable | list[str | Callable] | None = None,
    metric_values: dict[str, list[float]] | None = None,
    add_to_defaults: bool = False,
    metric_labels: dict[str, str] | bool | list | None = None,
    higher_is_better: dict[str, bool] | None = None,
    norm: Literal["per_metric", "global", "none"] = "per_metric",
    global_bounds: dict[str, tuple[float, float]] | None = None,
    min_radius: float = 0.05,
    clip_to_bounds: bool = True,
    title: str = "Regression Model Performance",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    colors: list[str] = None,
    bp_padding: float = 1.0,
    acov: str = "full",
    zero_at: Literal["N", "E", "S", "W"] = "E",
    clockwise: bool = True,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    mask_radius: bool = False,
    savefig: str | None = None,
    dpi: int = 300,
    kind: str = "polar",
    ax: Axes | None = None,
):
    # --- Input Validation and Warnings ---
    if not (0 < bp_padding <= 1):
        raise ValueError(
            "`bp_padding` must be between 0 and 1, and cannot be exactly 0."
        )
    if bp_padding < 0.5:
        warnings.warn(
            f"The value of `bp_padding` ({bp_padding}) is less than 0.5. "
            "This may cause the 'Best Performance' ring to become"
            " too small and unreadable. Consider increasing the padding"
            " or minimizing the figure for better clarity.",
            UserWarning,
            stacklevel=2,
        )

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

    # --- EARLY DELEGATION TO CARTESIAN ---
    kind = validate_kind(kind)
    cart_ax = maybe_delegate_cartesian(
        kind,
        _plot_regression_performance_cartesian,
        names,
        metric_names,
        normalized,
        tick_vals,
        tick_lbls,
        title=title,
        figsize=figsize,
        cmap=cmap,
        colors=colors,
        metric_labels=metric_labels,
        show_grid=show_grid,
        grid_props=grid_props,
        savefig=savefig,
        dpi=dpi,
        ax=ax,
    )
    if cart_ax is not None:
        return cart_ax

    # --- 3. Create the Polar Plot ---

    # Create/configure a polar Axes and get the span in radians
    canon = canonical_acov(acov, raise_on_invalid=False, fallback="defaut")
    warn_acov_preference(canon, preferred="default")

    fig, ax, span = setup_polar_axes(
        ax,
        acov=canon,
        figsize=figsize,
        zero_at=zero_at,  # Where θ=0 points (N/E/S/W)
        clockwise=clockwise,  # True => clockwise
    )

    # rescale radial max based on best performance padding.
    radial_max_best = radial_max * bp_padding

    # scale radias mak
    # Prepare angles and widths for the grouped bars
    n_metrics = len(metric_names)

    # group_angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    # bar_width = (2 * np.pi / n_metrics) / (len(names) + 1)
    group_angles = np.linspace(0, span, n_metrics, endpoint=False)
    bar_width = (span / n_metrics) / (len(names) + 1)

    # Get a color for each model
    # --- Get Colors ---
    colors = get_colors(
        len(names), colors=colors, cmap=cmap, default="viridis"
    )
    # cmap_obj = get_cmap(cmap, default="viridis")
    # colors = cmap_obj(np.linspace(0, 1, len(names)))

    # Draw the bars for each model
    for i, name in enumerate(names):
        radii = [normalized[m][i] * radial_max_best for m in metric_names]
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
    theta = np.linspace(0, span, 200)
    ax.plot(
        theta,
        np.full_like(theta, radial_max_best),
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
    fig.tight_layout()
    if savefig:
        fig.savefig(
            savefig,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()

    return ax


def _plot_regression_performance_cartesian(
    names: list[str],
    metric_names: list[str],
    normalized: dict[str, np.ndarray],
    tick_vals: list[float],
    tick_lbls: list[str],
    *,
    title: str = "Regression Model Performance",
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    colors: list[str] | None = None,
    metric_labels: dict[str, str] | bool | list | None = None,
    show_grid: bool = True,
    grid_props: dict[str, Any] | None = None,
    savefig: str | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> Axes:
    n_metrics = len(metric_names)
    n_models = len(names)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # one color per model (consistent with polar path)
    model_colors = get_colors(
        n_models, colors=colors, cmap=cmap, default="viridis"
    )

    x = np.arange(n_metrics)
    width = 0.8 / max(1, n_models)

    for i, name in enumerate(names):
        vals = np.array([normalized[m][i] for m in metric_names], dtype=float)
        offset = (i - (n_models - 1) / 2.0) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=model_colors[i],
            alpha=0.7,
            label=name,
        )

    # X tick labels (mirror polar’s metric_labels behavior)
    if metric_labels is False or (
        isinstance(metric_labels, list) and not metric_labels
    ):
        ax.set_xticklabels([])
        ax.set_xticks(x)
    elif isinstance(metric_labels, dict):
        ax.set_xticks(x)
        ax.set_xticklabels(
            [metric_labels.get(m, m) for m in metric_names], fontsize=10
        )
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)

    # Y ticks/labels
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_lbls)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Score")
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.legend(loc="upper right")

    if show_grid:
        ax.grid(True, **(grid_props or {}))
    else:
        ax.grid(False)

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
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
colors : list of str, optional
    A list of custom colors for the bars. If ``None``, the function
    will use the colormap specified in ``cmap`` to generate the colors.
bp_padding : float, default=1.0
    Padding factor for the "Best Performance" ring. A value of 1.0
    places the ring at the maximum radius, while smaller values
    decrease its size.
acov : str, default="full"
    The angular coverage for the plot. Can be one of:
        
    - 'full': 360° (default)
    - 'half': 180°
    - 'quarter': 90°
    - 'eighth': 45°
    
zero_at : {'N', 'E', 'S', 'W'}, default='E'
    The point where θ=0 is located on the plot. Options are:
        
    - 'N': North
    - 'E': East
    - 'S': South
    - 'W': West
    
clockwise : bool, default=True
    If ``True``, the plot is drawn clockwise. If ``False``, the plot
    is drawn counter-clockwise.
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
kind : {'polar', 'cartesian'}, default='polar'
    Rendering mode selector. When set to ``'polar'`` (default), the
    plot uses a Matplotlib polar projection and applies polar-specific
    options (``acov``, ``zero_at``, ``clockwise``) via internal helpers.
    When set to ``'cartesian'``, the function *delegates* to a
    Cartesian renderer (through ``maybe_delegate_cartesian``), keeping
    names/colors/figsize/grid behavior consistent while ignoring polar-
    only arguments (e.g., ``acov``, ``zero_at``, ``clockwise``). The
    return value is always the ``Axes`` actually used. The value is
    validated with ``validate_kind`` (case-insensitive); invalid values
    raise ``ValueError("kind must be 'polar' or 'cartesian'.")``.
ax : Axes, optional
    The Matplotlib Axes object to use for plotting. If ``None``,
    a new figure and axes will be created.
    
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

The radial bars are colored according to the model performance, 
and the best and worst performance rings are shown for reference.

The "Best Performance" ring is drawn at the maximum radial
distance, and the "Worst Performance" ring is drawn at the minimum
distance. Each model's performance is shown as a radial bar, with
the length of the bar corresponding to the normalized score.
    
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
