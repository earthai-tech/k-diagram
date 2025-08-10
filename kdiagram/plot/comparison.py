# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Model comparison plots."""
from __future__ import annotations

import warnings
from numbers import Real
from typing import Any, Callable, Literal

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..compat.matplotlib import get_cmap 
from ..compat.sklearn import StrOptions, type_of_target, validate_params
from ..utils.generic_utils import drop_nan_in
from ..utils.handlers import columns_manager
from ..utils.metric_utils import get_scorer
from ..utils.plot import set_axis_grid
from ..utils.validator import _assert_all_types, is_iterable, validate_yy

__all__ = ["plot_reliability_diagram", "plot_model_comparison"]


@validate_params(
    {
        "y_true": ["array-like"],
        "strategy": [StrOptions({"uniform", "quantile"})],
        "error_bars": [StrOptions({"wilson", "normal", "none"})],
        "counts_panel": [StrOptions({"none", "bottom"})],
        "counts_norm": [StrOptions({"fraction", "count"})],
    }
)
def plot_reliability_diagram(
    y_true,
    *y_preds,
    names: list[str] | None = None,
    sample_weight: list[float] | np.ndarray | None = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    positive_label: int | float | str = 1,
    class_index: int | None = None,
    clip_probs: tuple[float, float] = (0.0, 1.0),
    normalize_probs: bool = True,
    error_bars: str = "wilson",
    conf_level: float = 0.95,
    show_diagonal: bool = True,
    diagonal_kwargs: dict[str, Any] | None = None,
    show_ece: bool = True,
    show_brier: bool = True,
    counts_panel: str = "bottom",
    counts_norm: Literal["fraction", "count"] = "fraction",
    counts_alpha: float = 0.35,
    figsize: tuple[float, float] | None = (9, 7),
    title: str | None = None,
    xlabel: str | None = "Predicted probability",
    ylabel: str | None = "Observed frequency",
    cmap: str = "tab10",
    color_palette: list[Any] | None = None,
    marker: str = "o",
    s: int = 40,
    linewidth: float = 2.0,
    alpha: float = 0.9,
    connect: bool = True,
    legend: bool = True,
    legend_loc: str = "best",
    show_grid: bool = True,
    grid_props: dict | None = None,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    savefig: str | None = None,
    return_data: bool = False,
    **kw,  # for future extension
):
    # -------------- input handling -------------- #
    if len(y_preds) == 0:
        raise ValueError("Provide at least one prediction array via *y_preds.")

    names = columns_manager(names, to_string=True) or []
    if len(names) < len(y_preds):
        names.extend([f"Model_{i+1}" for i in range(len(names), len(y_preds))])
    if len(names) > len(y_preds):
        warnings.warn(
            (
                f"Received {len(names)} names for {len(y_preds)} models. "
                "Extra names ignored."
            ),
            UserWarning,
            stacklevel=2,
        )
        names = names[: len(y_preds)]

    y_true = np.asarray(y_true)
    if type_of_target(y_true) not in ("binary", "multiclass"):
        raise ValueError(
            "y_true must be a classification target. " "Binary reliability is expected."
        )
    y_bin = (y_true == positive_label).astype(int)

    prob_list: list[np.ndarray] = []
    for arr in y_preds:
        arr = np.asarray(arr)
        prob_list.append(_to_prob_vector(arr, class_index))

    if sample_weight is None:
        y_bin, *prob_list = drop_nan_in(y_bin, *prob_list, error="raise")
        w = np.ones_like(y_bin, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        y_bin, *prob_list, w = drop_nan_in(y_bin, *prob_list, w, error="raise")

    clip_lo, clip_hi = clip_probs
    clipped_flag = False
    new_probs = []
    for p in prob_list:
        p0 = p.copy()
        p1 = _prep_probs(p0, clip_lo, clip_hi, normalize_probs)
        if not np.allclose(p0, p1):
            clipped_flag = True
        new_probs.append(p1)
    prob_list = new_probs
    if clipped_flag:
        warnings.warn(
            (
                "Some predicted probabilities were normalized/clipped "
                f"to [{clip_lo}, {clip_hi}]."
            ),
            UserWarning,
            stacklevel=2,
        )

    edges, centers = _build_bins(prob_list, n_bins, strategy, clip_lo, clip_hi)
    z = _z_from_conf(conf_level)

    # -------------- colors & layout -------------- #
    colors = _colors(cmap, color_palette, len(prob_list))

    if counts_panel == "bottom":
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        axb = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axb = None

    # -------------- compute & plot -------------- #

    per_model: dict[str, pd.DataFrame] = {}

    for i, (name, p, col) in enumerate(zip(names, prob_list, colors)):
        stats = _bin_stats(p, y_bin, w, edges, error_bars, z)
        ece = float(np.nansum(stats["ece"]))
        br = _brier(p, y_bin, w)

        df = pd.DataFrame(
            {
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "bin_center": centers,
                "n": stats["n"],
                "w_sum": stats["wsum"],
                "p_mean": stats["pmean"],
                "y_rate": stats["yrate"],
                "y_low": stats["ylo"],
                "y_high": stats["yhi"],
                "ece_contrib": stats["ece"],
            }
        )
        per_model[name] = df

        valid = df["w_sum"].to_numpy() > 0
        x = df.loc[valid, "p_mean"].to_numpy()
        y = df.loc[valid, "y_rate"].to_numpy()
        ylo = df.loc[valid, "y_low"].to_numpy()
        yhi = df.loc[valid, "y_high"].to_numpy()

        if error_bars.lower() != "none":
            yerr = np.vstack([y - ylo, yhi - y])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=col,
                elinewidth=1.0,
                capsize=2,
                alpha=alpha * 0.85,
            )

        ax.scatter(x, y, c=[col], s=s, marker=marker, alpha=alpha)
        if connect and len(x) > 1:
            ax.plot(x, y, color=col, linewidth=linewidth, alpha=alpha)

        label = name
        pieces = []
        if show_ece:
            pieces.append(f"ECE={ece:.3f}")
        if show_brier:
            pieces.append(f"Brier={br:.3f}")
        if pieces:
            label = f"{label} ({', '.join(pieces)})"

        ax.plot(
            [],
            [],
            color=col,
            marker=marker,
            linestyle="-" if connect else "None",
            linewidth=linewidth,
            label=label,
        )

        if axb is not None:
            bw = edges[1:] - edges[:-1]
            slot = bw * 0.8 / max(1, len(prob_list))
            left = edges[:-1] + i * slot
            vals = df["w_sum"].to_numpy()
            if counts_norm == "fraction":
                denom = vals.sum() if vals.sum() > 0 else 1.0
                vals = vals / denom
            axb.bar(
                left,
                vals,
                width=slot,
                align="edge",
                color=col,
                alpha=counts_alpha,
                label=name,
            )

    # -------------- format axes -------------- #
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if show_diagonal:
        diag_kw = {
            "color": "gray",
            "linestyle": "--",
            "linewidth": 1.2,
            "alpha": 0.9,
        }
        if diagonal_kwargs:
            _assert_all_types(diagonal_kwargs, dict, objname="'diagonal_kwargs'")
            diag_kw.update(diagonal_kwargs)
        ax.plot((0.0, 1.0), (0.0, 1.0), **diag_kw)

    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    if legend:
        ax.legend(loc=legend_loc)

    if axb is not None:
        axb.set_xlim(*xlim)
        axb.axhline(0, color="gray", lw=0.8)
        axb.set_xlabel(xlabel or "Predicted probability")
        axb.set_ylabel("Frac." if counts_norm == "fraction" else "Count")
        set_axis_grid(axb, show_grid=True, grid_props={"alpha": 0.25})
        h, l = axb.get_legend_handles_labels()
        if h and l:
            axb.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if savefig:
        try:
            fig.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        plt.show()

    if return_data:
        return ax, per_model
    return ax


# ------------------ helpers ------------------ #
def _z_from_conf(cf: float) -> float:
    table = {
        0.80: 1.2815515655,
        0.90: 1.6448536269,
        0.95: 1.9599639845,
        0.975: 2.241402728,
        0.99: 2.5758293035,
    }
    return table.get(round(cf, 3), 1.9599639845)


def _to_prob_vector(arr: np.ndarray, ci: int | None) -> np.ndarray:
    if arr.ndim == 1:
        return arr.astype(float, copy=False)
    if arr.ndim == 2:
        idx = arr.shape[1] - 1 if ci is None else ci
        if idx < 0 or idx >= arr.shape[1]:
            raise ValueError(
                "class_index out of bounds for 2D predictions: "
                f"{idx} not in [0, {arr.shape[1]-1}]"
            )
        return arr[:, idx].astype(float, copy=False)
    raise ValueError(
        "Predictions must be 1D probabilities or " "(n_samples, n_classes) arrays."
    )


def _prep_probs(
    p: np.ndarray, clip_lo: float, clip_hi: float, do_norm: bool
) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if do_norm:
        pmin, pmax = np.nanmin(p), np.nanmax(p)
        if (pmin < -1e-9) or (pmax > 1.0 + 1e-9):
            rng = pmax - pmin
            if rng > 1e-12:
                p = (p - pmin) / rng
    p = np.clip(p, clip_lo, clip_hi)
    return p


def _build_bins(
    probs_list: list[np.ndarray], nb: int, strat: str, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    if strat == "uniform":
        edges = np.linspace(low, high, nb + 1)
    else:
        allp = np.concatenate(probs_list)
        q = np.linspace(0.0, 1.0, nb + 1)
        edges = np.quantile(allp, q)
        edges = np.unique(edges)
        if len(edges) - 1 < nb:
            warnings.warn(
                ("Not enough unique quantile edges; " "falling back to uniform bins."),
                UserWarning,
                stacklevel=2,
            )
            edges = np.linspace(low, high, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _bin_stats(
    p: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    edges: np.ndarray,
    ebars: str,
    zval: float,
) -> dict[str, np.ndarray]:
    nb = len(edges) - 1
    eps = 1e-12
    idx = np.digitize(p, edges, right=False) - 1
    idx[idx < 0] = 0
    idx[idx >= nb] = nb - 1

    n = np.zeros(nb, dtype=float)
    wsum = np.zeros(nb, dtype=float)
    pmean = np.zeros(nb, dtype=float)
    yr = np.zeros(nb, dtype=float)

    for b in range(nb):
        m = idx == b
        if not np.any(m):
            continue
        ww = w[m]
        pp = p[m]
        yy = y[m]
        wsum[b] = ww.sum()
        n[b] = float(m.sum())
        denom = max(wsum[b], eps)
        pmean[b] = float(np.dot(ww, pp) / denom)
        yr[b] = float(np.dot(ww, yy) / denom)

    if ebars.lower() == "none":
        ylo = np.full_like(yr, np.nan)
        yhi = np.full_like(yr, np.nan)
    elif ebars.lower() == "normal":
        neff = np.maximum(wsum, eps)
        se = np.sqrt(np.clip(yr * (1.0 - yr) / neff, 0.0, 1.0))
        ylo = np.clip(yr - zval * se, 0.0, 1.0)
        yhi = np.clip(yr + zval * se, 0.0, 1.0)
    else:
        neff = np.maximum(wsum, eps)
        ylo = np.empty_like(yr)
        yhi = np.empty_like(yr)
        for i in range(nb):
            ph = yr[i]
            n_ = neff[i]
            if n_ <= eps:
                ylo[i] = np.nan
                yhi[i] = np.nan
                continue
            denom = 1.0 + (zval**2) / n_
            center = (ph + (zval**2) / (2.0 * n_)) / denom
            rad = (
                zval * np.sqrt((ph * (1.0 - ph) + (zval**2) / (4.0 * n_)) / n_)
            ) / denom
            ylo[i] = np.clip(center - rad, 0.0, 1.0)
            yhi[i] = np.clip(center + rad, 0.0, 1.0)

    totw = max(w.sum(), eps)
    wbin = wsum / totw
    ece_contrib = wbin * np.abs(yr - pmean)

    return {
        "n": n,
        "wsum": wsum,
        "pmean": pmean,
        "yrate": yr,
        "ylo": ylo,
        "yhi": yhi,
        "wbin": wbin,
        "ece": ece_contrib,
    }


def _brier(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(np.average((p - y) ** 2, weights=w))


def _colors(cmap_name: str, palette: list[Any] | None, k: int) -> list[Any]:
    if palette is not None:
        return [palette[i % len(palette)] for i in range(k)]
    try:
        cmo = get_cmap(cmap_name)
    except ValueError:
        warnings.warn(
            f"Invalid cmap '{cmap_name}'. Using 'tab10' instead.",
            UserWarning,
            stacklevel=2,
        )
        cmo = get_cmap("tab10")
    if hasattr(cmo, "colors") and len(cmo.colors) >= k:
        return list(cmo.colors[:k])
    if k == 1:
        return [cmo(0.5)]
    return [cmo(i / (k - 1)) for i in range(k)]


plot_reliability_diagram.__doc__ = r"""\
Plot a reliability diagram (calibration plot) for one or more
classification models.

This compares **predicted probabilities** to **observed
frequencies** across bins of predicted probability. Perfect
calibration lies on the diagonal :math:`y=x`.

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground truth labels. For binary calibration, values are
    compared to ``positive_label`` after validation and
    flattening.

*y_preds : array-like(s)
    One or more model predictions. Each item may be:
    - 1D array of positive-class probabilities in ``[0, 1]``.
    - 2D array of shape ``(n_samples, n_classes)``; use
      ``class_index`` to select a column. If omitted, the
      last column is used.

names : list of str, optional
    Labels for each model curve. If fewer names are provided
    than models, placeholders like ``'Model_1'`` are appended.

sample_weight : array-like of shape (n_samples,), optional
    Per-sample weights used for observed frequencies, ECE,
    and Brier score. If ``None``, equal weights are used.

n_bins : int, default=10
    Number of probability bins.

strategy : {'uniform', 'quantile'}, default='uniform'
    Binning strategy.
    - ``'uniform'``: equally spaced edges in ``[0, 1]``.
    - ``'quantile'``: edges are empirical quantiles of the
      pooled predictions. If edges are not unique, the method
      falls back to uniform binning with a warning.

positive_label : int or float or str, default=1
    Label in ``y_true`` treated as the positive class when
    constructing the binary target.

class_index : int, optional
    Column index to pick from 2D probability arrays. If
    omitted, the last column is used.

clip_probs : tuple of (float, float), default=(0.0, 1.0)
    Inclusive clipping range applied to predictions. A warning
    is issued if clipping occurs.

normalize_probs : bool, default=True
    If ``True``, attempts to linearly rescale predictions into
    ``[0, 1]`` when minor out-of-range values are detected,
    then applies clipping.

error_bars : {'wilson', 'normal', 'none'}, default='wilson'
    Per-bin uncertainty for observed frequencies.
    - ``'wilson'``: Wilson interval using ``conf_level``.
    - ``'normal'``: normal approximation.
    - ``'none'``: no error bars.

conf_level : float, default=0.95
    Confidence level used for error bars when applicable.

show_diagonal : bool, default=True
    Draw the reference diagonal :math:`y=x`.

diagonal_kwargs : dict, optional
    Matplotlib keyword arguments for the diagonal reference
    line (e.g., ``linestyle``, ``color``).

show_ece : bool, default=True
    Compute Expected Calibration Error (ECE) and append a
    summary to each model label.

show_brier : bool, default=True
    Compute (weighted) Brier score and append a summary to
    each model label.

counts_panel : {'none', 'bottom'}, default='bottom'
    If not ``'none'``, draw a compact histogram below the main
    panel that shows per-bin totals for each model.

counts_norm : {'fraction', 'count'}, default='fraction'
    Normalization for the counts panel. ``'fraction'`` divides
    by the total weight; ``'count'`` shows raw weighted sums.

counts_alpha : float, default=0.35
    Alpha for bars in the counts panel.

figsize : tuple of (float, float), default=(9, 7)
    Figure size for the layout. When ``counts_panel='bottom'``,
    a two-row gridspec is used.

title : str, optional
    Title for the plot. If ``None``, no title is set.

xlabel : str, optional
    Label for the x-axis. Defaults to
    ``'Predicted probability'``.

ylabel : str, optional
    Label for the y-axis. Defaults to
    ``'Observed frequency'``.

cmap : str, default='tab10'
    Matplotlib colormap name used to generate model colors.

color_palette : list, optional
    Explicit list of colors. When provided, colors are cycled
    from this list instead of the colormap.

marker : str, default='o'
    Marker used for the bin points.

s : int, default=40
    Marker size for the bin points.

linewidth : float, default=2.0
    Line width used when connecting bin points.

alpha : float, default=0.9
    Alpha for points and lines in the main panel.

connect : bool, default=True
    Connect bin points with a line for each model.

legend : bool, default=True
    Display a legend. Summary metrics (ECE, Brier) are shown
    next to model names when enabled.

legend_loc : str, default='best'
    Legend location passed to Matplotlib.

show_grid : bool, default=True
    Toggle gridlines via the package helper ``set_axis_grid``.

grid_props : dict, optional
    Keyword arguments passed to ``set_axis_grid`` for grid
    customization (e.g., ``linestyle``, ``alpha``).

xlim : tuple of (float, float), default=(0.0, 1.0)
    X-axis limits.

ylim : tuple of (float, float), default=(0.0, 1.0)
    Y-axis limits.

savefig : str, optional
    If provided, save the figure to this path; otherwise the
    plot is shown interactively.

return_data : bool, default=False
    If ``True``, return ``(ax, data_dict)`` where values are
    per-model ``pandas.DataFrame`` objects with per-bin stats:
    ``['bin_left', 'bin_right', 'bin_center', 'n', 'w_sum',
    'p_mean', 'y_rate', 'y_low', 'y_high', 'ece_contrib']``.
    Otherwise, return only the Matplotlib axes.

Returns
-------
ax : matplotlib.axes.Axes
    Axes of the main calibration plot. When
    ``counts_panel='bottom'``, the second axes (counts panel)
    is not returned.

Notes
-----
Calibration compares *confidence* to *accuracy* within bins.
For bin :math:`b`, let :math:`\hat{p}_i` be predictions and
:math:`y_i\in\{0,1\}` be binary targets with weights
:math:`w_i\ge 0`. Define the weighted bin mean probability
and accuracy as

.. math::

   \bar{p}_b \;=\;
   \frac{\sum_{i\in b} w_i \hat{p}_i}
        {\sum_{i\in b} w_i},
   \qquad
   \bar{y}_b \;=\;
   \frac{\sum_{i\in b} w_i y_i}
        {\sum_{i\in b} w_i}.

The Expected Calibration Error (ECE) is

.. math::

   \mathrm{ECE}
   \;=\;
   \sum_b
   \left(
     \frac{\sum_{i\in b} w_i}{\sum_i w_i}
   \right)
   \left|
     \bar{y}_b - \bar{p}_b
   \right|.

The (weighted) Brier score is

.. math::

   \mathrm{Brier}
   \;=\;
   \frac{\sum_i
     w_i \left(\hat{p}_i - y_i\right)^2}
        {\sum_i w_i}.

Wilson confidence intervals for :math:`\bar{y}_b` use
:math:`z = \Phi^{-1}\!\left(\tfrac{1+\alpha}{2}\right)` and
effective count :math:`n_b=\sum_{i\in b} w_i`:

.. math::

   \mathrm{center}
   \;=\;
   \frac{\bar{y}_b + \frac{z^2}{2 n_b}}
        {1 + \frac{z^2}{n_b}},
   \qquad
   \mathrm{radius}
   \;=\;
   \frac{z}{1 + \frac{z^2}{n_b}}
   \sqrt{\frac{\bar{y}_b(1-\bar{y}_b)}{n_b}
         + \frac{z^2}{4 n_b^2}}.

The interval is
:math:`[\mathrm{center}-\mathrm{radius},
\mathrm{center}+\mathrm{radius}]`,
clipped to ``[0, 1]``. The normal interval replaces the term
with the usual standard error
:math:`\sqrt{\bar{y}_b(1-\bar{y}_b)/n_b}`.

When ``strategy='quantile'``, bin edges are the empirical
quantiles of the pooled predictions. If many identical values
exist, edges can collapse; in that case, the function falls
back to uniform edges with a warning.

Examples
--------
Binary example with quantile bins and Wilson intervals.

>>> import numpy as np
>>> from kdiagram.plot.comparison import \
...     plot_reliability_diagram
>>> rng = np.random.default_rng(0)
>>> y = (rng.random(1000) < 0.4).astype(int)
>>> p1 = 0.4 * np.ones_like(y) + 0.15 * rng.random(len(y))
>>> p2 = 0.4 * np.ones_like(y) + 0.05 * rng.random(len(y))
>>> ax = plot_reliability_diagram(
...     y, p1, p2,
...     names=['Wide', 'Tight'],
...     n_bins=12,
...     strategy='quantile',
...     error_bars='wilson',
...     counts_panel='bottom',
...     show_ece=True,
...     show_brier=True,
...     title=('Reliability Diagram '
...            '(Quantile bins + Wilson CIs)'),
... )
"""


@validate_params(
    {
        "train_times": ["array-like", None],
        "metrics": [str, "array-like", callable, None],
        "scale": [
            StrOptions(
                {
                    "norm",
                    "min-max",
                    "std",
                    "standard",
                }
            ),
            None,
        ],
        "lower_bound": [Real],
    }
)
def plot_model_comparison(
    y_true,
    *y_preds,
    train_times: float | list[float] | None = None,
    metrics: str | Callable | list[str | Callable] | None = None,
    names: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    colors: list[Any] | None = None,
    alpha: float = 0.7,
    legend: bool = True,
    show_grid: bool = True,
    grid_props: dict = None,
    scale: str | None = "norm",
    lower_bound: float = 0,
    savefig: str | None = None,
    loc: str = "upper right",
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
        y_true, *y_preds = drop_nan_in(y_true, *y_preds, error="raise")
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
        warnings.warn("No prediction arrays (*y_preds) provided.", stacklevel=2)
        return None  # Cannot plot without predictions

    # --- Handle Names ---
    if names is None:
        names = [f"Model_{i+1}" for i in range(n_models)]
    else:
        names = columns_manager(list(names), empty_as_none=False)  # Ensure list
        if len(names) < n_models:
            names += [f"Model_{i+1}" for i in range(len(names), n_models)]
        elif len(names) > n_models:
            warnings.warn(
                f"Received {len(names)} names for {n_models}"
                f" models. Extra names ignored.",
                UserWarning,
                stacklevel=2,
            )
            names = names[:n_models]

    # --- Handle Metrics ---
    if metrics is None:
        target_type = type_of_target(y_true)
        if target_type in ["continuous", "continuous-multioutput"]:
            # Default regression metrics
            metrics = ["r2", "mae", "mape", "rmse"]
        else:
            # Default classification metrics
            metrics = ["accuracy", "precision", "recall", "f1"]
        if verbose >= 1:
            print(
                f"[INFO] Auto-selected metrics for target type "
                f"'{target_type}': {metrics}"
            )

    metrics = is_iterable(metrics, exclude_string=True, transform=True)

    metric_funcs = []
    metric_names = []
    error_metrics = []  # Track metrics needing sign inversion

    for metric in metrics:
        try:
            if isinstance(metric, str):
                # get_scorer returns a callable scorer object
                scorer_func = get_scorer(metric)
                metric_funcs.append(scorer_func)
                metric_names.append(metric)
                # Identify error metrics (lower is better) for potential scaling flip
                if metric in ["mae", "mape", "rmse", "mse"]:  # Add others if needed
                    error_metrics.append(metric)
            elif callable(metric):
                metric_funcs.append(metric)
                m_name = getattr(metric, "__name__", f"func_{len(metric_names)}")
                metric_names.append(m_name)
                # Cannot easily determine if callable is error/score metric
            else:
                warnings.warn(
                    f"Ignoring invalid metric type: {type(metric)}", stacklevel=2
                )
        except Exception as e:
            warnings.warn(
                f"Could not retrieve scorer for metric '{metric}': {e}", stacklevel=2
            )

    if not metric_funcs:
        raise ValueError("No valid metrics found or specified.")

    # --- Handle Train Times ---
    train_time_vals = None
    if train_times is not None:
        if isinstance(train_times, (int, float, np.number)):  # Handle single value
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
        metric_names.append("Train Time (s)")  # Use clearer name
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
                    warnings.warn(
                        f"Could not compute metric "
                        f"'{metric_names[j]}' for model "
                        f"'{names[i]}': {e}. Setting to NaN.",
                        stacklevel=2,
                    )
                    results[i, j] = np.nan
            else:
                results[i, j] = np.nan  # Should not happen if logic is correct

    # --- Scale Results ---
    # Make copy for scaling to preserve original results if needed later
    results_scaled = results.copy()

    # Handle potential NaNs before scaling
    if np.isnan(results_scaled).any():
        warnings.warn(
            "NaN values found in metric results. Scaling might "
            "be affected or rows/cols dropped depending on method.",
            stacklevel=2,
        )
        # Option 1: Impute (e.g., with column mean) - complex
        # Option 2: Use nan-aware numpy functions
        # Let's use nan-aware functions

    # Note: Some metrics are better when *lower* (MAE, RMSE, MAPE, train_time).
    # For visualization where larger radius is better, we might invert these
    # before scaling, or adjust the interpretation. Let's scale first.
    if scale in ["norm", "min-max"]:
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

    elif scale in ["std", "standard"]:
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
    fig = plt.figure(figsize=figsize or (8, 8))  # Default figsize here
    ax = fig.add_subplot(111, polar=True)

    # Angles for each metric axis
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles_closed = angles + angles[:1]  # Repeat first angle to close plot

    # Colors
    if colors is None:
        # Use a robust colormap like tab10 if available
        try:
            cmap_obj = get_cmap("tab10")
            plot_colors = [cmap_obj(i % 10) for i in range(n_models)]
        except ValueError:  # Fallback if tab10 not found (unlikely)
            cmap_obj = get_cmap("viridis")
            plot_colors = [cmap_obj(i / n_models) for i in range(n_models)]
    else:
        plot_colors = colors  # Use user-provided list

    # Plot each model
    for i, row in enumerate(results_scaled):
        values = np.concatenate((row, [row[0]]))  # Close the polygon
        color = plot_colors[i % len(plot_colors)]  # Cycle colors if needed
        ax.plot(
            angles_closed,
            values,
            label=names[i],
            color=color,
            linewidth=1.5,
            alpha=alpha,
        )
        ax.fill(angles_closed, values, color=color, alpha=0.1)  # Lighter fill

    # --- Configure Axes ---
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_names)

    # Adjust radial limits and labels
    # If scaled to [0, 1], set limit slightly above 1
    # If std scaled, auto-limit might be better, but respect lower_bound
    if scale in ["norm", "min-max"]:
        ax.set_ylim(bottom=lower_bound, top=1.05)
        # Optional: Add radial ticks for [0, 1] scale
        ax.set_yticks(np.linspace(lower_bound, 1, 5))
    else:  # Raw or std scaled
        ax.set_ylim(bottom=lower_bound)
        # Let matplotlib auto-determine upper limit and ticks

    ax.tick_params(axis="y", labelsize=8)  # Smaller radial labels
    ax.tick_params(axis="x", pad=10)  # Pad angular labels outwards

    # Grid
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    # Legend
    if legend:
        ax.legend(loc=loc, bbox_to_anchor=(1.25, 1.05))  # Adjust position

    # Title
    ax.set_title(title or "Model Performance Comparison", y=1.15, fontsize=14)

    # --- Output ---
    plt.tight_layout(pad=2.0)  # Adjust layout

    if savefig:
        try:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig}")
        except Exception as e:
            print(f"Error saving plot to {savefig}: {e}")
    else:
        try:
            plt.show()
        except Exception as e:
            warnings.warn(
                f"Could not display plot interactively ({e})."
                f" Use savefig parameter.",
                UserWarning,
                stacklevel=2,
            )

    return ax
