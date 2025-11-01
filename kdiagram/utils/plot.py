#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scipy.stats import gaussian_kde

from ..api.typing import Acov
from ..compat.matplotlib import get_cmap
from .generic_utils import get_valid_kwargs
from .handlers import columns_manager


def set_axis_grid(
    ax, show_grid: bool = True, grid_props: dict = None
) -> None:
    """Robustly set grid properties on one or more matplotlib axes."""
    # Ensure grid_props is a dictionary.
    grid_props = (
        grid_props.copy()
        if grid_props is not None
        else {"linestyle": ":", "alpha": 0.7}
    )

    props = dict(grid_props or {})
    props.pop("visible", None)

    axes = ax if isinstance(ax, (list, tuple)) else [ax]

    which = props.pop("which", "both")
    axis = props.pop("axis", "both")

    for a in axes:
        if show_grid:
            # Turn on both major & minor. Apply props only when enabling.
            a.grid(True, which=which, axis=axis, **props)
        else:
            # Turn off both major & minor.
            a.grid(False, which=which, axis=axis)
            # And force-hide any already-created grid line artists.
            for gl in a.get_xgridlines() + a.get_ygridlines():
                gl.set_visible(False)


def _place_polar_ylabel(
    ax: Axes,
    text: str,
    *,
    angle: float = 225,
    x_in: float | None = None,
    y_in: float = 0.5,
    min_inside: float = 0.028,
    tight_safe: bool = True,
    labelpad: int =32, 
) -> tuple[float, float]:
    """
    Place the polar *radial* label inside the axes to avoid crowding/clipping.

    This is especially useful in multi-panel layouts where the default
    y-label (drawn outside the left edge) can overlap or be cropped,
    particularly when saving with ``bbox_inches='tight'``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A polar axes. If a non-polar axes is passed, a warning is issued
        and the function returns without re-positioning.
    text : str
        The label text to set.
    angle : float, default=225
        Quadrant for radial *tick labels* (not the y-label itself). Using
        225° keeps tick labels away from the title/top.
    x_in : float, optional
        X position of the y-label in *axes coordinates* (0 = left edge,
        1 = right). If omitted, a heuristic is used based on the subplot
        width to keep the label comfortably inside.
    y_in : float, default=0.5
        Y position of the y-label in axes coordinates.
    min_inside : float, default=0.028
        Smallest x-offset to keep even very wide subplots from placing the
        label too close to the edge.
    tight_safe : bool, default=True
        If True, the label is always placed inside the axes so that saving
        with ``bbox_inches='tight'`` will not crop it.

    Returns
    -------
    (x, y) : tuple of float
        The axes-coordinate position actually used for the label.

    Notes
    -----
    • This function does **not** add a figure-level label. If you want a
      single shared label across subplots, use ``fig.supylabel(...)`` and
      do **not** call this helper for per-axes labels.
    """
    # Basic safety checks
    try:
        is_polar = getattr(ax, "name", "") == "polar"
    except Exception:  # very defensive
        is_polar = False

    if not is_polar:
        warnings.warn(
            "_place_polar_ylabel was given a non-polar Axes; "
            "leaving the label at its default location.",
            stacklevel=2,
        )
        ax.set_ylabel(text, labelpad =labelpad)
        return (0.0, 0.5)

    # Set the text and move radial tick labels away from the top/title.
    ax.set_ylabel(text, labelpad =labelpad)
    ax.set_rlabel_position(angle)

    # Heuristic: narrower axes → push label further in.
    bb = ax.get_position()  # figure-fraction bbox
    if x_in is None:
        w = float(bb.width)
        if w < 0.33:
            x_in = 0.085
        elif w < 0.42:
            x_in = 0.060
        elif w < 0.50:
            x_in = 0.042
        else:
            x_in = 0.033
        x_in = max(x_in, min_inside)

    # If user really wants the default outside placement, they can call
    # ax.set_ylabel(...) themselves and skip this helper. When tight_safe=True,
    # we *always* keep the label inside the axes to be safe with
    # bbox_inches='tight'.
    if tight_safe:
        ax.yaxis.set_label_coords(x_in, y_in, transform=ax.transAxes)
    else:
        # Still place inside, but allow very small offsets if caller insists.
        ax.yaxis.set_label_coords(max(x_in, 0.0), y_in, transform=ax.transAxes)

    return (x_in, y_in)


def validate_kind(kind: str | None, *, default: str = "polar") -> str:
    """
    Normalize and validate the 'kind' switch used by plotting functions.

    Returns a lower-cased string. Raises ValueError with a stable message
    if invalid (used by tests across plots).
    """
    k = kind or default
    if not isinstance(k, str):
        raise ValueError("kind must be 'polar' or 'cartesian'.")
    k = k.lower()
    if k not in {"polar", "cartesian"}:
        raise ValueError("kind must be 'polar' or 'cartesian'.")
    return k


def maybe_delegate_cartesian(
    kind: str, cartesian_fn: Callable[..., Axes], /, *args, **kwargs
) -> Axes | None:
    """
    If kind == 'cartesian', call the provided cartesian renderer and return
    its Axes; otherwise return None so the caller can proceed with polar.
    """
    if validate_kind(kind) == "cartesian":
        return cartesian_fn(*args, **kwargs)
    return None


def is_valid_kind(
    kind: str,
    valid_kinds: list[str] | None = None,
    error: str = "raise",
) -> str:
    r"""
    Normalizes and validates plot type specifications,
    handling aliases and suffixes.

    Parameters:
        kind (str): Input plot type specification (flexible formatting).
        valid_kinds (Optional[List[str]]):
            Acceptable plot types to validate against.
        error (str): Error handling mode
        ('raise' to raise errors, others to return normalized kind).

    Returns:
        str: Normalized canonical plot type or custom kind.

    Raises:
        ValueError: If invalid plot type is provided and `error` is 'raise`.
    """
    SUFFIXES = ("plot", "graph", "chart", "diagram", "visual")

    # Expanded alias mappings
    KIND_ALIASES = {
        "boxplot": "box",
        "boxgraph": "box",
        "boxchart": "box",
        "plotbox": "box",
        "box_plot": "box",
        "violinplot": "violin",
        "violingraph": "violin",
        "violinchart": "violin",
        "violin_plot": "violin",
        "scatterplot": "scatter",
        "scattergraph": "scatter",
        "scatterchart": "scatter",
        "lineplot": "line",
        "linegraph": "line",
        "linechart": "line",
        "barchart": "bar",
        "bargraph": "bar",
        "barplot": "bar",
        "plotbar": "bar",
        "histogram": "hist",
        "histplot": "hist",
        "heatmap": "heatmap",
        "heat_map": "heatmap",
        "plotdensity": "density",
        "plot_density": "density",
        "densityplot": "density",
        "densitygraph": "density",
        "areachart": "area",
        "areagraph": "area",
    }

    # Canonical regex patterns (match anywhere in string)
    CANONICAL_PATTERNS = {
        "box": re.compile(r"box"),
        "violin": re.compile(r"violin"),
        "scatter": re.compile(r"scatter"),
        "line": re.compile(r"line"),
        "bar": re.compile(r"bar"),
        "hist": re.compile(r"hist"),
        "heatmap": re.compile(r"heatmap"),
        "density": re.compile(r"density"),
        "area": re.compile(r"area"),
    }

    def normalize(k: str) -> str:
        """Normalize input: clean, lowercase, remove suffixes."""
        # Remove non-alphanumeric chars and underscores
        k_clean = re.sub(r"[\W_]+", "", k.strip().lower())
        # Remove suffixes from the end
        for suffix in SUFFIXES:
            if k_clean.endswith(suffix):
                k_clean = k_clean[: -len(suffix)]
                break
        return k_clean

    normalized = normalize(kind)

    # 1. Check exact aliases
    canonical = KIND_ALIASES.get(normalized)

    # 2. Search for canonical patterns if no alias found
    if not canonical:
        for pattern, regex in CANONICAL_PATTERNS.items():
            if regex.search(normalized):
                canonical = pattern
                break

    final_kind = canonical if canonical else normalized

    # Validation against allowed kinds
    if valid_kinds is not None:
        # Normalize valid kinds using same rules
        valid_normalized = {normalize(k): k for k in valid_kinds}
        final_normalized = normalize(final_kind)

        # Check matches against original valid kinds or their normalized forms
        valid_match = False
        for valid_norm, orig_kind in valid_normalized.items():
            if (
                final_normalized == valid_norm
                or final_normalized == normalize(orig_kind)
            ):
                valid_match = True
                break

        if not valid_match and error == "raise":
            allowed = ", ".join(f"'{k}'" for k in valid_kinds)
            raise ValueError(
                f"Invalid plot type '{kind}'. Allowed: {allowed}"
            )

    return final_kind


def prepare_data_for_kde(
    data: np.ndarray, bandwidth: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Prepares the data for Kernel Density Estimate (KDE) calculation.

    This function cleans the data by removing non-finite values, then creates
    a grid for KDE evaluation. It then computes the Kernel Density Estimate
    using a Gaussian kernel.

    Parameters
    ----------
    data : np.ndarray
        The data to calculate the KDE from.
    bandwidth : float, optional
        Bandwidth for the KDE. Default is None, which uses Silverman's rule
        of thumb for bandwidth estimation.

    Returns
    -------
    grid : np.ndarray
        The x-values grid for KDE evaluation.
    pdf : np.ndarray
        The estimated probability density function (PDF) from the KDE.

    Notes
    -----
    The KDE is computed by finding the optimal bandwidth (if not provided)
    and applying a Gaussian kernel to the data. The bandwidth is chosen
    using Silverman's rule of thumb when not explicitly provided.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> grid, pdf = prepare_data_for_kde(data)
    >>> print(grid, pdf)

    See Also
    --------
    _kde_pdf : Helper function that calculates the KDE.

    References
    ----------
    [1] Silverman, B. W. (1986). Density Estimation for Statistics and Data
        Analysis. CRC Press.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError("`data` is empty after removing non-finite values.")

    lo, hi = np.min(data), np.max(data)
    grid = np.linspace(lo, hi, 512)

    pdf = _kde_pdf(data, grid, bandwidth=bandwidth)

    return grid, pdf


def _kde_pdf(
    x: np.ndarray, grid: np.ndarray, bandwidth: float | None = None
) -> np.ndarray:
    r"""
    Helper function to calculate the Kernel Density Estimate (KDE).

    This function computes the KDE for a given dataset and returns the
    probability density function (PDF) evaluated at the grid points.

    Parameters
    ----------
    x : np.ndarray
        The data values to estimate the KDE from.
    grid : np.ndarray
        The grid points at which to evaluate the KDE.
    bandwidth : float, optional
        Bandwidth for the KDE. Default is None, which uses Silverman's rule
        of thumb.

    Returns
    -------
    np.ndarray
        The estimated probability density function evaluated at the grid points.

    Notes
    -----
    Silverman's rule of thumb is used to calculate the bandwidth if it is
    not provided. The `gaussian_kde` function from SciPy is then used to
    estimate the probability density function.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> grid = np.linspace(-5, 5, 100)
    >>> pdf = _kde_pdf(data, grid)
    >>> print(pdf)

    References
    ----------
    [1] Silverman, B. W. (1986). Density Estimation for Statistics and Data
        Analysis. CRC Press.
    """

    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.zeros_like(grid)

    if bandwidth is None:
        std = np.std(x, ddof=1) if x.size > 1 else 1.0
        bandwidth = 1.06 * std * (x.size ** (-1 / 5)) if std > 0 else 1.0

    kde = gaussian_kde(
        x,
        bw_method=(
            bandwidth / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else None
        ),
    )

    return kde(grid)


def normalize_pdf(pdf: np.ndarray) -> np.ndarray:
    r"""
    Normalize the probability density function (PDF) to [0, 1].

    This function scales the PDF so that the maximum value is 1. It ensures
    that the PDF is normalized, making it suitable for visual comparison.

    Parameters
    ----------
    pdf : np.ndarray
        The probability density function to normalize.

    Returns
    -------
    np.ndarray
        The normalized probability density function.

    Notes
    -----
    The PDF is normalized by dividing by the maximum value to scale the
    entire distribution to the [0, 1] range.

    Examples
    --------
    >>> pdf = np.random.normal(0, 1, 1000)
    >>> normalized_pdf = normalize_pdf(pdf)
    >>> print(normalized_pdf)

    See Also
    --------
    prepare_data_for_kde : Function to prepare data and compute the PDF.
    """
    pdf_normalized = pdf.copy()
    if np.max(pdf_normalized) > 0:
        pdf_normalized = pdf_normalized / np.max(pdf_normalized)

    return pdf_normalized


def setup_plot_axes(
    figsize: tuple[float, float] = (8, 6),
    title: str = "",
    x_label: str = "Value",
    y_label: str = "Density",
) -> plt.Axes:
    r"""
    Setup and return the axes for the plot with common formatting.

    This function initializes a new plot with a title, and labels for the
    x-axis and y-axis, and returns the axes object for further customizations.

    Parameters
    ----------
    figsize : tuple of float, optional
        Size of the figure. Default is (8, 6).
    title : str, optional
        Title of the plot. Default is an empty string.

    Returns
    -------
    plt.Axes
        The axes object for the plot.

    Examples
    --------
    >>> ax = setup_plot_axes(figsize=(10, 8), title="My Plot")
    >>> ax.set_xlabel("X Axis Label")
    >>> ax.set_ylabel("Y Axis Label")
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    return ax


def add_kde_to_plot(
    grid: np.ndarray,
    pdf: np.ndarray,
    ax: plt.Axes,
    color: str = "orange",
    line_width: float = 2,
) -> None:
    r"""
    Add the KDE line to the plot.

    This function adds the KDE line (calculated previously)
    to the provided axes object (`ax`).

    Parameters
    ----------
    grid : np.ndarray
        The grid values over which to evaluate the KDE.
    pdf : np.ndarray
        The probability density function (PDF) values.
    ax : plt.Axes
        The axes object to which the KDE line will be added.
    color : str, optional
        The color of the KDE line. Default is 'orange'.
    line_width : float, optional
        The line width for the KDE plot. Default is 2.

    Returns
    -------
    None

    Examples
    --------
    >>> add_kde_to_plot(grid, pdf, ax, color='red', line_width=3)
    """
    ax.plot(
        grid,
        pdf,
        color=color,
        linewidth=line_width,
        label="KDE",
        zorder=3,  # Ensure KDE is drawn on top
    )
    # ax.legend()


def _add_histogram_to_plot(
    data: np.ndarray,
    ax: plt.Axes,
    bins: int = 50,
    hist_color: str = "skyblue",
    hist_edge_color: str = "white",
    hist_alpha: float = 0.7,
    **hist_kws,
) -> None:
    r"""
    Add the histogram to the plot.

    This function adds a histogram to the plot, taking the data values
    and customizes the appearance of the histogram bars.

    Parameters
    ----------
    data : np.ndarray
        The data to plot the histogram from.
    ax : plt.Axes
        The axes object to which the histogram will be added.
    bins : int, optional
        The number of bins for the histogram. Default is 50.
    hist_color : str, optional
        The color of the histogram bars. Default is 'skyblue'.
    hist_edge_color : str, optional
        The color of the histogram bars' edges. Default is 'white'.
    hist_alpha : float, optional
        The transparency of the histogram bars. Default is 0.7.

    Returns
    -------
    None

    Examples
    --------
    >>> add_histogram_to_plot(data, ax, bins=40, hist_color='green')
    """
    label = hist_kws.pop("label", "Histogram")
    zorder = hist_kws.pop(
        "zorder", 2
    )  # Ensure histogram is drawn below the KDE line,
    # pop density if exist .
    hist_kws.pop("density", True)

    hist_kws = get_valid_kwargs(ax.hist, hist_kws)
    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=hist_alpha,
        color=hist_color,
        edgecolor=hist_edge_color,
        label=label,
        zorder=zorder,
        **hist_kws,
    )


def add_histogram_to_plot(
    data: np.ndarray,
    ax: plt.Axes,
    bins: int = 50,
    hist_color: str = "skyblue",
    hist_edge_color: str = "white",
    hist_alpha: float = 0.7,
    **hist_kws,
) -> None:
    r"""
    Add the histogram to the plot.

    This function adds a histogram to the plot, taking the data values
    and customizes the appearance of the histogram bars.

    Parameters
    ----------
    data : np.ndarray
        The data to plot the histogram from.
    ax : plt.Axes
        The axes object to which the histogram will be added.
    bins : int, optional
        The number of bins for the histogram. Default is 50.
    hist_color : str, optional
        The color of the histogram bars. Default is 'skyblue'.
    hist_edge_color : str, optional
        The color of the histogram bars' edges. Default is 'white'.
    hist_alpha : float, optional
        The transparency of the histogram bars. Default is 0.7.

    Returns
    -------
    None

    Examples
    --------
    >>> add_histogram_to_plot(data, ax, bins=40, hist_color='green')
    """
    # extract label/zorder (don’t pass twice)
    label = hist_kws.pop("label", "Histogram")
    zorder = hist_kws.pop("zorder", 2)

    # we set these explicitly below; remove if present in hist_kws
    for k in (
        "bins",
        "alpha",
        "color",
        "edgecolor",
        "density",
        "label",
        "zorder",
    ):
        hist_kws.pop(k, None)

    # keep only kwargs that ax.hist actually accepts
    hist_kws = get_valid_kwargs(ax.hist, hist_kws)

    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=hist_alpha,
        color=hist_color,
        edgecolor=hist_edge_color,
        label=label,
        zorder=zorder,
        **hist_kws,
    )


def setup_polar_plot(
    figsize: tuple[float, float] = (8, 8), title: str = ""
) -> plt.Axes:
    r"""
    Setup and return the polar axes for radial density plotting.

    This function initializes a polar plot with a specified title. The radial
    labels and angular grid lines are customizable as needed.

    Parameters
    ----------
    figsize : tuple of float, optional
        Size of the figure. Default is (8, 8).
    title : str, optional
        Title of the plot. Default is an empty string.

    Returns
    -------
    plt.Axes
        The polar axes object for the plot.

    Examples
    --------
    >>> ax = setup_polar_plot(figsize=(10, 8), title="Radial Density Ring")
    >>> ax.set_rlabel_position(90)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_title(title, fontsize=14)

    return ax


def _sample_colors(
    cm: str | Colormap | None,
    n: int,
    *,
    trim: float = 0.08,
    default: str = "tab10",
) -> list[tuple[float, float, float, float]]:
    r"""
    Return *n* visually separated colors from a colormap.

    The function accepts either a colormap **name** or a Matplotlib
    :class:`~matplotlib.colors.Colormap` instance.  For short listed
    palettes it spreads selections over the available swatches,
    instead of taking the first ones.  For continuous maps it samples
    the colormap on an evenly spaced grid, trimming the darkest and
    lightest extremes to keep lines readable on typical backgrounds.

    Parameters
    ----------
    cm : str or Colormap or None
        The input colormap.  If a string, it is resolved using
        :func:`get_cmap`.  If ``None``, the *default* colormap is
        used.
    n : int
        Number of colors to return.  Must be a positive integer.
    trim : float, optional
        Fraction of the continuous colormap to trim at each end.
        Values are clipped to ``[0.0, 0.49]``.  The default is
        ``0.08``, which avoids near-black and near-white tones.
    default : str, optional
        Fallback colormap name when *cm* is invalid or ``None``.
        Defaults to ``"tab10"``.

    Returns
    -------
    list of tuple of float
        A list of length *n*.  Each element is an ``(r, g, b, a)``
        tuple with components in ``[0, 1]``.

    Notes
    -----
    *Short listed palettes.*  Palettes such as ``tab10`` or ``tab20``
    are "listed colormaps".  Taking the first *k* colors can bunch
    hues.  This function selects indices that are evenly spaced across
    the palette.  If *n* exceeds the palette size, it cycles while
    maintaining the spread for the residual part.

    *Continuous colormaps.*  Perceptual maps like ``viridis`` or
    ``magma`` begin with very dark tones and may end very light.  The
    *trim* parameter keeps lines visible by avoiding those extremes.

    Examples
    --------
    >>> from kdiagram.utils.plot import _sample_colors
    >>> # 4 colors from a continuous map, trimmed at both ends
    >>> _sample_colors("magma", 4)
    [(...), (...), (...), (...)]

    >>> # 12 colors from a short listed palette, spaced across entries
    >>> _sample_colors("tab10", 12)
    [(...), ..., (...)]

    See Also
    --------
    matplotlib.colormaps : Registry of available colormaps.
    matplotlib.colors.Colormap : Base class for colormaps.

    References
    ----------
    .. [1] Matplotlib Colormap Guide.
           https://matplotlib.org/stable/users/explain/colors/colormaps.html
    .. [2] Smith, K., van der Walt, S. (2015).  A better default
           colormap for Matplotlib.
           https://bids.github.io/colormap/
    """
    # ---- validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("`n` must be a positive integer.")

    # Resolve to a Colormap object using the project helper.
    if isinstance(cm, Colormap):
        cmap_obj = cm
    else:
        cmap_obj = get_cmap(cm, default=default)

    # Defensive bounds for trim
    trim = float(np.clip(trim, 0.0, 0.49))

    # Listed colormap path: prefer even index spacing over "first N"
    if hasattr(cmap_obj, "colors") and isinstance(
        cmap_obj.colors, (list, tuple)
    ):
        colors = list(cmap_obj.colors)
        m = len(colors)

        if m == 0:
            # Fall back to continuous sampling if something is odd
            xs = np.linspace(0.0 + trim, 1.0 - trim, n)
            return [tuple(cmap_obj(float(x))) for x in xs]

        if n <= m:
            # Evenly spaced integer indices in [0, m-1]
            idx = np.floor(np.linspace(0, m - 1, n)).astype(int)
            return [tuple(colors[i]) for i in idx]

        # n > m: take full cycles, then distribute the remainder evenly
        q, r = divmod(n, m)
        out: list[tuple[float, float, float, float]] = []
        out.extend([tuple(c) for c in colors] * q)
        if r:
            idx = np.floor(np.linspace(0, m - 1, r)).astype(int)
            out.extend([tuple(colors[i]) for i in idx])
        return out

    # Continuous colormap path: sample evenly with end trimming
    if n == 1:
        xs = np.array([(1.0 - 2.0 * trim) * 0.5 + trim], dtype=float)
    else:
        xs = np.linspace(trim, 1.0 - trim, n)
    return [tuple(cmap_obj(float(x))) for x in xs]


def _setup_axes_for_reliability(
    ax: Axes | None,
    counts_panel: str,
    figsize: tuple[float, float] | None,
) -> tuple:
    """
    Return (fig, ax, axb) given an optional `ax` and `counts_panel` setting.

    - If `ax` is None:
        * counts_panel == "bottom": new Figure + GridSpec (main + bottom)
        * else:                      new Figure + single Axes
    - If `ax` is provided:
        * counts_panel == "bottom": append a bottom axis to the same figure
                                     (using axes_grid1 divider) that shares x.
        * else:                      reuse `ax` and no bottom axis.
    """
    if ax is None:
        if counts_panel == "bottom":
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.12)
            ax = fig.add_subplot(gs[0, 0])
            axb = fig.add_subplot(gs[1, 0], sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axb = None
    else:
        fig = ax.figure
        if counts_panel == "bottom":
            # Append a bottom panel below the provided `ax`
            try:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
            except ModuleNotFoundError as err:
                raise RuntimeError(
                    "counts_panel='bottom' with a provided `ax` requires "
                    "mpl_toolkits.axes_grid1 (install matplotlib's toolkits)."
                ) from err
            divider = make_axes_locatable(ax)
            # ~20% height for counts; tweak pad as you like
            axb = divider.append_axes(
                "bottom", size="20%", pad=0.25, sharex=ax
            )
        else:
            axb = None
    return fig, ax, axb


def canonical_acov(
    key: str | None,
    *,
    raise_on_invalid: bool = True,
    fallback: str = "default",
) -> str:
    """
    Normalize an acov alias to its canonical key.

    Accepts: "default"/"full", "half_circle"/"half",
             "quarter_circle"/"quarter", "eighth_circle"/"eighth".
    Hyphens/underscores/case/whitespace are ignored.
    """
    if key is None:
        return fallback

    norm = str(key).strip().lower().replace("-", "_")

    # map aliases -> canonical keys
    alias = {
        "default": "default",
        "full": "default",
        "full_circle": "default",
        "half": "half_circle",
        "half_circle": "half_circle",
        "quarter": "quarter_circle",
        "quarter_circle": "quarter_circle",
        "eighth": "eighth_circle",
        "eighth_circle": "eighth_circle",
    }

    canon = alias.get(norm)
    if canon is not None:
        return canon

    if raise_on_invalid:
        valid = "', '".join(
            ["default", "half_circle", "quarter_circle", "eighth_circle"]
        )
        raise ValueError(
            f"Invalid acov={key!r}. Use one of: '{valid}', "
            "or their aliases: 'full', 'half', 'quarter', 'eighth'."
        )

    return fallback


def resolve_polar_span(acov: Acov = "default") -> float:
    """
    Return angular span (radians) for a coverage keyword.
    Accepted aliases:
      - default: 'default', 'full', 'full-circle', 'full_circle'
      - half:    'half', 'half-circle', 'half_circle'
      - quarter: 'quarter', 'quarter-circle', 'quarter_circle'
      - eighth:  'eighth', 'eighth-circle', 'eighth_circle'
    """
    # normalize: lower, trim, hyphen/space -> underscore
    key = acov or "default"
    key = str(key).strip().lower().replace("-", "_").replace(" ", "_")

    canon = canonical_acov(acov, raise_on_invalid=False, fallback=key)
    # canon = alias.get(key, key)

    spans = {
        "default": 2 * np.pi,  # 360°
        "half_circle": 1 * np.pi,  # 180°
        "quarter_circle": 0.5 * np.pi,  # 90°
        "eighth_circle": 0.25 * np.pi,  # 45°
    }
    try:
        return spans[canon]
    except KeyError as e:
        valid = (
            "{'default|full', 'half_circle|half', "
            "'quarter_circle|quarter', 'eighth_circle|eighth'}"
        )
        raise ValueError(f"Invalid acov={acov!r}. Use one of {valid}.") from e


def setup_polar_axes(
    ax: Axes | None,
    *,
    acov: Acov = "default",
    figsize: tuple[float, float] | None = None,
    zero_at: Literal["N", "E", "S", "W"] = "N",  # where theta=0 points
    clockwise: bool = True,  # plot direction
) -> tuple[plt.Figure, Axes, float]:
    """
    Ensure we have a polar Axes configured to the requested angular span.

    Returns (fig, ax, span_radians).
    """
    span = resolve_polar_span(acov)

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw={"projection": "polar"}
        )
    else:
        fig = ax.figure

    # Set zero direction
    zero_at = str(zero_at).upper()

    offsets = {"E": 0.0, "N": np.pi / 2, "W": np.pi, "S": 3 * np.pi / 2}

    if zero_at not in offsets.keys():
        warnings.warn(
            f"Unknow plot direction {zero_at!r}. Fallback to 'N'.",
            stacklevel=2,
        )
        zero_at = "N"

    ax.set_theta_offset(offsets[zero_at])

    # Clockwise or counter-clockwise
    ax.set_theta_direction(-1 if clockwise else 1)

    # Limit angular range to [0, span]
    ax.set_thetamin(0.0)
    ax.set_thetamax(np.degrees(span))

    return fig, ax, span


def map_theta_to_span(
    theta_raw: np.ndarray,
    *,
    span: float,
    theta_period: float | None = None,
    data_min: float | None = None,
    data_max: float | None = None,
) -> np.ndarray:
    """
    Map arbitrary theta values to [0, span].

    - If theta_period is given, uses modulo scaling.
    - Else if data_min/max are given, min-max scale.
    - Else assumes theta_raw already in [0, 2π] and rescales to [0, span].
    """
    theta_raw = np.asarray(theta_raw)

    if theta_period is not None:
        t = (theta_raw % theta_period) / theta_period  # [0,1]
        return t * span

    if (
        (data_min is not None)
        and (data_max is not None)
        and (data_max > data_min)
    ):
        t = (theta_raw - data_min) / (data_max - data_min)  # [0,1]
        return t * span

    # Default: assume input is in [0, 2π]
    return (theta_raw % (2 * np.pi)) / (2 * np.pi) * span


def _default_theta_ticks(span: float) -> tuple[np.ndarray, list[str]]:
    # nice ticks for 360/180/90/45 deg spans
    deg = np.degrees(span)
    if np.isclose(deg, 360):
        vals = np.arange(0, 360, 30)
    elif np.isclose(deg, 180):
        vals = np.arange(0, 180 + 1e-9, 15)
    elif np.isclose(deg, 90):
        vals = np.arange(0, 90 + 1e-9, 10)
    else:
        vals = np.arange(0, 45 + 1e-9, 5)

    return np.radians(vals), [f"{int(v)}°" for v in vals]


# Usage
# ticks, labels = _default_theta_ticks(span)
# ax.set_thetagrids(np.degrees(ticks), labels)


def acov_to_span(acov: Acov = "default") -> float:
    """
    Thin alias over `resolve_polar_span` so older call-sites keep
    working. Returns the angular span in radians.
    """
    return resolve_polar_span(acov)


def set_polar_angular_span(ax: Axes, acov: Acov = "default") -> float:
    """
    Apply the `[thetamin, thetamax]` corresponding to `acov` on an
    existing polar axes. Returns the span in radians.
    """
    span = resolve_polar_span(acov)
    ax.set_thetamin(0.0)
    ax.set_thetamax(np.degrees(span))
    return span


def resolve_polar_axes(
    ax: Axes | None = None,
    *,
    acov: Acov = "default",
    figsize: tuple[float, float] | None = None,
    zero_at: Literal["N", "E", "S", "W"] = "N",
    clockwise: bool = True,
) -> Axes:
    """
    Convenience wrapper that reuses/creates a polar axes configured
    with the requested angular coverage. Returns the Axes only.
    """
    _, ax, _ = setup_polar_axes(
        ax,
        acov=acov,
        figsize=figsize,
        zero_at=zero_at,
        clockwise=clockwise,
    )
    return ax


def _acov_to_deg(acov: str) -> float:
    """Safe degree lookup; returns NaN on invalid acov."""
    try:
        return float(np.degrees(resolve_polar_span(acov)))
    except Exception:
        return float("nan")


def _fmt_pref_list(
    prefs: Sequence[str],
    *,
    include_deg: bool = True,
) -> str:
    """Human-friendly list: 'default (360°)' or 'A or B'."""
    parts = []
    for p in prefs:
        if include_deg:
            d = _acov_to_deg(p)
            suf = f" ({int(round(d))}°)" if np.isfinite(d) else ""
        else:
            suf = ""
        parts.append(f"'{p}'{suf}")
    if len(parts) <= 1:
        return parts[0] if parts else ""
    if len(parts) == 2:
        return " or ".join(parts)
    return ", ".join(parts[:-1]) + f", or {parts[-1]}"


def warn_acov_preference(
    acov: str,
    *,
    preferred: str | Sequence[str] = "default",
    plot_name: str | None = None,
    reason: str | None = None,
    advice: str | None = None,
    warn_cls: type[Warning] = UserWarning,
    stacklevel: int = 2,
    enable: bool = True,
    include_deg: bool = True,
) -> bool:
    """
    Emit a gentle, configurable warning when `acov` deviates from a
    preferred set. Returns True iff a warning was issued.

    Parameters
    ----------
    acov
        The requested angular coverage keyword.
    preferred
        Single value or list of acceptable/ideal acov values.
    plot_name
        Short context label (e.g. 'relationship', 'fingerprint').
    reason
        Why the preference exists (readability, conventions, etc.).
    advice
        Closing note; defaults to 'proceeding with the requested span.'
    warn_cls
        Warning class to emit (UserWarning by default).
    stacklevel
        Passed to `warnings.warn` for correct source pointing.
    enable
        If False, does nothing and returns False.
    include_deg
        If True, append degree hints like '(360°)' to names.
    """
    if not enable:
        return False

    acov_l = (acov or "").lower()
    acov_l = canonical_acov(acov_l)
    prefs = columns_manager(preferred, empty_as_none=False, to_string=True)
    prefs = [canonical_acov(str(p).lower()) for p in prefs]

    if acov_l in prefs:
        return False

    # sensible defaults
    if reason is None:
        if "default" in prefs:
            reason = "a full 360° span is often clearest for comparison"
        elif "half_circle" in prefs:
            reason = "a half-circle (180°) often improves label readability"
        else:
            reason = "the preferred span usually yields better readability"

    if advice is None:
        advice = "proceeding with the requested span."

    # compose message
    d_req = _acov_to_deg(acov_l)
    req_suf = (
        f" ({int(round(d_req))}°)"
        if include_deg and np.isfinite(d_req)
        else ""
    )
    ctx = f" for {plot_name}" if plot_name else ""
    want = _fmt_pref_list(prefs, include_deg=include_deg)

    msg = (
        f"Using acov='{acov_l}'{req_suf}{ctx}. "
        f"Tip: {want} is preferred; {reason}; {advice}"
    )

    warnings.warn(msg, warn_cls, stacklevel=stacklevel)
    return True
