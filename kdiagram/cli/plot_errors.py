from __future__ import annotations

import argparse
from collections.abc import Iterable

from kdiagram.plot.errors import (
    plot_error_bands,
    plot_error_ellipses,
    plot_error_violins,
)

from ._utils import (
    ColumnsListAction,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_errors"]


# --------------------------- helpers ------------------------------
def _flatten_cols(items: Iterable | None) -> list[str]:
    if not items:
        return []
    flat: list[str] = []
    for it in items:
        if isinstance(it, (list, tuple)):
            flat.extend(str(x) for x in it)
        else:
            flat.append(str(it))
    return flat


# ----------------------- plot-error-violins -----------------------
def _cmd_plot_error_violins(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # collect error columns from flags
    err_cols = _flatten_cols(ns.error) + _flatten_cols(ns.error_cols)
    if not err_cols:
        raise SystemExit("Provide error columns via --error/--error-cols.")

    ensure_columns(df, err_cols)
    df = ensure_numeric(df, err_cols, copy=True, errors="raise")

    plot_error_violins(
        df,
        *err_cols,
        names=ns.names,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
        alpha=ns.alpha,
        edgecolor=ns.edgecolor,
    )


def _add_plot_error_violins_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-error-violins",
        help="Polar violins for error distributions.",
        description=(
            "Compare multiple error distributions (e.g., actual - "
            "predicted) as polar violin plots, one sector per model."
        ),
    )

    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path.",
    )
    p.add_argument(
        "-i",
        "--input",
        dest="input",
        help="Input table path (alt form).",
    )
    p.add_argument(
        "--format",
        default=None,
        help="Input format override.",
    )

    # columns
    p.add_argument(
        "--error",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="error",
        help=("Error columns (CSV or tokens). Repeat to add more columns."),
    )
    p.add_argument(
        "--error-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="error_cols",
        help="Alias for --error.",
    )
    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Names (CSV or space separated).",
    )

    # style
    p.add_argument(
        "--title",
        default="Comparison of Error Distributions",
        help="Figure title.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(9.0, 9.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap for violins.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Violin face alpha (0..1).",
    )
    p.add_argument(
        "--edgecolor",
        default=None,
        help="Violin edge color.",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save figure.",
    )

    p.set_defaults(func=_cmd_plot_error_violins)


# ------------------------ plot-error-bands ------------------------
def _cmd_plot_error_bands(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    ensure_columns(df, [ns.error_col, ns.theta_col])
    df = ensure_numeric(
        df,
        [ns.error_col, ns.theta_col],
        copy=True,
        errors="raise",
    )

    plot_error_bands(
        df=df,
        error_col=ns.error_col,
        theta_col=ns.theta_col,
        theta_period=ns.theta_period,
        theta_bins=ns.theta_bins,
        n_std=ns.n_std,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_angle=ns.mask_angle,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
        color=ns.color,
        alpha=ns.alpha,
    )


def _add_plot_error_bands_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-error-bands",
        help="Mean±k·std error vs angle bins (polar).",
        description=(
            "Bin an angle-like feature and show mean error with a "
            "shaded ±k·std band on polar axes."
        ),
    )

    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path.",
    )
    p.add_argument(
        "-i",
        "--input",
        dest="input",
        help="Input table path (alt form).",
    )
    p.add_argument(
        "--format",
        default=None,
        help="Input format override.",
    )

    p.add_argument(
        "--error-col",
        required=True,
        help="Column with errors (e.g., actual - pred).",
    )
    p.add_argument(
        "--theta-col",
        required=True,
        help="Binning feature (mapped to angle).",
    )
    p.add_argument(
        "--theta-period",
        type=float,
        default=None,
        help="Period of theta (wrap) if cyclical.",
    )
    p.add_argument(
        "--theta-bins",
        type=int,
        default=24,
        help="Number of angular bins.",
    )
    p.add_argument(
        "--n-std",
        type=float,
        default=1.0,
        help="Std multiples for band width.",
    )

    p.add_argument(
        "--title",
        default=None,
        help="Figure title (auto if omitted).",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap (not critical here).",
    )
    p.add_argument(
        "--color",
        default="#3498DB",
        help="Band face color.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Band alpha (0..1).",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
    )
    add_bool_flag(
        p,
        "mask-angle",
        False,
        "Hide angular tick labels.",
        "Show angular tick labels.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save figure.",
    )

    p.set_defaults(func=_cmd_plot_error_bands)


# ----------------------- plot-error-ellipses ----------------------
def _cmd_plot_error_ellipses(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    req = [ns.r_col, ns.theta_col, ns.r_std_col, ns.theta_std_col]
    ensure_columns(df, req)
    df = ensure_numeric(df, req, copy=True, errors="raise")

    # color column is optional; if provided, try numeric coercion
    if ns.color_col:
        ensure_columns(df, [ns.color_col])
        df = ensure_numeric(df, [ns.color_col], copy=True, errors="raise")

    plot_error_ellipses(
        df=df,
        r_col=ns.r_col,
        theta_col=ns.theta_col,
        r_std_col=ns.r_std_col,
        theta_std_col=ns.theta_std_col,
        color_col=ns.color_col,
        n_std=ns.n_std,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        mask_angle=ns.mask_angle,
        mask_radius=ns.mask_radius,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
        alpha=ns.alpha,
        edgecolor=ns.edgecolor,
        linewidth=ns.linewidth,
    )


def _add_plot_error_ellipses_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-error-ellipses",
        help="Polar error ellipses (2D uncertainty).",
        description=(
            "Draw ellipses for uncertainty where both radial and "
            "angular components have std devs."
        ),
    )

    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path.",
    )
    p.add_argument(
        "-i",
        "--input",
        dest="input",
        help="Input table path (alt form).",
    )
    p.add_argument(
        "--format",
        default=None,
        help="Input format override.",
    )

    p.add_argument(
        "--r-col",
        required=True,
        help="Mean radial column.",
    )
    p.add_argument(
        "--theta-col",
        required=True,
        help="Mean angular column (degrees).",
    )
    p.add_argument(
        "--r-std-col",
        required=True,
        help="Std dev in radial direction.",
    )
    p.add_argument(
        "--theta-std-col",
        required=True,
        help="Std dev in angle (degrees).",
    )
    p.add_argument(
        "--color-col",
        default=None,
        help="Optional column to color ellipses.",
    )
    p.add_argument(
        "--n-std",
        type=float,
        default=2.0,
        help="Std multiples for ellipse size.",
    )

    p.add_argument(
        "--title",
        default=None,
        help="Figure title (auto if omitted).",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap for ellipse colors.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Ellipse face alpha (0..1).",
    )
    p.add_argument(
        "--edgecolor",
        default=None,
        help="Ellipse edge color.",
    )
    p.add_argument(
        "--linewidth",
        type=float,
        default=0.5,
        help="Ellipse edge line width.",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
    )
    add_bool_flag(
        p,
        "mask-angle",
        False,
        "Hide angular tick labels.",
        "Show angular tick labels.",
    )
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show radial tick labels.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save figure.",
    )

    p.set_defaults(func=_cmd_plot_error_ellipses)


# ----------------------------- registrar --------------------------
def add_plot_errors(sub: argparse._SubParsersAction) -> None:
    """
    Register error-plot subcommands:
    - plot-error-violins
    - plot-error-bands
    - plot-error-ellipses
    """
    _add_plot_error_violins_subparser(sub)
    _add_plot_error_bands_subparser(sub)
    _add_plot_error_ellipses_subparser(sub)
