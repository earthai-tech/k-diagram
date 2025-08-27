# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse

from ..plot.context import plot_qq, plot_time_series
from ._utils import (
    ColumnsListAction,
    _flatten_cols,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = [
    "add_plot_time_series",
    "add_plot_qq",
    "add_ts_analyses",
]


def _cmd_plot_time_series(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # columns
    pred_cols = _flatten_cols(ns.pred) + _flatten_cols(ns.pred_cols)
    names = _flatten_cols(ns.names) if ns.names else None

    # exist
    need: list[str] = []
    if ns.x_col:
        need.append(ns.x_col)
    if ns.actual_col:
        need.append(ns.actual_col)
    need.extend(pred_cols)
    if ns.q_lower_col:
        need.append(ns.q_lower_col)
    if ns.q_upper_col:
        need.append(ns.q_upper_col)

    if need:
        ensure_columns(df, need)

    # numeric (exclude x_col which may be datetime)
    num_cols: list[str] = []
    if ns.actual_col:
        num_cols.append(ns.actual_col)
    num_cols.extend(pred_cols)
    if ns.q_lower_col:
        num_cols.append(ns.q_lower_col)
    if ns.q_upper_col:
        num_cols.append(ns.q_upper_col)
    if num_cols:
        df = ensure_numeric(df, num_cols, copy=True, errors="raise")

    plot_time_series(
        df=df,
        x_col=ns.x_col,
        actual_col=ns.actual_col,
        pred_cols=pred_cols or None,
        names=names,
        q_lower_col=ns.q_lower_col,
        q_upper_col=ns.q_upper_col,
        title=ns.title,
        xlabel=ns.xlabel,
        ylabel=ns.ylabel,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _cmd_plot_qq(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    ensure_columns(df, [ns.actual_col, ns.pred_col])
    df = ensure_numeric(
        df, [ns.actual_col, ns.pred_col], copy=True, errors="raise"
    )

    plot_qq(
        df=df,
        actual_col=ns.actual_col,
        pred_col=ns.pred_col,
        title=ns.title,
        xlabel=ns.xlabel,
        ylabel=ns.ylabel,
        figsize=ns.figsize,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def add_plot_time_series(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-time-series",
        help=(
            "Plot actual and model forecasts over time with "
            "optional prediction interval."
        ),
        description=(
            "Create a time-series plot from a table. Use --x-col "
            "for the horizontal axis (index is used if omitted). "
            "Supply --actual-col and/or one or more --pred columns. "
            "Optionally add a lower/upper interval."
        ),
    )

    # I/O
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. If omitted, try --input.",
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
        help="Format override (csv, parquet, ...).",
    )

    # columns
    p.add_argument(
        "--x-col",
        default=None,
        help="X-axis column (e.g., datetime).",
    )
    p.add_argument(
        "--actual-col",
        default=None,
        help="Observed values column (optional).",
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help="Prediction columns (CSV or tokens). Repeat as needed.",
    )
    p.add_argument(
        "--pred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_cols",
        help="Alias for --pred.",
    )
    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Legend names for predictions (CSV or tokens).",
    )
    p.add_argument(
        "--q-lower-col",
        default=None,
        help="Lower bound column of uncertainty interval.",
    )
    p.add_argument(
        "--q-upper-col",
        default=None,
        help="Upper bound column of uncertainty interval.",
    )

    # style
    p.add_argument(
        "--title",
        default=None,
        help="Figure title.",
    )
    p.add_argument(
        "--xlabel",
        default=None,
        help="X-axis label (defaults to column or index).",
    )
    p.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(12.0, 6.0),
        help="Figure size 'W,H' (e.g., 12,6).",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap for prediction series.",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid lines.",
        "Hide grid lines.",
    )

    # output
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure.",
    )

    p.set_defaults(func=_cmd_plot_time_series)


def add_plot_qq(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-qq",
        help=(
            "Q-Q plot of forecast errors (actual - predicted) "
            "against a normal distribution."
        ),
        description=(
            "Compute errors = actual - predicted and draw a "
            "Quantile-Quantile plot versus the normal law to "
            "assess normality of residuals."
        ),
    )

    # I/O
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. If omitted, try --input.",
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
        help="Format override (csv, parquet, ...).",
    )

    # columns
    p.add_argument(
        "--actual-col",
        required=True,
        help="Observed values column.",
    )
    p.add_argument(
        "--pred-col",
        required=True,
        help="Predicted values column.",
    )

    # style
    p.add_argument(
        "--title",
        default=None,
        help="Figure title.",
    )
    p.add_argument(
        "--xlabel",
        default=None,
        help="X-axis label.",
    )
    p.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(7.0, 7.0),
        help="Figure size 'W,H' (e.g., 7,7).",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid lines.",
        "Hide grid lines.",
    )

    # output
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure.",
    )

    p.set_defaults(func=_cmd_plot_qq)


def add_ts_analyses(sub: argparse._SubParsersAction) -> None:
    """Register the time-series analysis plot commands."""
    add_plot_time_series(sub)
    add_plot_qq(sub)
