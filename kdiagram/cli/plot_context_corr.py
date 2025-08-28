# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse

from ..plot.context import (
    plot_error_autocorrelation,
    plot_scatter_correlation,
)
from ._utils import (
    ColumnsListAction,
    _flatten_cols,
    _parse_models,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = [
    "add_plot_scatter_correlation",
    "add_plot_error_autocorrelation",
    "add_context_corr",
]


def _cmd_plot_scatter_correlation(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # collect pred columns from --pred/--pred-cols and --model
    pred_cols = _flatten_cols(ns.pred) + _flatten_cols(ns.pred_cols)
    m_cols, m_names = ([], [])
    if ns.model:
        m_cols, m_names = _parse_models(ns.model)

    # m_cols, m_names = _parse_models(ns.model)
    pred_cols.extend(m_cols)

    # names
    names = _flatten_cols(ns.names) if ns.names else None
    if not names and m_names:
        names = list(m_names)

    # required columns
    need: list[str] = [ns.actual_col] + pred_cols
    ensure_columns(df, need)

    # numeric enforcement (actual + preds)
    df = ensure_numeric(df, need, copy=True, errors="raise")

    plot_scatter_correlation(
        df=df,
        actual_col=ns.actual_col,
        pred_cols=pred_cols,
        names=names,
        title=ns.title,
        xlabel=ns.xlabel,
        ylabel=ns.ylabel,
        figsize=ns.figsize,
        cmap=ns.cmap,
        s=ns.s,
        alpha=ns.alpha,
        show_identity_line=ns.show_identity_line,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _cmd_plot_error_autocorrelation(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # single pred via --pred or --pred-col
    pred_cols = _flatten_cols(ns.pred) + _flatten_cols(ns.pred_col)
    if len(pred_cols) != 1:
        raise SystemExit(
            "Provide exactly one prediction column via "
            "--pred or --pred-col."
        )
    pred_col = pred_cols[0]

    need: list[str] = [ns.actual_col, pred_col]
    ensure_columns(df, need)

    # numeric enforcement (actual + pred)
    df = ensure_numeric(df, need, copy=True, errors="raise")

    plot_error_autocorrelation(
        df=df,
        actual_col=ns.actual_col,
        pred_col=pred_col,
        title=ns.title,
        xlabel=ns.xlabel,
        ylabel=ns.ylabel,
        figsize=ns.figsize,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def add_plot_scatter_correlation(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-scatter-corr",
        help=(
            "Scatter of true vs predicted values for one or " "more models."
        ),
        description=(
            "Visualize correlation by plotting actual values on "
            "the x-axis and predictions on the y-axis for one or "
            "more models. Optionally draw the y=x identity line."
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
        help="Ground-truth column used on the x-axis.",
    )
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec 'Name:col'. Repeat to add models.",
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction column names (CSV or tokens). Repeat to "
            "add multiple groups."
        ),
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
        help="Legend labels for models (CSV or tokens).",
    )

    # appearance
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--xlabel", default=None, help="X label.")
    p.add_argument("--ylabel", default=None, help="Y label.")
    p.add_argument("--cmap", default="viridis", help="Colormap.")
    p.add_argument("--s", type=int, default=50, help="Marker size.")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Marker alpha in [0,1].",
    )
    add_bool_flag(
        p,
        "show-identity-line",
        True,
        "Draw the y=x identity line.",
        "Do not draw the identity line.",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save figure.")

    p.set_defaults(func=_cmd_plot_scatter_correlation)


def add_plot_error_autocorrelation(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-error-autocorr",
        help=(
            "Autocorrelation (ACF) of forecast errors for a " "single model."
        ),
        description=(
            "Compute residuals (actual - predicted) and plot the "
            "autocorrelation function to diagnose remaining "
            "temporal structure."
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
        help="Ground-truth column name.",
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help="Prediction column (single).",
    )
    p.add_argument(
        "--pred-col",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_col",
        help="Alias for --pred.",
    )

    # appearance
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(10.0, 5.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--xlabel", default=None, help="X label.")
    p.add_argument("--ylabel", default=None, help="Y label.")
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save figure.")

    p.set_defaults(func=_cmd_plot_error_autocorrelation)


def add_context_corr(sub: argparse._SubParsersAction) -> None:
    """Register the context auto/correlation plot commands."""
    add_plot_scatter_correlation(sub)
    add_plot_error_autocorrelation(sub)
