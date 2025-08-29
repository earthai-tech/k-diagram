# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt

from ..plot.context import (
    plot_error_distribution,
    plot_error_pacf,
)
from ._utils import (
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = [
    "add_plot_error_distribution",
    "add_plot_error_pacf",
    "add_context_err",
]


def _cmd_plot_error_distribution(ns: argparse.Namespace) -> None:
    """
    CLI runner for plot-error-distribution.
    """
    df = load_df(ns.input, format=ns.format)

    need = [ns.actual_col, ns.pred_col]
    ensure_columns(df, need)
    df = ensure_numeric(df, need, copy=True, errors="raise")

    hist_kwargs: dict[str, Any] = {}
    if ns.figsize:
        hist_kwargs["figsize"] = ns.figsize
    if ns.bins is not None:
        hist_kwargs["bins"] = ns.bins
    hist_kwargs["kde"] = ns.kde
    hist_kwargs["density"] = ns.density
    if ns.hist_color:
        hist_kwargs["hist_color"] = ns.hist_color
    if ns.kde_color:
        hist_kwargs["kde_color"] = ns.kde_color
    if ns.alpha is not None:
        hist_kwargs["alpha"] = ns.alpha

    ax = plot_error_distribution(
        df=df,
        actual_col=ns.actual_col,
        pred_col=ns.pred_col,
        title=ns.title,
        xlabel=ns.xlabel,
        **hist_kwargs,
    )

    # the plotting function does not savefig; handle here
    if ns.savefig and ax is not None:
        fig = ax.figure
        fig.savefig(str(ns.savefig), dpi=ns.dpi, bbox_inches="tight")
        plt.close(fig)


def _cmd_plot_error_pacf(ns: argparse.Namespace) -> None:
    """
    CLI runner for plot-error-pacf.
    """
    df = load_df(ns.input, format=ns.format)

    need = [ns.actual_col, ns.pred_col]
    ensure_columns(df, need)
    df = ensure_numeric(df, need, copy=True, errors="raise")

    pacf_kwargs: dict[str, Any] = {}
    if ns.lags is not None:
        pacf_kwargs["lags"] = ns.lags
    if ns.alpha is not None:
        pacf_kwargs["alpha"] = ns.alpha
    if ns.method:
        pacf_kwargs["method"] = ns.method

    plot_error_pacf(
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
        **pacf_kwargs,
    )


# ------------------------------- parsers --------------------------------


def add_plot_error_distribution(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-error-dist",
        help=("Histogram + KDE of forecast errors " "(actual - predicted)."),
        description=(
            "Compute residuals and plot their distribution using "
            "a histogram and an optional KDE curve."
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
        "--pred-col",
        required=True,
        help="Prediction column name.",
    )

    # figure / styling
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 6.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--xlabel", default=None, help="X label.")
    p.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Histogram bins.",
    )
    add_bool_flag(
        p,
        "kde",
        True,
        "Overlay a KDE curve.",
        "Do not draw KDE.",
    )
    add_bool_flag(
        p,
        "density",
        True,
        "Normalize histogram to density.",
        "Use counts.",
    )
    p.add_argument(
        "--hist-color",
        default=None,
        help="Histogram color.",
    )
    p.add_argument(
        "--kde-color",
        default=None,
        help="KDE line color.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Bar alpha in [0,1].",
    )
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save figure.")

    p.set_defaults(func=_cmd_plot_error_distribution)


def add_plot_error_pacf(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-error-pacf",
        help=(
            "Partial autocorrelation of forecast errors "
            "(requires statsmodels)."
        ),
        description=(
            "Compute residuals (actual - predicted) and plot "
            "their partial autocorrelation (PACF)."
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
        "--pred-col",
        required=True,
        help="Prediction column name.",
    )

    # figure / styling
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

    # PACF options (forwarded to statsmodels)
    p.add_argument(
        "--lags",
        type=int,
        default=40,
        help="Number of lags for PACF.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Confidence level alpha.",
    )
    p.add_argument(
        "--method",
        default=None,
        help="Statsmodels PACF method (e.g., 'yw').",
    )

    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save figure.")

    p.set_defaults(func=_cmd_plot_error_pacf)


def add_context_err(sub: argparse._SubParsersAction) -> None:
    """Register the context error/distribution plot commands."""
    add_plot_error_distribution(sub)
    add_plot_error_pacf(sub)
