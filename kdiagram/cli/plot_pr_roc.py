# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse

from ..plot.evaluation import (
    plot_polar_pr_curve,
    plot_polar_roc,
)
from ._utils import (
    ColumnsListAction,
    _collect_point_preds,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = [
    "add_pr_roc",
]


# ---------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------
def _cmd_plot_polar_roc(ns: argparse.Namespace) -> None:
    """CLI runner for polar ROC curve."""
    df = load_df(ns.input, format=ns.format)

    # required cols
    ensure_columns(df, [ns.y_true])

    # numeric enforcement for labels + preds
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy()

    # collect preds (each group => exactly one column)
    yps, names_cli = _collect_point_preds(df, ns)

    # names
    names = list(ns.names) if ns.names else None
    if names is None and names_cli:
        names = list(names_cli)

    # all pred columns must be numeric
    pred_cols = [c for _, c in ns._pred_groups]  # set by collector
    if pred_cols:
        df = ensure_numeric(df, pred_cols, copy=True, errors="raise")

    plot_polar_roc(
        y_true,
        *yps,
        names=names,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _cmd_plot_polar_pr(ns: argparse.Namespace) -> None:
    """CLI runner for polar Precision–Recall curve."""
    df = load_df(ns.input, format=ns.format)

    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy()

    yps, names_cli = _collect_point_preds(df, ns)

    names = list(ns.names) if ns.names else None
    if names is None and names_cli:
        names = list(names_cli)

    # pred_cols = [c for _, c in ns._pred_groups]
    pred_cols = []
    for _, cols in ns._pred_groups:
        if isinstance(cols, (list, tuple)):
            pred_cols.append(cols[0])
        else:
            pred_cols.append(cols)

    if pred_cols:
        df = ensure_numeric(df, pred_cols, copy=True, errors="raise")

    plot_polar_pr_curve(
        y_true,
        *yps,
        names=names,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _common_io_and_preds(p: argparse.ArgumentParser) -> None:
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
        "--y-true",
        "--true-col",
        dest="y_true",
        required=True,
        help="Ground-truth binary labels column (0/1).",
    )
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec 'name:col'. Repeat to add models.",
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help="Prediction column (probability). Repeat to add models.",
    )
    p.add_argument(
        "--pred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_cols",
        help="Alias for --pred (CSV or tokens).",
    )
    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Model names (CSV or tokens).",
    )

    # figure + style
    p.add_argument("--cmap", default="viridis", help="Colormap name.")
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show polar grid.",
        "Hide polar grid.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save figure.")


def add_plot_polar_roc(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-polar-roc",
        help="Polar ROC curve for one or more models.",
        description=(
            "Plot the ROC curve in polar coordinates. Provide "
            "labels via --y-true and predictions via "
            "--pred/--pred-cols/--model. Each model is a single "
            "probability column."
        ),
    )
    _common_io_and_preds(p)
    p.set_defaults(func=_cmd_plot_polar_roc)


def add_plot_polar_pr_curve(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-polar-pr-curve",
        help="Polar Precision–Recall curve for one or more models.",
        description=(
            "Plot the Precision–Recall curve in polar coordinates. "
            "Provide labels via --y-true and predictions via "
            "--pred/--pred-cols/--model."
        ),
    )
    _common_io_and_preds(p)
    p.set_defaults(func=_cmd_plot_polar_pr)


def add_pr_roc(sub: argparse._SubParsersAction) -> None:
    """Register the predicion recall and roc plot commands."""
    add_plot_polar_roc(sub)
    add_plot_polar_pr_curve(sub)
