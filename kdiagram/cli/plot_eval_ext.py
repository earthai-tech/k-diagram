# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse

from ..plot.evaluation import (
    plot_pinball_loss,
    plot_polar_classification_report,
)
from ._utils import (
    ColumnsListAction,
    _flatten_cols,
    _parse_float_list,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = [
    "add_eval_extension",
]


# ---------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------
def _cmd_plot_polar_classification_report(
    ns: argparse.Namespace,
) -> None:
    """CLI runner for polar classification report."""
    df = load_df(ns.input, format=ns.format)

    # required columns (labels can be non-numeric)
    ensure_columns(df, [ns.y_true, ns.y_pred])

    plot_polar_classification_report(
        df[ns.y_true].to_numpy(),
        df[ns.y_pred].to_numpy(),
        class_labels=list(ns.class_labels) if ns.class_labels else None,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _cmd_plot_pinball_loss(ns: argparse.Namespace) -> None:
    """CLI runner for pinball-loss per quantile."""
    df = load_df(ns.input, format=ns.format)

    # columns
    qpred_cols = _flatten_cols(ns.qpreds)
    if not qpred_cols:
        raise SystemExit(
            "Provide quantile prediction columns via --qpreds/--qpred-cols."
        )

    # quantiles list
    quantiles = _parse_float_list(ns.quantiles)
    if not quantiles:
        raise SystemExit("Provide --quantiles values.")

    # lengths must match
    if len(qpred_cols) != len(quantiles):
        raise SystemExit(
            "Number of --qpreds columns must match the number of --quantiles."
        )

    # required numeric columns
    need = [ns.y_true] + qpred_cols
    ensure_columns(df, need)
    df = ensure_numeric(df, need, copy=True, errors="raise")

    y_true = df[ns.y_true].to_numpy()
    yq = df[qpred_cols].to_numpy()

    plot_pinball_loss(
        y_true=y_true,
        y_preds_quantiles=yq,
        quantiles=quantiles,  # function re-sorts internally
        names=None,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


# ---------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------
def add_plot_polar_classification_report(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-polar-cr",
        help=(
            "Polar grouped bars for per-class metrics (Precision/Recall/F1)."
        ),
        description=(
            "Compute a classification report per class and display it "
            "in polar coordinates. Provide ground-truth and predicted "
            "label columns."
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
        "--y-true",
        "--true-col",
        dest="y_true",
        required=True,
        help="Ground-truth labels column.",
    )
    p.add_argument(
        "--y-pred",
        "--pred-col",
        dest="y_pred",
        required=True,
        help="Predicted labels column.",
    )
    p.add_argument(
        "--class-labels",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="class_labels",
        help="Optional class labels (CSV or tokens).",
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
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show radial tick labels.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save.")

    p.set_defaults(func=_cmd_plot_polar_classification_report)


def add_plot_pinball_loss(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-pinball-loss",
        help="Polar plot of pinball loss across quantiles.",
        description=(
            "Visualize average pinball loss per quantile on a polar "
            "axis. Provide --y-true, a list of quantile prediction "
            "columns via --qpreds/--qpred-cols, and the corresponding "
            "--quantiles (e.g., 0.1 0.5 0.9)."
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
        "--y-true",
        "--true-col",
        dest="y_true",
        required=True,
        help="Ground-truth numeric column.",
    )
    p.add_argument(
        "--qpreds",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="qpreds",
        help="Quantile prediction columns (CSV or tokens).",
    )
    p.add_argument(
        "--qpred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="qpreds",
        help="Alias for --qpreds.",
    )
    p.add_argument(
        "--quantiles",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="quantiles",
        help="Quantile levels (e.g., 0.1 0.5 0.9).",
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
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show radial tick labels.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save.")

    p.set_defaults(func=_cmd_plot_pinball_loss)


def add_eval_extension(sub: argparse._SubParsersAction) -> None:
    """Register Evaluation extension module plot commands."""
    add_plot_polar_classification_report(sub)
    add_plot_pinball_loss(sub)
