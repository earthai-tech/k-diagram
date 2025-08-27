# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse

from ..plot.evaluation import (
    plot_polar_confusion_matrix,
    plot_polar_confusion_matrix_in,
)
from ._utils import (
    ColumnsListAction,
    _collect_point_preds,
    _flatten_cols,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = [
    "add_plot_polar_cm",
    "add_plot_polar_cm_in",
    "add_plot_polar_cm_multiclass",
    "add_confusion_matrices",
]


def _cmd_plot_polar_cm(ns: argparse.Namespace) -> None:
    """CLI runner for binary polar confusion matrix."""
    df = load_df(ns.input, format=ns.format)

    # y_true must exist and be numeric; preds numeric too
    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy()

    # predictions (each group must be exactly one column)
    yps, names_cli = _collect_point_preds(df, ns)

    # user-provided names override model-derived names
    names = list(ns.names) if ns.names else None
    if names is None and names_cli:
        names = list(names_cli)

    # run plot
    plot_polar_confusion_matrix(
        y_true,
        *yps,
        names=names,
        normalize=ns.normalize,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _cmd_plot_polar_cm_in(
    ns: argparse.Namespace,
) -> None:
    """CLI runner for multiclass polar confusion matrix."""
    df = load_df(ns.input, format=ns.format)

    # required columns (do NOT force numeric: labels may be strings)
    need = [ns.y_true, ns.y_pred]
    ensure_columns(df, need)

    y_true = df[ns.y_true].to_numpy()
    y_pred = df[ns.y_pred].to_numpy()

    class_labels = _flatten_cols(ns.class_labels) if ns.class_labels else None

    plot_polar_confusion_matrix_in(
        y_true,
        y_pred,
        class_labels=class_labels,
        normalize=ns.normalize,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


# -------------------------- parsers ---------------------------------


def _add_common_fig_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--cmap", default="viridis", help="Colormap.")
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
    )
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show radial tick labels.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save.")


def add_plot_polar_cm(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-polar-cm",
        help=(
            "Polar confusion matrix for binary classification "
            "with one or more prediction columns."
        ),
        description=(
            "Plot a polar confusion matrix (TP/FP/TN/FN) per model. "
            "Provide the ground-truth column with --y-true and one "
            "column per model via --pred/--pred-cols/--model."
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
        help="Ground-truth column (binary labels).",
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

    # options
    add_bool_flag(
        p,
        "normalize",
        True,
        "Normalize counts to proportions.",
        "Use raw counts.",
    )

    _add_common_fig_args(p)
    p.set_defaults(func=_cmd_plot_polar_cm)


def _add_cm_in_like_args(p: argparse.ArgumentParser) -> None:
    # columns
    p.add_argument(
        "--y-true",
        "--true-col",
        dest="y_true",
        required=True,
        help="Ground-truth class label column.",
    )
    p.add_argument(
        "--y-pred",
        "--pred-col",
        dest="y_pred",
        required=True,
        help="Predicted class label column.",
    )
    p.add_argument(
        "--class-labels",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="class_labels",
        help="Optional class labels (CSV or tokens).",
    )

    # logic
    add_bool_flag(
        p,
        "normalize",
        True,
        "Normalize rows (proportions).",
        "Use raw counts.",
    )
    _add_common_fig_args(p)


def add_plot_polar_cm_in(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-polar-cm-in",
        help="Polar confusion matrix (multiclass).",
        description=(
            "Grouped polar bars. --y-true and --y-pred required. "
            "Optionally pass --class-labels."
        ),
    )
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

    _add_cm_in_like_args(p)
    p.set_defaults(func=_cmd_plot_polar_cm_in)


def add_plot_polar_cm_multiclass(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-polar-cm-multiclass",
        help="Alias of plot-polar-cm-in.",
        description=(
            "Alias for multiclass polar confusion matrix. "
            "Same args as plot-polar-cm-in."
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

    # IMPORTANT: register the exact same options, incl. --savefig
    _add_cm_in_like_args(p)
    # IMPORTANT: call the same runner so ns.savefig is honored
    p.set_defaults(func=_cmd_plot_polar_cm_in)


def add_confusion_matrices(
    sub: argparse._SubParsersAction,
) -> None:
    """
    Register polar confusion-matrix commands.

    Adds:
      - plot-polar-cm
      - plot-polar-cm-in
      - plot-polar-cm-multiclass
    """
    add_plot_polar_cm(sub)
    add_plot_polar_cm_in(sub)
    add_plot_polar_cm_multiclass(sub)
