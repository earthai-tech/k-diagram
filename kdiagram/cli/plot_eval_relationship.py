from __future__ import annotations

import argparse

from kdiagram.plot.relationship import (
    plot_error_relationship,
    plot_residual_relationship,
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


# -------------------------- cmds --------------------------
def _cmd_plot_residual_relationship(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy(dtype=float)

    # preds (1 col per model)
    yps, names = _collect_point_preds(df, ns)

    plot_residual_relationship(
        y_true,
        *yps,
        names=names,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        s=ns.s,
        alpha=ns.alpha,
        show_zero_line=ns.show_zero_line,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _cmd_plot_error_relationship(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy(dtype=float)

    # preds (1 col per model)
    yps, names = _collect_point_preds(df, ns)

    plot_error_relationship(
        y_true,
        *yps,
        names=names,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        s=ns.s,
        alpha=ns.alpha,
        show_zero_line=ns.show_zero_line,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


# ----------------------- subparsers -----------------------
def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. If omitted, use --input.",
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
        help="Input format override (csv, parquet, ...).",
    )
    p.add_argument(
        "--y-true",
        "--true-col",
        dest="y_true",
        required=True,
        help="Ground-truth column name.",
    )


def _add_preds_and_names_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec 'NAME:col'. Repeat to add models. "
            "For these plots, each model must map to one col."
        ),
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction columns (space-separated). Repeat to "
            "add multiple models (one col per model)."
        ),
    )
    p.add_argument(
        "--pred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_cols",
        help="Alias for --pred (single CSV token allowed).",
    )
    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Model names (CSV or space separated).",
    )


def _add_style_and_output_args(
    p: argparse.ArgumentParser,
    *,
    default_title: str,
    default_s: int,
) -> None:
    p.add_argument(
        "--title",
        default=default_title,
        help="Figure title.",
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
        help="Matplotlib colormap name.",
    )
    p.add_argument(
        "--s",
        type=int,
        default=default_s,
        help="Marker size.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Marker alpha in [0, 1].",
    )
    add_bool_flag(
        p,
        "show-zero-line",
        True,
        "Show zero-error reference circle.",
        "Hide zero-error reference circle.",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show polar grid.",
        "Hide polar grid.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI when saving.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure. If omitted, show it.",
    )


def add_plot_eval_relationship(
    subparsers: argparse._SubParsersAction,
) -> None:
    # ---- plot-residual-relationship ----
    p = subparsers.add_parser(
        "plot-residual-relationship",
        help="Residual vs predicted (polar scatter).",
        description=(
            "Plot residuals (actual - predicted) against a "
            "monotone mapping of the predicted value on polar "
            "axes. Each model must provide one prediction col."
        ),
    )
    _add_common_io_args(p)
    _add_preds_and_names_args(p)
    _add_style_and_output_args(
        p,
        default_title="Residual vs. Predicted Relationship",
        default_s=50,
    )
    p.set_defaults(func=_cmd_plot_residual_relationship)

    # ---- plot-error-relationship ----
    p2 = subparsers.add_parser(
        "plot-error-relationship",
        help="Error vs true value (polar scatter).",
        description=(
            "Plot errors (actual - predicted) vs a monotone "
            "mapping of the true value on polar axes. Each "
            "model must provide one prediction col."
        ),
    )
    _add_common_io_args(p2)
    _add_preds_and_names_args(p2)
    _add_style_and_output_args(
        p2,
        default_title="Error vs. True Value Relationship",
        default_s=50,
    )
    add_bool_flag(
        p2,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show radial tick labels.",
    )
    p2.set_defaults(func=_cmd_plot_error_relationship)
