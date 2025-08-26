# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..plot.uncertainty import plot_actual_vs_predicted
from ._utils import (
    _parse_kv_list,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_vs"]


def _cmd_plot_actual_vs_predicted(ns: argparse.Namespace) -> None:
    essential = [ns.actual_col, ns.pred_col]
    naf = essential if ns.dropna else None

    df: pd.DataFrame = load_df(ns.input, format=ns.format, dropna=naf)

    cols = [ns.actual_col, ns.pred_col]
    if ns.theta_col:
        cols.append(ns.theta_col)
    ensure_columns(df, cols)

    df = ensure_numeric(df, essential, copy=True, errors="raise")

    aprops = _parse_kv_list(ns.actual_props)
    pprops = _parse_kv_list(ns.pred_props)

    plot_actual_vs_predicted(
        df=df,
        actual_col=ns.actual_col,
        pred_col=ns.pred_col,
        theta_col=ns.theta_col,
        acov=ns.acov,
        figsize=ns.figsize,
        title=ns.title,
        line=ns.line,
        r_label=ns.r_label,
        cmap=ns.cmap,
        alpha=ns.alpha,
        actual_props=aprops or None,
        pred_props=pprops or None,
        show_grid=ns.show_grid,
        grid_props=None,
        show_legend=ns.show_legend,
        mask_angle=ns.mask_angle,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


# --------------------------- registration -------------------------- #
def add_plot_vs(subparsers: argparse._SubParsersAction) -> None:
    """
    Register 'plot-actual-vs-predicted' CLI command.
    """
    p = subparsers.add_parser(
        "plot-actual-vs-predicted",
        help="Polar plot: actual vs predicted.",
        description=(
            "Compare actual (ground truth) to predicted values on a "
            "polar plot. Theta currently follows index order."
        ),
    )
    p.add_argument("input", type=str, help="Input table path.")
    p.add_argument(
        "--format",
        type=str,
        default=None,
        help="Explicit input format; else inferred.",
    )
    p.add_argument(
        "--actual-col",
        type=str,
        required=True,
        help="Column with actual values.",
    )
    p.add_argument(
        "--pred-col",
        type=str,
        required=True,
        help="Column with predicted values.",
    )
    p.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Optional angle-order column (ignored for order).",
    )
    p.add_argument(
        "--acov",
        type=str,
        default="default",
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage span.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH' (default 8,8).",
    )
    p.add_argument("--title", type=str, default=None, help="Title.")
    p.add_argument(
        "--r-label",
        type=str,
        default=None,
        help="Radial axis label.",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="Matplotlib colormap (currently unused).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Overlay alpha for line bands.",
    )
    p.add_argument(
        "--actual-props",
        nargs="+",
        default=None,
        help="Matplotlib props for actual (key=value ...).",
    )
    p.add_argument(
        "--pred-props",
        nargs="+",
        default=None,
        help="Matplotlib props for predicted (key=value ...).",
    )

    add_bool_flag(p, "line", True, "Draw connecting lines.", "No lines.")
    add_bool_flag(
        p, "show-grid", True, "Show polar grid.", "Hide polar grid."
    )
    add_bool_flag(
        p,
        "show-legend",
        True,
        "Show legend.",
        "Hide legend.",
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
        "dropna",
        True,
        "Drop rows with NaNs in essential cols.",
        "Keep rows with NaNs.",
    )

    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save figure to this path.",
    )

    p.set_defaults(func=_cmd_plot_actual_vs_predicted)
