# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..plot.uncertainty import plot_temporal_uncertainty
from ._utils import (
    _split_tokens,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_temporal"]


def _cmd_plot_temporal_uncertainty(ns: argparse.Namespace) -> None:
    # Load table (no dropna here; function handles alignment)
    df: pd.DataFrame = load_df(ns.input, format=ns.format)

    # q-cols: either ["auto"] or explicit list of names
    if ns.q_cols and len(ns.q_cols) == 1 and ns.q_cols[0] == "auto":
        q_cols: str | list[str] = "auto"
        cols_to_check: list[str] = []
    else:
        q_cols = _split_tokens(ns.q_cols or [])
        cols_to_check = q_cols

    # names: optional list
    names = _split_tokens(ns.names) if ns.names else None

    # Validation / coercion only when explicit columns provided
    if cols_to_check:
        ensure_columns(df, cols_to_check)
        df = ensure_numeric(df, cols_to_check, copy=True, errors="raise")

    # Call the plotting function
    plot_temporal_uncertainty(
        df=df,
        q_cols=q_cols,
        theta_col=ns.theta_col,
        names=names,
        acov=ns.acov,
        figsize=ns.figsize,
        title=ns.title,
        cmap=ns.cmap,
        normalize=ns.normalize,
        show_grid=ns.show_grid,
        grid_props=None,
        alpha=ns.alpha,
        s=ns.s,
        dot_style=ns.dot_style,
        legend_loc=ns.legend_loc,
        mask_label=ns.mask_label,
        mask_angle=ns.mask_angle,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


# ----------------------- registration API --------------------- #
def add_plot_temporal(
    subparsers: argparse._SubParsersAction,
) -> None:
    """
    Register the temporal scatter plotting subcommand:
      - plot-temporal-uncertainty
    """
    p = subparsers.add_parser(
        "plot-temporal-uncertainty",
        help="Polar scatter for multiple series (columns).",
        description=(
            "Visualize one or more columns over angle with optional "
            "per-series normalization."
        ),
    )
    p.add_argument(
        "input",
        type=str,
        help="Input table (CSV/Parquet/…).",
    )
    p.add_argument(
        "--format",
        type=str,
        default=None,
        help="Explicit input format.",
    )

    # columns / labels
    p.add_argument(
        "--q-cols",
        nargs="+",
        default=["auto"],
        help=(
            "Columns to plot. Use 'auto' to detect quantiles or "
            "provide names (space/comma separated)."
        ),
    )
    p.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Legend labels for series (match --q-cols).",
    )
    p.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Used only for NaN alignment (order is index).",
    )

    # plot options
    p.add_argument(
        "--acov",
        type=str,
        default="default",
        choices=[
            "default",
            "half_circle",
            "quarter_circle",
            "eighth_circle",
        ],
        help="Angular span.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title.",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="tab10",
        help="Matplotlib colormap.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Marker alpha.",
    )
    p.add_argument(
        "--s",
        type=int,
        default=25,
        help="Marker size.",
    )
    p.add_argument(
        "--dot-style",
        dest="dot_style",
        type=str,
        default="o",
        help="Matplotlib marker style (e.g. 'o', '.', 'x').",
    )
    p.add_argument(
        "--legend-loc",
        dest="legend_loc",
        type=str,
        default="upper right",
        help="Legend location (e.g. 'best').",
    )

    # toggles
    add_bool_flag(p, "normalize", True, "Normalize each series.", "Do not.")
    add_bool_flag(p, "show-grid", True, "Show polar grid.", "Hide grid.")
    add_bool_flag(p, "mask-label", False, "Hide labels.", "Show labels.")
    add_bool_flag(p, "mask-angle", True, "Hide angle ticks.", "Show ticks.")

    # output
    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save to path instead of showing.",
    )

    p.set_defaults(func=_cmd_plot_temporal_uncertainty)
