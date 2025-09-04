# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..plot.uncertainty import plot_velocity
from ._utils import (
    _split_tokens,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_velocities"]


def _cmd_plot_velocity(ns: argparse.Namespace) -> None:
    df: pd.DataFrame = load_df(ns.input, format=ns.format)

    q50_cols = _split_tokens(ns.q50_cols or [])
    ensure_columns(df, q50_cols)
    df = ensure_numeric(df, q50_cols, copy=True, errors="raise")

    plot_velocity(
        df=df,
        q50_cols=q50_cols,
        theta_col=ns.theta_col,
        cmap=ns.cmap,
        acov=ns.acov,
        normalize=ns.normalize,
        use_abs_color=ns.use_abs_color,
        figsize=ns.figsize,
        title=ns.title,
        s=ns.s,
        alpha=ns.alpha,
        show_grid=ns.show_grid,
        savefig=str(ns.savefig) if ns.savefig else None,
        cbar=ns.cbar,
        mask_angle=ns.mask_angle,
    )


def add_plot_velocities(
    subparsers: argparse._SubParsersAction,
) -> None:
    """
    Register the velocity plotting subcommand:
      - plot-velocity
    """
    p = subparsers.add_parser(
        "plot-velocity",
        help="Polar velocity plot from sequential q50 columns.",
        description=(
            "Visualize per-location velocity derived from a sequence "
            "of median (q50) columns across time or steps."
        ),
    )

    p.add_argument(
        "input",
        type=str,
        help="Input table path (CSV/Parquet/…).",
    )
    p.add_argument(
        "--format",
        type=str,
        default=None,
        help="Explicit input format; else inferred.",
    )
    p.add_argument(
        "--q50-cols",
        nargs="+",
        required=True,
        help=(
            "List of q50/median columns in order (space- or comma-separated)."
        ),
    )
    p.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Optional column for NaN alignment.",
    )

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
        help="Angular span of the polar plot.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(9.0, 9.0),
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
        default="viridis",
        help="Matplotlib colormap.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Marker alpha.",
    )
    p.add_argument(
        "--s",
        type=int,
        default=30,
        help="Marker size.",
    )

    # toggles
    add_bool_flag(
        p, "normalize", True, "Normalize series.", "Do not normalize."
    )
    add_bool_flag(
        p,
        "use-abs-color",
        True,
        "Color by |velocity|.",
        "Color by signed velocity.",
    )
    add_bool_flag(
        p, "show-grid", True, "Show polar grid.", "Hide polar grid."
    )
    add_bool_flag(p, "cbar", True, "Show colorbar.", "Hide colorbar.")
    add_bool_flag(
        p,
        "mask-angle",
        False,
        "Hide angular tick labels.",
        "Show angular tick labels.",
    )

    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save figure to this path instead of showing.",
    )

    p.set_defaults(func=_cmd_plot_velocity)
