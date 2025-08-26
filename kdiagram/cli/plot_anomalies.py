# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..plot.uncertainty import plot_anomaly_magnitude
from ._utils import (
    ColumnsPairAction, 
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_anomalies"]


def _cmd_plot_anomaly_magnitude(ns: argparse.Namespace) -> None:
    qlow, qup = ns.q_cols

    drop_cols = [ns.actual_col, qlow, qup] if ns.dropna else None

    df: pd.DataFrame = load_df(
        ns.input,
        format=ns.format,
        dropna=drop_cols,
    )

    # only essential cols; theta may be absent and is optional
    ensure_columns(df, [ns.actual_col, qlow, qup])

    # coerce numeric for actual + bounds
    df = ensure_numeric(
        df, [ns.actual_col, qlow, qup], copy=True, errors="raise"
    )

    plot_anomaly_magnitude(
        df=df,
        actual_col=ns.actual_col,
        q_cols=[qlow, qup],
        theta_col=ns.theta_col,
        acov=ns.acov,
        title=ns.title,
        figsize=ns.figsize,
        cmap_under=ns.cmap_under,
        cmap_over=ns.cmap_over,
        s=ns.s,
        alpha=ns.alpha,
        show_grid=ns.show_grid,
        verbose=ns.verbose,
        cbar=ns.cbar,
        savefig=str(ns.savefig) if ns.savefig else None,
        mask_angle=ns.mask_angle,
    )


def add_plot_anomalies(
    subparsers: argparse._SubParsersAction,
) -> None:
    """
    Register anomaly-related plotting subcommands.
    Currently: plot-anomaly-magnitude
    """
    p = subparsers.add_parser(
        "plot-anomaly-magnitude",
        help=(
            "Polar scatter of anomaly magnitudes for "
            "under/over predictions."
        ),
        description=(
            "Plot magnitude of violations where actual falls "
            "below lower bound or above upper bound."
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
    p.add_argument(
        "--actual-col",
        type=str,
        required=True,
        help="Observed target column.",
    )
    p.add_argument(
        "--q-cols",
        action=ColumnsPairAction,
        nargs="+",                  # accept 1 token ("low,up") or 2 tokens
        required=True,
        metavar="LOW,UP",           # single string, not a tuple
        help=(
            "Two columns 'lower,upper' that define the interval. "
            "Accepts 'low,up' or two tokens."
        ),
    )
    p.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Ordering column (optional).",
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
        help="Angular span.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Anomaly Magnitude Polar Plot",
        help="Figure title.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--cmap-under",
        dest="cmap_under",
        type=str,
        default="Blues",
        help="Colormap for under-predictions.",
    )
    p.add_argument(
        "--cmap-over",
        dest="cmap_over",
        type=str,
        default="Reds",
        help="Colormap for over-predictions.",
    )
    p.add_argument(
        "--s",
        type=int,
        default=30,
        help="Marker size.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Marker alpha (0..1).",
    )
    add_bool_flag(p, "show-grid", True, "Show polar grid.", "Hide grid.")
    add_bool_flag(p, "cbar", False, "Show colorbar.", "Hide colorbar.")
    add_bool_flag(p, "mask-angle", False, "Hide angle ticks.", "Show.")
    add_bool_flag(p, "dropna", True, "Drop NaNs in needed cols.", "Keep.")

    p.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level.",
    )
    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save to path instead of showing.",
    )

    p.set_defaults(func=_cmd_plot_anomaly_magnitude)
