# kdiagram/cli/plot_credibility.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from kdiagram.plot.probabilistic import plot_credibility_bands

from ._utils import (
    load_df,
    ensure_columns,
    ensure_numeric,
    parse_figsize,
    add_bool_flag,
    ColumnsListAction,
)

def _cmd_plot_credibility_bands(ns: argparse.Namespace) -> None:
    q_cols: List[str] = list(ns.q_cols or [])
    if len(q_cols) != 3:
        raise SystemExit(
            "--q-cols expects three names (low,med,up). "
            "Use 'low,med,up' or three tokens."
        )

    drop_cols = q_cols + [ns.theta_col]

    df = load_df(
        ns.input,
        format=ns.format,
        dropna=drop_cols if ns.dropna else None,
    )

    ensure_columns(df, drop_cols, error="raise")
    df = ensure_numeric(df, drop_cols, copy=True, errors="raise")

    fill_kws = {}
    if ns.fill_alpha is not None:
        fill_kws["alpha"] = ns.fill_alpha

    plot_credibility_bands(
        df=df,
        q_cols=(q_cols[0], q_cols[1], q_cols[2]),
        theta_col=ns.theta_col,
        theta_period=ns.theta_period,
        theta_bins=ns.theta_bins,
        title=ns.title,
        figsize=ns.figsize,
        color=ns.color,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
        **fill_kws,
    )


def add_plot_credibility(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """
    Register the 'plot-credibility-bands' command.
    """
    p = subparsers.add_parser(
        "plot-credibility-bands",
        help="Polar credibility bands from (q_low, q50, q_up).",
        description=(
            "Bin along an angular driver (e.g., hour/month) and "
            "plot mean median with a shaded band between mean "
            "lower/upper quantiles."
        ),
    )

    # I/O
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path.",
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
        help="Input format override.",
    )

    # Columns
    p.add_argument(
        "--q-cols",
        action=ColumnsListAction,
        nargs="+",
        required=True,
        dest="q_cols",
        metavar="LOW,MED,UP",
        help=(
            "Three columns for (lower, median, upper). "
            "Accepts 'low,med,up' or three tokens."
        ),
    )
    p.add_argument(
        "--theta-col",
        "--theta",
        required=True,
        dest="theta_col",
        help="Column used for angular binning.",
    )
    p.add_argument(
        "--theta-period",
        type=float,
        default=None,
        help="Cycle length for theta (e.g., 24 or 12).",
    )
    p.add_argument(
        "--theta-bins",
        type=int,
        default=24,
        help="Number of angular bins.",
    )

    # Style
    p.add_argument(
        "--title",
        default="Forecast Credibility Bands",
        help="Figure title.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--color",
        default="#3498DB",
        help="Fill color for the band.",
    )
    p.add_argument(
        "--fill-alpha",
        type=float,
        default=None,
        help="Override band alpha (fill_between).",
    )
    add_bool_flag(p, "show-grid", True, "Show grid.", "Hide grid.")
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show labels.",
    )

    # Output
    add_bool_flag(
        p,
        "dropna",
        True,
        "Drop rows with NaNs in needed columns.",
        "Do not drop.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Path to save figure.",
    )

    p.set_defaults(func=_cmd_plot_credibility_bands)
    return p
