# kdiagram/cli/plot_drift.py
# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..plot.uncertainty import (
    plot_model_drift,
    plot_uncertainty_drift,
)
from ._utils import (
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_cols_pair,
    parse_figsize,
)

__all__ = ["add_plot_drift"]


def _flatten_pairs(pairs: Iterable[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for a, b in pairs:
        out.extend([a, b])
    return out


def _cmd_plot_model_drift(ns: argparse.Namespace) -> None:
    # Accept either --q-cols (repeated) OR parallel lists
    q_pairs: list[tuple[str, str]] | None = ns.q_cols
    q10_cols: list[str] | None = ns.q10_cols
    q90_cols: list[str] | None = ns.q90_cols

    if q_pairs:
        # columns used for drop/validation
        essential = _flatten_pairs(q_pairs)
        q10_cols = [a for a, _ in q_pairs]
        q90_cols = [b for _, b in q_pairs]
    else:
        if not (q10_cols and q90_cols):
            raise SystemExit(
                "Provide either --q-cols (repeated) or both "
                "--q10-cols and --q90-cols."
            )
        if len(q10_cols) != len(q90_cols):
            raise SystemExit(
                "--q10-cols and --q90-cols must have same length."
            )
        essential = [*q10_cols, *q90_cols]
        q_pairs = list(zip(q10_cols, q90_cols))

    # add color metrics if any
    if ns.color_metric_cols:
        essential += list(ns.color_metric_cols)

    df: pd.DataFrame = load_df(
        ns.input,
        format=ns.format,
        dropna=essential if ns.dropna else None,
    )

    ensure_columns(df, essential)

    # coerce numeric for all involved columns
    df = ensure_numeric(
        df,
        essential,
        copy=True,
        errors="raise",
    )

    # horizons defaulting handled in plot function
    plot_model_drift(
        df=df,
        q_cols=q_pairs,
        q10_cols=None,  # not needed when q_cols is given
        q90_cols=None,
        horizons=ns.horizons,
        color_metric_cols=ns.color_metric_cols,
        acov=ns.acov,
        value_label=ns.value_label,
        cmap=ns.cmap,
        figsize=ns.figsize,
        title=ns.title,
        show_grid=ns.show_grid,
        annotate=ns.annotate,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


def _cmd_plot_uncertainty_drift(ns: argparse.Namespace) -> None:
    qlow = ns.qlow_cols
    qup = ns.qup_cols

    if len(qlow) != len(qup):
        raise SystemExit("--qlow-cols and --qup-cols must have same length.")

    essential = [*qlow, *qup]

    df: pd.DataFrame = load_df(
        ns.input,
        format=ns.format,
        dropna=essential if ns.dropna else None,
    )

    ensure_columns(df, essential)
    df = ensure_numeric(df, essential, copy=True, errors="raise")

    plot_uncertainty_drift(
        df=df,
        qlow_cols=qlow,
        qup_cols=qup,
        dt_labels=ns.dt_labels,
        theta_col=ns.theta_col,
        acov=ns.acov,
        base_radius=ns.base_radius,
        band_height=ns.band_height,
        cmap=ns.cmap,
        label=ns.label,
        alpha=ns.alpha,
        figsize=ns.figsize,
        title=ns.title,
        show_grid=ns.show_grid,
        show_legend=ns.show_legend,
        mask_angle=ns.mask_angle,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


def add_plot_drift(
    subparsers: argparse._SubParsersAction,
) -> None:
    """
    Register drift-related plotting subcommands:
      - plot-model-drift
      - plot-uncertainty-drift
    """

    # ---- plot-model-drift ---- #
    p1 = subparsers.add_parser(
        "plot-model-drift",
        help=(
            "Polar bar chart of average interval width per forecast horizon."
        ),
        description=(
            "Visualize how uncertainty (or another metric) evolves "
            "with horizon."
        ),
    )
    p1.add_argument(
        "input",
        type=str,
        help="Input table (CSV/Parquet/…).",
    )
    p1.add_argument(
        "--format",
        type=str,
        default=None,
        help="Explicit input format.",
    )

    # either repeated --q-cols LOW,UP or parallel lists
    p1.add_argument(
        "--q-cols",
        action="append",
        type=parse_cols_pair,
        metavar="LOW,UP",
        help=(
            "Repeatable: pair of columns for a horizon. "
            "Example: --q-cols q10_h1,q90_h1 --q-cols q10_h2,q90_h2"
        ),
    )
    p1.add_argument(
        "--q10-cols",
        nargs="+",
        default=None,
        help="Lower-quantile columns by horizon.",
    )
    p1.add_argument(
        "--q90-cols",
        nargs="+",
        default=None,
        help="Upper-quantile columns by horizon.",
    )
    p1.add_argument(
        "--horizons",
        nargs="+",
        default=None,
        help="Labels for horizons (order matches q-cols).",
    )
    p1.add_argument(
        "--color-metric-cols",
        nargs="+",
        default=None,
        help=(
            "Optional columns used for bar colors (e.g. RMSE per horizon)."
        ),
    )

    p1.add_argument(
        "--acov",
        type=str,
        default="quarter_circle",
        choices=[
            "default",
            "half_circle",
            "quarter_circle",
            "eighth_circle",
        ],
        help="Angular span.",
    )
    p1.add_argument(
        "--value-label",
        type=str,
        default="Uncertainty Width (Q90 - Q10)",
        help="Radial label text.",
    )
    p1.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap.",
    )
    p1.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8, 8),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p1.add_argument(
        "--title",
        type=str,
        default="Model Forecast Drift Over Time",
        help="Figure title.",
    )

    add_bool_flag(p1, "show-grid", True, "Show polar grid.", "Hide grid.")
    add_bool_flag(p1, "annotate", True, "Annotate bar values.", "Do not.")
    add_bool_flag(p1, "dropna", True, "Drop NaNs in needed cols.", "Keep.")

    p1.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save to path instead of showing.",
    )

    p1.set_defaults(func=_cmd_plot_model_drift)

    # ---- plot-uncertainty-drift ---- #
    p2 = subparsers.add_parser(
        "plot-uncertainty-drift",
        help=("Polar rings of normalized interval widths across time steps."),
        description=(
            "Draw a ring per time step (year/horizon) where radius "
            "encodes normalized interval width."
        ),
    )
    p2.add_argument(
        "input",
        type=str,
        help="Input table (CSV/Parquet/...).",
    )
    p2.add_argument(
        "--format",
        type=str,
        default=None,
        help="Explicit input format.",
    )
    p2.add_argument(
        "--qlow-cols",
        nargs="+",
        required=True,
        help="Lower-quantile columns ordered by time.",
    )
    p2.add_argument(
        "--qup-cols",
        nargs="+",
        required=True,
        help="Upper-quantile columns ordered by time.",
    )
    p2.add_argument(
        "--dt-labels",
        nargs="+",
        default=None,
        help="Labels for rings (e.g. years).",
    )
    p2.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Ordering column (currently ignored).",
    )
    p2.add_argument(
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
    p2.add_argument(
        "--base-radius",
        dest="base_radius",
        type=float,
        default=0.15,
        help="Base radius step between rings.",
    )
    p2.add_argument(
        "--band-height",
        dest="band_height",
        type=float,
        default=0.15,
        help="Height scale for normalized widths.",
    )
    p2.add_argument(
        "--cmap",
        type=str,
        default="tab10",
        help="Colormap for rings.",
    )
    p2.add_argument(
        "--label",
        type=str,
        default="Year",
        help="Legend title label.",
    )
    p2.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Line alpha.",
    )
    p2.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(9, 9),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p2.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title.",
    )

    add_bool_flag(p2, "show-grid", True, "Show polar grid.", "Hide grid.")
    add_bool_flag(p2, "show-legend", True, "Show legend.", "Hide legend.")
    add_bool_flag(p2, "mask-angle", True, "Hide angle ticks.", "Show.")
    add_bool_flag(p2, "dropna", True, "Drop NaNs in needed cols.", "Keep.")

    p2.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save to path instead of showing.",
    )

    p2.set_defaults(func=_cmd_plot_uncertainty_drift)
