# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse

import numpy as np

from ..plot.uncertainty import (
    plot_coverage,
    plot_coverage_diagnostic,
)
from ._utils import (
    ColumnsPairAction, 
    ColumnsListAction, 
    ensure_columns,
    ensure_numeric,
    load_df,
    resolve_ytrue_preds, 
    _parse_q_levels, 
    _infer_figsize,
    split_csv
)

__all__ = [
    "add_plot_coverage_subparser",
    "add_plot_coverage_diagnostic_subparser",
    "add_plot_coverages",
    "cmd_plot_coverage",
    "cmd_plot_coverage_diagnostic",
]


# ------------------------------ cmds --------------------------------

def cmd_plot_coverage(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # Resolve y_true + prediction specs (flexible flags or auto)
    y_true_name, specs = resolve_ytrue_preds(ns, df)

    # Validate and coerce
    ensure_columns(df, [y_true_name], error="raise")
    df = ensure_numeric(df, [y_true_name], copy=True, errors="raise")
    y_true = df[y_true_name].to_numpy(dtype=float)

    # Ensure all model columns exist & are numeric
    need_cols: list[str] = []
    for ms in specs:
        need_cols.extend(ms.cols)
    if need_cols:
        ensure_columns(df, need_cols, error="raise")
        df = ensure_numeric(df, need_cols, copy=True, errors="raise")

    # Build arrays per model (each may be 1 col or quantile set)
    y_preds: list[np.ndarray] = [
        df[ms.cols].to_numpy(dtype=float) for ms in specs
    ]

    # Names: CLI override wins, else from specs
    if ns.names:
        names = list(ns.names)  # already a list from the action
    else:
        names = [ms.name for ms in specs]


    # Quantile levels (optional; None is fine if not a q-set)
    q_levels = _parse_q_levels(ns.q_levels)

    figsize = _infer_figsize(ns.figsize)

    plot_coverage(
        y_true,
        *y_preds,
        names=names,
        q=q_levels,
        kind=ns.kind,
        cmap=ns.cmap,
        pie_startangle=ns.pie_startangle,
        pie_autopct=ns.pie_autopct,
        radar_color=ns.radar_color,
        radar_fill_alpha=ns.radar_fill_alpha,
        radar_line_style=ns.radar_line_style,
        cov_fill=ns.radar_fill,
        figsize=figsize,
        title=ns.title,
        savefig=ns.savefig,
        verbose=ns.verbose,
    )

def cmd_plot_coverage_diagnostic(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # q-cols: from ColumnsPairAction or comma string fallback
    if isinstance(ns.q_cols, (list, tuple)):
        qcols = list(ns.q_cols)
    else:
        qcols = split_csv(ns.q_cols)
    if len(qcols) != 2:
        raise ValueError("--q-cols expects two names [lower,upper].")

    need_cols = [ns.actual_col, *qcols]
    ensure_columns(df, need_cols, error="raise")
    df = ensure_numeric(df, need_cols, copy=True, errors="raise")

    levels = None
    if ns.gradient_levels:
        levels = [float(x) for x in split_csv(ns.gradient_levels)]

    figsize = tuple(ns.figsize) if ns.figsize else (8.0, 8.0)

    plot_coverage_diagnostic(
        df=df,
        actual_col=ns.actual_col,
        q_cols=qcols,
        theta_col=ns.theta_col,
        acov=ns.acov,
        figsize=figsize,
        title=ns.title,
        show_grid=ns.show_grid,
        cmap=ns.cmap,
        alpha=ns.alpha,
        s=ns.s,
        as_bars=ns.as_bars,
        coverage_line_color=ns.coverage_line_color,
        buffer_pts=ns.buffer_pts,
        fill_gradient=ns.fill_gradient,
        gradient_size=ns.gradient_size,
        gradient_cmap=ns.gradient_cmap,
        gradient_levels=levels,
        mask_angle=ns.mask_angle,
        savefig=ns.savefig,
        verbose=ns.verbose,
    )


# --------------------------- subparsers ------------------------------
def add_plot_coverage_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "plot-coverage",
        help="Plot overall coverage scores.",
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

    # --- Columns & models (all optional; resolver handles them) ---
    # y_true aliases
    p.add_argument(
        "--y-true",
        dest="y_true",
        default=None,
        help="Ground truth column.",
    )
    p.add_argument(
        "--actual-col",
        dest="actual_col",
        default=None,
        help="Alias of --y-true.",
    )
    p.add_argument(
        "--true-col",
        dest="actual_col",
        default=None,
        help="Alias of --y-true.",
    )

    # predictions: several styles, all repeatable
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec 'name:col1[,col2,...]'. "
            "Repeat for multiple models."
        ),
    )
    p.add_argument(
        "--pred",
        action="append",
        default=None,
        help=("Prediction COLS. CSV string. Repeatable."),
    )
    p.add_argument(
        "--y-pred",
        action="append",
        default=None,
        help=("Single prediction column. Repeatable."),
    )

    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Model names (CSV or space separated).",
    )

    p.add_argument(
        "--q-levels",
        default=None,
        help="Comma-separated quantiles.",
    )

    # style
    p.add_argument(
        "--kind",
        default="line",
        choices=["line", "bar", "pie", "radar"],
        help="Plot kind.",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap.",
    )
    p.add_argument(
        "--pie-startangle",
        type=float,
        default=140.0,
        help="Pie start angle.",
    )
    p.add_argument(
        "--pie-autopct",
        default="%1.1f%%",
        help="Pie autopct format.",
    )
    p.add_argument(
        "--radar-color",
        default="tab:blue",
        help="Radar line color.",
    )
    p.add_argument(
        "--radar-fill",
        action="store_true",
        help="Fill radar area.",
    )
    p.add_argument(
        "--radar-line-style",
        default="o-",
        help="Radar line/marker style.",
    )
    p.add_argument(
        "--radar-fill-alpha",
        type=float,
        default=0.25,
        help="Radar fill alpha.",
    )

    # figure & output
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Figure title.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save figure.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level.",
    )

    p.set_defaults(func=cmd_plot_coverage)
    return p

def add_plot_coverage_diagnostic_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "plot-coverage-diagnostic",
        help="Point-wise coverage diagnostic.",
    )

    # I/O: allow positional or -i/--input
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

    # columns (aliases + flexible pair)
    p.add_argument(
        "--actual-col",
        "--actual",
        dest="actual_col",
        required=True,
        help="Observed values column.",
    )
    p.add_argument(
        "--q-cols",
        action=ColumnsPairAction,
        nargs="+",
        required=True,
        metavar="LOW,UP",
        help=("Two cols defining interval: 'low up' or 'low,up'."),
    )
    p.add_argument(
        "--theta-col",
        "--theta",
        dest="theta_col",
        default=None,
        help="Ordering column (ignored).",
    )

    # geometry / style
    p.add_argument(
        "--acov",
        default="default",
        choices=[
            "default",
            "half_circle",
            "quarter_circle",
            "eighth_circle",
        ],
        help="Angular coverage span.",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--show-grid", action="store_true", help="Show grid.")
    p.add_argument("--cmap", default="RdYlGn", help="Point/bars cmap.")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Point/bar alpha.",
    )
    p.add_argument("--s", type=int, default=35, help="Marker size.")
    p.add_argument(
        "--as-bars",
        action="store_true",
        help="Use bars not scatter.",
    )
    p.add_argument(
        "--coverage-line-color",
        default="r",
        help="Avg coverage line color.",
    )
    p.add_argument(
        "--buffer-pts",
        type=int,
        default=500,
        help="Line smoothness samples.",
    )
    p.add_argument(
        "--fill-gradient",
        action="store_true",
        help="Fill radial background.",
    )
    p.add_argument(
        "--gradient-size",
        type=int,
        default=300,
        help="Gradient mesh size.",
    )
    p.add_argument(
        "--gradient-cmap",
        default="Greens",
        help="Background gradient cmap.",
    )
    p.add_argument(
        "--gradient-levels",
        default=None,
        help="Comma floats, e.g. 0.5,0.8,0.9",
    )
    p.add_argument(
        "--mask-angle",
        action="store_true",
        help="Hide angular tick labels.",
    )
    p.add_argument("--savefig", default=None, help="Path to save fig.")
    p.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level.",
    )

    p.set_defaults(func=cmd_plot_coverage_diagnostic)
    return p



def add_plot_coverages(
    subparsers: argparse._SubParsersAction,
) -> None:
    add_plot_coverage_subparser(subparsers)
    add_plot_coverage_diagnostic_subparser(subparsers)
