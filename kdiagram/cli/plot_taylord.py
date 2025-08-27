# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Taylor diagram CLI entrypoints.

This module wires three commands into the main CLI:

- plot-taylor-diagram
- plot-taylor-diagram-in
- taylor-diagram

Use ``add_plot_taylord(subparsers)`` to register them.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..plot.taylor_diagram import (
    plot_taylor_diagram,
    plot_taylor_diagram_in,
    taylor_diagram,
)
from ._utils import (
    ColumnsListAction,
    _collect_point_preds,
    _parse_float_list,
    _parse_norm_range,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_taylord"]


# ----------------------------- commands ----------------------------


def _cmd_plot_taylor_basic(ns: argparse.Namespace) -> None:
    """CLI runner for plot-taylor-diagram."""
    df = load_df(ns.input, format=ns.format)

    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy()

    # each group -> exactly one column
    yps, names = _collect_point_preds(df, ns)

    plot_taylor_diagram(
        *yps,
        reference=y_true,
        names=list(names) if names else None,
        acov=ns.acov,
        zero_location=ns.zero_location,
        direction=ns.direction,
        only_points=ns.only_points,
        ref_color=ns.ref_color,
        draw_ref_arc=ns.draw_ref_arc,
        angle_to_corr=ns.angle_to_corr,
        marker=ns.marker,
        corr_steps=ns.corr_steps,
        fig_size=ns.figsize,
        title=ns.title,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


def _cmd_plot_taylor_in(ns: argparse.Namespace) -> None:
    """CLI runner for plot-taylor-diagram-in."""
    df = load_df(ns.input, format=ns.format)

    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy()

    yps, names = _collect_point_preds(df, ns)

    plot_taylor_diagram_in(
        *yps,
        reference=y_true,
        names=list(names) if names else None,
        acov=ns.acov,
        zero_location=ns.zero_location,
        direction=ns.direction,
        only_points=ns.only_points,
        ref_color=ns.ref_color,
        draw_ref_arc=ns.draw_ref_arc,
        angle_to_corr=ns.angle_to_corr,
        marker=ns.marker,
        corr_steps=ns.corr_steps,
        cmap=ns.cmap,
        shading=ns.shading,
        shading_res=ns.shading_res,
        radial_strategy=ns.radial_strategy,
        norm_c=ns.norm_c,
        norm_range=_parse_norm_range(ns.norm_range),
        cbar=ns.cbar,
        fig_size=ns.figsize,
        title=ns.title,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


def _cmd_taylor_diagram(ns: argparse.Namespace) -> None:
    """
    Flexible runner for taylor-diagram.

    Two modes:
    (A) stats-mode: --stddev & --corrcoef
    (B) data-mode : --y-true + preds
    """
    stds = _parse_float_list(ns.stddev)
    cors = _parse_float_list(ns.corrcoef)
    names = list(ns.names) if ns.names else None

    ref_props: dict[str, Any] = {}
    if ns.ref_label:
        ref_props["label"] = ns.ref_label
    if ns.ref_lc:
        ref_props["lc"] = ns.ref_lc
    if ns.ref_color:
        ref_props["color"] = ns.ref_color
    if ns.ref_lw is not None:
        ref_props["lw"] = ns.ref_lw

    size_props: dict[str, Any] | None = None
    if ns.tick_size is not None or ns.label_size is not None:
        size_props = {
            "ticks": ns.tick_size if ns.tick_size is not None else 10,
            "label": ns.label_size if ns.label_size is not None else 12,
        }

    # ----- stats-mode
    if stds is not None or cors is not None:
        if not stds or not cors or len(stds) != len(cors):
            raise SystemExit(
                "--stddev and --corrcoef must be provided "
                "with equal lengths."
            )
        taylor_diagram(
            stddev=stds,
            corrcoef=cors,
            y_preds=None,
            reference=None,
            names=names,
            ref_std=ns.ref_std if ns.ref_std is not None else 1.0,
            cmap=ns.cmap,
            draw_ref_arc=ns.draw_ref_arc,
            radial_strategy=ns.radial_strategy,
            norm_c=ns.norm_c,
            power_scaling=ns.power_scaling,
            marker=ns.marker,
            ref_props=ref_props if ref_props else None,
            fig_size=ns.figsize,
            size_props=size_props,
            title=ns.title,
            savefig=str(ns.savefig) if ns.savefig else None,
        )
        return

    # ----- data-mode
    if not ns.y_true:
        raise SystemExit(
            "Provide --y-true (or --true-col) and predictions "
            "via --pred/--pred-cols/--model."
        )

    df = load_df(ns.input, format=ns.format)
    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy()

    yps, names_cli = _collect_point_preds(df, ns)
    if names is None:
        names = list(names_cli) if names_cli else None

    taylor_diagram(
        stddev=None,
        corrcoef=None,
        y_preds=yps,
        reference=y_true,
        names=names,
        ref_std=ns.ref_std if ns.ref_std is not None else 1.0,
        cmap=ns.cmap,
        draw_ref_arc=ns.draw_ref_arc,
        radial_strategy=ns.radial_strategy,
        norm_c=ns.norm_c,
        power_scaling=ns.power_scaling,
        marker=ns.marker,
        ref_props=ref_props if ref_props else None,
        fig_size=ns.figsize,
        size_props=size_props,
        title=ns.title,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


# ----------------------------- parsers -----------------------------


def add_plot_taylor_diagram(
    sub: argparse._SubParsersAction,
) -> None:
    """Register `plot-taylor-diagram`."""
    p = sub.add_parser(
        "plot-taylor-diagram",
        help=(
            "Taylor diagram for predictions vs. reference "
            "(std dev + correlation)."
        ),
        description=(
            "Plot a Taylor diagram (polar): radius is std dev, "
            "angle encodes correlation. Provide a ground-truth "
            "column and one column per model."
        ),
    )
    # I/O + y
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input path (try --input if omitted).",
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
    p.add_argument(
        "--y-true",
        "--true-col",
        dest="y_true",
        required=True,
        help="Ground-truth column.",
    )

    # predictions
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec 'name:col'. Repeat per model.",
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=("Prediction columns (CSV or tokens). Repeat to " "add models."),
    )
    p.add_argument(
        "--pred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_cols",
        help="Alias for --pred.",
    )
    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Model names (CSV or tokens).",
    )

    # diagram settings
    p.add_argument(
        "--acov",
        default="half_circle",
        choices=["default", "half_circle"],
        help="Angular coverage.",
    )
    p.add_argument(
        "--zero-location",
        default="W",
        choices=["N", "NE", "E", "S", "SW", "W", "NW", "SE"],
        help="Where corr=1 sits on the plot.",
    )
    p.add_argument(
        "--direction",
        type=int,
        default=-1,
        choices=[-1, 1],
        help="Angle direction (CW=-1, CCW=1).",
    )
    add_bool_flag(
        p,
        "only-points",
        False,
        "Plot only markers (no radial lines).",
        "Plot markers and radial lines.",
    )
    p.add_argument(
        "--ref-color",
        default="red",
        help="Reference color.",
    )
    add_bool_flag(
        p,
        "draw-ref-arc",
        True,
        "Draw reference std dev arc.",
        "Do not draw reference arc.",
    )
    add_bool_flag(
        p,
        "angle-to-corr",
        True,
        "Label angles with correlation.",
        "Label with degrees.",
    )
    p.add_argument(
        "--marker",
        default="o",
        help="Marker style.",
    )
    p.add_argument(
        "--corr-steps",
        type=int,
        default=6,
        help="# of correlation ticks.",
    )

    # figure
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(10.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Figure title.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure.",
    )
    p.set_defaults(func=_cmd_plot_taylor_basic)


def add_plot_taylor_diagram_in(
    sub: argparse._SubParsersAction,
) -> None:
    """Register `plot-taylor-diagram-in`."""
    p = sub.add_parser(
        "plot-taylor-diagram-in",
        help="Taylor diagram with background colormap.",
        description=(
            "Paint a background mesh (e.g., correlation or "
            "normalized radius) using a chosen strategy."
        ),
    )
    # I/O + y
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input path (try --input if omitted).",
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
    p.add_argument(
        "--y-true",
        "--true-col",
        dest="y_true",
        required=True,
        help="Ground-truth column.",
    )

    # predictions
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec 'name:col'. Repeat per model.",
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=("Prediction columns (CSV or tokens). Repeat to " "add models."),
    )
    p.add_argument(
        "--pred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_cols",
        help="Alias for --pred.",
    )
    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Model names (CSV or tokens).",
    )

    # diagram settings
    p.add_argument(
        "--acov",
        default=None,
        choices=["default", "half_circle"],
        help="Angular coverage.",
    )
    p.add_argument(
        "--zero-location",
        default="E",
        choices=["N", "NE", "E", "S", "SW", "W", "NW", "SE"],
        help="Where corr=1 sits on the plot.",
    )
    p.add_argument(
        "--direction",
        type=int,
        default=-1,
        choices=[-1, 1],
        help="Angle direction (CW=-1, CCW=1).",
    )
    add_bool_flag(
        p,
        "only-points",
        False,
        "Plot only markers (no radial lines).",
        "Plot markers and radial lines.",
    )
    p.add_argument(
        "--ref-color",
        default="red",
        help="Reference color.",
    )
    add_bool_flag(
        p,
        "draw-ref-arc",
        True,
        "Draw reference std dev arc.",
        "Do not draw reference arc.",
    )
    add_bool_flag(
        p,
        "angle-to-corr",
        True,
        "Label angles with correlation.",
        "Label with degrees.",
    )
    p.add_argument(
        "--marker",
        default="o",
        help="Marker style.",
    )
    p.add_argument(
        "--corr-steps",
        type=int,
        default=6,
        help="# of correlation ticks.",
    )

    # background
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Background colormap.",
    )
    p.add_argument(
        "--shading",
        default="auto",
        choices=["auto", "gouraud", "nearest"],
        help="pcolormesh shading.",
    )
    p.add_argument(
        "--shading-res",
        type=int,
        default=300,
        help="Background mesh resolution.",
    )
    p.add_argument(
        "--radial-strategy",
        default=None,
        choices=[
            "convergence",
            "norm_r",
            "performance",
            "rwf",
            "center_focus",
        ],
        help=(
            "Strategy for background. Unsupported options "
            "degrade to 'performance' with a warning."
        ),
    )
    add_bool_flag(
        p,
        "norm-c",
        False,
        "Normalize background colors.",
        "Do not normalize background colors.",
    )
    p.add_argument(
        "--norm-range",
        type=parse_figsize,
        default=None,
        help="Normalization range 'lo,hi'.",
    )
    add_bool_flag(
        p,
        "cbar",
        False,
        "Show colorbar.",
        "Hide colorbar.",
    )

    # figure
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(10.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Figure title.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure.",
    )
    p.set_defaults(func=_cmd_plot_taylor_in)


def add_taylor_diagram(sub: argparse._SubParsersAction) -> None:
    """Register `taylor-diagram` (stats-mode or data-mode)."""
    p = sub.add_parser(
        "taylor-diagram",
        help=(
            "Flexible Taylor diagram: either precomputed stats "
            "or ground-truth + model columns."
        ),
        description=(
            "Pass --stddev/--corrcoef (stats-mode) or "
            "--y-true + preds (data-mode). Optional background "
            "via --cmap and --radial-strategy."
        ),
    )
    # I/O (+ optional y for data-mode)
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input path (data-mode).",
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
    p.add_argument(
        "--y-true",
        "--true-col",
        dest="y_true",
        default=None,
        help="Ground-truth column (data-mode).",
    )

    # predictions for data-mode
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec 'name:col'. Repeat per model.",
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=("Prediction columns (CSV or tokens). Repeat to " "add models."),
    )
    p.add_argument(
        "--pred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_cols",
        help="Alias for --pred.",
    )

    # stats-mode inputs
    p.add_argument(
        "--stddev",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        help="Std dev values (CSV or tokens).",
    )
    p.add_argument(
        "--corrcoef",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        help="Correlation values (CSV or tokens).",
    )

    # names (both modes)
    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Labels for models (CSV or tokens).",
    )

    # reference styling / params
    p.add_argument(
        "--ref-std",
        type=float,
        default=None,
        help="Reference std (stats-mode or override).",
    )
    p.add_argument(
        "--ref-label",
        default=None,
        help="Reference legend label.",
    )
    p.add_argument(
        "--ref-lc",
        default=None,
        help="Reference arc line color/style.",
    )
    p.add_argument(
        "--ref-color",
        default=None,
        help="Reference point color (stats-mode).",
    )
    p.add_argument(
        "--ref-lw",
        type=float,
        default=None,
        help="Reference arc linewidth.",
    )
    add_bool_flag(
        p,
        "draw-ref-arc",
        False,
        "Draw reference std dev arc.",
        "Use reference point.",
    )

    # background (optional)
    p.add_argument(
        "--cmap",
        default=None,
        help="Background colormap (e.g., 'viridis').",
    )
    p.add_argument(
        "--radial-strategy",
        default="rwf",
        choices=[
            "rwf",
            "convergence",
            "center_focus",
            "performance",
        ],
        help="Background strategy.",
    )
    add_bool_flag(
        p,
        "norm-c",
        False,
        "Normalize background to [0,1].",
        "Do not normalize background.",
    )
    p.add_argument(
        "--power-scaling",
        type=float,
        default=1.0,
        help="Exponent for normalized background.",
    )
    p.add_argument(
        "--marker",
        default="o",
        help="Marker style.",
    )

    # figure & text sizes
    p.add_argument(
        "--tick-size",
        type=int,
        default=None,
        help="Tick font size.",
    )
    p.add_argument(
        "--label-size",
        type=int,
        default=None,
        help="Label font size.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 6.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Figure title.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure.",
    )
    p.set_defaults(func=_cmd_taylor_diagram)


# ----------------------------- registrar ---------------------------


def add_plot_taylord(sub: argparse._SubParsersAction) -> None:
    """
    Register all Taylor diagram subcommands on a parser.

    Parameters
    ----------
    sub :
        The result of ``parser.add_subparsers()``.
    """
    add_plot_taylor_diagram(sub)
    add_plot_taylor_diagram_in(sub)
    add_taylor_diagram(sub)
