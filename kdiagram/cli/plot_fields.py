# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..plot.uncertainty import (
    plot_polar_heatmap,
    plot_polar_quiver,
    plot_radial_density_ring,
)
from ._utils import (
    _split_tokens,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_fields"]


def _cols_from_tokens(tokens: list[str] | None) -> list[str]:
    return _split_tokens(tokens or [])


def _cmd_plot_radial_density_ring(ns: argparse.Namespace) -> None:
    df: pd.DataFrame = load_df(ns.input, format=ns.format)

    targets = _cols_from_tokens(ns.target_cols)
    ensure_columns(df, targets)
    df = ensure_numeric(df, targets, copy=True, errors="raise")

    plot_radial_density_ring(
        df=df,
        kind=ns.kind,
        target_cols=targets,
        title=ns.title,
        r_label=ns.r_label,
        figsize=ns.figsize,
        cmap=ns.cmap,
        alpha=ns.alpha,
        cbar=ns.cbar,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_angle=ns.mask_angle,
        bandwidth=ns.bandwidth,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
        show_yticklabels=ns.show_yticklabels,
    )


def _add_cmd_ring(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "plot-radial-density-ring",
        help=(
            "Radial density ring from direct values, widths, or velocities."
        ),
        description=(
            "Visualize a 1D distribution as a ring. When kind=width "
            "or velocity, values are derived from the provided columns."
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
        "--kind",
        type=str,
        default="direct",
        choices=["direct", "width", "velocity"],
        help="How to interpret target columns.",
    )
    p.add_argument(
        "--target-cols",
        nargs="+",
        required=True,
        help=(
            "Columns used for the ring. Space- or comma-separated. "
            "For kind=direct provide 1+ columns; for kind=width "
            "usually two (low,up); for kind=velocity provide an "
            "ordered sequence (q50 over time)."
        ),
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title.",
    )
    p.add_argument(
        "--r-label",
        type=str,
        default=None,
        help="Radial label text.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
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
        default=0.8,
        help="Artist alpha.",
    )
    p.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Optional KDE bandwidth.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Save DPI.",
    )

    add_bool_flag(p, "cbar", True, "Show colorbar.", "Hide colorbar.")
    add_bool_flag(
        p, "show-grid", True, "Show polar grid.", "Hide polar grid."
    )
    add_bool_flag(
        p,
        "mask-angle",
        True,
        "Hide angular ticks.",
        "Show angular ticks.",
    )
    add_bool_flag(
        p,
        "show-yticklabels",
        False,
        "Show radial tick labels.",
        "Hide radial tick labels.",
    )

    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save figure to this path.",
    )

    p.set_defaults(func=_cmd_plot_radial_density_ring)


# ------------------------- command: quiver ------------------------- #
def _cmd_plot_polar_quiver(ns: argparse.Namespace) -> None:
    df: pd.DataFrame = load_df(ns.input, format=ns.format)

    cols = [ns.r_col, ns.theta_col, ns.u_col, ns.v_col]
    if ns.color_col:
        cols.append(ns.color_col)
    ensure_columns(df, cols)
    df = ensure_numeric(df, cols, copy=True, errors="raise")

    qkws = {}
    if ns.scale is not None:
        qkws["scale"] = ns.scale
    if ns.width is not None:
        qkws["width"] = ns.width

    plot_polar_quiver(
        df=df,
        r_col=ns.r_col,
        theta_col=ns.theta_col,
        u_col=ns.u_col,
        v_col=ns.v_col,
        color_col=ns.color_col,
        theta_period=ns.theta_period,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        mask_angle=ns.mask_angle,
        mask_radius=ns.mask_radius,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
        **qkws,
    )


def _add_cmd_quiver(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "plot-polar-quiver",
        help="Polar vector (quiver) plot.",
        description=(
            "Plot vectors specified by (u,v) at polar coords "
            "(theta,r). Theta is in radians."
        ),
    )
    p.add_argument("input", type=str, help="Input table path.")
    p.add_argument(
        "--format",
        type=str,
        default=None,
        help="Explicit input format; else inferred.",
    )
    p.add_argument("--r-col", type=str, required=True, help="Radius col.")
    p.add_argument(
        "--theta-col",
        type=str,
        required=True,
        help="Angle col (radians).",
    )
    p.add_argument("--u-col", type=str, required=True, help="U col.")
    p.add_argument("--v-col", type=str, required=True, help="V col.")
    p.add_argument(
        "--color-col",
        type=str,
        default=None,
        help="Optional color column.",
    )
    p.add_argument(
        "--theta-period",
        type=float,
        default=None,
        help="Period to wrap theta (e.g. 2*pi).",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument("--title", type=str, default=None, help="Title.")
    p.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Save DPI.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Quiver scale.",
    )
    p.add_argument(
        "--width",
        type=float,
        default=None,
        help="Quiver width.",
    )

    add_bool_flag(
        p, "show-grid", True, "Show polar grid.", "Hide polar grid."
    )
    add_bool_flag(
        p,
        "mask-angle",
        False,
        "Hide angular ticks.",
        "Show angular ticks.",
    )
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial ticks.",
        "Show radial ticks.",
    )

    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save figure to this path.",
    )

    p.set_defaults(func=_cmd_plot_polar_quiver)


# ------------------------ command: heatmap ------------------------- #
def _cmd_plot_polar_heatmap(ns: argparse.Namespace) -> None:
    df: pd.DataFrame = load_df(ns.input, format=ns.format)

    cols = [ns.r_col, ns.theta_col]
    ensure_columns(df, cols)
    df = ensure_numeric(df, cols, copy=True, errors="raise")

    plot_polar_heatmap(
        df=df,
        r_col=ns.r_col,
        theta_col=ns.theta_col,
        theta_period=ns.theta_period,
        r_bins=ns.r_bins,
        theta_bins=ns.theta_bins,
        statistic=ns.statistic,
        cbar_label=ns.cbar_label,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        mask_angle=ns.mask_angle,
        mask_radius=ns.mask_radius,
        show_grid=ns.show_grid,
        grid_props=None,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _add_cmd_heatmap(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "plot-polar-heatmap",
        help="Binned polar heatmap.",
        description=(
            "Aggregate values over (theta,r) bins to form a polar "
            "heatmap. Theta is in radians."
        ),
    )
    p.add_argument("input", type=str, help="Input table path.")
    p.add_argument(
        "--format",
        type=str,
        default=None,
        help="Explicit input format; else inferred.",
    )
    p.add_argument("--r-col", type=str, required=True, help="Radius col.")
    p.add_argument(
        "--theta-col",
        type=str,
        required=True,
        help="Angle col (radians).",
    )
    p.add_argument(
        "--theta-period",
        type=float,
        default=None,
        help="Period to wrap theta (e.g. 2*pi).",
    )
    p.add_argument(
        "--r-bins",
        type=int,
        default=20,
        help="Number of radial bins.",
    )
    p.add_argument(
        "--theta-bins",
        type=int,
        default=36,
        help="Number of angular bins.",
    )
    p.add_argument(
        "--statistic",
        type=str,
        default="count",
        help="Aggregation statistic (e.g. count, mean, sum).",
    )
    p.add_argument(
        "--cbar-label",
        type=str,
        default=None,
        help="Colorbar label text.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument("--title", type=str, default=None, help="Title.")
    p.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Save DPI.",
    )

    add_bool_flag(
        p, "show-grid", True, "Show polar grid.", "Hide polar grid."
    )
    add_bool_flag(
        p,
        "mask-angle",
        False,
        "Hide angular ticks.",
        "Show angular ticks.",
    )
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial ticks.",
        "Show radial ticks.",
    )

    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save figure to this path.",
    )

    p.set_defaults(func=_cmd_plot_polar_heatmap)


# --------------------------- registration -------------------------- #
def add_plot_fields(subparsers: argparse._SubParsersAction) -> None:
    """
    Register field-style polar plotting subcommands:
      - plot-radial-density-ring
      - plot-polar-quiver
      - plot-polar-heatmap
    """
    _add_cmd_ring(subparsers)
    _add_cmd_quiver(subparsers)
    _add_cmd_heatmap(subparsers)
