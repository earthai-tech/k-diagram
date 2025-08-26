# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..plot.uncertainty import (
    plot_interval_consistency,
    plot_interval_width,
)
from ._utils import (
    ColumnsListAction,
    ColumnsPairAction,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_intervals"]


def _cmd_plot_interval_width(ns: argparse.Namespace) -> None:
    qlow, qup = ns.q_cols

    drop_cols = [qlow, qup]
    if ns.z_col:
        drop_cols.append(ns.z_col)

    df: pd.DataFrame = load_df(
        ns.input,
        format=ns.format,
        dropna=drop_cols if ns.dropna else None,
    )

    need_cols = [qlow, qup] + ([ns.z_col] if ns.z_col else [])
    ensure_columns(df, need_cols)

    coerce = [qlow, qup] + ([ns.z_col] if ns.z_col else [])
    df = ensure_numeric(df, coerce, copy=True, errors="raise")

    plot_interval_width(
        df=df,
        q_cols=[qlow, qup],
        theta_col=ns.theta_col,
        z_col=ns.z_col,
        acov=ns.acov,
        figsize=ns.figsize,
        title=ns.title,
        cmap=ns.cmap,
        s=ns.s,
        alpha=ns.alpha,
        show_grid=ns.show_grid,
        grid_props=None,
        cbar=ns.cbar,
        mask_angle=ns.mask_angle,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


def _as_list(x) -> list[str] | None:
    if x is None:
        return None
    if isinstance(x, str):
        # accept "a,b,c" or "a b c"
        s = x.replace(" ", ",")
        return [t.strip() for t in s.split(",") if t.strip()]
    return list(x)


def _cmd_plot_interval_consistency(ns: argparse.Namespace) -> None:
    qlow_cols = _as_list(ns.qlow_cols) or []
    qup_cols = _as_list(ns.qup_cols) or []
    q50_cols = _as_list(ns.q50_cols) if ns.q50_cols else None

    if len(qlow_cols) != len(qup_cols):
        raise SystemExit("qlow-cols and qup-cols must have same size.")
    if q50_cols and len(q50_cols) != len(qlow_cols):
        raise SystemExit("q50-cols must match qlow/qup length.")

    drop_cols = [*qlow_cols, *qup_cols, *(q50_cols or [])]

    df: pd.DataFrame = load_df(
        ns.input,
        format=ns.format,
        dropna=drop_cols if ns.dropna else None,
    )

    ensure_columns(df, drop_cols)
    df = ensure_numeric(df, drop_cols, copy=True, errors="raise")

    plot_interval_consistency(
        df=df,
        qlow_cols=qlow_cols,
        qup_cols=qup_cols,
        q50_cols=q50_cols,
        theta_col=ns.theta_col,
        use_cv=ns.use_cv,
        cmap=ns.cmap,
        acov=ns.acov,
        title=ns.title,
        figsize=ns.figsize,
        s=ns.s,
        alpha=ns.alpha,
        show_grid=ns.show_grid,
        mask_angle=ns.mask_angle,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


def add_plot_intervals(
    subparsers: argparse._SubParsersAction,
) -> None:
    """
    Register interval-family plot commands.
    """
    # ---- plot-interval-width -----------------------------------
    p = subparsers.add_parser(
        "plot-interval-width",
        help=("Polar scatter of interval width " "(upper - lower)."),
        description=(
            "Visualize the magnitude of prediction "
            "uncertainty as width between two "
            "quantile columns."
        ),
    )
    p.add_argument(
        "input",
        type=str,
        help="Input table (CSV/Parquet/...).",
    )
    p.add_argument(
        "--format",
        type=str,
        default=None,
        help="Explicit input format.",
    )
    p.add_argument(
        "--q-cols",
        action=ColumnsPairAction,
        nargs="+",  # accept 1 token ("low,up") or 2 tokens
        required=True,
        metavar="LOW,UP",  # single string, not a tuple
        help=(
            "Two columns 'lower,upper' that define the interval. "
            "Accepts 'low,up' or two tokens."
        ),
    )

    p.add_argument(
        "--z-col",
        type=str,
        default=None,
        help=("Optional column for color; defaults to width."),
    )
    p.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Ordering column (ignored).",
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
        default="viridis",
        help="Matplotlib colormap.",
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
    add_bool_flag(p, "show-grid", True, "Show grid.", "Hide grid.")
    add_bool_flag(p, "cbar", True, "Show colorbar.", "Hide.")
    add_bool_flag(p, "mask-angle", True, "Hide angle ticks.", "Show.")
    add_bool_flag(p, "dropna", True, "Drop NaNs in needed cols.", "Keep.")
    p.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save to path instead of showing.",
    )
    p.set_defaults(func=_cmd_plot_interval_width)

    # ---- plot-interval-consistency -----------------------------
    p2 = subparsers.add_parser(
        "plot-interval-consistency",
        help=("Polar scatter of temporal consistency " "of interval widths."),
        description=(
            "Compute CV or Std of widths across "
            "time columns, plot per-location."
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
        "--q10-cols",
        action=ColumnsListAction,
        nargs="+",
        required=True,
        dest="qlow_cols",
        metavar="LOW1,LOW2,...",
        help="Lower-bound columns by time.",
    )
    p2.add_argument(
        "--qup-cols",
        "--q90-cols",
        action=ColumnsListAction,
        nargs="+",
        required=True,
        dest="qup_cols",
        metavar="UP1,UP2,...",
        help="Upper-bound columns by time.",
    )
    p2.add_argument(
        "--q50-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="q50_cols",
        metavar="Q50A,Q50B,...",
        help="Optional Q50 columns by time.",
    )
    p2.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Ordering column (ignored).",
    )

    add_bool_flag(p2, "use-cv", True, "Use CV metric.", "Use Std Dev.")

    p2.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap.",
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
        "--title",
        type=str,
        default=None,
        help="Figure title.",
    )
    p2.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(9.0, 9.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p2.add_argument(
        "--s",
        type=float,
        default=30,
        help="Marker size.",
    )
    p2.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Marker alpha (0..1).",
    )
    add_bool_flag(p2, "show-grid", True, "Show grid.", "Hide grid.")
    add_bool_flag(p2, "mask-angle", False, "Hide angle ticks.", "Show.")
    add_bool_flag(p2, "dropna", True, "Drop NaNs in needed cols.", "Keep.")
    p2.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Save to path instead of showing.",
    )
    p2.set_defaults(func=_cmd_plot_interval_consistency)


def main() -> None:
    parser = argparse.ArgumentParser(prog="kdiagram-intervals")
    sp = parser.add_subparsers(dest="cmd")
    add_plot_intervals(sp)
    ns = parser.parse_args()
    if hasattr(ns, "func"):
        ns.func(ns)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
