# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse

from ..plot.feature_based import (
    plot_feature_fingerprint,
    plot_feature_interaction,
)
from ._utils import (
    ColumnsListAction,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = ["add_plot_feature_based"]


def _cmd_plot_feature_interaction(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # Required columns
    need = [ns.theta_col, ns.r_col, ns.color_col]
    ensure_columns(df, need)
    # Only r_col & color_col need to be numeric; theta can be ordinal/cyc.
    df = ensure_numeric(
        df, [ns.r_col, ns.color_col], copy=True, errors="raise"
    )

    plot_feature_interaction(
        df=df,
        theta_col=ns.theta_col,
        r_col=ns.r_col,
        color_col=ns.color_col,
        statistic=ns.statistic,
        theta_period=ns.theta_period,
        theta_bins=ns.theta_bins,
        r_bins=ns.r_bins,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def add_plot_feature_interaction(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-feature-interaction",
        help="Polar heatmap of interaction between two features.",
        description=(
            "Bin one feature to angle (theta) and another to radius, "
            "aggregate a target column per bin (mean/median/std/etc.), "
            "and render as a polar heatmap."
        ),
    )

    # I/O
    p.add_argument("input", nargs="?", default=None, help="Input table path.")
    p.add_argument(
        "-i", "--input", dest="input", help="Input table path (alt)."
    )
    p.add_argument(
        "--format", default=None, help="Format override (csv, parquet, ...)."
    )

    # Columns
    p.add_argument(
        "--theta-col",
        required=True,
        help="Column mapped to angle (often cyclical, e.g. month/hour).",
    )
    p.add_argument(
        "--r-col",
        required=True,
        help="Column mapped to radius (numeric).",
    )
    p.add_argument(
        "--color-col",
        required=True,
        help="Target/response column aggregated per bin for coloring.",
    )

    # Binning / statistic
    p.add_argument(
        "--statistic",
        default="mean",
        choices=["mean", "median", "std", "min", "max"],
        help="Aggregation applied to color column per (theta,r) bin.",
    )
    p.add_argument(
        "--theta-period",
        type=float,
        default=None,
        help="Period of theta feature (e.g. 12 for months, 24 for hours).",
    )
    p.add_argument(
        "--theta-bins", type=int, default=24, help="Number of theta bins."
    )
    p.add_argument(
        "--r-bins", type=int, default=10, help="Number of radial bins."
    )

    # Appearance
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument(
        "--cmap", default="viridis", help="Matplotlib colormap name."
    )
    add_bool_flag(p, "show-grid", True, "Show grid.", "Hide grid.")
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show radial tick labels.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save the figure.")

    p.set_defaults(func=_cmd_plot_feature_interaction)


# Feature Fingerprint (radar chart)
def _cmd_plot_feature_fingerprint(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # Importance matrix
    if not ns.cols:
        raise SystemExit(
            "Provide importance columns via --cols/--importances."
        )
    cols = list(ns.cols)
    ensure_columns(df, cols)
    df = ensure_numeric(df, cols, copy=True, errors="raise")

    imp = df[cols].to_numpy(dtype=float)
    # Allow users to interpret rows as features and columns as layers
    if ns.transpose:
        imp = imp.T  # (n_layers, n_features) expected by the plot function

    # Labels (layers)
    labels = None
    if ns.labels:
        labels = list(ns.labels)
    elif ns.labels_col:
        ensure_columns(df, [ns.labels_col])
        labels = df[ns.labels_col].astype(str).tolist()
        if ns.transpose:
            # When transposed, labels should come from column headers
            # Let explicit --labels override this; otherwise use cols
            labels = cols

    # Feature names
    features = None
    if ns.features:
        features = list(ns.features)
    else:
        # Default to the column names used for importances
        features = (
            cols
            if not ns.transpose
            else (
                df[ns.labels_col].astype(str).tolist()
                if ns.labels_col
                else None
            )
        )

    plot_feature_fingerprint(
        importances=imp,
        features=features,
        labels=labels,
        normalize=ns.normalize,
        fill=ns.fill,
        cmap=ns.cmap,
        title=ns.title,
        figsize=ns.figsize,
        show_grid=ns.show_grid,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


def add_plot_feature_fingerprint(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-feature-fingerprint",
        help="Radar chart for multi-layer feature importances.",
        description=(
            "Visualize feature-importance profiles across layers "
            "(e.g., models/years/zones) as radar polygons."
        ),
    )

    # I/O
    p.add_argument("input", nargs="?", default=None, help="Input table path.")
    p.add_argument(
        "-i", "--input", dest="input", help="Input table path (alt)."
    )
    p.add_argument(
        "--format", default=None, help="Format override (csv, parquet, ...)."
    )

    # Matrix selection
    p.add_argument(
        "--cols",
        "--importances",
        action=ColumnsListAction,
        nargs="+",
        required=True,
        dest="cols",
        help="Importance columns (CSV or space-separated).",
    )

    # Names / labels
    p.add_argument(
        "--labels-col",
        default=None,
        help="Optional column providing layer labels (row labels).",
    )
    p.add_argument(
        "--labels",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        help="Explicit layer labels (CSV or tokens).",
    )
    p.add_argument(
        "--features",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        help="Feature names. Defaults to --cols (or labels-col if --transpose).",
    )

    # Options
    add_bool_flag(
        p,
        "transpose",
        False,
        "Interpret rows as features and columns as layers.",
        "Interpret rows as layers (default).",
    )
    add_bool_flag(
        p,
        "normalize",
        True,
        "Row-wise normalize to [0,1].",
        "Disable normalization.",
    )
    add_bool_flag(
        p, "fill", True, "Fill radar polygons.", "Do not fill polygons."
    )
    p.add_argument(
        "--cmap", default="tab10", help="Colormap name or a discrete cmap."
    )
    p.add_argument(
        "--title", default="Feature Impact Fingerprint", help="Figure title."
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H'.",
    )
    add_bool_flag(p, "show-grid", True, "Show grid.", "Hide grid.")
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save the figure.")

    p.set_defaults(func=_cmd_plot_feature_fingerprint)


# ----------------------------- registrar -------------------------


def add_plot_feature_based(
    subparsers: argparse._SubParsersAction,
) -> None:
    add_plot_feature_interaction(subparsers)
    add_plot_feature_fingerprint(subparsers)
