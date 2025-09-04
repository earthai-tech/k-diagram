from __future__ import annotations

import argparse

from kdiagram.plot.relationship import (
    plot_conditional_quantiles,
    plot_relationship,
)

from ._utils import (
    ColumnsListAction,
    _collect_point_preds,
    _collect_pred_specs,
    _parse_q_levels,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

# -------------------------- commands --------------------------


def _cmd_plot_relationship(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy(dtype=float)

    # preds: one col per model
    yps, names = _collect_point_preds(df, ns)

    # z-values (tick labels) are optional; no numeric coercion
    z_vals = None
    if ns.z_col:
        ensure_columns(df, [ns.z_col])
        z_vals = df[ns.z_col].to_numpy()

    color_palette = list(ns.colors) if ns.colors else None

    plot_relationship(
        y_true,
        *yps,
        names=names,
        title=ns.title,
        theta_offset=ns.theta_offset,
        theta_scale=ns.theta_scale,
        acov=ns.acov,
        figsize=ns.figsize,
        cmap=ns.cmap,
        s=ns.s,
        alpha=ns.alpha,
        legend=ns.legend,
        show_grid=ns.show_grid,
        grid_props=None,
        color_palette=color_palette,
        xlabel=ns.xlabel,
        ylabel=ns.ylabel,
        z_values=z_vals,
        z_label=ns.z_label,
        savefig=str(ns.savefig) if ns.savefig else None,
    )


def _cmd_plot_conditional_quantiles(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy(dtype=float)

    # quantile predictions: exactly one group
    specs = _collect_pred_specs(ns)
    if not specs:
        raise SystemExit(
            "provide one quantile group via --pred/--pred-cols/--model"
        )
    if len(specs) != 1:
        raise SystemExit("expect exactly one group of quantile columns")

    name, cols = specs[0]
    ensure_columns(df, cols)
    ensure_numeric(df, cols, copy=True, errors="raise")
    yq = df[cols].to_numpy(dtype=float)

    # q-levels
    q = _parse_q_levels(ns.q_levels)
    if len(cols) != len(q):
        raise SystemExit(
            f"group {name!r} has {len(cols)} cols but --q-levels has {len(q)}"
        )

    # bands -> list[int] or None
    bands = [int(x) for x in ns.bands] if ns.bands else None

    plot_conditional_quantiles(
        y_true,
        yq,
        q,
        bands=bands,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        alpha_min=ns.alpha_min,
        alpha_max=ns.alpha_max,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


# ------------------------- subparsers -------------------------


def _add_common_io(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. If omitted, use --input.",
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
        help="Input format override (csv, parquet, ...).",
    )
    p.add_argument(
        "--y-true",
        "--true-col",
        dest="y_true",
        required=True,
        help="Ground-truth column name.",
    )


def _add_preds_and_names(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec 'NAME:col' or 'NAME:col1,col2,...'. "
            "For point plots, use one col per model. For "
            "quantile plots, pass one group with many cols."
        ),
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction columns (space-separated or CSV). "
            "Repeat to add groups/models."
        ),
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
        help="Model names (CSV or space separated).",
    )


def add_plot_cond_relationship(
    subparsers: argparse._SubParsersAction,
) -> None:
    # ---- plot-relationship ----
    p = subparsers.add_parser(
        "plot-relationship",
        help="Truth vs predictions (polar scatter).",
        description=(
            "Scatter predictions on polar axes using angles "
            "derived from y_true and radii from predictions. "
            "Compare many point predictions to the same truth."
        ),
    )
    _add_common_io(p)
    _add_preds_and_names(p)

    # mapping + style
    p.add_argument(
        "--theta-offset",
        type=float,
        default=0.0,
        help="Angular shift (radians) applied to all points.",
    )
    p.add_argument(
        "--theta-scale",
        choices=["proportional", "uniform"],
        default="proportional",
        help="Angle mapping: 'proportional' or 'uniform'.",
    )
    p.add_argument(
        "--acov",
        choices=[
            "default",
            "half_circle",
            "quarter_circle",
            "eighth_circle",
        ],
        default="default",
        help="Angular span (full/half/quarter/eighth).",
    )
    p.add_argument(
        "--title",
        default="Relationship Visualization",
        help="Figure title.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--cmap",
        default="tab10",
        help="Matplotlib colormap name.",
    )
    p.add_argument(
        "--colors",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="colors",
        help="Explicit colors (CSV or space separated).",
    )
    p.add_argument(
        "--s",
        type=float,
        default=50.0,
        help="Marker size.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Marker alpha in [0, 1].",
    )
    add_bool_flag(
        p,
        "legend",
        True,
        "Show legend.",
        "Hide legend.",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show polar grid.",
        "Hide polar grid.",
    )
    p.add_argument(
        "--xlabel",
        default=None,
        help="Radial axis label.",
    )
    p.add_argument(
        "--ylabel",
        default=None,
        help="Angular axis label.",
    )
    p.add_argument(
        "--z-col",
        default=None,
        help="Column for custom angular tick labels.",
    )
    p.add_argument(
        "--z-label",
        default=None,
        help="Label describing z-col tick labels.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI when saving.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure. If omitted, show it.",
    )
    p.set_defaults(func=_cmd_plot_relationship)

    # ---- plot-conditional-quantiles ----
    p2 = subparsers.add_parser(
        "plot-conditional-quantiles",
        help="Quantile bands vs true (polar bands).",
        description=(
            "Plot conditional quantile bands on polar axes. "
            "Provide one group of quantile columns and the "
            "matching --q-levels."
        ),
    )
    _add_common_io(p2)
    _add_preds_and_names(p2)

    p2.add_argument(
        "--q-levels",
        "--quantiles",
        dest="q_levels",
        required=True,
        help=(
            "Quantile levels (CSV). Example '0.1,0.5,0.9'. "
            "Must match the group's columns."
        ),
    )
    p2.add_argument(
        "--bands",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="bands",
        help="Interval %s to fill, e.g. '90 50'.",
    )
    p2.add_argument(
        "--title",
        default="Conditional Quantile Plot",
        help="Figure title.",
    )
    p2.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p2.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap for bands.",
    )
    p2.add_argument(
        "--alpha-min",
        type=float,
        default=0.2,
        dest="alpha_min",
        help="Min alpha for outer bands.",
    )
    p2.add_argument(
        "--alpha-max",
        type=float,
        default=0.5,
        dest="alpha_max",
        help="Max alpha for inner bands.",
    )
    add_bool_flag(
        p2,
        "show-grid",
        True,
        "Show polar grid.",
        "Hide polar grid.",
    )
    add_bool_flag(
        p2,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show radial tick labels.",
    )
    p2.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI when saving.",
    )
    p2.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure. If omitted, show it.",
    )
    p2.set_defaults(func=_cmd_plot_conditional_quantiles)
