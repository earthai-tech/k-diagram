from __future__ import annotations

import argparse

from kdiagram.plot.probabilistic import (
    plot_calibration_sharpness,
    plot_polar_sharpness,
)

from ._utils import (
    ColumnsListAction,
    _collect_pred_specs,
    _names_from_specs,
    _parse_q_levels,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)


# ----------------------- commands -----------------------
def _cmd_plot_polar_sharpness(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # collect prediction groups
    specs = _collect_pred_specs(ns)
    if not specs:
        raise SystemExit("provide groups via --pred/--pred-cols/--model")

    need = [c for _, cols in specs for c in cols]
    ensure_columns(df, need)
    df = ensure_numeric(df, need, copy=True, errors="raise")
    yqs = [df[cols].to_numpy(dtype=float) for _, cols in specs]

    names = ns.names if ns.names else _names_from_specs(specs)
    q = _parse_q_levels(ns.q_levels)

    plot_polar_sharpness(
        *yqs,
        quantiles=q,
        names=list(names),
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        marker=ns.marker,
        s=ns.s,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def _cmd_plot_calibration_sharpness(
    ns: argparse.Namespace,
) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true], error="raise")
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy(dtype=float)

    # prediction groups (support all styles)
    specs = _collect_pred_specs(ns)
    if not specs:
        raise SystemExit("provide groups via --pred/--pred-cols/--model")

    need = [c for _, cols in specs for c in cols]
    ensure_columns(df, need, error="raise")
    df = ensure_numeric(df, need, copy=True, errors="raise")
    yqs = [df[cols].to_numpy(dtype=float) for _, cols in specs]

    names = ns.names if ns.names else _names_from_specs(specs)
    q = _parse_q_levels(ns.q_levels)

    plot_calibration_sharpness(
        y_true,
        *yqs,
        quantiles=q,
        names=list(names),
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        marker=ns.marker,
        s=ns.s,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


# ----------------------- subparsers ---------------------
def add_plot_sharpness(
    subparsers: argparse._SubParsersAction,
) -> None:
    # -------- plot-polar-sharpness ----------
    p = subparsers.add_parser(
        "plot-polar-sharpness",
        help="Compare forecast sharpness (interval width).",
        description=(
            "Compare models by average interval width on polar "
            "axes. Provide each model via --model NAME:col1[,col2,..] "
            "or with --pred/--pred-cols. Use --q-levels for the "
            "shared quantile levels."
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
        help="Input format override (csv, parquet, ...).",
    )

    # predictions (flexible)
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=("Model spec 'NAME:col1[,col2,...]'. Repeat to add models."),
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction columns (CSV or space-separated). "
            "Repeat to add groups."
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
        help=(
            "Model names (CSV or space separated). If omitted, "
            "names come from --model or are auto-generated."
        ),
    )
    p.add_argument(
        "--q-levels",
        "--quantiles",
        dest="q_levels",
        required=True,
        help=(
            "Quantile levels (CSV), e.g. '0.1,0.5,0.9'. Must "
            "match each group's column count."
        ),
    )

    # style
    p.add_argument(
        "--title",
        default="Forecast Sharpness Comparison",
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
        default="viridis",
        help="Matplotlib colormap name.",
    )
    p.add_argument(
        "--marker",
        default="o",
        help="Marker style.",
    )
    p.add_argument(
        "--s",
        type=int,
        default=100,
        help="Marker size.",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
    )
    add_bool_flag(
        p,
        "mask-radius",
        False,
        "Hide radial tick labels.",
        "Show radial tick labels.",
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
        help="Path to save figure. If omitted, show it.",
    )
    p.set_defaults(func=_cmd_plot_polar_sharpness)

    # -------- plot-calibration-sharpness ----
    p2 = subparsers.add_parser(
        "plot-calibration-sharpness",
        help="Calibration vs sharpness trade-off (polar).",
        description=(
            "Quarter-circle where angle encodes calibration "
            "error (KS distance) and radius encodes average "
            "interval width (sharpness). Provide ground truth "
            "via --y-true/--true-col. Accepts --model and "
            "--pred/--pred-cols."
        ),
    )

    # I/O
    p2.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path.",
    )
    p2.add_argument(
        "-i",
        "--input",
        dest="input",
        help="Input table path (alt form).",
    )
    p2.add_argument(
        "--format",
        default=None,
        help="Input format override (csv, parquet, ...).",
    )

    # y_true
    p2.add_argument(
        "--y-true",
        "--true-col",
        required=True,
        dest="y_true",
        help="Ground-truth column name.",
    )

    # predictions (flexible; do NOT require only --model)
    p2.add_argument(
        "--model",
        action="append",
        default=None,
        help=("Model spec 'NAME:col1[,col2,...]'. Repeat to add models."),
    )
    p2.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction columns (CSV or space-separated). "
            "Repeat to add groups."
        ),
    )
    p2.add_argument(
        "--pred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_cols",
        help="Alias for --pred.",
    )
    p2.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help=(
            "Model names (CSV or space separated). If omitted, "
            "names come from --model or are auto-generated."
        ),
    )
    p2.add_argument(
        "--q-levels",
        "--quantiles",
        required=True,
        dest="q_levels",
        help=(
            "Quantile levels (CSV), e.g. '0.1,0.5,0.9'. Must "
            "match each group's column count."
        ),
    )

    # style
    p2.add_argument(
        "--title",
        default="Calibration vs. Sharpness Trade-off",
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
        help="Matplotlib colormap name.",
    )
    p2.add_argument(
        "--marker",
        default="o",
        help="Marker style.",
    )
    p2.add_argument(
        "--s",
        type=int,
        default=150,
        help="Marker size.",
    )
    add_bool_flag(
        p2,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
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
        help="Path to save figure. If omitted, show it.",
    )
    p2.set_defaults(func=_cmd_plot_calibration_sharpness)
