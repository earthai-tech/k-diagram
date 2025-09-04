from __future__ import annotations

import argparse

from kdiagram.plot.probabilistic import (
    plot_crps_comparison,
    plot_pit_histogram,
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

# --------------------------- PIT ---------------------------------


def _cmd_plot_pit_hist(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy(dtype=float)

    # predictions: exactly one group
    specs = _collect_pred_specs(ns)
    if not specs:
        raise SystemExit(
            "provide one pred group via --pred/--pred-cols/--model/--q-cols"
        )
    # Be permissive: take the first group if multiple were passed.
    if len(specs) > 1:
        specs = [specs[0]]

    _, cols = specs[0]
    ensure_columns(df, cols)
    df = ensure_numeric(df, cols, copy=True, errors="raise")
    yq = df[cols].to_numpy(dtype=float)

    # quantiles
    q = _parse_q_levels(ns.q_levels)

    plot_pit_histogram(
        y_true,
        yq,
        q,
        n_bins=ns.n_bins,
        title=ns.title,
        figsize=ns.figsize,
        color=ns.color,
        edgecolor=ns.edgecolor,
        alpha=ns.alpha,
        show_uniform_line=ns.show_uniform_line,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def add_plot_pit_histogram_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-pit-histogram",
        help="PIT polar histogram for one model.",
        description=(
            "Compute Probability Integral Transform (PIT) per "
            "observation and draw a polar histogram."
        ),
    )

    # I/O + y
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. If omitted, try --input.",
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
        help="Ground truth column name.",
    )

    # prediction specs (multiple styles)
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec 'name:col1[,col2,...]'. Repeat for "
            "multiple models. For PIT, exactly one group."
        ),
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction columns (CSV or space-separated). "
            "Repeat to add groups. For PIT, use one group."
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
    # legacy, optional single group
    p.add_argument(
        "--q-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="q_cols",
        help=(
            "Legacy single group (CSV or tokens). Kept for "
            "compatibility. For PIT, allow one group."
        ),
    )

    # quantiles
    p.add_argument(
        "--q-levels",
        "--quantiles",
        dest="q_levels",
        required=True,
        help=(
            "Quantile levels (CSV). Example: '0.1,0.5,0.9'. "
            "Must match the # of prediction columns."
        ),
    )

    # style
    p.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of PIT bins on [0, 1].",
    )
    p.add_argument(
        "--title",
        default="PIT Histogram",
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
        help="Bar color.",
    )
    p.add_argument(
        "--edgecolor",
        default="black",
        help="Bar edge color.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Bar alpha (0..1).",
    )
    add_bool_flag(
        p,
        "show-uniform-line",
        True,
        "Show uniform reference line.",
        "Hide uniform reference line.",
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
        help="Path to save the figure. If omitted, show it.",
    )

    p.set_defaults(func=_cmd_plot_pit_hist)


# --------------------------- CRPS --------------------------
def _cmd_plot_crps_comparison(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy(dtype=float)

    # collect groups
    specs = _collect_pred_specs(ns)
    if not specs:
        raise SystemExit(
            "provide one or more groups via --pred/--pred-cols/--model"
        )

    # quantiles, then check group sizes first
    q = _parse_q_levels(ns.q_levels)
    for name, cols in specs:
        if len(cols) != len(q):
            raise SystemExit(
                f"group {name!r} has {len(cols)} cols but "
                f"--q-levels has {len(q)}"
            )

    # columns -> arrays
    need = [c for _, cols in specs for c in cols]
    ensure_columns(df, need)
    df = ensure_numeric(df, need, copy=True, errors="raise")
    yqs = [df[cols].to_numpy(dtype=float) for _, cols in specs]

    # names (CLI override or auto from specs)
    names = ns.names if ns.names else _names_from_specs(specs)
    if names and any(isinstance(n, list) for n in names):
        names = [
            x for g in names for x in (g if isinstance(g, list) else [g])
        ]

    plot_crps_comparison(
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


def add_plot_crps_comparison_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-crps-comparison",
        help="Average CRPS per model on polar axes.",
        description=(
            "Compute mean CRPS for one or more models and "
            "plot them (lower is better)."
        ),
    )

    # I/O + y
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. If omitted, try --input.",
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
        help="Ground truth column name.",
    )

    # predictions
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=("Model spec 'name:col1[,col2,...]'. Repeat to add models."),
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction columns (CSV or space-separated). "
            "Repeat to add model groups."
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
            "Model names (CSV or space-separated). If omitted, "
            "names come from --model or are auto-generated."
        ),
    )

    p.add_argument(
        "--q-levels",
        "--quantiles",
        dest="q_levels",
        required=True,
        help=(
            "Quantile levels (CSV). Example: '0.1,0.5,0.9'. "
            "Must match each group's columns."
        ),
    )

    # style
    p.add_argument(
        "--title",
        default="Probabilistic Forecast Performance (CRPS)",
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
        help="Point marker style.",
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
        help="Path to save the figure. If omitted, show it.",
    )

    p.set_defaults(func=_cmd_plot_crps_comparison)


def add_plot_probs(
    subparsers: argparse._SubParsersAction,
) -> None:
    add_plot_pit_histogram_subparser(subparsers)
    add_plot_crps_comparison_subparser(subparsers)
