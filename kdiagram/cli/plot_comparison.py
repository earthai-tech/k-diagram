from __future__ import annotations

import argparse

from kdiagram.plot.comparison import (
    plot_horizon_metrics,
    plot_model_comparison,
)

from ._utils import (
    ColumnsListAction,
    _collect_point_preds,
    _parse_float_list,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)


# ------------------------- model comparison -----------------------
def _cmd_plot_model_comparison(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy()

    # predictions: one column per model (helper enforces this)
    yps, names = _collect_point_preds(df, ns)

    # train times
    ttimes = _parse_float_list(ns.train_times)
    if ttimes is not None:
        if len(ttimes) == 1 and len(yps) > 1:
            ttimes = [ttimes[0]] * len(yps)
        if len(ttimes) != len(yps):
            raise SystemExit(
                "len(--train-times) must equal #models "
                f"({len(yps)}), got {len(ttimes)}"
            )

    # metrics
    if ns.metrics is None:
        metrics = None
    else:
        metrics = (
            None
            if (len(ns.metrics) == 1 and ns.metrics[0].lower() == "auto")
            else list(ns.metrics)
        )

    # colors
    colors = list(ns.colors) if ns.colors else None

    # scale
    scale = None if ns.scale == "none" else ns.scale

    plot_model_comparison(
        y_true,
        *yps,
        train_times=ttimes,
        metrics=metrics,
        names=list(names),
        title=ns.title,
        figsize=ns.figsize,
        colors=colors,
        alpha=ns.alpha,
        legend=ns.legend,
        show_grid=ns.show_grid,
        grid_props=None,
        scale=scale,
        lower_bound=ns.lower_bound,
        savefig=str(ns.savefig) if ns.savefig else None,
        loc=ns.loc,
        verbose=ns.verbose,
    )


def add_plot_model_comparison_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-model-comparison",
        help=("Radar plot of multiple metrics for one or more models."),
        description=(
            "Compute selected metrics per model and show "
            "them on a radar (spider) plot. Accepts point "
            "prediction columns via --pred/--model."
        ),
    )

    # I/O + y
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. Or use --input.",
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

    # predictions (one col per model)
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec 'name:col'. Repeat to add models. "
            "Each model must have exactly one column."
        ),
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction columns (CSV or tokens). Repeat to "
            "add model groups (one col each)."
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
            "Model names (CSV or tokens). Defaults to "
            "parsed names or auto-generated."
        ),
    )

    # metrics / times
    p.add_argument(
        "--metrics",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="metrics",
        help=(
            "Metric names (CSV/tokens). Use 'auto' to let "
            "the function infer defaults by task."
        ),
    )
    p.add_argument(
        "--train-times",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="train_times",
        help=(
            "Training time per model (floats). Use one "
            "value to broadcast to all models."
        ),
    )

    # style / scaling
    p.add_argument(
        "--title",
        default=None,
        help="Figure title (optional).",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=None,
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--colors",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="colors",
        help="Custom colors (CSV/tokens), one per model.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Alpha for lines and fills.",
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
        "Show grid.",
        "Hide grid.",
    )
    p.add_argument(
        "--scale",
        default="norm",
        choices=["norm", "min-max", "std", "standard", "none"],
        help=(
            "Per-metric scaling across models. Use 'none' to disable scaling."
        ),
    )
    p.add_argument(
        "--lower-bound",
        type=float,
        default=0.0,
        help="Minimum radial value (inner circle).",
    )
    p.add_argument(
        "--loc",
        default="upper right",
        help="Legend location.",
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level.",
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
        help="Path to save figure.",
    )

    p.set_defaults(func=_cmd_plot_model_comparison)


# --------------------------- horizon metrics ----------------------
def _cmd_plot_horizon_metrics(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # ensure required lists
    if not ns.qlow or not ns.qup:
        raise SystemExit("both --qlow and --qup are required")
    qlow = list(ns.qlow)
    qup = list(ns.qup)
    q50 = list(ns.q50) if ns.q50 else None

    ensure_columns(df, qlow + qup + (q50 or []))
    df = ensure_numeric(
        df, qlow + qup + (q50 or []), copy=True, errors="raise"
    )

    xticks = list(ns.xtick_labels) if ns.xtick_labels else None

    plot_horizon_metrics(
        df=df,
        qlow_cols=qlow,
        qup_cols=qup,
        q50_cols=q50,
        xtick_labels=xticks,
        normalize_radius=ns.normalize_radius,
        show_value_labels=ns.show_value_labels,
        cbar_label=ns.cbar_label,
        r_label=ns.r_label,
        cmap=ns.cmap,
        acov=ns.acov,
        title=ns.title,
        figsize=ns.figsize,
        alpha=ns.alpha,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_angle=ns.mask_angle,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
        cbar=ns.cbar,
    )


def add_plot_horizon_metrics_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-horizon-metrics",
        help=(
            "Polar bar chart of mean interval width (and "
            "optional color metric) across horizons."
        ),
        description=(
            "Provide lower and upper quantile column lists per "
            "horizon. Optionally pass median (Q50) columns for "
            "color. Rows represent horizons/categories."
        ),
    )

    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. Or use --input.",
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
        "--qlow",
        action=ColumnsListAction,
        nargs="+",
        required=True,
        dest="qlow",
        help=("Lower quantile cols (CSV/tokens). Length must match --qup."),
    )
    p.add_argument(
        "--qup",
        action=ColumnsListAction,
        nargs="+",
        required=True,
        dest="qup",
        help=("Upper quantile cols (CSV/tokens). Length must match --qlow."),
    )
    p.add_argument(
        "--q50",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="q50",
        help=(
            "Optional median cols for bar colors. If not "
            "given, bar height is used for color."
        ),
    )
    p.add_argument(
        "--xtick-labels",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="xtick_labels",
        help=(
            "Angular tick labels (CSV/tokens). Length must "
            "match number of rows."
        ),
    )

    add_bool_flag(
        p,
        "normalize-radius",
        False,
        "Min-max scale bar heights.",
        "Do not normalize bar heights.",
    )
    add_bool_flag(
        p,
        "show-value-labels",
        True,
        "Show numeric value on bars.",
        "Hide numeric value on bars.",
    )
    p.add_argument(
        "--cbar-label",
        default=None,
        help="Colorbar label (optional).",
    )
    p.add_argument(
        "--r-label",
        default=None,
        help="Radial axis label (optional).",
    )
    p.add_argument(
        "--cmap",
        default="coolwarm",
        help="Colormap for bars.",
    )
    p.add_argument(
        "--acov",
        default="default",
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage span.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Figure title (optional).",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Alpha for bars.",
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
        "mask-angle",
        False,
        "Hide angular tick labels (if any).",
        "Show angular tick labels.",
    )
    add_bool_flag(
        p,
        "cbar",
        True,
        "Show colorbar.",
        "Hide colorbar.",
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
        help="Path to save figure.",
    )

    p.set_defaults(func=_cmd_plot_horizon_metrics)


# ----------------------------- registrar -------------------------
def add_plot_comparison(
    subparsers: argparse._SubParsersAction,
) -> None:
    add_plot_model_comparison_subparser(subparsers)
    add_plot_horizon_metrics_subparser(subparsers)
