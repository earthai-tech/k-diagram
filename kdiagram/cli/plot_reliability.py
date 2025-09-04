from __future__ import annotations

import argparse

from kdiagram.plot.comparison import (
    plot_polar_reliability,
    plot_reliability_diagram,
)

from ._utils import (
    ColumnsListAction,
    _collect_point_preds,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)


def _parse_label(value: str) -> int | float | str:
    """Best-effort parse of a label to int/float/str."""
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return value


# --------------------------- rectangular --------------------------
def _cmd_plot_reliability(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    y_true = df[ns.y_true].to_numpy()

    # sample weight
    sample_weight = None
    if ns.sample_weight:
        ensure_columns(df, [ns.sample_weight])
        df = ensure_numeric(df, [ns.sample_weight], copy=True, errors="raise")
        sample_weight = df[ns.sample_weight].to_numpy(dtype=float)

    # 1D point probabilities + names
    yps, names = _collect_point_preds(df, ns)

    # parse pairs
    clip_lo, clip_hi = ns.clip_probs
    xlim_lo, xlim_hi = ns.xlim
    ylim_lo, ylim_hi = ns.ylim

    plot_reliability_diagram(
        y_true,
        *yps,
        names=list(names),
        sample_weight=sample_weight,
        n_bins=ns.n_bins,
        strategy=ns.strategy,
        positive_label=_parse_label(ns.positive_label),
        class_index=ns.class_index,
        clip_probs=(clip_lo, clip_hi),
        normalize_probs=ns.normalize_probs,
        error_bars=ns.error_bars,
        conf_level=ns.conf_level,
        show_diagonal=ns.show_diagonal,
        diagonal_kwargs=None,
        show_ece=ns.show_ece,
        show_brier=ns.show_brier,
        counts_panel=ns.counts_panel,
        counts_norm=ns.counts_norm,
        counts_alpha=ns.counts_alpha,
        figsize=ns.figsize,
        title=ns.title,
        xlabel=ns.xlabel,
        ylabel=ns.ylabel,
        cmap=ns.cmap,
        color_palette=None,
        marker=ns.marker,
        s=ns.s,
        linewidth=ns.linewidth,
        alpha=ns.alpha,
        connect=ns.connect,
        legend=ns.legend,
        legend_loc=ns.legend_loc,
        show_grid=ns.show_grid,
        grid_props=None,
        xlim=(xlim_lo, xlim_hi),
        ylim=(ylim_lo, ylim_hi),
        savefig=str(ns.savefig) if ns.savefig else None,
        return_data=False,
    )


def add_plot_reliability_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-reliability-diagram",
        help=("Reliability (calibration) diagram for one or more models."),
        description=(
            "Compare predicted probabilities to observed "
            "frequencies across probability bins. Supports "
            "multiple models and a counts panel."
        ),
    )

    # I/O + y
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help=("Input table path. If omitted, try --input."),
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
        help="Ground-truth label column.",
    )
    p.add_argument(
        "--sample-weight",
        dest="sample_weight",
        default=None,
        help=("Optional weight column for bins, ECE, and Brier score."),
    )

    # predictions (point probs)
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec 'name:col' or 'name:c1,c2,...'. "
            "Repeat to add models. If multiple columns are "
            "given, a class index can be selected."
        ),
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction prob columns (CSV or tokens). "
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
            "Model names (CSV or tokens). Defaults to "
            "parsed names or auto-generated."
        ),
    )

    # calibration specifics
    p.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of probability bins.",
    )
    p.add_argument(
        "--strategy",
        default="uniform",
        choices=["uniform", "quantile"],
        help="Binning strategy.",
    )
    p.add_argument(
        "--positive-label",
        default="1",
        help=("Label in y_true treated as positive class (int/float/str)."),
    )
    p.add_argument(
        "--class-index",
        type=int,
        default=None,
        help=(
            "When a model group has multiple columns, "
            "choose the column index for the positive "
            "class (default: last)."
        ),
    )
    p.add_argument(
        "--clip-probs",
        type=parse_figsize,
        default=(0.0, 1.0),
        help=(
            "Inclusive clipping range 'lo,hi' for probs. "
            "Applies after optional normalization."
        ),
    )
    add_bool_flag(
        p,
        "normalize-probs",
        True,
        "Try to linearly rescale near-range probs to [0,1].",
        "Disable normalization; only clip to [0,1].",
    )
    p.add_argument(
        "--error-bars",
        default="wilson",
        choices=["wilson", "normal", "none"],
        help="Per-bin CI method for observed frequency.",
    )
    p.add_argument(
        "--conf-level",
        type=float,
        default=0.95,
        help="Confidence level for error bars.",
    )

    # appearance
    add_bool_flag(
        p,
        "show-diagonal",
        True,
        "Show y=x reference diagonal.",
        "Hide reference diagonal.",
    )
    add_bool_flag(
        p,
        "show-ece",
        True,
        "Append ECE summary to legend labels.",
        "Do not compute/show ECE.",
    )
    add_bool_flag(
        p,
        "show-brier",
        True,
        "Append Brier score to legend labels.",
        "Do not compute/show Brier score.",
    )
    p.add_argument(
        "--counts-panel",
        default="bottom",
        choices=["bottom", "none"],
        help=("Show compact per-bin totals below the plot or disable it."),
    )
    p.add_argument(
        "--counts-norm",
        default="fraction",
        choices=["fraction", "count"],
        help=("Normalize counts by total weight or show raw weighted sums."),
    )
    p.add_argument(
        "--counts-alpha",
        type=float,
        default=0.35,
        help="Alpha for counts panel bars.",
    )
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(9.0, 7.0),
        help="Figure size 'W,H' or 'WxH'.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Figure title (optional).",
    )
    p.add_argument(
        "--xlabel",
        default="Predicted probability",
        help="X-axis label.",
    )
    p.add_argument(
        "--ylabel",
        default="Observed frequency",
        help="Y-axis label.",
    )
    p.add_argument(
        "--cmap",
        default="tab10",
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
        default=40,
        help="Marker size.",
    )
    p.add_argument(
        "--linewidth",
        type=float,
        default=2.0,
        help="Line width for connecting points.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Alpha for points/lines.",
    )
    add_bool_flag(
        p,
        "connect",
        True,
        "Connect bin points per model.",
        "Do not connect points.",
    )
    add_bool_flag(
        p,
        "legend",
        True,
        "Show legend.",
        "Hide legend.",
    )
    p.add_argument(
        "--legend-loc",
        default="best",
        help="Legend location.",
    )
    add_bool_flag(
        p,
        "show-grid",
        True,
        "Show grid.",
        "Hide grid.",
    )
    p.add_argument(
        "--xlim",
        type=parse_figsize,
        default=(0.0, 1.0),
        help="X limits 'lo,hi'.",
    )
    p.add_argument(
        "--ylim",
        type=parse_figsize,
        default=(0.0, 1.0),
        help="Y limits 'lo,hi'.",
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

    p.set_defaults(func=_cmd_plot_reliability)


# ----------------------------- polar ------------------------------


def _cmd_plot_polar_reliability(ns: argparse.Namespace) -> None:
    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    y_true = df[ns.y_true].to_numpy()

    # point probabilities + names (1D each)
    yps, names = _collect_point_preds(df, ns)

    plot_polar_reliability(
        y_true,
        *yps,
        names=list(names),
        n_bins=ns.n_bins,
        strategy=ns.strategy,
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        show_cbar=ns.show_cbar,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def add_plot_polar_reliability_subparser(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-polar-reliability",
        help=(
            "Polar reliability (calibration spiral) for one or more models."
        ),
        description=(
            "Map reliability diagram into polar axes: angle is "
            "predicted probability, radius is observed "
            "frequency. Colors highlight calibration error."
        ),
    )

    # I/O + y
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help=("Input table path. If omitted, try --input."),
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
        help="Ground-truth label column.",
    )

    # predictions (point probs)
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec 'name:col' or 'name:c1,c2,...'. Repeat to add models."
        ),
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help=(
            "Prediction prob columns (CSV or tokens). "
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
            "Model names (CSV or tokens). Defaults to "
            "parsed names or auto-generated."
        ),
    )

    # binning + style
    p.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of probability bins.",
    )
    p.add_argument(
        "--strategy",
        default="uniform",
        choices=["uniform", "quantile"],
        help="Binning strategy.",
    )
    p.add_argument(
        "--title",
        default="Polar Reliability Diagram",
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
        default="coolwarm",
        help="Diverging colormap for error coloring.",
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
        "show-cbar",
        True,
        "Show colorbar (calibration error).",
        "Hide colorbar.",
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
        help="Savefig DPI.",
    )
    p.add_argument(
        "--savefig",
        default=None,
        help="Path to save the figure.",
    )

    p.set_defaults(func=_cmd_plot_polar_reliability)


# ---------------------------- registrar ---------------------------
def add_plot_reliability(
    subparsers: argparse._SubParsersAction,
) -> None:
    add_plot_reliability_subparser(subparsers)
    add_plot_polar_reliability_subparser(subparsers)
