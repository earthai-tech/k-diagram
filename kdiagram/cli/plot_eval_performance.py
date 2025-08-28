# License: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse

from ..plot.evaluation import plot_regression_performance
from ._utils import (
    ColumnsListAction,
    _collect_point_preds,
    _flatten_cols,
    _parse_metric_values,
    _parse_name_bool_map,
    _resolve_metric_labels,
    add_bool_flag,
    ensure_columns,
    ensure_numeric,
    load_df,
    parse_figsize,
)

__all__ = [
    "add_plot_regression_performance",
]


def _cmd_plot_regression_performance(
    ns: argparse.Namespace,
) -> None:
    """
    CLI runner for regression performance polar chart.
    Two modes:
      (A) values-mode via --metric-values
      (B) data-mode via --y-true + preds
    """

    # values-mode
    mv = _parse_metric_values(ns.metric_values)
    if mv:
        names = _flatten_cols(ns.names) if ns.names else None
        n_models = len(next(iter(mv.values())))
        if names and len(names) != n_models:
            raise SystemExit(
                f"--names has {len(names)} entries but "
                f"--metric-values imply {n_models} models"
            )

        plot_regression_performance(
            y_true=None,
            metric_values=mv,
            names=names,
            metrics=None,
            add_to_defaults=False,
            metric_labels=_resolve_metric_labels(ns),
            higher_is_better=_parse_name_bool_map(ns.higher_is_better),
            title=ns.title,
            figsize=ns.figsize,
            cmap=ns.cmap,
            show_grid=ns.show_grid,
            grid_props=None,
            mask_radius=ns.mask_radius,
            savefig=str(ns.savefig) if ns.savefig else None,
            dpi=ns.dpi,
        )
        return

    # data-mode
    if not ns.y_true:
        raise SystemExit(
            "Provide --y-true and predictions via "
            "--pred/--pred-cols/--model, or use "
            "--metric-values for values-mode."
        )

    df = load_df(ns.input, format=ns.format)

    # y_true
    ensure_columns(df, [ns.y_true])
    df = ensure_numeric(df, [ns.y_true], copy=True, errors="raise")
    y_true = df[ns.y_true].to_numpy()

    # collect predictions (each group => exactly one column)
    yps, names_cli = _collect_point_preds(df, ns)

    # names
    names = _flatten_cols(ns.names) if ns.names else None
    if names is None and names_cli:
        names = list(names_cli)

    # metrics list
    metrics = _flatten_cols(ns.metrics) if ns.metrics else None

    plot_regression_performance(
        y_true,
        *yps,
        names=names,
        metrics=metrics,
        metric_values=None,
        add_to_defaults=ns.add_to_defaults,
        metric_labels=_resolve_metric_labels(ns),
        higher_is_better=_parse_name_bool_map(ns.higher_is_better),
        title=ns.title,
        figsize=ns.figsize,
        cmap=ns.cmap,
        show_grid=ns.show_grid,
        grid_props=None,
        mask_radius=ns.mask_radius,
        savefig=str(ns.savefig) if ns.savefig else None,
        dpi=ns.dpi,
    )


def add_plot_regression_performance(
    sub: argparse._SubParsersAction,
) -> None:
    p = sub.add_parser(
        "plot-regression-performance",
        help=(
            "Polar performance chart for regression models using "
            "multiple metrics."
        ),
        description=(
            "Two modes:\n"
            "  (A) values-mode via --metric-values pairs "
            "(e.g. r2:0.81,0.75)\n"
            "  (B) data-mode via --y-true + "
            "--pred/--pred-cols/--model."
        ),
    )

    # I/O
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input table path. Omit in values-mode.",
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

    # data-mode columns
    p.add_argument(
        "--y-true",
        "--true-col",
        dest="y_true",
        default=None,
        help="Ground-truth column (data-mode).",
    )
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec 'name:col'. Repeat to add models.",
    )
    p.add_argument(
        "--pred",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred",
        help="Prediction column. Repeat to add models.",
    )
    p.add_argument(
        "--pred-cols",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="pred_cols",
        help="Alias for --pred (CSV or tokens).",
    )
    p.add_argument(
        "--names",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="names",
        help="Model names (CSV or tokens).",
    )

    # metrics control (data-mode)
    p.add_argument(
        "--metrics",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="metrics",
        help="Metric names/callables (tokens or CSV).",
    )
    add_bool_flag(
        p,
        "add-to-defaults",
        False,
        "Append user metrics to defaults.",
        "Replace defaults with user metrics.",
    )
    p.add_argument(
        "--higher-is-better",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="higher_is_better",
        help="Pairs 'metric:true/false' (space/CSV).",
    )

    # metric label controls
    add_bool_flag(
        p,
        "no-metric-labels",
        False,
        "Hide metric labels on the angular axis.",
        "Show metric labels (default).",
    )
    p.add_argument(
        "--metric-label",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="metric_label",
        help="Pairs 'orig:new' to rename metric labels.",
    )

    # values-mode input
    p.add_argument(
        "--metric-values",
        action=ColumnsListAction,
        nargs="+",
        default=None,
        dest="metric_values",
        help=(
            "Repeated pairs 'name:v1,v2,...' for values-mode "
            "(e.g. r2:0.8,0.7 rmse:5.2,6.0)."
        ),
    )

    # figure
    p.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 8.0),
        help="Figure size 'W,H'.",
    )
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--cmap", default="viridis", help="Colormap.")
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
    p.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    p.add_argument("--savefig", default=None, help="Path to save.")

    p.set_defaults(func=_cmd_plot_regression_performance)
