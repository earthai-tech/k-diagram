from __future__ import annotations

import argparse

try:
    from .. import __version__ as _VERSION
except Exception:  # pragma: no cover
    _VERSION = "1.3.0"

# subcommand registrars
from .plot_anomalies import add_plot_anomalies
from .plot_comparison import add_plot_comparison
from .plot_cond_relationship import add_plot_cond_relationship
from .plot_coverages import add_plot_coverages
from .plot_credibility import add_plot_credibility
from .plot_drift import add_plot_drift
from .plot_errors import add_plot_errors
from .plot_eval_relationship import add_plot_eval_relationship
from .plot_fields import add_plot_fields
from .plot_intervals import add_plot_intervals
from .plot_probs import add_plot_probs
from .plot_reliability import add_plot_reliability
from .plot_sharpness import add_plot_sharpness
from .plot_temporal import add_plot_temporal
from .plot_velocities import add_plot_velocities
from .plot_vs import add_plot_vs

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser and register all
    subcommands from kdiagram/cli/* modules.
    """
    parser = argparse.ArgumentParser(
        prog="kdiagram",
        description=(
            "KDiagram CLI — polar diagnostics for "
            "uncertainty, drift, fields, and more."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"kdiagram { _VERSION }",
        help="Show version and exit.",
    )

    sub = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        help="Run 'kdiagram <command> -h' for help.",
    )

    # register all commands (grouped by theme)
    # Uncertainty / coverage
    add_plot_coverages(sub)
    add_plot_intervals(sub)
    add_plot_anomalies(sub)

    # Probabilistic diagnostics (PIT, CRPS, sharpness)
    add_plot_probs(sub)
    add_plot_sharpness(sub)
    add_plot_credibility(sub)

    # Drift & temporal
    add_plot_drift(sub)
    add_plot_temporal(sub)

    # Fields & vectors
    add_plot_fields(sub)
    add_plot_velocities(sub)

    # Ground-truth vs prediction
    add_plot_vs(sub)

    # Relationship views (truth vs preds, conditional bands)
    add_plot_cond_relationship(sub)
    add_plot_eval_relationship(sub)

    # errors plot
    add_plot_errors(sub)

    # comparison plots
    add_plot_reliability(sub)
    add_plot_comparison(sub)

    return parser


def main(args: list[str] | None = None) -> None:
    """
    CLI entrypoint. Parse args and dispatch to the
    selected subcommand.
    """
    parser = build_parser()
    ns = parser.parse_args(args=args)
    if hasattr(ns, "func"):
        ns.func(ns)
    else:
        parser.print_help()
