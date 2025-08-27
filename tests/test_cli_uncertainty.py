from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from kdiagram.cli import build_parser


@pytest.fixture(scope="session")
def demo_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a small CSV with all columns needed by the CLIs."""
    tmp = tmp_path_factory.mktemp("kdiagram_cli_demo")
    p = tmp / "demo.csv"

    rng = np.random.default_rng(42)
    n = 80

    # base signals
    x = np.linspace(0, 4 * np.pi, n)
    actual = 10 + 3 * np.sin(x) + rng.normal(0, 0.6, n)

    # quantiles around 'actual'
    width = 1.0 + 0.5 * (1 + np.sin(x / 2))
    q50 = actual + rng.normal(0, 0.2, n)
    q10 = q50 - width
    q90 = q50 + width

    # another horizon (year) pair
    q10_2024 = q10 + 0.2
    q90_2024 = q90 + 0.6

    # yet another pair
    q10_2025 = q10 + 0.5
    q90_2025 = q90 + 1.0

    # velocity/central series over time
    v50_1 = q50
    v50_2 = q50 + rng.normal(0, 0.4, n)
    v50_3 = q50 + rng.normal(0, 0.6, n)

    # polar field helpers
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rval = np.clip(np.abs(np.sin(x)) + 0.1, 0, None)
    u = np.cos(theta)
    v = np.sin(theta)
    color_val = (rval - rval.min()) / (np.ptp(rval) + 1e-9)

    df = pd.DataFrame(
        {
            "actual": actual,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "q10_2024": q10_2024,
            "q90_2024": q90_2024,
            "q10_2025": q10_2025,
            "q90_2025": q90_2025,
            "v50_1": v50_1,
            "v50_2": v50_2,
            "v50_3": v50_3,
            "theta": theta,
            "rval": rval,
            "u": u,
            "v": v,
            "color_val": color_val,
        }
    )
    p.write_text(df.to_csv(index=False), encoding="utf-8")
    return p


def _run(argv: list[str]) -> None:
    """Parse and execute a CLI command (ns.func(ns))."""
    parser = build_parser()
    ns = parser.parse_args(argv)
    assert hasattr(ns, "func"), f"no func bound for: {argv}"
    ns.func(ns)


def _expect_file(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 0


def test_cli_subcommands_present():
    parser = build_parser()
    sub_names = set()
    for act in parser._subparsers._actions:  # type: ignore[attr-defined]
        if getattr(act, "choices", None):
            sub_names |= set(act.choices.keys())
    # expected names (keep in sync with add_* modules)
    expected = {
        "plot-coverage",
        "plot-coverage-diagnostic",
        "plot-interval-width",
        "plot-interval-consistency",
        "plot-anomaly-magnitude",
        "plot-model-drift",
        "plot-uncertainty-drift",
        "plot-temporal-uncertainty",
        "plot-velocity",
        "plot-radial-density-ring",
        "plot-polar-heatmap",
        "plot-polar-quiver",
        "plot-actual-vs-predicted",
    }
    missing = expected - sub_names
    assert not missing, f"missing subcommands: {sorted(missing)}"


def test_plot_interval_width_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "iw.png"
    _run(
        [
            "plot-interval-width",
            str(demo_csv),
            "--q-cols",
            "q10",
            "q90",
            "--z-col",
            "q50",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_interval_consistency_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "iw_consistency.png"
    _run(
        [
            "plot-interval-consistency",
            str(demo_csv),
            "--q10-cols",
            "q10",
            "q10_2024",
            "q10_2025",
            "--q90-cols",
            "q90",
            "q90_2024",
            "q90_2025",
            "--use-cv",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_anomaly_magnitude_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "anomaly.png"

    with pytest.warns(UserWarning):
        _run(
            [
                "plot-anomaly-magnitude",
                str(demo_csv),
                "--actual-col",
                "actual",
                "--q-cols",
                "q10",
                "q90",
                "--cbar",
                "--savefig",
                str(out),
            ]
        )
        _expect_file(out)


def test_plot_model_drift_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "model_drift.png"
    _run(
        [
            "plot-model-drift",
            str(demo_csv),
            "--q10-cols",
            "q10",
            "q10_2024",
            "q10_2025",
            "--q90-cols",
            "q90",
            "q90_2024",
            "q90_2025",
            "--horizons",
            "2023",
            "2024",
            "2025",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_uncertainty_drift_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "uncertainty_drift.png"
    _run(
        [
            "plot-uncertainty-drift",
            str(demo_csv),
            "--qlow-cols",
            "q10",
            "q10_2024",
            "q10_2025",
            "--qup-cols",
            "q90",
            "q90_2024",
            "q90_2025",
            "--dt-labels",
            "2023",
            "2024",
            "2025",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_temporal_uncertainty_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "temporal.png"
    _run(
        [
            "plot-temporal-uncertainty",
            str(demo_csv),
            "--q-cols",
            "q10",
            "q50",
            "q90",
            "--names",
            "Q10",
            "Q50",
            "Q90",
            "--normalize",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_velocity_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "velocity.png"
    _run(
        [
            "plot-velocity",
            str(demo_csv),
            "--q50-cols",
            "v50_1",
            "v50_2",
            "v50_3",
            "--cbar",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_polar_heatmap_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "heatmap.png"
    _run(
        [
            "plot-polar-heatmap",
            str(demo_csv),
            "--r-col",
            "rval",
            "--theta-col",
            "theta",
            "--r-bins",
            "16",
            "--theta-bins",
            "24",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_polar_quiver_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "quiver.png"
    _run(
        [
            "plot-polar-quiver",
            str(demo_csv),
            "--r-col",
            "rval",
            "--theta-col",
            "theta",
            "--u-col",
            "u",
            "--v-col",
            "v",
            "--color-col",
            "color_val",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_radial_density_ring_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "ring.png"
    _run(
        [
            "plot-radial-density-ring",
            str(demo_csv),
            "--kind",
            "direct",
            "--target-cols",
            "rval",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_actual_vs_predicted_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "avp.png"
    _run(
        [
            "plot-actual-vs-predicted",
            str(demo_csv),
            "--actual-col",
            "actual",
            "--pred-col",
            "q50",
            "--line",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


def test_plot_coverage_diagnostic_cli(demo_csv: Path, tmp_path: Path):
    out = tmp_path / "coverage_diag.png"
    _run(
        [
            "plot-coverage-diagnostic",
            str(demo_csv),
            "--actual-col",
            "actual",
            "--q-cols",
            "q10",
            "q90",
            "--savefig",
            str(out),
        ]
    )
    _expect_file(out)


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_coverage_cli_comma_sep_token(
    demo_csv: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "coverage.png"
    argv = [
        "plot-coverage",
        str(demo_csv),
        yflag,
        "actual",
        "--pred",
        "q10,q50,q90",
        "--pred",
        "q10_2024,q50,q90_2024",
        "--names",
        "M1",
        "M2",
        "--kind",
        "bar",
        "--savefig",
        str(out),
    ]
    parser = build_parser()
    ns = parser.parse_args(argv)
    assert hasattr(ns, "func")
    ns.func(ns)
    _expect_file(out)


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_coverage_cli_space_sep_token(
    demo_csv: Path, tmp_path: Path, yflag: str
):
    out = tmp_path / "coverage.png"
    argv = [
        "plot-coverage",
        str(demo_csv),
        yflag,
        "actual",
        "--pred",
        "q10,q50,q90",
        "--pred",
        "q10_2024,q50,q90_2024",
        "--names",
        "M1,M2",  # <-- one token, comma-separated
        "--kind",
        "bar",
        "--savefig",
        str(out),
    ]
    parser = build_parser()
    ns = parser.parse_args(argv)
    assert hasattr(ns, "func")
    ns.func(ns)
    _expect_file(out)
