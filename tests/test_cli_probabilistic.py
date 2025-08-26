# tests/test_cli_probs.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli import build_parser


# --------------------------- helpers ---------------------------------


def _expect_file(path: Path) -> None:
    assert path.exists(), f"missing: {path}"
    assert path.stat().st_size > 0, f"empty: {path}"


def _try_parse_and_run(
    argv_variants: Iterable[list[str]],
) -> None:
    """
    Try a list of argv variants until one parses, then run it.
    If none parse, raise the last SystemExit to surface details.
    """
    parser = build_parser()
    last_err: SystemExit | None = None
    for argv in argv_variants:
        try:
            ns = parser.parse_args(argv)
            assert hasattr(ns, "func"), "subcommand not bound"
            ns.func(ns)
            return
        except SystemExit as e:  # bad argv (unknown flags etc.)
            last_err = e
    # If we got here, nothing parsed; re-raise the last failure
    raise last_err  # type: ignore[misc]


# ---------------------------- fixtures --------------------------------


@pytest.fixture()
def demo_csv_prob(tmp_path: Path) -> Path:
    """
    Minimal CSV covering all new commands:

    - y_true: 'actual'
    - 3 quantile columns for two 'models'
    - a cyclic 'theta' column used by credibility plot
    """
    n = 120
    rng = np.random.default_rng(42)

    actual = rng.normal(10.0, 2.5, size=n)
    q = np.array([0.1, 0.5, 0.9])

    # Model 1 (base)
    scale1 = 2.5
    q10 = np.quantile(actual[:, None] + rng.normal(0, scale1, (n, 3)), q, axis=1)
    # cheat: construct q10/q50/q90 in a simple, monotone way
    q10_col = actual - abs(rng.normal(2.0, 0.3, n))
    q50_col = actual + rng.normal(0.0, 0.2, n)
    q90_col = actual + abs(rng.normal(2.0, 0.3, n))

    # Model 2 (slightly wider)
    q10_b = actual - abs(rng.normal(2.5, 0.4, n))
    q90_b = actual + abs(rng.normal(2.5, 0.4, n))

    # angle over [0, 2pi)
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)

    df = pd.DataFrame(
        {
            "actual": actual,
            "q10": q10_col,
            "q50": q50_col,
            "q90": q90_col,
            "q10_2024": q10_b,
            "q90_2024": q90_b,
            "theta": theta,
        }
    )
    path = tmp_path / "demo_probs.csv"
    df.to_csv(path, index=False)
    return path


# --------------------------- presence test ----------------------------


def test_prob_commands_present() -> None:
    parser = build_parser()
    sub_names = []
    for act in parser._actions:  # type: ignore[attr-defined]
        if isinstance(getattr(act, "choices", None), dict):
            sub_names.extend(list(act.choices.keys()))
    need = {
        "plot-pit-histogram",
        "plot-crps-comparison",
        "plot-polar-sharpness",
        "plot-calibration-sharpness",
        "plot-credibility-bands",
    }
    missing = need.difference(sub_names)
    assert not missing, f"missing subcommands: {sorted(missing)}"


# -------------------------- PIT histogram -----------------------------


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
@pytest.mark.parametrize("qflag", ["--quantiles", "--q-levels"])
def test_plot_pit_histogram_cli(
    demo_csv_prob: Path,
    tmp_path: Path,
    yflag: str,
    qflag: str,
) -> None:
    out = tmp_path / "pit_hist.png"
    primary = [
        "plot-pit-histogram",
        str(demo_csv_prob),
        yflag,
        "actual",
        "--pred",
        "q10,q50,q90",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    alt = [
        "plot-pit-histogram",
        str(demo_csv_prob),
        yflag,
        "actual",
        "--pred-cols",
        "q10,q50,q90",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([primary, alt])
    _expect_file(out)


# ------------------------- CRPS comparison ----------------------------


@pytest.mark.parametrize("qflag", ["--quantiles", "--q-levels"])
def test_plot_crps_comparison_cli(
    demo_csv_prob: Path,
    tmp_path: Path,
    qflag: str,
) -> None:
    out = tmp_path / "crps.png"
    # variant using --pred twice + explicit names
    v1 = [
        "plot-crps-comparison",
        str(demo_csv_prob),
        "--true-col",
        "actual",
        "--pred",
        "q10,q50,q90",
        "--pred",
        "q10_2024,q50,q90_2024",
        "--names",
        "M1",
        "M2",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    # variant using --model (name:cols) twice, no --names needed
    v2 = [
        "plot-crps-comparison",
        str(demo_csv_prob),
        "--y-true",
        "actual",
        "--model",
        "M1:q10,q50,q90",
        "--model",
        "M2:q10_2024,q50,q90_2024",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# --------------------------- Polar sharpness --------------------------


@pytest.mark.parametrize("qflag", ["--quantiles", "--q-levels"])
def test_plot_polar_sharpness_cli(
    demo_csv_prob: Path,
    tmp_path: Path,
    qflag: str,
) -> None:
    out = tmp_path / "sharpness.png"
    v1 = [
        "plot-polar-sharpness",
        str(demo_csv_prob),
        "--pred",
        "q10,q50,q90",
        "--pred",
        "q10_2024,q50,q90_2024",
        "--names",
        "A",
        "B",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-polar-sharpness",
        str(demo_csv_prob),
        "--model",
        "A:q10,q50,q90",
        "--model",
        "B:q10_2024,q50,q90_2024",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# --------------------- Calibration vs sharpness ----------------------


@pytest.mark.parametrize("qflag", ["--quantiles", "--q-levels"])
@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_calibration_sharpness_cli(
    demo_csv_prob: Path,
    tmp_path: Path,
    qflag: str,
    yflag: str,
) -> None:
    out = tmp_path / "cal_sharp.png"
    v1 = [
        "plot-calibration-sharpness",
        str(demo_csv_prob),
        yflag,
        "actual",
        "--pred",
        "q10,q50,q90",
        "--pred",
        "q10_2024,q50,q90_2024",
        "--names",
        "Good",
        "Wide",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-calibration-sharpness",
        str(demo_csv_prob),
        yflag,
        "actual",
        "--model",
        "Good:q10,q50,q90",
        "--model",
        "Wide:q10_2024,q50,q90_2024",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# ------------------------- Credibility bands -------------------------


@pytest.mark.parametrize("qcols_style", ["split", "csv"])
def test_plot_credibility_bands_cli(
    demo_csv_prob: Path,
    tmp_path: Path,
    qcols_style: str,
) -> None:
    out = tmp_path / "credibility.png"
    if qcols_style == "split":
        argv = [
            "plot-credibility-bands",
            str(demo_csv_prob),
            "--q-cols",
            "q10",
            "q50",
            "q90",
            "--theta-col",
            "theta",
            "--theta-bins",
            "12",
            "--savefig",
            str(out),
        ]
    else:
        argv = [
            "plot-credibility-bands",
            str(demo_csv_prob),
            "--q-cols",
            "q10,q50,q90",
            "--theta-col",
            "theta",
            "--theta-bins",
            "12",
            "--savefig",
            str(out),
        ]
    _try_parse_and_run([argv])
    _expect_file(out)
