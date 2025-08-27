from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


@pytest.fixture
def demo_csv_reliability(tmp_path: Path) -> Path:
    """
    Binary labels + two probability columns in [0, 1].
    """
    rng = np.random.default_rng(0)
    n = 800
    y = (rng.random(n) < 0.4).astype(int)
    p1 = np.clip(0.35 + 0.30 * rng.random(n), 0, 1)
    p2 = np.clip(0.40 + 0.20 * rng.random(n), 0, 1)

    df = pd.DataFrame({"y": y, "p_m1": p1, "p_m2": p2})
    path = tmp_path / "rel.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def demo_csv_regression(tmp_path: Path) -> Path:
    """
    Regression ground truth + 2 model predictions
    for the radar comparison plot.
    """
    rng = np.random.default_rng(42)
    n = 200
    y = rng.normal(10.0, 5.0, size=n)
    m1 = y + rng.normal(0.0, 1.0, size=n)
    m2 = y + rng.normal(0.0, 2.0, size=n)
    df = pd.DataFrame({"y": y, "m1": m1, "m2": m2})
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def demo_csv_horizons(tmp_path: Path) -> Path:
    """
    Rows are horizons/categories. Columns are quantile
    samples per horizon. We provide 2 samples for each
    of q10, q50, q90.
    """
    rng = np.random.default_rng(123)
    n_rows = 6
    # generate base center per row
    center = np.linspace(5.0, 20.0, n_rows)
    wid = np.linspace(2.0, 6.0, n_rows)

    q10_s1 = center - wid * 0.5 + rng.normal(0, 0.2, n_rows)
    q10_s2 = center - wid * 0.5 + rng.normal(0, 0.2, n_rows)
    q90_s1 = center + wid * 0.5 + rng.normal(0, 0.2, n_rows)
    q90_s2 = center + wid * 0.5 + rng.normal(0, 0.2, n_rows)
    q50_s1 = center + rng.normal(0, 0.1, n_rows)
    q50_s2 = center + rng.normal(0, 0.1, n_rows)

    df = pd.DataFrame(
        {
            "q10_s1": q10_s1,
            "q10_s2": q10_s2,
            "q90_s1": q90_s1,
            "q90_s2": q90_s2,
            "q50_s1": q50_s1,
            "q50_s2": q50_s2,
        }
    )
    path = tmp_path / "horizons.csv"
    df.to_csv(path, index=False)
    return path


# ----------------------- tests: reliability diagram ---------------------


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_reliability_diagram_cli(
    demo_csv_reliability: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "reliability.png"

    # for catching matplotlib layout issue
    # which is harmless.
    with pytest.warns(UserWarning):
        v1 = [
            "plot-reliability-diagram",
            str(demo_csv_reliability),
            yflag,
            "y",
            "--pred",
            "p_m1",
            "--pred",
            "p_m2",
            "--names",
            "A",
            "B",
            "--strategy",
            "quantile",
            "--n-bins",
            "12",
            "--savefig",
            str(out),
        ]
        v2 = [
            "plot-reliability-diagram",
            str(demo_csv_reliability),
            yflag,
            "y",
            "--model",
            "A:p_m1",
            "--model",
            "B:p_m2",
            "--show-brier",
            "--show-ece",
            "--savefig",
            str(out),
        ]
        _try_parse_and_run([v1, v2])
        _expect_file(out)


# ----------------------- tests: polar reliability -----------------------


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_polar_reliability_cli(
    demo_csv_reliability: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "polar_reliability.png"

    v1 = [
        "plot-polar-reliability",
        str(demo_csv_reliability),
        yflag,
        "y",
        "--pred",
        "p_m1",
        "--pred",
        "p_m2",
        "--n-bins",
        "15",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-polar-reliability",
        str(demo_csv_reliability),
        yflag,
        "y",
        "--model",
        "M1:p_m1",
        "--model",
        "M2:p_m2",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# ----------------------- tests: model comparison (radar) ----------------


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_model_comparison_cli(
    demo_csv_regression: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "model_comparison.png"

    # explicit metrics + train times
    v1 = [
        "plot-model-comparison",
        str(demo_csv_regression),
        yflag,
        "y",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--names",
        "Lin",
        "Tree",
        "--metrics",
        "r2",
        "mae",
        "rmse",
        "--train-times",
        "0.1",
        "0.5",
        "--scale",
        "norm",
        "--savefig",
        str(out),
    ]
    # using --model and auto metrics
    v2 = [
        "plot-model-comparison",
        str(demo_csv_regression),
        yflag,
        "y",
        "--model",
        "Lin:m1",
        "--model",
        "Tree:m2",
        "--metrics",
        "auto",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# ----------------------- tests: horizon metrics (polar bars) ------------


@pytest.mark.parametrize("acov", ["default", "half_circle"])
def test_plot_horizon_metrics_cli(
    demo_csv_horizons: Path,
    tmp_path: Path,
    acov: str,
) -> None:
    out = tmp_path / f"horizons_{acov}.png"

    labels = ["H+1", "H+2", "H+3", "H+4", "H+5", "H+6"]

    v1 = [
        "plot-horizon-metrics",
        str(demo_csv_horizons),
        "--qlow",
        "q10_s1",
        "q10_s2",
        "--qup",
        "q90_s1",
        "q90_s2",
        "--q50",
        "q50_s1",
        "q50_s2",
        "--xtick-labels",
        *labels,
        "--acov",
        acov,
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-horizon-metrics",
        str(demo_csv_horizons),
        "--qlow",
        "q10_s1",
        "q10_s2",
        "--qup",
        "q90_s1",
        "q90_s2",
        "--normalize-radius",
        "--show-value-labels",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)
