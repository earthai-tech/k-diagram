from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


# -------- fixtures ------------------------------------------------
@pytest.fixture
def demo_csv_rel(tmp_path: Path) -> Path:
    """
    Minimal CSV covering all new commands:

    - y_true: 'actual'
    - 3 quantile columns for two 'models'
    """
    n = 120
    rng = np.random.default_rng(42)

    actual = rng.normal(10.0, 2.5, size=n)

    # Model 1 (baseline quantiles)
    q10_col = actual - abs(rng.normal(2.0, 0.3, n))
    q50_col = actual + rng.normal(0.0, 0.2, n)
    q90_col = actual + abs(rng.normal(2.0, 0.3, n))

    # Model 2 (slightly wider) — include q50_2024 (needed by tests)
    q10_b = actual - abs(rng.normal(2.5, 0.4, n))
    q50_b = actual + rng.normal(0.0, 0.35, n)
    q90_b = actual + abs(rng.normal(2.5, 0.4, n))

    df = pd.DataFrame(
        {
            "actual": actual,
            "q10": q10_col,
            "q50": q50_col,
            "q90": q90_col,
            "q10_2024": q10_b,
            "q50_2024": q50_b,  # <-- added
            "q90_2024": q90_b,
        }
    )
    path = tmp_path / "demo_probs.csv"
    df.to_csv(path, index=False)
    return path


# -------- tests: plot-relationship -------------------------------
@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_relationship_cli(
    demo_csv_rel: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "relationship.png"

    # Use point predictions (q50 columns as proxies)
    v1 = [
        "plot-relationship",
        str(demo_csv_rel),
        yflag,
        "actual",
        "--pred",
        "q50",
        "--pred",
        "q50_2024",
        "--names",
        "M1",
        "M2",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-relationship",
        str(demo_csv_rel),
        yflag,
        "actual",
        "--model",
        "M1:q50",
        "--model",
        "M2:q50_2024",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# -------- tests: plot-conditional-quantiles ----------------------
@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
@pytest.mark.parametrize("qflag", ["--quantiles", "--q-levels"])
def test_plot_conditional_quantiles_cli(
    demo_csv_rel: Path,
    tmp_path: Path,
    yflag: str,
    qflag: str,
) -> None:
    out = tmp_path / "cond_quant.png"

    v1 = [
        "plot-conditional-quantiles",
        str(demo_csv_rel),
        yflag,
        "actual",
        "--pred",
        "q10,q50,q90",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-conditional-quantiles",
        str(demo_csv_rel),
        yflag,
        "actual",
        "--pred-cols",
        "q10,q50,q90",
        qflag,
        "0.1,0.5,0.9",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# -------- tests: plot-residual-relationship ----------------------
@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_residual_relationship_cli(
    demo_csv_rel: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "residual_rel.png"

    v1 = [
        "plot-residual-relationship",
        str(demo_csv_rel),
        yflag,
        "actual",
        "--pred",
        "q50",
        "--pred",
        "q50_2024",
        "--names",
        "A",
        "B",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-residual-relationship",
        str(demo_csv_rel),
        yflag,
        "actual",
        "--model",
        "A:q50",
        "--model",
        "B:q50_2024",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# -------- tests: plot-error-relationship -------------------------
@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_error_relationship_cli(
    demo_csv_rel: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "error_rel.png"

    v1 = [
        "plot-error-relationship",
        str(demo_csv_rel),
        yflag,
        "actual",
        "--pred",
        "q50",
        "--pred",
        "q50_2024",
        "--names",
        "A",
        "B",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-error-relationship",
        str(demo_csv_rel),
        yflag,
        "actual",
        "--model",
        "A:q50",
        "--model",
        "B:q50_2024",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)
