from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


@pytest.fixture
def demo_csv_taylor(tmp_path: Path) -> Path:
    """Small regression-like CSV: y, m1, m2."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.normal(0, 1.0, n)
    m1 = 0.9 * y + rng.normal(0, 0.3, n)
    m2 = 0.6 * y + rng.normal(0, 0.8, n)

    df = pd.DataFrame({"y": y, "m1": m1, "m2": m2})
    path = tmp_path / "taylord.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_taylor_diagram_cli(
    demo_csv_taylor: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "plot_taylor_basic.png"

    v1 = [
        "plot-taylor-diagram",
        str(demo_csv_taylor),
        yflag,
        "y",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--names",
        "A",
        "B",
        "--acov",
        "half_circle",
        "--zero-location",
        "W",
        "--direction",
        "-1",
        "--marker",
        "o",
        "--corr-steps",
        "6",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-taylor-diagram",
        str(demo_csv_taylor),
        yflag,
        "y",
        "--model",
        "A:m1",
        "--model",
        "B:m2",
        "--only-points",
        "--draw-ref-arc",
        "--angle-to-corr",
        "--savefig",
        str(out),
    ]

    _try_parse_and_run([v1, v2])
    _expect_file(out)


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_plot_taylor_diagram_in_cli(
    demo_csv_taylor: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    out = tmp_path / "plot_taylor_in.png"

    v1 = [
        "plot-taylor-diagram-in",
        str(demo_csv_taylor),
        yflag,
        "y",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--names",
        "A",
        "B",
        "--acov",
        "half_circle",
        "--zero-location",
        "E",
        "--direction",
        "-1",
        "--cmap",
        "viridis",
        "--shading",
        "auto",
        "--shading-res",
        "180",
        "--radial-strategy",
        "convergence",
        "--norm-c",
        "--norm-range",
        "0.0,1.0",
        "--cbar",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-taylor-diagram-in",
        str(demo_csv_taylor),
        yflag,
        "y",
        "--model",
        "A:m1",
        "--model",
        "B:m2",
        "--radial-strategy",
        "norm_r",
        "--cmap",
        "plasma",
        "--savefig",
        str(out),
    ]

    _try_parse_and_run([v1, v2])
    _expect_file(out)


def test_taylor_diagram_cli_stats_mode(tmp_path: Path) -> None:
    """Stats-mode: --stddev & --corrcoef (no input file)."""
    out = tmp_path / "taylor_stats.png"

    v = [
        "taylor-diagram",
        "--stddev",
        "1.1",
        "0.8",
        "--corrcoef",
        "0.92",
        "0.65",
        "--names",
        "M1",
        "M2",
        "--ref-std",
        "1.0",
        "--draw-ref-arc",
        "--cmap",
        "viridis",
        "--radial-strategy",
        "rwf",
        "--norm-c",
        "--power-scaling",
        "1.2",
        "--marker",
        "^",
        "--savefig",
        str(out),
    ]

    _try_parse_and_run([v])
    _expect_file(out)


@pytest.mark.parametrize("yflag", ["--true-col", "--y-true"])
def test_taylor_diagram_cli_data_mode(
    demo_csv_taylor: Path,
    tmp_path: Path,
    yflag: str,
) -> None:
    """Data-mode: y + preds."""
    out = tmp_path / "taylor_data.png"

    v1 = [
        "taylor-diagram",
        str(demo_csv_taylor),
        yflag,
        "y",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--names",
        "A",
        "B",
        "--cmap",
        "plasma",
        "--radial-strategy",
        "center_focus",
        "--norm-c",
        "--savefig",
        str(out),
    ]
    v2 = [
        "taylor-diagram",
        str(demo_csv_taylor),
        yflag,
        "y",
        "--model",
        "A:m1",
        "--model",
        "B:m2",
        "--marker",
        "s",
        "--savefig",
        str(out),
    ]

    _try_parse_and_run([v1, v2])
    _expect_file(out)
