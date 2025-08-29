from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


def _make_reg_csv(path: Path, n: int = 200) -> Path:
    rng = np.random.default_rng(0)
    y = rng.random(n) * 50.0
    m1 = y + rng.normal(0, 5, n)  # "good" model
    m2 = y - 10 + rng.normal(0, 2, n)  # biased model
    df = pd.DataFrame({"y": y, "m1": m1, "m2": m2})
    f = path / "reg.csv"
    df.to_csv(f, index=False)
    return f


def test_cli_reg_perf_data_mode_norm_none(tmp_path: Path) -> None:
    """Data mode: raw values (norm=none) should render and save."""
    csv = _make_reg_csv(tmp_path, n=180)
    out = tmp_path / "reg_perf_data_none.png"

    v = [
        "plot-regression-performance",
        str(csv),
        "--y-true",
        "y",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--names",
        "A",
        "B",
        "--norm",
        "none",
        "--title",
        "Perf (raw)",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v])
    _expect_file(out)


def test_cli_reg_perf_values_mode_global_bounds(tmp_path: Path) -> None:
    """
    Values mode: with global normalization & bounds.
    Ensures new flags are wired and output is produced.
    """
    out = tmp_path / "reg_perf_values_global.png"

    v = [
        "plot-regression-performance",
        "--metric-values",
        "r2:0.82,0.74",
        "neg_mean_absolute_error:-3.2,-3.6",
        "neg_root_mean_squared_error:-4.1,-4.8",
        "--names",
        "A",
        "B",
        "--norm",
        "global",
        "--global-bounds",
        "r2:0,1",
        "--global-bounds",
        "neg_mean_absolute_error:-5,0",
        "--global-bounds",
        "neg_root_mean_squared_error:-10,0",
        "--min-radius",
        "0.05",
        "--title",
        "Perf (global)",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v])
    _expect_file(out)


def test_cli_reg_perf_values_mode_global_no_clip(tmp_path: Path) -> None:
    """
    Values mode: global norm with out-of-bounds scores and clip disabled.
    Just validates the CLI runs and saves output.
    """
    out = tmp_path / "reg_perf_values_global_noclip.png"

    # r2 slightly > 1, RMSE slightly < min bound to test no-clip path
    v = [
        "plot-regression-performance",
        "--metric-values",
        "r2:1.05,0.95",
        "neg_mean_absolute_error:-6.0,-1.0",
        "neg_root_mean_squared_error:-12.0,-2.0",
        "--names",
        "A",
        "B",
        "--norm",
        "global",
        "--global-bounds",
        "r2:0,1",
        "--global-bounds",
        "neg_mean_absolute_error:-5,0",
        "--global-bounds",
        "neg_root_mean_squared_error:-10,0",
        "--no-clip-to-bounds",  # disable clipping
        "--title",
        "Perf (global no-clip)",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v])
    _expect_file(out)
