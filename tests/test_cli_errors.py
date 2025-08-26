from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli import build_parser


# ------------------------ helpers --------------------------------
def _expect_file(path: Path) -> None:
    assert path.exists(), f"missing: {path}"
    assert path.stat().st_size > 0, f"empty: {path}"


def _try_parse_and_run(variants: Iterable[list[str]]) -> None:
    """
    Try argv variants in order. If one parses and runs, stop.
    Raise the last error if all fail.
    """
    last_err: BaseException | None = None
    for argv in variants:
        parser = build_parser()
        try:
            ns = parser.parse_args(argv)
            if not hasattr(ns, "func"):
                raise SystemExit("no func bound")
            ns.func(ns)
            return
        except SystemExit as e:
            last_err = e
        except Exception as e:  # pragma: no cover
            last_err = e
    if last_err:
        raise last_err


# ------------------------ fixture --------------------------------
@pytest.fixture
def demo_csv_errors(tmp_path: Path) -> Path:
    """
    One CSV that supports all three error CLI tests.

    Columns:
      - error violins: err_a, err_b, err_c
      - error bands: err, month
      - error ellipses: r, theta_deg, r_std, theta_std_deg, priority
    """
    n = 180
    rng = np.random.default_rng(7)

    month = np.tile(np.arange(1, 13), n // 12 + (n % 12 != 0))[:n]
    season = 0.6 * np.sin(month * 2 * np.pi / 12.0)

    base = rng.normal(0.0, 1.5, size=n) + season
    err_a = base
    err_b = 1.2 * base + rng.normal(0, 0.4, size=n)
    err_c = rng.normal(0.0, 2.5, size=n)
    err = err_a

    r = rng.uniform(10.0, 50.0, size=n)
    theta_deg = (np.linspace(0.0, 360.0, n, endpoint=False)) % 360
    r_std = rng.uniform(0.5, 3.0, size=n)
    theta_std_deg = rng.uniform(2.0, 8.0, size=n)
    priority = rng.integers(1, 5, size=n)

    df = pd.DataFrame(
        {
            "month": month,
            "err": err,
            "err_a": err_a,
            "err_b": err_b,
            "err_c": err_c,
            "r": r,
            "theta_deg": theta_deg,
            "r_std": r_std,
            "theta_std_deg": theta_std_deg,
            "priority": priority,
        }
    )
    path = tmp_path / "demo_errors.csv"
    df.to_csv(path, index=False)
    return path


# -------------------- plot-error-violins --------------------------
def test_plot_error_violins_cli(
    demo_csv_errors: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "error_violins.png"

    v1 = [
        "plot-error-violins",
        str(demo_csv_errors),
        "--error",
        "err_a",
        "--error",
        "err_b,err_c",
        "--names",
        "A",
        "B",
        "C",
        "--alpha",
        "0.7",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-error-violins",
        str(demo_csv_errors),
        "--error-cols",
        "err_a,err_b,err_c",
        "--cmap",
        "plasma",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# --------------------- plot-error-bands ---------------------------
@pytest.mark.parametrize("use_period", [True, False])
def test_plot_error_bands_cli(
    demo_csv_errors: Path,
    tmp_path: Path,
    use_period: bool,
) -> None:
    out = tmp_path / f"error_bands_{use_period}.png"

    common = [
        "plot-error-bands",
        str(demo_csv_errors),
        "--error-col",
        "err",
        "--theta-col",
        "month",
        "--theta-bins",
        "12",
        "--n-std",
        "1.5",
        "--color",
        "#2980B9",
        "--alpha",
        "0.35",
        "--savefig",
        str(out),
    ]

    if use_period:
        common[6:6] = ["--theta-period", "12"]

    _try_parse_and_run([common])
    _expect_file(out)


# ------------------- plot-error-ellipses --------------------------
def test_plot_error_ellipses_cli(
    demo_csv_errors: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "error_ellipses.png"

    v1 = [
        "plot-error-ellipses",
        str(demo_csv_errors),
        "--r-col",
        "r",
        "--theta-col",
        "theta_deg",
        "--r-std-col",
        "r_std",
        "--theta-std-col",
        "theta_std_deg",
        "--color-col",
        "priority",
        "--n-std",
        "2.0",
        "--alpha",
        "0.7",
        "--edgecolor",
        "black",
        "--linewidth",
        "0.5",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-error-ellipses",
        str(demo_csv_errors),
        "--r-col",
        "r",
        "--theta-col",
        "theta_deg",
        "--r-std-col",
        "r_std",
        "--theta-std-col",
        "theta_std_deg",
        "--n-std",
        "1.5",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)
