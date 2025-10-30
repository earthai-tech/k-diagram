
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib

# Headless backend for CI
matplotlib.use("Agg")

from kdiagram.cli import build_parser


@pytest.fixture
def demo_csv_ctx(tmp_path: Path) -> Path:
    """Minimal dataset for correlation/error context CLIs."""
    n = 64
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 1, n)
    m1 = actual + rng.normal(0, 0.5, n)
    m2 = actual + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"actual": actual, "m1": m1, "m2": m2})
    p = tmp_path / "ctx.csv"
    df.to_csv(p, index=False)
    return p


def _run(argv: list[str]) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    assert hasattr(ns, "func"), f"no func bound for: {argv}"
    ns.func(ns)


def _expect_file(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 0


def test_plot_scatter_corr_cli(monkeypatch: pytest.MonkeyPatch,
                               demo_csv_ctx: Path, tmp_path: Path) -> None:
    # Mock the heavy plotting function at the CLI module level
    def fake_scatter_correlation(**kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if kwargs.get("show_identity_line", True):
            ax.plot([0, 1], [0, 1])
        sf = kwargs.get("savefig")
        if sf:
            fig.savefig(sf, dpi=kwargs.get("dpi", 100))
        plt.close(fig)

    monkeypatch.setattr(
        "kdiagram.cli.plot_context_corr.plot_scatter_correlation",
        fake_scatter_correlation,
        raising=True,
    )

    out = tmp_path / "scatter_corr.png"
    _run(
        [
            "plot-scatter-corr",
            str(demo_csv_ctx),
            "--actual-col", "actual",
            "--pred-cols", "m1", "m2",
            "--names", "M1", "M2",
            "--show-identity-line",
            "--savefig", str(out),
        ]
    )
    _expect_file(out)


def test_plot_error_autocorr_cli_and_guard(monkeypatch: pytest.MonkeyPatch,
                                           demo_csv_ctx: Path, tmp_path: Path):
    # Fake ACF plotter; just produce an image when savefig is passed
    def fake_error_acf(**kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sf = kwargs.get("savefig")
        if sf:
            fig.savefig(sf, dpi=kwargs.get("dpi", 100))
        plt.close(fig)

    monkeypatch.setattr(
        "kdiagram.cli.plot_context_corr.plot_error_autocorrelation",
        fake_error_acf,
        raising=True,
    )

    # Happy path: exactly one pred (as enforced by CLI) → saves a figure
    out = tmp_path / "error_acf.png"
    _run(
        [
            "plot-error-autocorr",
            str(demo_csv_ctx),
            "--actual-col", "actual",
            "--pred", "m1",           # exactly one
            "--savefig", str(out),
        ]
    )
    _expect_file(out)

    # Guard path: two preds → SystemExit from CLI (pre-plot) check
    parser = build_parser()
    ns = parser.parse_args(
        ["plot-error-autocorr", str(demo_csv_ctx),
         "--actual-col", "actual", "--pred", "m1", "m2"]
    )
    with pytest.raises(SystemExit):  # enforced in CLI layer
        ns.func(ns)


def test_plot_error_dist_cli(monkeypatch: pytest.MonkeyPatch,
                             demo_csv_ctx: Path, tmp_path: Path):
    # plot_error_distribution returns an Axes; CLI handles saving the figure
    def fake_error_dist(**kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist([0, 1, 2])
        return ax

    monkeypatch.setattr(
        "kdiagram.cli.plot_context_err.plot_error_distribution",
        fake_error_dist,
        raising=True,
    )

    out = tmp_path / "err_dist.png"
    _run(
        [
            "plot-error-dist",
            str(demo_csv_ctx),
            "--actual-col", "actual",
            "--pred-col", "m1",
            "--bins", "20",
            "--savefig", str(out),
        ]
    )
    _expect_file(out)


def test_plot_error_pacf_cli(monkeypatch: pytest.MonkeyPatch,
                             demo_csv_ctx: Path, tmp_path: Path):
    # Avoid statsmodels dependency by mocking the PACF plotter
    def fake_pacf(**kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sf = kwargs.get("savefig")
        if sf:
            fig.savefig(sf, dpi=kwargs.get("dpi", 100))
        plt.close(fig)

    monkeypatch.setattr(
        "kdiagram.cli.plot_context_err.plot_error_pacf",
        fake_pacf,
        raising=True,
    )

    out = tmp_path / "err_pacf.png"
    _run(
        [
            "plot-error-pacf",
            str(demo_csv_ctx),
            "--actual-col", "actual",
            "--pred-col", "m1",
            "--lags", "24",
            "--savefig", str(out),
        ]
    )
    _expect_file(out)

