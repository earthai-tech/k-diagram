
from __future__ import annotations

from pathlib import Path
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import pytest

import kdiagram.utils.plot as kplot

matplotlib.use("Agg")  # headless for CI

@pytest.fixture(autouse=True)
def patch_get_cmap(monkeypatch):
    """Signature-compatible shim for compat.get_cmap used by utils.plot.

    Matches kdiagram.compat.matplotlib.get_cmap(name, default='viridis',
    allow_none=False, error=None, failsafe='continuous', **kw) and uses
    modern Matplotlib APIs to avoid deprecation warnings.
    """
    def _resolve(name):
        # Allow passing an actual Colormap instance
        if isinstance(name, Colormap):
            return name
        if name is None:
            return None
        try:
            # Modern, non-deprecated API (MPL ≥3.5)
            return matplotlib.colormaps.get_cmap(name)
        except Exception:
            return None

    def shim(name, default="viridis", allow_none=False, error=None, failsafe="continuous", **kw):
        if name is None and allow_none:
            return None
        cmap = _resolve(name)
        if cmap is None:
            # fall back to default, then to viridis; stay silent in tests
            cmap = _resolve(default) or _resolve("viridis")
        return cmap

    # Patch the symbol that utils.plot imported at import-time
    monkeypatch.setattr(kplot, "get_cmap", shim, raising=True)

# --- grid & kind utilities -------------------------------------------------

def test_set_axis_grid_toggles():
    fig, ax = plt.subplots()
    # on → gridlines exist and are visible
    kplot.set_axis_grid(ax, True, {"linestyle": "--", "alpha": 0.3})
    assert any(gl.get_visible() for gl in ax.get_xgridlines() + ax.get_ygridlines())
    # off → all gridlines hidden
    kplot.set_axis_grid(ax, False)
    assert all(not gl.get_visible() for gl in ax.get_xgridlines() + ax.get_ygridlines())
    plt.close(fig)

def test_set_axis_grid_multi_axes():
    fig, axs = plt.subplots(1, 2)
    kplot.set_axis_grid(list(axs), True)
    assert all(any(gl.get_visible() for gl in a.get_xgridlines() + a.get_ygridlines()) for a in axs)
    kplot.set_axis_grid(list(axs), False)
    assert all(all(not gl.get_visible() for gl in a.get_xgridlines() + a.get_ygridlines()) for a in axs)
    plt.close(fig)

def test_is_valid_kind_normalization_and_validation():
    assert kplot.is_valid_kind("LineChart") == "line"
    assert kplot.is_valid_kind("heat_map") == "heatmap"
    assert kplot.is_valid_kind("box_plot", valid_kinds=["scatter", "box"]) == "box"
    with pytest.raises(ValueError):
        kplot.is_valid_kind("spiderweb", valid_kinds=["line", "scatter"])


# --- KDE prep / normalization (gaussian_kde mocked) ------------------------

def test_prepare_data_for_kde_and_normalize(monkeypatch):
    # Fake gaussian_kde so SciPy isn't required
    class FakeKDE:
        def __init__(self, x, bw_method=None):
            self.x = np.asarray(x)
            self.bw = bw_method
        def __call__(self, grid):
            g = np.asarray(grid)
            # simple bell-ish curve for testing
            mu = np.mean(self.x)
            sig = np.std(self.x) if np.std(self.x) > 0 else 1.0
            return np.exp(-0.5 * ((g - mu) / sig) ** 2)

    monkeypatch.setattr(kplot, "gaussian_kde", FakeKDE, raising=True)

    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=256)
    grid, pdf = kplot.prepare_data_for_kde(data)
    assert grid.shape == (512,)
    assert pdf.shape == (512,)
    nrm = kplot.normalize_pdf(pdf)
    assert nrm.max() == pytest.approx(1.0)

    # Empty after filtering → ValueError
    with pytest.raises(ValueError):
        kplot.prepare_data_for_kde(np.array([np.nan, np.inf]))


# --- histogram / simple axes helpers --------------------------------------

def test_setup_axes_and_drawers(tmp_path: Path):
    ax = kplot.setup_plot_axes(title="Demo", x_label="X", y_label="Y")
    # add KDE line
    x = np.linspace(-1, 1, 100)
    y = np.sin(np.pi * x) ** 2
    kplot.add_kde_to_plot(x, y, ax, color="k", line_width=1.5)
    # add histogram
    kplot.add_histogram_to_plot(np.random.randn(200), ax, bins=20, hist_color="gray")
    out = tmp_path / "axes_demo.png"
    ax.figure.savefig(out)
    assert out.exists() and out.stat().st_size > 0
    plt.close(ax.figure)


# --- color sampling --------------------------------------------------------

def test_sample_colors_listed_and_continuous():
    # listed palette (tab10) should cycle/spread when n > m
    cols = kplot._sample_colors("tab10", 12)
    assert len(cols) == 12
    # continuous map with trim
    cols2 = kplot._sample_colors("viridis", 3, trim=0.1)
    assert len(cols2) == 3
    # invalid n
    with pytest.raises(ValueError):
        kplot._sample_colors("viridis", 0)


# --- polar axis tools ------------------------------------------------------

def test_canonical_acov_and_resolve_span():
    assert kplot.canonical_acov("full") == "default"
    assert kplot.canonical_acov("Half-Circle") == "half_circle"
    assert math.isclose(kplot.resolve_polar_span("default"), 2 * math.pi)
    assert math.isclose(kplot.resolve_polar_span("quarter"), 0.5 * math.pi)
    with pytest.raises(ValueError):
        kplot.canonical_acov("weird", raise_on_invalid=True)

def test_setup_polar_axes_and_set_span_warnings():
    fig, ax, span = kplot.setup_polar_axes(None, acov="half", zero_at="E", clockwise=False)
    assert math.isclose(span, math.pi)
    # invalid zero_at → warns and falls back to 'N'
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _, ax2, span2 = kplot.setup_polar_axes(None, acov="quarter", zero_at="X")
        assert any("Fallback to 'N'" in str(w.message) for w in rec)
        assert math.isclose(span2, 0.5 * math.pi)
    # set span on existing polar axes
    s = kplot.set_polar_angular_span(ax, acov="eighth")
    assert math.isclose(s, 0.25 * math.pi)
    plt.close(fig); plt.close(ax2.figure)

def test_resolve_polar_axes_and_map_theta(tmp_path: Path):
    ax = kplot.resolve_polar_axes(acov="default", zero_at="N", clockwise=True)
    assert getattr(ax, "name", "") == "polar"
    span = 0.5 * math.pi
    th = np.array([0.0, math.pi, 2 * math.pi, 3 * math.pi / 2])
    m = kplot.map_theta_to_span(th, span=span)
    # 0 → 0 ; π → span/2 ; 2π → 0 ; 3π/2 → 3/4 span
    assert np.allclose(m, [0.0, span / 2.0, 0.0, 0.75 * span])
    ax.figure.savefig(tmp_path / "polar.png")
    plt.close(ax.figure)

def test_default_theta_ticks_and_acov_aliases():
    ticks, labels = kplot._default_theta_ticks(2 * math.pi)
    assert len(ticks) == 12 and labels[0] == "0°" and labels[-1] == "330°"
    assert math.isclose(kplot.acov_to_span("full"), 2 * math.pi)

def test_fmt_pref_list_and_warn_preference(monkeypatch):
    # stable degrees for message
    msg = kplot._fmt_pref_list(["default", "half_circle"])
    assert "360°" in msg and "180°" in msg

    # columns_manager may coerce inputs; ensure it behaves in test
    monkeypatch.setattr(kplot, "columns_manager", lambda x, **k: list(x) if isinstance(x, (list, tuple)) else [x], raising=True)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        issued = kplot.warn_acov_preference(
            "half",
            preferred="default",
            plot_name="demo",
            reason="better comparison",
            advice="still OK.",
        )
        assert issued is True
        assert any("Using acov='half_circle' (180°) for demo."[:18] in str(w.message) for w in rec)

    # No warning if matches preference
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        issued2 = kplot.warn_acov_preference("default", preferred="default")
        assert issued2 is False


# --- reliability axes helper ----------------------------------------------

def test_setup_axes_for_reliability_variants():
    # ax is None + bottom panel
    fig, ax, axb = kplot._setup_axes_for_reliability(None, "bottom", (6, 4))
    assert ax is not None and axb is not None
    plt.close(fig)

    # reuse existing ax, append bottom panel using axes_grid1
    fig2, ax2 = plt.subplots()
    fig3, ax3, axb3 = kplot._setup_axes_for_reliability(ax2, "bottom", None)
    assert ax3 is ax2 and axb3 is not None
    plt.close(fig2); plt.close(fig3)
