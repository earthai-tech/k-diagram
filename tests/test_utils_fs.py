
import os
import warnings
import pytest
import matplotlib

# Use a non-interactive backend for tests
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from kdiagram.utils.fs import savefig as safe_savefig  # noqa: E402


def test_creates_parents_and_unique_naming(tmp_path):
    fig, ax = plt.subplots()

    target = tmp_path / "deep" / "nested" / "plot.png"

    # 1st save → creates folders and 'plot.png'
    p1 = safe_savefig(target, fig, dpi=72, overwrite=False)
    assert p1 is not None and p1.exists()
    assert p1.name == "plot.png"

    # 2nd save to same target → 'plot (1).png'
    p2 = safe_savefig(target, fig, dpi=72, overwrite=False)
    assert p2.exists() and p2.name == "plot (1).png"

    # 3rd save passing an Axes → 'plot (2).png'
    p3 = safe_savefig(target, ax, dpi=72, overwrite=False)
    assert p3.exists() and p3.name == "plot (2).png"


def test_directory_hint_and_default_extension(tmp_path):
    fig, _ = plt.subplots()

    # Trailing separator → treated as directory with default 'figure.png'
    dir_hint = str(tmp_path / "as_dir") + os.sep
    p_dir = safe_savefig(dir_hint, fig, dpi=72)
    assert p_dir.exists()
    assert p_dir.parent.name == "as_dir"
    assert p_dir.name == "figure.png"

    # No extension provided → defaults to .png
    noext = tmp_path / "my_figure"
    p_noext = safe_savefig(noext, fig, dpi=72)
    assert p_noext.exists()
    assert p_noext.suffix == ".png"
    assert p_noext.name == "my_figure.png"


def test_error_handling_warn_and_raise(tmp_path, monkeypatch):
    fig, _ = plt.subplots()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    # Warn mode: returns None and emits a warning
    monkeypatch.setattr(fig, "savefig", boom, raising=True)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        out = safe_savefig(tmp_path / "x.png", fig, error="warn")
        assert out is None
        assert any("Failed to save figure" in str(w.message) for w in ws)

    # Raise mode: propagates the exception
    monkeypatch.setattr(fig, "savefig", boom, raising=True)
    with pytest.raises(RuntimeError):
        safe_savefig(tmp_path / "y.png", fig, error="raise")


def test_savefig_none_is_noop():
    fig, _ = plt.subplots()
    out = safe_savefig(None, fig)
    assert out is None

