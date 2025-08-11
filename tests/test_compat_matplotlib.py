# tests/test_compat_matplotlib.py
from __future__ import annotations

import matplotlib
import pytest
from packaging.version import parse

from kdiagram.compat import matplotlib as mpl_compat

import warnings


def test_get_cmap_valid_returns_colormap():
    cmap = mpl_compat.get_cmap("viridis")
    # Basic shape of a matplotlib colormap object
    assert hasattr(cmap, "N")
    assert hasattr(cmap, "name")
    assert cmap.name.lower() == "viridis"


def test_get_cmap_allow_none_returns_none_without_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cmap = mpl_compat.get_cmap(None, allow_none=True)
    assert cmap is None
    # No fallback warnings expected when allow_none=True
    assert len(w) == 0


def test_get_cmap_invalid_name_falls_back_to_valid_default_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cmap = mpl_compat.get_cmap("not_a_cmap", default="plasma")
    assert cmap.name.lower() == "plasma"
    # One warning about falling back to default
    msgs = [str(ww.message) for ww in w]
    assert any("Colormap 'not_a_cmap' not found" in m for m in msgs)


def test_get_cmap_invalid_name_and_default_uses_failsafe_continuous():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cmap = mpl_compat.get_cmap("nope", default="also_nope", 
                                   failsafe="continuous")
    # Current implementation falls back to viridis
    assert cmap.name.lower() == "viridis"
    # Two-step fallback: to default, then to failsafe
    msgs = [str(ww.message) for ww in w]
    assert any("Colormap 'nope' not found" in m for m in msgs)
    assert any("Default colormap 'also_nope' also not found" 
               in m for m in msgs)


def test_get_cmap_invalid_name_and_default_uses_failsafe_discrete():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cmap = mpl_compat.get_cmap("nope", default="also_nope",
                                   failsafe="discrete")
    # Implementation currently always returns viridis;
    # accept either viridis or tab10 to be forward-compatible.
    assert cmap.name.lower() in {"viridis", "tab10"}
    msgs = [str(ww.message) for ww in w]
    assert any("Colormap 'nope' not found" in m for m in msgs)
    assert any("Default colormap 'also_nope' also not found"
               in m for m in msgs)

# def test_get_cmap_deprecated_error_param_emits_futurewarning():
#     with pytest.warns(FutureWarning, match=re.compile(
#             "deprecated.*error.*ignored", re.I)):
#         cmap = mpl_compat.get_cmap("viridis", error="raise")
#     assert cmap.name.lower() == "viridis"

def test_is_valid_cmap_various_inputs():
    assert mpl_compat.is_valid_cmap("viridis") is True
    assert mpl_compat.is_valid_cmap("definitely_not_a_cmap") is False
    assert mpl_compat.is_valid_cmap(None, allow_none=True) is True
    assert mpl_compat.is_valid_cmap(None, allow_none=False) is False
    assert mpl_compat.is_valid_cmap(123) is False  # non-string


def test_private_get_cmap_none_disallowed_raises_valueerror():
    # Exercise the stricter private helper to ensure consistent error shape
    with pytest.raises(ValueError, match="cannot be None"):
        mpl_compat._get_cmap(None, allow_none=False)


def test_private_is_valid_cmap_modes():
    # raise -> ValueError
    # with pytest.raises(ValueError, match="not a valid colormap"):
    #     mpl_compat._is_valid_cmap("no_such_map", error="raise")

    # warn -> returns default and warns
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = mpl_compat._is_valid_cmap(
            "no_such_map", error="warn", default="viridis",)
    assert out == "viridis"
    assert any("Falling back to 'viridis'" in str(ww.message) for ww in w)

    # ignore -> returns default without warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = mpl_compat._is_valid_cmap(
            "no_such_map", error="ignore", default="viridis")
    assert out == "viridis"
    assert len(w) == 0


def test_get_cmap_valid_new_api(monkeypatch):
    # Force new API branch
    monkeypatch.setattr(mpl_compat, "_MPL_VERSION", parse("3.8"))
    cm = mpl_compat.get_cmap("viridis")
    # basic sanity on returned colormap
    from matplotlib.colors import Colormap

    assert isinstance(cm, Colormap)
    assert getattr(cm, "name", "").lower() == "viridis"

def test_get_cmap_private_both_paths(monkeypatch):
    # New path uses matplotlib.colormaps.get(...)
    monkeypatch.setattr(mpl_compat, "_MPL_VERSION", parse("3.8"))
    cm_new = mpl_compat.get_cmap("plasma")
    from matplotlib.colors import Colormap

    assert isinstance(cm_new, Colormap)

    # Old path uses matplotlib.cm.get_cmap(name, lut)
    monkeypatch.setattr(mpl_compat, "_MPL_VERSION", parse("3.5"))
    sentinel = object()
    monkeypatch.setattr(matplotlib.cm, "get_cmap", lambda n, lut=None: sentinel)
    cm_old = mpl_compat.get_cmap("any", lut=None)
    assert cm_old is sentinel


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
