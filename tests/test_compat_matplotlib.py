# tests/test_compat_matplotlib.py
from __future__ import annotations

import matplotlib
import pytest
from packaging.version import parse

from kdiagram.compat import matplotlib as mpl_compat


def test_get_cmap_valid_new_api(monkeypatch):
    # Force new API branch
    monkeypatch.setattr(mpl_compat, "_MPL_VERSION", parse("3.8"))
    cm = mpl_compat.get_cmap("viridis")
    # basic sanity on returned colormap
    from matplotlib.colors import Colormap

    assert isinstance(cm, Colormap)
    assert getattr(cm, "name", "").lower() == "viridis"


def test_get_cmap_old_api_calls_cm_get_cmap(monkeypatch):
    # Force old API path (<3.6) and stub cm.get_cmap
    monkeypatch.setattr(mpl_compat, "_MPL_VERSION", parse("3.5"))
    sentinel = object()

    def fake_get_cmap(name, lut=None):
        # Ensure arguments are relayed
        assert name == "magma"
        assert lut == 16
        return sentinel

    monkeypatch.setattr(matplotlib.cm, "get_cmap", fake_get_cmap)
    out = mpl_compat.get_cmap("magma", lut=16)
    assert out is sentinel


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


# def test_is_valid_cmap_new_registry(monkeypatch):
#     monkeypatch.setattr(mpl_compat, "_MPL_VERSION", parse("3.9"))
#     # Valid returns unchanged
#     assert mpl_compat.is_valid_cmap("viridis") == "viridis"

#     # Invalid: raise
#     with pytest.raises(ValueError):
#         mpl_compat.is_valid_cmap("definitely_not_a_cmap")

#     # Invalid: warn -> returns default
#     with pytest.warns(UserWarning, match="Invalid `cmap` name"):
#         out = mpl_compat.is_valid_cmap(
#             "definitely_not_a_cmap", default="magma", error="warn"
#         )
#     assert out == "magma"

#     # Invalid: ignore -> returns default silently
#     out = mpl_compat.is_valid_cmap("nope", default="inferno", error="ignore")
#     assert out == "inferno"


def test_is_valid_cmap_old_registry(monkeypatch):
    # Force old path and fake the old registry with exactly two names
    monkeypatch.setattr(mpl_compat, "_MPL_VERSION", parse("3.5"))
    fake_registry = {"oldViridis": object(), "oldMagma": object()}
    monkeypatch.setattr(matplotlib.cm, "cmap_d", fake_registry, raising=False)

    with pytest.raises(ValueError):
        mpl_compat.is_valid_cmap("notThere")

    with pytest.warns(UserWarning):
        out = mpl_compat.is_valid_cmap("notThere", default="oldMagma", error="warn")
    assert out == "oldMagma"

    out = mpl_compat.is_valid_cmap("stillNotThere", default="oldMagma", error="ignore")
    assert out == "oldMagma"


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
