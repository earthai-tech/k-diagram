# tests/test_datasets_property_more.py
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

import kdiagram.datasets._property as dsprop
from kdiagram.datasets._property import (
    RemoteMetadata,
    download_file_if,
)


def test_download_file_if_bad_error_value(tmp_path: Path):
    with pytest.raises(ValueError):
        download_file_if("x.csv", data_home=str(tmp_path), error="boom")


def test_remote_metadata_missing_fields_raises(tmp_path: Path):
    # missing data_module
    meta1 = RemoteMetadata(
        file="x.csv",
        url="https://x/",
        checksum=None,
        descr_module=None,
        data_module=None,
    )
    with pytest.raises(ValueError, match="must include 'data_module'"):
        download_file_if(meta1, data_home=str(tmp_path))

    # missing url
    meta2 = RemoteMetadata(
        file="x.csv",
        url=None,
        checksum=None,
        descr_module=None,
        data_module="a.b",
    )
    with pytest.raises(ValueError, match="must include 'url'"):
        download_file_if(meta2, data_home=str(tmp_path))


def test_force_download_but_download_disabled_warns(
    tmp_path: Path, monkeypatch
):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        # bogus module to ensure we don't find package resources
        meta = RemoteMetadata("x.txt", "https://x/", None, None, "not.a.pkg")
        got = download_file_if(
            meta,
            data_home=str(tmp_path),
            download_if_missing=False,
            force_download=True,
            error="warn",
            verbose=False,
        )
        assert got is None
        assert any("Cannot force download" in str(w.message) for w in rec)


def test_package_resource_lookup_warns_when_module_missing(
    tmp_path: Path, monkeypatch
):
    # Use non-existent package; download disabled so we stop early
    meta = RemoteMetadata("x.txt", "https://x/", None, None, "not.a.pkg")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = download_file_if(
            meta,
            data_home=str(tmp_path),
            download_if_missing=False,
            error="warn",
            verbose=True,
        )
        assert got is None
        # A resource warning message should appear
        assert any(
            "Could not check package resources" in str(w.message) for w in rec
        )


def _make_pkg_with_file(
    tmp_path: Path, pkg: str, sub: str, filename: str, content: str
) -> str:
    base = tmp_path / pkg
    (base / sub).mkdir(parents=True, exist_ok=True)
    (base / "__init__.py").write_text("", encoding="utf-8")
    (base / sub / "__init__.py").write_text("", encoding="utf-8")
    (base / sub / filename).write_text(content, encoding="utf-8")
    return f"{pkg}.{sub}"


def test_package_to_cache_copy_failure_falls_back(
    monkeypatch, tmp_path: Path
):
    mod = _make_pkg_with_file(
        tmp_path, "kdtmp_pkg_prop", "data", "hi.txt", "hello"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    # Monkeypatch copyfile to fail to hit the warning + fallback branch
    def boom_copy(src, dst, *, follow_symlinks=True):
        raise OSError("disk full")

    cache = tmp_path / "cache"
    meta = RemoteMetadata("hi.txt", "https://example/", None, None, mod)

    with pytest.warns(UserWarning, match="Could not copy dataset"):
        monkeypatch.setattr(
            dsprop.shutil, "copyfile", boom_copy, raising=True
        )
        out = download_file_if(
            meta,
            data_home=str(cache),
            download_if_missing=False,
            verbose=False,
        )
        # Since cache file doesn't exist, function returns the package path
        assert out.endswith(os.path.join("kdtmp_pkg_prop", "data", "hi.txt"))


def test_not_found_and_download_disabled_returns_none(tmp_path: Path):
    meta = RemoteMetadata(
        "nope.csv", "https://example/", None, None, "not.pkg"
    )
    got = download_file_if(
        meta,
        data_home=str(tmp_path),
        download_if_missing=False,
        error="ignore",
        verbose=False,
    )
    assert got is None


def test_string_metadata_defaults_missing_warn_and_raise(
    tmp_path: Path, monkeypatch
):
    # Blank defaults → warn + None
    monkeypatch.setattr(dsprop, "KD_REMOTE_DATA_URL", "", raising=True)
    monkeypatch.setattr(dsprop, "KD_DMODULE", "", raising=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = download_file_if("a.csv", data_home=str(tmp_path), error="warn")
        assert got is None
        assert any(
            "Default remote URL or data module path not configured"
            in str(w.message)
            for w in rec
        )

    # Raise path
    with pytest.raises(ValueError):
        download_file_if("b.csv", data_home=str(tmp_path), error="raise")


def test_download_failure_warns_and_returns_none(tmp_path: Path, monkeypatch):
    # Valid defaults: inject a fake downloader that raises
    monkeypatch.setattr(
        dsprop, "KD_REMOTE_DATA_URL", "https://x/", raising=True
    )
    monkeypatch.setattr(dsprop, "KD_DMODULE", "not.a.pkg", raising=True)

    def failing_downloader(**kwargs):
        raise RuntimeError("net down")

    monkeypatch.setattr(
        dsprop, "fancier_downloader", failing_downloader, raising=True
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = download_file_if(
            "c.csv",
            data_home=str(tmp_path),
            download_if_missing=True,
            error="warn",
            verbose=False,
        )
        assert got is None
        assert any(
            "Failed to download 'c.csv'" in str(w.message) for w in rec
        )
