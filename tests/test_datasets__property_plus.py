from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import kdiagram.datasets._property as dsprop
from kdiagram.datasets._property import (
    RemoteMetadata,
    download_file_if,
    get_data,
    remove_data,
)

# ----------------------------- get_data / remove_data ------------------------


def test_get_data_respects_env_and_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    env_dir = tmp_path / "kdiagram_cache_env"
    monkeypatch.setenv("KDIAGRAM_DATA", str(env_dir))
    out = get_data()  # use env var
    assert Path(out) == env_dir
    assert env_dir.exists() and env_dir.is_dir()

    # explicit path wins over env
    custom = tmp_path / "mycache" / "nested"
    out2 = get_data(str(custom))
    assert Path(out2) == custom
    assert custom.exists() and custom.is_dir()


def test_get_data_warns_on_makedirs_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Simulate OSError on os.makedirs
    def boom(path, exist_ok=False):
        raise OSError("nope")

    monkeypatch.setenv("KDIAGRAM_DATA", str(tmp_path / "will_fail"))
    monkeypatch.setattr(dsprop.os, "makedirs", boom, raising=True)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = get_data()  # will warn but still return expanded path
        assert any(
            "Could not create data directory" in str(w.message) for w in rec
        )
    assert Path(out).as_posix().endswith("will_fail")


def test_remove_data_existing_and_missing(tmp_path: Path):
    d = tmp_path / "to_remove"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")
    assert d.exists()
    remove_data(str(d))
    assert not d.exists()

    # missing path → no crash
    remove_data(str(d))  # prints message but nothing to assert


# ----------------------------- package resource helper -----------------------


def _make_pkg_with_file(
    tmp_path: Path, pkg_name: str, rel_mod: str, filename: str, content: str
) -> str:
    """
    Create a temporary importable package structure:

        <tmp>/<pkg_name>/<rel_mod>/__init__.py
        <tmp>/<pkg_name>/<rel_mod>/<filename>

    Returns the full module path (e.g., "pkg_name.rel_mod").
    """
    base = tmp_path / pkg_name
    (base / rel_mod).mkdir(parents=True, exist_ok=True)
    (base / "__init__.py").write_text("", encoding="utf-8")
    (base / rel_mod / "__init__.py").write_text("", encoding="utf-8")
    (base / rel_mod / filename).write_text(content, encoding="utf-8")
    return f"{pkg_name}.{rel_mod}"


# ----------------------------- download_file_if ------------------------------


def test_download_file_if_package_resource_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Create a fake package with a data submodule containing the file
    mod = _make_pkg_with_file(
        tmp_path, "kdtmp_pkgA", "data", "hello.txt", "hi"
    )
    monkeypatch.syspath_prepend(str(tmp_path))  # make importable

    cache = tmp_path / "cacheA"
    cache.mkdir()

    meta = RemoteMetadata(
        file="hello.txt",
        url="https://example.invalid/base/",
        checksum=None,
        descr_module=None,
        data_module=mod,  # use our temp package
    )

    got = download_file_if(
        meta, data_home=str(cache), download_if_missing=False, verbose=False
    )
    assert Path(got) == cache / "hello.txt"
    assert (cache / "hello.txt").read_text(encoding="utf-8") == "hi"


def test_download_file_if_force_download_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Mock downloader to "write" the file into dstpath
    def fake_downloader(*, url, filename, dstpath, **kwargs):
        p = Path(dstpath) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("remote", encoding="utf-8")

    monkeypatch.setattr(
        dsprop, "fancier_downloader", fake_downloader, raising=True
    )

    cache = tmp_path / "cacheB"
    meta = RemoteMetadata(
        file="remote.txt",
        url="https://example.invalid/data/",
        checksum=None,
        descr_module=None,
        data_module="kdtmp_pkgB.data",
    )
    got = download_file_if(
        meta,
        data_home=str(cache),
        download_if_missing=True,
        force_download=True,
        verbose=False,
    )
    assert Path(got) == cache / "remote.txt"
    assert (cache / "remote.txt").read_text(encoding="utf-8") == "remote"


def test_download_file_if_force_download_raises_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def failing_downloader(*, url, filename, dstpath, **kwargs):
        raise RuntimeError("net down")

    monkeypatch.setattr(
        dsprop, "fancier_downloader", failing_downloader, raising=True
    )

    cache = tmp_path / "cacheC"
    meta = RemoteMetadata(
        file="remote_fail.txt",
        url="https://example.invalid/data/",
        checksum=None,
        descr_module=None,
        data_module="kdtmp_pkgC.data",
    )
    with pytest.raises(RuntimeError, match="Forced download failed"):
        download_file_if(
            meta,
            data_home=str(cache),
            download_if_missing=True,
            force_download=True,
            error="raise",
            verbose=False,
        )


def test_download_file_if_download_missing_success_with_string_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """String metadata path uses module constants; ensure we mock them + downloader."""
    # Point constants to our temp package AND benign URL
    mod = _make_pkg_with_file(tmp_path, "kdtmp_pkgD", "data", "__keep__", "")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(dsprop, "KD_DMODULE", mod, raising=True)
    monkeypatch.setattr(
        dsprop,
        "KD_REMOTE_DATA_URL",
        "https://example.invalid/repo/",
        raising=True,
    )

    def fake_downloader(*, url, filename, dstpath, **kwargs):
        p = Path(dstpath) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("downloaded", encoding="utf-8")

    monkeypatch.setattr(
        dsprop, "fancier_downloader", fake_downloader, raising=True
    )

    cache = tmp_path / "cacheD"
    got = download_file_if(
        "toy.csv",
        data_home=str(cache),
        download_if_missing=True,
        verbose=False,
    )
    assert Path(got) == cache / "toy.csv"
    assert (cache / "toy.csv").read_text(encoding="utf-8") == "downloaded"


def test_download_file_if_cache_preferred_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Create both a package resource and a cache copy; function should return cache path.
    mod = _make_pkg_with_file(
        tmp_path, "kdtmp_pkgE", "data", "dup.txt", "pkg"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    cache = tmp_path / "cacheE"
    cache.mkdir()
    (cache / "dup.txt").write_text("cache", encoding="utf-8")

    meta = RemoteMetadata(
        file="dup.txt",
        url="https://x/",
        checksum=None,
        descr_module=None,
        data_module=mod,
    )
    got = download_file_if(
        meta, data_home=str(cache), download_if_missing=False, verbose=False
    )
    assert Path(got) == cache / "dup.txt"
    assert (cache / "dup.txt").read_text(encoding="utf-8") == "cache"


def test_download_file_if_invalid_params_and_types(tmp_path: Path):
    with pytest.raises(ValueError):
        download_file_if(
            "x",
            data_home=str(tmp_path),
            download_if_missing=True,
            error="boom",
        )

    with pytest.raises(TypeError):
        download_file_if(123, data_home=str(tmp_path))
