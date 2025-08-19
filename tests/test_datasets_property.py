import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import kdiagram.datasets._property as prop_mod
from kdiagram.datasets._property import (
    RemoteMetadata,
    download_file_if,
    get_data,
    remove_data,
)


# --- helper: fake downloader that "creates" the file in cache
def fake_dl(url, filename, dstpath, **kwargs):
    dstpath = Path(dstpath)  # <- IMPORTANT: your code passes str
    dstpath.mkdir(parents=True, exist_ok=True)
    (dstpath / filename).write_bytes(b"ok")


def test_get_data_respects_env_and_creates(monkeypatch, tmp_path):
    target = tmp_path / "kd_data_env"
    monkeypatch.setenv("KDIAGRAM_DATA", str(target))
    path = get_data()
    # get_data returns a string path
    assert path == str(target)
    assert target.exists() and target.is_dir()


def test_get_data_custom_path(tmp_path):
    custom = tmp_path / "my_custom_cache"
    path = get_data(str(custom))
    assert path == str(custom)
    assert custom.exists() and custom.is_dir()


def test_remove_data_deletes(tmp_path, capsys):
    d = tmp_path / "to_delete"
    d.mkdir()
    (d / "touch.txt").write_text("x")
    remove_data(str(d))
    assert not d.exists()


def test_download_file_if_creates_file_in_cache(monkeypatch, tmp_path):
    # point data cache to our temp dir
    monkeypatch.setenv("KDIAGRAM_DATA", str(tmp_path))

    # patch the downloader WHERE IT'S LOOKED UP
    monkeypatch.setattr(prop_mod, "fancier_downloader", fake_dl)

    # use simple string metadata (filename only; module fills defaults)
    filename = "toy.csv"

    # make sure the cache is empty
    cache_file = tmp_path / filename
    assert not cache_file.exists()

    # exercise
    out = prop_mod.download_file_if(
        filename, download_if_missing=True, verbose=False
    )

    # assert file is "downloaded"
    assert out == str(cache_file)
    assert cache_file.exists() and cache_file.read_bytes() == b"ok"


def _write_temp_file(dirpath, name, content=b"OK"):
    p = dirpath / name
    p.write_bytes(content)
    return p


@contextmanager
def _fake_resources_path(_package, name, file_path):
    # Mimic importlib.resources.path(...)
    yield file_path


def test_download_from_package_resources_copies_to_cache(
    monkeypatch, tmp_path
):
    """
    Simulate a file shipped inside the package resources and ensure it is
    copied into the user cache dir and returned.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create a "package" file somewhere, we will yield its path via fake resources.path
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    pkg_file = _write_temp_file(pkg_dir, "data.txt", b"FROM_PKG")

    # Monkeypatch importlib.resources API used by the module
    monkeypatch.setattr(
        prop_mod.resources, "is_resource", lambda pkg, nm: nm == "data.txt"
    )
    monkeypatch.setattr(
        prop_mod.resources,
        "path",
        lambda pkg, nm: _fake_resources_path(pkg, nm, pkg_file),
    )

    # Ensure module constants are reasonable
    monkeypatch.setattr(prop_mod, "KD_DMODULE", "kdiagram.datasets.data")
    monkeypatch.setattr(
        prop_mod, "KD_REMOTE_DATA_URL", "https://example.com/datasets/"
    )

    out = download_file_if(
        "data.txt",
        data_home=str(cache_dir),
        download_if_missing=True,
        verbose=False,
    )
    assert out is not None
    assert os.path.isfile(out)
    # copied into cache
    assert os.path.dirname(out) == str(cache_dir)
    assert open(out, "rb").read() == b"FROM_PKG"


def test_download_when_missing_calls_downloader(monkeypatch, tmp_path):
    """
    If not in resources and not in cache, we should call the downloader
    which writes to cache. No network—stub the downloader to create the file.
    """
    cache_dir = tmp_path / "cache2"
    cache_dir.mkdir()
    fname = "remote.bin"

    # Not present in package resources
    monkeypatch.setattr(
        prop_mod.resources, "is_resource", lambda *a, **k: False
    )

    created = {}

    def fake_downloader(url, filename, dstpath, **kwargs):
        # Create the file in dstpath with the given filename
        p = os.path.join(dstpath, filename)
        with open(p, "wb") as f:
            f.write(b"DOWNLOADED")
        created["p"] = p

    monkeypatch.setattr(prop_mod, "fancier_downloader", fake_downloader)
    monkeypatch.setattr(prop_mod, "KD_DMODULE", "kdiagram.datasets.data")
    monkeypatch.setattr(
        prop_mod, "KD_REMOTE_DATA_URL", "https://example.com/base/"
    )

    out = download_file_if(
        fname,
        data_home=str(cache_dir),
        download_if_missing=True,
        verbose=False,
    )
    assert out == created["p"]
    assert os.path.isfile(out)
    assert open(out, "rb").read() == b"DOWNLOADED"


def test_force_download_overrides(monkeypatch, tmp_path):
    """
    When force_download=True, downloader is invoked regardless of resources.
    """
    cache_dir = tmp_path / "cache3"
    cache_dir.mkdir()
    fname = "forced.txt"

    # Pretend resource exists, but we still want forced download
    monkeypatch.setattr(
        prop_mod.resources, "is_resource", lambda *a, **k: True
    )
    monkeypatch.setattr(
        prop_mod.resources,
        "path",
        lambda pkg, nm: _fake_resources_path(pkg, nm, tmp_path / "ignored"),
    )

    wrote = {}

    def fake_downloader(url, filename, dstpath, **kwargs):
        p = os.path.join(dstpath, filename)
        with open(p, "wb") as f:
            f.write(b"FORCED")
        wrote["p"] = p

    monkeypatch.setattr(prop_mod, "fancier_downloader", fake_downloader)
    monkeypatch.setattr(prop_mod, "KD_DMODULE", "kdiagram.datasets.data")
    monkeypatch.setattr(
        prop_mod, "KD_REMOTE_DATA_URL", "https://example.com/base/"
    )

    out = download_file_if(
        fname,
        data_home=str(cache_dir),
        download_if_missing=True,
        force_download=True,
        verbose=False,
    )
    assert out == wrote["p"]
    assert open(out, "rb").read() == b"FORCED"


def test_download_file_if_type_errors_and_value_errors(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache4"
    cache_dir.mkdir()

    # Bad error policy
    with pytest.raises(ValueError):
        download_file_if("x.bin", data_home=str(cache_dir), error="meh")

    # Bad metadata type
    with pytest.raises(TypeError):
        download_file_if(
            12345, data_home=str(cache_dir)
        )  # not str / RemoteMetadata

    # Missing fields in RemoteMetadata
    bad_meta = RemoteMetadata(
        file="x.bin", url="", checksum=None, descr_module=None, data_module=""
    )
    with pytest.raises(ValueError):
        download_file_if(bad_meta, data_home=str(cache_dir))


def test_download_file_if_not_found_and_not_allowed(monkeypatch, tmp_path):
    """
    When not in resources/cache and download_if_missing=False, return None.
    """
    cache_dir = tmp_path / "cache5"
    cache_dir.mkdir()

    monkeypatch.setattr(
        prop_mod.resources, "is_resource", lambda *a, **k: False
    )
    monkeypatch.setattr(prop_mod, "KD_DMODULE", "kdiagram.datasets.data")
    monkeypatch.setattr(
        prop_mod, "KD_REMOTE_DATA_URL", "https://example.com/base/"
    )

    out = download_file_if(
        "nope.txt",
        data_home=str(cache_dir),
        download_if_missing=False,
        verbose=False,
    )
    assert out is None
