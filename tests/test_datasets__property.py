import shutil
from pathlib import Path

import pytest

from kdiagram.datasets import _property as prop

# ---------- helpers ----------


def _touch(p: Path, data: bytes = b"data"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


class _PathCtx:
    """Simple context manager to mimic importlib.resources.path."""

    def __init__(self, p: Path):
        self._p = p

    def __enter__(self):
        return self._p

    def __exit__(self, exc_type, exc, tb):
        return False


# ---------- get_data / remove_data ----------


def test_get_data_env_and_custom(tmp_path, monkeypatch):
    # env variable path takes precedence
    env_dir = tmp_path / "envdir"
    monkeypatch.setenv("KDIAGRAM_DATA", str(env_dir))
    got = prop.get_data()
    assert Path(got) == env_dir
    assert env_dir.exists()

    # explicit path overrides env
    custom = tmp_path / "custom"
    got2 = prop.get_data(str(custom))
    assert Path(got2) == custom
    assert custom.exists()


def test_remove_data_deletes_dir(tmp_path):
    d = tmp_path / "delme"
    # create via get_data to mimic normal behavior
    prop.get_data(str(d))
    assert d.exists()
    prop.remove_data(str(d))
    assert not d.exists()


# ---------- download_file_if : package resource path ----------


def test_download_from_package_copies_to_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    filename = "sample.txt"
    pkg_file = tmp_path / "pkg" / filename
    _touch(pkg_file, b"pkgdata")

    # Mock importlib.resources APIs used by the module
    monkeypatch.setattr(
        prop.resources, "is_resource", lambda mod, name: name == filename
    )
    monkeypatch.setattr(prop.resources, "path", lambda mod, name: _PathCtx(pkg_file))

    # Ensure downloader is never called in this path
    monkeypatch.setattr(
        prop, "fancier_downloader", lambda **k: pytest.fail("downloader should not run")
    )

    out = prop.download_file_if(
        filename, data_home=str(cache_dir), download_if_missing=False, verbose=False
    )
    assert out is not None
    out_path = Path(out)
    assert out_path.parent == cache_dir
    assert out_path.read_bytes() == b"pkgdata"


def test_download_uses_cache_if_already_present(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    filename = "already.txt"
    cached = cache_dir / filename
    _touch(cached, b"cached")

    # Pretend package also has it; function should still return cache path
    monkeypatch.setattr(prop.resources, "is_resource", lambda mod, name: True)
    monkeypatch.setattr(
        prop.resources, "path", lambda mod, name: _PathCtx(tmp_path / "other" / name)
    )

    out = prop.download_file_if(
        filename, data_home=str(cache_dir), download_if_missing=False, verbose=False
    )
    assert Path(out) == cached
    assert Path(out).read_bytes() == b"cached"


# ---------- download_file_if : forced download ----------


def test_force_download_calls_downloader_and_writes_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    filename = "force.txt"

    # No package resource
    monkeypatch.setattr(prop.resources, "is_resource", lambda mod, name: False)

    # Stub downloader: create the file in the cache dir
    def fake_downloader(url, filename, dstpath, **kw):
        Path(dstpath).mkdir(parents=True, exist_ok=True)
        (Path(dstpath) / filename).write_bytes(b"dl")

    monkeypatch.setattr(prop, "fancier_downloader", fake_downloader)

    out = prop.download_file_if(
        filename,
        data_home=str(cache_dir),
        download_if_missing=True,
        force_download=True,
        verbose=False,
    )
    assert Path(out) == cache_dir / filename
    assert Path(out).read_bytes() == b"dl"


def test_force_download_warns_when_disabled(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    filename = "force_disabled.txt"

    # No package, no cache, downloader should not be called
    monkeypatch.setattr(prop.resources, "is_resource", lambda mod, name: False)
    monkeypatch.setattr(
        prop, "fancier_downloader", lambda **k: pytest.fail("should not be called")
    )

    with pytest.warns(UserWarning):
        out = prop.download_file_if(
            filename,
            data_home=str(cache_dir),
            download_if_missing=False,
            force_download=True,
            verbose=False,
        )
    # Nothing available => returns None
    assert out is None

def test_download_if_missing_success(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    filename = "flow.txt"

    monkeypatch.setattr(prop.resources, "is_resource", lambda mod, name: False)

    def fake_downloader(url, filename, dstpath, **kw):
        Path(dstpath).mkdir(parents=True, exist_ok=True)
        (Path(dstpath) / filename).write_text("ok")

    monkeypatch.setattr(prop, "fancier_downloader", fake_downloader)

    out = prop.download_file_if(
        filename, data_home=str(cache_dir), download_if_missing=True, verbose=False
    )
    assert Path(out) == cache_dir / filename
    assert Path(out).read_text() == "ok"


def test_download_if_missing_disabled_returns_none(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    filename = "nomiss.txt"

    monkeypatch.setattr(prop.resources, "is_resource", lambda mod, name: False)
    monkeypatch.setattr(
        prop, "fancier_downloader", lambda **k: pytest.fail("should not be called")
    )

    out = prop.download_file_if(
        filename, data_home=str(cache_dir), download_if_missing=False, verbose=False
    )
    assert out is None


# ---------- error handling & metadata validation ----------


def test_invalid_error_value_raises(tmp_path):
    with pytest.raises(ValueError):
        prop.download_file_if("x.txt", error="nope", verbose=False)


def test_invalid_metadata_type_raises():
    with pytest.raises(TypeError):
        prop.download_file_if(123, verbose=False)  # not str / RemoteMetadata


def test_remote_metadata_missing_fields_raise(tmp_path):
    # Missing data_module
    meta1 = prop.RemoteMetadata(
        file="a.txt",
        url=prop.KD_REMOTE_DATA_URL,
        checksum=None,
        descr_module=None,
        data_module="",
    )
    with pytest.raises(ValueError):
        prop.download_file_if(meta1, verbose=False)

    # Missing url
    meta2 = prop.RemoteMetadata(
        file="a.txt",
        url="",
        checksum=None,
        descr_module=None,
        data_module=prop.KD_DMODULE,
    )
    with pytest.raises(ValueError):
        prop.download_file_if(meta2, verbose=False)


def test_download_returns_none_when_downloader_silent_failure(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    filename = "silent.txt"

    monkeypatch.setattr(prop.resources, "is_resource", lambda mod, name: False)

    # Downloader does NOT create the file (e.g., error='ignore' path)
    def fake_downloader(url, filename, dstpath, **kw):
        Path(dstpath).mkdir(parents=True, exist_ok=True)
        # do nothing else

    monkeypatch.setattr(prop, "fancier_downloader", fake_downloader)

    out = prop.download_file_if(
        filename,
        data_home=str(cache_dir),
        download_if_missing=True,
        verbose=False,
        error="ignore",
    )
    assert out is None


def test_package_copy_failure_falls_back_to_package_path_or_cache(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    filename = "copyfail.txt"
    pkg_file = tmp_path / "pkg2" / filename
    _touch(pkg_file, b"pkg2")

    # Mock resource present
    monkeypatch.setattr(prop.resources, "is_resource", lambda mod, name: True)
    monkeypatch.setattr(prop.resources, "path", lambda mod, name: _PathCtx(pkg_file))

    # Force copyfile to fail to hit the warning branch
    def bad_copy(src, dst):
        raise OSError("nope")

    monkeypatch.setattr(shutil, "copyfile", bad_copy)

    # Wrap the function call to catch the expected UserWarning
    with pytest.warns(UserWarning, match="Could not copy dataset"):
        out = prop.download_file_if(
            filename, data_home=str(cache_dir), download_if_missing=False,
            verbose=False
        )
    
    # Function returns cache path if exists, else package path.
    # Here cache doesn't exist, so it should return package or cache per code.
    assert out in {str(cache_dir / filename), str(pkg_file)}
    
if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
