# tests/test_io.py
from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

import kdiagram.utils.io as io_mod


class FakeResponse:
    """Context-manager mock for requests.get(..., stream=True)."""

    def __init__(
        self, data: bytes, status: int = 200, content_length: int | None = None
    ):
        self._data = data
        self.status_code = status
        clen = len(data) if content_length is None else content_length
        self.headers = {"content-length": str(clen)}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            # mimic requests.HTTPError without importing requests
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        # yield data in chunks
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]


def test_fancier_downloader_requires_requests(monkeypatch):
    monkeypatch.setattr(io_mod, "HAS_REQUESTS", False)
    with pytest.raises(ImportError):
        io_mod.fancier_downloader("http://x", "f.bin")


def test_fancier_downloader_invalid_error_value(monkeypatch):
    monkeypatch.setattr(io_mod, "HAS_REQUESTS", True)
    with pytest.raises(ValueError):
        io_mod.fancier_downloader("http://x", "f.bin", error="nope")


def test_fancier_downloader_fallback_without_tqdm(monkeypatch, tmp_path):
    # make tqdm import raise ImportError
    real_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm" or (name == "tqdm" and fromlist):
            raise ImportError("no tqdm")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _failing_import)

    called = {}

    def fake_download(url, filename, dstpath=None):
        called["args"] = (url, filename, dstpath)
        return "returned-path"

    monkeypatch.setattr(io_mod, "download_file", fake_download)

    with pytest.warns(UserWarning, match="tqdm is not installed"):
        out = io_mod.fancier_downloader("http://x", "f.bin", dstpath=str(tmp_path))
    assert out == "returned-path"
    assert called["args"] == ("http://x", "f.bin", str(tmp_path))


def test_fancier_downloader_success_with_progress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = b"abcdef" * 10  # 60 bytes
    monkeypatch.setattr(io_mod.requests, "get", lambda *a, **k: FakeResponse(data))

    # provide a tiny tqdm stand-in so import succeeds (if tqdm not installed)
    if "tqdm" not in sys.modules:
        fake_tqdm_mod = types.SimpleNamespace(
            tqdm=lambda **kw: types.SimpleNamespace(
                update=lambda *a, **k: None, close=lambda: None
            )
        )
        sys.modules["tqdm"] = types.SimpleNamespace(tqdm=fake_tqdm_mod.tqdm)

    out = io_mod.fancier_downloader("http://x", "ok.bin", verbose=False)
    assert out == "ok.bin"
    assert Path("ok.bin").is_file()
    assert Path("ok.bin").stat().st_size == len(data)


@pytest.mark.parametrize("mode", ["warn", "raise", "ignore"])
def test_fancier_downloader_check_size_mismatch(monkeypatch, tmp_path, mode):
    # header advertises 100 bytes, we download 50 -> mismatch (>1%)
    monkeypatch.chdir(tmp_path)
    data = b"x" * 50
    monkeypatch.setattr(
        io_mod.requests, "get", lambda *a, **k: FakeResponse(data, content_length=100)
    )

    # Provide minimal tqdm stub if missing
    if "tqdm" not in sys.modules:
        sys.modules["tqdm"] = types.SimpleNamespace(
            tqdm=lambda **kw: types.SimpleNamespace(
                update=lambda *a, **k: None, close=lambda: None
            )
        )

    kwargs = dict(
        url="http://x",
        filename="mismatch.bin",
        check_size=True,
        error=mode,
        verbose=False,
    )

    if mode == "raise":
        with pytest.raises(RuntimeError, match="Downloaded file size"):
            io_mod.fancier_downloader(**kwargs)
    elif mode == "warn":
        with pytest.warns(UserWarning, match="Downloaded file size"):
            out = io_mod.fancier_downloader(**kwargs)
            assert out == "mismatch.bin"
    else:  # ignore
        out = io_mod.fancier_downloader(**kwargs)
        assert out == "mismatch.bin"


def test_fancier_downloader_moves_to_dst(monkeypatch, tmp_path):
    data = b"hello world"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(io_mod.requests, "get", lambda *a, **k: FakeResponse(data))
    if "tqdm" not in sys.modules:
        sys.modules["tqdm"] = types.SimpleNamespace(
            tqdm=lambda **kw: types.SimpleNamespace(
                update=lambda *a, **k: None, close=lambda: None
            )
        )

    dest = tmp_path / "dest"
    out = io_mod.fancier_downloader(
        "http://x", "move.bin", dstpath=str(dest), verbose=False
    )
    # moved => returns None
    assert out is None
    assert not Path("move.bin").exists()
    assert (dest / "move.bin").is_file()


def test_fancier_downloader_move_error_handling(monkeypatch, tmp_path):
    data = b"data"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(io_mod.requests, "get", lambda *a, **k: FakeResponse(data))
    if "tqdm" not in sys.modules:
        sys.modules["tqdm"] = types.SimpleNamespace(
            tqdm=lambda **kw: types.SimpleNamespace(
                update=lambda *a, **k: None, close=lambda: None
            )
        )

    def boom(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(io_mod.os, "replace", boom)
    dest = tmp_path / "d"

    # warn path
    with pytest.warns(UserWarning, match="Failed to move"):
        out = io_mod.fancier_downloader(
            "http://x", "f.bin", dstpath=str(dest), error="warn", verbose=False
        )
        assert out is None  # function returns None after handling

    # raise path
    with pytest.raises(RuntimeError, match="Failed to move"):
        io_mod.fancier_downloader(
            "http://x", "g.bin", dstpath=str(dest), error="raise", verbose=False
        )


def test_fancier_downloader_download_exception(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def raise_get(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(io_mod.requests, "get", raise_get)

    # warn
    with pytest.warns(UserWarning, match="Failed to download"):
        out = io_mod.fancier_downloader(
            "http://x", "bad.bin", error="warn", verbose=False
        )
        assert out is None

    # ignore
    out = io_mod.fancier_downloader(
        "http://x", "bad2.bin", error="ignore", verbose=False
    )
    assert out is None

    # raise
    with pytest.raises(RuntimeError, match="Failed to download"):
        io_mod.fancier_downloader("http://x", "bad3.bin", error="raise", verbose=False)


# ---- download_file tests ----------------------------------------------------


def test_download_file_returns_filename(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    data = b"A" * 17

    monkeypatch.setattr(io_mod.requests, "get", lambda *a, **k: FakeResponse(data))

    out = io_mod.download_file("http://x", "plain.bin")
    # returns absolute path to file in CWD
    assert out == str(tmp_path / "plain.bin")
    assert Path(out).is_file()
    assert Path(out).stat().st_size == len(data)

    # ensure it prints the banners (do not assert exact, just some substring)
    captured = capsys.readouterr().out
    assert "downloading" in captured.lower()
    assert "ok" in captured.lower()


def test_download_file_with_dst_calls_move(monkeypatch, tmp_path):
    data = b"B" * 8
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(io_mod.requests, "get", lambda *a, **k: FakeResponse(data))

    called = {}

    def fake_move(src, dst):
        called["src"] = src
        called["dst"] = dst

    monkeypatch.setattr(io_mod, "move_file", fake_move)

    out = io_mod.download_file("http://x", "to-move.bin", dstpath=str(tmp_path / "dst"))
    assert out is None
    assert "src" in called and "dst" in called
    assert Path(called["src"]).name == "to-move.bin"
    assert called["dst"] == str(tmp_path / "dst")


# ---- check_file_exists tests -----------------------------------------------


def test_check_file_exists_true_false(monkeypatch):
    import importlib.resources as pkg_resources

    # monkeypatch the function used inside
    monkeypatch.setattr(pkg_resources, "is_resource", lambda *a, **k: True)
    assert io_mod.check_file_exists("any.pkg", "res.ext") is True

    monkeypatch.setattr(pkg_resources, "is_resource", lambda *a, **k: False)
    assert io_mod.check_file_exists("any.pkg", "res.ext") is False


# ---- move_file tests --------------------------------------------------------


def test_move_file_creates_directory_and_moves(tmp_path):
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    f = src_dir / "x.txt"
    f.write_text("content")

    io_mod.move_file(str(f), str(dst_dir))
    assert not f.exists()
    assert (dst_dir / "x.txt").exists()
    assert (dst_dir / "x.txt").read_text() == "content"


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
