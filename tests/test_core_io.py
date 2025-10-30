
from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from kdiagram.core.io import read_data, write_data
import kdiagram.core.io as core_io


def test_read_errors_policy_validation():
    with pytest.raises(ValueError, match="errors must be one of"):
        read_data(io.StringIO("x"), format="csv", errors="boom")


def test_read_filelike_without_format_warns_and_returns_empty():
    buf = io.StringIO("a,b\n1,2\n")
    with pytest.warns(UserWarning):
        out = read_data(buf, errors="warn")  # no format → warn + empty df
    assert isinstance(out, pd.DataFrame) and out.empty

    with pytest.raises(ValueError):
        read_data(buf, errors="raise")       # no format → raise


def test_read_csv_with_postprocess(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text("g,x,y\nb,1,\na,,3\nc,4,5\n", encoding="utf-8")

    df = read_data(
        p,
        index_col="g",
        drop_na=["x", "y"],   # drop rows where x and y are both NA
        fillna={"y": 0},
        sort_index=True,
    )
    assert list(df.index) == ["a", "b", "c"]
    # a: x NA, y=3 → kept; b: x=1, y NA→ kept; c: both present → kept
    assert df.loc["a", "y"] == 3
    assert df.loc["b", "x"] == 1


def test_read_html_first_concat_list_monkeypatched(monkeypatch, tmp_path: Path):
    # Stub pd.read_html to avoid lxml/bs4 install; return two tables
    def fake_read_html(source, **kwargs):
        t1 = pd.DataFrame({"a": [1]})
        t2 = pd.DataFrame({"a": [2]})
        return [t1, t2]

    monkeypatch.setattr(pd, "read_html", fake_read_html, raising=True)

    p = tmp_path / "t.html"
    p.write_text("<table><tr><th>a</th></tr><tr><td>1</td></tr></table>",
                 encoding="utf-8")

    df_first = read_data(p, html="first")
    assert isinstance(df_first, pd.DataFrame)
    assert df_first.iloc[0, 0] == 1

    df_concat = read_data(p, html="concat")
    assert list(df_concat["a"]) == [1, 2]

    lst = read_data(p, html="list")
    assert isinstance(lst, list) and len(lst) == 2


def test_read_sql_file_with_connection_and_error(tmp_path: Path):
    sqlp = tmp_path / "q.sql"
    sqlp.write_text("SELECT 1 AS x UNION ALL SELECT 2;", encoding="utf-8")
    con = sqlite3.connect(":memory:")
    df = read_data(sqlp, sql_con=con)
    assert list(df["x"]) == [1, 2]

    with pytest.raises(ValueError, match="requires 'sql_con'"):
        read_data(sqlp, errors="raise")


def test_read_chunks_iterator(tmp_path: Path):
    p = tmp_path / "big.csv"
    pd.DataFrame({"a": range(13)}).to_csv(p, index=False)
    it = read_data(p, chunksize=5)  # -> iterator
    chunks = list(it)
    assert all(isinstance(c, pd.DataFrame) for c in chunks)
    assert [len(c) for c in chunks] == [5, 5, 3]


def test_read_unsupported_format_policies(tmp_path: Path):
    p = tmp_path / "x.foo"
    p.write_text("whatever", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported format"):
        read_data(p, errors="raise")

    with pytest.warns(UserWarning):
        df = read_data(p, errors="warn")
        assert isinstance(df, pd.DataFrame) and df.empty

    df2 = read_data(p, errors="ignore")
    assert isinstance(df2, pd.DataFrame) and df2.empty


def test_write_str_dict_rec_and_filelike(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2]})

    s = write_data(df, None, format="str")
    assert isinstance(s, str) and "a" in s

    d = write_data(df, None, format="dict")
    assert isinstance(d, dict) and d["a"] == [1, 2]

    r = write_data(df, None, format="rec")
    # writer includes index by default -> dtype names typically ('index', 'a')
    assert hasattr(r, "dtype")
    assert r.dtype.names is not None and "a" in r.dtype.names


def test_write_overwrite_and_dest_none_policy(tmp_path: Path):
    df = pd.DataFrame({"a": [1]})
    p = tmp_path / "z.csv"
    assert write_data(df, p) == p and p.exists()

    with pytest.raises(FileExistsError):
        write_data(df, p, overwrite=False)

    # dest=None for path-style format → error/warn/ignore
    with pytest.raises(ValueError, match="Destination is required"):
        write_data(df, None, format="csv", errors="raise")
    with pytest.warns(UserWarning):
        assert write_data(df, None, format="csv", errors="warn") is None
    assert write_data(df, None, format="csv", errors="ignore") is None


def test_write_unsupported_format_policies(tmp_path: Path):
    df = pd.DataFrame({"a": [1]})
    p = tmp_path / "out.foo"

    with pytest.raises(ValueError, match="Unsupported format"):
        write_data(df, p, errors="raise")

    with pytest.warns(UserWarning):
        assert write_data(df, p, errors="warn") is None

    assert write_data(df, p, errors="ignore") is None


def test_storage_options_pass_through_reader(monkeypatch, tmp_path: Path):
    # Replace handlers so we control the reader callable and inspect kwargs
    class FakeHandlers:
        def __init__(self):
            self.parsers = {".csv": self._reader}

        def _reader(self, source, **kwargs):
            # storage_options should be forwarded into kwargs if provided
            assert kwargs.get("storage_options") == {"anon": True}
            # Return a tiny DataFrame to satisfy read_data
            return pd.DataFrame({"a": [1]})

        def writers(self, df):
            return {}

    monkeypatch.setattr(core_io, "PandasDataHandlers", lambda: FakeHandlers())

    p = tmp_path / "f.csv"
    p.write_text("a\n1\n", encoding="utf-8")
    df = read_data(p, storage_options={"anon": True})
    assert list(df["a"]) == [1]


def test_storage_options_pass_through_writer(monkeypatch, tmp_path: Path):
    # Swap writers to capture kwargs; keep ext mapping minimal
    class FakeHandlers:
        def __init__(self):
            self._written = {}
        def writers(self, df):
            def _writer(dest: Path, **kwargs):
                # ensure storage_options forwarded
                assert kwargs.get("storage_options") == {"token": "X"}
                # simulate writing
                dest.write_text("ok", encoding="utf-8")
            return {".csv": _writer}

    fake = FakeHandlers()
    monkeypatch.setattr(core_io, "PandasHeaders", None, raising=False)  # guard
    monkeypatch.setattr(core_io, "PandasDataHandlers", lambda: fake)

    p = tmp_path / "w.csv"
    out = write_data(pd.DataFrame({"a": [1]}), p, storage_options={"token": "X"})
    assert out == p and p.exists()
