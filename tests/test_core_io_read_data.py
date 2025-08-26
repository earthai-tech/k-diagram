from __future__ import annotations

import gzip
import io
import sqlite3
import warnings
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from kdiagram.core.io import read_data


def _csv_write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_csv_basic_read(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    _csv_write(p, "id,val\n1,10\n2,20\n")
    df = read_data(p)
    exp = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    # dtype may differ; compare by values
    pdt.assert_frame_equal(
        df.reset_index(drop=True).astype("int64"),
        exp.astype("int64"),
    )


def test_csv_gz_infer_ext(tmp_path: Path) -> None:
    p = tmp_path / "b.csv.gz"
    raw = b"id,val\n1,10\n2,20\n"
    with gzip.open(p, "wb") as f:
        f.write(raw)
    df = read_data(p)
    assert list(df.columns) == ["id", "val"]
    assert df.shape == (2, 2)


def test_filelike_requires_format_raises() -> None:
    buf = io.StringIO("id,val\n1,2\n")
    with pytest.raises(ValueError):
        _ = read_data(buf)


def test_filelike_with_format_csv() -> None:
    buf = io.StringIO("id,val\n1,2\n")
    df = read_data(buf, format="csv")
    assert df.loc[0, "val"] == 2


def test_html_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    t1 = pd.DataFrame({"a": [1], "b": [2]})
    t2 = pd.DataFrame({"a": [3], "b": [4]})

    def fake_read_html(_src, **_kw):
        return [t1, t2]

    monkeypatch.setattr(pd, "read_html", fake_read_html)

    df_first = read_data("page.html", html="first")
    pdt.assert_frame_equal(df_first, t1)

    df_concat = read_data("page.html", html="concat")
    assert df_concat.shape == (2, 2)

    lst = read_data("page.html", html="list")
    assert isinstance(lst, list) and len(lst) == 2


def test_sql_reading(tmp_path: Path) -> None:
    # create small sqlite db in memory
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute("create table t(id integer, v text)")
    cur.executemany("insert into t values(?,?)", [(1, "a"), (2, "b")])
    con.commit()

    q = "select * from t order by id"
    sql_file = tmp_path / "q.sql"
    sql_file.write_text(q, encoding="utf-8")

    df = read_data(sql_file, sql_con=con)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["id", "v"]


def test_post_process_fillna_drop_index_sort(
    tmp_path: Path,
) -> None:
    p = tmp_path / "c.csv"
    _csv_write(p, "id,x,y\n2,,3\n1,5,\n")
    df = read_data(
        p,
        fillna={"x": 0, "y": 0},
        index_col="id",
        sort_index=True,
    )
    assert list(df.index) == [1, 2]
    assert df.loc[1, "y"] == 0
    assert df.loc[2, "x"] == 0


def test_dropna_subset(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    _csv_write(p, "id,x,y\n1,,3\n2,5,\n3,,\n")
    df = read_data(p, drop_na=["x", "y"])
    # row 3 removed
    assert df.shape[0] == 2


def test_errors_policy_warn(tmp_path: Path) -> None:
    p = tmp_path / "file.foo"
    p.write_text("dummy", encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = read_data(p, errors="warn")
        assert df.empty
        assert any("Unsupported format" in str(x.message) for x in w)


def test_chunked_csv_iterator(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    _csv_write(p, "id\n1\n2\n3\n")
    it = read_data(p, format="csv", chunksize=2)
    assert hasattr(it, "__iter__")
    chunks = list(it)  # type: ignore[arg-type]
    assert isinstance(chunks[0], pd.DataFrame)
    sizes = [c.shape[0] for c in chunks]
    assert sizes == [2, 1]


def test_storage_options_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    p = tmp_path / "f.csv"
    _csv_write(p, "id\n1\n")

    seen: dict[str, object] = {}

    # keep original before patching
    orig_read_csv = pd.read_csv

    def fake_read_csv(path, **kw):
        seen["storage_options"] = kw.get("storage_options")
        kw2 = {k: v for k, v in kw.items() if k != "storage_options"}
        # call the original to avoid recursion
        return orig_read_csv(path, **kw2)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)

    df = read_data(
        p,
        storage_options={"anon": True},
        format="csv",
    )
    assert df.shape == (1, 1)
    assert seen["storage_options"] == {"anon": True}
