from __future__ import annotations

import io
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kdiagram.core.io import read_data, write_data


def _df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "val": [10, 20]})


def test_write_csv_roundtrip(tmp_path: Path) -> None:
    df = _df()
    p = tmp_path / "out.csv"
    out = write_data(df, p, index=False)
    assert out == p and p.exists()
    df2 = read_data(p)
    pdt.assert_frame_equal(
        df2.astype("int64"),
        df.astype("int64"),
        check_like=True,
    )


def test_write_overwrite_policy(tmp_path: Path) -> None:
    df = _df()
    p = tmp_path / "dup.csv"
    p.write_text("id,val\n9,9\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _ = write_data(df, p, overwrite=False, errors="raise")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = write_data(
            df,
            p,
            overwrite=False,
            errors="warn",
        )
        assert out is None
        assert any("Destination exists" in str(x.message) for x in w)


def test_write_dict_returns_object(tmp_path: Path) -> None:
    df = _df()
    # dest=None → just return the dict
    d = write_data(df, None, format="dict")
    assert isinstance(d, dict) and d["id"] == [1, 2]

    # with a path → still return dict, warn about path
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d2 = write_data(df, tmp_path / "x.dict", format="dict")
        assert isinstance(d2, dict)
        assert any(".dict" in str(x.message) for x in w)


def test_write_str_to_string_and_path(tmp_path: Path) -> None:
    df = _df()
    # return string when dest=None
    s = write_data(df, None, format="str")
    assert isinstance(s, str) and "id" in s and "val" in s

    # write string to file path
    p = tmp_path / "frame.txt"
    out = write_data(df, p, format="str")
    assert out == p and p.exists()
    text = p.read_text(encoding="utf-8")
    assert "id" in text and "val" in text

    # write string to file-like
    buf = io.StringIO()
    out2 = write_data(df, buf, format="str")
    assert out2 is None and len(buf.getvalue()) > 0


def test_write_gbq_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _df()
    seen = {"called": False, "kw": None}

    def fake_to_gbq(self, **kw):
        seen["called"] = True
        seen["kw"] = kw
        return "JOB123"

    monkeypatch.setattr(pd.DataFrame, "to_gbq", fake_to_gbq)
    job = write_data(df, None, format="gbq", project_id="p")
    assert job == "JOB123"
    assert seen["called"] and seen["kw"]["project_id"] == "p"


def test_write_sql_roundtrip() -> None:
    df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    con = sqlite3.connect(":memory:")
    _ = write_data(
        df,
        None,
        format="sql",
        sql_con=con,
        sql_table="t",
        if_exists="replace",
        index=False,
    )
    out = pd.read_sql("select * from t order by id", con=con)
    pdt.assert_frame_equal(
        out.astype({"id": "int64"}),
        df.astype({"id": "int64"}),
        check_like=True,
    )


def test_write_requires_dest_for_path_based() -> None:
    df = _df()
    with pytest.raises(ValueError):
        _ = write_data(df, None, format="csv")


def test_unsupported_format_warn(tmp_path: Path) -> None:
    df = _df()
    p = tmp_path / "bad.foo"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = write_data(df, p, errors="warn")
        assert out is None
        assert any("Unsupported format" in str(x.message) for x in w)


def test_storage_options_and_index_kw_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = _df()
    p = tmp_path / "s.csv"
    seen = {"storage": None, "index": None}

    def fake_to_csv(self, path, **kw):
        seen["storage"] = kw.get("storage_options")
        seen["index"] = kw.get("index")
        Path(path).write_text("id,val\n1,10\n2,20\n", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    out = write_data(
        df,
        p,
        storage_options={"anon": True},
        index=False,
    )
    assert out == p
    assert seen["storage"] == {"anon": True}
    assert seen["index"] is False


def test_write_rec_returns_recarray(tmp_path: Path) -> None:
    df = _df()
    rec = write_data(df, None, format="rec")
    assert isinstance(rec, np.recarray)
