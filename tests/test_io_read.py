
import io, sqlite3, pandas as pd
from pathlib import Path
import pytest
from kdiagram.core.io import read_data

def test_read_csv_path_and_buffer(tmp_path: Path):
    p = tmp_path / "a.csv"; p.write_text("a,b\n1,2\n")
    df = read_data(p)
    assert df.shape == (1, 2)

    buf = io.StringIO("a,b\n3,4\n")
    df2 = read_data(buf, format="csv")  # file-like needs explicit format
    assert df2.iloc[0].tolist() == [3,4]

def test_read_compressed_csv(tmp_path: Path):
    p = tmp_path / "b.csv.gz"
    pd.DataFrame({"a":[1]}).to_csv(p, index=False, compression="gzip")
    df = read_data(p)
    assert list(df.columns) == ["a"]  # .csv.gz → .csv detected

def test_read_html_first_concat_list(monkeypatch, tmp_path: Path):
    # Stub out pandas.read_html so we don't need lxml/bs4+html5lib
    def fake_read_html(source, **kwargs):
        t1 = pd.DataFrame({"a": [1]})
        t2 = pd.DataFrame({"a": [2]})
        return [t1, t2]

    # Ensure our stub is used regardless of what’s installed
    monkeypatch.setattr(pd, "read_html", fake_read_html, raising=True)

    # Any .html path triggers the HTML branch & extension inference
    p = tmp_path / "t.html"
    p.write_text("<table><tr><th>a</th></tr><tr><td>1</td></tr></table>", encoding="utf-8")

    df_first = read_data(p, html="first")
    assert isinstance(df_first, pd.DataFrame)
    assert df_first.shape == (1, 1)
    assert df_first.iloc[0, 0] == 1

    df_concat = read_data(p, html="concat")
    assert isinstance(df_concat, pd.DataFrame)
    assert list(df_concat["a"]) == [1, 2]

    lst = read_data(p, html="list")
    assert isinstance(lst, list) and len(lst) == 2
    assert [df.shape for df in lst] == [(1, 1), (1, 1)]
    
def test_read_sql_file_with_connection(tmp_path: Path):
    # write a .sql file with a simple query
    sqlp = tmp_path / "q.sql"; sqlp.write_text("SELECT 1 AS x;", encoding="utf-8")
    con = sqlite3.connect(":memory:")
    df = read_data(sqlp, sql_con=con)
    assert df.iloc[0,0] == 1
    # missing sql_con with errors='raise' → ValueError
    with pytest.raises(ValueError):
        read_data(sqlp, errors="raise")

