
import io
import pandas as pd
from pathlib import Path
from kdiagram.core._io_utils import _normalize_ext, _get_valid_kwargs, _post_process

def test_normalize_ext_explicit_and_compressed(tmp_path: Path):
    p = tmp_path / "d.csv.gz"
    p.write_text("a,b\n1,2\n")
    assert _normalize_ext(p) == ".csv"   # compressed suffix handling
    assert _normalize_ext(p, explicit="json") == ".json"
    assert _normalize_ext(io.StringIO("x")) is None  # file-like → None
    assert _normalize_ext("noext") is None

def test_get_valid_kwargs_filters():
    def f(a, b): return a + b
    kw = {"a": 1, "b": 2, "c": 3}
    assert _get_valid_kwargs(f, kw) == {"a": 1, "b": 2}  # filters extras

def test_post_process_fill_drop_index_sort():
    df = pd.DataFrame({"k":[2,1], "x":[1,None], "y":[None,3]})
    out = _post_process(df, index_col="k", sort_index=True,
                        drop_na=["x","y"], fillna={"y":0})
    assert list(out.index) == [1, 2]                     # sorted by index
    assert out.loc[1, "y"] == 3 and out.loc[2, "y"] == 0 # fillna/subset drop
