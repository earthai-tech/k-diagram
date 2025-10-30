
import numpy as np, pandas as pd, pytest
from pathlib import Path
from kdiagram.core.io import write_data

def test_write_csv_and_json(tmp_path: Path):
    df = pd.DataFrame({"a":[1,2]})
    p_csv = tmp_path / "o.csv"
    p_json = tmp_path / "o.json"
    assert write_data(df, p_csv) == p_csv and p_csv.exists()
    assert write_data(df, p_json) == p_json and p_json.exists()

def test_write_special_str_dict_rec(tmp_path: Path):
    df = pd.DataFrame({"a":[1,2]})
    s = write_data(df, None, format="str")
    assert isinstance(s, str) and "a" in s  
    d = write_data(df, None, format="dict")
    assert isinstance(d, dict) and "a" in d  
    r = write_data(df, None, format="rec")
    assert isinstance(r, np.recarray)        

def test_write_overwrite_policy(tmp_path: Path):
    df = pd.DataFrame({"a":[1]})
    p = tmp_path / "z.csv"
    write_data(df, p)
    with pytest.raises(FileExistsError):
        write_data(df, p, overwrite=False)  
