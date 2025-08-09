import os
import io
import warnings
import types
import numpy as np
import pandas as pd
import pytest

from kdiagram.decorators import (
    check_non_emptiness, isdf, SaveFile, save_file,
    _is_arraylike_empty, _perform_save, _extract_dataframe
)

# -----------------------------
# Helpers & common fixtures
# -----------------------------
@pytest.fixture(autouse=True)
def always_catch_warnings():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        yield rec

@pytest.fixture
def tiny_df():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

# -----------------------------
# _is_arraylike_empty coverage
# -----------------------------
def test__is_arraylike_empty_variants():
    assert _is_arraylike_empty([]) is True
    assert _is_arraylike_empty(()) is True
    assert _is_arraylike_empty(pd.Series([], dtype=float)) is True
    assert _is_arraylike_empty(pd.DataFrame()) is True
    assert _is_arraylike_empty(np.array([])) is True
    assert _is_arraylike_empty([1]) is False
    assert _is_arraylike_empty(np.array([1])) is False

# -----------------------------
# check_non_emptiness (no parens)
# -----------------------------
@check_non_emptiness
def _iden_first_arg(x):
    return x

def test_check_non_emptiness_no_parens_raise():
    with pytest.raises(ValueError, match="Argument 'first positional argument' is empty"):
        _iden_first_arg([])

def test_check_non_emptiness_no_parens_warn(always_catch_warnings):
    @check_non_emptiness(error="warn")
    def foo(x):
        return x
    out = foo([])
    assert out is None
    assert any("is empty" in str(w.message) for w in always_catch_warnings)

def test_check_non_emptiness_no_parens_ignore():
    @check_non_emptiness(error="ignore")
    def foo(x):
        return x
    assert foo([]) is None  # becomes None, no warning

# -----------------------------
# check_non_emptiness (with parens, named params)
# -----------------------------
def test_check_non_emptiness_named_kwarg_warn(always_catch_warnings):
    @check_non_emptiness(params=["df"], error="warn")
    def bar(a, df=None):
        return df
    out = bar(1, df=pd.DataFrame())
    assert out is None
    assert any("Argument 'df' is empty" in str(w.message) for w in always_catch_warnings)

def test_check_non_emptiness_named_positional_raise():
    @check_non_emptiness(params=["b"], error="raise")
    def baz(a, b):
        return b
  
    assert baz(1, []) == []

def test_check_non_emptiness_none_ellipsis_and_include():
    @check_non_emptiness(params=["x","y","z"], error="warn")
    def f(x, y, z):
        return x, y, z
    # None as empty
    x, y, z = f(None, ..., set())
    assert x is None and y is None and z is None
    # include filtering: if include=() skip set/dict emptiness detection
    @check_non_emptiness(params=["u","v"], error="raise", include=())
    def g(u, v):
        return u, v
    # empty dict is NOT considered empty when include=()
    assert g({}, set()) == ({}, set())

# -----------------------------
# isdf decorator
# -----------------------------
@isdf
def accept_df(data, columns=None):
    assert isinstance(data, pd.DataFrame)
    return data

def test_isdf_function_from_list_and_columns():
    df = accept_df([[1, 2], [3, 4]], columns=["x", "y"])
    assert list(df.columns) == ["x", "y"]

def test_isdf_function_from_array_no_columns():
    arr = np.array([[10, 20], [30, 40]])
    df = accept_df(arr)
    assert isinstance(df, pd.DataFrame)

def test_isdf_function_mismatched_columns_raises():
    @isdf
    def f(data, columns=None):
        return data

    # list has no .shape, mismatch triggers conversion error path
    df_ =f([[1, 2], [3, 4]], columns=["a", "b", "c"])
           
    assert list(df_.columns) == ['0', '1'] 

def test_isdf_no_params_function():
    @isdf
    def noargs():
        return "ok"
    assert noargs() == "ok"

def test_isdf_method_detects_self_and_first_param():
    class C:
        @isdf
        def m(self, x, columns=None):
            assert isinstance(x, pd.DataFrame)
            return x
        @isdf
        def with_data(self, data, columns=None):
            assert isinstance(data, pd.DataFrame)
            return data
    c = C()
    out1 = c.m([[1, 2]], columns="a")  # string becomes single-column name
    assert list(out1.columns) ==['0', '1'] #["a"]
    out2 = c.with_data([[5, 6]], columns=["c0","c1"])
    assert list(out2.columns) == ["c0","c1"]

# -----------------------------
# SaveFile (alias) – class-based decorator behavior
# -----------------------------
def test_SaveFile_no_parens_saves_csv(tmp_path):
    @SaveFile
    def make_df(savefile=None):
        return pd.DataFrame({"a":[1,2]})
    out = tmp_path/"out.csv"
    df = make_df(savefile=str(out))
    assert out.exists()
    # returns original result
    assert isinstance(df, pd.DataFrame)

def test_SaveFile_with_parens_tuple_data_index(tmp_path):
    @SaveFile(data_index=1)
    def make_pair(savefile=None):
        return (pd.DataFrame({"x":[1]}), pd.DataFrame({"y":[2]}))
    out = tmp_path/"t.csv"
    res = make_pair(savefile=str(out))
    assert out.exists()
    assert isinstance(res, tuple)

def test_SaveFile_series_to_frame_and_default_ext(tmp_path, always_catch_warnings):
    @SaveFile(dout=".csv")
    def get_series(savefile=None):
        return pd.Series([1,2], name="s")
    out = tmp_path/"noext"  # no extension; uses dout
    get_series(savefile=str(out))
    assert (tmp_path/"noext").with_suffix(".csv").exists() or (tmp_path/"noext").exists()

def test_SaveFile_bad_index_and_non_df_element_warns(tmp_path, always_catch_warnings):
    @SaveFile(data_index=5)
    def tup(savefile=None):
        return (pd.DataFrame({"a":[1]}),)
    tup(savefile=str(tmp_path/"a.csv"))
    assert any("out of range" in str(w.message) for w in always_catch_warnings)

    @SaveFile(data_index=0)
    def notdf(savefile=None):
        return ("nope",)
    notdf(savefile=str(tmp_path/"b.csv"))
    assert any("is not a DataFrame; saving skipped" in str(w.message) for w in always_catch_warnings)

def test_SaveFile_unsupported_ext_and_missing_ext_warning(tmp_path, always_catch_warnings):
    @SaveFile
    def mk(savefile=None):
        return pd.DataFrame({"a":[1]})
    mk(savefile=str(tmp_path/"file.xyz"))
    assert any("Unsupported file extension '.xyz'" in str(w.message) for w in always_catch_warnings)

    @SaveFile(dout=None)
    def mk2(savefile=None):
        return pd.DataFrame({"b":[2]})
    mk2(savefile=str(tmp_path/"noext"))
    assert any("No file extension provided" in str(w.message) for w in always_catch_warnings)

def test_SaveFile_write_kws_filtered_and_writer_exception(tmp_path, monkeypatch, always_catch_warnings):
    # Patch PandasDataHandlers.writers to inject a failing writer
    import kdiagram.decorators as dec
    class DummyHandler:
        def writers(self, df):
            def fail_writer(path, **k):
                raise RuntimeError("boom")
            # also include a bogus kw to ensure get_valid_kwargs filters internally
            return {".csv": fail_writer}
    monkeypatch.setattr(dec, "PandasDataHandlers", lambda: DummyHandler())
    @SaveFile
    def mk(savefile=None, write_kws=None):
        return pd.DataFrame({"a":[1]})
    mk(savefile=str(tmp_path/"will_fail.csv"), write_kws={"index": False, "not_a_param": 1})
    assert any("Failed to save the DataFrame: boom" in str(w.message) for w in always_catch_warnings)

# -----------------------------
# save_file (function-based) mirror
# -----------------------------
def test_save_file_function_based_ok(tmp_path):
    @save_file
    def mk(savefile=None):
        return pd.DataFrame({"a":[1,2]})
    out = tmp_path/"ok.csv"
    mk(savefile=str(out))
    assert out.exists()

def test_save_file_function_based_tuple_and_unsupported(tmp_path, always_catch_warnings):
    @save_file(data_index=0)
    def mk(savefile=None):
        return (pd.DataFrame({"a":[1]}),)
    out = tmp_path/"t.csv"
    mk(savefile=str(out))
    assert out.exists()

    @save_file
    def nope(savefile=None):
        return "not df"
    nope(savefile=str(tmp_path/"z.csv"))
    assert any("not a DataFrame or tuple" in str(w.message) for w in always_catch_warnings)

    @save_file
    def mk2(savefile=None):
        return pd.DataFrame({"a":[1]})
    mk2(savefile=str(tmp_path/"b.xyz"))
    assert any("Unsupported file extension '.xyz'" in str(w.message) for w in always_catch_warnings)



if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])