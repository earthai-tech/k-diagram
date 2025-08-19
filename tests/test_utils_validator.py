import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from kdiagram.utils import validator as V


# -----------------------
# validate_length_range
# -----------------------
def test_validate_length_range_happy_and_sorting_and_errors():
    # auto-sorts when sorted_values=True
    assert V.validate_length_range((202, 25)) == (25, 202)
    # already sorted
    assert V.validate_length_range((1, 3)) == (1, 3)
    # equality after sorting -> error
    with pytest.raises(ValueError):
        V.validate_length_range((5, 5))
    # non-numeric -> error
    with pytest.raises(ValueError):
        V.validate_length_range(("a", 2))
    # sorted_values=False keeps order and allows first>second here (no extra checks)
    assert V.validate_length_range((9, 1), sorted_values=False) == (9, 1)


# ------------
# validate_yy
# ------------
def test_validate_yy_shapes_types_and_expected_type():
    # flatten=True
    y_true = np.array([[0.0], [1.0], [2.0]])
    y_pred = np.array([[0.0], [1.0], [2.0]])
    yt, yp = V.validate_yy(y_true, y_pred, flatten=True)
    assert yt.ndim == yp.ndim == 1

    # auto flatten: "auto" string path
    yt2, yp2 = V.validate_yy(y_true, y_pred, flatten="auto")
    assert yt2.ndim == yp2.ndim == 1

    # wrong ndim -> error (suggests flatten=True)
    with pytest.raises(ValueError):
        V.validate_yy(np.array([[1, 2]]), np.array([[1, 2]]), flatten=False)

    # length mismatch through check_consistent_length -> error
    with pytest.raises(ValueError):
        V.validate_yy([0, 1], [0], flatten=True)

    # expected_type mismatch -> error
    with pytest.raises(ValueError):
        V.validate_yy([0.1, 0.2], [0.1, 0.3], expected_type="binary", flatten=True)

    # expected_type correct (continuous)
    yt3, yp3 = V.validate_yy(
        [0.1, 0.2], [0.1, 0.3], expected_type="continuous", flatten=True
    )
    assert yt3.shape == yp3.shape == (2,)


# ------------------------
# contains_nested_objects
# ------------------------
def test_contains_nested_objects_strict_and_non_strict():
    nested = [{1, 2}, [3, 4], {"k": "v"}]
    mixed = [1, 2, [3]]
    flat = [1, 2, 3]
    assert V.contains_nested_objects(nested) is True
    assert V.contains_nested_objects(nested, strict=True) is True
    assert V.contains_nested_objects(mixed) is True
    assert V.contains_nested_objects(mixed, strict=True) is False
    assert V.contains_nested_objects(flat) is False


# ------------------------
# check_consistent_length / _num_samples
# ------------------------
class _HasFit:
    def fit(self):  # pragma: no cover - behavior is in _num_samples
        return self


def test_check_consistent_length_and_num_samples_errors():
    # mismatch
    with pytest.raises(ValueError):
        V.check_consistent_length([1, 2, 3], [1])
    # estimator-like with .fit -> TypeError in _num_samples
    with pytest.raises(TypeError):
        V._num_samples(_HasFit())
    # 0-dim array -> TypeError
    with pytest.raises(TypeError):
        V._num_samples(np.array(1.0))
    # normal paths
    assert V._num_samples(np.array([[1, 2], [3, 4]])) == 2
    assert V._num_samples([1, 2, 3]) == 3


# -----------
# is_in_if
# -----------
def test_is_in_if_variants_raise_warn_ignore_and_returns():
    o = ["apple", "banana", "cherry"]

    # single string present (no error)
    assert V.is_in_if(o, "banana") is None

    # missing -> raise
    with pytest.raises(ValueError):
        V.is_in_if(o, "date", error="raise")

    # warn -> returns None
    with pytest.warns(UserWarning):
        V.is_in_if(o, ["date"], error="warn")

    # return_diff forces ignore
    diff = V.is_in_if(o, ["banana", "date"], return_diff=True)
    assert diff == ["date"]

    # return_intersect forces ignore
    inter = V.is_in_if(o, ["banana", "date"], return_intersect=True)
    assert inter == ["banana"]


# ----------------
# exist_features
# ----------------
def test_exist_features_happy_and_modes_and_errors():
    df = pd.DataFrame({"f1": [1], "f2": [2]})

    # happy
    assert V.exist_features(df, ["f1", "f2"]) is True

    # string feature
    assert V.exist_features(df, "f1") is True

    # numpy array and Index supported
    assert V.exist_features(df, np.array(["f1"])) is True
    assert V.exist_features(df, pd.Index(["f2"])) is True

    # invalid df param
    with pytest.raises(ValueError):
        V.exist_features(["not", "df"], ["a"])

    # invalid error param
    with pytest.raises(ValueError):
        V.exist_features(df, ["x"], error="boom")

    # warn branch returns False
    with pytest.warns(UserWarning):
        assert V.exist_features(df, ["f1", "x"], error="warn") is False

    # ignore branch returns False
    assert V.exist_features(df, ["x"], error="ignore") is False


# -------------------
# _assert_all_types
# -------------------
def test__assert_all_types_valid_and_error_message():
    # valid
    assert V._assert_all_types([1, 2], list) == [1, 2]
    # invalid with custom name
    with pytest.raises(TypeError, match="must be of type"):
        V._assert_all_types([1, 2], tuple, objname="features")


# --------
# is_frame
# --------
def test_is_frame_detects_df_or_series_and_modes_and_deprecation_bridge():
    df = pd.DataFrame({"A": [1, 2]})
    s = pd.Series([1, 2], name="A")

    assert V.is_frame(df) is True
    assert V.is_frame(s) is True
    assert V.is_frame(s, df_only=True, error="ignore") is False

    # error='raise'
    with pytest.raises(TypeError):
        V.is_frame(123, df_only=True, error="raise", objname="Input")

    # warn
    with pytest.warns(UserWarning):
        assert V.is_frame(123, df_only=False, error="warn") is False

    # deprecated raise_exception=True flips to raise even if error!='raise'
    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError):
            V.is_frame(123, df_only=True, error="ignore", raise_exception=True)


# -------------
# build_data_if
# -------------
def test_build_data_if_paths_and_type_coercions_and_col_conversion():
    # dict -> DataFrame (columns from keys)
    out1 = V.build_data_if({"a": [1, 2], "b": [3, 4]}, to_frame=True)
    assert isinstance(out1, pd.DataFrame) and list(out1.columns) == ["a", "b"]

    # list of lists -> force autogenerated columns
    data = [[1, "2021-01-01"], [2, "bad-date"], [3, "2021-03-01"]]
    with pytest.warns(UserWarning):
        # start_incr_at non-int with error='warn'
        # triggers warning and defaults to 0
        out2 = V.build_data_if(
            data,
            to_frame=True,
            columns=None,
            force=True,
            error="warn",
            coerce_datetime=True,
            coerce_numeric=True,
            start_incr_at="nope",
            col_prefix="c_",
        )
    assert isinstance(out2, pd.DataFrame)
    # column names auto-generated with prefix
    assert list(out2.columns) == ["c_0", "c_1"]
    # dtype coercion: first col numeric, second stays
    # date/object depending on parse success
    assert np.issubdtype(out2.dtypes.iloc[0], np.number)

    # Series -> DataFrame
    ser = pd.Series([1, 2, 3], name="s")
    out3 = V.build_data_if(ser, to_frame=True)
    assert isinstance(out3, pd.DataFrame)

    # to_frame True but columns missing and not forced -> raises
    with pytest.raises(TypeError):
        V.build_data_if(
            np.array([[1, 2], [3, 4]]),
            to_frame=True,
            columns=None,
            force=False,
            error="raise",
        )

    # int columns converted to strings with prefix
    df_intcols = pd.DataFrame([[1, 2], [3, 4]])
    out4 = V._convert_int_columns_to_str(df_intcols, col_prefix="col_")
    assert list(out4.columns) == ["col_0", "col_1"]
    out4b = V._convert_int_columns_to_str(df_intcols, col_prefix=None)
    assert list(out4b.columns) == ["0", "1"]


# -------------------
# recheck_data_types
# -------------------

def test_recheck_data_types_handles_mixed_types_and_warns():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "3"],
            "b": ["2021-01-01", "nope", "2021-03-01"], 
            "c": ["1.1", "2.2", "3.3"],
        }
    )
    
    # Expect the warning from the mixed-type 'b' column
    with pytest.warns(UserWarning, match="Could not infer format"):
        out = V.recheck_data_types(
            df, coerce_numeric=True, coerce_datetime=True, 
            return_as_numpy=False
        )

    # Assert that types were coerced correctly where possible
    assert np.issubdtype(out["a"].dtype, np.number)
    assert np.issubdtype(out["c"].dtype, np.floating)
    # Assert that the problematic column remains an object
    assert pd.api.types.is_object_dtype(out["b"].dtype)

def test_recheck_data_types_handles_non_df_input():
    # This input will produce a datetime warning because the function
    # attempts to convert all object columns to datetime by default.
    list_input = [["1", "2"], ["3", "4"]]
    
    # FIX: Expect the UserWarning from pd.to_datetime
    with pytest.warns(UserWarning, match="Could not infer format"):
        out_np = V.recheck_data_types(list_input, return_as_numpy="auto")
    
    # Assertions remain the same
    assert isinstance(out_np, np.ndarray)
    # Check that the data was correctly converted to numeric
    # after the datetime conversion failed.
    assert np.issubdtype(out_np.dtype, np.number)
    
# ---------------
# array_to_frame
# ---------------
def test_array_to_frame_paths_and_warnings_and_sparse_passthrough():
    X = np.array([[1, 2], [3, 4]])

    # generate columns with force
    out = V.array_to_frame(X, to_frame=True, columns=None, input_name="X", force=True)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == ["X_0", "X_1"]

    # raising when to_frame and columns None and not forced
    with pytest.raises(ValueError):
        V.array_to_frame(
            X,
            to_frame=True,
            columns=None,
            input_name="X",
            force=False,
            raise_exception=True,
        )

    # warn (and return original X) when to_frame True, missing columns, not forced
    with pytest.warns(UserWarning):
        out2 = V.array_to_frame(
            X,
            to_frame=True,
            columns=None,
            input_name="X",
            force=False,
            raise_warning=True,
        )
    assert isinstance(out2, np.ndarray)

    # with explicit columns to DataFrame
    out3 = V.array_to_frame(
        X, to_frame=True, columns=["a", "b"], input_name="X", force=False
    )
    assert isinstance(out3, pd.DataFrame) and list(out3.columns) == ["a", "b"]

    # sparse matrix path: convert_array_to_pandas
    # should skip conversion (returns sparse)
    XS = sp.csr_matrix([[1, 0], [0, 1]])
    out4 = V.array_to_frame(
        XS, to_frame=True, columns=["s1", "s2"], input_name="S", force=False
    )
    assert sp.issparse(out4)


# -------------------------
# convert_array_to_pandas
# -------------------------
def test_convert_array_to_pandas_all_branches():
    # string X -> TypeError
    with pytest.raises(TypeError):
        V.convert_array_to_pandas("oops", to_frame=True, columns=["a"])

    # invalid array-like -> TypeError
    class NotArrayLike: ...

    with pytest.raises(TypeError):
        V.convert_array_to_pandas(NotArrayLike(), to_frame=True, columns=["a"])

    # to_frame True without columns -> ValueError
    with pytest.raises(ValueError):
        V.convert_array_to_pandas(np.array([[1, 2]]), to_frame=True, columns=None)

    # 1D -> Series
    ser, cols = V.convert_array_to_pandas(
        np.array([1, 2, 3]), to_frame=True, columns=["v"]
    )
    assert isinstance(ser, pd.Series) and ser.name == "v"

    # 2D -> DataFrame with matching cols
    df, cols2 = V.convert_array_to_pandas(
        np.array([[1, 2], [3, 4]]), to_frame=True, columns=["a", "b"]
    )
    assert isinstance(df, pd.DataFrame) and list(df.columns) == ["a", "b"]

    # mismatched shape/columns -> ValueError
    with pytest.raises(ValueError):
        V.convert_array_to_pandas(
            np.array([[1, 2], [3, 4]]), to_frame=True, columns=["a"]
        )


# -----------
# ensure_2d
# -----------
def test_ensure_2d_array_frame_and_auto():
    # 1D -> column vector
    x = np.array([1, 2, 3])
    arr = V.ensure_2d(x, output_format="array")
    assert isinstance(arr, np.ndarray) and arr.shape == (3, 1)

    # DataFrame stays DataFrame (frame)
    df = pd.DataFrame([1, 2, 3])
    out = V.ensure_2d(df, output_format="frame")
    assert isinstance(out, pd.DataFrame)

    # list -> auto (returns DataFrame because not df
    # originally? function returns np array unless df?)
    out2 = V.ensure_2d([1, 2, 3], output_format="auto")
    # auto returns DataFrame if is_dataframe else original array;
    # here it's np.array column vector
    assert isinstance(out2, (np.ndarray, pd.DataFrame))

    # invalid output_format -> parameter validator raises
    with pytest.raises(ValueError):
        V.ensure_2d([1, 2], output_format="nope")


# ---------------------
# parameter_validator
# ---------------------
def test_parameter_validator_contains_and_exact_and_no_raise():
    validate = V.parameter_validator(
        "outlier_method",
        ["z_score", "iqr"],
        match_method="contains",
        raise_exception=True,
        deep=True,
    )
    assert validate("Z_sco") == "z_score"

    validate_exact = V.parameter_validator(
        "mode", ["train", "test"], match_method="exact", raise_exception=True
    )
    assert validate_exact("train") == "train"
    with pytest.raises(ValueError):
        validate_exact("tra")

    validate_no_raise = V.parameter_validator(
        "fill", ["median", "mean"], match_method="contains", raise_exception=False
    )
    assert validate_no_raise("average") is None


# -------------------
# normalize_string
# -------------------
def test_normalize_string_modes_and_targets_and_errors():
    # no targets => normalized lowercased
    assert V.normalize_string("Hello World") == "hello world"

    # exact match
    assert (
        V.normalize_string("train", ["train", "test"], match_method="exact") == "train"
    )

    # contains
    assert (
        V.normalize_string("this-is-iqr", ["z_score", "iqr"], match_method="contains")
        == "this-is-iqr"
    )

    # startswith
    norm, target = V.normalize_string(
        "Goodbye World", ["hello", "goodbye"], num_chars_check=7, return_target_str=True
    )
    assert norm.startswith("goodbye") and target == "goodbye"

    # deep
    assert V.normalize_string("abc", ["xyzabc123"], deep=True) == "abc"

    # return_target_only matched
    assert V.normalize_string("MODE", ["mode"], return_target_only=True) == "mode"

    # not found + raise
    with pytest.raises(ValueError):
        V.normalize_string("X", ["a", "b"], raise_exception=True)

    # not found + no raise
    assert V.normalize_string("X", ["a", "b"]) == ""


# -------------
# is_iterable
# -------------
def test_is_iterable_variants_and_parse_string_errors():
    # boolean checks
    assert V.is_iterable("x", exclude_string=True) is False
    assert V.is_iterable([1, 2]) is True

    # transform wrapping
    assert V.is_iterable(5, transform=True) == [5]

    # parse_string requires transform=True
    with pytest.raises(ValueError):
        V.is_iterable("a b", parse_string=True, transform=False)

    # parse_string with transform
    out = V.is_iterable("col1, col2", parse_string=True, transform=True)
    assert isinstance(out, list) and out[0] in {"col1", "col2"}


# -----------------------
# check_spatial_columns
# -----------------------
def test_check_spatial_columns_paths():
    df = pd.DataFrame({"longitude": [-1], "latitude": [0], "v": [3]})

    # ok
    assert V.check_spatial_columns(df) is None

    # not df
    with pytest.raises(TypeError):
        V.check_spatial_columns(["not", "df"])

    # wrong tuple size
    with pytest.raises(ValueError):
        V.check_spatial_columns(df, spatial_cols=("lon",))

    # missing columns
    with pytest.raises(ValueError):
        V.check_spatial_columns(df, spatial_cols=("lon", "lat"))


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
