# tests/test_compat_numpy.py
import warnings

import numpy as np
import pytest

from kdiagram.compat import numpy as knp


def test_flags_consistent_with_numpy_version():
    v = np.__version__
    assert knp.IS_NP2 == (tuple(map(int, v.split(".")[:2])) >= (2, 0))
    assert knp.IS_NP125_PLUS == (tuple(map(int, v.split(".")[:2])) >= (1, 25))


@pytest.mark.parametrize(
    "py_t, np_check",
    [
        (knp.int_, np.integer),
        (knp.float_, np.floating),
        (knp.bool_, np.bool_),
        (knp.complex_, np.complexfloating),
        (knp.str_, np.str_),
        (knp.object_, np.object_),
    ],
)
def test_builtin_aliases_work_as_dtypes(py_t, np_check):
    arr = np.asarray([0, 1], dtype=py_t)
    assert np.issubdtype(arr.dtype, np_check)


@pytest.mark.parametrize(
    "np_scalar",
    [knp.NP_INT, knp.NP_FLOAT, knp.NP_BOOL, knp.NP_COMPLEX, knp.NP_STR],
)
def test_numpy_scalar_aliases_exist(np_scalar):
    # Just ensure these are usable as dtypes
    arr = np.asarray([0], dtype=np_scalar)
    assert isinstance(arr, np.ndarray)


def test_in1d_and_row_stack_and_trapz():
    assert np.all(knp.in1d([1, 2], [2, 3]) == np.isin([1, 2], [2, 3]))
    stacked = knp.row_stack(([1, 2], [3, 4]))
    assert stacked.shape == (2, 2)

    # trapz should integrate y=x over x=[0,1,2] -> area = 2
    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    area = knp.trapz(y, x)
    assert pytest.approx(area) == 2.0


def test_axiserror_available():
    # Raise AxisError on bad axis via numpy call that checks axes
    with pytest.raises(knp.AxisError):
        np.sum(np.arange(6).reshape(2, 3), axis=2)


def test_asarray_copy_kw_no_crash_and_correct_dtype():
    # Should never crash whether NumPy 1.x or 2.x
    a = [1, 2, 3]
    out = knp.asarray(a, dtype=float, copy=True)
    assert out.dtype.kind == "f"
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_find_common_type_matches_result_type_no_deprecation():
    arr_dtypes = [np.int32, np.float64]
    expected = np.result_type(*arr_dtypes)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = knp.find_common_type(arr_dtypes, [])
        # On NumPy >= 1.25, our shim must NOT emit a DeprecationWarning
        if knp.IS_NP125_PLUS:
            assert not any(issubclass(wi.category, DeprecationWarning) for wi in w)
    assert got == np.dtype(expected)


def test_set_promotion_warn_no_error():
    # No-op on 1.x; available on 2.x. Should not raise.
    knp.set_promotion_warn("weak_and_warn")


if __name__ == "__main__":
    pytest.main([__file__])
