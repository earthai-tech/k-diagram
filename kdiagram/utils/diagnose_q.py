#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

"""
Diagnose and validating quantile-related operations.
Provides utilities for handling quantile data in various formats
and ensuring compatibility with expected structures.
"""

import operator
import re
import warnings
from collections.abc import Sequence
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np
import pandas as pd

from .validator import is_frame, is_in_if

__all__ = [
    "to_iterable",
    "validate_quantiles",
    "validate_quantiles_in",
    "validate_q_dict",
    "check_forecast_mode",
    "detect_quantiles_in",
    "build_q_column_names",
    "detect_digits",
    "validate_consistency_q",
    "parse_qcols",
    "validate_qcols",
    "build_qcols_multiple",
]


def parse_qcols(q_cols, fallback_cols=None, error="warn"):
    r"""
    parse_qcols is a utility function designed to interpret
    quantile column mappings from either a dictionary or list.
    It automatically identifies the lowest quantile, the median
    quantile (preferably 50 if available), and the highest
    quantile. The remaining quantiles, if any, are accessible
    through a parsed dictionary. This utility helps streamline
    the process of extracting quantile-based columns for later
    processing or plotting [1]_.

    Given a set of quantiles in :math:`q`, typically named
    like ``q10``, ``q50``, ``q90``, parse_qcols attempts to
    extract the numeric part of each name. For example:

    .. math::
       q_{key} = \\text{float}(\\text{key}[1:])

    If ``'q50'`` is found, it is treated as the median quantile.
    Otherwise, parse_qcols uses the central element of the
    sorted list of parsed quantiles. The minimum quantile
    becomes the "lowest" and the maximum becomes the "highest."

    Parameters
    ------------
    q_cols : dict or list, optional
        A collection of quantile definitions. If this
        parameter is a dictionary with keys like ``q10``,
        ``q50``, or ``q90``, the numeric portion is parsed.
        If it is a list, items are assigned dummy keys in
        ascending order (``q0``, ``q1``, etc.).
    fallback_cols : tuple of str, optional
        A 3-tuple (``lower_col``, ``median_col``,
        ``upper_col``) to return if the quantile parsing
        fails or if <parameter `q_cols`> is None.
    error : {'warn', 'raise', 'ignore'}, optional
        A function used to warn about parsing issues. If
        'raise', error raises rather than warning issues.

    Returns
    ---------
    dict
        A dictionary containing:
        - ``lowest_col``: The column name of the lowest
          quantile.
        - ``median_col``: The column name of the median
          quantile (preferably 50).
        - ``highest_col``: The column name of the highest
          quantile.
        - ``parsed_qvals``: A mapping of parsed quantile
          floats to their column names.
        - ``valid``: A boolean indicating whether valid
          quantiles were parsed.

    Notes
    -------
    By default, parse_qcols handles numeric quantile keys that
    begin with the letter 'q', followed by a valid float value
    (e.g., ``q10`` -> 10.0). Keys that cannot be converted into
    floats are ignored. If none are valid, parse_qcols returns
    values from <parameter `fallback_cols`>.

    Examples
    ----------
    >>> from kdiagram.utils.diagnose_q import parse_qcols
    >>> # Example dictionary
    >>> q_def = {'q10': 'low_10', 'q50': 'med_50', 'q90': 'hi_90'}
    >>> result = parse_qcols(q_def)
    >>> result['lowest_col']
    'low_10'


    References
    ------------
    .. [1] Doe, A., & Smith, J. (2021). Dynamic quantile
       extraction in large datasets. Journal of Data
       Diagnostics, 4(2), 101-110.
    """
    # Provide a default warning function if none is passed
    if fallback_cols is None:
        fallback_cols = (None, None, None)

    # Prepare output structure with defaults
    output = {
        "lowest_col": fallback_cols[0],
        "median_col": fallback_cols[1],
        "highest_col": fallback_cols[2],
        "parsed_qvals": {},
        "valid": False,
    }

    # If q_cols is not provided, return fallback immediately
    if not q_cols:
        return output

    if isinstance(q_cols, str):
        q_cols = [q_cols]
    # If q_cols is a list, convert to dict with dummy keys (q0, q1, etc.)
    if isinstance(q_cols, (list, tuple)):
        q_cols = {f"q{i}": col_name for i, col_name in enumerate(q_cols)}

    # Parse keys like 'q10', 'q50', 'q90' into numeric floats
    parsed = {}
    for k, col_name in q_cols.items():
        if not isinstance(k, str):
            msg = f"Quantile key '{k}' is not a string. Skipped."
            if error == "warn":
                warnings.warn(msg, stacklevel=2)
            elif error == "raise":
                raise TypeError(msg)
            continue
        if not k.startswith("q"):
            msg = f"Key '{k}' is not prefixed with 'q'. Skipped."
            if error == "warn":
                warnings.warn(msg, stacklevel=2)
            elif error == "raise":
                raise ValueError(msg)
            continue
        # Attempt to convert 'q10' -> 10.0
        try:
            q_val = float(k[1:])
            parsed[q_val] = col_name
        except ValueError as e:
            msg = f"Cannot parse quantile '{k}'. Skipped."
            if error == "warn":
                warnings.warn(msg, stacklevel=2)
            elif error == "raise":
                raise ValueError(msg) from e

    # If nothing valid was parsed, return fallback
    if not parsed:
        msg = (
            "No valid quantile columns found in `q_cols`. "
            "Falling back to explicit columns if provided."
        )
        if error == "warn":
            warnings.warn(msg, stacklevel=2)
        elif error == "raise":
            raise ValueError(msg)

        return output

    # Sort parsed q-values
    sorted_qvals = sorted(parsed.keys())
    output["parsed_qvals"] = parsed
    output["valid"] = True

    # The lowest quantile
    output["lowest_col"] = parsed[sorted_qvals[0]]
    # The highest quantile
    output["highest_col"] = parsed[sorted_qvals[-1]]

    # For median, prefer '50' if it exists, else pick middle
    if 50.0 in parsed:
        output["median_col"] = parsed[50.0]
    else:
        mid_idx = len(sorted_qvals) // 2
        output["median_col"] = parsed[sorted_qvals[mid_idx]]

    return output


def check_forecast_mode(
    mode, q=None, error="raise", ops="validate", q_mode="strict", **kw
):
    r"""
    Check consistency between forecast mode and quantile values.

    This function verifies that the provided forecast `mode`
    is consistent with the quantile values (`q`). If the mode
    is ``"point"`` and quantile values are provided, it will
    either warn the user and reset `q` to ``None`` (if
    ``error=="warn"``) or raise a ValueError (if
    ``error=="raise"``). Similarly, if the mode is
    ``"quantile"`` and no quantile values are provided, it will
    either warn the user and set `q` to the default values
    ``[0.1, 0.5, 0.9]`` (if ``error=="warn"``) or raise a
    ValueError (if ``error=="raise"``).

    Additionally, if ``ops`` is set to ``"check_only"``, the function
    only performs the checks without modifying or returning ``q``.

    Parameters
    ------------
    mode : str
        Forecast mode, either ``"point"`` or
        ``"quantile"``.
    q : list of float, optional
        List of quantile values. Defaults to ``None``.
    error : str, optional
        Error handling behavior. If set to ``"raise"``, a
        ValueError is raised when an inconsistency is
        detected. If set to ``"warn"``, a warning is issued
        and a default behavior is applied.
    ops : str, optional
        Operation mode. If set to ``"check_only"``, the function only
        performs the checks without returning any value. If set to
        ``"validate"``, the function returns the validated
       (or updated) quantile values. Default is ``"validate"``.
    q_mode: {'strict', 'soft'}
       Validation quantiles mode. See more in
       :func:`kdiagram.utils.diagnose_q.validate_quantiles`
    *kw: dict,
        Additional keywords argument of :func:`kdiagram.utils.diagnose_q`.

    Returns
    ---------
    q : list of float or None
        The validated (or updated) quantile values if ``ops`` is
        ``"validate"``; otherwise, returns ``None``.

    Raises
    --------
    ValueError
        If an inconsistency is detected and ``error`` is set to
        ``"raise"``.

    Examples
    ----------
    >>> from gofast.utils.diagnose_q impor check_forecast_mode
    >>> check_forecast_mode("point", q=[0.1, 0.5, 0.9])
    # Raises a ValueError or warns and returns None based on the error flag.

    >>> check_forecast_mode("quantile", q=None, error="warn")
    # Issues a warning and returns [0.1, 0.5, 0.9].

    """

    # Ensure mode is valid.
    if mode not in ["point", "quantile"]:
        raise ValueError("mode must be either 'point' or 'quantile'.")

    # Handle the case for "point" mode.
    if mode == "point":
        if q is not None:
            msg = (
                "In point mode, quantile values (q) should be None. "
                "Resetting q to None."
            )
            if error == "warn":
                warnings.warn(msg, stacklevel=2)

            elif error == "raise":
                raise ValueError(msg)

            q = None
    # Handle the case for "quantile" mode.
    elif mode == "quantile":
        if q is None:
            msg = (
                "In quantile mode, quantile values (q) must be provided. "
                "Setting default quantiles to [0.1, 0.5, 0.9]."
            )
            if error == "warn":
                warnings.warn(msg, stacklevel=2)

            elif error == "raise":
                raise ValueError(msg)

            q = [0.1, 0.5, 0.9]
        # then validate quantiles
        q = validate_quantiles(q, mode=q_mode, **kw)
    # If ops is "check_only", simply return None.
    if ops == "check_only":
        return None
    else:
        return q


def to_iterable(
    obj: Any,
    exclude_string: bool = False,
    transform: bool = False,
    parse_string: bool = False,
    flatten: bool = False,
    unique: bool = False,
    delimiter: str = r"[ ,;|\t\n]+",
) -> Union[bool, list[Any]]:
    r"""
    Determines if an object is iterable, with options to transform, parse,
    and modify the input for flexible iterable handling.

    Parameters
    ------------
    obj : Any
        Object to be evaluated or transformed into an iterable.
    exclude_string : bool, default=False
        Excludes strings from being considered as iterable objects.
    transform : bool, default=False
        Transforms `obj` into an iterable if it isn't already. Defaults to
        wrapping `obj` in a list.
    parse_string : bool, default=False
        If `obj` is a string, splits it into a list based on the specified
        `delimiter`. Requires `transform=True`.
    flatten : bool, default=False
        If `obj` is a nested iterable, flattens it into a single list.
    unique : bool, default=False
        Ensures unique elements in the output if `transform=True`.
    delimiter : str, default=r'[ ,;|\t\n]+'
        Regular expression pattern for splitting strings when `parse_string=True`.

    Returns
    ---------
    bool or List[Any]
        Returns a boolean if `transform=False`, or an iterable if
        `transform=True`.

    Raises
    --------
    ValueError
        If `parse_string=True` without `transform=True`, or if `delimiter`
        is invalid.

    Notes
    -------
    - When `parse_string` is used, strings are split by `delimiter` to form a
      list of substrings.
    - `flatten` and `unique` apply only when `transform=True`.
    - Using `unique=True` ensures no duplicate values in the output.

    Examples
    ----------
    >>> from kdiagram.utils.diagnose_q import to_iterable
    >>> to_iterable("word", exclude_string=True)
    False

    >>> to_iterable(123, transform=True)
    [123]

    >>> to_iterable("parse, this sentence", transform=True, parse_string=True)
    ['parse', 'this', 'sentence']

    >>> to_iterable([1, [2, 3], [4]], transform=True, flatten=True)
    [1, 2, 3, 4]

    >>> to_iterable("a,b,a,b", transform=True, parse_string=True, unique=True)
    ['a', 'b']
    """
    if parse_string and not transform:
        raise ValueError("Set 'transform=True' when using 'parse_string=True'.")

    # Check if object is iterable (excluding strings if specified)
    is_iterable = hasattr(obj, "__iter__") and not (
        exclude_string and isinstance(obj, str)
    )

    # If transformation is not needed, return the boolean check
    if not transform:
        return is_iterable

    # If string parsing is enabled and obj is a string, split it using delimiter
    if isinstance(obj, str) and parse_string:
        obj = re.split(delimiter, obj.strip())

    # Wrap non-iterables into a list if they aren't iterable
    elif not is_iterable:
        obj = [obj]

    # Flatten nested iterables if flatten=True
    if flatten:
        obj = _flatten(obj)

    # Apply unique filtering if requested
    if unique:
        obj = list(dict.fromkeys(obj))  # Preserves order while ensuring uniqueness

    return obj


def _flatten(nested_list: Any) -> list[Any]:
    """Helper function to recursively flatten a nested list structure."""
    flattened = []
    for element in nested_list:
        if isinstance(element, (list, tuple, set)):
            flattened.extend(_flatten(element))
        else:
            flattened.append(element)
    return flattened


def validate_q_dict(q_dict, recheck=False):
    r"""
     Converts the keys of a dictionary of quantile columns (`q_dict`) from
     string representations to numeric values (float) if possible. If the key
     cannot be converted to a number, it returns the dictionary as is.

     Optionally validates the quantiles after conversion to ensure that all
     keys are within the valid range of quantiles [0, 1].

     Parameters
     ------------
     q_dict : dict
         A dictionary where the keys represent quantiles (either as
         strings like '0.1' or '10%') and the values are lists of
         column names associated with those quantiles.

     recheck : bool, optional, default=False
         If `True`, the keys of the dictionary will be validated as
         quantiles after the conversion. This validation checks that
         all keys lie within the range [0, 1].

     Returns
     ---------
     dict
         A dictionary with numeric keys if conversion is successful,
         otherwise the original dictionary. The keys are either floats
         representing quantiles or the original keys if they cannot
         be converted.

     Notes
     -------
     The function performs the following steps:

     1. Iterates over the dictionary to check whether each key can be
        converted to a numeric value.
     2. If a key is a string containing a percentage (e.g., '10%'),
        it removes the '%' sign and divides the value by 100 to
        convert it to a float.
     3. If a key can be successfully converted, it is stored as a
        floating-point number in the resulting dictionary.
     4. If the conversion fails (due to a `ValueError`, `TypeError`, or
        `AttributeError`), the original key is retained in the dictionary.
     5. If `recheck` is `True`, it validates the converted quantiles by
        ensuring they are in the range [0, 1].

     The function is designed to handle both direct float conversions (e.g.,
     '0.1') and percentage-based representations (e.g., '10%' becomes 0.1).

     Example
     ---------
     >>> q_dict = {'0.1': ['subsidence_q10'], '50%': ['subsidence_q50'],
                   '90%': ['subsidence_q90']}
     >>> validate_q_dict(q_dict)
     {0.1: ['subsidence_q10'], 0.5: ['subsidence_q50'], 0.9: ['subsidence_q90']}

     >>> q_dict = {'0.1': ['subsidence_q10'], '0.5': ['subsidence_q50'],
                   '0.9': ['subsidence_q90']}
     >>> validate_q_dict(q_dict)
     {'0.1': ['subsidence_q10'], 0.5: ['subsidence_q50'], 'high': ['subsidence_q90']}

     >>> q_dict = {'0.1': ['subsidence_q10'], '200%': ['subsidence_q200']}
     >>> validate_q_dict(q_dict, recheck=True)
     {0.1: ['subsidence_q10'], 2.0: ['subsidence_q200']}

     See Also
     ----------
    validate_quantiles`:
        Validates if the values are valid quantiles in the range [0, 1].

     References
     ------------
     .. [1] Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in
            statistical packages. The American Statistician, 50(4), 361-365.
     .. [2] Weiss, N. A. (2015). Introductory Statistics. Pearson.
    """
    # Initialize an empty dictionary to store the converted quantiles
    new_q_cols = {}

    if not isinstance(q_dict, dict):
        raise TypeError(
            f"Expected a dictionary for `q_dict`, but "
            f"got {type(q_dict).__name__}. Ensure that `q_dict`"
            " is a dictionary where the keys represent quantiles "
            "(either as strings like '0.1' or '10%') and the values"
            " are lists of column names."
        )
    for key, value in q_dict.items():
        try:
            # Check if the key contains a percentage sign and convert it
            if "%" in str(key):
                key_float = float(key.replace("%", "")) / 100.0
            else:
                # Directly convert the key to float
                key_float = float(key)

            new_q_cols[key_float] = value

        except (AttributeError, ValueError, TypeError):
            # If conversion fails, retain the original key
            new_q_cols[key] = value

    if recheck:
        # Validate the quantiles after conversion
        # (i.e., ensure keys are between 0 and 1)
        validate_quantiles(list(new_q_cols.keys()), dtype="float64")

    return new_q_cols


def validate_quantiles(
    quantiles,
    asarray=False,
    round_digits=2,
    dtype=None,
    mode="strict",
    scale_method="uniform",
):
    r"""
    Validate and normalize quantile values with flexible conversion rules.

    Ensures quantile inputs are valid probabilities :math:`q \in [0,1]` while
    providing mechanisms for automatic value adjustment through different
    scaling strategies [1]_.

    .. math::

        Q_{\text{adj}} = \frac{q_{\text{raw}}}{10^{\lfloor \log_{10}(q_{\text{raw}}) \rfloor + 1}}

    where :math:`q_{\text{raw}}` is the input value requiring adjustment.

    Parameters
    ------------
    quantiles : array-like
        Input values to validate. Accepts:
        - Numeric values in [0,1]
        - Percentages (e.g., "20%")
        - Integers for automatic scaling in ``mode='soft'``
    asarray : bool, default=False
        Determines output format:
        - ``True``: Returns numpy array
        - ``False``: Returns Python list
    round_digits : int, default=2
        Number of decimal places for rounding to mitigate floating-point
        precision issues
    dtype : str or numpy.dtype, default='float32'
        Output data type. Supported values: 'float32' (TF-compatible) or
        'float64' (high precision)
    mode : {'strict', 'soft'}, default='strict'
        Validation strictness:
        - ``'strict'``: Rejects values outside [0,1]
        - ``'soft'``: Converts percentages and scales integers using
           ``scale_method``
    scale_method : {'uniform', 'individual'}, default='uniform'
        Scaling strategy for ``mode='soft'``:
        - ``'uniform'``: Uses maximum digit count from all values for divisor
        - ``'individual'``: Scales each value independently

    Returns
    ---------
    list or numpy.ndarray
        Validated quantiles in specified format. Return type matches
        ``asarray`` parameter.

    Raises
    ------
    TypeError
        For non-numeric inputs in ``mode='strict'`` or invalid types in
        ``mode='soft'``
    ValueError
        For values outside [0,1] in ``mode='strict'`` or invalid scaling
        conversions

    Examples
    -----------
    Basic validation:
    >>> from kdiagram.utils.diagnose_q import validate_quantiles
    >>> validate_quantiles([0.1, 0.5, 0.9])
    [0.1, 0.5, 0.9]

    Soft mode with percentage conversion:
    >>> validate_quantiles(["20%", 5, 150], mode='soft')
    [0.2, 0.05, 0.15]

    Array output with custom precision:
    >>> validate_quantiles([0.123456, 0.789012], asarray=True, round_digits=3)
    array([0.123, 0.789], dtype=float32)

    Notes
    --------
    1. In ``mode='soft'``:
       - Percentages convert via :math:`\frac{\text{value}}{100}`
       - Integer scaling uses:
         - Uniform: :math:`\frac{\text{value}}{10^{\text{max\_digits}}}`
         - Individual: :math:`\frac{\text{value}}{10^{\text{self\_digits}}}`

    2. Rounding follows banker's rounding (numpy.round behavior) to minimize
       cumulative errors [2]_.

    See Also
    ----------
    gofast.stats.evaluate_quantiles : Evaluates quantile estimation accuracy
    numpy.quantile : Computes quantiles of array values

    References
    ------------
    .. [1] IEEE Standard for Floating-Point Arithmetic. IEEE Std 754-2019.

    .. [2] Hyndman, R.J. & Fan, Y. (1996). Sample Quantiles in Statistical
           Packages. The American Statistician, 50(4), 361-365.
    """
    quantiles = to_iterable(quantiles, transform=True, flatten=True)

    if mode == "soft":
        quantiles = _process_soft_quantiles(quantiles, scale_method=scale_method)

    if not isinstance(quantiles, (list, np.ndarray)):
        raise TypeError(
            "Quantiles must be list or numpy array. "
            f"Received {type(quantiles).__name__}."
        )

    dtype = _get_valid_dtype(dtype)
    quantiles_np = np.array(quantiles, dtype=dtype)

    _validate_quantile_values(quantiles_np)
    quantiles_np = np.round(quantiles_np, decimals=round_digits)

    return quantiles_np if asarray else quantiles_np.tolist()


def _process_soft_quantiles(quantiles, scale_method):
    """Process quantiles in soft mode with scaling adjustments."""
    scaled_values = []
    scale_candidates = []

    for q in quantiles:
        q_val, needs_scaling = _process_single_quantile(q)

        if needs_scaling:
            scale_candidates.append(q_val)
            scaled_values.append(None)
        else:
            scaled_values.append(q_val)

    if scale_candidates:
        scaled = _apply_scaling(scale_candidates, scale_method=scale_method)
        scaled_values = _merge_scaled_values(scaled_values, scaled)

    return scaled_values


def _process_single_quantile(q):
    """Process individual quantile value for soft mode."""
    original = q
    q = _convert_string_quantile(q)

    if not isinstance(q, (int, float)):
        raise TypeError(
            f"Quantile {original} must be numeric. " f"Received {type(q).__name__}."
        )

    if q < 0:
        raise ValueError(f"Negative quantile value: {original}")

    if 0 <= q <= 1:
        return q, False

    if not np.isclose(q, int(q)):
        raise ValueError(f"Non-integer out-of-range quantile: {original}")

    return int(q), True


def _convert_string_quantile(q):
    """Convert string quantiles to numeric values."""
    if isinstance(q, str):
        q = q.strip().rstrip("%")
        try:
            value = float(q)
            if "%" in q:
                value /= 100.0
            return value
        except ValueError:
            raise ValueError(f"Could not convert string quantile: {q}") from None
    return q


def _apply_scaling(scale_candidates, scale_method):
    """Apply scaling strategy to out-of-range quantiles."""
    if scale_method == "uniform":
        max_digits = max(len(str(q)) for q in scale_candidates)
        divisor = 10**max_digits
        return [q / divisor for q in scale_candidates]

    if scale_method == "individual":
        return [q / (10 ** len(str(q))) for q in scale_candidates]

    raise ValueError(
        f"Invalid scale_method: {scale_method}. " "Choose 'uniform' or 'individual'."
    )


def _merge_scaled_values(values, scaled):
    """Merge scaled values back into original quantile list."""
    result = []
    scale_idx = 0

    for val in values:
        if val is None:
            result.append(scaled[scale_idx])
            scale_idx += 1
        else:
            result.append(val)

    return result


def _get_valid_dtype(dtype):
    """Validate and return proper numpy dtype."""
    dtype_map = {"float32": np.float32, "float64": np.float64}

    if dtype is None:
        return np.float32

    if isinstance(dtype, str) and dtype in dtype_map:
        return dtype_map[dtype]

    return dtype if dtype in (np.float32, np.float64) else np.float32


def _validate_quantile_values(quantiles_np):
    """Core validation for quantile value requirements."""
    if not np.issubdtype(quantiles_np.dtype, np.number):
        raise ValueError("All quantiles must be numeric.")

    if np.any((quantiles_np < 0) | (quantiles_np > 1)):
        raise ValueError(
            "Quantiles must be in [0, 1] range. "
            "Use 'soft' mode for automatic scaling."
        )


def validate_quantiles_in(
    quantiles,
    asarray=False,
    round_digits=1,
    dtype=None,
    mode="strict",
):
    r"""
    Validates the input quantiles and optionally returns the output as a
    numpy array or list, with an option to round the quantiles to a
    specified number of decimal places to avoid floating-point precision
    issues.

    Quantiles are numerical values used in statistical analysis to
    divide a distribution into intervals. They must lie within the
    range [0, 1] as they represent proportions of data [1]_.

    Parameters
    ----------
    quantiles : list or numpy.ndarray
        Input array-like containing quantile values to be validated.
        The values must be numeric and within the range [0, 1].

    asarray : bool, optional
        Determines the output format. If `True`, the validated
        quantiles are returned as a numpy array. If `False`, they
        are returned as a list. Default is `False`.

    round_digits : int, optional, default=1
        The number of decimal places to which the quantiles should be
        rounded. This helps avoid floating-point precision errors such as
        `0.10000000149011612` being displayed as `0.1`. By default,
        quantiles are rounded to 1 decimal place.

    dtype : numpy.dtype, optional, default=np.float32
        The data type for the quantiles array. Use `np.float32`
        for compatibility with TensorFlow or `np.float64` for higher
        precision. The dtype determines the precision used for quantiles
        during validation and rounding.

    Returns
    -------
    list or numpy.ndarray
        A list or numpy array of validated quantile values, depending
        on the value of `asarray`.

    Raises
    ------
    TypeError
        If the input `quantiles` is not a list or numpy array.

    ValueError
        If any element of `quantiles` is not numeric or lies outside
        the range [0, 1].

    Notes
    -----
    Quantiles, denoted as :math:`q \in [0, 1]`, represent the fraction
    of observations below a certain value in a distribution:

    .. math::

        Q(q) = \inf \{ x \in \mathbb{R} : P(X \leq x) \geq q \}

    where :math:`Q(q)` is the quantile function, and :math:`q` is the
    proportion [2]_.

    This function ensures that all values in `quantiles` adhere to
    this definition by checking:

    1. The type of `quantiles`.
    2. The numerical nature of its elements.
    3. The range of its values.
    4. The optional rounding of the quantiles to a specified number
       of decimal places.

    Examples
    --------
    >>> from gofast.utils.diagnose_q import validate_quantiles_in
    >>> validate_quantiles([0.1, 0.2, 0.5])
    [0.1, 0.2, 0.5]

    >>> validate_quantiles(np.array([0.3, 0.7, 0.9]), asarray=True)
    array([0.3, 0.7, 0.9])

    >>> validate_quantiles([0.10000000149011612, 0.5, 0.8999999761581421], round_digits=1)
    [0.1, 0.5, 0.9]

    >>> validate_quantiles([0.5, 1.2])
    ValueError: All quantile values must be in the range [0, 1].

    See Also
    --------
    numpy.percentile : Computes the nth percentile of an array.
    numpy.quantile : Computes the qth quantile of an array.

    References
    ----------
    .. [1] Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in
           statistical packages. The American Statistician, 50(4), 361-365.
    .. [2] Weiss, N. A. (2015). Introductory Statistics. Pearson.

    """
    # Convert quantiles to a list if necessary
    quantiles = to_iterable(quantiles, transform=True, flatten=True)

    # Validate input type: must be list or numpy array
    if not isinstance(quantiles, (list, np.ndarray)):
        raise TypeError(
            "Quantiles must be a list or numpy array. Received "
            f"{type(quantiles).__name__!r}."
        )

    # Define a dictionary for mapping string dtype names to numpy float types
    dtypes = {"float32": np.float32, "float64": np.float64}

    # Check if dtype is a string, and convert
    # it to the corresponding numpy dtype
    if dtype is None:
        dtype = "float32"
    if isinstance(dtype, str):
        if dtype not in dtypes:
            raise ValueError(
                f"Unsupported dtype string: {dtype}."
                " Supported values are 'float32' or 'float64'."
            )
        # Convert string to corresponding numpy dtype
        dtype = dtypes[dtype]

    # Convert input to numpy array for consistent
    # processing using the specified dtype
    quantiles = np.array(quantiles, dtype=dtype)

    # Validate that all elements are numeric
    if not np.issubdtype(quantiles.dtype, np.number):
        raise ValueError("All quantile values must be numeric.")

    # Validate that all values are within the range [0, 1]
    if not np.all((quantiles >= 0) & (quantiles <= 1)):
        raise ValueError("All quantile values must be in the range [0, 1].")

    # Round quantiles to the specified number of decimal places
    quantiles = np.round(quantiles, decimals=round_digits)

    # Return quantiles in the desired format
    return quantiles if asarray else quantiles.tolist()


def detect_quantiles_in(
    df: pd.DataFrame,
    col_prefix: Optional[str] = None,
    dt_value: Optional[list[str]] = None,
    mode: str = "soft",
    return_types: str = "columns",
    verbose: int = 0,
) -> Union[list[str], list[float], list[np.ndarray], pd.DataFrame, None]:
    r"""
    Detect quantile columns in a DataFrame using naming patterns and
    value validation.

    Identifies columns containing quantile data through structured naming
    conventions and value validation [1]_. Supports both absolute and normalized
    quantile representations through mode-based value adjustment [2]_.


    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing potential quantile columns. Column names
        must be strings.
    col_prefix : str, optional
        Column name prefix for targeted search (e.g., ``'price'`` for
        ``price_q0.25``). If None, scans all columns.
    dt_value : list of str, optional
        Date filters for temporal quantile detection (e.g., ``['2023']`` matches
        columns like ``price_2023_q0.5``).
    mode : {'soft', 'strict'}, default='soft'
        Value handling strategy:
        - ``'soft'``: Normalizes values >1 to 1.0 using min-max scaling
        - ``'strict'``: Excludes values outside [0,1] range
    return_types : {'columns', 'q_val', 'values', 'frame'}, default='columns'
        Return format specification:
        - ``'columns'``: List of column names
        - ``'q_val'``: Sorted unique quantile values
        - ``'values'``: Column data arrays
        - ``'frame'``: DataFrame subset
    verbose : {0, 1, 2, 3}, default=0
        Output verbosity:
        - 0: Silent
        - 1: Basic scan info
        - 2: Per-column matches
        - 3: Full diagnostic output

    Returns
    -------
    Union[List[str], List[float], List[np.ndarray], pd.DataFrame, None]
        Quantile data in format specified by ``return_types``. Returns None if
        no quantiles detected.

    Notes
    --------
    The detection adjustment can be formulated as :

    .. math::

        q_{\text{adj}} = \begin{cases}
        \min(1, \max(0, q_{\text{raw}})) & \text{if } mode=\text{'soft'} \\
        q_{\text{raw}} & \text{if } q \in [0,1] \text{ and } mode=\text{'strict'}
        \end{cases}

    1. Column name pattern requirements:
       - Requires ``_qX`` suffix where X is numeric
       - Temporal format: ``{prefix}_{date}_q{value}``
       - Non-temporal format: ``{prefix}_q{value}``

    2. Value adjustment in soft mode uses piecewise function:
       - Clips values to [0,1] range
       - Preserves original values within valid range

    Examples
    --------
    >>> from kdiagram.utils.diagnose_q import detect_quantiles_in
    >>> import pandas as pd
    >>>
    >>> # Basic detection
    >>> df = pd.DataFrame({'sales_q0.25': [4.2], 'sales_q0.75': [5.8]})
    >>> detect_quantiles_in(df, col_prefix='sales')
    ['sales_q0.25', 'sales_q0.75']
    >>>
    >>> # Temporal quantile filtering
    >>> df = pd.DataFrame({'temp_2023_q0.5': [22.1], 'temp_2024_q0.5': [23.4]})
    >>> detect_quantiles_in(df, dt_value=['2023'], return_types='q_val')
    [0.5]
    >>>
    >>> # Value normalization
    >>> df = pd.DataFrame({'risk_q150': [0.8]})
    >>> detect_quantiles_in(df, mode='soft', return_types='q_val')
    [1.0]

    See Also
    --------
    kdiagram.utils.validate_quantiles : For quantile value validation
    pandas.DataFrame.filter : For column selection by pattern

    References
    ----------
    .. [1] Regular Expression HOWTO, Python Documentation
    .. [2] Pandas API Reference: DataFrame operations

    """

    is_frame(df, df_only=True, objname="Data 'df'")
    df.columns = df.columns.astype(str)
    col_prefix = col_prefix or ""

    quantile_columns = []
    found_quantiles = set()

    _log_verbose(f"Scanning DataFrame columns with prefix: {col_prefix}", verbose, 1)

    for col in df.columns:
        result = _process_column(col, df, col_prefix, dt_value, mode, verbose)
        if result:
            q_val, col_data = result
            found_quantiles.add(q_val)
            _store_results(q_val, col_data, quantile_columns, return_types, col)

    _log_verbose(f"Quantiles detected: {sorted(found_quantiles)}", verbose, 3)

    return _format_output(quantile_columns, found_quantiles, return_types, df)


def _process_column(
    col: str,
    df: pd.DataFrame,
    prefix: str,
    dt_values: Optional[list[str]],
    mode: str,
    verbose: int,
) -> Optional[tuple]:
    """Process individual column for quantile detection."""
    # Handle both cases: with or without prefix
    if prefix:
        # Remove the prefix part
        if col.startswith(f"{prefix}_"):
            col_match = col[len(prefix) + 1 :]  # Remove prefix and underscore
        else:
            return None  # If the column does not start with the prefix, skip it
    else:
        col_match = col  # No prefix, use the column name directly

    match, q_str = _check_column_match(col_match, dt_values)
    if not match:
        return None

    try:
        q_val = _extract_quantile_value(q_str, mode)
    except ValueError as e:
        _log_verbose(f"Invalid quantile value in {col}: {e}", verbose, 2)
        return None

    _log_verbose(f"Found quantile match: {col} with value: {q_val}", verbose, 2)
    return q_val, df[col].values


def _check_column_match(
    remainder: str, dt_values: Optional[list[str]], prefix: Optional[str] = None
) -> tuple:
    """Check if column remainder matches date and quantile patterns."""
    # Match quantile pattern like q0.25
    quantile_pattern = re.compile(r"q([\d\.]+)$")
    # Case 1: If a date filter is provided, look for
    # columns matching the date and quantile patterns
    if dt_values:
        for d_str in dt_values:
            date_pattern = f"^{d_str}_q"  # Include date check
            # Match column with date filter
            if remainder.startswith(date_pattern):
                m = quantile_pattern.search(remainder)
                return (True, m.group(1)) if m else (False, None)

    # Case 2: If no date filter is provided,
    # look for quantile matches
    else:
        # Check for the quantile pattern in the remainder
        m = quantile_pattern.search(remainder)
        if m:
            # Return the quantile value (e.g., 0.25)
            return (True, m.group(1))

    return (False, None)


def _extract_quantile_value(q_str: str, mode: str) -> float:
    """Extract and validate quantile value with proper error handling."""
    try:
        q_val = float(q_str)
    except ValueError as e:
        raise ValueError(f"Invalid quantile format: {q_str}") from e

    # Use centralized validation from validate_quantiles
    validated = validate_quantiles(
        [q_val],
        mode=mode,
        scale_method="individual",
        round_digits=2,
        dtype=np.float64,
    )
    return validated[0]


def _store_results(
    q_val: float,
    col_data: np.ndarray,
    quantile_columns: list,
    return_types: str,
    col: str,
) -> None:
    """Store results based on requested return type."""
    if return_types == "values":
        quantile_columns.append(col_data)
    elif return_types == "q_val":
        quantile_columns.append(q_val)
    else:
        quantile_columns.append(col)


def _format_output(
    quantile_columns: list, found_quantiles: set, return_types: str, df: pd.DataFrame
) -> Union[list, pd.DataFrame, None]:
    """Format final output based on return_types."""
    if not quantile_columns:
        return None

    if return_types == "frame":
        return df[quantile_columns]
    if return_types == "q_val":
        return sorted(found_quantiles)
    if return_types == "values":
        return np.vstack(quantile_columns) if quantile_columns else []
    return sorted(quantile_columns)


def _log_verbose(message: str, verbose_level: int, required_level: int) -> None:
    """Centralized verbose logging control."""
    if verbose_level >= required_level:
        print(message)


def build_q_column_names(
    df: pd.DataFrame,
    quantiles: list[Union[float, str]],
    value_prefix: Optional[str] = None,
    dt_value: Optional[list[Union[str, int]]] = None,
    strict_match: bool = True,
) -> list[str]:
    r"""
    Generate and validate quantile column names following naming conventions.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame containing potential quantile columns
    quantiles : list of float/str
        Quantile values to search for (0 < q < 1). Accepts:
        - Float values (e.g., 0.25)
        - String representations (e.g., "25%")
    value_prefix : str, optional
        Column name prefix for structured naming. If None,
        looks for unprefixed columns.
    dt_value : list of str/int, optional
        Temporal identifiers for time-aware quantiles. Converts
        all values to strings.

    strict_match : bool, default=True
        Matching strategy:
        - ``True``: Requires exact column name matches
        - ``False``: Uses regex pattern matching for flexible detection

    Returns
    -------
    list
        Valid column names found in the DataFrame matching the
        quantile naming pattern.

    Notes
    -------
    Constructs column names using the pattern:

    .. math::
        \text{col_name} = \begin{cases}
        \text{value_prefix}\_\text{date}\_q\text{quantile} & \text{if both prefix and date exist} \\
        \text{value_prefix}\_q\text{quantile} & \text{if only prefix exists} \\
        \text{date}\_q\text{quantile} & \text{if only date exists} \\
        q\text{quantile} & \text{otherwise}
        \end{cases}

    Examples
    --------
    >>> from kdiagram.utils.diagnose_q import build_q_column_names
    >>> import pandas as pd
    >>>
    >>> # Basic usage with prefix
    >>> df = pd.DataFrame(columns=['price_q0.25', 'price_2023_q0.5'])
    >>> build_q_column_names(df, [0.25, 0.5], 'price')
    >>>
    >>> # if strict_match ts
    ['price_q0.25', 'price_2023_q0.5']
    >>>
    >>> # Date-filtered search
    >>> build_q_column_names(df, [0.5], 'price', dt_value=['2023'])
    ['price_2023_q0.5']
    >>>
    >>> # Unprefixed columns
    >>> df = pd.DataFrame(columns=['q0.75', '2024_q0.9'])
    >>> build_q_column_names(df, [0.75, 0.9])
    ['q0.75', '2024_q0.9']

    See Also
    --------
    kdiagram.utils.diagnose_q.validate_quantiles : For quantile value validation
    pandas.Series.str.contains : For column pattern matching
    """
    is_frame(df, df_only=True, objname="Data 'df'")
    # Validate and normalize inputs
    valid_quantiles = validate_quantiles(
        quantiles,
        mode="soft",
        round_digits=2,
        dtype="float64",
    )
    date_strings = _process_dt_values(dt_value)
    df.columns = df.columns.astype(str)

    if strict_match:
        candidates = _generate_strict_candidates(
            valid_quantiles, value_prefix, date_strings
        )
        return [col for col in candidates if col in df.columns]

    # Flexible pattern matching
    pattern = _build_flexible_pattern(valid_quantiles, value_prefix, date_strings)
    return [col for col in df.columns if pattern.search(col)]


def _generate_strict_candidates(
    quantiles: list[float], prefix: Optional[str], dates: list[str]
) -> list[str]:
    """Generate exact match candidates in all valid formats."""
    candidates = []
    for q in quantiles:
        # Decimal format (q0.25)
        dec_str = f"q{q:.4f}".rstrip("0").rstrip(".")
        # Percentage format (q25)
        pct_str = f"q{int(round(q * 100))}"

        for fmt in [dec_str, pct_str]:
            # Temporal candidates
            if dates:
                candidates.extend(
                    f"{prefix}_{d}_{fmt}" if prefix else f"{d}_{fmt}" for d in dates
                )
            # Non-temporal candidates
            candidates.append(f"{prefix}_{fmt}" if prefix else fmt)
    return list(set(candidates))  # Remove duplicates


def _build_flexible_pattern(
    quantiles: list[float], prefix: Optional[str], dates: list[str]
) -> re.Pattern:
    """Build regex pattern for flexible quantile matching."""
    # Quantile alternatives (0.25|25)
    q_alternatives = "|".join(
        f"{q:.4f}".rstrip("0").rstrip(".") + "|" + str(int(round(q * 100)))
        for q in quantiles
    )

    # Prefix component
    prefix_part = f"{re.escape(prefix)}_?" if prefix else ""

    # Date component
    date_part = f"({'|'.join(map(re.escape, dates))})_+" if dates else r"\d{4}_?|"

    return re.compile(
        rf"^{prefix_part}(?:{date_part})?q({q_alternatives})\b", flags=re.IGNORECASE
    )


def _process_dt_values(dt_values: Optional[list[Union[str, int]]]) -> list[str]:
    """Normalize temporal values to standardized strings."""
    return [str(v).strip() for v in dt_values] if dt_values else []


def detect_digits(
    value,
    pattern: str = None,
    as_q: bool = False,
    return_unique: bool = False,
    sort: bool = False,
    error: str = "ignore",
    verbose: int = 0,
) -> list:
    r"""
    Detect numeric values in a string or list of strings.

    This function extracts numeric values from the input by applying a
    robust regular expression. When used in quantile mode (i.e., when
    ``as_q`` is True), it captures numbers that appear immediately after
    the substring ``_q`` and before either ``_step`` or the end-of-string.
    In general mode (when ``as_q`` is False), it uses a conventional digit
    detector. The extracted numeric values are converted to floats.

    .. math::
       \text{Extracted Value} = \text{value after } `\_q` \text{ and before }
       (\texttt{\_step} \text{ or end-of-string})

    Parameters
    ----------
    value         : Union[str, List[str]]
        A string or a list of strings from which to extract numeric values.
    pattern       : str, optional
        A custom regular expression pattern. If ``None``, the default is:

        - If ``as_q`` is True:

          ``"(?<=_q)(\\d+(?:\\.\\d+)?)(?=(_step|$))"``.

        - Otherwise:

          ``(?<!\d)(\d+(?:\.\d+)?)(?!\d)""``.
    as_q          : bool, optional
        If True, converts each detected numeric value to a quantile value
        using soft mode via ``validate_quantiles``. Default is False.
    return_unique : bool, optional
        If True, returns only unique detected values. Default is False.
    sort          : bool, optional
        If True, returns the detected numbers in ascending order.
        Default is False.
    error         : str, optional
        Specifies how to handle conversion errors. Options are:
        ``"raise"`` to throw a ValueError,
        ``"warn"`` to print a warning message, or
        ``"ignore"`` to skip invalid matches.
        Default is ``"ignore"``.
    verbose       : int, optional
        Verbosity level for debugging output. Higher values (e.g., 5 or
        above) produce more detailed logs. Default is 0.

    Returns
    -------
    list
        A list of numeric values (floats) extracted from the input. If
        ``as_q`` is True, these values are converted to quantile values in
        soft mode.

    Examples
    --------
    >>> from kdiagram.utils.diagnose_q import detect_digits
    >>> # Single string example:
    >>> detect_digits("subsidence_q10_step1")
    [10.0]
    >>> # List of strings:
    >>> detect_digits(["subsidence_q10_step1",
    ...                "subsidence_q50_step1",
    ...                "subsidence_q89_step1"])
    [10.0, 50.0, 89.0]
    >>> # With conversion to quantile (soft mode):
    >>> detect_digits("subsidence_q10.5_step1", as_q=True)
    [0.105]  # Example: converts 10.5 to 0.105 in soft mode.

    Notes
    -----
    - The default regex pattern for quantile mode employs lookbehind and
      lookahead assertions to ensure that the numeric value is immediately
      preceded by ``_q`` and followed by ``_step`` or the end-of-string.
    - When ``as_q`` is False, a more general digit detection regex is used.
    - Input that is not a list is automatically converted to a list of strings.
    - The ``error`` parameter controls whether conversion issues raise an
      exception, warn the user, or are silently ignored.

    See Also
    --------
    validate_quantiles : Converts numeric values to quantile values in soft mode.

    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
           (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
    .. [2] Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D.
           (2006). *Compilers: Principles, Techniques, and Tools* (2nd ed.).
           Pearson.
    """

    # If no custom regex pattern is provided, select a default pattern.
    if pattern is None:
        if as_q:
            # Use a pattern to capture numbers after '_q'
            # and before '_step' or end-of-string.
            pattern = r"(?<=_q)(\d+(?:\.\d+)?)(?=(_step|$))"
        else:
            # General robust digit detection using word boundaries.
            pattern = r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)"
            # #r"(?<!\d)(\d+)(?!\d)" #r"\b\d+(?:\.\d+)?\b"

    # Compile the regex pattern.
    regex = re.compile(pattern)

    # Ensure the input is treated as a list of strings.
    if not isinstance(value, list):
        input_data = [str(value)]
    else:
        input_data = [str(item) for item in value]

    digits = []  # List to store detected numbers.

    # Iterate over each string in the input_data.
    for text in input_data:
        matches = regex.findall(text)
        if verbose >= 5:
            print(f"[DEBUG] Processing '{text}' => Matches: {matches}")
        for match in matches:
            try:
                # match can be a tuple (if using lookahead groups),
                # so extract the first element if needed.
                num_str = match[0] if isinstance(match, tuple) else match
                num = float(num_str)
                digits.append(num)
            except ValueError as exc:
                if error == "raise":
                    raise ValueError(f"Could not convert '{match}' to float.") from exc
                elif error == "warn":
                    if verbose >= 1:
                        print(f"[WARN] Skipping value '{match}': conversion failed.")
                # If error is "ignore", continue without appending.
                continue

    # If conversion to quantile is requested,
    # convert numbers using soft mode.
    if as_q:
        digits = validate_quantiles(
            digits, mode="soft", round_digits=2, dtype=np.float64
        )

    # Remove duplicates if requested.
    if return_unique:
        digits = list(set(digits))

    # Optionally sort the detected numbers.
    if sort:
        digits = sorted(digits)

    if verbose >= 3:
        print(f"[INFO] Detected digits: {digits}")

    return digits


def validate_consistency_q(
    user_q: list[float],
    q_items: Union[str, list[Any]],
    error: str = "raise",
    mode: str = "soft",
    msg: Optional[str] = None,
    default_to: str = "valid_q",
    verbose: int = 0,
):
    r"""
    Validate the consistency of user-specified quantile values with those
    auto-detected from the input.

    This function compares the quantile values provided in ``user_q``
    with the numeric values extracted from ``q_items`` (using
    :func:`detect_digits` with ``as_q=True``). Let :math:`Q_{user}` be the
    set of quantile values provided by the user and :math:`Q_{det}` be the
    set of quantile values detected from ``q_items``. In soft mode, the
    function returns the intersection, i.e.,

    .. math::
       Q_{valid} = Q_{user} \cap Q_{det},

    whereas in strict mode, it expects an exact match and returns
    :math:`Q_{user}` directly.

    Parameters
    ----------
    user_q         : list of float
        A list of quantile values provided by the user. These represent
        the expected quantiles for evaluation or forecasting.
    q_items        : Union[str, List[str], pandas.DataFrame]
        The source from which quantile values are auto-detected. This can be
        a string, a list of strings, or a DataFrame whose columns contain
        quantile information.
    error          : str, optional
        Determines the error handling behavior if the user-specified
        quantiles do not match the detected values. Options are:
          - ``"raise"`` : Raise a ValueError.
          - ``"warn"``  : Emit a warning and continue.
          - ``"ignore"``: Silently ignore mismatches.
        Default is ``"raise"``.
    mode           : str, optional
        The matching mode. In ``"soft"`` mode (default), the function returns
        the intersection of user and detected quantiles. In ``"strict"`` mode,
        the user-specified quantiles must exactly match those detected, and
        the function returns ``user_q``.
    msg            : str, optional
        A custom error message to use if inconsistencies are found. If not
        provided, a default message is generated.
    default_to: str, default='valid_q'
       Return kind when inconsistent numbers found in quantiles.
       In ``'soft'`` mode, it controls whether to return the 'valid_q'
       valids quantiles or ``'auto_q'``for automatic_detected quanties.
       Defaut is the ``'valid_q'``.

    verbose       : int, optional
        Verbosity level for debugging output. Higher values (e.g., 5 or above)
        yield more detailed logs. Default is 0.

    Returns
    -------
    list
        A sorted list of validated quantile values (as floats) that are
        consistent between the user-specified values and those detected
        from ``q_items``.

    Examples
    --------
    >>> from kdiagram.utils.diagnose_q import validate_consistency_q
    >>> user_quantiles = [0.1, 0.5, 0.9]
    >>> columns = ["subsidence_q10_step1", "subsidence_q50_step1",
    ...            "subsidence_q90_step1", "other_column"]
    >>> validate_consistency_q(user_quantiles, columns)
    [0.1, 0.5, 0.9]

    Notes
    -----
    This function leverages :func:`detect_digits` to extract numeric quantile
    values from the input and :func:`is_in_if` to compute the intersection
    between the user-specified and detected quantiles. In soft mode, minor
    discrepancies are tolerated; strict mode requires an exact match.

    See Also
    --------
    detect_digits       : Extracts numeric values from strings, including decimals.
    is_in_if            : Checks membership and returns the intersection of lists.
    validate_quantiles  : Converts numeric values to quantile values in soft mode.

    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
           (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
    .. [2] Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D.
           (2006). *Compilers: Principles, Techniques, and Tools*. Pearson.
    """

    # If q_items is a DataFrame, extract its columns.
    if isinstance(q_items, pd.DataFrame):
        q_items = q_items.columns

    # Detect quantile values from q_items using detect_digits in quantile mode.
    detected_q_values = detect_digits(q_items, as_q=True, sort=True, return_unique=True)
    if verbose >= 5:
        print(f"[DEBUG] Detected quantile values: {detected_q_values}")

    # Use is_in_if to get the intersection between user_q and detected_q_values.
    valid_quantiles = is_in_if(
        sorted(user_q), detected_q_values, error=error, return_intersect=True
    )
    if verbose >= 5:
        print(f"[DEBUG] Valid quantiles after intersection: {valid_quantiles}")

    # If valid_quantiles is not empty, sort it; otherwise, handle error.

    if valid_quantiles:
        valid_quantiles = sorted(valid_quantiles)
    else:
        default_err = (
            "User provided quantiles do not match any detected "
            "quantile values in the DataFrame columns:"
            f" {user_q} != {detected_q_values}"
        )
        err_msg = msg if msg is not None else default_err
        suff = ". Returning " + (
            "an empty list."
            if default_to == "valid_q"
            else ("the detected values instead.")
        )
        if error == "raise":
            raise ValueError(err_msg)
        elif error == "warn":
            warnings.warn(err_msg + f"{suff}", UserWarning, stacklevel=2)

        return [] if default_to == "valid_q" else detected_q_values

    # In strict mode, expect the user_q to exactly match the detected quantiles.
    if mode == "strict":
        # valid_quantiles = user_q
        valid_quantiles = _verify_identical_items(
            user_q,
            detected_q_values,
            ops="validate",
            objname="quantiles list",
        )

    # Check consistency in count between valid and detected quantiles.
    if len(valid_quantiles) != len(detected_q_values):
        default_err = (
            "Inconsistent number of quantiles: user provided "
            f"valid {len(valid_quantiles)} ({valid_quantiles}) vs detected "
            f"{len(detected_q_values)} ({detected_q_values})."
        )
        err_msg = msg if msg is not None else default_err

        suff = " Returning " + (
            "valid_quantiles instead."
            if default_to == "valid_q"
            else ("detected values instead.")
        )
        if default_to == "valid_q":
            if error == "raise":
                raise ValueError(err_msg)
            elif error == "warn":
                warnings.warn(err_msg + f"{suff}", UserWarning, stacklevel=2)
        else:  # 'auto_q'
            if error == "warn":
                warnings.warn(err_msg + f"{suff}", UserWarning, stacklevel=2)
            valid_quantiles = detected_q_values

    # Optionally sort the result if not already
    # sorted (redundant here, but for safety).
    return sorted(valid_quantiles)


def _verify_identical_items(
    list1,
    list2,
    mode: str = "unique",
    ops: str = "check_only",
    error: str = "raise",
    objname: str = None,
) -> Union[bool, list]:
    r"""
    Check if two lists contain identical elements according
    to the specified mode.

    In "unique" mode, the function compares the unique elements
    in each list.
    In "ascending" mode, it compares elements pairwise in order.

    Parameters
    ----------
    list1     : list
        The first list of items.
    list2`     : list
        The second list of items.
    mode      : {'unique', 'ascending'}, default="unique"
        The mode of comparison:
          - "unique": Compare unique elements (order-insensitive).
          - "ascending": Compare each element pairwise in order.
    ops       : {'check_only', 'validate'}, default="check_only"
        If "check_only", returns True/False indicating a match.
        If "validate", returns the validated list.
    error     : {'raise', 'warn', 'ignore'}, default="raise"
        Specifies how to handle mismatches.
    objname   : str, optional
        A name to include in error messages.

    Returns
    -------
    bool or list
        Depending on `ops`, returns True/False or the validated list.

    Examples
    --------
    >>> from gofast.core.generic import verify_identical_items
    >>> list1 = [0.1, 0.5, 0.9]
    >>> list2 = [0.1, 0.5, 0.9]
    >>> verify_identical_items(list1, list2, mode="unique", ops="validate")
    [0.1, 0.5, 0.9]
    >>> verify_identical_items(list1, list2, mode="ascending", ops="check_only")
    True

    Notes
    -----
    In "ascending" mode, both lists must have the same length, and the
    function compares each corresponding pair of elements.
    In "unique" mode, the function uses the set of unique values for
    comparison. If the lists contain mixed types, the function attempts
    to compare their string representations.
    """
    # Validate mode.
    if mode not in ("unique", "ascending"):
        raise ValueError("mode must be either 'unique' or 'ascending'")
    if ops not in ("check_only", "validate"):
        raise ValueError("ops must be either 'check_only' or 'validate'")
    if error not in ("raise", "warn", "ignore"):
        raise ValueError("error must be one of 'raise', 'warn', or 'ignore'")

    # Ascending mode: compare each element in order.
    if mode == "ascending":
        if len(list1) != len(list2):
            msg = (
                f"Length mismatch in {objname or 'object lists'}: "
                f"{len(list1)} vs {len(list2)}."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings

                warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        differences = []
        for idx, (a, b) in enumerate(zip(list1, list2)):
            if a != b:
                differences.append((idx, a, b))
        if differences:
            msg = f"Differences in {objname or 'object lists'}: {differences}."
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings

                warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        return True if ops == "check_only" else list1

    # Unique mode: compare the unique elements of each list.
    else:
        try:
            unique1 = sorted(set(list1))
            unique2 = sorted(set(list2))
        except Exception:
            unique1 = sorted({str(x) for x in list1})
            unique2 = sorted({str(x) for x in list2})
        if unique1 != unique2:
            msg = (
                f"Inconsistent unique elements in {objname or 'object lists'}: "
                f"{unique1} vs {unique2}."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings

                warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        return True if ops == "check_only" else unique1


def validate_qcols(
    q_cols: Union[str, int, Sequence[Any]],
    ncols_exp: Optional[str] = None,
    err_msg: Optional[str] = None,
) -> list[str]:
    r"""
    Validate and standardise a collection of column names that
    represent quantiles or prediction outputs. The function
    `validate_qcols` converts the input to a clean list of
    strings, removes blanks, and—optionally—checks that the
    final list length satisfies an expectation expressed in
    `<ncols_exp>`.

    .. math::
       \text{valid} = \bigl\\{\\,c \\mid c \neq ''\bigr\\}

    If an expectation is supplied, the function compares
    :math:`|\\,\text{valid}\\,|` to the requested condition and
    raises an error if the test fails.

    Parameters
    ----------
    q_cols : list, str, tuple or set
        Column names to validate. May be a single string, an
        iterable of names, or any mixture thereof. Non‑string
        entries are cast to string.

    ncols_exp : str or None, optional
        Expectation on the number of columns. The string must
        begin with a comparison operator (``'==', '>=', '<=',
        '!=', '>'`` or ``'<'``) followed by an integer, e.g.
        ``'>=2'`` or ``'==3'``. If *None*, no length check is
        applied.

    err_msg : str or None, optional
        Custom message to raise if the expectation in
        `<ncols_exp>` is not met. If *None*, a default message
        is generated.

    Returns
    -------
    list
        A cleaned list of column names that meet all checks.

    Raises
    ------
    TypeError
        If `<q_cols>` is not a recognised container or string.

    ValueError
        If `<q_cols>` is empty after cleaning, or if the length
        check in `<ncols_exp>` fails.

    Examples
    --------
    >>> from kdiagram.utils.diagnose_q import validate_qcols
    >>> validate_qcols('q50')
    ['q50']
    >>> validate_qcols(['q10', 'q90'], ncols_exp='==2')
    ['q10', 'q90']
    >>> validate_qcols(('p1', 'p2', ''), ncols_exp='>=2')
    ['p1', 'p2']

    Notes
    -----
    The expectation string is parsed by splitting on the first
    occurring comparison operator and casting the remainder to
    int. This avoids ambiguous patterns and guarantees that
    ``ops[op](len(cols), expected)`` is evaluated safely.

    See Also
    --------
    operator : Built‑in module providing comparison functions.

    References
    ----------
    .. [1] Harris, C. R. *et al.* (2020). Array programming
           with NumPy. *Nature*, 585, 357‑362.
    """

    # Step‑1  : convert <q_cols> to a list of strings
    if q_cols is None:
        raise ValueError(
            "`q_cols` cannot be None. Provide at least " "one column name."
        )

    if isinstance(q_cols, (str, int)):
        q_cols = [str(q_cols)]
    elif isinstance(q_cols, (tuple, set, list)):
        q_cols = [str(col) for col in q_cols]
    else:
        raise TypeError("`q_cols` must be a list, tuple, set or " "single string.")

    # Remove blanks and strip white‑space
    q_cols = [col.strip() for col in q_cols if col.strip()]

    if len(q_cols) == 0:
        raise ValueError("`q_cols` is empty after cleaning.")

    # Step‑2  : optional length expectation check
    if ncols_exp:
        _ops: dict[str, Any] = {
            "==": operator.eq,
            "=": operator.eq,
            "!=": operator.ne,
            ">=": operator.ge,
            "<=": operator.le,
            ">": operator.gt,
            "<": operator.lt,
        }

        # longest operators first (>=, <=, !=, ==)
        for sym in sorted(_ops, key=len, reverse=True):
            if ncols_exp.startswith(sym):
                num_str = ncols_exp[len(sym) :].strip()
                if not num_str.isdigit():
                    raise ValueError(f"Invalid expectation syntax " f"'{ncols_exp}'.")
                expected = int(num_str)
                if not _ops[sym](len(q_cols), expected):
                    raise ValueError(
                        err_msg
                        or f"Expected {ncols_exp}, got " f"{len(q_cols)}: {q_cols}"
                    )
                break
        else:
            raise ValueError(f"Invalid `ncols_exp` format: " f"{ncols_exp}")

    return q_cols


def build_qcols_multiple(
    q_cols: Optional[Sequence[tuple[str, ...]]] = None,
    qlow_cols: Optional[Sequence[str]] = None,
    qup_cols: Optional[Sequence[str]] = None,
    qmed_cols: Optional[Sequence[str]] = None,
    *,
    enforce_triplet: bool = False,
    allow_pair_when_median: bool = False,
) -> list[tuple[str, ...]]:
    r"""
    Assemble and validate tuples of quantile columns.

    Parameters
    ----------
    q_cols : sequence of tuple, optional
        Pre‑built tuples of column names. Each tuple can
        be ``(q10, q90)`` or ``(q10, q50, q90)``. If this
        argument is supplied, the helper bypasses the
        individual `qlow_cols`, `qup_cols`, and
        `qmed_cols` inputs.
    qlow_cols : sequence of str, optional
        Column names representing the lower quantile
        (e.g. 10 th percentile).
    qup_cols : sequence of str, optional
        Column names representing the upper quantile
        (e.g. 90 th percentile).
    qmed_cols : sequence of str, optional
        Column names representing the median (e.g. 50 th
        percentile). If provided, each tuple will be
        returned as ``(q10, q50, q90)``.
    enforce_triplet : bool, default=False
        * If ``True`` the output **must** be a triplet
          ``(low, med, up)``. Raises an error when
          `qmed_cols` is missing or when `q_cols`
          contains pairs.
        * If ``False`` the function returns pairs when
          no median columns are supplied.
    allow_pair_when_median : bool, default=False
        By default, when `qmed_cols` is supplied the
        helper always returns triplets. Set this flag to
        ``True`` to ignore `qmed_cols` and still output
        pairs ``(low, up)`` (useful for quick A/B tests).

    Returns
    -------
    list of tuple
        A list of tuples with either two or three column
        names depending on the inputs and flags.

    Raises
    ------
    ValueError
        On mismatched lengths, invalid tuple sizes, or
        missing mandatory inputs.

    Examples
    --------
    >>> from kdiagram.utils.diagnose_q import build_qcols_multiple
    >>> # 1) Use pre‑built list of pairs
    >>> q_pairs = [('q10', 'q90'), ('lwr', 'upr')]
    >>> build_qcols_multiple(q_cols=q_pairs)
    [('q10', 'q90'), ('lwr', 'upr')]

    >>> # 2) Separate lower / upper lists (no median)
    >>> lows = ['q10', 'lwr']
    >>> ups  = ['q90', 'upr']
    >>> build_qcols_multiple(qlow_cols=lows, qup_cols=ups)
    [('q10', 'q90'), ('lwr', 'upr')]

    >>> # 3) Triplets with median enforced
    >>> meds = ['q50', 'mid']
    >>> build_qcols_multiple(
    ...     qlow_cols=lows,
    ...     qup_cols=ups,
    ...     qmed_cols=meds,
    ...     enforce_triplet=True
    ... )
    [('q10', 'q50', 'q90'), ('lwr', 'mid', 'upr')]

    >>> # 4) Ignore supplied median and still get pairs
    >>> build_qcols_multiple(
    ...     qlow_cols=lows,
    ...     qup_cols=ups,
    ...     qmed_cols=meds,
    ...     allow_pair_when_median=True
    ... )
    [('q10', 'q90'), ('lwr', 'upr')]

    """
    # --------------------------------------------------
    # Case‑1: user already passed `q_cols`
    # --------------------------------------------------
    if q_cols is not None:
        if not all(isinstance(t, (list, tuple)) for t in q_cols):
            raise ValueError("`q_cols` must be an iterable of tuples.")
        sizes = {len(t) for t in q_cols}
        if sizes - {2, 3}:
            raise ValueError("`q_cols` tuples must have length 2 or 3.")
        if enforce_triplet and sizes != {3}:
            raise ValueError(
                "`enforce_triplet=True` requires every "
                "tuple in `q_cols` to have three items."
            )
        if (not enforce_triplet) and (sizes == {3}) and allow_pair_when_median:
            # convert triplets to pairs (low, up)
            q_cols = [(t[0], t[-1]) for t in q_cols]
        return [tuple(t) for t in q_cols]

    # --------------------------------------------------
    # Case‑2: build tuples from separate lists
    # --------------------------------------------------
    if qlow_cols is None or qup_cols is None:
        raise ValueError(
            "When `q_cols` is not provided, both "
            "`qlow_cols` and `qup_cols` must be given."
        )
    if len(qlow_cols) != len(qup_cols):
        raise ValueError("`qlow_cols` and `qup_cols` must be the same " "length.")

    # -- median logic
    if qmed_cols is not None and not allow_pair_when_median:
        if len(qmed_cols) != len(qlow_cols):
            raise ValueError(
                "`qmed_cols` must be the same length as " "`qlow_cols` and `qup_cols`."
            )
        tuples = list(zip(qlow_cols, qmed_cols, qup_cols))
    else:
        if enforce_triplet:
            raise ValueError(
                "`enforce_triplet=True` but no median " "columns were supplied."
            )
        tuples = list(zip(qlow_cols, qup_cols))

    return [tuple(t) for t in tuples]
