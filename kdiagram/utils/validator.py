#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

import numbers
import warnings
from collections.abc import Iterable
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .generic_utils import smart_format, str2columns


def validate_length_range(length_range, sorted_values=True, param_name=None):
    r"""
    Validates the review length range ensuring it's a tuple with two integers
    where the first value is less than the second.

    Parameters:
    ----------
    length_range : tuple
        A tuple containing two values that represent the minimum and maximum
        lengths of reviews.
    sorted_values: bool, default=True
        If True, the function expects the input length range to be sorted in
        ascending order and will automatically sort it if not. If False, the
        input length range is not expected to be sorted, and it will remain
        as provided.
    param_name : str, optional
        The name of the parameter being validated. If None, the default name
        'length_range' will be used in error messages.

    Returns
    -------
    tuple
        The validated length range.

    Raise
    ------
    ValueError
        If the length range does not meet the requirements.

    Examples
    --------
    >>> from kdiagram.utils.validator import validate_length_range
    >>> validate_length_range ( (202, 25) )
    (25, 202)
    >>> validate_length_range ( (202,) )
    ValueError: length_range must be a tuple with two elements.
    """
    param_name = param_name or "length_range"
    if not isinstance(length_range, (list, tuple)) or len(length_range) != 2:
        raise ValueError(f"{param_name} must be a tuple with two elements.")

    min_length, max_length = length_range

    if not all(
        isinstance(x, (float, int, np.integer, np.floating))
        for x in length_range
    ):
        raise ValueError(f"Both elements in {param_name} must be numeric.")

    if sorted_values:
        length_range = tuple(sorted([min_length, max_length]))
        if length_range[0] >= length_range[1]:
            raise ValueError(
                f"The first element in {param_name} must be less than the second."
            )
    else:
        length_range = tuple([min_length, max_length])

    return length_range


def validate_yy(
    y_true,
    y_pred,
    expected_type=None,
    *,
    validation_mode="strict",
    flatten=False,
    allow_2d_pred=False,
):
    r"""
    Validates the shapes and types of actual and predicted target arrays.

    Ensures arrays are compatible for metrics calculation, handling cases
    where predictions might be two-dimensional (e.g., quantiles).

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    expected_type : str, optional
        The expected scikit-learn type of the target ('binary', etc.).
    validation_mode : str, optional
        Validation strictness. Currently, only 'strict' is implemented,
        which requires y_true and y_pred to have the same shape and match the
        expected_type.
    flatten : bool, default=False
        If True, both y_true and y_pred are flattened to 1D arrays.
    allow_2d_pred : bool, default=False
        If True, allows y_pred to be a 2D array (e.g., for quantiles)
        while y_true must be 1D. The number of samples (rows) must
        still be consistent.

    Raises
    ------
    ValueError
        If inputs do not meet the validation criteria.

    Returns
    -------
    tuple
        The validated y_true and y_pred as NumPy arrays.
    """
    from ..compat.sklearn import type_of_target

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if flatten:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    # After potential flattening, check dimensions
    if allow_2d_pred:
        # Special mode for functions where y_pred is a 2D quantile array
        if y_true.ndim != 1:
            raise ValueError(
                f"y_true must be 1D when allow_2d_pred=True, but got "
                f"shape {y_true.shape}."
            )
        if y_pred.ndim != 2:
            raise ValueError(
                f"y_pred must be 2D when allow_2d_pred=True, but got "
                f"shape {y_pred.shape}."
            )
    else:
        # Default mode: expect both arrays to be 1D
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError(
                "Both y_true and y_pred must be 1D arrays. "
                f"Got shapes {y_true.shape} and {y_pred.shape}. "
                "Consider setting `flatten=True`."
            )

    # Check for consistent number of samples (first dimension)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: "
            f"[{y_true.shape[0]}, {y_pred.shape[0]}]"
        )

    if expected_type is not None:
        actual_type_y_true = type_of_target(y_true)
        actual_type_y_pred = type_of_target(y_pred)
        if validation_mode == "strict" and (
            actual_type_y_true != expected_type
            or actual_type_y_pred != expected_type
        ):
            msg = (
                f"Validation failed in strict mode. Expected type '{expected_type}'"
                f" for both y_true and y_pred, but got '{actual_type_y_true}'"
                f" and '{actual_type_y_pred}' respectively."
            )
            raise ValueError(msg)

    return y_true, y_pred


def contains_nested_objects(lst, strict=False, allowed_types=None):
    r"""
    Determines whether a list contains nested objects.

    Parameters
    ----------
    lst : list
        The list to be checked for nested objects.
    strict : bool, optional
        If True, all items in the list must be nested objects. If False, the function
        returns True if any item is a nested object. Default is False.
    allowed_types : tuple of types, optional
        A tuple of types to consider as nested objects. If None, common nested types
        like list, set, dict, and tuple are checked. Default is None.

    Returns
    -------
    bool
        True if the list contains nested objects according to the given parameters,
        otherwise False.

    Notes
    -----
    A nested object is defined as any item within the list that is not a primitive
    data type (e.g., int, float, str) or is a complex structure like lists, sets,
    dictionaries, etc. The function can be customized to check for specific types
    using the `allowed_types` parameter.

    Examples
    --------
    >>> from kdiagram.utils.validator import contains_nested_objects
    >>> example_list1 = [{1, 2}, [3, 4], {'key': 'value'}]
    >>> example_list2 = [1, 2, 3, [4]]
    >>> example_list3 = [1, 2, 3, 4]
    >>> contains_nested_objects(example_list1)
    True  # non-strict, contains nested objects
    >>> contains_nested_objects(example_list1, strict=True)
    True  # strict, all are nested objects
    >>> contains_nested_objects(example_list2)
    True  # non-strict, contains at least one nested object
    >>> contains_nested_objects(example_list2, strict=True)
    False  # strict, not all are nested objects
    >>> contains_nested_objects(example_list3)
    False  # non-strict, no nested objects
    >>> contains_nested_objects(example_list3, strict=True)
    False  # strict, no nested objects
    """
    if allowed_types is None:
        allowed_types = (
            list,
            set,
            dict,
            tuple,
            pd.Series,
            pd.DataFrame,
            np.ndarray,
        )  # Default nested types

    # Function to check if an item is a nested type
    def is_nested(item):
        return isinstance(item, allowed_types)

    if strict:
        # Check if all items are nested objects
        return all(is_nested(item) for item in lst)
    else:
        # Check if any item is a nested object
        return any(is_nested(item) for item in lst)


def check_consistent_length(*arrays):
    r"""Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(le) for le in lengths]
        )


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = f"Expected sequence or array-like, got {type(x)}"
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                f"Singleton array {x!r} cannot be considered a valid collection."
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def is_in_if(
    o: Iterable,
    items: Union[str, Iterable],
    error: str = "raise",
    return_diff: bool = False,
    return_intersect: bool = False,
) -> Union[list, None]:
    r"""
    Assert the Presence of Items within an Iterable Object.
    
    The ``is_in_if`` function verifies whether specified ``items`` exist within 
    an iterable object ``o``. It offers flexibility in handling missing items by 
    allowing users to either raise errors, ignore them, or retrieve differences 
    and intersections based on the provided parameters.
    
    .. math::
        \text{Presence Check} = 
        \begin{cases}
            \text{Raise Error} & \text{if items are missing and error='raise'} \\
            \text{Return Differences} & \text{if return_diff=True} \\
            \text{Return Intersection} & \text{if return_intersect=True}
        \end{cases}
    
    Parameters
    ----------
    o : `Iterable`
        The iterable object in which to check for the presence of ``items``.
    
    items : Union[`str`, `Iterable`]
        The item or collection of items to assert their presence within ``o``.
        If a single string is provided, it is treated as a single-item iterable.
    
    error : `str`, default=`'raise'`
        Determines how the function handles missing items.
        
        - ``'raise'``: Raises a ``ValueError`` if any ``items`` are not found 
          in ``o``.
        - ``'ignore'``: Suppresses errors and allows the function to proceed.
    
    return_diff : `bool`, default=`False`
        If ``True``, returns a list of items that are missing from ``o``.
        When set to ``True``, the ``error`` parameter is automatically set to 
        ``'ignore'``.
    
    return_intersect : `bool`, default=`False`
        If ``True``, returns a list of items that are present in both ``o`` and 
        ``items``.
        When set to ``True``, the ``error`` parameter is automatically set to 
        ``'ignore'``.
    
    Returns
    -------
    Union[List, None]
        - If ``return_diff`` is ``True``, returns a list of missing items.
        - If ``return_intersect`` is ``True``, returns a list of intersecting items.
        - If neither is ``True``, returns ``None`` unless an error is raised.
    
    Raises
    ------
    ValueError
        - If ``error`` is set to ``'raise'`` and any ``items`` are missing in ``o``.
        - If an unsupported value is provided for ``error``.
    
    TypeError
        - If ``o`` is not an iterable object.
    
    Examples
    --------
    >>> from kdiagram.utils.validator import is_in_if
    >>> 
    >>> # Example 1: Check presence with error raising
    >>> o = ['apple', 'banana', 'cherry']
    >>> is_in_if(o, 'banana')
    # No output, validation passed
    >>> is_in_if(o, 'date')
    ValueError: Item 'date' is missing in the list ['apple', 'banana', 'cherry'].
    >>> 
    >>> # Example 2: Check multiple items with some missing
    >>> items = ['banana', 'date']
    >>> is_in_if(o, items)
    ValueError: Items 'date' are missing in the list ['apple', 'banana', 'cherry'].
    >>> 
    >>> # Example 3: Return missing items without raising error
    >>> missing = is_in_if(o, 'date', error='ignore', return_diff=True)
    >>> print(missing)
    ['date']
    >>> 
    >>> # Example 4: Return intersecting items
    >>> intersect = is_in_if(o, ['banana', 'date'], 
    ...                      error='ignore', return_intersect=True)
    >>> print(intersect)
    ['banana']
    
    Notes
    -----
    - **Flexible Input Handling**: The function accepts both single items 
      (as strings) and multiple items (as iterables), providing versatility 
      in usage scenarios.
    
    - **Automatic Error Handling Adjustment**: Setting ``return_diff`` or 
      ``return_intersect`` to ``True`` automatically changes the ``error`` 
      parameter to ``'ignore'`` to facilitate the retrieval of differences 
      or intersections without interruption.
    
    - **Performance Considerations**: For large iterables and item lists, 
      converting them to sets can enhance performance during intersection 
      and difference operations.
    
    See Also
    --------
    list : Built-in Python list type.
    set : Built-in Python set type.
    
    References
    ----------
    .. [1] Python Documentation: set.intersection.  
       https://docs.python.org/3/library/stdtypes.html#set.intersection  
    .. [2] Python Documentation: set.difference.  
       https://docs.python.org/3/library/stdtypes.html#set.difference  
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """

    if error not in {"raise", "warn", "ignore"}:
        raise ValueError("error must be 'raise', 'warn', or 'ignore'")

    if isinstance(items, str):
        items = [items]
    elif not isinstance(o, Iterable):
        raise TypeError(
            f"Expected an iterable object for 'o', got {type(o).__name__!r}."
        )

    # Convert to sets for efficient operations
    set_o = set(o)
    set_items = set(items)

    intersect = list(set_o.intersection(set_items))

    # to make a difference be sure to select the long set
    # Always: items that are not in o
    missing_items = list(set_items.difference(set_o))

    # if len(set_items) >= len(set_o):
    #     missing_items = list(set_items.difference(set_o))
    # else:
    #     missing_items = list(set_o.difference(set_items))

    if return_diff or return_intersect:
        error = "ignore"

    if missing_items:
        formatted_items = ", ".join(f"'{item}'" for item in missing_items)
        verb = "is" if len(missing_items) == 1 else "are"
        msg = (
            f"Item{'' if len(missing_items) == 1 else 's'} {formatted_items} "
            f"{verb} missing in the {type(o).__name__.lower()} {list(o)}."
        )
        if error == "raise":
            raise ValueError(msg)
        elif error == "warn":
            warnings.warn(msg, stacklevel=2)

    if return_diff:
        return missing_items if missing_items else []
    elif return_intersect:
        return intersect if intersect else []

    return None


def exist_features(
    df: pd.DataFrame, features, error="raise", name="Feature"
) -> bool:
    r"""
    Check whether the specified features exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to be checked.
    features : list or str
        List of feature names (str) to check for in the dataframe.
        If a string is provided, it will be treated as a list with
        a single feature.
    error : str, optional, default 'raise'
        Action to take if features are not found. Can be one of:
        - 'raise' (default): Raise a ValueError.
        - 'warn': Issue a warning and return False.
        - 'ignore': Do nothing if features are not found.
    name : str, optional, default 'Feature'
        Name of the feature(s) being checked (default is 'Feature').

    Returns
    -------
    bool
        Returns True if all features exist in the dataframe, otherwise False.

    Raises
    ------
    ValueError
        If 'error' is 'raise' and features are not found.

    Warns
    -----
    UserWarning
        If 'error' is 'warn' and features are missing.

    Notes
    -----
    This function ensures that all the specified features exist in the
    dataframe. If the 'error' parameter is set to 'warn', the function
    will issue a warning instead of raising an error when a feature
    is missing, and return False.

    References
    ----------
    - pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Examples
    --------
    >>> from kdiagram.utils.validator import exist_features
    >>> import pandas as pd

    >>> # Sample DataFrame
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3],
    >>>     'feature2': [4, 5, 6],
    >>>     'feature3': [7, 8, 9]
    >>> })

    >>> # Check for missing features with 'raise' error
    >>> exist_features(df, ['feature1', 'feature4'], error='raise')
    Traceback (most recent call last):
        ...
    ValueError: Features feature4 not found in the dataframe.

    >>> # Check for missing features with 'warn' error
    >>> exist_features(df, ['feature1', 'feature4'], error='warn')
    UserWarning: Features feature4 not found in the dataframe.

    >>> # Check for missing features with 'ignore' error
    >>> exist_features(df, ['feature1', 'feature4'], error='ignore')
    False
    """
    # Validate if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas DataFrame.")

    # Normalize the error parameter to lowercase and strip whitespace
    error = error.lower().strip()

    # Validate the 'error' parameter
    if error not in ["raise", "ignore", "warn"]:
        raise ValueError(
            "Invalid value for 'error'. Expected"
            " one of ['raise', 'ignore', 'warn']."
        )

    # Ensure 'features' is a list-like structure
    if isinstance(features, str):
        features = [features]

    # Validate that 'features' is one of the allowed types
    features = _assert_all_types(features, list, tuple, np.ndarray, pd.Index)

    # Get the intersection of features with the dataframe columns
    existing_features = set(features).intersection(df.columns)

    # If all features exist, return True
    if len(existing_features) == len(features):
        return True

    # Calculate the missing features
    missing_features = set(features) - existing_features

    # If there are missing features, handle according to 'error' type
    if missing_features:
        msg = f"{name}{'s' if len(features) > 1 else ''}"

        if error == "raise":
            raise ValueError(
                f"{msg} {smart_format(missing_features)}"
                " not found in the dataframe."
            )

        elif error == "warn":
            warnings.warn(
                f"{msg} {smart_format(missing_features)}"
                " not found in the dataframe.",
                UserWarning,
                stacklevel=2,
            )
            return False

        # If 'error' is 'ignore', simply return False
        return False

    return True


def _assert_all_types(
    obj: object,
    *expected_objtype: Union[type[Any], tuple[type[Any], ...]],
    objname: Optional[str] = None,
) -> object:
    r"""
    Robust type checking with enhanced error handling and formatting.

    Parameters
    ----------
    obj : object
        Object to validate
    *expected_objtype : type or tuple of types
        Acceptable type(s). Can pass multiple types or tuples of types
    objname : str, optional
        Custom name for object in error messages

    Returns
    -------
    object
        Original object if validation passes

    Raises
    ------
    TypeError
        If type validation fails
    """
    # Flatten nested type specifications
    expected_types = []
    for typ in expected_objtype:
        if isinstance(typ, tuple):
            expected_types.extend(typ)
        else:
            expected_types.append(typ)

    # Convert numpy dtypes to python types
    py_types = []
    for t in expected_types:
        if isinstance(t, type):
            py_types.append(t)
        elif isinstance(t, np.dtype):
            py_types.append(np.dtype(t).type)
        else:
            py_types.append(t)

    expected_types = tuple(py_types)

    if not isinstance(obj, expected_types):
        # Build human-readable type list
        type_names = []
        for t in expected_types:
            try:
                name = t.__name__
            except AttributeError:
                name = str(t)
            type_names.append(name)

        # Format error message components
        obj_name = f"'{objname}'" if objname else "Object"
        plural = "s" if len(expected_types) > 1 else ""
        expected_str = smart_format(type_names)
        actual_type = type(obj).__name__

        raise TypeError(
            f"{obj_name} must be of type{plural} {expected_str}, "
            f"but got type {actual_type!r}"
        )

    return obj


def is_frame(
    arr,
    df_only=False,
    raise_exception=False,
    objname=None,
    error="raise",
):
    r"""
    Check if `arr` is a pandas DataFrame or Series.

    If ``df_only=True``, the function checks strictly for a pandas
    DataFrame. Otherwise, it accepts either a pandas DataFrame or
    Series. This utility is often used to validate input data before
    processing, ensuring that the input conforms to expected types.

    Parameters
    ----------
    arr : object
        The object to examine. Typically a pandas DataFrame or Series,
        but can be any Python object.
    df_only : bool, optional
        If True, only verifies that `arr` is a DataFrame. If False,
        checks for either a DataFrame or a Series. Default is False.
    raise_exception : bool, optional
        If True, this will override `error="raise"`. This parameter
        is deprecated and will be removed soon. Default is False.
    error : str, optional
        Determines the action when `arr` is not a valid frame. Can be:
        - ``"raise"``: Raises a TypeError.
        - ``"warn"``: Issues a warning.
        - ``"ignore"``: Does nothing. Default is ``"raise"``.
    objname : str or None, optional
        A custom name used in the error message if `error` is set to
        ``"raise"``. If None, a generic name is used.

    Returns
    -------
    bool
        True if `arr` is a DataFrame or Series (or strictly a DataFrame
        if `df_only=True`), otherwise False.

    Raises
    ------
    TypeError
        If `error="raise"` and `arr` is not a valid frame. The error
        message guides the user to provide the correct type
        (`DataFrame` or `DataFrame or Series`).

    Notes
    -----
    This function does not convert or modify `arr`. It merely checks
    its compatibility with common DataFrame/Series interfaces by
    examining attributes such as `'columns'` or `'name'`. For a
    DataFrame, `arr.columns` should exist, and for a Series, a `'name'`
    attribute is often present. Both DataFrame and Series implement
    `__array__`, making them NumPy array-like.

    Examples
    --------
    >>> import pandas as pd
    >>> from kdiagram.utils.validator import is_frame

    >>> df = pd.DataFrame({'A': [1,2,3]})
    >>> is_frame(df)
    True

    >>> s = pd.Series([4,5,6], name='S')
    >>> is_frame(s)
    True

    >>> is_frame(s, df_only=True)
    False

    If `error="raise"`:

    >>> is_frame(s, df_only=True, error="raise", objname='Input')
    Traceback (most recent call last):
        ...
    TypeError: 'Input' parameter expects a DataFrame. Got 'Series'
    """

    # Handle deprecation for `raise_exception`
    if raise_exception and error != "raise":
        warnings.warn(
            "'raise_exception' is deprecated and will be replaced by 'error'."
            " The 'error' parameter is now used for specifying error handling.",
            stacklevel=2,
            category=DeprecationWarning,
        )
        error = "raise"  # Fall back to 'raise' if raise_exception is True

    # Determine if arr qualifies as a frame based on df_only
    if df_only:
        obj_is_frame = hasattr(arr, "__array__") and hasattr(arr, "columns")
    else:
        obj_is_frame = hasattr(arr, "__array__") and (
            hasattr(arr, "name") or hasattr(arr, "columns")
        )

    # If not valid and error is set to 'raise', raise TypeError
    if not obj_is_frame:
        if error == "raise":
            objname = objname or "Input"
            objname = f"{objname!r} parameter expects"
            expected = "a DataFrame" if df_only else "a DataFrame or Series"
            raise TypeError(
                f"{objname} {expected}. Got {type(arr).__name__!r}"
            )
        elif error == "warn":
            warning_msg = (
                f"Warning: {objname or 'Input'} expects "
                f"a DataFrame or Series. Got {type(arr).__name__!r}."
            )
            warnings.warn(warning_msg, stacklevel=2, category=UserWarning)

    return obj_is_frame


def build_data_if(
    data,
    columns=None,
    to_frame=True,
    input_name="data",
    col_prefix="col_",
    force=False,
    error="warn",
    coerce_datetime=False,
    coerce_numeric=True,
    start_incr_at=0,
    **kw,
):
    r"""
    Validates and converts ``data`` into a pandas DataFrame
    if requested, optionally enforcing consistent column
    naming. Intended to standardize data structures for
    downstream analysis.

    See more in :func:`gofast.utils.data_utils.build_df` for
    documentation details.

    """

    force = True if (force == "auto" and columns is None) else force

    # Attempt to ensure start_incr_at is an integer
    try:
        start_incr_at = int(start_incr_at)
    except (TypeError, ValueError) as err:
        # If the user provided a non-integer, handle it
        # based on the value of `error`
        if error == "raise":
            raise TypeError(
                f"Expected integer for start_incr_at, got "
                f"{type(start_incr_at)} instead."
            ) from err
        elif error == "warn":
            warnings.warn(
                f"Provided 'start_incr_at'={start_incr_at} is not "
                "an integer. Defaulting to 0.",
                UserWarning,
                stacklevel=2,
            )
        # Gracefully default to 0 if error='ignore' or we
        # just want to continue
        start_incr_at = 0

    # Convert from dict to DataFrame if needed. If it's a dict,
    # we can directly create a DataFrame from it
    if isinstance(data, dict):
        data = pd.DataFrame(data)
        # Overwrite columns if they come from dict's keys
        columns = list(data.columns)

    # Convert list or tuple to NumPy array for uniform handling
    elif isinstance(data, (list, tuple)):
        data = np.array(data)

    # If data is a Series, convert it to a DataFrame
    elif isinstance(data, pd.Series):
        data = data.to_frame()

    # Ensure data is 2D by using a helper function
    data = ensure_2d(data)

    # If user wants a DataFrame but we don't have one yet:
    if to_frame and not isinstance(data, pd.DataFrame):
        # If columns are not specified and force=False,
        # we warn or raise accordingly
        if columns is None and not force:
            msg = (
                f"Conversion of '{input_name}' to DataFrame requires "
                "column names. Provide `columns` or set `force=True` to "
                "auto-generate them."
            )
            if error == "raise":
                raise TypeError(msg)
            elif error == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)

        # If forced, generate column names automatically if not given
        if force and columns is None:
            columns = [
                f"{col_prefix}{i + start_incr_at}"
                for i in range(data.shape[1])
            ]

        # Perform final DataFrame conversion
        data = pd.DataFrame(data, columns=columns)

    # Perform an array-to-frame conversion with potential
    # re-checking of columns
    data = array_to_frame(
        data,
        columns=columns,
        to_frame=to_frame,
        input_name=input_name,
        force=force,
    )

    # Optionally apply data-type checks or conversions, like
    # datetime or numeric coercion
    if isinstance(data, pd.DataFrame):
        data = recheck_data_types(
            data,
            coerce_datetime=coerce_datetime,
            coerce_numeric=coerce_numeric,
            return_as_numpy=False,
            column_prefix=col_prefix,
        )

    # Convert integer column names to strings, if needed
    data = _convert_int_columns_to_str(data, col_prefix=col_prefix)

    # Return the final validated and (optionally) converted DataFrame
    return data


def _convert_int_columns_to_str(
    df: pd.DataFrame, col_prefix: Optional[str] = "col_"
) -> pd.DataFrame:
    r"""
    Convert integer columns in a DataFrame to string form,
    optionally adding a prefix.
    """
    # If it's not a DataFrame, just return it as-is
    if not isinstance(df, pd.DataFrame):
        return df

    # Check if every column name is an integer
    if all(isinstance(col, int) for col in df.columns):
        # Copy to avoid mutating the original
        df_converted = df.copy()

        if col_prefix is None:
            # Convert to str without prefix
            df_converted.columns = df_converted.columns.astype(str)
        else:
            # Convert to str with user-provided prefix
            df_converted.columns = [
                f"{col_prefix}{col}" for col in df_converted.columns
            ]
        return df_converted
    else:
        # Return a copy of the original if columns are not all int
        return df.copy()


def recheck_data_types(
    data: Union[pd.DataFrame, pd.Series, list, dict],
    coerce_numeric: bool = True,
    coerce_datetime: bool = True,
    column_prefix: str = "col",
    return_as_numpy: Union[bool, str] = "auto",
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    r"""
    Rechecks and coerces column data types in a DataFrame to the most appropriate
    numeric or datetime types if initially identified as objects. It can also handle
    non-DataFrame inputs by attempting to construct a DataFrame before processing.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, list, or dict
        The data to process. If not a DataFrame, an attempt will be made to convert it.
    coerce_numeric : bool, default=True
        If True, tries to convert object columns to numeric data types.
    coerce_datetime : bool, default=True
        If True, tries to convert object columns to datetime data types.
    column_prefix : str, default="col"
        Prefix for column names when constructing a DataFrame from non-DataFrame input.
    return_as_numpy : bool or str, default="auto"
        If True or "auto", converts the DataFrame to a NumPy array upon returning.
        If "auto", the output type matches the input type.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        The processed data, either as a DataFrame or a NumPy array.

    Examples
    --------
    >>> data = {'a': ['1', '2', '3'], 'b': ['2021-01-01', '2021-02-01', 'not a date'],
                'c': ['1.1', '2.2', '3.3']}
    >>> df = pd.DataFrame(data)
    >>> df = recheck_data_types(df)
    >>> print(df.dtypes)
    a             int64
    b            object  # remains object due to mixed valid and invalid dates
    c           float64
    """
    return_as_numpy = parameter_validator(
        "return_as_numpy", target_strs={"auto", True, False}
    )(return_as_numpy)
    is_frame = True
    if not isinstance(data, pd.DataFrame):
        is_frame = False
        try:
            data = pd.DataFrame(
                data,
                columns=[column_prefix + str(i) for i in range(len(data))],
            )
        except Exception as e:
            raise ValueError(
                "Failed to construct a DataFrame from the provided data. "
                "Ensure that your input data is structured correctly, such as "
                "a list of lists or a dictionary with equal-length lists. "
                "Alternatively, provide a DataFrame directly."
            ) from e

    for column in data.columns:
        if data[column].dtype == "object":
            if coerce_datetime:
                try:
                    data[column] = pd.to_datetime(data[column])
                    continue  # Skip further processing if datetime conversion is successful
                except (TypeError, ValueError):
                    pass  # Continue if datetime conversion fails

            if coerce_numeric:
                try:
                    data[column] = pd.to_numeric(data[column])
                except ValueError:
                    pass  # Keep as object if conversion fails

    if return_as_numpy == "auto" and not is_frame:
        return_as_numpy = (
            True  # Automatically determine if output should be a NumPy array
        )

    if return_as_numpy is True:  # Explicitly set to True since "auto" is True
        return data.to_numpy()

    return data


def array_to_frame(
    X,
    *,
    to_frame=False,
    columns=None,
    raise_exception=False,
    raise_warning=True,
    input_name="",
    force=False,
):
    r"""
    Validates and optionally converts an array-like object to a pandas DataFrame,
    applying specified column names if provided or generating them if the `force`
    parameter is set.

    Parameters
    ----------
    X : array-like
        The array to potentially convert to a DataFrame.
    columns : str or list of str, optional
        The names for the resulting DataFrame columns or the Series name.
    to_frame : bool, default=False
        If True, converts `X` to a DataFrame if it isn't already one.
    input_name : str, default=''
        The name of the input variable, used for error and warning messages.
    raise_warning : bool, default=True
        If True and `to_frame` is True but `columns` are not provided,
        a warning is issued unless `force` is True.
    raise_exception : bool, default=False
        If True, raises an exception when `to_frame` is True but columns
        are not provided and `force` is False.
    force : bool, default=False
        Forces the conversion of `X` to a DataFrame by generating column names
        based on `input_name` if `columns` are not provided.

    Returns
    -------
    pd.DataFrame or pd.Series
        The potentially converted DataFrame or Series, or `X` unchanged.

    Examples
    --------
    >>> from kdiagram.utils.validator import array_to_frame
    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> X = data.data
    >>> array_to_frame(X, to_frame=True, columns=['sepal_length', 'sepal_width',
                                                  'petal_length', 'petal_width'])
    """
    # Determine if conversion to frame is needed
    if to_frame and not isinstance(X, (pd.DataFrame, pd.Series)):
        # Handle force conversion without provided column names
        if columns is None and force:
            columns = [f"{input_name}_{i}" for i in range(X.shape[1])]
        elif columns is None:
            msg = (
                f"Array '{input_name}' requires column names for conversion to a DataFrame. "
                "Provide `columns` or set `force=True` to auto-generate column names."
            )
            if raise_exception:
                raise ValueError(msg)
            if raise_warning and raise_warning not in (
                "silence",
                "ignore",
                "mute",
            ):
                warnings.warn(msg, stacklevel=2)
            return X  # Early return if no columns and not forcing

        # Proceed with conversion using the provided or generated column names
        X, _ = convert_array_to_pandas(
            X, to_frame=True, columns=columns, input_name=input_name
        )

    return X


def convert_array_to_pandas(
    X, *, to_frame=False, columns=None, input_name="X"
):
    r"""
    Converts an array-like object to a pandas DataFrame or Series, applying
    provided column names or series name.

    Parameters
    ----------
    X : array-like
        The array to convert to a DataFrame or Series.
    to_frame : bool, default=False
        If True, converts the array to a DataFrame. Otherwise, returns the array unchanged.
    columns : str or list of str, optional
        Name(s) for the columns of the resulting DataFrame or the name of the Series.
    input_name : str, default='X'
        The name of the input variable; used in constructing error messages.

    Returns
    -------
    pd.DataFrame or pd.Series
        The converted DataFrame or Series. If `to_frame` is False, returns `X` unchanged.
    columns : str or list of str
        The column names of the DataFrame or the name of the Series, if applicable.

    Raises
    ------
    TypeError
        If `X` is not array-like or if `columns` is neither a string nor a list of strings.
    ValueError
        If the conversion to DataFrame is requested but `columns` is not provided,
        or if the length of `columns` does not match the number of columns in `X`.
    """
    # Check if the input is string, which is a common mistake
    if isinstance(X, str):
        raise TypeError(
            f"The parameter '{input_name}' should be an array-like"
            " or sparse matrix, but a string was passed."
        )

    # Validate the type of X
    if not (
        hasattr(X, "__array__")
        or isinstance(X, (np.ndarray, pd.Series, list))
        or sp.issparse(X)
    ):
        raise TypeError(
            f"The parameter '{input_name}' should be array-like"
            f" or a sparse matrix. Received: {type(X).__name__!r}"
        )

    # Preserve existing DataFrame or Series column names
    if hasattr(X, "columns"):
        columns = X.columns
    elif hasattr(X, "name"):
        columns = X.name

    if to_frame and not sp.issparse(X):
        if columns is None:
            raise ValueError(
                "Columns must be provided for DataFrame conversion."
            )

        # Ensure columns is list-like for DataFrame conversion, single string for Series
        if isinstance(columns, str):
            columns = [columns]

        if not hasattr(columns, "__len__") or isinstance(columns, str):
            raise TypeError(
                f"Columns for {input_name} must be a list or a single string."
            )

        # Convert to Series or DataFrame based on dimensionality
        if (
            X.ndim == 1 or len(X) == len(columns) == 1
        ):  # 1D array or single-column DataFrame
            X = pd.Series(X, name=columns[0])
        elif X.ndim == 2:  # 2D array to DataFrame
            if X.shape[1] != len(columns):
                raise ValueError(
                    f"Shape of passed values is {X.shape},"
                    f" but columns implied {len(columns)}"
                )
            X = pd.DataFrame(X, columns=columns)
        else:
            raise ValueError(
                f"{input_name} cannot be converted to DataFrame with given columns."
            )

    return X, columns


def ensure_2d(X, output_format="auto"):
    r"""
    Ensure that the input X is converted to a 2-dimensional structure.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        The input data to convert. Can be a list, numpy array, or DataFrame.
    output_format : str, optional
        The format of the returned object. Options are "auto", "array", or "frame".
        "auto" returns a DataFrame if X is a DataFrame, otherwise a numpy array.
        "array" always returns a numpy array.
        "frame" always returns a pandas DataFrame.

    Returns
    -------
    ndarray or DataFrame
        The converted 2-dimensional structure, either as a numpy array or DataFrame.

    Raises
    ------
    ValueError
        If the `output_format` is not one of the allowed values.

    Examples
    --------
    >>> import numpy as np
    >>> from kdiagram.utils.validator import ensure_2d
    >>> X = np.array([1, 2, 3])
    >>> ensure_2d(X, output_format="array")
    array([[1],
           [2],
           [3]])
    >>> df = pd.DataFrame([1, 2, 3])
    >>> ensure_2d(df, output_format="frame")
       0
    0  1
    1  2
    2  3
    """
    # Check for allowed output_format values
    output_format = parameter_validator(
        "output_format", target_strs=["auto", "array", "frame"]
    )(output_format)

    # Detect if the input is a DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)

    # Ensure X is at least 2-dimensional
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X[:, np.newaxis]
    elif isinstance(X, pd.DataFrame):
        if X.shape[1] == 0:  # Implies an empty DataFrame or misshapen
            X = X.values.reshape(-1, 1)  # reshape and handle as array
            is_dataframe = False
    else:
        X = np.array(X)  # Convert other types like lists to np.array
        if X.ndim == 1:
            X = X[:, np.newaxis]

    # Decide on return type based on output_format
    if output_format == "array":
        return X if isinstance(X, np.ndarray) else X.values
    elif output_format == "frame":
        return pd.DataFrame(X) if not is_dataframe else X
    else:  # auto handling
        if is_dataframe:
            return X
        return pd.DataFrame(X) if is_dataframe else X


def parameter_validator(
    param_name,
    target_strs,
    match_method="contains",
    raise_exception=True,
    **kws,
):
    r"""
    Creates a validator function for ensuring a parameter's value matches one
    of the allowed target strings, optionally applying normalization.

    This higher-order function returns a validator that can be used to check
    if a given parameter value matches allowed criteria, optionally raising
    an exception or normalizing the input.

    Parameters
    ----------
    param_name : str
        Name of the parameter to be validated. Used in error messages to
        indicate which parameter failed validation.
    target_strs : list of str
        A list of acceptable string values for the parameter.
    match_method : str, optional
        The method used to match the input string against the target strings.
        The default method is 'contains', which checks if the input string
        contains any of the target strings.
    raise_exception : bool, optional
        Specifies whether an exception should be raised if validation fails.
        Defaults to True, raising an exception on failure.
    **kws: dict,
       Keyword arguments passed to :func:`gofast.core.utils.normalize_string`.
    Returns
    -------
    function
        A closure that takes a single string argument (the parameter value)
        and returns a normalized version of it if the parameter matches the
        target criteria. If the parameter does not match and `raise_exception`
        is True, it raises an exception; otherwise, it returns the original value.

    Examples
    --------
    >>> from kdiagram.utils.validator import parameter_validator
    >>> validate_outlier_method = parameter_validator(
    ...  'outlier_method', ['z_score', 'iqr'])
    >>> outlier_method = "z_score"
    >>> print(validate_outlier_method(outlier_method))
    'z_score'

    >>> validate_fill_missing = parameter_validator(
    ...  'fill_missing', ['median', 'mean', 'mode'], raise_exception=False)
    >>> fill_missing = "average"  # This does not match but won't raise an exception.
    >>> print(validate_fill_missing(fill_missing))
    'average'

    Notes
    -----
    - The function leverages a custom utility function `normalize_string`
      from a module named `gofast.core.utils`. This utility is assumed to handle
      string normalization and matching based on the provided `match_method`.
    - If `raise_exception` is set to False and the input does not match any
      target string, the input string is returned unchanged. This behavior
      allows for optional enforcement of the validation rules.
    - The primary use case for this function is to validate and optionally
      normalize parameters for configuration settings or function arguments
      where only specific values are allowed.
    """

    def validator(param_value):
        """Validate param value from :func:`~normalize_string`"""
        if param_value:
            return normalize_string(
                param_value,
                target_strs=target_strs,
                return_target_only=True,
                match_method=match_method,
                raise_exception=raise_exception,
                **kws,
            )
        return param_value  # Return the original value if it's None or empty

    return validator


def normalize_string(
    input_str: str,
    target_strs: Optional[list[str]] = None,
    num_chars_check: Optional[int] = None,
    deep: bool = False,
    return_target_str: bool = False,
    return_target_only: bool = False,
    raise_exception: bool = False,
    ignore_case: bool = True,
    match_method: str = "exact",
    error_msg: str = None,
) -> Union[str, tuple[str, Optional[str]]]:
    r"""
    Normalizes a string by applying various transformations and optionally checks
    against a list of target strings based on different matching methods.

    Function normalizes a string by stripping leading/trailing whitespace,
    converting to lowercase,and optionally checks against a list of target
    strings. If specified, returns the target string that matches the
    conditions. Raise an exception if the string is not found.

    Parameters
    ----------
    input_str : str
        The string to be normalized.
    target_strs : List[str], optional
        A list of target strings for comparison.
    num_chars_check : int, optional
        The number of characters at the start of the string to check
        against each target string.
    deep : bool, optional
        If True, performs a deep substring check within each target string.
    return_target_str : bool, optional
        If True and a target string matches, returns the matched target string
        along with the normalized string.
    return_target_only: bool, optional
       If True and a target string  matches, returns only the matched string
       target.
    raise_exception : bool, optional
        If True and the input string is not found in the target strings,
        raises an exception.
    ignore_case : bool, optional
        If True, ignores case in string comparisons. Default is True.
    match_method : str, optional
        The string matching method: 'exact', 'contains', or 'startswith'.
        Default is 'exact'.
    error_msg: str, optional,
       Message to raise if `raise_exception` is ``True``.

    Returns
    -------
    Union[str, Tuple[str, Optional[str]]]
        The normalized string. If return_target_str is True and a target
        string matches, returns a tuple of the normalized string and the
        matched target string.

    Raises
    ------
    ValueError
        If raise_exception is True and the input string is not found in
        the target strings.

    Examples
    --------
    >>> from gofast.core.utils import normalize_string
    >>> normalize_string("Hello World", target_strs=["hello", "world"], ignore_case=True)
    'hello world'
    >>> normalize_string("Goodbye World", target_strs=["hello", "goodbye"],
                         num_chars_check=7, return_target_str=True)
    ('goodbye world', 'goodbye')
    >>> normalize_string("Hello Universe", target_strs=["hello", "world"],
                         raise_exception=True)
    ValueError: Input string not found in target strings.
    """
    normalized_str = str(input_str).lower() if ignore_case else input_str

    if not target_strs:
        return normalized_str
    target_strs = is_iterable(
        target_strs, exclude_string=True, transform=True
    )
    normalized_targets = (
        [str(t).lower() for t in target_strs] if ignore_case else target_strs
    )
    matched_target = None

    for target in normalized_targets:
        if num_chars_check is not None:
            condition = (
                normalized_str[:num_chars_check] == target[:num_chars_check]
            )
        elif deep:
            condition = normalized_str in target
        elif match_method == "contains":
            condition = target in normalized_str
        elif match_method == "startswith":
            condition = normalized_str.startswith(target)
        else:  # Exact match
            condition = normalized_str == target

        if condition:
            matched_target = target
            break

    if matched_target is not None:
        if return_target_only:
            return matched_target
        return (
            (normalized_str, matched_target)
            if return_target_str
            else normalized_str
        )

    if raise_exception:
        error_msg = error_msg or (
            f"Invalid input. Expect {smart_format(target_strs, 'or')}."
            f" Got {input_str!r}."
        )
        raise ValueError(error_msg)

    if return_target_only:
        return matched_target

    return ("", None) if return_target_str else ""


def is_iterable(
    y,
    exclude_string: bool = False,
    transform: bool = False,
    parse_string: bool = False,
    delimiter: str = r"[ ,;|\t\n]+",
) -> Union[bool, list]:
    r"""
    Asserts whether `<y>` is iterable and optionally transforms
    `<y>` into a list or parses it as columns if it is a string.

    If `<exclude_string>` is True and `<y>` is a string, the
    function returns `False` for the iterability check.
    If `<transform>` is True, the function returns `<y>` as-is
    if already iterable or wraps `<y>` in a list. If
    `<parse_string>` is True (and `<transform>` is also True),
    a string input is split into columns via `str2columns`.

    Parameters
    ----------
    y : any
        Object to evaluate for iterability or transform
        into an iterable.
    exclude_string : bool, default=False
        If True, treats any string `<y>` as non-iterable.
    transform : bool, default=False
        If True, transforms `<y>` into an iterable if
        not already one. By default, wraps `<y>` into
        a list.
    parse_string : bool, default=False
        If True and `<y>` is a string, attempts to parse
        using `str2columns`. Requires `<transform>` = True.

    Returns
    -------
    bool or list
        If `<transform>` is False, returns a boolean
        indicating whether `<y>` is considered iterable.
        If `<transform>` is True, returns either `<y>` (if
        it is already iterable) or `[y]`. If `<parse_string>`
        is also True, a string is split into columns.

    Raises
    ------
    ValueError
        If `<parse_string>` is True but `<transform>` is
        False while `<y>` is a string.

    Examples
    --------
    >>> from kdiagram.utils.validator import is_iterable
    >>> is_iterable('iterable', exclude_string=True)
    False
    >>> is_iterable('iterable', exclude_string=True, transform=True)
    ['iterable']
    >>> is_iterable('parse this', parse_string=True, transform=True)
    ['parse', 'this']
    """
    # If user wants to parse string but not transform,
    # raise error because result wouldn't be an iterable.
    if parse_string and not transform and isinstance(y, str):
        raise ValueError(
            "Cannot parse the given string. Set 'transform' to True "
            "or use 'str2columns' directly."
        )

    # If parse_string is True, convert string to columns
    if isinstance(y, str) and parse_string:
        y = str2columns(y, pattern=delimiter)

    # Check iterability, but optionally treat string
    # objects as non-iterable
    is_iter = not (exclude_string and isinstance(y, str)) and hasattr(
        y, "__iter__"
    )

    # If transform is True, return y as-is if it is
    # iterable, otherwise wrap it in a list.
    if transform:
        return y if is_iter else [y]

    # Otherwise, just return boolean indicating
    # iterability
    return is_iter


def check_spatial_columns(
    df: pd.DataFrame,
    spatial_cols: Optional[tuple] = ("longitude", "latitude"),
) -> None:
    r"""
    Validate the spatial columns in the DataFrame.

    Ensures that the specified `spatial_cols` are present in the DataFrame and
    consist of exactly two columns representing longitude and latitude.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing geographical data.

    spatial_cols : tuple, optional, default=('longitude', 'latitude')
        A tuple containing the names of the longitude and latitude columns.
        Must consist of exactly two elements.

    Raises
    ------
    ValueError
        - If `spatial_cols` is not a tuple or does not contain exactly two elements.
        - If any of the specified `spatial_cols` are not present in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from kdiagram.utils.validator import check_spatial_columns

    >>> # Valid spatial columns
    >>> df = pd.DataFrame({
    ...     'longitude': [-100, -99, -98],
    ...     'latitude': [35, 36, 37],
    ...     'value': [1, 2, 3]
    ... })
    >>> check_spatial_columns(df, spatial_cols=('longitude', 'latitude'))
    # No output, validation passed

    >>> # Invalid spatial columns
    >>> check_spatial_columns(df, spatial_cols=('lon', 'lat'))
    ValueError: The following spatial_cols are not present in the dataframe: {'lat', 'lon'}

    Notes
    -----
    - The function strictly requires `spatial_cols` to contain exactly two
      column names representing longitude and latitude.

    See Also
    --------
    plot_spatial_distribution : Function to plot spatial distributions.

    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Spatial columns check requires a dataframe `df`"
            f" to be set. Got {type(df).__name__!r}"
        )

    if (
        spatial_cols is None
        or not isinstance(spatial_cols, (tuple, list))
        or len(spatial_cols) != 2
    ):
        raise ValueError(
            "spatial_cols must be a tuple of exactly two elements "
            "(longitude and latitude)."
        )

    missing_cols = set(spatial_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"The following spatial_cols are not present in the dataframe: {missing_cols}"
        )
