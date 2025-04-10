# -*- coding: utf-8 -*-

import re 
import inspect
import warnings
from typing import ( 
    Any,
    Dict, 
    Optional, 
    List 
)
import numpy as np 

def str2columns(
    text: str,
    regex: Optional[re.Pattern] = None,
    pattern: Optional[str] = None
) -> List[str]:
    """
    Split the input string `<text>` into words or
    column names using a regular expression.

    By default, if both `<regex>` and `<pattern>` are
    None, returns `[text]`. If `<pattern>` is given,
    compiles it into a regex to split `<text>`. If
    `<regex>` is provided, uses it directly.

    Parameters
    ----------
    text : str
        The string to split into words/columns.
    regex : re.Pattern, optional
        Precompiled regular expression used for
        splitting the text. If provided, overrides
        `<pattern>`.
    pattern : str, optional
        Regex pattern to compile if `<regex>` is not
        given. Defaults to
        ``r'[#&.*@!_,;\s-]\s*'`` if only `pattern`
        is used.
    Returns
    -------
    List[str]
        List of tokens from `<text>`, split
        according to the pattern or `<regex>`.

    Examples
    --------
    >>> from gofast.core.checks import str2columns
    >>> text = "this.is an-example"
    >>> str2columns(text)
    ['this','is','an','example']
    """
    # If no regex or pattern is provided,
    # just wrap the entire text in a list
    if regex is None and pattern is None:
        return [text]

    # If the user provided a compiled regex,
    # we use it directly
    if regex is not None:
        splitter = regex
    else:
        # Otherwise compile from <pattern>
        splitter = re.compile(pattern, flags=re.IGNORECASE)

    # Split and filter out empty parts
    parts = splitter.split(text)
    return list(filter(None, parts))

def smart_format(
    iter_obj,
    choice: str = 'and'
) -> str:
    """
    Smartly format an iterable object into a readable
    string with a specific connector (e.g. `'and'`).

    Parameters
    ----------
    iter_obj : iterable
        The iterable to format. If it is not truly
        iterable, it is returned as a string
        representation.
    choice : str, default='and'
        The connector word between the second last
        and last items (e.g. `'and'`, `'or'`).

    Returns
    -------
    str
        A user-friendly string representation of
        `<iter_obj>`, e.g. '"foo", "bar" and "baz"'.

    Examples
    --------
    >>> _smart_format(['apple', 'banana', 'cherry'])
    '"apple","banana" and "cherry"'
    >>> _smart_format(['apple'])
    '"apple"'
    >>> _smart_format('banana')
    'banana'
    """
    # Attempt to ensure it's iterable
    try:
        _ = iter(iter_obj)
    except: # TypeError >
        return f"{iter_obj}"

    # Convert each element to string
    items = [str(obj) for obj in iter_obj]
    if not items:
        return ""

    if len(items) == 1:
        return ','.join([f"{i!r}" for i in items])

    # Multiple items: join all but last with commas,
    # then add the connector word and final item
    body = ','.join([f"{i!r}" for i in items[:-1]])
    return f"{body} {choice} {items[-1]!r}"

def count_functions(
    module_name, 
    include_class=False, 
    return_counts=True, 
    include_private=False, 
    include_local=False
    ):
    """
    Count and list the number of functions and classes in a specified module.

    Parameters
    ----------
    module_name : str
        The name of the module to inspect, in the format `package.module`.
    include_class : bool, optional
        Whether to include classes in the count and listing. Default is 
        `False`.
    return_counts : bool, optional
        Whether to return only the count of functions and classes (if 
        ``include_class`` is `True`). If `False`, returns a list of functions 
        and classes in alphabetical order. Default is `True`.
    include_private : bool, optional
        Whether to include private functions and classes (those starting with 
        `_`). Default is `False`.
    include_local : bool, optional
        Whether to include local (nested) functions in the count and listing. 
        Default is `False`.

    Returns
    -------
    int or list
        If ``return_counts`` is `True`, returns the count of functions and 
        classes (if ``include_class`` is `True`). If ``return_counts`` is 
        `False`, returns a list of function and class names (if 
        ``include_class`` is `True`) in alphabetical order.

    Notes
    -----
    This function dynamically imports the specified module and analyzes its 
    Abstract Syntax Tree (AST) to count and list functions and classes. It 
    provides flexibility to include or exclude private and local functions 
    based on the parameters provided.

    The process can be summarized as:

    .. math::
        \text{total\_count} = 
        \text{len(functions)} + \text{len(classes)}

    where:

    - :math:`\text{functions}` is the list of functions found in the module.
    - :math:`\text{classes}` is the list of classes found in the module 
      (if ``include_class`` is `True`).

    Examples
    --------
    >>> from gofast.api.util import count_functions_classes
    >>> count_functions_classes('gofast.api.util', include_class=True,
                                return_counts=True)
    10

    >>> count_functions('gofast.api.util', include_class=True,
                                return_counts=False)
    ['ClassA', 'ClassB', 'func1', 'func2', 'func3']

    >>> count_functions('gofast.api.util', include_class=False, 
                                return_counts=True, include_private=True)
    15

    >>> count_functions('gofast.api.util', include_class=False, 
                                return_counts=False, include_private=True)
    ['_private_func1', '_private_func2', 'func1', 'func2']

    See Also
    --------
    ast : Abstract Syntax Tree (AST) module for parsing Python source code.

    References
    ----------
    .. [1] Python Software Foundation. Python Language Reference, version 3.9. 
       Available at http://www.python.org
    .. [2] Python `ast` module documentation. Available at 
       https://docs.python.org/3/library/ast.html
    """
 
    try:
        import ast
    except ImportError as e:  # Catch the specific ImportError exception
        raise ImportError(
            "The 'ast' module could not be imported. This module is essential"
            " for analyzing Python source code to count functions and classes."
            " Ensure that you are using a standard Python distribution, which"
            " includes the 'ast' module by default."
        ) from e
    
    import importlib
    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Get the source code of the module
    source = inspect.getsource(module)

    # Parse the source code into an AST
    tree = ast.parse(source)

    # Initialize lists to store function and class names
    functions = []
    classes = []

    def is_local_function(node):
        """Determine if the function is local (nested)."""
        while node:
            if isinstance(node, ast.FunctionDef):
                return True
            node = getattr(node, 'parent', None)
        return False

    # Add parent references to each node
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    # Traverse the AST to find function and class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if (include_private or not node.name.startswith('_')) and \
               (include_local or not is_local_function(node.parent)):
                functions.append(node.name)
        elif isinstance(node, ast.ClassDef) and include_class:
            if include_private or not node.name.startswith('_'):
                classes.append(node.name)

    # Combine and sort the lists if needed
    if include_class:
        result = sorted(functions + classes)
    else:
        result = sorted(functions)

    if return_counts:
        return len(result)
    else:
        return result
    
def drop_nan_in(y_true, *y_preds, error="raise", nan_policy=None):
    """
    Drop NaN values from `y_true` and corresponding predictions in `y_preds`.
    
    This function filters out the samples where `y_true` contains NaN values,
    and also removes the corresponding entries from each predicted value array
    provided in `y_preds`. The resulting arrays have the same length as the
    number of non-NaN values in `y_true`.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        The true values. Must be numeric, one-dimensional, and of the same length 
        as the arrays in `y_preds`.

    y_preds : array-like (one or more), shape (n_samples,)
        Predicted values from one or more models. Each `y_pred` must have the same
        length as `y_true`. Multiple predicted arrays can be passed, and each 
        will be filtered based on the non-NaN values in `y_true`.

    error : {"raise", "warn", "ignore"}, default="raise"
        Defines the action to take when NaN values are found in `y_true`.
        - "raise" : Raises a ValueError (default behavior).
        - "warn" : Issues a warning but proceeds.
        - "ignore" : Silently ignores the NaN values.
        
    nan_policy : {"raise", "propagate", "omit"}, default=None
        Defines the behavior for handling NaNs:
        - "raise" : Raises an error if NaN is encountered.
        - "propagate" : Propagates NaNs without raising an error (default).
        - "omit" : Omits NaN values (similar to removing them).
        
    Returns
    -------
    y_true : array-like, shape (n_samples,)
        The filtered true values with NaNs removed.

    y_preds : tuple of array-like, shape (n_samples,)
        A tuple of the filtered predicted values, one for each model passed as `y_preds`.
        Each array has the same number of non-NaN entries as `y_true`.
        
    Notes
    -----
    The function ensures that NaN values in `y_true` are dropped, and corresponding 
    entries in all predicted arrays are also removed, maintaining alignment.
    
    If `nan_policy` is set, the `error` parameter will not take effect. When `nan_policy` 
    is None, the behavior is controlled by the `error` parameter.
    
    Example
    -------
    >>> y_true = [1, 2, np.nan, 4]
    >>> y_pred1 = [0.9, 1.8, 3.1, 4.2]
    >>> y_pred2 = [1.1, 1.9, 3.0, 4.0]
    >>> _drop_nan_in(y_true, y_pred1=y_pred1, y_pred2=y_pred2, error="warn")
    ([1, 2, 4], ([0.9, 1.8, 4.2], [1.1, 1.9, 4.0]))
    """
    # Check for NaNs in y_true
    not_nan_indices = ~np.isnan(y_true)
    
    # Handle error or nan_policy logic
    if np.any(np.isnan(y_true)) and error == "raise":
        raise ValueError("NaN values found in y_true, cannot proceed with NaN values.")
    elif np.any(np.isnan(y_true)) and error == "warn":

        warnings.warn("NaN values found in y_true, they will be ignored.")
    elif np.any(np.isnan(y_true)) and error == "ignore":
        pass  # Silently ignore NaNs in y_true
    
    # Apply nan_policy if set
    if nan_policy == "raise" and np.any(np.isnan(y_true)):
        raise ValueError("NaN values found in y_true with nan_policy='raise'.")
    elif nan_policy == "omit":
        not_nan_indices = ~np.isnan(y_true)  # Continue to omit NaNs
    
    # Filter y_true and all y_preds based on the non-NaN indices
    y_true = y_true[not_nan_indices]
    y_preds = tuple(y_pred[not_nan_indices] for y_pred in y_preds)

    return y_true, *y_preds

def get_valid_kwargs(
        callable_obj: Any, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
    """
    Filter and return only the valid keyword arguments for a given 
    callable object, while warning about any invalid kwargs.
    
    Parameters
    ----------
    callable_obj : callable
        The callable object (function, lambda function, method, or class) 
        for which the keyword arguments need to be validated.
    
    kwargs : dict
        Dictionary of keyword arguments to be validated against the callable object.
    
    Returns
    -------
    valid_kwargs : dict
        Dictionary containing only the valid keyword arguments 
        for the callable object.
    """
    # If the callable_obj is an instance, get its class
    if not inspect.isclass(callable_obj) and not callable(callable_obj):
        callable_obj = callable_obj.__class__
    
    try:
        # Retrieve the signature of the callable object
        signature = inspect.signature(callable_obj)
    except ValueError:
        # If signature cannot be obtained, return empty kwargs and warn
        warnings.warn(
            "Unable to retrieve signature of the callable object. "
            "No keyword arguments will be passed."
        )
        return {}
    
    # Extract parameter names from the function signature
    valid_params = set(signature.parameters.keys())
    
    # Identify valid and invalid kwargs
    valid_kwargs = {}
    invalid_kwargs = {}
    for k, v in kwargs.items():
        if k in valid_params:
            valid_kwargs[k] = v
        else:
            invalid_kwargs[k] = v
    
    # Warn the user about invalid kwargs
    if invalid_kwargs:
        invalid_keys = ', '.join(invalid_kwargs.keys())
        warnings.warn(
            f"The following keyword arguments are invalid"
            f" and will be ignored: {invalid_keys}"
        )
    
    return valid_kwargs

def error_policy(
    error: str | None,
    *,
    policy: str = 'auto',
    base: str = 'ignore',
    exception: type[Exception] = None,
    msg: str | None = None, 
    valid_policies: set =None 
) -> str:
    """
    Manage error-handling policies like 'warn', 'raise', or 'ignore'.

    The `error_policy` function determines how to handle potential
    errors by mapping the user-provided ``error`` argument to a valid
    policy. It helps standardize responses such as warnings, raised
    exceptions, or silent ignores. The function can adapt to different
    modes, allowing for strict or flexible behavior depending on the
    ``policy`` and ``base`` settings.

    Parameters
    ----------
    error : str or None
        The user-provided error setting. Can be `'warn'`, `'raise'`,
        `'ignore'`, or `None`. If `None`, the behavior is resolved
        based on ``policy`` and ``base``.
    
    policy : str, default='auto'
        Determines how to interpret a `None` error setting. Valid
        options:
        
        - `'auto'`: Resolve `None` to the default `base` policy.
        - `'strict'`: Disallows `None` for `error`; raises an error
          if encountered.
        - `None`: Defers strictly to `base`.

    base : str, default='ignore'
        The fallback error policy when `None` is encountered and
        `policy='auto'` or `policy=None`. Must be one of `'warn'`,
        `'raise'`, or `'ignore'`.

    exception : type of Exception, default=ValueError
        The exception class to be raised if an invalid policy or
        error is encountered.

    msg : str, optional
        A custom message for the raised exception if an invalid
        `error` or `policy` is detected. If omitted, a default is
        used.

    Returns
    -------
    str
        A valid error policy: one of `'warn'`, `'raise'`, or
        `'ignore'`.

    Raises
    ------
    ValueError
        If `policy` is invalid or if `None` is not permitted by
        `policy='strict'` but is used. Also raised if `error` cannot
        be resolved to a valid policy or if `base` is invalid when
        `policy='auto'`.

    Notes
    -----
    - If `error` is already a valid policy (`'warn'`, `'raise'`,
      `'ignore'`), it is returned immediately.
    - When `error=None`, the behavior depends on the `policy` and
      `base` parameters. Setting `policy='strict'` disallows `None`
      for `error`.
      
      
    .. math::
       \\text{error\\_policy}:
       \\begin{cases}
         \\text{'warn'}, & \\text{issue a warning} \\\\
         \\text{'raise'}, & \\text{raise an exception} \\\\
         \\text{'ignore'}, & \\text{do nothing}
       \\end{cases}


    Examples
    --------
    >>> from gofast.core.utils import error_policy
    >>> # Basic usage:
    >>> resolved_error = error_policy('warn')
    >>> print(resolved_error)
    'warn'
    
    >>> # Using 'auto' policy with a default base of 'ignore'
    >>> resolved_error = error_policy(None, policy='auto',
    ...                                base='warn')
    >>> print(resolved_error)
    'warn'
    
    >>> # Strict policy disallows None
    >>> error_policy(None, policy='strict')
    ValueError: In strict policy, `None` is not acceptable as error.

    See Also
    --------
    gofast.utils.validator.validate_nan_policy : A function that
        validate NaN policies.
    """  # noqa: E501

    # Predefined valid policies.
    valid_policies = valid_policies or {'warn', 'raise', 'ignore'}

    # Default message if none is provided.
    default_msg = (
        "Invalid error policy: '{error}'. Valid options are "
        f"{valid_policies}."
    )
    if exception is None:
        exception = ValueError

    # Use custom message or default if not supplied.
    msg = msg or default_msg

    # Validate the `policy` argument.
    if policy not in {'auto', 'strict', None}:
        raise ValueError(
            f"Invalid policy: '{policy}'. Valid options are "
            "'auto', 'strict', or None."
        )

    # Resolve None values for `error` according to `policy`.
    if error is None:
        if policy == 'auto':
            # If policy='auto', fallback to `base` if no override is set.
            error = base or 'ignore'
        elif policy == 'strict':
            # If policy='strict', disallow None for `error`.
            raise ValueError(
                "In strict policy, `None` is not acceptable as an "
                "error. Please set `error` explicitly or switch "
                "policy to 'auto'."
            )
        else:
            # policy=None means strictly use `base` for resolution.
            if base not in valid_policies:
                raise ValueError(
                    f"Invalid base policy: '{base}'. Must be one of "
                    f"{valid_policies} when `error` is None and "
                    "policy is None."
                )
            error = base

    # Final check to ensure `error` is valid.
    if error not in valid_policies:
        # Raise the specified exception if the policy is invalid.
        raise exception(msg.format(error=error))

    # Return the resolved error policy.
    return error