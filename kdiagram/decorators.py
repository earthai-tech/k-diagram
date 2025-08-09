# -*- coding: utf-8 -*-

from __future__ import print_function
import os 
import inspect 
from functools import wraps
import warnings

import numpy as np
import pandas as pd
from .core.property import PandasDataHandlers 
from .utils.generic_utils import get_valid_kwargs

__all__= ['check_non_emptiness', 'isdf', 'SaveFile']

def check_non_emptiness(
    params=None,
    *,
    error: str = "raise",
    none_as_empty: bool = True,
    ellipsis_as_empty: bool = True,
    include: tuple = ("set", "dict")
):
    """
    Decorator to check for non-emptiness of specified
    parameters in a function. By default, it checks
    array-like structures (lists, tuples, Series,
    DataFrame, NumPy arrays) but can also consider
    sets, dicts, or other types via <include inline>.

    If the function is used without parentheses
    (e.g. ``@check_emptiness``), the first positional
    argument is checked by default. Otherwise, specify
    a list of parameter names in ``params``.

    Parameters
    ----------
    params : list of str, optional
        Names of arguments whose emptiness will be
        checked. If None and the decorator is used
        without parentheses, the first positional
        argument is checked.
    error : str, optional
        How to handle an empty argument:
          - "raise": raise ValueError
          - "warn": issue a warning
          - "ignore": do nothing
        Default "raise".
    none_as_empty : bool, optional
        If True, consider None as empty. Default True.
    ellipsis_as_empty : bool, optional
        If True, consider the Ellipsis object
        (``...``) as empty. Default True.
    include : tuple, optional
        Additional types to treat as potentially
        empty, e.g., ("set", "dict"). Default
        ("set", "dict").

    Returns
    -------
    callable
        The decorated function that checks emptiness
        for the specified arguments.

    Examples
    --------
    
    1) Decorator used without parentheses:
       >>> from kdiagram.decorators import check_non_emptiness
       >>> @check_non_emptiness
       ... def func_first_arg(x):
       ...     return x
       ...
       >>> # Here, if x is empty, handle it according
       ... # to <error inline>.

    2) Specify which parameters to check:
       >>> @check_non_emptiness(params=['df'],
       ...                  error='warn',
       ...                  none_as_empty=True)
       ... def process_data(a, df=None):
       ...     return df

    3) Check multiple parameters:
       >>> @check_non_emptiness(params=['arr', 'df'],
       ...                  error='raise')
       ... def model_fit(arr, df=None, *args):
       ...     return (arr, df)

    Notes
    -----
    1. If <none_as_empty inline> is True and an
       argument is None, it is considered empty.
    2. If <ellipsis_as_empty inline> is True and
       an argument is Ellipsis (``...``), it is
       considered empty.
    3. For additional types, such as <set inline>
       or <dict inline>, set them in <include
       inline>. By default, sets and dicts are
       also checked for emptiness.
    4. By default, standard array-like structures
       (lists, tuples, Series, DataFrames, numpy
       arrays) are checked.

    """
    # Case A: If decorator is used without parentheses,
    # then `params` is actually the function object itself.
    if callable(params):
        func = params

        @wraps(func)
        def _wrapper(*args, **kwargs):
            if args:
                converted_args = list(args)
                converted_args[0] = _check_and_handle_emptiness(
                    converted_args[0],
                    error=error,
                    none_as_empty=none_as_empty,
                    ellipsis_as_empty=ellipsis_as_empty,
                    include=include,
                    param_name="first positional argument"
                )
                args = tuple(converted_args)
            return func(*args, **kwargs)

        return _wrapper

    # Case B: If decorator is used with parentheses
    # then `params` is the list of names or None.
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            valid_params = params or []

            # If no params were specified, check the
            # first positional argument (if present).
            if not valid_params:
                if args:
                    converted_args = list(args)
                    converted_args[0] = _check_and_handle_emptiness(
                        converted_args[0],
                        error=error,
                        none_as_empty=none_as_empty,
                        ellipsis_as_empty=ellipsis_as_empty,
                        include=include,
                        param_name="first positional argument"
                    )
                    args = tuple(converted_args)
            else:
                # For each name in valid_params,
                # check emptiness.
                sig = inspect.signature(func)
                parameters = list(sig.parameters.keys())

                for name in valid_params:
                    if name in kwargs:
                        kwargs[name] = _check_and_handle_emptiness(
                            kwargs[name],
                            error=error,
                            none_as_empty=none_as_empty,
                            ellipsis_as_empty=ellipsis_as_empty,
                            include=include,
                            param_name=name
                        )
                    else:
                        # Possibly positional
                        try:
                            idx = parameters.index(name)
                            if idx < len(args):
                                converted_args = list(args)
                                converted_args[idx] = _check_and_handle_emptiness(
                                    converted_args[idx],
                                    error=error,
                                    none_as_empty=none_as_empty,
                                    ellipsis_as_empty=ellipsis_as_empty,
                                    include=include,
                                    param_name=name
                                )
                                args = tuple(converted_args)
                        except ValueError:
                            # param not found in function signature
                            pass

            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def _check_and_handle_emptiness(
    value,
    *,
    error: str,
    none_as_empty: bool,
    ellipsis_as_empty: bool,
    include: tuple,
    param_name: str
):
    """
    Internal helper to detect emptiness of `value`
    and handle it according to <error inline>
    policy.
    """
    if value is None and none_as_empty:
        # Consider None as empty
        return _handle_empty(error, param_name)
    if value is Ellipsis and ellipsis_as_empty:
        # Consider Ellipsis (...) as empty
        return _handle_empty(error, param_name)

    # Check standard array-likes:
    if _is_arraylike_empty(value):
        return _handle_empty(error, param_name)

    # Check optional sets, dicts, etc.
    # We only check if 'set' or 'dict' is in include
    # (by default, we do check them).
    include = include or []
    if include: 
        if "set" in include and isinstance(value, set):
            if len(value) == 0:
                return _handle_empty(error, param_name)
        if "dict" in include and isinstance(value, dict):
            if len(value) == 0:
                return _handle_empty(error, param_name)

    # If it's not considered empty, return as is.
    return value

def _is_arraylike_empty(value):
    """
    Heuristic check if `value` is an empty
    array-like structure (list, tuple, Series,
    DataFrame, numpy array).
    """
    # check list or tuple
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return True

    # check pandas Series/DataFrame
    if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
        if value.empty:
            return True

    # check numpy array
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return True

    return False


def _handle_empty(error_policy, param_name):
    """
    Handle an empty argument based on the
    <error_policy inline>.
    """
    msg = f"Argument '{param_name}' is empty."
    if error_policy == "raise":
        raise ValueError(msg)
    elif error_policy == "warn":
        warnings.warn(msg, UserWarning)
    elif error_policy == "ignore":
        pass
    # Return None so that we effectively "clear"
    # the argument if emptiness was found
    return None


def isdf(func):
    """
    Decorator that ensures the first positional argument passed to the 
    decorated callable is a pandas DataFrame. If it's not, attempts to convert
    it to a DataFrame using an optional `columns` keyword argument. 
    
    Function is designed to be flexible and efficient, suitable for 
    both functions and methods.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the signature of the function
        sig = inspect.signature(func)
        params = sig.parameters
        param_list = list(params.values())
        
        # Check if the function has any parameters
        if not param_list:
            # No parameters to process
            return func(*args, **kwargs)
        
        # Determine if we're decorating a method (with 'self' or 'cls')
        is_method = False
        start_idx = 0
        if param_list[0].name in ('self', 'cls'):
            is_method = True
            start_idx = 1  # Skip 'self' or 'cls'

        # Map arguments to their names
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Identify the data parameter name
        # Prefer 'data' if it's among the parameters
        data_param_name = None
        for idx, param in enumerate(param_list[start_idx:], start=start_idx):
            if param.name == 'data':
                data_param_name = 'data'
                break
        else:
            # If 'data' is not a parameter, use the first positional
            # parameter after 'self'/'cls'
            if ( len(param_list) > start_idx) or is_method:
                data_param_name = param_list[start_idx].name
            else:
                # No parameters left to consider
                return func(*args, **kwargs)

        # Get 'data' argument from bound arguments
        data = bound_args.arguments.get(data_param_name, None)
        columns = bound_args.arguments.get('columns', kwargs.get('columns', None))
        if isinstance(columns, str):
            columns = [columns]
        
        # Proceed with conversion if necessary
        if data is not None and not isinstance(data, pd.DataFrame):
            if isinstance(data, list): 
                data = np.asarray(data)
            try:
                if columns and len(columns) != data.shape[1]:
                    data = pd.DataFrame(data)
                else: 
                    data = pd.DataFrame(data, columns=columns)
            except Exception as e:
                raise ValueError(
                    f"Unable to convert {type(data).__name__!r} to DataFrame: {e}"
                )
            data.columns = data.columns.astype(str)
            # Update the bound arguments with the new data
            bound_args.arguments[data_param_name] = data
        
        # Call the original function with the updated arguments
        return func(*bound_args.args, **bound_args.kwargs)
    
    return wrapper

class SaveFile:
    """
    SaveFile Decorator for Smartly Saving DataFrames in Various Formats.
    
    The `SaveFile` decorator enables automatic saving of DataFrames returned 
    by decorated functions or methods. It intelligently handles different 
    return types, such as single DataFrames or tuples containing DataFrames, 
    and utilizes the `PandasDataHandlers` class to manage file writing based 
    on provided file extensions.
    
    The decorator extracts the `savefile` keyword argument from the decorated 
    function or method. If `savefile` is specified, it determines the 
    appropriate writer based on the file extension and saves the DataFrame 
    accordingly. If the decorated function does not include a `savefile` 
    keyword argument, the decorator performs no action and simply returns 
    the original result.
    
    Parameters
    ----------
    savefile : str, optional
        The file path where the DataFrame should be saved. If `None`, no file 
        is saved.
    data_index : int, default=0
        The index to extract the DataFrame from the returned tuple. Applicable
         only if the decorated function returns a tuple.
    dout : int, default='.csv'
        The default output to save the dataframe if the extension of the file 
        is not provided by the user. 
    writer_kws: dict, optional 
       keywords argument of the writer function. If not passed, assume the 
       'csv' format and turned index to False. 
    verbose: int, default=1 
       Minimum diplaying message. 
       
    Methods
    -------
    __call__(self, func):
        Makes the class instance callable and applies the decorator logic to the 
        decorated function.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from kdiagram.decorators import SaveFile
    >>> from gofast.utils.datautils import to_categories
    
    >>> # Sample DataFrame
    >>> data = {
    ...     'value': np.random.uniform(0, 100, 1000)
    ... }
    >>> df = pd.DataFrame(data)
    
    >>> # Define a function that categorizes and returns the DataFrame
    >>> @SaveFile(data_index=0)
    ... def categorize_values(df, savefile=None):
    ...     df = to_categories(
    ...         df=df,
    ...         column='value',
    ...         categories='auto'
    ...     )
    ...     return df
    
    >>> # Execute the function with savefile parameter
    >>> df = categorize_values(df, savefile='output/value_categories.csv')
    
    >>> # The categorized DataFrame is saved to 'output/value_categories.csv'
    
    >>> # Define a function that returns a tuple containing multiple DataFrames
    >>> @SaveFile(data_index=1)
    ... def process_data(df, savefile=None):
    ...     categorized_df = to_categories(
    ...         df=df,
    ...         column='value',
    ...         categories='auto'
    ...     )
    ...     summary_df = df.describe()
    ...     return (categorized_df, summary_df)
    
    >>> # Execute the function with savefile parameter targeting the summary DataFrame
    >>> categorized, summary = process_data(df, savefile='output/summary_stats.xlsx')
    
    >>> # The summary DataFrame is saved to 'output/summary_stats.xlsx'
    
    Notes
    -----
    - The decorator leverages the `PandasDataHandlers` class to support a wide 
      range of file formats based on the provided file extension.
    - If the decorated function does not include a `savefile` keyword argument, 
      the decorator does not perform any saving operations and simply returns the 
      original result.
    - When dealing with tuple returns, ensure that the `data_index` corresponds 
      to the position of the DataFrame within the tuple.
    - Unsupported file extensions will trigger a warning, and the DataFrame will 
      not be saved.

    See Also
    --------
    PandasDataHandlers : Class for handling Pandas data parsing and writing.
    pandas.DataFrame.to_csv : Method to write a DataFrame to a CSV file.
    pandas.DataFrame.to_excel : Method to write a DataFrame to an Excel file.
    pandas.DataFrame.to_json : Method to write a DataFrame to a JSON file.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame.to_csv. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    .. [2] Pandas Documentation: pandas.DataFrame.to_excel. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html
    .. [3] Pandas Documentation: pandas.DataFrame.to_json. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html
    .. [4] Python Documentation: functools.wraps. 
       https://docs.python.org/3/library/functools.html#functools.wraps
    .. [5] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: 
           L2 theory. *Probability Theory and Related Fields*, 57(5), 453-476.
    """
    def __init__(
        self, 
        # func=None, 
        # *, 
        data_index=0, 
        dout='.csv', 
        writer_kws=None, 
        verbose=1, 
    ):
        # Store the function if passed directly (no parentheses),
        # otherwise store None until __call__ is invoked again.
        # self.func = func
        self.data_index = data_index
        self.dout = dout
        self.data_handler = PandasDataHandlers()
        self.verbose=verbose 
        self.writer_kws= writer_kws or {'index': False}

    def __call__(self, func):
        # If self.func is None, it means the decorator is used with parentheses.
        # The first arg in 'args' will be the actual function to decorate.
        # if self.func is None:
        #     func = args[0]
        #     return SaveFile(
        #         func,
        #         data_index=self.data_index,
        #         dout=self.dout
        #     )
        
        # Otherwise, self.func is already set => define the real wrapper
        @wraps(func)
        def wrapper(*w_args, **w_kwargs):
            # Execute the original function
            result = func(*w_args, **w_kwargs)

            # Check if a 'savefile' kwarg is provided
            savefile = w_kwargs.get('savefile', None)
            
            # otherwirte the writter kws if use provides it explicitely.  
            writer_kws= w_kwargs.get('write_kws', None)
            self.writer_kws = writer_kws or self.writer_kws 
            
            if savefile is not None:
                # Extract extension or use self.dout if none is provided
                _, ext = os.path.splitext(savefile)
                if not ext:
                    if (
                        self.dout is not None 
                        and isinstance(self.dout, str) 
                        and self.dout.startswith('.')
                    ):
                        ext = self.dout.lower()
                    else:
                        warnings.warn(
                            "No file extension provided for `savefile`. "
                            "Cannot save the DataFrame."
                        )
                        return result

                # Determine which DataFrame to save
                if isinstance (result, pd.Series): 
                    result= result.to_frame() 
                    
                if isinstance(result, pd.DataFrame):
                    df_to_save = result
                elif isinstance(result, tuple):
                    try:
                        df_to_save = result[self.data_index]
                    except IndexError:
                        warnings.warn(
                            f"`data_index` {self.data_index} is out of range "
                            "for the returned tuple."
                        )
                        return result
                    
                    except Exception as e: 
                        # If something wrong happend
                        warnings.warn(
                            f"An unexpected error occurred: {e}."
                            " Data cannot be saved; skipped."
                        )
                        return result 
                        
                    
                    if isinstance (df_to_save, pd.Series): 
                        df_to_save = df_to_save.to_frame() 
                    
                    if not isinstance(df_to_save, pd.DataFrame):
                        warnings.warn(
                            f"Element at `data_index` {self.data_index} "
                            "is not a DataFrame; saving skipped"
                        )
                        return result
                else:
                    warnings.warn(
                        f"Return type '{type(result)}' is not a"
                        " DataFrame or tuple; skip saving data."
                    )
                    return result

                # Get the appropriate writer based on file extension
                writers_dict = self.data_handler.writers(df_to_save)
                writer_func = writers_dict.get(ext.lower())
                self.writer_kws= get_valid_kwargs(writer_func, self.writer_kws)
                
                if writer_func is None:
                    warnings.warn(
                        f"Unsupported file extension '{ext}'. "
                        "Cannot save the DataFrame."
                    )
                    return result

                # Attempt to save the DataFrame
                try:
                    writer_func(
                        savefile,
                        **self.writer_kws
                        # index=False
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to save the DataFrame: {e}"
                    )
                else: 
                    if self.verbose:
                        print(
                            "[INFO] DataFrame saved to "
                            f"'{savefile}'."
                        )

            return result

        # Return the wrapper function
        return wrapper
    
    @classmethod 
    def save_file(
        cls, 
        func=None, 
        *, 
        data_index=0, 
        dout='.csv', 
        writer_kws=None, 
        verbose=1,
    ): 
        if func is not None:
            return cls(data_index, dout, writer_kws, verbose)(func)
        return cls(data_index, dout, writer_kws, verbose)
    
    
# Class-based decorator to save returned DataFrame(s) to file.
# Allows usage with parentheses (e.g. @SaveFile(data_index=1)) 
# or without parentheses (e.g. @SaveFile).
SaveFile = SaveFile.save_file 

def save_file(func=None, *, data_index=0, dout='.csv'):
    """
    Both save_file (function-based) and SaveFile (class-based) decorators
    are designed to allow users to save the returned DataFrame(s) from a 
    decorated function to a file, if needed. For more details and advanced
    usage, please refer to the documentation of :class:`kdiagram.decorators.SaveFile`,
    as both operate in a similar manner.
    
    * When to Use SaveFile vs. save_file? 
    
    - SaveFile is a class-based decorator.
    - save_file is a function-based decorator.
    
    Their behavior is essentially the same, so you can choose whichever 
    style fits your coding preference or project standards. Both will look 
    for a savefile argument at runtime and, if present, save the DataFrame 
    result using the rules described above.
    
    For full documentation and more advanced usage details, please check 
    the documentation of :class:`kdiagram.decorators.SaveFile`.
    """
    # If called without parentheses, `func` is the function object.
    # If called with parentheses, `func` is None on first pass
    # and we return a decorator that expects the function later.
    if func is None:
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                result = f(*args, **kwargs)
                savefile = kwargs.get('savefile', None)
                if savefile is not None:
                    df_to_save, ext = _get_df_to_save (
                        savefile, dout, result, data_index 
                    )
                    if df_to_save is None:
                        return result
                    _perform_save(df_to_save, savefile, ext)
                return result
            return wrapper
        return decorator
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            savefile = kwargs.get('savefile', None)
            if savefile is not None:
                df_to_save, ext = _get_df_to_save (
                    savefile, dout, result, data_index 
                )
                if df_to_save is None:
                    return result
                _perform_save(df_to_save, savefile, ext)
            return result
        return wrapper

def _get_df_to_save (savefile, dout, result, data_index ): 
    _, ext = os.path.splitext(savefile)
    if not ext:
        if dout and isinstance(dout, str) and dout.startswith('.'):
            ext = dout.lower()
        else:
            warnings.warn(
                "No file extension provided for `savefile`. "
                "Cannot save the DataFrame."
            )
            return result
    df_to_save = _extract_dataframe(result, data_index)
    return df_to_save, ext 

    
def _extract_dataframe(result, data_index):
    if isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, tuple):
        try:
            df = result[data_index]
        except IndexError:
            warnings.warn(
                f"`data_index` {data_index} is out of range "
                "for the returned tuple."
            )
            return None
        if not isinstance(df, pd.DataFrame):
            warnings.warn(
                f"Element at `data_index` {data_index} is not a DataFrame."
            )
            return None
        return df
    else:
        warnings.warn(
            f"Return type '{type(result)}' is not a DataFrame or tuple."
        )
        return None

def _perform_save(df_to_save, savefile, ext):
    # map of extension to saving method
    # Get the appropriate writer based on file extension
    data_handler = PandasDataHandlers()
    writers_dict= data_handler.writers(df_to_save)
    writer_func = writers_dict.get(ext.lower())
    
    if writer_func is None:
        warnings.warn(
            f"Unsupported file extension '{ext}'. "
            "Cannot save DataFrame."
        )
        return
    # Attempt to save the DataFrame
    try:
        writer_func(
            savefile,
            index=False
        )
    except Exception as e:
        warnings.warn(
            f"Failed to save the DataFrame: {e}"
        )

save_file.__doc__= SaveFile.__doc__ 