# File: kdiagram/compat/sklearn.py
# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

"""
A compatibility module to handle API changes across different
versions of Matplotlib.
"""

import warnings

import matplotlib
from packaging.version import parse

# Get the installed Matplotlib version
_MPL_VERSION = parse(matplotlib.__version__)


__all__ = ["get_cmap", "is_valid_cmap"]


def get_cmap(name="viridis", lut=None, allow_none=False):
    """Get a Matplotlib colormap with version compatibility.

    This function acts as a robust wrapper to retrieve a colormap
    object, ensuring consistent behavior across different
    Matplotlib versions. It handles the API change from
    `matplotlib.cm.get_cmap` to `matplotlib.colormaps.get`.

    Parameters
    ----------
    name : str or None, optional
        The name of the colormap to retrieve. Defaults to "viridis".
    lut : int, optional
        The number of colors in the lookup table. This parameter
        is only used for Matplotlib versions older than 3.6 and
        is ignored in newer versions. Defaults to None.
    allow_none : bool, optional
        If True, passing `name=None` will return `None` without
        raising an error. If False (default), passing `name=None`
        will raise a `ValueError`. This is the primary control for
        preventing unexpected `TypeError` exceptions downstream.

    Returns
    -------
    matplotlib.colors.Colormap or None
        The requested colormap instance. Returns `None` only if
        `name` is None and `allow_none` is True.

    Raises
    ------
    ValueError
        If `name` is `None` and `allow_none` is False, if `name` is
        not a string (and not None), or if the named colormap
        does not exist.
    """
    if name is None:
        if allow_none:
            return None
        else:
            raise ValueError(
                "Colormap `name` cannot be None when `allow_none` is False."
            )

    if not isinstance(name, str):
        raise ValueError(f"Colormap name must be a string, not {type(name)}.")

    if _MPL_VERSION >= parse("3.6"):
        # The new API raises KeyError for invalid names. We catch it and
        # re-raise as ValueError to match the old API's behavior. This
        # ensures that try/except ValueError blocks in the main code
        # continue to work as expected.
        
        try:
            return matplotlib.colormaps.get(name)
        except (TypeError, KeyError, ValueError) as err:
            raise ValueError(
                f"'{name}' is not a valid colormap name."
            ) from err
    else:
        # The old API raises ValueError for invalid names.
        
        return matplotlib.cm.get_cmap(name, lut)

def is_valid_cmap(cmap, default="viridis", error="raise"):
    """Check if a colormap name is valid by attempting to retrieve it.

    This utility leverages the robust `get_cmap` wrapper to handle
    different Matplotlib versions and input types seamlessly.

    Parameters
    ----------
    cmap : str
        The name of the colormap to validate.
    default : str, optional
        The colormap name to return if `cmap` is found to be
        invalid. Defaults to "viridis".
    error : {'raise', 'warn', 'ignore'}, optional
        Defines how to handle an invalid `cmap` name.
        - 'raise': Raise a `ValueError` (default).
        - 'warn': Issue a `UserWarning` and return the `default`.
        - 'ignore': Silently return the `default` value.

    Returns
    -------
    str
        A valid colormap name, which is either the original `cmap`
        or the `default` value if `cmap` was invalid.
        
    Raises
    ------
    ValueError
        If `cmap` is invalid and `error` is 'raise', or if the 
        `error` parameter itself is given an invalid value.
    """
    try:
        # We call get_cmap with its safe default (allow_none=False)
        # because this function's purpose is to validate names.
        # It will raise a ValueError for any invalid input.
        get_cmap(cmap, allow_none=False)
        return cmap
    except ValueError:
        # The name is invalid, so handle it based on the 'error' flag.
        if error == "raise":
            raise ValueError(f"'{cmap}' is not a valid colormap name.")
        elif error == "warn":
            warnings.warn(
                f"Invalid `cmap` name '{cmap}'. Falling back to '{default}'.",
                UserWarning,
                stacklevel=2,
            )
            return default
        elif error == "ignore":
            return default
        else:
            raise ValueError(
                "Invalid value for 'error' parameter. Must be 'raise', "
                "'warn', or 'ignore'."
            )
            