# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

"""
A compatibility layer so k-diagram works under both NumPy 1.x and 2.0+
"""

import numpy as _np
from numpy.lib import NumpyVersion as _NV

# Detect NumPy major version
_NVERSION = _NV(_np.__version__)
IS_NP2    = _NVERSION >= "2.0.0"

# === Type aliases ===
# NumPy 2.0 removed aliases like np.int, np.float, np.bool, etc.
# Provide safe fallbacks to the corresponding Python built-ins.
int_     = int     if IS_NP2 else _np.int
float_   = float   if IS_NP2 else _np.float
bool_    = bool    if IS_NP2 else _np.bool
object_  = object  if IS_NP2 else _np.object
complex_ = complex if IS_NP2 else _np.complex
str_     = str     if IS_NP2 else _np.str_

# === Moved / renamed functions ===
# These existed in the root namespace in 1.x but were
# removed/moved in 2.0 (NEP 52).
in1d      = _np.isin         # replaced alias
row_stack = _np.vstack       # replaced alias
trapz     = _np.trapezoid    # replaced alias

# === AxisError import compatibility ===
# In NumPy 2.x it lives under numpy.exceptions.
try:
    from numpy.exceptions import AxisError
except ImportError:
    from numpy import AxisError

# === Promotion-warnings helper ===
def set_promotion_warn(state: str = "weak_and_warn") -> None:
    """
    During testing you can enable warnings on changed type-promotion
    behavior (NumPy 2.0+).  E.g.:

        import warnings
        warnings.simplefilter('error')
        compat_numpy.set_promotion_warn()

    to turn those into errors and catch any unintended changes.
    """
    if IS_NP2:
        _np._set_promotion_state(state)


# === asarray with copy keyword signature ===
def asarray(x, dtype=None, copy=None):
    """
    Wrapper around numpy.asarray that accepts the
    signature (dtype=None, copy=None) under both 1.x and 2.x.
    """
    if not IS_NP2:
        # NumPy 1.x ignores copy kwarg on array(), so drop it
        return _np.asarray(x, dtype=dtype)
    return _np.asarray(x, dtype=dtype, copy=copy)


# === Default integer dtype ===
# Expose what the “default” int type is in this NumPy build.
default_int = _np.intp if IS_NP2 else _np.int_

# === Public API ===
__all__ = [
    "IS_NP2", "int_", "float_", "bool_", "object_", "complex_", "str_",
    "in1d", "row_stack", "trapz", "AxisError",
    "set_promotion_warn", "asarray", "default_int",
]
