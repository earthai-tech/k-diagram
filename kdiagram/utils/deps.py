from __future__ import annotations

from ._deps import ensure_pkg as _ensure_pkg

# Make docs/import paths show the public module rather than _deps
_ensure_pkg.__module__ = "kdiagram.utils.deps"

ensure_pkg = _ensure_pkg

__all__ = ["ensure_pkg"]

#   License: Apache-2.0
#   Author: LKouadio <etanoyau@gmail.com>

__doc__ = r"""
Dependency gating utilities for *kdiagram*.

This module exposes a single public decorator,
:func:`ensure_pkg`, that validates importable
dependencies before a function, method, or class
executes.  It can enforce a minimum version, and
optionally attempt a one-shot upgrade or install
via ``pip`` when requested.

The goal is to fail *early* and *clearly*, while
offering an escape hatch for interactive or
controlled environments that allow runtime
installation.

Public API
----------
ensure_pkg
    Decorator that checks import availability,
    optional version constraints, and optional
    auto-install behavior.  Works on callables
    and on classes (prior to instantiation).

Quick Start
-----------
Basic gate on ``numpy`` 1.23 or newer::

    >>> from kdiagram.utils.deps import ensure_pkg
    >>> @ensure_pkg("numpy", min_version="1.23")
    ... def to_array(x):
    ...     import numpy as np
    ...     return np.asarray(x)

Warn but continue if the requirement is missing
or too old::

    >>> @ensure_pkg("pandas", min_version="2.0",
    ...             errors="warn")
    ... def summarize(df):
    ...     import pandas as pd
    ...     return df.describe()

Guard a class before heavy ``__init__`` logic::

    >>> @ensure_pkg("matplotlib", errors="raise")
    ... class Plotter:
    ...     def __init__(self):
    ...         import matplotlib.pyplot as plt
    ...         self._plt = plt

Behavior
--------
errors : {'raise', 'warn', 'ignore'}
    - ``'raise'``  → raise an informative error.
    - ``'warn'``   → warn and continue anyway.
    - ``'ignore'`` → continue silently.

Key Details
-----------
- *Import vs distribution names.*  Some projects
  import as one name but distribute under another
  (e.g., ``skimage`` vs ``scikit-image``).  Use
  ``dist_name`` to disambiguate when needed.

- *Version parsing.*  When available, the
  ``packaging`` library is used for PEP 440
  compliant comparisons [1]_.  Otherwise a
  best-effort fallback may be used.

- *Caching.*  Checks may be cached per
  ``(name, dist_name, min_version)`` so repeated
  calls are fast within a single process.

- *Auto-install caveats.*  Enabling
  ``auto_install=True`` modifies the current
  environment and may require network access,
  permissions, and isolation.  Prefer project
  pins and environment files for production.

- *Conda.*  ``use_conda`` is accepted for API
  stability but currently ignored.  Prefer
  managing conda packages with environment
  tooling until a conda backend is provided.

See Also
--------
importlib.import_module
    Standard library facility to import modules
    by dotted path.
importlib.metadata.version
    Query installed distribution versions.
packaging.version.Version
    PEP 440 compliant version objects and
    comparisons.

References
----------
.. [1] PEP 440 — Version Identification and
       Dependency Specification.
.. [2] importlib.metadata — Accessing package
       metadata.  Python Standard Library.
.. [3] Packaging library — Version parsing and
       comparisons, https://packaging.pypa.io/
.. [4] pip User Guide — Installing packages,
       https://pip.pypa.io/

"""
