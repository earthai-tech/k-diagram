# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0
#
# ------------------------------------------------------------------
# Core properties, constants, and defaults for k-diagram.
# Parts may be adapted or inspired by code in the 'gofast'
# package:
# https://github.com/earthai-tech/gofast
# Original 'gofast' code is licensed under BSD-3-Clause.
# Modifications and 'k-diagram' are under Apache License 2.0.
# ------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

__all__ = ["PandasDataHandlers"]

r"""
Core properties for :mod:`kdiagram.core.property`.

This module exposes small, shared utilities used across
``kdiagram``.  Currently it provides a lightweight mapping
between common file extensions and pandas IO functions for
reading and writing tabular data.

Notes
-----
The mappings are intentionally conservative.  They cover
popular formats shipped with pandas.  If your build of
pandas lacks a given reader or writer, importing or using
that entry may raise at runtime.

Examples
--------
Get a parser and read a CSV file::

    >>> from kdiagram.core.property import PandasDataHandlers
    >>> h = PandasDataHandlers()
    >>> df = h.parsers[".csv"]("data.csv")

Get a writer and export to JSON::

    >>> w = h.writers(df)[".json"]
    >>> w("out.json")

See Also
--------
pandas.read_* : Family of read functions.
pandas.DataFrame.to_* : Family of write methods.

References
----------
.. [1] McKinney, W.  *Data Structures for Statistical
       Computing in Python*.  Proc. SciPy 2010.
"""


class PandasDataHandlers:
    r"""
    Small container that surfaces pandas IO helpers.

    The goal is to centralize a consistent set of readers
    (``parsers``) and writers (``writers``) keyed by file
    extension.  This keeps IO routing simple and uniform
    in higher level APIs.

    Attributes
    ----------
    parsers : dict
        Maps extension to a pandas reader callable.

    Notes
    -----
    The mappings return the raw pandas callables.  You may
    pass any pandas-specific keyword arguments to them.  For
    remote filesystems, many readers accept ``storage_options``.
    """

    @property
    def parsers(self) -> dict[str, Callable[..., pd.DataFrame]]:
        r"""
        Return a mapping of extension to pandas readers.

        Returns
        -------
        dict
            Keys are extensions (dot-prefixed).  Values are
            pandas read callables.
        """
        return {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".json": pd.read_json,
            ".html": pd.read_html,
            ".sql": pd.read_sql,
            ".xml": pd.read_xml,
            ".fwf": pd.read_fwf,
            ".pkl": pd.read_pickle,
            ".sas": pd.read_sas,
            ".spss": pd.read_spss,
            ".txt": pd.read_csv,
        }

    @staticmethod
    def writers(
        obj: pd.DataFrame,
    ) -> dict[str, Callable[..., Any]]:
        r"""
        Return a mapping of extension to pandas writers.

        Parameters
        ----------
        obj : pandas.DataFrame
            The frame for which writers will be bound.

        Returns
        -------
        dict
            Keys are extensions (dot-prefixed).  Values are
            bound methods that write ``obj`` to the target
            format.
        """
        return {
            ".csv": obj.to_csv,
            ".hdf": obj.to_hdf,
            ".sql": obj.to_sql,
            ".dict": obj.to_dict,
            ".xlsx": obj.to_excel,
            ".json": obj.to_json,
            ".html": obj.to_html,
            ".feather": obj.to_feather,
            ".tex": obj.to_latex,
            ".stata": obj.to_stata,
            ".gbq": obj.to_gbq,
            ".rec": obj.to_records,
            ".str": obj.to_string,
            ".clip": obj.to_clipboard,
            ".md": obj.to_markdown,
            ".parq": obj.to_parquet,
            ".pkl": obj.to_pickle,
        }
