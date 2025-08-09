# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0 (see LICENSE file)

# -------------------------------------------------------------------
# Defines core properties, constants, and default settings for k-diagram.
# Parts may be adapted or inspired by code originally found in the
# 'gofast' package: https://github.com/earthai-tech/gofast
# Original 'gofast' code licensed under BSD-3-Clause.
# Modifications and 'k-diagram' are under Apache License 2.0.
# -------------------------------------------------------------------
"""
Core Properties and Constants (:mod:`kdiagram.core.property`)
===============================================================

This module defines shared constants, default parameter values, string
enumerations (like accepted options for plot arguments), configuration
settings, or other core properties used internally throughout the
`k-diagram` package to ensure consistency.
"""

import pandas as pd


class PandasDataHandlers:
    """ 
    A container for data parsers and writers based on Pandas, supporting a 
    wide range of formats for both reading and writing DataFrames. This class 
    simplifies data I/O by mapping file extensions to Pandas functions, making 
    it easier to manage diverse file formats in the Gofast package.
    
    Attributes
    ----------
    parsers : dict
        A dictionary mapping common file extensions to Pandas functions for 
        reading files into DataFrames. Each entry links a file extension to 
        a specific Pandas reader function, allowing for standardized and 
        convenient data import.

    Methods
    -------
    writers(obj)
        Returns a dictionary mapping file extensions to Pandas functions for 
        writing a DataFrame to various formats. Enables easy exporting of data 
        in multiple file formats, ensuring flexibility in data storage.
        
    Notes
    -----
    The `PandasDataHandlers` class centralizes data handling functions, 
    allowing for a unified interface to access multiple data formats, which 
    simplifies data parsing and file writing in the Gofast package.

    This class does not take any parameters on initialization and is used 
    to manage I/O options for DataFrames exclusively.

    Examples
    --------
    >>> from kdiagram.core.property import PandasDataHandlers
    >>> data_handler = PandasDataHandlers()
    
    # Reading a CSV file
    >>> parser_func = data_handler.parsers[".csv"]
    >>> df = parser_func("data.csv")
    
    # Writing to JSON
    >>> writer_func = data_handler.writers(df)[".json"]
    >>> writer_func("output.json")

    The above example illustrates how to access reader and writer functions 
    for specified file extensions, allowing for simplified data import and 
    export with Pandas.

    See Also
    --------
    pandas.DataFrame : Provides comprehensive data structures and methods for 
                       managing tabular data.
                       
    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing 
           in Python." In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    """

    @property
    def parsers(self):
        """
        A dictionary mapping file extensions to Pandas functions for reading 
        data files. Each extension is associated with a Pandas function 
        capable of parsing the respective format and returning a DataFrame.

        Returns
        -------
        dict
            A dictionary of file extensions as keys, and their respective 
            Pandas parsing functions as values.

        Examples
        --------
        >>> data_handler = PandasDataHandlers()
        >>> csv_parser = data_handler.parsers[".csv"]
        >>> df = csv_parser("data.csv")

        Notes
        -----
        The `parsers` attribute simplifies data import across diverse formats 
        supported by Pandas. As new formats are integrated into Pandas, this 
        dictionary can be expanded to include additional file types.
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
            ".txt": pd.read_csv
        }

    @staticmethod
    def writers(obj):
        """
        A dictionary mapping file extensions to Pandas functions for writing 
        DataFrames. The `writers` method generates file-specific writing 
        functions to enable export of DataFrames in various formats.

        Parameters
        ----------
        obj : pandas.DataFrame
            The DataFrame to be written to a specified format.
        
        Returns
        -------
        dict
            A dictionary of file extensions as keys, mapped to the DataFrame 
            writer functions in Pandas that allow exporting to that format.

        Examples
        --------
        >>> data_handler = PandasDataHandlers()
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> json_writer = data_handler.writers(df)[".json"]
        >>> json_writer("output.json")

        Notes
        -----
        The `writers` method provides a flexible solution for exporting data 
        to multiple file formats. This method centralizes data export 
        functionality by associating file extensions with Pandas writer 
        methods, making it straightforward to save data in different formats.
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

