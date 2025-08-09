# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

import os
import re
import shutil
import warnings


def beautify_dict(d, space=4, key=None, max_char=None):
    """
    Format a dictionary with custom indentation and aligned keys and values.
    
    Parameters:
    ----------
    d : dict
        The dictionary to format.
    space : int, optional
        The number of spaces to indent the dictionary entries.
    key : str, optional
        An optional key to nest the dictionary under.
    max_char : int, optional
        Maximum characters a value can have before being truncated.
    
    Returns:
    -------
    str
        A string representation of the dictionary with custom formatting.

    Examples:
    --------
    >>> from kdiagram.api.util import beautify_dict
    >>> dictionary = {
    ...     3: 'Home & Garden',
    ...     2: 'Health & Beauty',
    ...     4: 'Sports',
    ...     0: 'Electronics',
    ...     1: 'Fashion'
    ... }
    >>> print(beautify_dict(dictionary, space=4))
    """
    
    if not isinstance(d, dict):
        raise TypeError("Expected input to be a 'dict',"
                        f" received '{type(d).__name__}' instead.")
    
    if max_char is None: 
        # get it automatically 
        max_char, _ =get_terminal_size()
    # Determine the longest key for alignment
    if len(d)==0: 
        max_key_length=0 
    else:
        max_key_length = max(len(str(k)) for k in d.keys())
    
    # Create a list of formatted rows
    formatted_rows = []
    for dkey, value in sorted(d.items()):
        value_str = str(value)
        if max_char is not None and len(value_str) > max_char:
            value_str = value_str[:max_char] + "..."
        # Ensure all keys are right-aligned to the longest key length
        formatted_row = f"{str(dkey):>{max_key_length}}: '{value_str}'"
        formatted_rows.append(formatted_row)

    # Join all rows into a single string with custom indentation
    indent = ' ' * space
    inner_join = ',\n' + indent
    formatted_dict = '{\n' + indent + inner_join.join(formatted_rows) + '\n}'

    if key:
        # Prepare outer key indentation and format
           # Slightly less than the main indent
        outer_indent = ' ' * (space - 2 + len(key) + max_key_length)  
        # Construct a new header with the key
        formatted_dict = f"{key} : {formatted_dict}"
        # Split lines and indent properly to align with the key
        lines = formatted_dict.split('\n')
        for i in range(1, len(lines)):
            lines[i] = outer_indent + lines[i]
            if max_char is not None and len(lines[i]) > max_char:
                lines[i] = lines[i][:max_char] + "..."
        # format lins -1 
        #lines [-1] = outer_indent + lines [-1]
        formatted_dict = '\n'.join(lines)
        
    return formatted_dict

def to_camel_case(text, delimiter=None, use_regex=False):
    """
    Converts a given string to CamelCase. The function handles strings with or
    without delimiters, and can optionally use regex for splitting based on
    non-alphanumeric characters.

    Parameters
    ----------
    text : str
        The string to convert to CamelCase.
    delimiter : str, optional
        A character or string used as a delimiter to split the input string. 
        Common delimiters include underscores ('_') or spaces (' '). If None 
        and use_regex is ``False``, the function tries to automatically detect 
        common delimiters like spaces or underscores. 
        If `use_regex` is ``True``, it splits the string at any non-alphabetic 
        character.
    use_regex : bool, optional
        Specifies whether to use regex for splitting the string on non-alphabetic
        characters.
        Defaults to ``False``.

    Returns
    -------
    str
        The CamelCase version of the input string.

    Examples
    --------
    >>> from kdiagram.api.util import to_camel_case
    >>> to_camel_case('outlier_results', '_')
    'OutlierResults'

    >>> to_camel_case('outlier results', ' ')
    'OutlierResults'

    >>> to_camel_case('outlierresults')
    'Outlierresults'

    >>> to_camel_case('data science rocks')
    'DataScienceRocks'

    >>> to_camel_case('data_science_rocks')
    'DataScienceRocks'

    >>> to_camel_case('multi@var_analysis', use_regex=True)
    'MultiVarAnalysis'
    
    >>> to_camel_case('OutlierResults')
    'OutlierResults'

    >>> to_camel_case('BoxFormatter')
    'BoxFormatter'

    >>> to_camel_case('MultiFrameFormatter')
    'MultiFrameFormatter'
    """
    # Remove any leading/trailing whitespace
    text = str(text).strip()

    # Check if text is already in CamelCase and return it as is
    if text and text[0].isupper() and text[1:].islower() == False:
        return text

    if use_regex:
        # Split text using any non-alphabetic character as a delimiter
        words = re.split('[^a-zA-Z]', text)
    elif delimiter is None:
        if ' ' in text and '_' in text:
            # Both space and underscore are present, replace '_' with ' ' then split
            text = text.replace('_', ' ')
            words = text.split()
        elif ' ' in text:
            words = text.split(' ')
        elif '_' in text:
            words = text.split('_')
        else:
            # No common delimiter found, handle as a single word
            words = [text]
    else:
        # Use the specified delimiter
        words = text.split(delimiter)

    # Capitalize the first letter of each word and join them without spaces
       # Ensure empty strings from split are ignored
    return ''.join(word.capitalize() for word in words if word)  


def to_snake_case(name, mode="standard"):
    """
    Converts a string to snake_case.

    Parameters
    ----------
    name : str
        The string to convert to snake_case.
    mode : str, optional
        If 'soft', extra whitespace and case inconsistencies are
        handled to produce clean snake_case.

    Returns
    -------
    str
        The snake_case version of the input string.
    """
    name = str(name)
    
    if mode == "soft":
        # Convert to lowercase and replace multiple spaces 
        # or non-word characters with a single underscore
        name = re.sub(r'\W+', ' ', name)  # Replace non-word characters with spaces
        name = re.sub(r'\s+', ' ', name).strip()  # Normalize whitespace
        name = name.lower().replace(' ', '_')  # Convert spaces to underscores
    
    else:
        # Standard snake_case conversion without additional processing
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()  # Convert CamelCase to snake_case
        name = re.sub(r'\W+', '_', name)  # Replace non-word characters with '_'
        name = re.sub(r'_+', '_', name)  # Replace multiple underscores with a single '_'
    
    return name.strip('_')  # Remove any leading or trailing underscores


def get_table_size(width="auto", error="warn", return_height=False):
    """
    Determines the appropriate width (and optionally height) for table display
    based on terminal size, with options for manual width adjustment.

    Parameters
    ----------
    width : int or str, optional
        The desired width for the table. If set to 'auto', the terminal width
        is used. If an integer is provided, it will be used as the width, 
        default is 'auto'.
    error : str, optional
        Error handling strategy when specified width exceeds terminal 
        width: 'warn' or 'ignore'.
        Default is 'warn'.
    return_height : bool, optional
        If True, the function also returns the height of the table. 
        Default is False.

    Returns
    -------
    int or tuple
        The width of the table as an integer, or a tuple of (width, height) 
        if return_height is True.

    Examples
    --------
    >>> table_width = get_table_size()
    >>> print("Table width:", table_width)
    >>> table_width, table_height = get_table_size(return_height=True)
    >>> print("Table width:", table_width, "Table height:", table_height)
    """
    auto_width, auto_height = get_terminal_size()
    if width == "auto":
        width = auto_width
    else:
        try:
            width = int(width)
            if width > auto_width:
                if error == "warn":
                    warnings.warn(
                        f"Specified width {width} exceeds terminal width {auto_width}. "
                        "This may cause display issues."
                    )
        except ValueError:
            raise ValueError(
                "Width must be 'auto' or an integer; got {type(width).__name__!r}")

    if return_height:
        return (width, auto_height)
    return width

def get_terminal_size():
    """
    Retrieves the current terminal size (width and height) to help dynamically 
    set the maximum width for displaying data columns.

    Returns
    -------
    tuple
        A tuple containing two integers:
        - The width of the terminal in characters.
        - The height of the terminal in lines.

    Examples
    --------
    >>> from kdiagram.api.util import get_terminal_size
    >>> terminal_width, terminal_height = get_terminal_size()
    >>> print("Terminal Width:", terminal_width)
    >>> print("Terminal Height:", terminal_height)
    """
    # Use shutil.get_terminal_size if available (Python 3.3+)
    # This provides a fallback of (80, 24) which is a common default size
    if hasattr(shutil, "get_terminal_size"):
        size = shutil.get_terminal_size(fallback=(80, 24))
    else:
        # Fallback for Python versions before 3.3
        try:
            # UNIX-based systems
            size = os.popen('stty size', 'r').read().split()
            return int(size[1]), int(size[0])
        except Exception:
            # Default fallback size
            size = (80, 24)
    return size.columns, size.lines