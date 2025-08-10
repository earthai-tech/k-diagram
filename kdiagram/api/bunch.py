# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0 (see LICENSE file)

"""
Bunch Data Structure (:mod:`kdiagram.bunch`)
==========================================

Provides a dictionary-like object that allows accessing keys as
attributes.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

__all__ = ["Bunch", "FlexDict"]


class Bunch(dict):
    """Dictionary-like container providing attribute-style access.

    Acts as a standard Python dictionary but allows accessing keys as
    attributes for convenience, similar to structures used in libraries
    like scikit-learn for holding datasets or results.

    Parameters
    ----------
    *args : tuple
        Arguments passed directly to the `dict` constructor. Allows
        initialization from mapping objects or iterables of key-value
        pairs.
    **kwargs : dict
        Keyword arguments passed directly to the `dict` constructor.
        Allows initialization using key=value pairs.

    Examples
    --------
    >>> from kdiagram.bunch import Bunch
    >>> b = Bunch(a=1, b='hello')
    >>> b.a
    1
    >>> b['b']
    'hello'
    >>> b.c = [1, 2, 3]
    >>> b['c']
    [1, 2, 3]
    >>> print(b)
    Bunch({'a': 1, 'b': 'hello', 'c': [1, 2, 3]})
    >>> 'a' in b
    True
    >>> list(b.keys())
    ['a', 'b', 'c']
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize Bunch using dict constructor."""
        super().__init__(*args, **kwargs)
        # Bind __getattr__, __setattr__, __delattr__ early
        # This helps ensure attribute access works immediately even if
        # these methods were somehow masked by initial keys.
        self.__dict__["__getattr__"] = super().__getattribute__("__getattr__")
        self.__dict__["__setattr__"] = super().__getattribute__("__setattr__")
        self.__dict__["__delattr__"] = super().__getattribute__("__delattr__")

    def __getattr__(self, name: str) -> Any:
        """Retrieve item via attribute access.

        Called when `bunch.key` attribute access is used. Raises
        AttributeError if the key is not found, mimicking standard
        attribute behavior.

        Parameters
        ----------
        name : str
            The attribute (key) name being accessed.

        Returns
        -------
        Any
            The value associated with the key `name`.

        Raises
        ------
        AttributeError
            If `name` is not found as a key in the Bunch object.
        """
        try:
            # Access the item using standard dictionary lookup
            return self[name]
        except KeyError:
            # If key doesn't exist, raise AttributeError
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None  # Suppress context from KeyError

    def __setattr__(self, name: str, value: Any) -> None:
        """Set item via attribute access.

        Called when `bunch.key = value` attribute assignment is used.
        Stores the item as a standard dictionary key-value pair.

        Parameters
        ----------
        name : str
            The attribute (key) name being assigned.
        value : Any
            The value to assign to the key `name`.
        """
        # Set the item using standard dictionary assignment
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete item via attribute access.

        Called when `del bunch.key` attribute deletion is used. Removes
        the key-value pair from the dictionary. Raises AttributeError
        if the key is not found.

        Parameters
        ----------
        name : str
            The attribute (key) name to delete.

        Raises
        ------
        AttributeError
            If `name` is not found as a key in the Bunch object.
        """
        try:
            # Delete the item using standard dictionary deletion
            del self[name]
        except KeyError:
            # If key doesn't exist, raise AttributeError
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None  # Suppress context from KeyError

    def __repr__(self) -> str:
        """Return a string representation indicating it's a Bunch."""
        # Get the standard dictionary representation
        dict_repr = super().__repr__()
        # Wrap it to show it's a Bunch object
        return f"Bunch({dict_repr})"

    def __dir__(self) -> Iterable[str]:
        """Provide attributes for tab-completion.

        Includes dictionary keys in the list of attributes available
        for interactive completion (e.g., in IPython/Jupyter).

        Returns
        -------
        Iterable[str]
            Combined list of standard attributes/methods and keys.
        """
        # Start with standard attributes/methods from dict
        dynamic_attrs = set(super().__dir__())
        # Add the dictionary keys
        dynamic_attrs.update(self.keys())
        return sorted(list(dynamic_attrs))

    def copy(self) -> Bunch:
        """Return a shallow copy of the Bunch."""
        return Bunch(super().copy())

    def __copy__(self) -> Bunch:
        """Support copy.copy()."""
        return self.copy()

    def __deepcopy__(self, memodict=None) -> Bunch:
        """Support copy.deepcopy()."""
        if memodict is None:
            memodict = {}
        return Bunch(copy.deepcopy(dict(self), memodict))


class FlexDict(dict):
    """
    A `FlexDict` is a dictionary subclass that provides flexible attribute-style
    access to its items, allowing users to interact with the dictionary as if it
    were a regular object with attributes. It offers a convenient way to work with
    dictionary keys without having to use the bracket notation typically required by
    dictionaries in Python. This makes it especially useful in environments where
    quick and easy access to data is desired.

    The `FlexDict` class extends the built-in `dict` class, so it inherits all the
    methods and behaviors of a standard dictionary. In addition to the standard
    dictionary interface, `FlexDict` allows for the setting, deletion, and access
    of keys as if they were attributes, providing an intuitive and flexible
    interface for managing dictionary data.

    Examples
    --------
    Here is how you can use a `FlexDict`:

    >>> from kdiagram.api.bunch import FlexDict
    >>> fd = FlexDict(pkg='gofast', goal='simplify tasks', version='1.0')
    >>> fd['pkg']  # Standard dictionary access
    'gofast'
    >>> fd.pkg     # Attribute access
    'gofast'
    >>> fd.goal    # Another example of attribute access
    'simplify tasks'
    >>> fd.version # Accessing another attribute
    '1.0'
    >>> fd.new_attribute = 'New Value'  # Setting a new attribute
    >>> fd['new_attribute']             # The new attribute is accessible as a key
    'New Value'

    Notes
    -----
    - While `FlexDict` adds convenience, it is important to avoid key names that
      clash with the methods and attributes of a regular dictionary. Such conflicts
      can result in unexpected behavior, as method names would take precedence over
      key names during attribute access.

    - The behavior of `FlexDict` under serialization (e.g., when using pickle) may
      differ from that of a standard dictionary due to the attribute-style access.
      Users should ensure that serialization and deserialization processes are
      compatible with `FlexDict`'s approach to attribute access.

    - Since `FlexDict` is built on the Python dictionary, it maintains the same
      performance characteristics for key access and management. However, users
      should be mindful of the additional overhead introduced by supporting
      attribute access when considering performance-critical applications.

    By providing a dictionary that can be accessed and manipulated as if it were a
    regular object, `FlexDict` offers an enhanced level of usability, particularly
    in situations where the more verbose dictionary syntax might be less desirable.
    """

    def __init__(self, **kwargs):
        """
        Initialize a FlexDict with keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs to initialize the FlexDict.
        """
        super().__init__(**kwargs)
        self.__dict__ = self

    def __getattr__(self, item):
        """
        Allows attribute-style access to the dictionary keys.

        Parameters
        ----------
        item : str
            The attribute name corresponding to the dictionary key.

        Returns
        -------
        The value associated with 'item' in the dictionary.

        Raises
        ------
        AttributeError
            If 'item' is not found in the dictionary.
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'FlexDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """
        Enables setting dictionary items directly as object attributes,
        with a special rule:
        if the attribute name contains any of the designated special symbols
        ('**', '%%', '&&', '||', '$$'), only the substring before the first
        occurrence of any of these symbols will be used as the key.

        Parameters
        ----------
        key : str
            The attribute name to be added or updated in the dictionary. If
            the key contains any special symbols ('**', '%%', '&&', "||", '$$'),
            it is truncated before the first occurrence of these symbols.
        value : any
            The value to be associated with 'key'.

        Example
        -------
        If the key is 'column%%stat', it will be truncated to 'column', and
        only 'column' will be used as the key.
        """
        # List of special symbols to check in the key.
        special_symbols = ["**", "%%", "&&", "||", "$$"]
        # Iterate over the list of special symbols.
        for symbol in special_symbols:
            # Check if the current symbol is in the key.
            if symbol in key:
                # Split the key by the symbol and take the
                # first part as the new key.
                key = key.split(symbol)[0]
                # Exit the loop after handling the first
                # occurrence of any special symbol
                break

        # Set the item in the dictionary using the potentially modified key.
        self[key] = value

    def __setstate__(self, state):
        """
        Ensures that FlexDict can be unpickled correctly.
        """
        self.update(state)
        self.__dict__ = self

    def __dir__(self):
        """
        Ensures that auto-completion works in interactive environments.
        """
        return list(self.keys())

    def __repr__(self):
        """
        Provides a string representation of the FlexDict object, including the keys.
        """
        keys = ", ".join(self.keys())
        return f"<FlexDict with keys: {keys}>"
