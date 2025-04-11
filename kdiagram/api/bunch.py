# -*- coding: utf-8 -*-


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
        special_symbols = ['**', '%%', '&&', '||', '$$']
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
        keys = ', '.join(self.keys())
        return f"<FlexDict with keys: {keys}>"
