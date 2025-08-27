from pathlib import Path
import re
from typing import (
    Any, List, Union, Optional, Tuple, Sequence
)
import warnings
from inspect import signature, Parameter
import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib as mpl
import scanpy as sc
import mudata as md


def make_logspace(start, stop, num, endpoint=True, dtype=None, axis=0):
    '''
    Wrapper for np.logspace but input unlogged start and stop. Because
    unlogged data is provided, base parameter is irrelevant and therefore
    not inputtable.
    
    start : array_like
        ``log(start)`` is the starting value of the sequence.
    stop : array_like
        ``log(stop)`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.


    returns: `num` samples, equally spaced on a log scale.
    ''' 

    return np.logspace(start=np.log10(start), stop=np.log10(stop), num=num, endpoint=endpoint, dtype=dtype, axis=axis)
    

def is_listlike(obj):
    return isinstance(obj, (list, set, tuple, np.ndarray, pd.Series)) and not isinstance(obj, str)


def is_valid_filename(fn: Union[str, Path]) -> bool:
    """Check if the Path object represents a valid filename.

    Parameters
    ----------
    fn : Union[str, Path]
        The filename string or Path object to check.

    Returns
    -------
    bool
        True if the filename is valid, False otherwise.
    """
    filename = Path(fn).name

    accepted_chars = bool(re.match(r'^[^\\/:*?"<>|\r\n]+$', filename))
    invalid_final_chars = (filename[-1] in [" ", "."])

    return accepted_chars and not invalid_final_chars


def is_filepath_available(filepath: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """Check if a filepath is available for writing.

    Checks for valid filename, existing directory, and if file already exists.

    Parameters
    ----------
    filepath : Union[str, Path]
        The path to check.

    Returns
    -------
    Tuple[bool, Optional[str]]
        A tuple containing a boolean indicating availability and a message
        string if not available.
    """
    filepath = Path(filepath)
    # Check if the filename is valid
    if not is_valid_filename(filepath.name):
        return (False, "Invalid filename")

    # Check if the parent directory exists
    if not filepath.parent.is_dir():
        return (False, "Invalid directory")

    # Check if the file does not exist
    if filepath.exists():
        return (False, "File already exists.")

    return (True, None)


def generate_unique_barcodes(
        n: int, 
        length: int=16) -> List[str]:
    """
    Generates a list of n unique cell barcodes using numpy.random.

    Each barcode consists of a 16-nucleotide sequence from 'A', 'C', 'G', 'T'
    followed by '-1'. The function guarantees that all generated
    barcodes in the list are unique.

    Parameters
    ----------
    n : int
        The number of unique barcodes to generate.

    Returns
    -------
    List[str]
        A list containing n unique cell barcodes.
    """
    nucleotides = np.array(list("ACGT"))
    barcodes = set()

    # The number of possible barcodes (4**16) is very large, so collisions
    # are rare for typical values of n. This loop ensures uniqueness.
    while len(barcodes) < n:
        # Generate a random 16-nucleotide sequence
        sequence_array = np.random.choice(nucleotides, length)
        barcode = "".join(sequence_array) + "-1"
        barcodes.add(barcode)

    return list(barcodes)


class UniqueDataFrame(pd.DataFrame):
    
    @property
    def _constructor(self):
        """Ensures that methods returning a new DataFrame also create a UniqueDataFrame."""
        return UniqueDataFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_uniqueness()

    def __setattr__(self, name, value):
        """
        Intercepts attribute setting to enforce uniqueness for 'columns' and 'index'
        before the assignment happens.
        """
        if name == 'columns':
            self._validate_unique_columns(value)
        if name == 'index':
            self._validate_unique_index(value)
            
        super().__setattr__(name, value)

    def _validate_uniqueness(self):
        """Helper method to check both index and columns."""
        self._validate_unique_columns(self.columns)
        self._validate_unique_index(self.index)

    def _validate_unique_columns(self, columns):
        """Raises ValueError if columns are not unique."""
        if not pd.Index(columns).is_unique:
            duplicates = pd.Index(columns)[pd.Index(columns).duplicated()].unique()
            raise ValueError(f"Columns must be unique. Found duplicate(s): {list(duplicates)}")

    def _validate_unique_index(self, index):
        """Raises ValueError if the index is not unique."""
        if not pd.Index(index).is_unique:
            duplicates = pd.Index(index)[pd.Index(index).duplicated()].unique()
            raise ValueError(f"Index must be unique. Found duplicate(s): {list(duplicates)}")


def reorder_like(a: Sequence, b: Sequence) -> List:
    """Reorders a list-like object 'a' according to the order of elements in 'b'.

    Elements in 'a' that are also present in 'b' are ordered according to their
    first appearance in 'b'. Elements in 'a' that are not in 'b' are appended
    to the end of the list, maintaining their original relative order.

    Parameters
    ----------
    a : Sequence
        The list-like object to be reordered.
    b : Sequence
        The list-like object providing the desired order.

    Returns
    -------
    List
        The reordered list.

    Examples
    --------
    >>> a = ['cat', 'dog', 'mouse', 'fish', 'bird']
    >>> b = ['fish', 'cat', 'lion']
    >>> reorder_like(a, b)
    ['fish', 'cat', 'dog', 'mouse', 'bird']
    
    >>> a = ['a', 'b', 'c', 'd']
    >>> b = ['d', 'c', 'b', 'a']
    >>> reorder_like(a, b)
    ['d', 'c', 'b', 'a']
    """
    # Create a mapping of item to its first index in b for efficient lookup.
    order_map = {val: i for i, val in reversed(list(enumerate(b)))}
    set_b = set(b)

    # Separate 'a' into two lists: one for items found in 'b', one for items not found.
    in_b = [item for item in a if item in set_b]
    not_in_b = [item for item in a if item not in set_b]

    # Sort the 'in_b' list based on the order defined by 'b'.
    in_b.sort(key=lambda x: order_map[x])

    # Combine the sorted list with the list of items not found in 'b'.
    return in_b + not_in_b


def get_config(user_config, default_config, *, protected=None):
    """
    Merges configs, protecting specified keys. Defaults to None pattern.
    """
    # Create an empty set if none was provided
    if protected is None:
        protected = set()
        
    provided = user_config or {}
    
    forbidden_keys = set(protected).intersection(provided.keys())

    if forbidden_keys:
        raise ValueError(f"Attempted to override protected settings: {', '.join(forbidden_keys)}")

    for key in provided:
        if key.endswith('_kwargs'):
            provided[key] = get_config(provided[key], default_config.get(key, {}))

    return {**default_config, **provided}


class ParamManager:
    """
    Manages, validates, and resolves complex parameter sets based on a
    pre-defined set of rules.
    """

    REQUIRED_KEYS = ['value', 'type', 'prop', 'setter', 'error']
    REQUIRED_TYPES = ['d', 'm', 'a']

    def __init__(self, defaults, func=None, error_on=[]):
        if isinstance(defaults, list):
            defaults = self.list_to_dict(defaults)
        self._validate_defaults_and_func(defaults, func)
        self.defaults = defaults
        self.error_on = error_on


    def list_to_dict(self, defaults):
        if not isinstance(defaults, list):
            raise TypeError("Defaults must be a list.")
        if not all(isinstance(i, list) and len(i) == 1 + len(self.REQUIRED_KEYS) for i in defaults):
            raise TypeError(f"Defaults must be a list of lists with {1 + len(self.REQUIRED_KEYS)} elements each.")

        df = pd.DataFrame(defaults, columns=['param'] + self.REQUIRED_KEYS)
        defaults_dict = df.set_index('param').T.to_dict()
        return defaults_dict
        

    def _validate_defaults_and_func(self, defaults, func, warn_on_kwargs=False):
            """Internal method to validate the defaults dict and check for conflicts with the func signature."""
            if not isinstance(defaults, dict):
                raise TypeError("Defaults must be a dictionary.")

            # --- Part 1: Standard validation of the defaults dictionary ---

            for name, info in defaults.items():
                if any(key not in info for key in self.REQUIRED_KEYS):
                    raise ValueError(f"Missing required keys in defaults for '{name}'.")
                if info['type'] not in self.REQUIRED_TYPES:
                    raise ValueError(f"Parameter 'type' for '{name}' must be one of {self.REQUIRED_TYPES}.")
                if not isinstance(info['error'], list):
                    raise TypeError(f"Parameter 'error' for '{name}' must be a list.")
                if info['type'] == 'm' and not callable(info['setter']):
                    raise TypeError(f"A 'setter' callable is required for meta-parameter '{name}'.")
                if info['type'] != 'm' and info['prop'] is not None:
                    raise ValueError(f"Parameter 'prop' for '{name}' must be None for non-meta parameters.")
            # --- Part 2: Validate against the function signature using inspect ---
            if not func:
                return # If no function is provided, we're done.

            if not callable(func):
                raise TypeError("If provided, 'func' must be a callable.")

            sig = signature(func)
            func_params = set(sig.parameters.keys())
            
            # Check if the function accepts arbitrary keyword arguments (**kwargs)
            accepts_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values())

            # If the function takes **kwargs, any parameter name is technically valid,
            # so this validation isn't meaningful. We can either skip or warn.
            if accepts_kwargs and warn_on_kwargs:
                warnings.warn(
                    f"Function '{func.__name__}' accepts **kwargs. "
                    "Cannot definitively validate against its signature for parameter name clashes.",
                    UserWarning
                )
                return

            for name, info in defaults.items():
                # A meta or auxiliary parameter should NOT have the same name as a
                # direct parameter in the underlying function's signature.
                if info['type'] in ['m', 'a'] and name in func_params:
                    raise ValueError(
                        f"Configuration error: Parameter '{name}' is of type '{info['type']}' "
                        f"but it conflicts with an explicit argument in the signature of '{func.__name__}'. "
                        "Meta and auxiliary parameters must have names that are distinct from the wrapped function's arguments."
                    )


    def get_params(self, user_params=None):
        """
        Resolves parameters based on defaults and user-provided inputs.
        
        This follows a two-stage process:
        1. Resolve default values, warning on any internal conflicts.
        2. Update with user values, warning on any user-specified conflicts.
        """
        if user_params is None:
            user_params = {}

        params = {}
        
        # --- Stage 1: Resolve Default Values ---
        set_by_default = {}
        # First, process default meta-parameters and propogate auxiliary parameters
        for name, info in self.defaults.items():
            if info['type'] == 'm' and info['value'] is not None:
                new_params = info['setter'](info['value'])
                if info['prop'] is True:
                    params[name] = info['value']
                    set_by_default[name] = name
                for k, v in new_params.items():
                    params[k] = v
                    set_by_default[k] = name
            elif info['type'] in ['a']:
                params[name] = info['value']
                set_by_default[name] = name
        
        # Second, process default direct parameters
        for name, info in self.defaults.items():
            if info['type'] in ['d']:
                # Warn if a default 'd' or 'a' conflicts with a default 'm'
                if name in set_by_default and params[name] != info['value']:
                    warnings.warn(
                        f"Conflicting defaults for '{name}'. "
                        f"Was '{params[name]}' set by meta-param '{set_by_default[name]}', "
                        f"but default value is '{info['value']}'. Using '{info['value']}'.",
                        UserWarning
                    )
                params[name] = info['value']
                set_by_default[name] = name

        # --- Stage 2: Resolve User-Provided Values ---
        set_by_user = {}
        
        # First, process user-provided meta-parameters
        for name, value in user_params.items():
            if self.defaults.get(name, {}).get('type') == 'm':
                if self.defaults.get(name, {}).get('prop') is True:
                    params[name] = value
                    set_by_user[name] = name
                new_params = self.defaults[name]['setter'](value)
                for k, v in new_params.items():
                    # The setter's keys are what get updated.
                    params[k] = v
                    set_by_user[k] = name

        # Second, process user-provided direct parameters
        for name, value in user_params.items():
            if self.defaults.get(name, {}).get('type') != 'm':
                if name in set_by_user and params[name] != value:
                    warnings.warn(
                        f"Conflicting user settings for '{name}'. "
                        f"Was '{params[name]}' set by '{set_by_user[name]}', "
                        f"but is being overridden by explicit setting. Using '{value}'.",
                        UserWarning
                    )
                params[name] = value
                set_by_user[name] = name
                    
        # Add any extra user parameters that are not defined in defaults
        extras = {k: v for k, v in user_params.items() if k not in self.defaults}
        params.update(extras)

        # --- Stage 3: Perform Final Error Checking ---
        self._check_for_errors(params, user_params)

        return params
    

    def _check_for_errors(self, final_params, user_params):
        """Checks for user-defined error conditions on the final parameter set."""
        for param in final_params:
            if param in self.error_on:
                raise ValueError(self.error_on[param])
        for name, info in self.defaults.items():
            if name not in user_params:
                continue

            for param_to_check, condition in info['error']:
                if param_to_check not in user_params:
                    continue
                
                rule = condition['on']
                message = condition['message']
                
                should_error = False
                if rule == 'any':
                    if final_params.get(param_to_check) != self.defaults.get(param_to_check, {}).get('value'):
                        should_error = True
                elif callable(rule):
                    if rule(final_params.get(name), final_params.get(param_to_check)):
                        should_error = True
                        
                if should_error:
                    raise ValueError(f"Parameter conflict: {message}")
                

def get_mask(df, filters):

    """
    Provides a boolean mask to filter a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        filters (dict): A dictionary where keys are column names
                        and values are lists of items to check for
                        inclusion in that column.
    """
    # Start with a mask that includes all rows
    mask = pd.Series(True, index=df.index)

    # Cumulatively apply each filter
    for col, values in filters.items():
        mask &= df[col].isin(values)

    return mask


def pklop(*args):
    """
    Saves or loads a pickle file based on the number of arguments.

    Args:
        *args: Variable length argument list.
            - To save: Provide the object and the file path (e.g., pickle_op(my_obj, 'file.pkl')).
            - To load: Provide only the file path (e.g., my_obj = pickle_op('file.pkl')).

    Returns:
        The loaded object if in 'load' mode, otherwise None.
        
    Raises:
        ValueError: If the number of arguments is not 1 or 2.
    """

    def _check_path(path):
        if not isinstance(path, (str, Path)):
            raise ValueError("Path must be a string or Path object.")
        elif isinstance(path, str):
            path = Path(path)
        if not path.suffix == '.pkl':
            raise ValueError("Path must end with '.pkl' extension.")
        return path

    if len(args) == 2:
        # Save mode: (object, path)
        obj, path = args
        path = _check_path(path)

        with open(path, 'wb') as f:
            pkl.dump(obj, f)

    elif len(args) == 1:
        # Load mode: (path)
        path = args[0]
        path = _check_path(path)
        
        with open(path, 'rb') as f:
            return pkl.load(f)
    else:
        raise ValueError("Function accepts 1 argument (path) to load or 2 arguments (object, path) to save.")