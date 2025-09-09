# Built-ins
from pathlib import Path
import re
from typing import (
    List, Union, Optional, Tuple, Sequence, Any, Dict, Callable 
)
import warnings
from inspect import signature, Parameter
import pickle as pkl
from numbers import Number

# Standard libs
import numpy as np
import pandas as pd


def make_logspace(start: Number,
                  stop: Number,
                  num: int,
                  endpoint: bool = True,
                  dtype: Optional[np.dtype] = None,
                  axis: int = 0) -> np.ndarray:
    """
    Creates a sequence of numbers that are evenly spaced on a log scale.

    This is a wrapper for `numpy.logspace` that accepts the start and stop
    values in their original (un-logged) scale.

    Parameters
    ----------
    start : Number
        The starting value of the sequence. Must be > 0.
    stop : Number
        The final value of the sequence, unless `endpoint` is False.
    num : int
        The number of samples to generate.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
    dtype : np.dtype, optional
        The data type of the output array.
    axis : int, optional
        The axis in the result to store the samples.

    Returns
    -------
    np.ndarray
        An array of `num` samples, equally spaced on a log scale.
    """

    return np.logspace(start=np.log10(start), stop=np.log10(stop), num=num, endpoint=endpoint, dtype=dtype, axis=axis)
    

def is_listlike(obj: Any) -> bool:
    """
    Checks if an object is list-like.

    An object is considered list-like if it is an instance of list, set, tuple,
    numpy.ndarray, or pandas.Series, but not a string.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is list-like, False otherwise.
    """
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


def generate_unique_barcodes(n: int,
                             length: int = 16) -> List[str]:
    """
    Generates a list of n unique random DNA barcodes.

    Each barcode consists of a sequence of 'A', 'C', 'G', 'T' of a specified
    length, followed by '-1'. The function guarantees that all generated
    barcodes in the list are unique.

    Parameters
    ----------
    n : int
        The number of unique barcodes to generate.
    length : int, optional
        The length of the nucleotide sequence part of the barcode.

    Returns
    -------
    List[str]
        A list containing n unique barcodes.
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
    """
    A pandas DataFrame subclass that enforces unique index and columns.

    This class raises a `ValueError` if an attempt is made to create or modify
    the DataFrame with duplicate values in either the index or the columns.
    """
    
    @property
    def _constructor(self) -> Callable[..., 'UniqueDataFrame']:
        """Ensures that methods returning a new DataFrame also create a UniqueDataFrame."""
        return UniqueDataFrame

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the UniqueDataFrame and validates uniqueness.
        """
        super().__init__(*args, **kwargs)
        self._validate_uniqueness()

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Intercepts attribute setting to enforce uniqueness for 'columns' and 'index'.
        """
        if name == 'columns':
            self._validate_unique_columns(value)
        if name == 'index':
            self._validate_unique_index(value)
            
        super().__setattr__(name, value)

    def _validate_uniqueness(self) -> None:
        """Helper method to check both index and columns for uniqueness."""
        self._validate_unique_columns(self.columns)
        self._validate_unique_index(self.index)

    def _validate_unique_columns(self, columns: pd.Index) -> None:
        """Raises ValueError if columns are not unique."""
        if not pd.Index(columns).is_unique:
            duplicates = pd.Index(columns)[pd.Index(columns).duplicated()].unique()
            raise ValueError(f"Columns must be unique. Found duplicate(s): {list(duplicates)}")

    def _validate_unique_index(self, index: pd.Index) -> None:
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


def get_config(user_config: Optional[Dict],
               default_config: Dict,
               *,
               protected: Optional[set] = None) -> Dict:
    """
    Merges a user configuration with a default configuration.

    This function safely combines two dictionaries, allowing defaults to be
    overridden by user settings, except for keys specified as `protected`.

    Parameters
    ----------
    user_config : dict, optional
        The user-provided configuration dictionary.
    default_config : dict
        The default configuration dictionary.
    protected : set, optional
        A set of keys in the default configuration that cannot be overridden.

    Returns
    -------
    dict
        The merged configuration dictionary.

    Raises
    ------
    ValueError
        If `user_config` attempts to override a protected key.
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
    Manages, validates, and resolves complex parameter sets for function calls.

    This class provides a structured way to handle function parameters by
    defining defaults, validation rules, and relationships between different
    types of parameters (direct, meta, auxiliary).

    Parameters
    ----------
    defaults : list or dict
        A structure defining the parameters, their default values, types,
        and validation rules.
    func : callable, optional
        The target function whose signature will be used for validation.
    error_on : list, optional
        A list of parameter names that are forbidden in user input.
    """

    REQUIRED_KEYS = ['value', 'type', 'prop', 'setter', 'error']
    REQUIRED_TYPES = ['d', 'm', 'a']

    def __init__(self,
                 defaults: Union[List, Dict],
                 func: Optional[Callable] = None,
                 error_on: List = []) -> None:
        if isinstance(defaults, list):
            defaults = self.list_to_dict(defaults)
        self._validate_defaults_and_func(defaults, func)
        self.defaults = defaults
        self.error_on = error_on


    def list_to_dict(self, defaults: List) -> Dict:
        """
        Converts a list-based defaults definition to a dictionary.

        Parameters
        ----------
        defaults : list
            The list of parameter definitions.

        Returns
        -------
        dict
            The dictionary of parameter definitions.
        """
        if not isinstance(defaults, list):
            raise TypeError("Defaults must be a list.")
        if not all(isinstance(i, list) and len(i) == 1 + len(self.REQUIRED_KEYS) for i in defaults):
            raise TypeError(f"Defaults must be a list of lists with {1 + len(self.REQUIRED_KEYS)} elements each.")

        df = pd.DataFrame(defaults, columns=['param'] + self.REQUIRED_KEYS)
        defaults_dict = df.set_index('param').T.to_dict()
        return defaults_dict
        

    def _validate_defaults_and_func(self,
                                    defaults: Dict,
                                    func: Optional[Callable],
                                    warn_on_kwargs: bool = False) -> None:
        """
        Validates the defaults dictionary and checks for conflicts with a function signature.

        Parameters
        ----------
        defaults : dict
            The defaults dictionary to validate.
        func : callable, optional
            The function to validate against.
        warn_on_kwargs : bool, optional
            If True, warns if the function accepts `**kwargs`, which can limit
            validation accuracy.
        """
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


    def get_params(self, user_params: Optional[Dict] = None) -> Dict:
        """
        Resolves the final parameter set based on defaults and user input.

        This method follows a two-stage process:
        1. Resolve default values, handling meta-parameters and their effects.
        2. Update with user-provided values, overriding defaults and handling
           user-specified meta-parameters.

        Parameters
        ----------
        user_params : dict, optional
            A dictionary of user-provided parameters.

        Returns
        -------
        dict
            The final, resolved dictionary of parameters to be passed to a function.
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
    

    def _check_for_errors(self, final_params: Dict, user_params: Dict) -> None:
        """
        Checks the final parameter set against user-defined error conditions.

        Parameters
        ----------
        final_params : dict
            The fully resolved parameter dictionary.
        user_params : dict
            The original user-provided parameters.

        Raises
        ------
        ValueError
            If a forbidden parameter is found or if a conflict rule is triggered.
        """
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
                

def get_mask(df: pd.DataFrame, filters: Dict[str, List]) -> pd.Series:
    """
    Creates a boolean mask to filter a DataFrame based on multiple criteria.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be filtered.
    filters : dict
        A dictionary where keys are column names and values are lists of
        items to include for that column.

    Returns
    -------
    pd.Series
        A boolean Series that can be used to index the DataFrame.
    """
    # Start with a mask that includes all rows
    mask = pd.Series(True, index=df.index)

    # Cumulatively apply each filter
    for col, values in filters.items():
        mask &= df[col].isin(values)

    return mask


def pklop(*args: Any) -> Optional[Any]:
    """
    Saves or loads a Python object to/from a pickle file.

    This function acts as a simple wrapper around the `pickle` module.
    - To save: `pklop(my_object, 'file.pkl')`
    - To load: `my_object = pklop('file.pkl')`

    Parameters
    ----------
    *args : Any
        - If one argument is provided, it is treated as the file path to load from.
        - If two arguments are provided, they are treated as `(object_to_save, file_path)`.

    Returns
    -------
    Any or None
        The loaded object if in 'load' mode, otherwise None.

    Raises
    ------
    ValueError
        If the number of arguments is not 1 or 2, or if the file path is invalid.
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
    
    
def normalize(arr: np.ndarray,
              filter: Optional[Tuple[Optional[float], Optional[float]]] = None,
              clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
              lower: Optional[float] = None,
              upper: Optional[float] = None,) -> np.ndarray:
    """
    Filters, clips, and/or normalizes an array to a specified range.

    The function applies operations in the following order:
    1. **Filter**: Values outside the `filter` range are set to `np.nan`.
    2. **Clip**: Values are clipped to the `clip` range.
    3. **Normalize**: The resulting values are rescaled to a new target range
       defined by `lower` and `upper`.

    Parameters
    ----------
    arr : np.ndarray
        The input NumPy array to normalize.
    filter : tuple of (float or None, float or None), optional
        A tuple `(min_filter, max_filter)` to filter the array. Values outside
        this range will be replaced with `np.nan` before any other
        operations. If a value is None, no filtering is performed on that side.
    clip : tuple of (float or None, float or None), optional
        A tuple `(min_clip, max_clip)` to clip the array values after
        filtering but before normalization. If a value is None, no clipping is
        performed on that side. If `clip` is `(None, None)`, a warning is
        issued. Defaults to None (no clipping).
    lower : float, optional
        The lower bound of the target normalization range. Defaults to 0.0.
    upper : float, optional
        The upper bound of the target normalization range. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        The filtered, clipped, and normalized NumPy array.

    Raises
    ------
    ValueError
        - If `filter` or `clip` is not a tuple of length 2.
        - If an upper bound for `filter` or `clip` is not greater than its lower bound.
        - If the final normalization range is invalid (e.g., upper < lower).

    Warns
    -----
    UserWarning
        - If `clip` is provided as `(None, None)`, as no clipping will occur.
        - If `filter` is provided as `(None, None)`, as no filtering will occur.

    Examples
    --------
    >>> data = np.array([-10, 0, 50, 100, 120])

    # Filter out values < 0 and > 100, then normalize to [0, 1]
    >>> normalize(data, filter=(0, 100))
    array([ nan, 0. , 0.5, 1. ,  nan])

    # Clip to [0, 100] then normalize to [0, 1]
    >>> normalize(data, clip=(0, 100))
    array([0.  , 0.  , 0.5 , 1.  , 1.  ])

    # Clip and normalize to a new range using lower/upper
    >>> normalize(data, clip=(0, 100), lower=10, upper=20)
    array([10. , 10. , 15. , 20. , 20. ])
    """

    if arr.size == 0:
        return arr.copy() # Return a copy of the empty array

    # --- Input Parameter Validation ---
    work_arr = arr.copy().astype(float) # Use float to accommodate np.nan

    # --- Handle Filtering ---
    if filter is not None:
        if not isinstance(filter, tuple) or len(filter) != 2:
            raise ValueError("`filter` must be a tuple of length 2, like (min, max).")
        min_filter, max_filter = filter
        if min_filter is not None and max_filter is not None and max_filter <= min_filter:
            raise ValueError(f"In `filter`, max must be > min, but got min={min_filter}, max={max_filter}")
        if min_filter is not None:
            work_arr[work_arr < min_filter] = np.nan
        if max_filter is not None:
            work_arr[work_arr > max_filter] = np.nan

    # --- Handle Clipping ---
    if clip is not None:
        if not isinstance(clip, tuple) or len(clip) != 2:
            raise ValueError("`clip` must be a tuple of length 2, like (min, max).")
        min_clip, max_clip = clip
        if min_clip is not None and max_clip is not None and max_clip <= min_clip:
            raise ValueError(f"In `clip`, max must be > min, but got min={min_clip}, max={max_clip}")
        if min_clip is None and max_clip is None:
            import warnings
            warnings.warn("`clip` was set to (None, None). No clipping will be performed.")
        work_arr = np.clip(work_arr, min_clip, max_clip)

    # --- Determine Normalization Source Range (from filtered/clipped data) ---
    min_val = np.nanmin(work_arr)
    max_val = np.nanmax(work_arr)
    range_val = max_val - min_val

    # --- Determine Normalization Target Range ---
    target_lower = lower if lower is not None else 0.0
    target_upper = upper if upper is not None else 1.0
    target_spread = target_upper - target_lower

    if target_spread < 0:
        raise ValueError(
            f"Invalid target range: spread cannot be negative. "
            f"Derived from lower={target_lower}, upper={target_upper}"
        )

    if range_val == 0:
        # Calculate the midpoint of the target range
        midpoint = target_lower + target_spread / 2.0
        
        out = np.full_like(work_arr, midpoint)
        out[np.isnan(work_arr)] = np.nan
        return out

    # --- Perform Normalization ---
    # Scale to [0, 1] first
    normalized_arr = (work_arr - min_val) / range_val
    
    # Then scale to target range
    return (normalized_arr * target_spread) + target_lower


def get_functional_dependency(df: pd.DataFrame, 
                              cols: Tuple[str, str]) -> Dict[Any, Any]:
    """
    Checks for a functional dependency and returns the mapping using a groupby approach.

    A functional dependency exists if each unique value in the independent
    column maps to exactly one unique value in the dependent column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    cols : Tuple[str, str]
        A 2-tuple of column names (independent_col, dependent_col). The
        independent column provides the keys of the mapping, and the dependent
        column provides the values.

    Returns
    -------
    Dict[Any, Any]
        A dictionary mapping unique values from `independent_col` to their
        corresponding values in `dependent_col`.

    Raises
    ------
    ValueError
        If columns are not in the DataFrame or if a functional
        dependency does not exist.
    """
    independent_col, dependent_col = cols
    if independent_col not in df.columns or dependent_col not in df.columns:
        raise ValueError("One or both columns not in DataFrame.")

    # Group by the independent column and count unique dependent values
    counts = df.groupby(independent_col, observed=True)[dependent_col].nunique()

    # Check if any group has more than one unique dependent value
    if (counts > 1).any():
        offending_keys = counts[counts > 1].index.tolist()
        raise ValueError(
            f"No functional dependency from '{independent_col}' to '{dependent_col}'. "
            f"Keys with multiple values: {offending_keys}"
        )

    # If dependency holds, create the mapping from the unique pairs
    unique_pairs = df[[independent_col, dependent_col]].drop_duplicates()
    mapping = pd.Series(unique_pairs[dependent_col].values, 
                        index=unique_pairs[independent_col]).to_dict()
    return mapping