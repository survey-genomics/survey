# Built-ins
import warnings
import itertools as it
from typing import Any, Dict, List, Optional, Tuple, Union

# Standard libs
import numpy as np
import pandas as pd

# Single-cell libs
import scanpy as sc


class QuietScanpyLoad:
    """
    A context manager for temporarily setting the verbosity level of scanpy 
    and suppressing a specific UserWarning about non-unique variable names.

    Usage:
    with QuietScanpyLoad(0):
        # Your code here

    This will set the verbosity level to 0 and suppress some warnings (see below).
    When the with block is exited, the original verbosity level is restored and 
    the warnings are reset to their original state.

    Parameters
    ----------
    new_verbosity : int
        The verbosity level to use within the context.

    Attributes
    ----------
    original_verbosity : int
        The original verbosity level, saved when the context is entered.

    Notes
    -----
    This context manager is intended to be used with scanpy's read/write.
    Current ignored warnings:

    warnings.filterwarnings("ignore", category=UserWarning, message=".*Variable names are not unique.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*dtype argument is deprecated.*")

    """
    def __init__(self, new_verbosity: int) -> None:
        self.new_verbosity = new_verbosity
        self.original_verbosity = None

    def __enter__(self) -> None:
        """
        Sets the scanpy verbosity and suppresses warnings upon entering the context.
        """
        self.original_verbosity = sc.settings.verbosity
        sc.settings.verbosity = self.new_verbosity
        
        if self.new_verbosity < 1:
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Variable names are not unique.*")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*dtype argument is deprecated.*")

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """
        Restores the original scanpy verbosity and warning filters upon exiting the context.
        """
        sc.settings.verbosity = self.original_verbosity
        warnings.resetwarnings()


def filter_var(df: pd.DataFrame, 
               lib_tag_dict: Dict[str, Dict[str, str]], 
               mod: str) -> pd.DataFrame:
    """Filter a .var DataFrame based on the given modality.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter (e.g., `adata.var`).
    lib_tag_dict : dict
        A dictionary specifying the filtering criteria for a given library tag.
    mod : str
        The modality to look for in the dictionary.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame.

    Notes
    -----
    var string codes:
    - `id`: filter based on index
    - `gi`: filter based on `gene_ids` column
    - `ft`: filter based on `feature_types` column

    string_methods filtering:
    - `sw`: starts with
    - `ew`: ends with
    - `cn`: contains

    Examples
    --------
    >>> lib_tag_dict_example1 = {'hto': {'id_sw': 'tsb_hto'}}
    >>> lib_tag_dict_example2 = {
    ...     'sbc': {'id_sw': 'sbc', 'ft': 'Antibody Capture'},
    ...     'adt': {'gi_sw': 'A', 'ft': 'Antibody Capture'}
    ... }
    >>> # filtered_var = filter_var(adatas[lib_tag].var, lib_tag_dict=lib_tag_dict_example1, mod='hto')
    """

    # Mapping for special string codes to .var columns
    col_map = {
        "id": (
            df.index.name if df.index.name else None
        ),  # Ensuring we map 'id' to the index name, not its values
        "gi": "gene_ids",
        "ft": "feature_types",
    }

    # Retrieve the criteria for filtering
    criteria = lib_tag_dict.get(mod, {})

    if len(criteria) == 0:
        raise ValueError(f"No filtering criteria found for modality '{mod}' in the provided dictionary.")

    # Loop through the criteria to filter the dataframe
    for string_code, filter_string in criteria.items():
        code_parts = string_code.split("_")
        operation = code_parts[-1] if len(code_parts) > 1 else None
        col_name = col_map.get(code_parts[0], code_parts[0])

        if operation == "sw":
            if col_name == df.index.name:
                df = df[df.index.str.startswith(filter_string)]
            else:
                df = df[df[col_name].str.startswith(filter_string)]
        elif operation == "ew":
            if col_name == df.index.name:
                df = df[df.index.str.endswith(filter_string)]
            else:
                df = df[df[col_name].str.endswith(filter_string)]
        elif operation == "cn":
            if col_name == df.index.name:
                df = df[df.index.str.contains(filter_string)]
            else:
                df = df[df[col_name].str.contains(filter_string)]
        else:
            # If no special string code is present, assume exact match
            if col_name == df.index.name:
                df = df[df.index == filter_string]
            else:
                df = df[df[col_name] == filter_string]

    return df


def clr_normalize_column(x: np.ndarray) -> np.ndarray:
    """Apply centered log-ratio (CLR) normalization to a column.

    This function computes the geometric mean of the positive values in the input
    array and scales the values such that the sum of squares of the scaled values
    equals the number of non-zero entries.

    Parameters
    ----------
    x : np.ndarray
        Input array (column).

    Returns
    -------
    np.ndarray
        CLR-normalized array.
    """
    # Calculate the geometric mean of the positive elements of the array
    # The +1 in len(x+1) is to avoid division by zero for empty arrays, though
    # this case might not be robustly handled.
    normed_column = np.log1p((x) / (np.exp(sum(np.log1p((x)[x > 0])) / len(x + 1))))
    return normed_column


def clr_normalize(x: np.ndarray) -> np.ndarray:
    """Apply centered log-ratio (CLR) normalization to a matrix row-wise.

    Parameters
    ----------
    x : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        CLR-normalized matrix.
    """
    normed_matrix = np.apply_along_axis(clr_normalize_column, 1, x)
    return normed_matrix


def convert_to_categorical(obs_df: pd.DataFrame, 
                           col: str, 
                           via: Optional[str] = None, 
                           verbose: bool = True) -> pd.DataFrame:
    """
    Converts a column in a DataFrame to the 'category' dtype.

    Optionally, the column can be converted to an intermediate data type
    before being converted to a category.

    Parameters
    ----------
    obs_df : pd.DataFrame
        The DataFrame containing the column to convert.
    col : str
        The name of the column to convert.
    via : str, optional
        An intermediate data type to convert the column to first (e.g., 'str').
        If None, no intermediate conversion is performed. Default is None.
    verbose : bool, optional
        If True, prints a message indicating which column is being converted.
        Default is True.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified column converted to a categorical dtype.
    """
    # warnings.warn(f"Converting column '{col}' to categorical dtype.")
    if verbose:
        print(f"Converting column '{col}' to categorical dtype.")
    if via is not None:
        obs_df[col] = obs_df[col].astype(via)
    obs_df[col] = obs_df[col].astype('category')
    return obs_df

    
def get_color_mapper(cats: List[Any], 
                     cycle: bool = False) -> Dict[Any, str]:
    """
    Creates a mapping from categories to colors using scanpy's default palettes.

    The function selects a color palette based on the number of categories.
    If the number of categories exceeds the available colors in the chosen
    palette, it can cycle through the colors.

    Parameters
    ----------
    cats : list
        A list of unique categories to be mapped to colors.
    cycle : bool, optional
        If True and the number of categories is greater than 20, the default_20
        palette will be cycled to generate enough colors. Default is False.

    Returns
    -------
    dict
        A dictionary mapping each category to a color hex string.
    """
    if len(cats) <= 20 or cycle:
        mapper = {cat: color for cat, color in zip(cats, it.cycle(sc.pl.palettes.default_20))}
    elif len(cats) <= 28:
        mapper = {cat: color for cat, color in zip(cats, sc.pl.palettes.default_28)}
    else:
        mapper = {cat: color for cat, color in zip(cats, it.cycle(sc.pl.palettes.default_102))}
    return mapper


def freq_table(data: Union[sc.AnnData, pd.DataFrame],
               cols: Tuple[Union[str, int], Union[str, int]],
               observed: bool = True) -> pd.DataFrame:
    '''
    Create a frequency table for two categorical columns in an AnnData object.
    Note that reported frequencies will not include any rows with NaN values in either 
    column.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing the observation data.
    cols : tuple of str or int
        A tuple containing the names or indices of the two columns to analyze.
    observed : bool, optional
        If True, only the observed categories in the data will be considered.
        Passed to `groupby` method. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the frequency table, with one column as the index
        and the other as columns.
    '''
    if not isinstance(cols, (list, tuple)) or len(cols) != 2:
        raise ValueError("`cols` must be a list or tuple of exactly two elements.")
    if not isinstance(data, (sc.AnnData, pd.DataFrame)):
        raise ValueError("`data` must be an AnnData object or a DataFrame.")
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = data.obs
    c1, c2 = cols
    return df[cols].groupby(cols, observed=observed).size().unstack(fill_value=0)


