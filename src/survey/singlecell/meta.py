# Built-ins
from numbers import Number
from typing import Any, Dict, List, Optional, Union, Tuple

# Standard libs
import pandas as pd

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.genutils import UniqueDataFrame, is_listlike
from survey.singlecell.scutils import get_color_mapper, convert_to_categorical

# Design/Implementation
# Functions are single purpose and explicit about what they are checking and doing.
# Convenience functions will *mostly* string together single-purpose functions.
# Choosing to use a regular Index instead of a CategoricalIndex for the index of 
# UniqueDataFrame, since it already represents the categories of the categorical column.
# "Explicit is better than implicit"
#   - All functions will use validators to check data is well-formed and consistent before proceeding.
#   - It is up to user to make sure the data is well-formed and consistent before calling functions.
#   - Will only convert columns to categorical implicitly with warning.
#   - Will only add colors if they exist already with overwrite=True.
#
# Nomenclature
# - `key(s)`
#   - refers to columns in adata.obs, which should be categorical columns if metadata is to be added.
#   - it also refers to the keys of the metadata dictionary (adata.uns['meta']).
#   - a key in adata.uns['meta'] does not exist without a metadata DataFrame in adata.uns['meta'][key].
#     therefore, a key can also implicity refer to its value, the metadata DataFrame in adata.uns['meta'][key].
# - `cat(s)`
#   - refers to the categories of the categorical column in adata.obs.
#   - it also refers to the values in the index of the metadata DataFrame in adata.uns['meta'][key].
# - `prop(s)`
#   - refers to the metadata properties, which are the columns in the metadata DataFrame in adata.uns['meta'][key].
# 
# Consistency States
# - adata is always checked, and it is assumed .obs exists
# - adata.obs[col] can exist or not
# - adata.obs[col] can be categorical or not
# - uns['meta'] can exist or not
# - uns['meta'][col] can exist or not
# - uns['meta'][col][prop] can exist or not
# - uns['meta'][col].index can be consistent with adata.obs[col].cat.categories or not
# Each of these states should have its own validation function. And the functions should
# be explicit about what they are checking and when (in the docstring).

PROTECTED_KEYS = ['color']
MAX_LEN_KEYS_REPR = 40


# Validator Functions

def is_adata(input: Any, error: bool = True) -> bool:
    """
    Validates if the input is an AnnData object.

    Parameters
    ----------
    input : Any
        The object to validate.
    error : bool, optional
        If True, raises a TypeError if validation fails. If False, returns
        a boolean. Default is True.

    Returns
    -------
    bool
        True if `input` is an AnnData object, False otherwise (if `error` is False).

    Raises
    ------
    TypeError
        If `input` is not an AnnData object and `error` is True.
    """
    if not isinstance(input, sc.AnnData):
        if error:
            raise TypeError("Input must be an instance of sc.AnnData")
        else:
            return False    
    return True


def key_exists(adata: sc.AnnData, key: str, error: bool = True) -> bool:
    """
    Validates if a key exists as a column in `adata.obs`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to check.
    key : str
        The column name to look for in `adata.obs`.
    error : bool, optional
        If True, raises a ValueError if the key is not found. If False,
        returns a boolean. Default is True.

    Returns
    -------
    bool
        True if the key exists, False otherwise (if `error` is False).

    Raises
    ------
    ValueError
        If the key is not in `adata.obs.columns` and `error` is True.
    """
    if key not in adata.obs.columns:
        if error:
            raise ValueError(f"Column '{key}' not found in adata.obs. Please ensure the column exists.")
        else:
            return False
    return True


def is_key_categorical(adata: sc.AnnData, key: str, error: bool = True) -> bool:
    """
    Validates if a column in `adata.obs` is of categorical dtype.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to check.
    key : str
        The column name in `adata.obs` to validate.
    error : bool, optional
        If True, raises a TypeError if the column is not categorical.
        If False, returns a boolean. Default is True.

    Returns
    -------
    bool
        True if the column is categorical, False otherwise (if `error` is False).

    Raises
    ------
    TypeError
        If `adata.obs[key]` is not a categorical series and `error` is True.
    """
    if not isinstance(adata.obs[key].dtype, pd.CategoricalDtype):
        if error:
            raise TypeError(f"Column '{key}' is not categorical. Please convert it to categorical before proceeding.")
        else:
            return False
    return True


def meta_exists(adata: sc.AnnData, error: bool = True) -> bool:
    """
    Validates if `adata.uns['meta']` exists.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to check.
    error : bool, optional
        If True, raises a ValueError if `adata.uns['meta']` does not exist.
        If False, returns a boolean. Default is True.

    Returns
    -------
    bool
        True if `adata.uns['meta']` exists, False otherwise (if `error` is False).

    Raises
    ------
    ValueError
        If `adata.uns['meta']` does not exist and `error` is True.
    """
    if 'meta' not in adata.uns:
        if error:
            raise ValueError(
                "Input AnnData object does not contain 'meta' in .uns. Please use "
                "survey.singlecell.meta.add_meta() for easy category-level metadata "
                "management. If coming from scanpy-stored colors, use add_meta_from_scanpy()."
        )
        else:
            return False
    return True


def meta_key_exists(adata: sc.AnnData, key: str, error: bool = True) -> bool:
    """
    Validates if a key exists within `adata.uns['meta']`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing the metadata.
    key : str
        The metadata key to check for in `adata.uns['meta']`.
    error : bool, optional
        If True, raises a ValueError if the key does not exist. If False,
        returns a boolean. Default is True.

    Returns
    -------
    bool
        True if the key exists in `adata.uns['meta']`, False otherwise (if `error` is False).

    Raises
    ------
    ValueError
        If the key is not in `adata.uns['meta']` and `error` is True.
    """
    if key not in adata.uns['meta']:
        if error:
            raise ValueError(f"Metadata for key '{key}' not found in adata.uns['meta']. "
                             "Please add it using add_meta().")
        else:
            return False
    return True


def meta_key_prop_exists(adata: sc.AnnData, 
                         key: str, 
                         prop: str, 
                         error: bool = True) -> bool:
    """
    Validates if a property exists for a given key in `adata.uns['meta']`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing the metadata.
    key : str
        The metadata key.
    prop : str
        The property (column) to check for in `adata.uns['meta'][key]`.
    error : bool, optional
        If True, raises a ValueError if the property does not exist.
        If False, returns a boolean. Default is True.

    Returns
    -------
    bool
        True if the property exists, False otherwise (if `error` is False).

    Raises
    ------
    ValueError
        If the property is not in `adata.uns['meta'][key].columns` and `error` is True.
    """
    if prop not in adata.uns['meta'][key].columns:
        if error:
            raise ValueError(f"Metadata for key '{key}' with property '{prop}' not found in adata.uns['meta'][key]. "
                             "Please add it using add_metadata().")
        else:
            return False
    return True


def is_meta_key_consistent(adata: sc.AnnData, key: str, error: bool = True) -> bool:
    """
    Validates if `adata.obs` categories and `adata.uns['meta']` index are consistent.

    Checks if the sorted categories of `adata.obs[key]` match the sorted
    index of `adata.uns['meta'][key]`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object.
    key : str
        The key to check for consistency.
    error : bool, optional
        If True, raises a ValueError if inconsistent. If False, returns a boolean.
        Default is True.

    Returns
    -------
    bool
        True if consistent, False otherwise (if `error` is False).

    Raises
    ------
    ValueError
        If the categories and index are inconsistent and `error` is True.
    """
    if sorted(adata.uns['meta'][key].index) != sorted(adata.obs[key].cat.categories):
        if error:
            raise ValueError(
                f"Column '{key}' in adata.obs does not match the categories "
                f"in adata.uns['meta']['{key}']."
            )
        else:
            return False
    return True


# Validate Multiple

def validate_meta(adata: sc.AnnData, 
                  key: str, 
                  prop: str, 
                  error: bool = True) -> Optional[Dict[str, bool]]:
    '''
    Validate the metadata structure and consistency for a given key and property.

    This function runs a suite of validation checks. If `error` is True, it
    will raise an exception on the first failure. If `error` is False, it
    will run all checks and return a dictionary of their boolean results.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to validate.
    key : str
        The metadata key to validate.
    prop : str
        The metadata property to validate.
    error : bool, optional
        If True, raises an exception on the first validation failure. If False,
        returns a dictionary of validation results. Default is True.

    Returns
    -------
    dict or None
        If `error` is False, a dictionary where keys are check names and values
        are booleans indicating success or failure. If `error` is True and all
        checks pass, returns None.
    '''
    errors = {}
    errors['is_adata'] = is_adata(adata, error=error)
    errors['key_exists'] = key_exists(adata, key, error=error)
    errors['is_key_categorical'] = is_key_categorical(adata, key, error=error)
    errors['meta_exists'] = meta_exists(adata, error=error)
    errors['meta_key_exists'] = meta_key_exists(adata, key, error=error)
    errors['meta_key_prop_exists'] = meta_key_prop_exists(adata, key, prop, error=error)
    errors['is_meta_key_consistent'] = is_meta_key_consistent(adata, key, error=error)

    if not error:
        return errors

    # If we reached this point, it means all checks passed
    return None


# Basic Operation Functions

def add_meta(adata: sc.AnnData) -> None:
    '''
    Initializes `adata.uns['meta']` as an empty dictionary if it does not exist.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    '''
    is_adata(adata, error=True)
    if 'meta' not in adata.uns:
        adata.uns['meta'] = {}
    return


def add_meta_keys(adata: sc.AnnData, keys: Union[str, List[str]]) -> None:
    '''
    Adds one or more keys to `adata.uns['meta']`.

    For each key, if it's not already categorical in `adata.obs`, it will be
    converted. An empty `UniqueDataFrame` is created for each new key in
    `adata.uns['meta']`, indexed by the categories from `adata.obs[key]`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    keys : str or list of str
        The key or keys to add to the metadata structure.
    '''
    is_adata(adata, error=True)
    meta_exists(adata, error=True)

    if not is_listlike(keys):
        keys = [keys]

    for key in keys:
        key_exists(adata, key, error=True)

    for key in keys:
        if not is_key_categorical(adata, key, error=False):
            adata.obs = convert_to_categorical(adata.obs, key)
        if key not in adata.uns['meta']:
            adata.uns['meta'][key] = UniqueDataFrame(index=pd.Index(adata.obs[key].cat.categories))
    return


def reset_meta_keys(adata: sc.AnnData, keys: Union[str, List[str]]) -> None:
    '''
    Resets the metadata for one or more keys in `adata.uns['meta']`.

    This function replaces any existing metadata for the specified keys with
    a new, empty `UniqueDataFrame`, effectively clearing all properties for
    that key.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    keys : str or list of str
        The key or keys to reset.
    '''
    is_adata(adata, error=True)
    meta_exists(adata, error=True)

    if not is_listlike(keys):
        keys = [keys]

    for key in keys:
        key_exists(adata, key, error=True)

    for key in keys:
        if not is_key_categorical(adata, key, error=False):
            adata.obs = convert_to_categorical(adata.obs, key)
        adata.uns['meta'][key] = UniqueDataFrame(index=pd.Index(adata.obs[key].cat.categories))
    return


def drop_meta_keys(adata: sc.AnnData, keys: Union[str, List[str]]) -> None:
    '''
    Drops one or more keys from `adata.uns['meta']`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    keys : str or list of str
        The key or keys to remove from `adata.uns['meta']`.
    '''
    is_adata(adata, error=True)
    meta_exists(adata, error=True)

    if not is_listlike(keys):
        keys = [keys]

    for key in keys:
        try:
            del(adata.uns['meta'][key])
        except KeyError:
            pass
    return


def add_metadata(adata: sc.AnnData, 
                 key: str, 
                 prop: str, 
                 mapper: Dict[Any, Any]) -> None:
    '''
    Adds a metadata property to a key's metadata DataFrame.

    This function adds a new column (property) to the DataFrame at
    `adata.uns['meta'][key]`, mapping values from the `mapper` dictionary
    to the categories in the DataFrame's index.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    key : str
        The key whose metadata DataFrame will be modified.
    prop : str
        The name of the new property (column) to add.
    mapper : dict
        A dictionary mapping categories (from `adata.obs[key]`) to property values.
        Must contain all categories for the given key.

    Raises
    ------
    ValueError
        If `key` is a list, or if `mapper` does not contain all categories.
    TypeError
        If `mapper` is not a dictionary.
    '''
    is_adata(adata, error=True)
    meta_exists(adata, error=True)
    key_exists(adata, key, error=True)
    is_key_categorical(adata, key, error=True)
    meta_key_exists(adata, key, error=True)
    is_meta_key_consistent(adata, key, error=True)

    if is_listlike(key):
        raise ValueError("key must be a single string, not a list or other iterable.")

    if not isinstance(mapper, dict):
        raise TypeError("mapper must be a dictionary mapping categories to values")
    if not all([cat in mapper.keys() for cat in adata.uns['meta'][key].index]):
        raise ValueError("mapper must contain all categories in adata.uns['meta'][key].index")
    
    adata.uns['meta'][key][prop] = adata.uns['meta'][key].index.map(mapper)
    
    return


def remove_metadata(adata: sc.AnnData, key: str, prop: str) -> None:
    '''
    Removes a metadata property from a key's metadata DataFrame.

    This function drops a column (property) from `adata.uns['meta'][key]`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    key : str
        The key whose metadata will be modified.
    prop : str
        The property (column) to remove.
    '''
    is_adata(adata, error=True)
    meta_exists(adata, error=True)
    # key_exists(adata, key, error=True) # don't care
    # is_key_categorical(adata, key, error=True) # don't care
    meta_key_exists(adata, key, error=True)
    meta_key_prop_exists(adata, key, prop, error=True)

    adata.uns['meta'][key].drop(columns=[prop], inplace=True)
    
    return


def update_meta_cats(adata: sc.AnnData, key: str) -> None:
    '''
    Synchronizes the metadata index with `adata.obs` categories for a key.

    This function reindexes `adata.uns['meta'][key]` to match the categories
    in `adata.obs[key].cat.categories`. This adds new categories with NaN
    values and removes categories that are no longer in `.obs`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    key : str
        The key to update.
    '''
    is_adata(adata, error=True)
    meta_exists(adata, error=True)
    key_exists(adata, key, error=True)
    is_key_categorical(adata, key, error=True)
    meta_key_exists(adata, key, error=True)

    target_index = adata.obs[key].cat.categories
    
    # A single reindex handles additions, removals, and reordering.
    adata.uns['meta'][key] = adata.uns['meta'][key].reindex(target_index)

    return


# Simple Convenience Functions

def get_cat_keys(adata: sc.AnnData) -> pd.Index:
    '''
    Gets all keys in `adata.obs` that are of a categorical dtype.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to inspect.

    Returns
    -------
    pd.Index
        An index of column names that are categorical.
    '''
    is_adata(adata)
    return adata.obs.select_dtypes(include=pd.CategoricalDtype()).columns


def get_key_dict(adata: sc.AnnData) -> Dict[str, pd.CategoricalIndex]:
    '''
    Gets a dictionary mapping categorical keys to their categories.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to inspect.

    Returns
    -------
    dict
        A dictionary where keys are categorical column names from `adata.obs`
        and values are the `pd.CategoricalIndex` of their categories.
    '''
    is_adata(adata)
    return {key: adata.obs[key].cat.categories for key in get_cat_keys(adata)}


def get_cat_dict(adata: sc.AnnData, key: str, prop: str) -> Dict[Any, Any]:
    '''
    Gets a dictionary mapping categories to property values for a given key.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object.
    key : str
        The metadata key.
    prop : str
        The metadata property.

    Returns
    -------
    dict
        A dictionary mapping categories to values for the specified property.
    '''
    is_adata(adata)
    meta_exists(adata, error=True)
    # key_exists(adata, key, error=True) # don't care
    # is_key_categorical(adata, key, error=True) # don't care
    meta_key_exists(adata, key, error=True)
    meta_key_prop_exists(adata, key, prop, error=True)

    return adata.uns['meta'][key][prop].to_dict()


def get_combined_metadata(adata: sc.AnnData, prop: str) -> pd.DataFrame:
    '''
    Combines a specific property from all metadata keys into a single DataFrame.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object.
    prop : str
        The property to extract from each key's metadata.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each column represents a key and contains the values
        for the specified property.

    Raises
    ------
    ValueError
        If the property is not found in any of the metadata keys.
    '''
    is_adata(adata)
    meta_exists(adata, error=True)

    combined = []

    for key in adata.uns['meta']:
        if prop in adata.uns['meta'][key].columns:
            combined.append(adata.uns['meta'][key][prop].rename_axis(key, axis=0).reset_index())
    if len(combined) == 0:
        raise ValueError(f"No metadata found with property '{prop}' in any keys.")
    else:
        return pd.concat(combined, axis=1).fillna('')


def add_prop_to_obs(adata: sc.AnnData, 
                    key: str, 
                    prop: str, 
                    overwrite: bool = False) -> None:
    """
    Maps a metadata property from `uns` to a new column in `obs`.

    This function takes a property from `adata.uns['meta'][key]` and creates
    a new column in `adata.obs` by mapping the values based on the
    categorical `adata.obs[key]` column.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    key : str
        The categorical key in `adata.obs` used for mapping.
    prop : str
        The property from `adata.uns['meta'][key]` to add to `adata.obs`.
    overwrite : bool, optional
        If True, allows overwriting an existing column in `adata.obs`.
        Default is False.

    Raises
    ------
    ValueError
        If a column with the same name as `prop` already exists in `adata.obs`
        and `overwrite` is False.
    """
    is_adata(adata)
    key_exists(adata, key, error=True)
    meta_exists(adata, error=True)
    meta_key_exists(adata, key, error=True)
    meta_key_prop_exists(adata, key, prop, error=True)

    if prop in adata.obs.columns:
        if not overwrite:
            raise ValueError(
                f"Column '{prop}' already exists in adata.obs. "
                "Set `overwrite=True` to replace it."
            )
    adata.obs[prop] = adata.obs[key].map(adata.uns['meta'][key][prop].to_dict())


def transfer_meta(mdata: md.MuData, 
                  mods: Tuple[str, str], 
                  key: str, 
                  prop: str) -> None:
    """
    Transfers metadata for a key between modalities in a MuData object.

    This is typically used after transferring an `.obs` column from a source
    modality to a target modality (e.g., via `muon.pp.transfer_obs`).
    It assumes `mdata[target_mod].obs` contains a column named
    `f'{source_mod}.{key}'`.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object.
    mods : tuple of (str, str)
        A tuple containing the (source_mod, target_mod) names.
    key : str
        The original key in the source modality.
    prop : str
        The metadata property to transfer.

    Raises
    ------
    TypeError
        If `mdata` is not a MuData object.
    ValueError
        If `mods` is not a valid tuple of two modality names present in `mdata`.
    """

    if not isinstance(mdata, md.MuData):
        raise TypeError("data must be an instance of md.MuData")

    # Check that the mods param is valid
    mods_valid = isinstance(mods, tuple) and all([isinstance(mod, str) for mod in mods]) and len(mods) == 2
    mods_in_mdata = all([mod in mdata.mod.keys() for mod in mods])
    if not mods_valid or not mods_in_mdata:
        raise ValueError("Param `mods` must be a tuple of (source_mod, target_mod) present in mdata.")

    mapper = get_cat_dict(mdata[mods[0]], key, prop)

    add_metadata(mdata[mods[1]], '.'.join([mods[0], key]), prop, mapper=mapper)


# Complex Convenience Functions

def add_colors(adata: sc.AnnData, 
               key: str, 
               by_size: bool = True, 
               overwrite: bool = False) -> None:
    '''
    Generates and adds a color mapping for a categorical key.

    The colors are generated using `scutils.get_color_mapper` and added as
    the 'color' property in `adata.uns['meta'][key]`.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    key : str
        The categorical key in `adata.obs` for which to generate colors.
    by_size : bool, optional
        If True, categories are ordered by their frequency in `adata.obs[key]`
        before assigning colors, giving more frequent categories earlier colors
        in the palette. If False, categories are ordered alphabetically.
        Default is True.
    overwrite : bool, optional
        If True, allows overwriting an existing color mapping. Default is False.

    Raises
    ------
    ValueError
        If a color mapping already exists for the key and `overwrite` is False.
    '''
    # is_adata(adata) # checked in add_metadata()
    # meta_exists(adata, error=True) # checked in add_metadata()
    key_exists(adata, key, error=True)
    is_key_categorical(adata, key, error=True)
    meta_key_exists(adata, key, error=True) # Need to check here in case overwrite

    if meta_key_prop_exists(adata, key, 'color', error=False):
        if not overwrite:
            raise ValueError(
                f"Color mapping for key '{key}' already exists. "
                "Set `overwrite=True` to replace it."
            )

    if by_size:
        cats = adata.obs[key].value_counts().sort_values(ascending=False).index
    else:
        cats = adata.obs[key].cat.categories

    mapper = get_color_mapper(cats)
    
    add_metadata(adata, key, 'color', mapper)
    
    return


def add_meta_df(adata: sc.AnnData, 
                key: str, 
                df: pd.DataFrame, 
                how: str = 'left') -> None:
    '''
    Joins a DataFrame to a key's metadata DataFrame.

    This allows adding multiple properties to a key's metadata at once.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to modify.
    key : str
        The key whose metadata DataFrame will be joined.
    df : pd.DataFrame
        The DataFrame to join. Its index should correspond to the categories
        of `adata.obs[key]`.
    how : {'left', 'outer'}, optional
        The type of join to perform. 'left' preserves the existing categories,
        while 'outer' preserves all categories from both DataFrames.
        Default is 'left'.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    ValueError
        If `df` contains protected column names (e.g., 'color'), if `how` is
        not 'left' or 'outer', or if `df` has columns that already exist in
        the target metadata DataFrame.
    '''
    is_adata(adata)
    meta_exists(adata, error=True)
    key_exists(adata, key, error=True)
    is_key_categorical(adata, key, error=True)
    meta_key_exists(adata, key, error=True)
    is_meta_key_consistent(adata, key, error=True)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if any([i in PROTECTED_KEYS for i in df.columns]):
        raise ValueError(
            f"Cannot add metadata with protected columns: {PROTECTED_KEYS}. "
            "Please renames the columns in the DataFrame before adding."
        )
    
    if how not in ['left', 'outer']:
        raise ValueError("Param `how` must be either 'left' or 'outer'")

    # Proactively check for overlapping columns
    existing_props = adata.uns['meta'][key].columns
    new_props = df.columns
    overlap = existing_props.intersection(new_props)

    if not overlap.empty:
        raise ValueError(
            f"Columns in the DataFrame to be added overlap with existing columns in '{key}'. "
            "Please rename or remove the columns in the provided DataFrame or in "
            f"self.meta['{key}'] before adding. Overlapping columns: {list(overlap)}"
        )

    adata.uns['meta'][key] = adata.uns['meta'][key].join(df, how=how, validate='1:1')
    
    return


def get_key_props(mdata: md.MuData, 
                  mods: Tuple[str, str], 
                  keys: List[str], 
                  props: Optional[Union[str, Number, List, Dict]] = None) -> Dict[str, pd.Index]:
    """
    Prepares a dictionary of properties to transfer between modalities.

    This function is a helper for setting up metadata transfer. It identifies
    which properties exist for a given set of keys in a source modality and
    prepares the target modality to receive them.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object.
    mods : tuple of (str, str)
        A tuple containing the (source_mod, target_mod) names.
    keys : list of str
        The list of keys from the source modality to be transferred.
    props : str, list, dict, optional
        The properties to transfer.
        - If None, all properties for each key are selected.
        - If a list, only properties in the list are selected for each key.
        - If a dict, specifies properties per key.
        Default is None.

    Returns
    -------
    dict
        A dictionary where keys are the original keys and values are the
        properties to be transferred for that key.

    Raises
    ------
    TypeError
        If `mdata` is not a MuData object or `props` is of an invalid type.
    ValueError
        If `mods` is invalid, or if `props` are not found.
    """

    if not isinstance(mdata, md.MuData):
        raise TypeError("data must be an instance of md.MuData")

    # Check that the mods param is valid
    mods_valid = isinstance(mods, tuple) and all([isinstance(mod, str) for mod in mods]) and len(mods) == 2
    mods_in_mdata = all([mod in mdata.mod.keys() for mod in mods])
    if not mods_valid or not mods_in_mdata:
        raise ValueError("Param `mods` must be a tuple of (source_mod, target_mod) present in mdata.")
    
    
    if not meta_exists(mdata[mods[1]], error=False):
        add_meta(mdata[mods[1]])

    keys_to_add = []
    for key in keys:
        if not meta_key_exists(mdata[mods[1]], '.'.join([mods[0], key]), error=False):
            keys_to_add.append('.'.join([mods[0], key]))
    # print(keys_to_add)
    # print(mdata[mods[1]].uns['meta'].keys())
    add_meta_keys(mdata[mods[1]], keys=keys_to_add)

    if props is None:
        key_props = {k: mdata[mods[0]].uns['meta'][k].columns for k in keys}
    elif isinstance(props, dict):
        if not all([k in keys for k in props.keys()]):
            raise ValueError("If `props` is a dict, all keys must be in `columns`.")
        for k in key_props:
            for prop in key_props[k]:
                meta_key_prop_exists(mdata[mods[0]], k, prop, error=True)
        key_props = props
    else: # prop or list-like of props, which can partially match the keys
        if isinstance(props, (str, Number)):
            props = [props]
        elif is_listlike(props):
            pass
        else:
            raise TypeError(
                "Param `props` must be None, a string, a number, a list of strings/numbers, or a dict."
                )
        key_props = {}
        for k in keys:
            key_props[k] = [prop for prop in props if prop in mdata[mods[0]].uns['meta'][k].columns]
        if all(len(key_props[k]) == 0 for k in key_props):
            raise ValueError(
                "None of the provided properties in `props` were found in any of the keys' metadata."
                "Please check the properties and try again."
            )

    return key_props

