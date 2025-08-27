# Built-ins
import numbers
import warnings
from typing import Callable, Optional, Union, Dict, Any

# Standard libs
import pandas as pd
import matplotlib as mpl

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.genutils import is_listlike
from survey.singlecell.scutils import convert_to_categorical

# Hierarchy of modalities, which are checked in order for containing various data
MOD_ORDER = ['rna', 'adt', 'xyz', 'csp', 'hto', 'sbc', 'atc']
DEFAULT_MOD = MOD_ORDER[0]

    
def check_type(stype: str) -> Callable[[pd.Series], bool]:
    """
    Returns a function that checks if a pandas Series is of a specific type.

    This is a factory function that provides a convenient way to get a type-checking
    function for pandas Series.

    Parameters
    ----------
    stype : {'str', 'num', 'cat', 'bool'}
        The type to check for.

    Returns
    -------
    function
        A function that takes a pandas Series and returns True if it matches
        the specified type, False otherwise.

    Raises
    ------
    ValueError
        If `stype` is not one of the recognized types.
    """

    def is_string(s):
        return pd.api.types.is_string_dtype(s)

    def is_numeric(s):
        return pd.api.types.is_numeric_dtype(s)
    
    def is_categorical(s):
        return isinstance(s.dtype, pd.CategoricalDtype)
    
    def is_bool(s):
        return pd.api.types.is_bool_dtype(s)

    if stype == 'str':
        return is_string
    elif stype == 'num':
        return is_numeric
    elif stype == 'cat':
        return is_categorical
    elif stype == 'bool':
        return is_bool
    else:
        raise ValueError(f"Unknown series type: {stype}")
    

def determine_color_type(adata: sc.AnnData, color: Any) -> Optional[str]:
    """
    Determines the type of a color argument for plotting.

    The function checks if the `color` argument corresponds to a column in
    `adata.obs`, a gene in `adata.var_names`, or a valid Matplotlib color.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object.
    color : any
        The color argument to be evaluated.

    Returns
    -------
    str or None
        The determined color type: 'cat' (categorical), 'num' (numerical),
        'named' (e.g., 'red'), or 'rgba' (e.g., (1,0,0,1)). Returns None if
        the type cannot be determined.
    """

    color_type = None

    if color in adata.obs.columns:
        s = adata.obs[color]

        # First check if it's not categorical and it's a string or bool
        # In that order, since s_is_bool will error if it's categorical
        if not check_type('cat')(s) and (check_type('str')(s) or check_type('bool')(s)):
            convert_to_categorical(adata.obs, color)
        
        if check_type('cat')(s):
            color_type = 'cat'
        elif check_type('num')(s):
            color_type = 'num'
        else:
            raise ValueError(f"Column '{color}' is not str, cat, or num.")

    elif color in adata.var_names:
        color_type = 'num'
    elif mpl.colors.is_color_like(color):
        is_rgba = (is_listlike(color) and 
                    all([isinstance(i, numbers.Number) for i in color]) and 
                    len(color) in [3, 4])
        if is_rgba:
            color_type = 'rgba'
        else:
            color_type = 'named'

    return color_type


def determine_data(data: Union[sc.AnnData, md.MuData],
                   mod: Optional[str] = None,
                   color: Optional[Any] = None,
                   basis: Optional[str] = None,
                   error: bool = True) -> Dict[str, Any]:
    """
    Determines the data types and locations for plotting parameters.

    This function inspects an AnnData or MuData object to find the appropriate
    modality and data type for `color` and `basis` arguments, which is crucial
    for downstream plotting functions.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The input data object.
    mod : str, optional
        A specific modality to search within if `data` is a MuData object.
    color : any, optional
        The color argument to locate.
    basis : str, optional
        The basis (embedding) to locate (e.g., 'umap').
    error : bool, optional
        If True, raises a ValueError if a requested parameter cannot be found.

    Returns
    -------
    dict
        A dictionary with keys 'data', 'color', and 'basis', containing
        information about their types and locations.
        - 'data': 'adata' or 'mdata'.
        - 'color': A dict {'mod': str, 'type': str} or None.
        - 'basis': The modality name (str) or None.

    Raises
    ------
    ValueError
        If `data` is not a valid type, or if a requested `color` or `basis`
        cannot be found and `error` is True.
    """
    results = {'data': None, 'color': None, 'basis': None}
    if isinstance(data, md.MuData):
        results['data'] = 'mdata'
    elif isinstance(data, sc.AnnData):
        results['data'] = 'adata'
    else:
        raise ValueError('Param `data` must be an AnnData or MuData object.')

    if all(param is None for param in (mod, color, basis)):
        return results

    if mod is not None:
        if results['data'] == 'mdata' and mod not in data.mod:
            raise ValueError(f"Mod '{mod}' not found in mdata.")
        elif results['data'] == 'adata':
            warnings.warn('Param `mod` is ignored when `data` is an AnnData object.')
        adata_ref = data[mod] if results['data'] == 'mdata' else data

        if color is not None:
            color_type = determine_color_type(adata_ref, color)
            if color_type is not None:
                results['color'] = {'mod': mod if results['data'] == 'mdata' else None, 'type': color_type}
            elif error:
                raise ValueError(f"Color '{color}' not found in specified modality/data.")

        if basis is not None:
            if f'X_{basis}' in adata_ref.obsm.keys():
                results['basis'] = mod if results['data'] == 'mdata' else basis
            elif error:
                raise ValueError(f"Basis '{basis}' not found in specified modality/data.")
            
    else:  # mod is None
        if results['data'] == 'mdata':
            # Joint search for color and basis
            if color is not None and basis is not None:
                found_mods = [m for m in MOD_ORDER if m in data.mod and f'X_{basis}' in data[m].obsm and determine_color_type(data[m], color) is not None]
                if found_mods:
                    best_mod = found_mods[0]
                    results['color'] = {'mod': best_mod, 'type': determine_color_type(data[best_mod], color)}
                    results['basis'] = best_mod
                    if len(found_mods) > 1:
                        warnings.warn(f"Found color '{color}' and basis '{basis}' in multiple modalities: {found_mods}. Using '{best_mod}'.")
            
            # Separate searches if not found jointly or not requested jointly
            if color is not None and results['color'] is None:
                found_mods_color = [m for m in MOD_ORDER if m in data.mod and determine_color_type(data[m], color) is not None]
                if found_mods_color:
                    best_mod = found_mods_color[0]
                    results['color'] = {'mod': best_mod, 'type': determine_color_type(data[best_mod], color)}
                    if len(found_mods_color) > 1:
                        warnings.warn(f"Found color '{color}' in multiple modalities: {found_mods_color}. Using '{best_mod}'.")

            if basis is not None and results['basis'] is None:
                found_mods_basis = [m for m in MOD_ORDER if m in data.mod and f'X_{basis}' in data[m].obsm]
                if found_mods_basis:
                    results['basis'] = found_mods_basis[0]
                    if len(found_mods_basis) > 1:
                        warnings.warn(f"Found basis '{basis}' in multiple modalities: {found_mods_basis}. Using '{found_mods_basis[0]}'.")

        else:  # adata
            if color is not None:
                color_type = determine_color_type(data, color)
                if color_type is not None:
                    results['color'] = {'mod': None, 'type': color_type}
            if basis is not None and f'X_{basis}' in data.obsm.keys():
                results['basis'] = basis

        # Ccheck for missing items
        if error:
            if color is not None and results['color'] is None:
                raise ValueError(f"Color '{color}' not found in data.")
            if basis is not None and results['basis'] is None:
                raise ValueError(f"Basis '{basis}' not found in data.")
            
    # Final check to make sure color is returned correctly
    if results['color'] is None:
        results['color'] = {'mod': None, 'type': None}

    return results

