# Built-ins
import warnings
from typing import Optional, Union, List, Tuple, Dict

# Standard libs
import pandas as pd

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.genutils import is_listlike
from survey.singlecell.meta import key_exists, get_key_props, transfer_meta


def get_obs_df(data: Union[sc.AnnData, md.MuData],
               obs_keys: Optional[List[str]] = None,
               obsm_keys: Optional[List[str]] = None,
               features: Optional[List[str]] = None,
               layer: Optional[str] = None,
               mod: Optional[str] = None,
               empty: bool = True) -> pd.DataFrame:
    """
    Create a DataFrame from `.obs`, `.obsm`, and feature expression data.

    This function provides a flexible way to extract various pieces of cell-level
    data from an AnnData or MuData object and combine them into a single
    pandas DataFrame.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The input single-cell data object.
    obs_keys : list of str, optional
        A list of column names to select from `.obs`. If None and `empty` is
        False, all columns are selected.
    obsm_keys : list of str, optional
        A list of keys to select from `.obsm`. Each key corresponds to a
        multidimensional array that will be flattened and included in the
        output DataFrame.
    features : list of str, optional
        A list of gene/feature names to include. Their expression values will
        be extracted.
    layer : str, optional
        The data layer to use for extracting feature expression values.
    mod : str, optional
        If `data` is a MuData object, this specifies which modality to use.
    empty : bool, optional
        If True, returns an empty DataFrame for `.obs` and `.obsm` if their
        respective keys are not provided.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the concatenated data.

    Raises
    ------
    ValueError
        If `data` is a MuData object and `mod` is not specified.
    """

    if isinstance(data, sc.AnnData):
        if mod is not None:
            warnings.warn("AnnData object provided, ignoring `mod`.")
        obs = data.obs
        obsm = data.obsm
    elif isinstance(data, md.MuData):
        if mod is None:
            raise ValueError("For MuData objects, `mod` must be specified.")
        data = data[mod]
        obs = data.obs
        obsm = data.obsm
    else:
        raise ValueError("Invalid data type")

    # If no keys are provided, use all columns
    if obs_keys is None:
        if empty:
            obs_keys = []
        else:
            obs_keys = obs.columns.tolist()
    elif not is_listlike(obs_keys):
        obs_keys = [obs_keys]

    if obsm_keys is None:
        if empty:
            obsm_keys = []
        else:
            obsm_keys = list(obsm.keys())
    elif not is_listlike(obsm_keys):
        obsm_keys = [obsm_keys]

    # Select the desired columns
    obs = obs[obs_keys]

    # Create a DataFrame for each numpy array in obsm
    obsm_dfs = []
    for key in obsm_keys:
        array = obsm[key]
        df = pd.DataFrame(array, index=obs.index, columns=[f"{key}_{i}" for i in range(array.shape[1])])
        obsm_dfs.append(df)
    
    # Create a DataFrame for features
    if features is not None:
        features_df = data[:, features].to_df(layer=layer)
    else:
        features_df = pd.DataFrame()
    
    # Concatenate the obs and obsm data
    df = pd.concat([obs] + obsm_dfs + [features_df], axis=1)

    return df


def add_subset_obsm(adata: sc.AnnData,
                    sub_adata: sc.AnnData,
                    subset_tag: str,
                    obsm_key: str = 'umap') -> None:
    """
    Adds embedding coordinates from a subset of data to a full AnnData object.

    This function is useful when an embedding (e.g., UMAP) is calculated on a
    subset of cells. It transfers these coordinates to the main `AnnData`
    object, filling with `NaN` for cells not in the subset.

    Parameters
    ----------
    adata : sc.AnnData
        The full AnnData object to which the subset coordinates will be added.
    sub_adata : sc.AnnData
        The AnnData object containing the subset of data with the embedding.
    subset_tag : str
        A tag to append to the new `.obsm` key, which will be formatted as
        `'X_<obsm_key>_<subset_tag>'`.
    obsm_key : str, optional
        The key for the embedding in `.obsm` (e.g., 'umap', 'pca'). The
        function will look for `'X_<obsm_key>'` in `sub_adata.obsm`.

    Returns
    -------
    None
        The function modifies `adata` in-place.

    Raises
    ------
    TypeError
        If `adata` or `sub_adata` are not AnnData objects.
    KeyError
        If `'X_<obsm_key>'` is not found in `sub_adata.obsm`.
    """

    if not all(isinstance(x, sc.AnnData) for x in [adata, sub_adata]):
        raise TypeError("Both adata and sub_adata must be instances of AnnData.")

    source_obsm_key = f'X_{obsm_key}'
    if source_obsm_key not in sub_adata.obsm:
        raise KeyError(f"Source key '{source_obsm_key}' not found in sub_adata.obsm.")

    new_coords = pd.DataFrame(sub_adata.obsm[source_obsm_key], index=sub_adata.obs_names)
    
    # Reindex with the full adata index, this will create NaNs for missing cells
    joined_df = new_coords.reindex(adata.obs_names)

    new_obsm_key = f'X_{obsm_key}_{subset_tag}'
    adata.obsm[new_obsm_key] = joined_df.values

    return


def transfer_obs(mdata: md.MuData,
                 columns: Union[str, List[str]],
                 mods: Tuple[str, str],
                 overwrite: bool = False,
                 meta: bool = False,
                 props: Optional[Union[str, List[str], Dict[str, List[str]]]] = None) -> None:
    """
    Transfers observation data and optionally metadata between modalities.

    This function copies one or more columns from the `.obs` of a source
    modality to a target modality within a MuData object. It can also transfer
    associated metadata from `survey.singlecell.meta`.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object.
    columns : str or list of str
        The column name(s) in the source `.obs` to transfer.
    mods : tuple of (str, str)
        A tuple of (source_modality, target_modality).
    overwrite : bool, optional
        If True, allows overwriting existing columns in the target modality.
    meta : bool, optional
        If True, also transfers associated metadata for the columns.
    props : str, list, or dict, optional
        Specifies which metadata properties to transfer if `meta` is True.
        See `get_key_props` for details.

    Raises
    ------
    TypeError
        If `mdata` is not a MuData object.
    ValueError
        If `mods` is invalid, or if a column to be transferred already exists
        in the target and `overwrite` is False.
    """
    # Check mdata is a mudata object
    if not isinstance(mdata, md.MuData):
        raise TypeError("data must be an instance of md.MuData")

    # Check that the mods param is valid
    mods_valid = isinstance(mods, tuple) and all([isinstance(mod, str) for mod in mods]) and len(mods) == 2
    mods_in_mdata = all([mod in mdata.mod.keys() for mod in mods])
    if not mods_valid or not mods_in_mdata:
        raise ValueError("Param `mods` must be a tuple of (source_mod, target_mod) present in mdata.")
    
    # Check that the columns are present and not already present in the target mod, unless overwrite is True
    if not is_listlike(columns):
        columns = [columns]

    for column in columns:
        key_exists(mdata[mods[0]], column, error=True)
    
    if not overwrite:
        for column in columns:
            col_name_add = '.'.join([mods[0], column])
            if key_exists(mdata[mods[1]], col_name_add, error=False):
                raise ValueError(
                    f"Column '{col_name_add}' already exists in target modality '{mods[1]}'. "
                    "Use overwrite=True to overwrite."
                )

    # Perform the transfer
    for column in columns:
        new_col_name = '.'.join([mods[0], column])
        
        # Only transfer values for barcodes that exist in both modalities
        source_obs = mdata[mods[0]].obs
        target_obs_names = mdata[mods[1]].obs_names
        
        common_barcodes = source_obs.index.intersection(target_obs_names)
        
        # Create a new series with data for common barcodes, then reindex to match target obs
        new_col_data = pd.Series(source_obs.loc[common_barcodes, column], index=common_barcodes)
        mdata[mods[1]].obs[new_col_name] = new_col_data.reindex(target_obs_names)

    if meta:

        key_props = get_key_props(mdata, mods, keys=columns, props=props)
                
        for k in key_props:
            for prop in key_props[k]:
                transfer_meta(mdata, mods, k, prop)

