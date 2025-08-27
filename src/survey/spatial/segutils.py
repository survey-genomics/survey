# Built-ins
from functools import reduce
from typing import Optional, Union, List

# Standard libs
import pandas as pd
import mudata as md

# Survey libs
from survey.spatial.core import validate_spatial_mdata, validate_chipnums
from survey.genutils import is_listlike, get_mask

def get_all_seg_groups(mdata: md.MuData,
                       chipnum: Optional[Union[int, List[int]]] = None) -> pd.Index:
    """
    Gets all unique segmentation group names across one or more chips.

    This function inspects the `.seg` DataFrame of the specified chip(s)
    and returns a unified index of all segmentation column names.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing the spatial experiment data.
    chipnum : int or list of int, optional
        The chip number(s) to inspect. If None, all chips in the dataset
        are inspected.

    Returns
    -------
    pd.Index
        A pandas Index containing all unique segmentation group names found.
    """
    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    if chipnum is None:
        chipnums = list(chipset.chips.keys())
    else:
        chipnums = validate_chipnums(chipset, chipnum)

    all_seg_columns = [chipset.chips[chipnum].seg.columns for chipnum in chipnums]
    if all_seg_columns:
        all_groups = reduce(pd.Index.union, all_seg_columns)
    else:
        all_groups = pd.Index([])

    return all_groups



def get_seg_keys(mdata: md.MuData) -> List[str]:
    """
    Gets segmentation group names that are present as columns in `mdata['xyz'].obs`.

    This function identifies which segmentation groups have already been pulled
    into the main observation DataFrame of the 'xyz' modality.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object to inspect.

    Returns
    -------
    list of str
        A list of segmentation group names found as columns in `mdata['xyz'].obs`.
    """

    validate_spatial_mdata(mdata)
    groups = get_all_seg_groups(mdata)
    groups_present = mdata['xyz'].obs.columns.intersection(groups).tolist()
    
    return groups_present



def pull_seg(mdata: md.MuData,
             chipnum: Optional[Union[int, List[int]]] = None,
             groups: Optional[Union[str, List[str]]] = None,
             overwrite: bool = False) -> None:
    """
    Pulls segmentation data from `chip.seg` into `mdata['xyz'].obs`.

    This function maps segmentation annotations (e.g., tissue regions) from
    the well level to the cell level, creating new columns in the 'xyz'
    modality's `.obs` DataFrame.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object to be modified.
    chipnum : int or list of int, optional
        The chip number(s) from which to pull segmentation data. If None,
        data is pulled from all chips.
    groups : str or list of str, optional
        The specific segmentation group(s) to pull. If None, all available
        groups for the specified chips are pulled.
    overwrite : bool, optional
        If True, allows overwriting existing columns in `mdata['xyz'].obs`.

    Returns
    -------
    None
        The function modifies `mdata` in-place.

    Raises
    ------
    ValueError
        If a group to be pulled already exists in `mdata['xyz'].obs` and
        `overwrite` is False.
    """

    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    if chipnum is None:
        chipnums = list(chipset.chips.keys())
    else:
        chipnums = validate_chipnums(chipset, chipnum)

    if groups is None:
        groups = get_all_seg_groups(mdata, chipnums)
    elif not is_listlike(groups):
        groups = [groups]
    else:
        raise ValueError("Invalid groups specification.")

    groups_present = mdata['xyz'].obs.columns.intersection(groups)
    
    if not groups_present.empty:
        if not overwrite:
            raise ValueError(f"Groups {groups_present.tolist()} already exist and overwrite is False.")

    for group in groups:
        if group not in mdata['xyz'].obs.columns:
            mdata['xyz'].obs[group] = pd.NA
        for chipnum in chipnums:
            chip = chipset.chips[chipnum]
            try:
                group_mapper = chip.seg[group].to_dict()
            except KeyError:
                # Group not found in segmentation data for this chip
                continue
            mask = get_mask(mdata['xyz'].obs, {chipset.chip_key_prop: [chipnum]})
            cbcs = mdata['xyz'][mask].obs_names
            mdata['xyz'].obs.loc[cbcs, group] = mdata['xyz'][mask].obs['id'].map(group_mapper)

    for group in groups:
        mdata['xyz'].obs[group] = mdata['xyz'].obs[group].astype('category')
    
    return