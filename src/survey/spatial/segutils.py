# Built-ins
from functools import reduce
from typing import Optional, Union, List

# Standard libs
import numpy as np
import pandas as pd
import mudata as md
from scipy.ndimage import binary_dilation, binary_erosion

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


def add_borders(mdata: md.MuData,
                chipnum: Optional[Union[int, List[int]]] = None,
                from_group: Optional[Union[str, List[str]]] = None,
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


def get_segmask(chip, group, seg=None):
    """
    Get a binary array representing the segmentation mask of a chipmap.

    Parameters
    ----------
    chip : Chip
        The Chip object containing the chipmap and layout information.
    group : str
        The name of the segmentation group in chip.seg.columns to use for the mask.
    seg : str, optional
        The specific segment within the group to use for the mask. If None, all 
        segments in the group are included.
    
    Returns
    -------
    ndarray: 2D binary array representing the segmentation mask, where True indicates
    wells belonging to the specified segment(s) and False indicates wells that do not belong.
    """
    chipmap = chip.get_welldata()
    if seg is not None:
        group_bool = chip.seg[group] == seg
    else:
        group_bool = chip.seg[group].notna()
    group_ids = chip.seg[group][group_bool].index
    wells = chipmap[['arr-row', 'arr-col']].loc[group_ids].values.T

    mask = np.zeros(chip.array.arr_shape, dtype=bool)
    mask[wells[0], wells[1]] = True

    return mask, group_ids

def add_borders(mdata, chipnums, from_group, segments, name, modify=None, add=True, return_bordermasks=False, overwrite=False):
    """
    Add a segmentation group to chip.seg which marks wells at the interface
    between two (or more) segmentation groups. The borders are calculated by dilating
    the binary arrays of the segments (using default square connectivity of 1) in 
    'segments', and then taking the intersection of the dilated arrays. Optionally, the 
    borders can be further modified by dilating, eroding, or including additional 
    segments in the border calculation. 

    Parameters
    ----------
    mdata : object
        The mdata object containing a SurveyMap.
    chipnums : list
        The chipnums found in ChipSet.chips to process.
    from_group : str
        The name of the segment group Chip.seg.columns.
    segments : list
        A list of segments to include in the border calculation.
    modify : dict, optional
        A dictionary of modifications to apply to the borders. Keys can be 'dilate', 'erode', or 'include'.
        Values for 'dilate' and 'erode' should be the number of iterations to apply to the binary array. 
        Values for 'include' should be a list of segments within the 'from_group' to include in the border.
        Modifications are applied in the order they are provided.
    add : bool, optional
        Whether to add the borders as a new segment in the chip map. Default is True.
    name : str, optional
        The name of the new segment to add to the chip map. If not provided, a default "border" name is 
        generated from 'segments'.
    return_segmask : bool, optional
        Whether to return the binary arrays representing the borders. Default is False. If a single chip is 
        processed, the array is returned directly. If multiple chipnums are processed, a dictionary mapping
        chip identifiers to 2D binary arrays is returned.
    overwrite : bool, optional
        Whether to overwrite an existing segment inChip.seg.columns, if it exists. Default is False.

    Returns
    -------
    If add, modifies mdata['xyz'].uns['survey'] in-place. If return_segmask, returns 2D binary array(s) 
    representing the borders.
    """

    if is_listlike(chipnums):
        pass
    else:
        chipnums = [chipnums]
    
    if not all([chipnum in mdata['xyz'].uns['survey'].chips for chipnum in chipnums]):
        raise ValueError("All chipnums must be keys in mdata['xyz'].uns['survey'].chips.")

    segs_present = []
    for chipnum in chipnums:
        chip = mdata['xyz'].uns['survey'].chips[chipnum]
        for seg in segments:
            segs_present.append(seg in chip.seg[from_group].cat.categories)
    if not all(segs_present):
        raise ValueError("All segments must be present in the chip segmentation data.")
    
    def modify_bordermask(keyword, value, bordermask):
        if keyword == 'dilate':
            if not isinstance(value, int) or value < 0:
                raise ValueError("Value for 'dilate' must be a non-negative integer.")
            return binary_dilation(bordermask, iterations=value, mask=mask)
        
        if keyword == 'erode':
            if not isinstance(value, int) or value < 0:
                raise ValueError("Value for 'erode' must be a non-negative integer.")
            return binary_erosion(bordermask, iterations=value, mask=mask)
        
        elif keyword == 'include':
            if not is_listlike(value):
                value = [value]

            include_bordermasks =list()
            for include_seg in value:
                include_bordermask, include_ids = get_segmask(chip, from_group, seg=include_seg)
                include_bordermasks.append(include_bordermask)
                
            bordermask = np.any([bordermask] + include_bordermasks, axis=0)
            return bordermask
        else:
            raise ValueError("Keyword not recognized. Must be one of 'dilate', 'erode', or 'include'.")
        
    if return_bordermasks:
        bordermasks = dict()
    
    for chipnum in chipnums:

        chip = mdata['xyz'].uns['survey'].chips[chipnum]
        chipmap = chip.get_welldata()
        chipmap_arr = chipmap.reset_index(drop=False).set_index(['arr-col', 'arr-row'])

        segmasks = []

        mask, group_ids = get_segmask(chip, from_group, seg=None)

        # Find the borders by doing a single binary dilation and then keeping the overlap
        for seg in segments:
            segmask, seg_ids = get_segmask(chip, from_group, seg=seg)
            segmask = binary_dilation(segmask, iterations=1, mask=mask)

            segmasks.append(segmask)

        bordermask = np.all(segmasks, axis=0)

        # Perform any requested modifications, *in order*
        if modify is not None:
            for keyword, value in modify.items():
                bordermask = modify_bordermask(keyword, value, bordermask)

        if add:
            if name is None:
                name = '_'.join(segments) + '_border'
            else:
                assert isinstance(name, str)
            
            if not overwrite:
                if name in mdata['xyz'].uns['survey'].chips[chipnum].seg.columns:
                    raise ValueError("Segment name already exists in chip. Set overwrite=True to overwrite.")

            mdata['xyz'].uns['survey'].chips[chipnum].seg[name] = np.nan
            mdata['xyz'].uns['survey'].chips[chipnum].seg[name] = mdata['xyz'].uns['survey'].chips[chipnum].seg[name].astype('category').cat.set_categories(['border', 'nonborder'])
            nonborder_ids = group_ids
            border_ids = chipmap_arr.loc[list(map(tuple, np.argwhere(bordermask)[:, ::-1])), 'id'].values
            
            mdata['xyz'].uns['survey'].chips[chipnum].seg.loc[nonborder_ids, name] = 'nonborder'
            mdata['xyz'].uns['survey'].chips[chipnum].seg.loc[border_ids, name] = 'border'
        
        
        if return_bordermasks:
            bordermasks[chipnum] = bordermask
    
    if return_bordermasks:
        if len(chipnums) == 1:
            return bordermasks[chipnums[0]]
        else:
            return bordermasks