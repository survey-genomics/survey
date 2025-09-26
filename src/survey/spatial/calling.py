# Built-ins
from pathlib import Path
import warnings
import tempfile
import itertools as it
from functools import reduce
from typing import Optional, Union, List, Tuple, Dict, Any

# Standard libs
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.sparse import SparseEfficiencyWarning
from scipy.spatial.distance import pdist

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.genplot import subplots
from survey.singlecell import meta
from survey.spatial.core import Chip, ChipSet
from survey.genutils import is_listlike, get_config
from survey.genplot import create_gif_from_pngs


def validate_mdata_chipset(mdata: md.MuData,
                           chipset: Optional[ChipSet] = None,
                           chip_key_prop: str = 'chip-num') -> Tuple[md.MuData, ChipSet]:
    """
    Validates consistency between a MuData object and a ChipSet object.

    This function ensures that:
    - The MuData object contains modalities for each barcode type in the ChipSet.
    - The chip numbers in the ChipSet are present in the MuData object's observations.
    - For each chip, the set of cell barcodes is identical across all relevant modalities.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object to validate.
    chipset : ChipSet, optional
        The ChipSet object to validate against. If None, it is inferred from
        `mdata['xyz'].uns['chipset']`.
    chip_key_prop : str, optional
        The key in `.obs` that identifies the chip number for each cell.

    Returns
    -------
    tuple
        A tuple containing the validated `mdata` and `chipset` objects.

    Raises
    ------
    TypeError
        If `mdata` or `chipset` are not of the correct type.
    ValueError
        If inconsistencies are found between the `mdata` and `chipset`.
    """
    if not isinstance(mdata, md.MuData):
        raise TypeError("mdata must be an instance of mudata.MuData.")
    
    if chipset is None:
        if 'xyz' not in mdata.mod:
            raise ValueError("mdata must have 'xyz' modality for spatial plotting.")
        if 'chipset' not in mdata['xyz'].uns:
            raise ValueError("mdata['xyz'] must have 'chipset' modality for spatial plotting.")
        chipset = mdata['xyz'].uns['chipset']


    if not isinstance(chipset, ChipSet):
        raise TypeError("chipset must be an instance of svp.core.ChipSet.")
    for bctype in chipset.bctypes:
        adata_bctype = mdata[bctype]
        meta.key_exists(adata_bctype, chip_key_prop)
        meta.is_key_categorical(adata_bctype, chip_key_prop)
        if bctype not in mdata.mod:
            raise ValueError(f"mdata must contain modality for barcode type '{bctype}'.")

        # mdata_chip_nums = adata_bctype.obs[chip_key_prop].cat.categories

        # if not all([num in mdata_chip_nums for num in chipset.chips]):
        #     raise ValueError(
        #         f"Not all chip numbers in chipset are present in mdata['{bctype}'].obs['{chip_key_prop}']. "
        #         f"Missing chip numbers: {list(set(chipset.chips) - set(mdata_chip_nums))}. "
        #     )
        
    # Make sure that, for every chip, the set of cell barcodes for each bctype are identical
    for chip_num, chip in chipset.chips.items():
        reference_bcs = None
        
        for bctype in chip.layout.bctypes:
            adata_bctype = mdata[bctype]
            cbc_bool = adata_bctype.obs[chip_key_prop].isin([chip_num])
            current_cell_bcs = frozenset(adata_bctype.obs_names[cbc_bool])
            
            if reference_bcs is None:
                reference_bcs = current_cell_bcs
            elif reference_bcs != current_cell_bcs:
                raise ValueError(
                    f"Cell barcodes for chip {chip_num} are not consistent across barcode types. "
                    f"Mismatch found in bctype '{bctype}'."
                )
        
    return mdata, chipset
    

def get_faulty_bcs(mdata: md.MuData,
                   chipset: Optional[ChipSet] = None,
                   chip_nums: Optional[Union[int, List[int]]] = None,
                   chip_key_prop: str = 'chip-num',
                   check_thresh: Optional[Union[int, Dict[Any, Any]]] = None) -> Tuple[Dict, Dict]:
    """
    Identifies barcodes with high counts that are not in the provided layout.

    This function is a diagnostic tool to detect potential issues with a chip
    layout file, such as missing barcodes that still show significant counts.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing the count data.
    chipset : ChipSet, optional
        The ChipSet object with layout information. If None, it's inferred from `mdata`.
    chip_nums : int or list of int, optional
        The chip number(s) to check. If None, all chips in the chipset are checked.
    chip_key_prop : str, optional
        The key in `.obs` that identifies the chip number.
    check_thresh : int or dict, optional
        The count threshold above which a non-layout barcode is flagged as "faulty".
        This can be:
        1. An integer threshold applied to all checks.
        2. A dictionary `{bctype: threshold}` for bctype-specific thresholds.
        3. A dictionary `{chip_num: threshold}` for chip-specific thresholds.
        4. A nested dictionary `{chip_num: {bctype: threshold}}` for chip- and
           bctype-specific thresholds.

    Returns
    -------
    faulty_bcs : dict
        A nested dictionary `{chip_num: {bctype: [barcodes]}}` listing the faulty barcodes.
    faulty_counts : dict
        A nested dictionary `{chip_num: {bctype: pd.Series}}` containing the counts
        for all non-layout barcodes.
    """


    mdata, chipset = validate_mdata_chipset(mdata, chipset, chip_key_prop)

    faulty_bcs = {}
    faulty_counts = {}

    if chip_nums is None:
        chip_nums = list(chipset.chips.keys())
    elif not is_listlike(chip_nums):
        chip_nums = [chip_nums]
    elif not all([num in chipset.chips for num in chip_nums]):
        raise ValueError(f"chip_nums must be a list of chip numbers from the chipset: {list(chipset.chips.keys())}.")

    # Validate check_thresh structure
    if check_thresh is None:
        check_thresh = 500
    if isinstance(check_thresh, dict):
        if not check_thresh: # empty dict
            pass
        else:
            first_key = next(iter(check_thresh))
            if isinstance(first_key, int) and first_key in chipset.chips: # chip-num keyed
                if isinstance(check_thresh[first_key], dict): # Nested dict
                    all_bctypes = chipset.bctypes
                    for cn in check_thresh:
                        if not all(bt in all_bctypes for bt in check_thresh[cn]):
                            raise ValueError("Inner keys of check_thresh must be valid bctypes.")
                elif not all(isinstance(v, int) for v in check_thresh.values()):
                    raise ValueError("Values of chip-keyed check_thresh dict must be integers.")
            elif isinstance(first_key, str) and first_key in chipset.bctypes: # bctype keyed
                if not all(isinstance(v, int) for v in check_thresh.values()):
                    raise ValueError("Values of bctype-keyed check_thresh dict must be integers.")
            else:
                raise ValueError("check_thresh dict keys must be chip numbers or bctypes.")
    elif not isinstance(check_thresh, int):
        raise TypeError("check_thresh must be an int or a dictionary.")

    for chip_num in chip_nums:
        chip = chipset.chips[chip_num]

        faulty_bcs[chip_num] = {}
        faulty_counts[chip_num] = {}

        for bctype in set(chip.layout.bctypes):
            adata_bctype = mdata[bctype]
            meta.key_exists(adata_bctype, chip_key_prop)
            
            cbc_bool = adata_bctype.obs[chip_key_prop].isin([chip_num])
            excluded_bcs = adata_bctype.var_names.difference(chip.layout.all_bcs)

            excluded_counts = adata_bctype[cbc_bool, excluded_bcs].to_df().sum(0)

            current_thresh = 500 # Default
            if isinstance(check_thresh, int):
                current_thresh = check_thresh
            elif isinstance(check_thresh, dict):
                first_key = next(iter(check_thresh))
                if isinstance(first_key, int): # chip-num keyed
                    if isinstance(check_thresh.get(chip_num), dict):
                        current_thresh = check_thresh[chip_num].get(bctype, 500)
                    else:
                        current_thresh = check_thresh.get(chip_num, 500)
                else: # bctype keyed
                    current_thresh = check_thresh.get(bctype, 500)

            above_thresh_bool = excluded_counts > current_thresh
            bcs_above_thresh = above_thresh_bool.index[above_thresh_bool].to_list()

            faulty_bcs[chip_num][bctype] = bcs_above_thresh
            faulty_counts[chip_num][bctype] = excluded_counts

        if any([len(faulty_bcs[chip_num][i]) > 0 for i in faulty_bcs[chip_num]]):
            faulty_bcs_show = {bctype: ', '.join(bcs) for bctype, bcs in faulty_bcs[chip_num].items() if len(bcs) > 0}

            warnings.warn(
                f"Spatial barcode counts for chip {chip_num} exceed threshold for barcodes: "
                f"{faulty_bcs_show}, which were not found in layout {chip.layout.id}. "
                "This suggests the chip layout is incorrect, please check your layout file.")
            
    return faulty_bcs, faulty_counts


def dilate_array(bool_array: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 num_iters: Optional[int] = None,
                 struct: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Performs iterative binary dilation on an array.

    Each new element added during dilation is assigned a value corresponding to
    the iteration number, effectively creating a distance map from the initial
    `True` elements.

    Parameters
    ----------
    bool_array : np.ndarray
        The input binary array to dilate.
    mask : np.ndarray, optional
        A boolean array that limits the dilation.
    num_iters : int, optional
        The maximum number of dilation iterations to perform.
    struct : np.ndarray, optional
        The structuring element for dilation. If None, a default is used.

    Returns
    -------
    np.ndarray
        An integer array where each element's value is the dilation iteration
        at which it was added.
    """
    result = np.zeros_like(bool_array, dtype=int)
    iteration = 1
    last_result = None
    
    while not np.array_equal(result, last_result):
        last_result = result.copy()
        old_array = bool_array.copy()
        bool_array = binary_dilation(bool_array, mask=mask, structure=struct)
        newly_added = bool_array & ~old_array
        result[newly_added] = iteration
        iteration += 1
        
        # Break if we've reached the specified number of iterations
        if num_iters is not None and iteration > num_iters:
            break
    
    return result


def get_adj(arr: np.ndarray, max_dist: int) -> np.ndarray:
    """
    Calculates an adjacency tensor based on spatial proximity.

    For each barcode ID in the input array, this function finds its neighbors
    at different distances and calculates the proportion of each neighbor type.

    Parameters
    ----------
    arr : np.ndarray
        An integer-encoded 2D array representing a spatial layout.
    max_dist : int
        The maximum distance to consider for neighbors.

    Returns
    -------
    np.ndarray
        A 3D tensor of shape `(max_dist, num_barcodes, num_barcodes)` where
        `adj[d, i, j]` is the proportion of barcode `j` among the neighbors
        of barcode `i` at distance `d+1`.
    """
    # Get all unique barcode IDs, excluding the -1 placeholder for empty wells
    unique_bcids = np.unique(arr[arr != -1])
    if unique_bcids.size == 0:
        # Handle case with no barcodes
        return np.zeros((max_dist, 0, 0), dtype=float)
    
    num_barcodes = unique_bcids.max() + 1
    adj = np.zeros((max_dist, num_barcodes, num_barcodes), dtype=float)

    for j in range(1, max_dist + 1):
        for bcid in unique_bcids:
            # Find neighbors at distance j
            neighbor_mask = dilate_array(arr == bcid, num_iters=max_dist) == j
            
            # Get the barcode IDs of the neighbors, excluding empty wells
            neighbor_bcs = arr[neighbor_mask]
            neighbor_bcs = neighbor_bcs[neighbor_bcs != -1]

            if neighbor_bcs.size > 0:
                vals, counts = np.unique(neighbor_bcs, return_counts=True)
                props = counts / counts.sum()
                for v, p in zip(vals, props):
                    adj[j - 1, bcid, v] = p
    return adj


def distribute_by_proportions(v: np.ndarray, 
                              p: np.ndarray,
                              seed: int=0) -> List[np.ndarray]:
    """
    Randomly distributes values from an array into subarrays based on proportions.

    Parameters
    ----------
    v : np.ndarray
        1D array of values to distribute.
    p : np.ndarray
        1D array of proportions, which will be normalized to sum to 1.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of np.ndarray
        A list of subarrays containing the distributed values.
    """
    
    rng = np.random.default_rng(seed)


    v = np.array(v)
    p = np.array(p)
    
    # Normalize proportions to ensure they sum to 1
    psum = p.sum()
    if psum == 0:
        raise ValueError("Proportions must not sum to zero.")
    p = p / psum

    # Calculate target sizes for each subarray
    n_total = len(v)
    sizes = np.round(p * n_total).astype(int)
    
    # Adjust for rounding errors to ensure we use all values
    diff = n_total - sizes.sum()
    if diff > 0:
        # Add extra values to largest proportions
        indices = np.argsort(p)[-diff:]
        sizes[indices] += 1
    elif diff < 0:
        # Remove values from largest proportions
        indices = np.argsort(p)[diff:]
        sizes[indices] -= 1
    
    # Shuffle the values randomly
    shuffled_v = rng.permutation(v)
    
    # Split into subarrays
    result = []
    start_idx = 0
    for size in sizes:
        end_idx = start_idx + size
        result.append(shuffled_v[start_idx:end_idx])
        start_idx = end_idx
    
    return result


def convert_names(vars: Union[str, List[str]],
                  mode: str = 'flip',
                  prefix: Optional[str] = None,
                  return_map: bool = False) -> Union[str, List[str], Dict[str, str]]:
    """
    Adds, removes, or flips prefixes on a list of strings.

    Parameters
    ----------
    vars : str or list of str
        The string(s) to be converted.
    mode : {'flip', 'strip', 'add'}, optional
        - 'flip': Adds a prefix if not present, strips it if present.
        - 'strip': Removes the prefix if present.
        - 'add': Adds the prefix if not present.
    prefix : str, optional
        The prefix to use. Required for 'add' and 'flip' modes.
    return_map : bool, optional
        If True, returns a dictionary mapping original names to converted names.

    Returns
    -------
    str, list of str, or dict
        The converted string(s) or a mapping dictionary.

    Raises
    ------
    ValueError
        If `mode` is invalid or `prefix` is required but not provided.
    """
    is_string = False
    if isinstance(vars, str):
        is_string = True
        vars = [vars]
        

    if mode == 'add' and prefix is None:
        raise ValueError("Prefix must be provided if mode is 'add'.")
    elif mode == 'flip' and prefix is None and any('.' not in var for var in vars):
        raise ValueError("Prefix must be provided if mode is 'flip' and at least one variable does not contain a '.'.")

    if mode == 'flip':
        def flip_var(var):
            if '.' in var:
                return var.split('.')[1]
            else:
                return prefix + '.' + var
        converted = [flip_var(var) for var in vars]
    elif mode == 'strip':
        def strip_var(var):
            if '.' in var:
                return var.split('.')[1]
            else:
                return var
        converted = [strip_var(var) for var in vars]
    elif mode == 'add':
        converted = [(prefix + '.' + var) if '.' not in var else var for var in vars]
    else:
        raise ValueError("Invalid mode. Choose from 'flip', 'strip', or 'add'.")
    if not return_map:
        if is_string:
            return converted[0]
        else:
            return converted
    else:
        return dict(zip(vars, converted))


class SpatialNormalizer:
    """
    A class to normalize spatial barcode counts by simulating count bleeding.

    This class models and corrects for the phenomenon where counts from a
    high-expression well "bleed" into adjacent wells, artificially inflating
    their counts. It iteratively identifies over-represented "donor" barcodes
    and redistributes a portion of their counts to neighboring "acceptor"
    barcodes.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing the count data.
    chip : Chip
        The `Chip` object describing the physical and barcode layout.
    chip_key_prop : str, optional
        The key in `.obs` that identifies the chip number.
    max_dist : int, optional
        The maximum distance (in wells) to consider for neighbor interactions.
    p_donate : float, optional
        The proportion of a barcode's median count required to be considered
        a potential "donor".
    nb_params : dict, optional
        Parameters for the neighbor-based donation logic, including thresholds
        for count ratios and iteration limits.
    """

    def __init__(self,
                 mdata: md.MuData,
                 chip: Chip,
                 chip_key_prop: str = 'chip-num',
                 bctypes: Optional[Union[str, List[str]]] = None,
                 max_dist: int = 3,
                 p_donate: float = 0.3,
                 nb_params: Optional[Dict] = None) -> None:
        
        default_nb_config = {
            'ratio_thresh': 1.5,
            'ratio_min_prop_thresh': 0.5,
            'stop_iteration_donors': 4,
            'max_iteration_donate': 100
        }

        nb_params = get_config(nb_params, default_nb_config)

        counts = {}
        adjs = {}

        if bctypes is None:
            bctypes = chip.layout.bctypes
        elif isinstance(bctypes, str):
            bctypes = [bctypes]
        elif not all([bt in chip.layout.bctypes for bt in bctypes]):
            raise ValueError(f"bctypes must be a list of barcode types from the chip layout: {chip.layout.bctypes}.")

        for spidx, bctype in enumerate(chip.layout.bctypes):
            if bctype not in bctypes:
                continue
            cbc_bool = mdata[bctype].obs[chip_key_prop].isin([chip.num])
            var_bool = chip.layout.mappers[spidx].index
            counts[spidx] = mdata[bctype][cbc_bool, var_bool].to_df()
            counts[spidx].columns = counts[spidx].columns.map(chip.layout.mappers[spidx].to_dict())
            adjs[spidx] = get_adj(chip.layout.da[:, :, spidx], max_dist=max_dist)
        
        self.mdata = mdata
        self.chip = chip
        self.chip_key_prop = chip_key_prop
        self.bctypes = bctypes
        self.max_dist = max_dist
        self.p_donate = p_donate
        self.nb_params = nb_params

        self.layout = self.chip.layout
        self.welldata = self.chip.get_welldata()
        self.counts = counts
        self.adjs = adjs

        self.all_donors = self.get_possible_donors()

    @property
    def sorted_bcids(self) -> Dict[int, pd.Series]:
        """
        Returns a dictionary of total counts per barcode ID, sorted descendingly.
        Keys are spatial indices (0, 1, ...), values are pandas Series.
        """
        return {i: self.counts[i].sum(0).sort_values(ascending=False) for i in self.counts}

    @property
    def sorted_bcs(self) -> Dict[int, pd.Series]:
        """
        Returns a dictionary of total counts per barcode string, sorted descendingly.
        Keys are spatial indices, values are pandas Series.
        """
        sorted_bcids = self.sorted_bcids
        for i in sorted_bcids:
            sorted_bcids[i].index = sorted_bcids[i].index.map({v: k for k, v in self.layout.mappers[i].to_dict().items()})
        return sorted_bcids
    
    
    def plot_bcids(self,
                   spidx: int,
                   label: str = 'bc',
                   order_by: Optional[str] = None,
                   fss: float = 5,
                   ar: float = 5,
                   **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots a bar chart of barcode counts for a given spatial index.

        Parameters
        ----------
        spidx : int
            The spatial index (e.g., 0 or 1) to plot.
        label : {'bc', 'bcid'}, optional
            Whether to label the x-axis with barcode strings ('bc') or integer IDs ('bcid').
        order_by : str, optional
            A column from the welldata (e.g., 'arr-row') to order the barcodes by.
            If None, orders by count.
        fss : float, optional
            Figure size scaler.
        ar : float, optional
            Aspect ratio of the plot.
        **kwargs
            Additional arguments passed to `matplotlib.pyplot.bar`.

        Returns
        -------
        tuple
            A tuple containing the Matplotlib Figure and Axes objects.
        """

        if label == 'bc':
            joiner = self.sorted_bcs
            spidx_col = f'sp{spidx}'
        elif label == 'bcid':
            joiner = self.sorted_bcids
            spidx_col = f'int-bc-id{spidx}'
        else:
            raise ValueError("label must be 'bc' or 'bcid'.")

        if order_by is None:
            df = joiner[spidx].rename('counts').to_frame()
        else:
            # In order to only show each barcode once in the plot, there can't be duplicates...
            if self.welldata[[order_by, spidx_col]].dropna().drop_duplicates().shape[0] != joiner[spidx].shape[0]:
                raise ValueError("Provided `order_by` does not have a one-to-one mapping with barcodes. Please use a different `order_by`.")

            df = self.welldata.set_index(spidx_col)[[order_by]].join(joiner[spidx].rename('counts')).sort_values(order_by).dropna().drop_duplicates()

        fig, ax = subplots(1, ar=ar, fss=fss)
        ax.bar(height=df['counts'], x=df.index, **kwargs)
        ax.set_xticks(range(df.shape[0]))
        ax.set_xticklabels(df.index, rotation=90, ha='center', size=10)

        return fig, ax


    def get_probs(self, spidx: int, bcid: int) -> np.ndarray:
        """
        Retrieves neighbor probabilities for a given barcode ID.

        Parameters
        ----------
        spidx : int
            The spatial index.
        bcid : int
            The barcode ID.

        Returns
        -------
        np.ndarray
            An array of probabilities of encountering each neighbor barcode.
        """
        # Current implemention: just use the layer 0 neighbors
        # Future implementation: incorporate multiple layers and probabilty decay
        layer = 0
        probs = self.adjs[spidx][layer][bcid]
        return probs
        

    def get_possible_donors(self) -> Dict[int, pd.Index]:
        """
        Identifies potential "donor" barcodes with high initial counts.

        A barcode is considered a possible donor if its total count is above a
        certain proportion of the median count for its spatial index.

        Returns
        -------
        dict
            A dictionary mapping each spatial index to a pandas Index of
            possible donor barcode IDs.
        """
        # Possible donors are those whose starting counts are at least 20% of the 
        # median count across all barcodes for each spatial index
        # It should only be called at the start of normalization and it's result 
        # set to self.all_donors

        sorted_bcids = self.sorted_bcids
        possible_donors = {}
        for spidx in sorted_bcids:
            possible_donors[spidx] = sorted_bcids[spidx].index[sorted_bcids[spidx] > self.p_donate*sorted_bcids[spidx].median()]
        return possible_donors
    

    def get_current_donors(self, spidx: int) -> Dict[int, Dict[int, float]]:
        """
        Determines which barcodes should donate counts in the current iteration.

        This method checks possible donors to see if their count is significantly
        higher than their neighbors, meeting the criteria to become an active
        donor for this iteration.

        Parameters
        ----------
        spidx : int
            The spatial index to process.

        Returns
        -------
        dict
            A dictionary where keys are donor barcode IDs and values are
            dictionaries mapping acceptor barcode IDs to their donation proportions.
        """
        sorted_bcids = self.sorted_bcids[spidx]
        lower_counts = pd.Series(True, index=sorted_bcids.index)
        donors = {}
        for i, candidate_donor in enumerate(self.all_donors[spidx]):
            lower_counts.loc[:candidate_donor] = False

            # Get the probabilities of acceptors based on the adjacency matrix
            raw_probs = self.get_probs(spidx, candidate_donor)
            probs_masked = np.where(lower_counts.sort_index().values, raw_probs, 0)

            # Normalize the probabilities, continue if no probabilities for donation
            probs_sum = probs_masked.sum()
            if probs_sum == 0:
                continue

            # The acceptor probabilities get normalized by sum, then reformat into dict
            acceptor_probs = probs_masked/probs_sum
            acceptors = dict([i for i in enumerate(acceptor_probs) if i[1] > 0])

            # Compute the ratios and combine ratios and probs into single df called stats
            ratios = (sorted_bcids.loc[candidate_donor]/sorted_bcids.loc[list(acceptors.keys())]).rename('r')
            stats = ratios.to_frame().join(pd.Series(acceptors, name='p'))

            # Determine if the candidate should be a donor based on the stored neighbor parameters
            prop_above_thresh = stats['p'][stats['r'] > self.nb_params['ratio_thresh']].sum()

            if prop_above_thresh >= self.nb_params['ratio_min_prop_thresh']:
                donors[candidate_donor] = stats['p'].to_dict()

        return donors


    def donate(self, spidx: int, donors: Dict) -> None:
        """
        Executes one round of count donation from donors to acceptors.

        Modifies the internal count matrix for the specified spatial index in-place.

        Parameters
        ----------
        spidx : int
            The spatial index to process.
        donors : dict
            The dictionary of donors and their acceptor proportions, as returned
            by `get_current_donors`.
        """
        # Donates counts from donors to acceptors

        counts = self.counts[spidx]#.copy()

        for donor in donors:
            donor_counts = counts[donor]
            cbc_idxes_donate = np.argwhere(donor_counts).flatten()
            donor_counts.iloc[cbc_idxes_donate] -= 1

            acceptor_dict = donors[donor]
            props = [acceptor_dict[acceptor] for acceptor in acceptor_dict]
            cbc_idxes_split = dict(zip(acceptor_dict.keys(), distribute_by_proportions(cbc_idxes_donate, props)))
            for acceptor in cbc_idxes_split:
                acceptor_counts = counts[acceptor]
                acceptor_counts.iloc[cbc_idxes_split[acceptor]] += 1


    def iterate_donations(self,
                          save_gif_dir: Optional[Union[str, Path]] = None,
                          rm_figs: bool = True,
                          **kwargs: Any) -> None:
        """
        Iteratively performs count normalization until convergence.

        This method repeatedly calls `get_current_donors` and `donate` until the
        number of active donors drops below a threshold or the maximum number
        of iterations is reached.

        Parameters
        ----------
        save_gif_dir : str or Path, optional
            If provided, saves a GIF visualizing the normalization process for
            each spatial index in this directory.
        rm_figs : bool, optional
            If True, removes the intermediate PNG frames after creating the GIF.
        **kwargs
            Additional arguments passed to `create_gif_from_pngs`.
        """

        if save_gif_dir is not None:
            if not isinstance(save_gif_dir, Path):
                save_gif_dir = Path(save_gif_dir)
            if not save_gif_dir.exists():
                save_gif_dir.mkdir(parents=True)

        if self.layout.format != 'rowcol':
            # Because it's specifically ordering by arr-coord, which will error for
            # every other format due to a lack of one-to-one mapping. Otherwise nothing 
            # fundamentally wrong, so open to generalizing this in the future.
            if save_gif_dir is not None:
                warnings.warn("Saving GIFs is only supported for rowcol format layouts.")
                save_gif_dir = None
        
        max_iter = self.nb_params['max_iteration_donate']
        stop_iter_donors = self.nb_params['stop_iteration_donors']
        
        for spidx in self.counts:
            tmp_dir = Path(tempfile.mkdtemp(dir=save_gif_dir, prefix=f'.sp{spidx}_'))

            coord = self.layout.coords[spidx]
            if save_gif_dir is not None:
                if coord is not None:
                    coord_text = f'{coord}'
                else:
                    coord_text = '<no_coord>'
                spidx_text = f'sp{spidx}'
                
                fig, ax = self.plot_bcids(spidx, label='bcid', order_by=f'arr-{coord}')
                name = '-'.join([spidx_text, coord_text, 'img0'])
                ax.set_title(name)
                fig.savefig(tmp_dir / f'{name}.png', bbox_inches='tight')
                ylim = ax.get_ylim()
                plt.close()
            for i in range(max_iter):
                donors = self.get_current_donors(spidx)
                if len(donors) <= stop_iter_donors:
                    break
                self.donate(spidx, donors)
                if save_gif_dir is not None:
                    fig, ax = self.plot_bcids(spidx, label='bcid', order_by=f'arr-{coord}')
                    name = '-'.join([spidx_text, coord_text, f'img{i + 1}'])
                    ax.set_title(name)
                    fig.savefig(tmp_dir / f'{name}.png', bbox_inches='tight')
                    ax.set_ylim(ylim)
                    plt.close()
            if save_gif_dir is not None:
                create_gif_from_pngs(tmp_dir, output_gif=save_gif_dir / f'sp{spidx}{coord_text}.gif', **kwargs)
                if rm_figs:
                    for f in tmp_dir.glob('*.png'):
                        f.unlink()
                tmp_dir.rmdir()

    
    def normalize(self) -> None:
        """
        Applies the final normalized counts back to the MuData object.

        This method updates the `.X` matrix of the relevant modalities in the
        `mdata` object with the normalized count values, modifying it in-place.
        """

        for spidx, bctype in enumerate(self.chip.layout.bctypes):
            if bctype not in self.bctypes:
                continue

            cbc_bool = self.mdata[bctype].obs[self.chip_key_prop].isin([self.chip.num])
            row_indices = np.argwhere(cbc_bool.values).flatten()

            bcs = self.counts[spidx].columns.map({v: k for (k, v) in self.layout.mappers[spidx].to_dict().items()})
            col_indices = self.mdata[bctype].var_names.get_indexer_for(bcs)

            # # The goal here is to change the values of the .X on a subset of the data and avoid having
            # # to create entirely new AnnData objects that I need to later concatenate, which could not
            # # be done on a chip-by-chip basis as is intended with this class.

            # # I origianlly tried the following, but it raised a SparseEfficiencyWarning
            # self.mdata[bctype].X[np.ix_(row_indices, col_indices)] = self.counts[spidx].values
            # 
            # To avoid that warning, I tried the below, which raised am AnnData has FutureWarning about lil format
            # original_matrix = mdata[self.layout.bctypes[spidx]].X
            # mdata[self.layout.bctypes[spidx]].X = original_matrix.tolil()
            # mdata[self.layout.bctypes[spidx]].X[np.ix_(row_indices, col_indices)] = self.counts[spidx].values
            # mdata[self.layout.bctypes[spidx]].X = mdata[self.layout.bctypes[spidx]].X.tocsr()

            # So instead, I suppress the SparseEfficiencyWarning from scipy because I'm being deliberate about it
            # and it would be still more efficient than densifying the entire array (the only other way I see fit to
            # avoid either warning)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SparseEfficiencyWarning)
                self.mdata[bctype].X[np.ix_(row_indices, col_indices)] = self.counts[spidx].values
                

class SpatialCaller:
    """
    A class to call spatial barcodes based on the spatial layout and counts.
    """

    CALL_STRS = {
        # `filler_call_string`: used for initialization within calling methods, 
        # should not be present in final calls; checked at the end of add_spatial_calls
        'fill': '_',

        # `none_call_string`: used to initialize the calls column if it does not exist, 
        # will persist in final calls if any of the chips were not even attempted to be called
        # ... actually, I think this is actually not used in current implementation and is
        # carry-over from previous implementation when all cells were called together, but now
        # since this is done chip-by-chip, it might not appear at all in final calls; leave for 
        # now but consider removing in future
        'none': 'None',

        # `no_call_string`: used to identify cells that couldn't be called, either because
        # 1) they were attempted to be called but their spatial barcode profile was ambiguous
        # 2) they returned a spatial barcodes combination that does not exist in the layout
        'amb': 'no_call'
    }


    class CallResults:
        """
        A class to hold the results and state of the spatial calling.
        """
        def __init__(self):
            self.calls = None
            self.methods = []
            self.stats = []
            self.run = {
                'add_mods': False,
                'add_metrics': False,
                'add_calls': False,
            }

        def reset(self, state):

            if state == 'metrics':
                self.calls = None
                self.methods = None
                self.stats = None
                self.run['add_metrics'] = False
                self.run['add_calls'] = False
            elif state == 'calls':
                self.calls = None
                self.methods = None
                self.stats = None
                self.run['add_calls'] = False
            else:
                raise ValueError(f"Unknown state: {state}")


    def __init__(self, mdata: md.MuData, chip: Chip, chip_key_prop=None):
        """
        Initialize the SpatialCaller with mdata and chip.
        
        Parameters:
        - mdata: MuData object containing spatial data.
        - chip: Chip object defining the spatial layout.
        - chip_key_prop: Key for chip metadata property.
        """
        
        self.mdata = mdata
        self.chip = chip
        self.chip_key_prop = chip_key_prop

        self.spmods = None
        self.metrics = None
        self.spatial = self.CallResults()


    def reset(self, state=None):

        if state is None or state == 'mods':
            self.spmods = None
            self.metrics = None
            self.spatial = self.CallResults()
        elif state == 'metrics':
            self.metrics = None
            self.spatial.reset(state)
        elif state == 'calls':
            self.spatial.reset(state)
        else:
            raise ValueError(f"Unknown state: {state}")


    def add_mods(self):
        spmods = {}

        for spidx in range(len(self.chip.layout.bctypes)):
            spmods[spidx] = []

        for spidx, bctype in enumerate(self.chip.layout.bctypes):
            cbc_bool = self.mdata[bctype].obs[self.chip_key_prop].isin([self.chip.num])
            var_bool = self.chip.layout.mappers[spidx].index
            spmods[spidx] = self.mdata[bctype][cbc_bool, var_bool]

        if not reduce(lambda x, y: x.obs_names.equals(y.obs_names), spmods.values()):
            raise ValueError(
                "Cell barcodes are not consistent across barcode types for the specified chip. "
                "Please run validate_mdata_chipset to identify the issue."
                )

        self.spmods = spmods
        self.spatial.run['add_mods'] = True


    def add_metrics(self, top_bcs=3, gini_eps=1e-10, verbose=True):


        def compute_snrs(x, c):

            x_rev = x[::-1].values

            try:
                i, j = x_rev[c[0]], x_rev[c[1]]
                if i > 0 and j > 0:
                    return i/j
                elif i > 0 and j == 0:
                    return np.inf
                elif i == 0 and j == 0:
                    return 1
                else: # should never occur
                    return 0
            except IndexError:
                warnings.warn("Index out of bounds in SNR computation. Returning -1 for some comparisons.")
                return -1
            

        def compute_top_stats(x, t, snr_comps):

            # Get the top t unique values, the series is already sorted
            x_unique = x.drop_duplicates(keep='last')
            top_values = x_unique.values[-t:]
            
            indices = [x[x == value].index.tolist() for value in top_values]

            top_tags = indices[::-1] # Nested lists of non-unique

            top_counts = x.values[::-1][:t] # Non-unique
            top_counts_norm = (x/x.sum()).values[::-1][:t] # Non-unique
            top_snrs = [compute_snrs(x, c) for c in snr_comps] # Non-unique

            # top_counts_unique = top_values[::-1] # Unique
            # top_snrs_unique = [compute_snrs(x_unique, c) for c in snr_comps] # Unique

            return top_tags, top_counts, top_counts_norm, top_snrs#, top_counts_unique, top_snrs_unique

        
        def compute_gini(sorted, spidx, eps):
            
            gini_sorted_use = sorted[spidx] + eps

            ns = gini_sorted_use.groupby('cbc').size()
            unique_ns = ns.unique()
            indices_dict = dict(zip(unique_ns, [np.arange(1,n+1) for n in unique_ns]))
            indices = ns.map(indices_dict)

            def _compute_gini(x):
                """
                Calculate the Gini coefficient of a numpy array.
                Assumes array is flat, non-negative and pre-sorted.
                """
                # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
                # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
                n = ns[x.name]
                index = indices[x.name]
                return ((np.sum((2 * index - n  - 1) * x.values)) / (n * np.sum(x.values))) # Gini coefficient
            
            # Need ['count'], otherwise will divide by total counts across all tags
            s = gini_sorted_use.groupby('cbc')['count'].apply(_compute_gini).rename('gini')

            return s

        
        def compute_neg_entropy(sorted, spidx):
            '''
            (Negative) entropy = higher values are better.
            '''
            s = sorted[spidx]['count'].apply(lambda x: 0 if x == 0 else -1*x*np.log10(x))
            s = s.groupby('cbc').sum().rename('neg_entropy')
            return s
        

        def get_sorted(top_bcs):

            sorted = {spidx: [] for spidx in self.spmods}

            for spidx in self.spmods:
                cbc_bool = self.spmods[spidx].obs[self.chip_key_prop].isin([self.chip.num])
                vars = self.chip.layout.mappers[spidx].index
                if len(vars) < top_bcs:
                    raise ValueError(f"Number of spatial barcodes ({len(vars)}) is less than the requested top_bcs ({top_bcs}). Please reduce top_bcs.")

                df = self.spmods[spidx][cbc_bool, vars].to_df()

                df = df.melt(ignore_index=False, var_name='spbc', value_name='count').rename_axis('cbc')

                # Here ascending=True is critical because it is assumed for efficiency in all subsequent calculations
                df = df.sort_values(by=['cbc', 'count'], ascending=True).set_index('spbc', append=True)

                sorted[spidx] = df
            
            return sorted


        def get_topstats(sorted, top_bcs, snr_comps, seed=0):

            rng = np.random.default_rng(seed)

            topstats = {spidx: {} for spidx in self.spmods}

            for spidx in self.spmods:

                # The reset_index then groupby speeds up computation because we access the index in apply()
                topstats_s = sorted[spidx].reset_index('cbc').groupby('cbc')['count'].apply(compute_top_stats, top_bcs, snr_comps)
                tags = pd.DataFrame(data=[x[0] for x in topstats_s], index=topstats_s.index)
                tags = tags.map(lambda x: [] if x is None else x)

                topstats[spidx]['tags'] = tags.map(lambda x: x if len(x) == 1 else rng.permutation(x))
                # topstats[spidx]['tags'] = tags.map(lambda x: x if len(x) == 1 else ['Tie'])

                topstats[spidx]['counts'] = pd.DataFrame(data=[x[1] for x in topstats_s], index=topstats_s.index)
                topstats[spidx]['norm_counts'] = pd.DataFrame(data=[x[2] for x in topstats_s], index=topstats_s.index)
                topstats[spidx]['snrs'] = pd.DataFrame(data=[x[3] for x in topstats_s], index=topstats_s.index)

                
            return topstats

        # Make the snr_comps, which are the pairs of indices to compute SNRs for
        snr_comps = [(i, i+1) for i in range(top_bcs)]

        # Sort the counts, this is done first to speed up the computation of top stats
        if verbose:
            print("Sorting counts by cell barcode and count...")
        sorted = get_sorted(top_bcs=top_bcs)

        # Add the top stats, which are the top barcodes, their counts, normalized counts, and SNRs
        if verbose:
            print(f"Computing stats for the top {top_bcs} spatial barcodes in each index...")
        
        topstats = get_topstats(sorted, top_bcs, snr_comps)

        # Begin building the metrics dict
        metrics = {spidx: [] for spidx in self.spmods}

        if verbose:
            print("Adding the metrics for each spatial index...")

        for spidx in metrics:

            # Add gini index
            metrics[spidx].append(compute_gini(sorted, spidx, gini_eps))

            # Add negative entropy
            metrics[spidx].append(compute_neg_entropy(sorted, spidx))

            # Add top snrs
            ts_snrs = topstats[spidx]['snrs'].rename(columns=dict(enumerate(['snr_%d-%d' % comp for comp in snr_comps])))
            metrics[spidx].append(ts_snrs)

            # Add top barcodes from the top stats
            values = topstats[spidx]['tags'].apply(lambda x: np.concatenate(x)[:top_bcs], axis=1).tolist()
            top_bc_names = pd.DataFrame(values, index=topstats[spidx]['tags'].index, columns=['top%d' % i for i in range(top_bcs)])
            metrics[spidx].append(top_bc_names)

            # Add top raw counts and normalized counts
            ts_rcounts = topstats[spidx]['counts'].fillna(0).rename(columns=dict(enumerate(['rcount%d' % i for i in range(top_bcs)])))
            metrics[spidx].append(ts_rcounts)
            ts_ncounts = topstats[spidx]['norm_counts'].fillna(0).rename(columns=dict(enumerate(['ncount%d' % i for i in range(top_bcs)])))
            metrics[spidx].append(ts_ncounts)

            # Add the total raw counts per barcode
            tcounts = pd.Series(np.array(self.spmods[spidx].X.sum(1)).flatten(), index=self.spmods[spidx].obs.index, name='tcount')
            metrics[spidx].append(tcounts)

            # Bring it all together
            metrics[spidx] = pd.concat(metrics[spidx], axis=1).rename_axis('cbc', axis=0)

        self.metrics = metrics
        self.spatial.run['add_metrics'] = True

    
    def get_metrics_gb(self, gs):
        if not self.metrics:
            raise ValueError("Metrics have not been computed yet. Please run `add_metrics()` first.")
        return {mod: self.metrics[mod].join(self.spmods[mod].obs[gs]) for mod in self.metrics}


    def add_spatial_calls(self, method, method_kwargs={}, gb=None, reset=False, verbose=True):
        """
        Add spatial calls using one of the specified methods (`m01` or `m02`). The function 
        assigns combinatorial barcodes to cells and updates the `spcall` column.

        Parameters
        ----------
        method : str
            The spatial calling method to use. Options are:
            - 'm01': Basic method based on top-ranked barcodes.
            - 'm02': Advanced method incorporating spatial distances and barcode prioritization.
        method_kwargs : dict, optional
            Keyword arguments specific to the chosen method (default is an empty dictionary). See
            details below for method-specific parameters.
        gb : tuple, optional
            A 2-tuple for filtering cells by groups in Surveyor.mdata.obs.columns. The first element
            is a string specifying the grouping key (e.g., 'chip'), and the second is a list-like object 
            of values to include (default is None).
        reset : bool, optional
            Whether to reset the `spcall` column before running the method (default is True).

        Method-Specific Parameters
        --------------------------
        For `m01`:

        Basic method based on top-ranked barcodes. Metrics are queried and the top-ranked barcode
        is assigned to the cell.

        - call_qd : dict of str, optional
            Query strings for filtering barcodes to call, keyed by modality.
            (e.g., `{'sp1': "snr_0-1 > 5", 'sp2': "snr_1-2 > 3"}`).

        For `m02`:

        Advanced method incorporating spatial distances and barcode prioritization. Metrics are queried
        for exclusion (no_call) and inclusion (top barcode). For remaining cells, pairwise distances are
        calculated among top _n_ candidate barcodes in each spatial index to determine if any pair are 
        within `max_dist` of each other. If multiple candidate barcode pairs pass, the combinatorial 
        barcode pairs are scored according to a weighted combination of rank and the spatial distance 
        between them. Of the winning pair, the combinatorial barcode with the higher average rank is 
        chosen.

        - no_call_qd : dict of str, optional
            Query strings for filtering barcodes to exclude from calling, keyed by modality.
            (e.g., `{'sp1': "max_count < 10", 'sp2': "gini < 0.5"}`).
        - call_qd : dict of str, optional
            Query strings for filtering barcodes to call, keyed by modality.
            (e.g., `{'sp1': "snr_0-1 > 5", 'sp2': "snr_1-2 > 3"}`).
        - top_bcs : int, optional
            The number of top-ranked barcodes to consider for pairwise distance calculations
            (default is 3).
        - max_dist : float, optional
            The maximum allowed distance (in specified units) between barcodes to be considered
            as valid pairs (default is 500).
        - dist_units : str, optional
            Units for spatial distances. Options are 'wells' or 'um' (default is 'um').
        - r : float, optional
            Weight for the rank score in the combined distance-score metric (default is 0.8).
        - d : float, optional
            Weight for the spatial distance in the combined distance-score metric (default is 0.2).

        Returns
        -------
        None
            Both functions modify the `mdata` object in place:
            - `reset_spatial_calls`: Resets `spcall` to its default state.
            - `add_spatial_calls`: Updates `spcall` based on the chosen method.

        Raises
        ------
        ValueError
            If an invalid method is provided, required parameters are missing, or invalid
            `gb` or `dist_units` values are given.
        """

        def get_calls_calllist(calls_list):
            # Concatenate the list of DataFrames along columns and drop rows with any NaN values
            calls = pd.concat(calls_list, axis=1).dropna()

            # Join the values in each row with a hyphen
            calls = calls.apply(lambda x: '-'.join(x), axis=1)

            if calls.empty:
                return pd.Series()
            else:
                return calls

        def m01(metrics, mods, call_qd=None):
            
            mod0 = mods[0] # As the example mod
            stats = {}

            # Initialize spcall with a default value "_" for all cells
            bcs = metrics[mod0].index
            spcall = pd.Series(self.CALL_STRS['fill'], index=bcs, dtype='string')
            
            calls_list = []
            if call_qd is None:
                for mod in mods:
                    calls_list.append(metrics[mod]['top0'])
            else:
                for mod in mods:
                    query = call_qd[mod]
                    calls_list.append(metrics[mod].query(query)['top0'])

            calls = get_calls_calllist(calls_list)

            # Update spcall only for the indices present in calls
            spcall.loc[:] = self.CALL_STRS['amb']
            spcall.update(calls)

            # Stats are not necessary because only one filter
            stat = len(calls)/len(bcs) if len(bcs) > 0 else 0
            stats.update({'%called': stat})

            return spcall, stats

        def m02(metrics, mods, 
                no_call_qd=None, call_qd=None, top_bcs=3, max_dist=500, 
                dist_units='um', r=0.8, d=0.2):
            
            def score_pairs(rank_pairs, dists, min_dist, r, d):
                rank_score_raw = np.array([np.ravel(x).sum() for x in rank_pairs])
                rank_score = (max_rank - rank_score_raw)/max_rank
                # Should fix next line, right now not possible to be 1 because dists are always at least pitch
                denom = max_dist - min_dist
                if denom == 0:
                    d_score = 1
                else:
                    d_score = 1 - (dists - min_dist)/denom
                d_score = 1 - (dists - min_dist)/denom if denom > 0 else 1
                return r*rank_score + d*d_score


            def get_call(arrs, lyt, unit_convert, r, d):

                c = arrs['counts']
                t = arrs['tops']
                spidx_bcs_keep = [t_c[c_c > 0] for (c_c, t_c) in zip(c.T, t.T)]

                # Get combinations and ranks
                # Ranks only works because it's assumed/guaranteed that columns are sorted descendingly and we're only 
                # looking at the top_bcs, so the ranks are just the range(len(i))

                combs = list(it.product(*spidx_bcs_keep))
                ranks = list(it.product(*[range(len(i)) for i in spidx_bcs_keep]))
                combs_str = ['-'.join(i) for i in combs]

                # if check_bcs: # Originally had this as a kwarg, but decided to just always check
                comb_ranks_keep = [(i, j, k) for (i, j, k) in zip(combs_str, combs, ranks) if i in lyt._dfstacked.index]
                try:
                    combs_str, combs, ranks = zip(*comb_ranks_keep)
                except ValueError: # not enough values to unpack (expected 3, got 0)
                    return self.CALL_STRS['amb'], {}
                
                dists = pdist(lyt._dfstacked.loc[list(combs_str)].values)*unit_convert
                comb_pairs = np.fromiter(it.combinations(combs, 2), dtype='object')
                rank_pairs = np.fromiter(it.combinations(ranks, 2), dtype='object')
                keep_dists_bool =  (dists <= max_dist) # Exclude self-distances

                if keep_dists_bool.any():
                    dists_keep, rank_pairs_keep, comb_pairs_keep = dists[keep_dists_bool], rank_pairs[keep_dists_bool], comb_pairs[keep_dists_bool]
                    scores = score_pairs(rank_pairs_keep, dists_keep, min_dist=unit_convert, r=r, d=d)

                    sorted_idx_ranks = sorted(enumerate(scores), key=lambda x: x[1])[::-1]

                    top_idx = sorted_idx_ranks[0][0]
                    top_pair_ranks = rank_pairs_keep[top_idx]
                    
                    comb_call = '-'.join(comb_pairs_keep[top_idx][np.argmin([sum(i) for i in top_pair_ranks])])
                    return comb_call, {'top_pair_ranks': top_pair_ranks, 'score': scores[top_idx], 'dist': dists_keep[top_idx]}
                else:
                    mindist = dists.min() if dists.size > 0 else np.nan
                    return self.CALL_STRS['amb'], {'smallest_dist': mindist}

            def get_arrs(dfs, cbc):
                arrs = {}
                
                arrs['snrs'] = np.vstack([df.loc[cbc, df.columns.str.startswith('snr')].values.astype(float) for df in dfs]).T

                # Removes the case that there are n > top_bcs barcodes all with the same count, don't call, user should increase top_bcs
                if ((arrs['snrs'] == 1).sum(0) == arrs['snrs'].shape[0]).any(): 
                    return {}, False
                
                arrs['counts'] = np.vstack([df.loc[cbc, df.columns.str.startswith('rcount')].values.astype(int) for df in dfs]).T

                if (arrs['counts'].sum(0) == 0).any(): 
                    return {}, False
                # arrs['counts_norm'] = np.vstack([df.loc[cbc, df.columns.str.startswith('ncount')].values.astype(float) for df in dfs]).T

                arrs['tops'] = np.vstack([df.loc[cbc, df.columns.str.startswith('top')].values.astype(str) for df in dfs]).T
                # arrs['scores'], arrs['priorities'] = score_cell(arrs['snrs'], arrs['counts'])
                return arrs, True


            def get_keep_bcs():
                return spcall[spcall == self.CALL_STRS['fill']].index


            def get_rate(x, y):
                return len(x) / len(y) if len(y) > 0 else 0


            stats = {}
            mod0 = 0 # As the example mod
            
            # Initialize spcall with a default value "_" for all cells
            bcs = metrics[mod0].index
            spcall = pd.Series(self.CALL_STRS['fill'], index=bcs, dtype='string')
            

            # Set unit conversion based on the chip's array
            if dist_units == 'wells':
                unit_convert = 1
            elif dist_units == 'um':
                unit_convert = self.chip.array.pitch
            else:
                raise ValueError("Invalid dist_units.")

            # Step 1: Filter out cells with no_call_qd
            if no_call_qd is not None:
                no_call_cbc_list = []
                if verbose:
                    print("Filtering out cells with no_call_qd... ")
                for mod in mods:
                    no_call_cbc_list.append(metrics[mod].query(no_call_qd[mod]).index)    
                no_call_cbcs = reduce(np.intersect1d, no_call_cbc_list)

                spcall.loc[no_call_cbcs] = self.CALL_STRS['amb']

            else:
                if verbose:
                    print("No no_call_qd provided. Skipping filtering.")

            keep_bcs = get_keep_bcs()
            stats.update({f"%{self.CALL_STRS['amb']}_step1": 1 - get_rate(keep_bcs,bcs)})

            # Step 2: Call the cells using the call_qd
            if call_qd is not None:
                call_list = []
                if verbose:
                    print("Calling cells using call_qd... ")
                for mod in mods:
                    call_list.append(metrics[mod].loc[keep_bcs].query(call_qd[mod])['top0'])
                calls = get_calls_calllist(call_list)
                
                spcall.update(calls)

            else:
                calls = []
                if verbose:
                    print("No call_qd provided. Skipping calling.")
            
            keep_bcs = get_keep_bcs()
            stats.update({'%called_step2': get_rate(calls,bcs)})

            # Step 3: For remaining cells, call each cell's top_bc using multiple bcs
                    
            max_rank = (top_bcs*len(mods))*2 - 1

            snr_cols = ['snr_%d-%d' % (i, i+1) for i in range(top_bcs-1)]
            rcount_cols = ['rcount%d' % i for i in range(top_bcs)]
            top_cols = ['top%d' % i for i in range(top_bcs)]

            try:
                dfs = [metrics[mod].loc[keep_bcs, snr_cols + rcount_cols + top_cols] for mod in mods]
            except KeyError:
                err_msg = "Top-ranked barcode columns not found in metrics. Please run add_metrics with the " \
                    "appropriate number of `top_bcs` requested here (%d) ." % top_bcs
                raise ValueError(err_msg)
            
            if verbose:
                print("Computing spatial calls for remaining cells...")

            no_call_cbcs = []
            calls = []
            step3_stats = []

            for cbc in keep_bcs:
                arrs, passed = get_arrs(dfs, cbc)
                if passed:

                    lyt = self.chip.layout
                    
                    call, stat = get_call(arrs, lyt, unit_convert, r, d)
                    calls.append([cbc, call])
                    step3_stats.append([cbc, stat])
                else:
                    no_call_cbcs.append(cbc)
            
            calls = pd.DataFrame(calls, columns=['cbc', 'call']).set_index('cbc')['call']
            step3_stats = pd.DataFrame(step3_stats, columns=['cbc', 'stats']).set_index('cbc')['stats']

            spcall.update(calls)
            spcall.loc[no_call_cbcs] = self.CALL_STRS['amb']

            stats.update({f"%{self.CALL_STRS['amb']}_step3.1": get_rate(no_call_cbcs,bcs)})
            stats.update({f"%{self.CALL_STRS['amb']}_step3.2": get_rate(calls[calls == self.CALL_STRS['amb']],bcs)})
            stats.update({f"%{self.CALL_STRS['amb']}_step3.2": get_rate(calls[calls != self.CALL_STRS['amb']],bcs)})
            stats.update({'rank_info_step3': step3_stats})
                    
            return spcall, stats

        method_mapper = {'m01': m01, 'm02': m02}

        # check gb is a 2-tuple, gb[0] is str and gb[1] is list-like
        if gb is not None:
            gb_valid = isinstance(gb, tuple) and len(gb) == 2 and is_listlike(gb[1])
            if not gb_valid:
                raise ValueError("Param `gb` must be a 2-tuple with the first element as a str and the second element as a list-like.")
            gb = (gb[0], list(gb[1])) # To be sure that it can be saved in AnnData.uns later...
            
        mods = list(self.spmods.keys())


        all_comb_bcs = self.chip.layout.df_stacked.index.tolist() + list(self.CALL_STRS.values())

        if gb is not None:
            metrics = self.get_metrics_gb(gs=gb[0])
            metrics = {mod: metrics[mod][metrics[mod][gb[0]].isin(gb[1])] for mod in metrics}
        else:
            metrics = self.metrics
        
        spcall, stats = method_mapper[method](metrics, mods, **method_kwargs)
        spcall = spcall.astype('category')
        spcall = spcall.cat.add_categories([bc for bc in all_comb_bcs if bc not in spcall.cat.categories])
        if any([i not in all_comb_bcs for i in spcall.cat.categories]):
            warnings.warn(f"Some spatial calls for chip {self.chip.num} are not in the list of valid combinatorial " \
                          f"barcodes from the layout. These will be set to {self.CALL_STRS['amb']}. See method stats for details.")
            
            spcall[~spcall.isin(all_comb_bcs)] = self.CALL_STRS['amb']
            invalid_comb_bcs = [i for i in spcall.cat.categories if i not in all_comb_bcs]
            stats.update({'invalid_combinatorial_barcodes': invalid_comb_bcs})
            spcall = spcall.cat.remove_categories(invalid_comb_bcs)

        # Use the first mod as reference, they should all be the same (checked in add_mods)
        obs_names = self.spmods[0].obs_names

        if reset:
            self.reset(state='calls')
        elif self.spatial.calls is None:
            calls = pd.Series(self.CALL_STRS['none'], index=obs_names, dtype='string', name='spcall').astype('category')
            calls = calls.cat.add_categories([bc for bc in all_comb_bcs if bc not in calls.cat.categories])
            self.spatial.calls = calls
        
        self.spatial.calls.update(spcall)

        if self.spatial.calls.isin([self.CALL_STRS['fill']]).any():
            raise ValueError(
                f"Bug: Some cells were not processed and still have the filler call string '{self.CALL_STRS['fill']}'. "
                "Please confirm the spatial calling method is accurately handling cell calls.")
        
        self.spatial.methods.append({'method': method, 'method_kwargs': method_kwargs, 'gb': gb})
        self.spatial.stats.append(stats)
        self.spatial.run['add_calls'] = True

        return
        

    def get_call_rate(self, by=None):

        def _call_rate(x):
            return 100 * (1 - x.value_counts().loc[self.CALL_STRS['amb']] / x.shape[0])

        if by is None:
            by = self.chip_key_prop
        if self.spatial.calls is None:
            raise ValueError("No spatial calls have been made yet. Please run `add_spatial_calls()` first.")
        mod0 = 0 # As the example mod
        by_frame = self.spmods[mod0].obs[by].to_frame()

        return by_frame.join(self.spatial.calls).groupby([by], observed=True)['spcall'].apply(_call_rate)


class SpatialPositioner:

    # Note, the meanings of the variable names (meta, coords) is only within the 
    # context of this class, and not necessarily the same as is used elsewhere in
    # the survey package. Meta here is essentially the .obs of the resulting anndata
    # and coords is the .obsm[<spatial>] of the resulting anndata.

    class PositionResults:
        """
        A class to hold the results and state of the spatial calling.
        """
        def __init__(self):
            self.coords = None
            self.meta = None
            self.adata = None
            self.run = {
                'add_meta': False,
                'add_coords': False,
                'add_xyz': False,
            }

        def reset(self, state):
            if state == 'coords':
                self.coords = None
                self.run['add_coords'] = False
                self.run['add_xyz'] = False
                self.adata = None
            elif state == 'xyz':
                self.adata = None
                self.run['add_xyz'] = False
            else:
                raise ValueError(f"Unknown state: {state}")


    def __init__(self, mdata: md.MuData, chip: Chip, calls: pd.Series, chip_key_prop=None):
        """
        Initialize the SpatialPositioner with mdata and chip.
        
        Parameters:
        - mdata: MuData object containing spatial data.
        - chip: Chip object defining the spatial layout.
        - chip_key_prop: Key for chip metadata property.
        """

        bctype = chip.layout.bctypes[0] # Use the first bctype to get the cell barcodes
        chip_cbcs = frozenset(mdata[bctype].obs_names[mdata[bctype].obs[chip_key_prop] == chip.num])
        calls_cbcs = frozenset(calls.index)
        if chip_cbcs != calls_cbcs:
            raise ValueError("Cell barcodes in calls do not match those in mdata for the specified chip.")
        
        if chip.array.flat_top and chip.array.n == 4:
            # Assuming this is a square well because rectangular wells don't exist
            points_in_poly = False # Square wells will coincide with coordinate system
        else:
            points_in_poly = True

        self.mdata = mdata
        self.chip = chip
        self.calls = calls
        self.chip_key_prop = chip_key_prop

        self.points_in_poly = points_in_poly

        self.position = self.PositionResults()


    def add_meta(self):

        welldata_columns_pass = ['id', 'arr-row', 'arr-col', 'arr-x', 'arr-y', 'arr-center_x', 'arr-center_y', 'sp0', 'sp1']
        welldata_columns_pass_int = ['id', 'arr-row', 'arr-col', 'arr-x', 'arr-y']

        arr_positions_mappers = self.chip.get_welldata().dropna(axis=0).reset_index(drop=False).set_index('barcode')[welldata_columns_pass].to_dict()
        meta = pd.concat([self.calls.map(arr_positions_mappers[col]).rename(col) for col in welldata_columns_pass], axis=1).dropna(axis=0)
        meta[welldata_columns_pass_int] = meta[welldata_columns_pass_int].astype(int)

        meta[self.chip_key_prop] = self.chip.num

        self.position.meta = meta


    def add_coords(self, method='m01', seed=0, **kwargs):

        def _get_ranges(id):
            verts = self.chip.array.verts[id]
            range_min, range_max = verts.min(0), verts.max(0)
            xrange = range_min[0], range_max[0]
            yrange = range_min[1], range_max[1]

            return xrange, yrange

        def m01(id, count):
            """
            Uniformly randomly assign `count` positions within the well defined by `id`.
            """
            xrange, yrange = _get_ranges(id)

            if self.points_in_poly:
                # In the event the well is not square shaped and flat top
                # will need to check that the point is in the polygon defined by the well
                # not just the bounding box the well is inscribing
                # Will probably involve a while loop to confirm the point is in the polygon
                raise ValueError(
                    "Positioning within non-rectangular polygon not implemented yet."
                )
            else:
                centers = zip(*[rng.uniform(*r, size=count) for r in (xrange, yrange)])

            # print((id, centers.shape))

            for center in centers:
                yield center

        def m02(id, count, scale_norm=0.1):
            """
            Normally randomly assign `count` positions within the well defined by `id`.
            """
            xrange, yrange = _get_ranges(id)

            xrange_diff = xrange[1] - xrange[0]
            yrange_diff = yrange[1] - yrange[0]

            range_center = (xrange[0] + xrange_diff/2, yrange[0] + yrange_diff/2)
            
            # Create normal distribution parameters (mean, std) for each dimension
            x_params = (range_center[0], xrange_diff * scale_norm)
            y_params = (range_center[1], yrange_diff * scale_norm)
            
            # Generate centers
            centers = zip(*[rng.normal(*params, size=count) for params in (x_params, y_params)])

            for center in centers:
                yield center

        rng = np.random.default_rng(seed)

        possible_methods = ['m01', 'm02']
        if method not in possible_methods:
            raise ValueError(f"method must be one of {possible_methods}.")

        method_mapper = {
            'm01': m01,
            'm02': m02
        }

        default_method_params = {
            'm01': {},
            'm02': {'scale_norm': 0.1}
        }

        method_params = get_config(kwargs, default_method_params[method])

        id_counts = self.position.meta['id'].value_counts()
        assigners = {id: method_mapper[method](id, count, **method_params) for id, count in id_counts.items()}
        coords_arr = np.array([i for i in self.position.meta['id'].map(lambda x: next(assigners[x]))])
        coords = pd.DataFrame(coords_arr, index=self.position.meta.index, columns=['x', 'y'])

        self.position.coords = coords


    def add_xyz(self):

        xyz = sc.AnnData(X=self.position.meta[['arr-x', 'arr-y']].values,
                        obs=self.position.meta,
                        var=pd.DataFrame(index=['arr-x', 'arr-y']),
                        obsm={'X_survey': self.position.coords.values})

        self.position.xyz = xyz


class Survey:


    def __init__(self, mdata: md.MuData, 
                 chipset: ChipSet, 
                 chip_meta_key_prop=None, 
                 overwrite=False,
                 check_thresh=None) -> None:
        """
        A high-level class to orchestrate the spatial calling workflow.

        This class acts as a pipeline manager for processing Survey's single-cell
        spatial data. It takes a MuData object with raw counts and a ChipSet object
        defining the physical layouts, and provides methods to normalize counts,
        call cell barcodes, and assign spatial positions. The processing is
        performed on a chip-by-chip basis, and the final results are
        consolidated and applied back to the MuData object.

        Parameters
        ----------
        mdata : md.MuData
            The MuData object containing the raw count data, with separate
            modalities for each barcode type (e.g., 'bctype1', 'bctype2').
        chipset : ChipSet
            The ChipSet object that defines the layouts and properties of all
            chips in the experiment.
        chip_meta_key_prop : tuple, optional
            A tuple `(meta_key, chip_key_prop)` used to link cell metadata to
            chip numbers. `meta_key` is the column in a metadata file and
            `chip_key_prop` is the name of the new column to be created in `.obs`.
            Defaults to `('rxn', 'chip-num')`.
        overwrite : bool, optional
            If True, allows overwriting of existing chip number information in
            the MuData object. Defaults to False.

        Attributes
        ----------
        mdata : md.MuData
            The MuData object being processed.
        chipset : ChipSet
            The ChipSet definition for the experiment.
        chip_key_prop : str
            The key in `.obs` that identifies the chip number for each cell.
        faulty_bcs : dict
            A dictionary of barcodes with high counts that are not in the layout.
        faulty_counts : dict
            A dictionary of counts for the faulty barcodes.
        normers : dict
            A dictionary of `SpatialNormalizer` objects, keyed by chip number,
            after running `normalize_counts`.
        callers : dict
            A dictionary of `SpatialCaller` objects, keyed by chip number,
            after running `call_cells`.
        positioners : dict
            A dictionary of `SpatialPositioner` objects, keyed by chip number,
            after running `position_cells`.

        """

        if chip_meta_key_prop is None:
            meta_key, chip_key_prop = ('rxn', 'chip-num')
        else:
            if not isinstance(chip_meta_key_prop, tuple) or len(chip_meta_key_prop) != 2:
                raise TypeError("chip_meta_key_prop must be a tuple of (meta_key, chip_key_prop).")
            meta_key, chip_key_prop = chip_meta_key_prop

        for bctype in chipset.bctypes:
            if bctype not in mdata.mod:
                raise ValueError(f"Chip layout bctype {bctype} not found in mdata.mod. Please check your chip layout and mdata.")

        for mod in chipset.bctypes:
            # Adding props will check that metadata exists
            meta.add_prop_to_obs(mdata[mod], meta_key, chip_key_prop, overwrite=overwrite)
            mdata[mod].obs[chip_key_prop] = mdata[mod].obs[chip_key_prop].astype(int)
            meta.convert_to_categorical(mdata[mod].obs, chip_key_prop, verbose=False)
            
        # spmods = get_spmods_dict(mdata, chipset=chipset, chip_key_prop=chip_key_prop)
        faulty_bcs, faulty_counts = get_faulty_bcs(mdata, chipset=chipset, chip_key_prop=chip_key_prop, check_thresh=check_thresh)

        self.chip_key_prop = chip_key_prop
        self.mdata = mdata
        self.chipset = chipset
        self.faulty_bcs = faulty_bcs
        self.faulty_counts = faulty_counts

        self.spatial = None
        self.callers = {}


    def normalize_counts(self, bctypes=None):
        # save_gifs_dir=None, **kwargs
        """
        Normalize the counts in the spatial adatas.
        """

        # default_config = {}
        # kwargs = get_config(kwargs, default_config, protected=['save_gif_dir'])

        normers = {}

        for chip_num in self.chipset.chips:
            # if save_gifs_dir is not None:
            #     save_gif_dir = save_gifs_dir / f'chp{chip_num}/'
            # else:
            #     save_gif_dir = None

            chip = self.chipset.chips[chip_num]

            normer = SpatialNormalizer(self.mdata, chip, chip_key_prop=self.chip_key_prop, bctypes=bctypes)
            
            # normer.iterate_donations(save_gif_dir=save_gif_dir, **kwargs)
            normer.iterate_donations()

            normer.normalize()

            normers[chip_num] = normer
        
        self.normers = normers


    def call_cells(self, 
                   chipnums=None, 
                   metrics_kwargs=None, 
                   spatial_call_kwargs=None, 
                   verbose=True):
        """
        Call cells using the spatial calling methods.
        This will modify the counts in place.
        """

        if chipnums is None:
            chipnums = list(self.chipset.chips.keys())
        elif not is_listlike(chipnums):
            chipnums = [chipnums]
        chipnums = [c for c in chipnums if c in self.chipset.chips]
        if len(chipnums) == 0:
            raise ValueError("No valid chip numbers provided.")
        # chipnums = list(self.chipset.chips.keys())

        default_metrics_kwargs = {
            'top_bcs': 5,
            'gini_eps': 1e-10,
            'verbose': False
        }

        if len(self.chipset.bclens) > 1:
            raise ValueError(
                "Survey calling is not yet implemented for chipsets with a mixed number of" \
                " combinatorial spatial indices across chips (i.e. combinatorial barcode of " \
                "only 2 indices on some chips and 3 or more on others). Please call each chip " \
                "separately using SpatialCaller instead."
            )
        else:
            num_spidxes = list(self.chipset.bclens)[0]
            no_call_qd = dict(it.product(range(num_spidxes), ['(`tcount` < 5)']))
            call_qd = dict(it.product(range(num_spidxes), ['(`snr_0-1` >= 2)']))

        default_spatial_call_kwargs = {
            'method': 'm02',
            'method_kwargs': {
                'no_call_qd': no_call_qd,
                'call_qd': call_qd,
                'top_bcs': 5,
                'max_dist': 250,
                'dist_units': 'um'
            },
            'gb': None,
            'reset': False,
            'verbose': False
        }

        metrics_kwargs = get_config(metrics_kwargs, default_metrics_kwargs)
        spatial_call_kwargs = get_config(spatial_call_kwargs, default_spatial_call_kwargs)

        if spatial_call_kwargs['method'] == 'm01': # Only call_qd is needed
            method_kwargs = {}
            for key in ['call_qd']:
                method_kwargs[key] = spatial_call_kwargs['method_kwargs'].get(key)
            spatial_call_kwargs['method_kwargs'] = method_kwargs
            requested_top_bcs = metrics_kwargs.get('top_bcs')
        elif spatial_call_kwargs['method'] == 'm02':
            requested_top_bcs = max([metrics_kwargs.get('top_bcs'), 
                                     spatial_call_kwargs['method_kwargs'].get('top_bcs')])

        callers = {}

        not_enough_bcs = []
        for chipnum in chipnums:
            chip = self.chipset.chips[chipnum]
            for spidx in chip.layout.mappers:
                num_bcs_spidx = chip.layout.mappers[spidx].shape[0]
                if num_bcs_spidx < requested_top_bcs:
                    not_enough_bcs.append((chipnum, spidx, num_bcs_spidx))
        
        if len(not_enough_bcs) > 0:
            for chipnum, spidx, num_bcs_spidx in not_enough_bcs:
                msg = f"Chip {chipnum}, spatial index {spidx} has only {num_bcs_spidx} barcodes, " \
                 f"which is less than the requested top_bcs of {requested_top_bcs}. Please adjust the " \
                 "top_bcs parameter accordingly."
                print(msg)
            raise ValueError("Some chips do not have enough barcodes for the requested top_bcs.")

        for chip_num in chipnums:
            chip = self.chipset.chips[chip_num]

            if verbose:
                print(f"Spatial calling chip {chip_num}...")

            caller = SpatialCaller(self.mdata, chip, chip_key_prop=self.chip_key_prop)
            
            # Add spatial mods
            caller.add_mods()

            # Add metrics
            caller.add_metrics(**metrics_kwargs)

            # Add spatial calls
            caller.add_spatial_calls(**spatial_call_kwargs)

            callers[chip_num] = caller

        self.callers.update(callers)


    def get_call_rates(self, by=None):
        if self.callers is None:
            raise ValueError("No callers found. Please run `call_cells()` first.")
        return pd.concat([caller.get_call_rate(by=by) for caller in self.callers.values()])


    def position_cells(self, chips_ignore=None, add_coords_kwargs={}):

        default_positioner_kwargs = {
            'method': 'm02'
        }

        add_coords_kwargs = get_config(add_coords_kwargs, default_positioner_kwargs)

        positioners = {}

        for chip_num in self.chipset.chips:
            if chips_ignore is not None and chip_num in chips_ignore:
                continue
            chip = self.chipset.chips[chip_num]
            caller = self.callers[chip_num]

            positioner = SpatialPositioner(self.mdata, chip, caller.spatial.calls, chip_key_prop=self.chip_key_prop)

            positioner.add_meta()

            positioner.add_coords(**add_coords_kwargs)

            positioner.add_xyz()

            positioners[chip_num] = positioner

        self.positioners = positioners


    def apply(self):
        # Spatial modalities
        spmods = {}

        for spidx in range(self.chipset.bclens[0]):
            spmods[spidx] = []

        for chip_num in self.callers:
            for spidx in self.callers[chip_num].spmods:
                spmods[spidx].append(self.callers[chip_num].spmods[spidx])

        for spidx in spmods:
            spmods[spidx] = sc.concat(spmods[spidx], uns_merge=None)
            # Need to convert the var_names to be unique across modalities
            spmods[spidx].var_names = convert_names(spmods[spidx].var_names, prefix=f'sp{spidx}')

        # XYZ modality
        xyzs = []

        for chip in self.positioners:
            xyzs.append(self.positioners[chip].position.xyz)
        xyz = sc.concat(xyzs, uns_merge=None)

        ## Add the chip key property to make future usage more streamlined
        self.chipset._chip_key_prop = self.chip_key_prop

        xyz.uns['survey'] = self.chipset

        # Add everything to mdata
        self.mdata.mod['xyz'] = xyz
        for spidx in spmods:
            self.mdata.mod[f'sp{spidx}'] = spmods[spidx]

        self.mdata.update()