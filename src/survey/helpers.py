# Built-ins
import re
from typing import Dict, Tuple, Union, Optional

# Standard libs
import pandas as pd

def get_chipset_params(meta_rxn: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Extracts array parameters and chip metadata from a reaction metadata DataFrame.

    This function parses a DataFrame, typically derived from a "Compiled Results"
    spreadsheet in Survey Genomics data processing, to create the necessary
    parameter dictionaries for constructing `ChipSet` objects.

    Parameters
    ----------
    meta_rxn : pd.DataFrame
        A DataFrame containing reaction metadata with expected columns like
        'chip-version', 'well-shape', 'chip-num', 'layout-id', etc.

    Returns
    -------
    array_params : dict
        A dictionary where keys are chip versions and values are dictionaries
        of the physical array parameters (n, s, w, arr_shape).
    chip_meta : pd.DataFrame
        A DataFrame indexed by chip number containing metadata for each chip,
        such as version, layout ID, and offset.

    Raises
    ------
    KeyError
        If required columns are missing from the input DataFrame.
    ValueError
        If there are inconsistencies in the metadata, such as multiple
        different definitions for the same chip version or invalid offset formats.
    TypeError
        If `meta_rxn` is not a pandas DataFrame.
    """

    shape_to_n = {
        'Square': 4,
        'Hexagon': 6,
    }

    array_params_cols_mapper = {
        'well-shape': 'n',
        'well-side-length': 's',
        'wall-width': 'w',
    }

    chip_meta_cols_mapper = {
        'chip-version': 'version',
        'chip-num': 'num',
        'layout-id': 'lyt',
        'layout-arr-offset': 'offset',
        'image_file': 'img',
    }

    def convert_offset(offset_string: str, match: bool = False) -> Union[Tuple[int, int], Optional[re.Match]]:
        """
        Converts or validates an offset string '(x, y)'.

        Parameters
        ----------
        offset_string : str
            The string to process, e.g., '(0, 1)'.
        match : bool, optional
            If True, validates the string format and returns a regex match
            object or None. If False, converts the string to a tuple of ints.

        Returns
        -------
        tuple of (int, int) or re.Match or None
            The converted tuple or the result of the regex match.
        """
        if match:
            return re.match(r'^\(\s*\d+\s*,\s*\d+\s*\)$', offset_string)
        else:
            return tuple([int(i.strip()) for i in offset_string.strip('()').split(',')])
        
    def get_array_params(meta_rxn: pd.DataFrame) -> Dict:
        """
        Extracts and formats physical array parameters from the metadata.

        Parameters
        ----------
        meta_rxn : pd.DataFrame
            The reaction metadata DataFrame.

        Returns
        -------
        dict
            A dictionary of array parameters, keyed by chip version.
        """

        array_params_cols = ['chip-version', 'well-shape', 'well-side-length', 'wall-width', 'arr-num-rows', 'arr-num-cols']
        try:
            pre_array_params = meta_rxn[array_params_cols]
        except KeyError as e:
            raise KeyError(f"Meta DataFrame must contain the following columns: {array_params_cols}. Missing column: {e.args[0]}")
        try:
            pre_array_params = pre_array_params.drop_duplicates().set_index('chip-version')
        except ValueError as e:
            raise ValueError(
                f"Meta DataFrame must have identical parameters for the same 'chip-version', "
                f"but it appears to have inconsistent duplicates. Error: {e.args[0]}"
                )
        
        pre_array_params.rename(columns=array_params_cols_mapper, inplace=True)
        pre_array_params['n'] = pre_array_params['n'].map(shape_to_n)
        pre_array_params = pre_array_params.to_dict(orient='index')
        
        array_params = {}
        keys_keep = ['n', 's', 'w']
        for chip_version in pre_array_params:
            array_params[chip_version] = {k: pre_array_params[chip_version][k] for k in keys_keep}
            arr_shape = (int(pre_array_params[chip_version]['arr-num-rows']), int(pre_array_params[chip_version]['arr-num-cols']))
            array_params[chip_version]['arr_shape'] = arr_shape
        return array_params
    
    def get_chip_meta(meta_rxn: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and formats chip-specific metadata.

        Parameters
        ----------
        meta_rxn : pd.DataFrame
            The reaction metadata DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame of chip metadata, indexed by chip number.
        """

        chip_meta = meta_rxn[~meta_rxn['chip-num'].duplicated()].reset_index()[list(chip_meta_cols_mapper.keys())].rename(columns=chip_meta_cols_mapper).set_index('num')
        chip_meta.index = chip_meta.index.astype(int)

        offset_match = chip_meta['offset'].apply(convert_offset, match=True)
        if not offset_match.all():
            raise ValueError(f"Invalid layout array offset format at: {chip_meta['offset'][~offset_match]}")

        chip_meta['offset'] = chip_meta['offset'].apply(convert_offset, match=False)
        chip_meta['extent'] = 'auto'

        return chip_meta
    
    if not isinstance(meta_rxn, pd.DataFrame):
        raise TypeError("meta_rxn must be a pandas DataFrame")

    array_params = get_array_params(meta_rxn)
    chip_meta = get_chip_meta(meta_rxn)

    return array_params, chip_meta
