# Built-ins
import os
import re
from typing import Dict, Tuple, Union, Optional, List
from pathlib import Path

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

class SurveyPaths:
    """
    Class to manage and validate directory structure for Survey Genomics datasets.
    This class helps in organizing and accessing the directory structure of datasets
    provided by Survey Genomics, which may include multiple experiments or tissues.

    Parameters
    ----------
    data_dir : str or Path
        The root directory containing the dataset. This directory should have
        subdirectories for experiments or a 'general' directory for tissues.
    data_id : str
        An identifier for the dataset, which can be either an experiment ID
        (e.g., 'exp001') or a tissue name (e.g., 'liver').

    Attributes
    ----------
    cr : Path or None
        Path to the cellranger directory if an experiment ID is provided; None for tissue IDs.
    h5s : Path or None
        Path to the directory containing .h5mu files. For experiment IDs, this is
        under the experiment directory; for tissue IDs, it's under 'general/h5s'.
    imgs : Path or None
        Path to the directory containing images. This may be None if no images are available.

    Methods
    -------
    get_h5_paths(name=None, last=False, formats=None)
        Retrieves paths to .h5mu files in the h5s directory based on specified criteria.
    
    Raises
    ------
    ValueError
        If the provided data_dir does not exist, if the data_id is not recognized,
        or if required subdirectories are missing.
    TypeError
        If data_id is not a string.
    """

    def __init__(self, data_dir, data_id):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if not data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
        else:
            self.data_dir = data_dir

        id_detected = False

        if not isinstance(data_id, str):
            raise ValueError("ID must be a string")
        if re.match(r'exp\d{3}', data_id):
            if not (self.data_dir / data_id).exists():
                raise ValueError(f"Experiment directory {self.data_dir / data_id} does not exist")
            else:
                is_exp = True
                exp = data_id
                id_detected = True
        else:
            is_exp = False
            tissue = data_id

        if is_exp:
            cr_dir = self.data_dir / exp / 'cr'
            if cr_dir.exists():
                self.cr = cr_dir
            else:
                raise ValueError(f"The cellranger directory {cr_dir} does not exist. Please ensure its synced.")

            h5s_dir = self.data_dir / exp / 'h5s'
            if h5s_dir.exists():
                self.h5s = h5s_dir
            else:
                self.h5s = None
            
            imgs_dir = self.data_dir / exp / 'imgs'
            if imgs_dir.exists():
                self.imgs = imgs_dir
            else:
                self.imgs = None
        else:
            # The id represents a tissue
            general_dir = self.data_dir / 'general'
            if not general_dir.exists():
                raise ValueError(f"The general directory {general_dir} does not exist. Please ensure its synced.")

            general_h5s_dir = general_dir / 'h5s'
            if not general_h5s_dir.exists():
                raise ValueError(f"The general h5s directory {general_h5s_dir} exists. Please provide an experiment ID instead of a tissue ID.")

            tissue_h5s_dir = general_h5s_dir / tissue
            if tissue_h5s_dir.exists():
                id_detected = True
                self.cr = None
                self.h5s = tissue_h5s_dir
            
            imgs_dir = general_dir / 'imgs'
            if imgs_dir.exists():
                self.imgs = imgs_dir
            else:
                self.imgs = None

        if not id_detected:
            raise ValueError(f"ID {data_id} is not recognized as a valid experiment or tissue in {data_dir}")
    
    def get_h5_paths(self, name=None, last=False, formats=None):
        """
        Retrieves paths to .h5mu files in the h5s directory based on specified criteria.

        Parameters
        ----------
        name : str, optional
            A substring to filter the .h5mu files by name. If provided, only files
            containing this substring in their name will be returned.
        last : bool, optional
            If True, returns only the most recently modified .h5mu file.
            Cannot be used in conjunction with `name`.
        formats : list of str, optional
            A list of file formats/extensions to look for. Defaults to ['h5mu'].

        Returns
        -------
        list of Path or Path
            A list of paths to the .h5mu files matching the criteria, or a single
            Path if `name` or `last` is specified and only one file matches.
        """
        
        if formats is None:
            formats = ['h5mu']

        if name is not None and last:
            raise ValueError("Only one of 'name' or 'last' should be provided")
        
        formats = [i.strip('.') for i in formats] # in case any were provided with leading '.'
        suffixes = ['.' + f for f in formats]

        h5_paths = [self.h5s / i for i in map(Path, sorted(os.listdir(self.h5s))) if i.suffix in suffixes]

        if len(h5_paths) == 0:
            raise ValueError(f"No files with suffixes {suffixes} found in {self.h5s}")

        if name is not None:
            h5_paths_filt = [i for i in h5_paths if name == i.stem]
            if len(h5_paths_filt) == 0:
                raise ValueError(f"No files found with name containing '{name}' in {self.h5s}")
            elif len(h5_paths_filt) == 1:
                return h5_paths_filt[0]
        elif last:
            h5_paths_filt = h5_paths[-1]
            return h5_paths_filt

        if len(h5_paths) == 1:
            return h5_paths[0]
        
        return h5_paths

# # Not sure what this was for?
# def update_cell_type_annotations(
#     mdata: md.MuData,
#     cttypes: List[str],
#     ct_annots_path: Path,
#     ct_meta_path: Path
# ) -> None:
#     """
#     Update cell type annotations across modalities in a MuData object.
    
#     This function updates cell type annotations by:
#     1. Detecting existing cell type columns across all modalities
#     2. Removing detected columns in non-RNA modalities after user confirmation
#     3. Loading new annotations and applying them to the RNA modality
#     4. Updating metadata in the RNA modality
#     5. Transferring annotations to other modalities
    
#     Parameters
#     ----------
#     mdata : md.MuData
#         MuData object containing multiple modalities.
#     cttypes : list of str
#         List of cell type column names to update (e.g., ['ct1', 'ct2', 'ct3']).
#     ct_annots_path : Path
#         Path to CSV file containing cell type annotations with index matching 
#         observation names.
#     ct_meta_path : Path
#         Path to pickle file containing metadata dictionary for cell type columns.
#         Must contain all cttypes as keys.
    
#     Raises
#     ------
#     ValueError
#         If 'rna' modality doesn't exist, if chained modality prefixes are detected,
#         if cttypes are missing from metadata file, or if user cancels the operation.
#     FileNotFoundError
#         If annotation or metadata files don't exist.
    
#     Notes
#     -----
#     The function detects columns matching patterns:
#     - `<cttype>` (direct column)
#     - `<mod>.<cttype>` (transferred column from another modality)
    
#     Chained patterns like `xyz.rna.ct3` will raise an error.

    
#     Metadata file should have been created with the following:
#     ```
#     paths['ctmeta'] = paths['data'] / 'general/pkls/tmp-ct-meta.pkl'
#     cttypes = ['ct1', 'ct2', 'ct3']
#     ctmeta = {}
#     for cttype in cttypes:
#         ctmeta[cttype] = mdata['rna'].uns['meta'][cttype].copy()
#     pklop(ctmeta, paths['ctmeta'])
#     ```
    
#     Examples
#     --------
#     >>> from pathlib import Path
#     >>> update_cell_type_annotations(
#     ...     mdata,
#     ...     cttypes=['ct1', 'ct2', 'ct3'],
#     ...     ct_annots_path=Path('data/ct-annots.csv'),
#     ...     ct_meta_path=Path('data/ct-meta.pkl')
#     ... )
#     """
#     import re
#     from survey.genutils import pklop
    
#     print("=" * 80)
#     print("CELL TYPE ANNOTATION UPDATE")
#     print("=" * 80)
    
#     # Check that RNA modality exists
#     print("\n[1/8] Validating modalities...")
#     if 'rna' not in mdata.mod.keys():
#         raise ValueError("The 'rna' modality must exist in the MuData object.")
#     print("  ✓ RNA modality found")
    
#     # Check that cttypes columns exist in RNA modality
#     print("\n[2/8] Validating cttype columns in RNA modality...")
#     missing_columns = [ct for ct in cttypes if ct not in mdata['rna'].obs.columns]
#     if missing_columns:
#         raise ValueError(
#             f"The following cttype columns are missing from RNA modality: {missing_columns}"
#         )
#     print(f"  ✓ All cttype columns found: {cttypes}")
    
#     # Check that files exist
#     print("\n[3/8] Validating input files...")
#     if not ct_annots_path.exists():
#         raise FileNotFoundError(f"Annotations file not found: {ct_annots_path}")
#     if not ct_meta_path.exists():
#         raise FileNotFoundError(f"Metadata file not found: {ct_meta_path}")
#     print(f"  ✓ Annotations file: {ct_annots_path}")
#     print(f"  ✓ Metadata file: {ct_meta_path}")
    
#     # Load and validate metadata
#     print("\n[4/8] Loading metadata...")
#     ct_meta = pklop(ct_meta_path)
#     missing_cttypes = [ct for ct in cttypes if ct not in ct_meta.keys()]
#     if missing_cttypes:
#         raise ValueError(
#             f"The following cttypes are missing from metadata file: {missing_cttypes}"
#         )
#     print(f"  ✓ Metadata loaded with keys: {list(ct_meta.keys())}")
    
#     # Detect existing cell type columns across all modalities
#     print("\n[5/8] Detecting existing cell type columns...")
    
#     # Pattern: either direct column name or <mod>.<cttype>
#     # NOT allowing chained like xyz.rna.ct3
#     pattern = re.compile(r'^(?:([a-z]{3})\.)?(' + '|'.join(re.escape(ct) for ct in cttypes) + ')$')
    
#     detected_cols = {}  # {modality: [columns_to_remove]}
    
#     for mod in mdata.mod.keys():
#         detected_cols[mod] = []
#         for col in mdata[mod].obs.columns:
#             match = pattern.match(col)
#             if match:
#                 prefix, cttype = match.groups()
                
#                 # Check for chained modality (e.g., xyz.rna.ct3)
#                 if prefix and '.' in prefix:
#                     raise ValueError(
#                         f"Detected chained modality prefix in column '{col}' "
#                         f"in modality '{mod}'. This is not supported."
#                     )
                
#                 detected_cols[mod].append(col)
    
#     # Filter out modalities with no detected columns
#     detected_cols = {mod: cols for mod, cols in detected_cols.items() if cols}
    
#     if not detected_cols:
#         print("  ℹ No existing cell type columns detected")
#     else:
#         print("  ⚠ Detected cell type columns:")
#         for mod, cols in detected_cols.items():
#             print(f"    - {mod}: {cols}")
    
#     # Confirm removal with user
#     if detected_cols:
#         print("\n[6/8] Confirming column removal...")
#         # Filter out RNA modality from removal
#         cols_to_remove = {mod: cols for mod, cols in detected_cols.items() if mod != 'rna'}
        
#         if not cols_to_remove:
#             print("  ℹ Only RNA modality columns detected, skipping removal...")
#         else:
#             print("  ⚠ Will remove columns from non-RNA modalities:")
#             for mod, cols in cols_to_remove.items():
#                 print(f"    - {mod}: {cols}")
            
#             response = input("  Remove all detected columns from non-RNA modalities? (yes/no): ").strip().lower()
#             if response not in ['yes', 'y']:
#                 raise ValueError("Operation cancelled by user.")
            
#             # Remove detected columns (excluding RNA)
#             print("  Removing columns...")
#             for mod, cols in cols_to_remove.items():
#                 mdata[mod].obs.drop(columns=cols, inplace=True)
#                 print(f"    ✓ Removed from {mod}: {cols}")
#     else:
#         print("\n[6/8] No columns to remove, skipping...")
    
#     # Load annotations and apply to RNA modality
#     print("\n[7/8] Updating RNA modality annotations...")
#     ctannots = pd.read_csv(ct_annots_path, index_col=0)
#     print(f"  ✓ Loaded annotations with shape: {ctannots.shape}")
    
#     for cttype in cttypes:
#         print(f"  - Updating {cttype}...")
#         mdata['rna'].obs[cttype] = mdata['rna'].obs[cttype].astype(str)
#         mdata['rna'].obs.update(ctannots[[cttype]])
#         mdata['rna'].obs[cttype] = mdata['rna'].obs[cttype].astype('category')
#         print(f"    ✓ {cttype} updated")
    
#     # Update metadata in RNA modality
#     print("  - Updating metadata...")
#     if 'meta' not in mdata['rna'].uns:
#         mdata['rna'].uns['meta'] = {}
    
#     for cttype in cttypes:
#         # Remove old metadata if it exists
#         if cttype in mdata['rna'].uns['meta']:
#             del mdata['rna'].uns['meta'][cttype]
#         # Add new metadata
#         mdata['rna'].uns['meta'][cttype] = ct_meta[cttype]
#         print(f"    ✓ Metadata updated for {cttype}")
    
#     # Transfer to other modalities
#     print("\n[8/8] Transferring annotations to other modalities...")
    
#     # Get list of modalities that need transfer (excluding RNA)
#     target_mods = [mod for mod in detected_cols.keys() if mod != 'rna']
    
#     if not target_mods:
#         print("  ℹ No other modalities detected, skipping transfer")
#     else:
#         for target_mod in target_mods:
#             print(f"  - Transferring to {target_mod}...")
#             svc.obs.transfer_obs(
#                 mdata, 
#                 mods=('rna', target_mod), 
#                 columns=cttypes, 
#                 meta=True
#             )
#             print(f"    ✓ Transfer complete")
    
#     print("\n" + "=" * 80)
#     print("UPDATE COMPLETE")
#     print("=" * 80)

# # See Docstring Notes on how the metadata file should be created

# # Usage
# paths['ctannots'] = paths['ct_annots'] = paths['data'] / f'general/vals/{tissue}/ct-annots.csv'
# paths['ctmeta'] = paths['data'] / 'general/pkls/tmp-ct-meta.pkl'
# cttypes = ['ct1', 'ct2', 'ct3']

# update_cell_type_annotations(
#     mdata,
#     cttypes=cttypes,
#     ct_annots_path=paths['ctannots'],
#     ct_meta_path=paths['ctmeta']
# )