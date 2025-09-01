# Built-ins
from pathlib import Path
import re
import os
from functools import reduce
import warnings
from typing import Union, Dict, List, Optional, Tuple

# Standard libs
import numpy as np
import pandas as pd

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.singlecell.scutils import QuietScanpyLoad, filter_var
from survey.genutils import pklop

ADATA_SUFFIX = '.h5ad'
MDATA_SUFFIX = '.h5mu'

def parse_experiment_string(input_string: str, 
                            is_match: bool = False) -> Union[bool, Dict[str, Optional[str]]]:
    """
    Parses an experiment string to extract experiment, sample, and tag information.

    The expected format is 'exp<number>_S<number>[_tag]'. Case-insensitive.

    Parameters
    ----------
    input_string : str
        The string to parse, e.g., 'exp001_S1_SBC'.
    is_match : bool, optional
        If True, returns a boolean indicating if the string matches the format.
        If False (default), returns a dictionary with parsed components or raises
        a ValueError.

    Returns
    -------
    bool or dict
        If `is_match` is True, returns True if `input_string` matches the pattern,
        False otherwise.
        If `is_match` is False, returns a dictionary with keys 'exp', 'sample',
        and 'tag'. 'tag' is None if not present.

    Raises
    ------
    ValueError
        If `is_match` is False and `input_string` does not match the expected
        format.
    """
    # Define the regex
    pattern = r"(?i)(exp)(\d+)_S(\d+)(?:_(\w+))?"
    
    # Match the input string
    match = re.match(pattern, input_string)

    if is_match:
        if match:
            return True
        else:
            return False
    else:
        if match:
            # Extract the groups into a dictionary
            return {
                "exp": 'exp' + match.group(2),  # The number after "exp"
                "sample": "S" + match.group(3),    # The number after "S"
                "tag": match.group(4)      # The optional third part (e.g., SBC)
            }
        else:
            # Raise an error if the string doesn't match the required format
            raise ValueError(f"Expected format for input string is 'exp###_S#[_tag]', but got '{input_string}'.")


def detect_cellranger_run(path_to_outs: Union[str, Path]) -> bool:
    """
    Detects if a Cell Ranger run was 'multi' or 'count'.

    Checks for the presence of characteristic files/directories for either
    `cellranger multi` or `cellranger count` in the provided 'outs' directory.

    Parameters
    ----------
    path_to_outs : str or Path
        Path to the Cell Ranger 'outs' directory.

    Returns
    -------
    bool
        True if it's a 'multi' run, False if it's a 'count' run.

    Raises
    ------
    ValueError
        If the directory structure does not clearly match 'multi' or 'count',
        or if key files are missing.
    """

    cellranger_multi_files = ['multi', 'per_sample_outs']
    cellranger_count_files = ['raw_feature_bc_matrix.h5', 'filtered_feature_bc_matrix']

    in_outs = os.listdir(path_to_outs)

    if all([i in in_outs for i in cellranger_multi_files]):
        multi_run = True
    elif any([i in in_outs for i in cellranger_multi_files]):
        raise ValueError(
            f"One of the cellranger multi files is missing at {path_to_outs}. "
            "Expected files/dir: " + ', '.join(cellranger_multi_files) + ". "
            "Cannot determine cellranger run type.")
    elif all([i in in_outs for i in cellranger_count_files]):
        multi_run = False
    elif any([i in in_outs for i in cellranger_count_files]):
        raise ValueError(
            f"One of the cellranger count files is missing at {path_to_outs}. "
            "Expected files/dir: " + ', '.join(cellranger_count_files) + ". "
            "Cannot determine cellranger run type.")
    else:
        raise ValueError(
            f"Neither cellranger multi nor cellranger count files were found at {path_to_outs}. "
            "Expected files/dir: " + ', '.join(cellranger_multi_files + cellranger_count_files) + ". "
            "Cannot determine cellranger run type.")
    return multi_run


def is_multi_run_multiplexed(id: str, 
                             path_to_crout: Union[str, Path]) -> Tuple[bool, Optional[List[str]]]:
    """
    Determines if a `cellranger multi` run was multiplexed.

    Checks the 'per_sample_outs' directory for multiple sample subdirectories.

    Parameters
    ----------
    id : str
        The identifier of the run, expected to match the single sample directory
        name if not multiplexed.
    path_to_crout : str or Path
        Path to the Cell Ranger output directory (the parent of 'outs').

    Returns
    -------
    multiplexed : bool
        True if the run was multiplexed (more than one sample found).
    multiplexed_samples : list of str or None
        A list of sample names if multiplexed, otherwise None.

    Raises
    ------
    ValueError
        If the 'per_sample_outs' directory is empty or contains a single sample
        that does not match the provided `id`.
    """

    samples_in_per_sample_outs = []

    for i in os.listdir(path_to_crout / 'outs/per_sample_outs'):
        if (path_to_crout / 'outs/per_sample_outs' / i).is_dir():
            samples_in_per_sample_outs.append(i)

    if len(samples_in_per_sample_outs) == 1:
        if samples_in_per_sample_outs[0] != id:
            raise ValueError(
                f"Expected single sample directory in 'per_sample_outs' at {path_to_crout}, "
                f"but found {samples_in_per_sample_outs[0]}.")
        else:
            multiplexed = False
            multiplexed_samples = None
    elif len(samples_in_per_sample_outs) == 0:
        raise ValueError(
            f"No sample directories found in 'per_sample_outs' at {path_to_crout}."
        )
    else:
        multiplexed = True
        multiplexed_samples = samples_in_per_sample_outs
    
    return multiplexed, multiplexed_samples


class CellrangerOutdir:
    """
    Represents a single Cell Ranger output directory.

    This class parses the directory name to extract metadata and determines the
    run type (e.g., 'multi', multiplexed) and file paths for downstream analysis.

    Parameters
    ----------
    path_to_crout : str or Path
        The path to the Cell Ranger output directory. The directory name is
        expected to follow the format 'exp<number>_S<number>[_tag]'.

    Attributes
    ----------
    path : Path
        The path to the Cell Ranger output directory.
    id : str
        The identifier of the run, derived from the directory name.
    info : dict
        A dictionary containing parsed 'exp', 'sample', and 'tag' from the id.
    sampletag : str
        A combined string of sample and tag (e.g., 'S1_SBC').
    multi_run : bool
        True if the output is from `cellranger multi`.
    multiplexed : bool
        True if the `multi` run was multiplexed.
    multiplexed_samples : list of str or None
        List of sample names if multiplexed, otherwise None.
    paths : dict
        A nested dictionary mapping sample tags to their relevant file paths,
        such as filtered barcodes.
    raw_h5_path : Path
        Path to the raw feature-barcode matrix HDF5 file.
    """

    def __init__(self, path_to_crout: Union[str, Path]) -> None:
        if not isinstance(path_to_crout, Path):
            path_to_crout = Path(path_to_crout)
        
        if not path_to_crout.exists():
            raise FileNotFoundError(f"Path {path_to_crout} does not exist.")

        self.path = path_to_crout
        
        self.id = path_to_crout.stem

        self.info = parse_experiment_string(self.id)
        self.sampletag = self.info['sample'] + '_' + self.info['tag'] if self.info['tag'] else self.info['sample']
        
        path_to_outs = path_to_crout / 'outs'
        
        if not path_to_outs.exists():
            raise ValueError(f"Expected 'outs' directory in {path_to_crout}, but it was not found.")
        
        self.multi_run = detect_cellranger_run(path_to_outs)

        self.paths = {}

        if self.multi_run:
            self.multiplexed, self.multiplexed_samples = is_multi_run_multiplexed(self.id, path_to_crout)
        else:
            self.multiplexed = False
            self.multiplexed_samples = None
            
        
        if self.multi_run:
            raw_h5_path = path_to_crout / 'outs/multi/count/raw_feature_bc_matrix.h5'
            self.raw_h5_path = raw_h5_path

            if not self.multiplexed:
                self.paths[self.sampletag] = {}
                filtered_bcs_path = path_to_crout / f'outs/per_sample_outs/{self.id}/count/sample_filtered_feature_bc_matrix/barcodes.tsv.gz'
                # self.paths[self.sampletag]['raw_h5'] = raw_h5_path
                self.paths[self.sampletag]['filtered_bcs'] = filtered_bcs_path
            else:
                for sample in self.multiplexed_samples:
                    self.paths[sample] = {}
                    filtered_bcs_path = path_to_crout / f'outs/per_sample_outs/{sample}/count/sample_filtered_feature_bc_matrix/barcodes.tsv.gz'
                    # self.paths[sample]['raw_h5'] = raw_h5_path
                    self.paths[sample]['filtered_bcs'] = filtered_bcs_path
        else:
            self.paths[self.sampletag] = {}
            raw_h5_path = path_to_crout / 'outs/raw_feature_bc_matrix.h5'

            self.raw_h5_path = raw_h5_path
            # self.paths[self.sampletag]['raw_h5'] = raw_h5_path

            self.paths[self.sampletag]['filtered_bcs'] = path_to_crout / 'outs/filtered_feature_bc_matrix/barcodes.tsv.gz'

    def __str__(self):
        return f"CellrangerOutdir({self.sampletag})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Experiment:    
    """
    Represents a single-cell experiment composed of one or more samples.

    This class aggregates multiple `CellrangerOutdir` objects, organizing them
    by sample. It provides methods to validate paths and retrieve consolidated
    data like cell barcodes across different runs for a given sample.

    Parameters
    ----------
    path_to_exp : str or Path
        Path to the experiment directory. This directory is expected to contain
        a 'cr' subdirectory, which in turn holds sample directories (e.g., 'S1', 'S2').
    use_samples : list of str, optional
        A list of sample names (e.g., ['S1', 'S3']) to load. If None (default),
        all sample directories found in the 'cr' directory will be loaded.

    Attributes
    ----------
    exp_name : str
        Name of the experiment, derived from the directory name.
    samples : dict
        A dictionary mapping sample names to a list of `CellrangerOutdir` objects
        belonging to that sample.
    """

    def __init__(self, 
                 path_to_exp: Union[str, Path], 
                 use_samples: Optional[List[str]] = None) -> None:

        if not isinstance(path_to_exp, Path):
            path_to_exp = Path(path_to_exp)

        if not path_to_exp.exists():
            raise FileNotFoundError(f"Path {path_to_exp} does not exist.")

        self.exp_name = path_to_exp.stem

        if 'cr' not in os.listdir(path_to_exp):
            raise ValueError(f"Expected 'cr' directory in {path_to_exp}, but it was not found.")

        cr_path = path_to_exp / 'cr'

        if use_samples is None:
            samples = [i for i in os.listdir(cr_path) if re.match(r'^S(\d+)$', i.upper())]
        else:
            samples = [i for i in os.listdir(cr_path) if re.match(r'^S(\d+)$', i.upper()) and i in use_samples]
        

        if not samples:
            raise ValueError(f"No sample directories found in {cr_path}. Expected directories starting with 'S'.")
        
        self.samples = {}
        
        for sample in sorted(samples):
            sample_dir = cr_path / sample
            fnames = os.listdir(sample_dir)
            path_to_crouts_names = [f for f in fnames if parse_experiment_string(f, is_match=True)]
            self.samples[sample] = [CellrangerOutdir(sample_dir / name) for name in path_to_crouts_names]

    def validate_paths(self, error: bool = False) -> Optional[List[Path]]:
        """
        Validates that all file paths stored in the `CellrangerOutdir` objects exist.

        Parameters
        ----------
        error : bool, optional
            If True, raises a `FileNotFoundError` upon finding the first missing
            path. If False (default), prints a message and returns a list of all
            missing paths.

        Returns
        -------
        list or None
            If `error` is False, returns a list of missing `Path` objects. If no
            paths are missing, returns None.

        Raises
        ------
        FileNotFoundError
            If `error` is True and a path does not exist.
        """
        missing = []
        for sample, crouts in self.samples.items():
            for crout in crouts:
                if not crout.path.exists():
                    if error:
                        raise FileNotFoundError(f"Cellranger output directory {crout.path} does not exist.")
                    else:
                        missing.append(crout.path)
                for sampletag in crout.paths:
                    for path in crout.paths[sampletag].values():
                        if not path.exists():
                            if error:
                                raise FileNotFoundError(f"Expected file {path} does not exist in {crout.path}.")
                            else:
                                missing.append(path)
        if missing:
            print("Missing paths!")
            return missing
        else:
            return None
    
    def validate_multiplexed(self, error: bool = True) -> None:
        """
        Validates that all runs for a sample have a consistent multiplexed status.

        Parameters
        ----------
        error : bool, optional
            If True (default), raises a `ValueError` if statuses are inconsistent.
            Currently, only `error=True` is implemented.

        Returns
        -------
        None
            If validation passes.

        Raises
        ------
        ValueError
            If runs for a sample have mixed multiplexed statuses.
        NotImplementedError
            If `error` is set to False.
        """

        if not error:
            raise NotImplementedError("This method is only implemented for error=True.")

        for sample in self.samples:
            if not len(np.unique([crout.multiplexed for crout in self.samples[sample]])) == 1:
                raise ValueError(f"Expected all runs for sample {sample} to have the same multiplexed status, but found mixed statuses.")
        
        return None
        
    def get_bcs(self,
                sample: str,
                use: str = 'union',
                warn_threshold: float = 0.05) -> np.ndarray:
        """
        Gets cell barcodes for a given sample, consolidating from multiple runs.

        Parameters
        ----------
        sample : str
            The sample identifier (e.g., 'S1').
        use : {'union', 'intersection', 'first'}, optional
            Method to consolidate barcodes from multiple runs.
            - 'union' (default): Returns the union of all barcodes.
            - 'intersection': Returns the intersection of all barcodes.
            - 'first': Returns barcodes from the first run only.
        warn_threshold : float, optional
            If the relative difference in the number of barcodes between runs
            exceeds this threshold, a warning is issued. Default is 0.05 (5%).

        Returns
        -------
        numpy.ndarray
            An array of unique cell barcodes.
        """

        bcs = []

        for crout in self.samples[sample]:
            if crout.multi_run:
                if not crout.multiplexed:
                    for sampletag in crout.paths: # should only be one
                        bcs_path = crout.paths[sampletag]['filtered_bcs']
                        bcs.append(pd.read_csv(bcs_path, sep='\t', header=None).values.flatten().tolist())
                else:
                    sub_bcs = []
                    for sample in crout.multiplexed_samples: # multiple, take the union no matter what
                        bcs_path = crout.paths[sampletag]['filtered_bcs']
                        sub_bcs.append(pd.read_csv(bcs_path, sep='\t', header=None).values.flatten().tolist())
                    bcs.append(np.union1d(sub_bcs))
            else:
                bcs_path = crout.paths[crout.sampletag]['filtered_bcs']
                bcs.append(pd.read_csv(bcs_path, sep='\t', header=None).values.flatten().tolist())

        bcsets_lengths = [len(bcset) for bcset in bcs]

        if (max(bcsets_lengths) - min(bcsets_lengths))/ max(bcsets_lengths) > warn_threshold:
            warnings.warn(
                f"Cell barcodes across cr out dirs for sample {sample} differ "
                f"in length by more than {warn_threshold*100}%. "
                f"Max length: {max(bcsets_lengths)}, Min length: {min(bcsets_lengths)}. "
                "This may lead to unexpected results. Consider checking the data. If "
                '"Feature Barcode Only" mode was used, consider re-running cellranger with Gene Expression for '
                "all libraries.")
            
        if use == 'union':
            return_bcs = reduce(np.union1d, bcs)
        elif use == 'intersection':
            return_bcs = reduce(np.intersect1d, bcs)
        elif use == 'first':
            return_bcs = np.array(bcs[0])
        else:
            raise ValueError(f"Unknown use option: {use}. Use 'union', 'intersection', or 'first'.")
        
        return return_bcs

    def get_tags(self, sample: str) -> List[str]:
        """
        Retrieves all library tags associated with a given sample.

        Parameters
        ----------
        sample : str
            The sample identifier (e.g., 'S1').

        Returns
        -------
        list of str
            A list of all tags (e.g., ['SBC', 'GEX']) for the sample.
        """
        
        tags = []
        for crout in self.samples[sample]:
            tags.append(crout.info['tag'])
        
        return tags
    
    def get_crout_tag(self, sample: str, tag: str) -> CellrangerOutdir:
        """
        Finds the specific CellrangerOutdir object for a given sample and tag.

        Parameters
        ----------
        sample : str
            The sample identifier (e.g., 'S1').
        tag : str
            The library tag (e.g., 'SBC').

        Returns
        -------
        CellrangerOutdir
            The matching `CellrangerOutdir` object.

        Raises
        ------
        ValueError
            If the tag is not found for the sample, or if multiple matches are found.
        """
        tags = self.get_tags(sample)

        if tag not in tags:
            raise ValueError(f"Tag {tag} not found in sample {sample}. Available tags: {tags}.")

        crout_match = [crout for crout in self.samples[sample] if crout.info['tag'] == tag]

        if len(crout_match) != 1:
            raise ValueError(f"Expected one crout for tag {tag} in sample {sample}, found {len(crout_match)}.")
        
        return crout_match[0]

    def __str__(self) -> str:
        return f"Experiment({self.exp_name})"

    def __repr__(self) -> str:
        return self.__str__()


def get_ind_adata(adata: sc.AnnData, 
                  obs_names: List[str], 
                  var_names: pd.Index, 
                  batch: Union[int, str]) -> sc.AnnData:
    """
    Subsets an AnnData object and appends a batch identifier to cell barcodes.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to subset.
    obs_names : list of str
        A list of observation names (cell barcodes) to keep.
    var_names : list of str
        A list of variable names (features/genes) to keep.
    batch : int or str
        The batch identifier to append to each cell barcode.

    Returns
    -------
    anndata.AnnData
        A new, subsetted AnnData object with updated observation names.
    """
    adata = adata[obs_names][:, var_names].copy()
    adata.obs_names = adata.obs_names.str.split('-').str[0] + '-' + str(batch)
    return adata


def make_mudata(exp_path: Union[str, Path], 
                exp_data_mapper: Dict, 
                identifiers: Optional[Dict] = None, 
                use_bcs: str = 'union', 
                use_gex: Optional[str] = None,
                verbosity: int = 0) -> md.MuData:
    """
    Create a MuData object from 10x Genomics data for an experiment.

    This function orchestrates the loading of data from multiple Cell Ranger
    runs, organizes it by modality (e.g., 'rna', 'adt'), and concatenates
    the results into a single `MuData` object.

    Parameters
    ----------
    exp_path : str or Path
        Path to the experiment directory, which should be initialized as an
        `Experiment` object.
    exp_data_mapper : dict
        A dictionary defining which samples and libraries to load. It must
        contain 'samples' (a list of sample names) and 'libs_dicts' (a list
        of dictionaries mapping library tags to modalities).
        Example:
        {
            'samples': ['S1', 'S2'],
            'libs_dicts': [
                {'GEX': ['rna'], 'SBC': ['adt']},  # For S1
                {'GEX': ['rna'], 'SBC': ['adt']}   # For S2
            ]
        }
    identifiers : dict, optional
        A dictionary to add observation-level metadata. Keys become column
        names in `.obs`, and values are lists that correspond to the samples
        in `exp_data_mapper`. If 'batch' is not provided, it will be
        auto-generated.
    use_bcs : {'union', 'intersection', 'first'}, optional
        The method for consolidating cell barcodes if a sample has multiple
        runs. Defaults to 'union'. See `Experiment.get_bcs`.
    use_gex : str, optional
        The library tag corresponding to the Gene Expression data to use for
        defining the primary set of cells and genes. If None (default), the
        first library tag from the first sample's `libs_dict` is used.
    verbosity : int, optional
        Scanpy's verbosity level for loading data. Defaults to 0 (silent).

    Returns
    -------
    mudata.MuData
        A `MuData` object containing all specified modalities and samples,
        concatenated and annotated with batch information.

    Raises
    ------
    ValueError
        If samples in `exp_data_mapper` are not found in the `Experiment` object.

    Examples
    --------
    >>> exp_data_mapper = {
    ...     'samples': ['S1'],
    ...     'libs_dicts': [{'GEX': ['rna'], 'SBC': ['adt']}]
    ... }
    >>> identifiers = {'condition': ['treated']}
    >>> mdata = make_mudata(
    ...     exp_path='/path/to/experiment',
    ...     exp_data_mapper=exp_data_mapper,
    ...     identifiers=identifiers
    ... )
    """
    
    exp = Experiment(exp_path)

    if any([i not in exp.samples for i in exp_data_mapper['samples']]):
        raise ValueError(f"Some samples in exp_data_mapper are not present in the experiment: {exp_data_mapper['samples']}. "
                            f"Available samples: {list(exp.samples)}.")

    exp.validate_paths(error=True)
    exp.validate_multiplexed(error=True)

    default_batches = range(len(exp_data_mapper['samples']))

    if identifiers is None:
        identifiers = {'batch': default_batches}
    else:
        if 'batch' not in identifiers:
            identifiers['batch'] = default_batches

    mdatas = []
    for sample, libs_dict, batch in zip(exp_data_mapper['samples'], exp_data_mapper['libs_dicts'], identifiers['batch']):
        bcs = exp.get_bcs(sample, use=use_bcs)

        if use_gex is None:
            use_gex = list(libs_dict.keys())[0]
        
        crout = exp.get_crout_tag(sample, use_gex)    

        adatas_split = {}
        
        # Add the gex_use rna
        with QuietScanpyLoad(verbosity):
            adata = sc.read_10x_h5(crout.raw_h5_path, gex_only=False)
            adata.var_names_make_unique()

        gex_vars = adata.var_names[adata.var['feature_types'] == 'Gene Expression']
        
        adatas_split['rna'] = get_ind_adata(adata, bcs, gex_vars, batch)

        # Add all the tag data

        for tag in libs_dict:
            crout = exp.get_crout_tag(sample, tag)    
            with QuietScanpyLoad(verbosity):
                adata = sc.read_10x_h5(crout.raw_h5_path, gex_only=False)
                adata.var_names_make_unique()
            for mod in libs_dict[tag]:
                tag_var = filter_var(adata.var, libs_dict[tag], mod)
                adatas_split[mod] = get_ind_adata(adata, bcs, tag_var.index, batch)
        mdatas.append(md.MuData(adatas_split))

    for id_col in identifiers:
        for mdata, val in zip(mdatas, identifiers[id_col]):
            mdata.obs[id_col] = val
            mdata.obs[id_col] = mdata.obs[id_col].astype(str).astype('category')
            mdata.push_obs()

    mdata = md.concat(mdatas)
    return mdata


def write_data(data: Union[sc.AnnData, md.MuData],
               filepath: Union[str, Path],
               uns_to_pickle: Optional[Union[List[str], Dict[str, List[str]]]] = None,
               overwrite: bool = False) -> None:
    """
    Writes an AnnData or MuData object, pickling specified .uns keys memory-efficiently.

    This function modifies the data object in-place to remove specified `.uns`
    elements, saves it to an HDF5-based file (h5ad/h5mu), and writes the removed
    elements to a single sidecar pickle file.

    A `try...finally` block ensures that the removed elements are restored to the
    original object, leaving it in its original state in your session, even if
    an error occurs during file writing. This avoids the need for a full memory-
    intensive copy of the data object.

    Parameters
    ----------
    data : AnnData or MuData
        The single-cell data object to save. It will be modified temporarily.
    filepath : str or Path
        The path to write the main data file (e.g., 'my_data.h5ad'). The
        sidecar pickle file will be saved with the same stem (e.g., 'my_data.pkl').
    uns_to_pickle : list[str] or dict[str, list[str]], optional
        The key(s) in `.uns` to pickle.
        - For AnnData: A list of keys in `adata.uns`.
        - For MuData: A dictionary where keys are modality names (or 'base' for
          `mdata.uns`) and values are lists of `.uns` keys for that modality.
          If a simple list is provided for MuData, it will be applied to all
          modalities, including the base `mdata.uns`.
    overwrite : bool, optional (default: False)
        If True, overwrite existing files.
    """
    filepath = Path(filepath)
    pickle_path = filepath.with_suffix(".pkl")
    
    if not isinstance(data, (sc.AnnData, md.MuData)):
        raise TypeError("`data` must be an AnnData or MuData object.")

    if not overwrite and (filepath.exists() or pickle_path.exists()):
        raise FileExistsError(
            f"File(s) already exist: {filepath} / {pickle_path}. Use overwrite=True to replace them."
        )

    if isinstance(data, sc.AnnData):
        if filepath.suffix != ADATA_SUFFIX:
            raise ValueError(f"AnnData file path must end with '{ADATA_SUFFIX}'")
        if uns_to_pickle and not isinstance(uns_to_pickle, list):
            raise TypeError("`uns_to_pickle` must be a list of strings for an AnnData object.")
    
    elif isinstance(data, md.MuData):
        if filepath.suffix != MDATA_SUFFIX:
            raise ValueError(f"MuData file path must end with '{MDATA_SUFFIX}'")
        if uns_to_pickle and not isinstance(uns_to_pickle, (list, dict)):
            raise TypeError("For MuData, `uns_to_pickle` must be a dict or a list.")

    restore_jar = {}
    pickle_jar = {}
    
    try:
        if uns_to_pickle:
            if isinstance(data, sc.AnnData):
                for key in uns_to_pickle:
                    if key in data.uns:
                        pickle_jar[key] = data.uns.pop(key)
                if pickle_jar:
                    restore_jar = pickle_jar
                    data.uns['survey_io'] = {'uns_keys_pickled': list(restore_jar.keys())}

            elif isinstance(data, md.MuData):
                mods_to_process = {}
                if isinstance(uns_to_pickle, list):
                    keys_to_check = uns_to_pickle
                    mods_to_process = {'base': keys_to_check, **{m: keys_to_check for m in data.mod.keys()}}
                elif isinstance(uns_to_pickle, dict):
                    mods_to_process = uns_to_pickle

                for mod, keys in mods_to_process.items():
                    target_obj = data if mod == 'base' else data[mod]
                    mod_uns_data = {}
                    for key in keys:
                        if key in target_obj.uns:
                            mod_uns_data[key] = target_obj.uns.pop(key)
                    if mod_uns_data:
                        pickle_jar[mod] = mod_uns_data
                
                if pickle_jar:
                    restore_jar = pickle_jar
                    data.uns['survey_io'] = {'uns_keys_pickled': uns_to_pickle}

        # Write the main data file
        if isinstance(data, sc.AnnData):
            data.write_h5ad(filepath)
        else: # MuData
            md.write(str(filepath), data)

        # Write the single sidecar pickle file if anything was collected
        if pickle_jar:
            # with open(pickle_path, 'wb') as f:
            #     pkl.dump(pickle_jar, f)
            pklop(pickle_jar, pickle_path)

    finally:
        # This block now only handles restoration, and the return is gone.
        if restore_jar:
            if isinstance(data, sc.AnnData):
                data.uns.update(restore_jar)
            elif isinstance(data, md.MuData):
                for mod, uns_data in restore_jar.items():
                    target_obj = data if mod == 'base' else data.mod[mod]
                    target_obj.uns.update(uns_data)
            
            if 'survey_io' in data.uns:
                del data.uns['survey_io']


def read_data(filepath: Union[str, Path]) -> Union[sc.AnnData, md.MuData]:
    """
    Reads an AnnData or MuData object, re-inserting pickled .uns objects.

    This function reads an HDF5-based file (h5ad/h5mu) and looks for a
    corresponding sidecar pickle file (e.g., 'my_data.pkl') created by
    `write_data`. If found, it loads the pickled objects and re-inserts
    them into the correct `.uns` locations.

    Parameters
    ----------
    filepath : str or Path
        The path to the main data file (e.g., 'my_data.h5ad').

    Returns
    -------
    AnnData or MuData
        The fully reconstructed data object.
    """
    filepath = Path(filepath)
    pickle_path = filepath.with_suffix('.pkl')

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Read the main object first
    if filepath.suffix == ADATA_SUFFIX:
        data = sc.read_h5ad(filepath)
    elif filepath.suffix == MDATA_SUFFIX:
        data = md.read_h5mu(filepath)
    else:
        raise ValueError(f"File suffix must be '{ADATA_SUFFIX}' or '{MDATA_SUFFIX}'")
    
    # Check if there's anything to do
    if 'survey_io' not in data.uns:
        return data

    if not pickle_path.exists():
        warnings.warn(f"`survey_io` metadata found in .uns, but pickle file not found at: {pickle_path}")
        return data

    # Load the pickled data and re-insert it
    pickle_jar = pklop(pickle_path)
    # with open(pickle_path, 'rb') as f:
        # pickle_jar = pkl.load(f)

    # AnnData re-insertion
    if isinstance(data, sc.AnnData):
        data.uns.update(pickle_jar)

    # MuData re-insertion
    elif isinstance(data, md.MuData):
        for mod, uns_data in pickle_jar.items():
            target_obj = data if mod == 'base' else data.mod[mod]
            target_obj.uns.update(uns_data)
            
    # Clean up by removing the metadata key
    del data.uns['survey_io']
    
    return data

