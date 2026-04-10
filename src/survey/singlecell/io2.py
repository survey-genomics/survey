# Built-ins
from pathlib import Path
import re
import os
from functools import reduce
import warnings
from typing import Union, Dict, List, Optional, Tuple, Any

# Standard libs
import numpy as np
import pandas as pd

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.singlecell.scutils import QuietScanpyLoad, filter_var
from survey.genutils import get_config, pklop, is_listlike, get_functional_dependency
from survey.singlecell.iohelp import extract_cb_h5

ADATA_SUFFIX = '.h5ad'
MDATA_SUFFIX = '.h5mu'

# Constants for validation
MTX_COUNTS_DIRS = ['cr']  # Only CR returns an mtx folder
COUNTS_DIR_REGISTRY = {
    'cr': 'CellrangerOutdir',
    'cb': 'CellbenderOutdir',
}

# Expected columns in data_mapper.tsv
DATAMAPPER_COLUMNS = [
    'exp', 'counts_dir', 'samp-id', 'tag', 'mod',
    'id', 'id_sw', 'id_ew', 'id_cn',
    'gi', 'gi_sw', 'gi_ew', 'gi_cn', 
    'ft', 'ft_sw', 'ft_ew', 'ft_cn',
    'use_bcs', 'gex_tag', 'use_mtx', 'add_metrics', 'metrics_dir'
]

# Filter code groups - if the exact match column is not null, the suffix columns must be null
FILTER_CODE_GROUPS = ['id', 'gi', 'ft']


def read_data_mapper(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Read a data_mapper.tsv file into a DataFrame.
    
    This function reads a TSV file and handles the expected data types
    appropriately, converting empty strings to NaN.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the data_mapper.tsv file.
        
    Returns
    -------
    pd.DataFrame
        The data_mapper DataFrame ready for use with Experiments.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"data_mapper file not found: {filepath}")
    
    df = pd.read_csv(filepath, sep='\t', na_values=['', 'NA', 'NaN', 'nan'])
    
    return df


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
    exp = None
    sample = None
    tag = None

    try:
        exp = input_string.split('_')[0]

        tag = input_string[::-1].split('_')[0][::-1]

        if tag in ['rerun', 'reseq']:
            tag = '_'.join(input_string[::-1].split('_')[:2])[::-1]
        if re.match(r'^S[\d_-]+$', tag):
            sample = tag
            tag = None
        
        if sample is None:
            sample = input_string.split('_', maxsplit=1)[1].split(tag)[0].strip('_')

    except IndexError as e:
        if is_match:
            return False
        else:
            # Raise an error if the string doesn't match the required format
            raise ValueError(f"Expected format for input string is 'exp###_S(#ID)[_tag]', but got '{input_string}'.")
    
    if not exp.startswith('exp') or not re.match(r'^S.*$', sample):
        if is_match:
            return False
        else:
            raise ValueError(f"Expected format for input string is 'exp###_S(#ID)[_tag]', but got '{input_string}'.")

    if is_match:
        return True
    else:
        return_dict = {
            'exp': exp,
            'sample': sample,
            'tag': tag
        }
        return return_dict
            
            
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

    Notes
    -----

    This function is designed to detect the cellranger run type based on the files and
    directories that are present only when syncing the "typically necessary" files for
    downstream analysis, usually done with:

    ```
    aws s3 sync s3://${bucket}/path/to/exp/ ./ 
        --exclude "*" 
        --include "*filtered_feature_bc_matrix/barcodes.tsv.gz" 
        --include "*raw_feature_bc_matrix.h5" 
        --include "*web_summary.html" 
        --include "*metrics_summary.csv"
    ```

    Cell Ranger v8 and v10 have different expected files for 'multi' runs. In v8,
    the outs folder only contained either the 'multi' directory or 'per_sample_outs'.
    In v10, 'multi' _may_ be absent (have not tested to confirm this is always the case).
    """

    cellranger_v8_multi_files = ['multi', 'per_sample_outs']
    cellranger_v10_multi_files = ['per_sample_outs', 'filtered_feature_bc_matrix', 'raw_feature_bc_matrix.h5', 'raw_feature_bc_matrix']
    cellranger_count_files = ['raw_feature_bc_matrix.h5', 'filtered_feature_bc_matrix', 'metrics_summary.csv', 'web_summary.html']

    in_outs = os.listdir(path_to_outs)

    all_v8 = all([i in in_outs for i in cellranger_v8_multi_files])
    all_v10 = all([i in in_outs for i in cellranger_v10_multi_files])

    if all_v8:
        multi_run = True
        crv10 = False
    elif all_v10:
        multi_run = True
        crv10 = True
    elif all([i in in_outs for i in cellranger_count_files]): # this must come last to avoid mis-classifying v10 multi as count
        multi_run = False
        crv10 = False
    else:
        raise ValueError(
            f"Could not determine if Cell Ranger run at {path_to_outs} is 'multi' or 'count'. "
            "Expected files for 'multi' or 'count' runs were not found. "
            "Expected files/dirs in /path/to/cr_out/outs/ must be exactly:\n"
            f"  For 'multi' (v8): {cellranger_v8_multi_files}\n"
            f"  For 'multi' (v10): {cellranger_v10_multi_files}\n"
            f"  For 'count': {cellranger_count_files}\n"
        )
    return multi_run, crv10


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
        such as filtered barcodes and metrics summaries.
    raw_h5_path : Path
        Path to the raw feature-barcode matrix HDF5 file.
    """

    def __init__(self, 
                 path_to_crout: Union[str, Path]) -> None:
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
        
        self.multi_run, self.crv10 = detect_cellranger_run(path_to_outs)

        self.paths = {}

        if self.multi_run:
            self.multiplexed, self.multiplexed_samples = is_multi_run_multiplexed(self.id, path_to_crout)
        else:
            self.multiplexed = False
            self.multiplexed_samples = None
            
        
        if self.multi_run:
            if not self.crv10:
                raw_h5_path = path_to_crout / 'outs/multi/count/raw_feature_bc_matrix.h5'
            else:
                raw_h5_path = path_to_crout / 'outs/raw_feature_bc_matrix.h5'
            self.raw_h5_path = raw_h5_path

            if not self.multiplexed:
                self.paths[self.sampletag] = {}
                if not self.crv10:
                    filtered_bcs_path = path_to_crout / f'outs/per_sample_outs/{self.id}/count/sample_filtered_feature_bc_matrix/barcodes.tsv.gz'
                else:
                    filtered_bcs_path = path_to_crout / f'outs/per_sample_outs/{self.id}/sample_filtered_feature_bc_matrix/barcodes.tsv.gz'
                # self.paths[self.sampletag]['raw_h5'] = raw_h5_path
                self.paths[self.sampletag]['filtered_bcs'] = filtered_bcs_path

                metrics_summary_path = path_to_crout / f'outs/per_sample_outs/{self.id}/metrics_summary.csv'
                if not metrics_summary_path.exists():
                    metrics_summary_path = None
                self.paths[self.sampletag]['metrics_summary'] = metrics_summary_path
            else:
                for sample in self.multiplexed_samples:
                    self.paths[sample] = {}
                    if not self.crv10:
                        filtered_bcs_path = path_to_crout / f'outs/per_sample_outs/{sample}/count/sample_filtered_feature_bc_matrix/barcodes.tsv.gz'
                    else:
                        filtered_bcs_path = path_to_crout / f'outs/per_sample_outs/{sample}/sample_filtered_feature_bc_matrix/barcodes.tsv.gz'
                    # self.paths[sample]['raw_h5'] = raw_h5_path
                    self.paths[sample]['filtered_bcs'] = filtered_bcs_path
                    
                    metrics_summary_path = path_to_crout / f'outs/per_sample_outs/{sample}/metrics_summary.csv'
                    if not metrics_summary_path.exists():
                        metrics_summary_path = None
                    self.paths[sample]['metrics_summary'] = metrics_summary_path
        else:
            self.paths[self.sampletag] = {}
            raw_h5_path = path_to_crout / 'outs/raw_feature_bc_matrix.h5'

            self.raw_h5_path = raw_h5_path
            # self.paths[self.sampletag]['raw_h5'] = raw_h5_path

            self.paths[self.sampletag]['filtered_bcs'] = path_to_crout / 'outs/filtered_feature_bc_matrix/barcodes.tsv.gz'

            metrics_summary_path = path_to_crout / 'outs/metrics_summary.csv'
            if not metrics_summary_path.exists():
                metrics_summary_path = None
            self.paths[self.sampletag]['metrics_summary'] = metrics_summary_path

    def __str__(self):
        return f"CellrangerOutdir({self.sampletag})"
    
    def __repr__(self) -> str:
        return self.__str__()


class CellbenderOutdir:
    """
    Represents a single CellBender output directory.

    This class parses the directory name to extract metadata and determines the
    run type (e.g., 'multi', multiplexed) and file paths for downstream analysis.

    Parameters
    ----------
    path_to_cbout : str or Path
        The path to the CellBender output directory. The directory name is
        expected to follow the format 'exp<number>_S<number>[_tag]'.

    Attributes
    ----------
    path : Path
        The path to the CellBender output directory.
    id : str
        The identifier of the run, derived from the directory name.
    info : dict
        A dictionary containing parsed 'exp', 'sample', and 'tag' from the id.
    sampletag : str
        A combined string of sample and tag (e.g., 'S1_SBC').
    filepaths : dict
        A dictionary mapping file types to their respective paths, such as raw, filtered, 
        posterior HDF5 files, PDFs, logs, cell barcodes, metrics, and reports.
    """

    def __init__(self, 
                 path_to_cbout: Union[str, Path]) -> None:
        
        if not isinstance(path_to_cbout, Path):
            path_to_cbout = Path(path_to_cbout)
        
        if not path_to_cbout.exists():
            raise FileNotFoundError(f"Path {path_to_cbout} does not exist.")

        self.path = path_to_cbout
        
        self.id = path_to_cbout.stem

        self.info = parse_experiment_string(self.id)
        self.sampletag = self.info['sample'] + '_' + self.info['tag'] if self.info['tag'] else self.info['sample']

        self.rootname = None

        h5s = list(path_to_cbout.glob('**/*.h5'))
        if len(h5s) == 0:
            raise ValueError(f"No .h5 files found in {path_to_cbout}. Expected raw, filtered, and/or posterior h5s.")
        for h5file in h5s:
            if h5file.stem.endswith('filtered') or h5file.stem.endswith('posterior'):
                pass
            else:
                self.rootname = h5file.stem
        if self.rootname is None:
            raise ValueError(f"No raw .h5 file found in {path_to_cbout}. Please make sure it exists.")
        
        self.filepaths = {}

        raw_path = self.path / (self.rootname + '.h5')
        self.filepaths['raw'] = raw_path if raw_path.exists() else None
        filt_path = self.path / (self.rootname + '_filtered.h5')
        self.filepaths['filtered'] = filt_path if filt_path.exists() else None
        posterior_path = self.path / (self.rootname + '_posterior.h5')
        self.filepaths['posterior'] = posterior_path if posterior_path.exists() else None
        pdf_path = self.path / (self.rootname + '.pdf')
        self.filepaths['pdf'] = pdf_path if pdf_path.exists() else None
        log_path = self.path / (self.rootname + '.log')
        self.filepaths['log'] = log_path if log_path.exists() else None
        cbcs_path = self.path / (self.rootname + '_cell_barcodes.csv')
        self.filepaths['cbcs'] = cbcs_path if cbcs_path.exists() else None
        metrics_path = self.path / (self.rootname + '_metrics.csv')
        self.filepaths['metrics'] = metrics_path if metrics_path.exists() else None
        report_path = self.path / (self.rootname + '_report.html')
        self.filepaths['report'] = report_path if report_path.exists() else None


    def __str__(self):
        return f"CellbenderOutdir({self.sampletag})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Tag:
    """
    Represents a single tag (modality/library type) within a sample.
    
    A Tag can have multiple output directories from different counts_dir types
    (e.g., both cellranger 'cr' and cellbender 'cb' outputs for the same tag).
    
    Parameters
    ----------
    exp_name : str
        The experiment name (e.g., 'exp371').
    sample_id : str
        The sample identifier (e.g., 'S1').
    tag_name : str
        The tag name (e.g., 'SBC', 'CSP').
    tag_rows : pd.DataFrame
        Subset of the data_mapper DataFrame for this tag.
    path_to_exp : Path
        Path to the experiment directory.
        
    Attributes
    ----------
    name : str
        The tag name.
    mod : str
        The modality name (e.g., 'sbc', 'csp').
    filter_criteria : dict
        Dictionary of filter criteria for filter_var().
    outdirs : dict
        Dictionary mapping counts_dir type to output directory objects.
    """
    
    def __init__(self,
                 exp_name: str,
                 sample_id: str,
                 tag_name: str,
                 tag_rows: pd.DataFrame,
                 path_to_exp: Path) -> None:
        
        self.name = tag_name
        self.exp_name = exp_name
        self.sample_id = sample_id
        
        # Get unique modality (should be one per tag based on functional dependency)
        mods = tag_rows['mod'].unique()
        if len(mods) != 1:
            raise ValueError(f"Expected exactly one modality for tag '{tag_name}', got {mods}")
        self.mod = mods[0]
        
        # Build filter criteria from first row (should be same for all rows of this tag)
        first_row = tag_rows.iloc[0]
        self.filter_criteria = self._build_filter_criteria(first_row)
        
        # Store additional metadata
        self.use_bcs = first_row.get('use_bcs')
        self.use_mtx = first_row.get('use_mtx', False)
        if pd.isna(self.use_mtx):
            self.use_mtx = False
        
        # Store which counts_dir to use (from data_mapper)
        self.counts_dir = first_row.get('counts_dir', 'cr')
        
        # Build output directories - check ALL possible counts_dirs, not just those in data_mapper
        # The data_mapper tells us which to USE, but we store all that exist
        self.outdirs = {}
        outdir_name = f"{exp_name}_{sample_id}_{tag_name}"
        
        for counts_dir_type, outdir_class in COUNTS_DIR_REGISTRY.items():
            outdir_path = path_to_exp / counts_dir_type / sample_id / outdir_name
            
            if outdir_path.exists():
                if counts_dir_type == 'cr':
                    self.outdirs['cr'] = CellrangerOutdir(outdir_path)
                elif counts_dir_type == 'cb':
                    self.outdirs['cb'] = CellbenderOutdir(outdir_path)
    
    def _build_filter_criteria(self, row: pd.Series) -> Dict[str, str]:
        """Build filter criteria dictionary from a data_mapper row."""
        criteria = {}
        
        for code in FILTER_CODE_GROUPS:
            # Check exact match first
            if pd.notna(row.get(code)):
                criteria[code] = row[code]
            else:
                # Check suffix columns
                for suffix in ['sw', 'ew', 'cn']:
                    col = f"{code}_{suffix}"
                    if pd.notna(row.get(col)):
                        criteria[f"{code}_{suffix}"] = row[col]
        
        return criteria
    
    def get_outdir(self, counts_dir: Optional[str] = None) -> Union['CellrangerOutdir', 'CellbenderOutdir', None]:
        """Get the output directory for a specific counts_dir type.
        
        If counts_dir is None, uses self.counts_dir (from data_mapper).
        """
        if counts_dir is None:
            counts_dir = self.counts_dir
        return self.outdirs.get(counts_dir)
    
    def get_raw_h5_path(self, counts_dir: Optional[str] = None) -> Optional[Path]:
        """Get the path to the raw H5 file.
        
        If counts_dir is None, uses self.counts_dir (from data_mapper).
        """
        if counts_dir is None:
            counts_dir = self.counts_dir
        outdir = self.get_outdir(counts_dir)
        if outdir is None:
            return None
        if counts_dir == 'cr':
            return outdir.raw_h5_path
        elif counts_dir == 'cb':
            # return outdir.filepaths.get('filtered') or outdir.filepaths.get('raw')
            return outdir.filepaths.get('raw')
        return None
    
    def get_filtered_bcs_path(self, counts_dir: Optional[str] = None) -> Optional[Path]:
        """Get the path to filtered barcodes.
        
        If counts_dir is None, uses self.counts_dir (from data_mapper).
        """
        if counts_dir is None:
            counts_dir = self.counts_dir
        outdir = self.get_outdir(counts_dir)
        if outdir is None:
            return None
        if counts_dir == 'cr':
            return outdir.paths.get(outdir.sampletag, {}).get('filtered_bcs')
        elif counts_dir == 'cb':
            return outdir.filepaths.get('cbcs')
        return None

    def __str__(self) -> str:
        return f"Tag({self.name}, use={self.counts_dir}, available={list(self.outdirs.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Sample:
    """
    Represents a single sample within an experiment.
    
    A Sample contains one or more Tags, each representing a different
    sequencing modality/library type.
    
    Parameters
    ----------
    exp_name : str
        The experiment name (e.g., 'exp371').
    sample_id : str
        The sample identifier (e.g., 'S1').
    sample_rows : pd.DataFrame
        Subset of the data_mapper DataFrame for this sample.
    path_to_exp : Path
        Path to the experiment directory.
        
    Attributes
    ----------
    id : str
        The sample identifier.
    tags : dict
        Dictionary mapping tag names to Tag objects.
    gex_tag : str or None
        The tag that contains gene expression data.
    """
    
    def __init__(self,
                 exp_name: str,
                 sample_id: str,
                 sample_rows: pd.DataFrame,
                 path_to_exp: Path) -> None:
        
        self.id = sample_id
        self.exp_name = exp_name
        self.path_to_exp = path_to_exp
        
        # Get gex_tag (should be unique or None for this sample)
        gex_tags = sample_rows['gex_tag'].dropna().unique()
        if len(gex_tags) > 1:
            raise ValueError(f"Multiple gex_tags found for {exp_name}/{sample_id}: {gex_tags}")
        self.gex_tag = gex_tags[0] if len(gex_tags) == 1 else None
        
        # Store metadata about metrics collection
        metrics_rows = sample_rows[sample_rows['add_metrics'] == True]
        self.add_metrics = len(metrics_rows) > 0
        self.metrics_dir = metrics_rows['metrics_dir'].iloc[0] if self.add_metrics else None
        
        # Build Tag objects
        self.tags = {}
        for tag_name in sample_rows['tag'].unique():
            tag_rows = sample_rows[sample_rows['tag'] == tag_name]
            self.tags[tag_name] = Tag(exp_name, sample_id, tag_name, tag_rows, path_to_exp)
    
    def get_tag(self, tag_name: str) -> Optional[Tag]:
        """Get a Tag by name."""
        return self.tags.get(tag_name)
    
    def get_gex_tag_obj(self) -> Optional[Tag]:
        """Get the Tag object that contains gene expression data."""
        if self.gex_tag is None:
            return None
        return self.tags.get(self.gex_tag)
    
    def get_bcs(self, use: str = 'union', warn_threshold: float = 0.05) -> np.ndarray:
        """
        Get consolidated cell barcodes across all tags.
        
        Parameters
        ----------
        use : str
            How to consolidate barcodes:
            - 'union': Union of all barcodes across tags
            - 'intersection': Intersection of all barcodes
            - A tag name: Use barcodes from that specific tag
            
        Returns
        -------
        np.ndarray
            Array of cell barcodes.
        """
        # Collect barcodes from each tag
        bc_sets = []
        
        for tag_name, tag in self.tags.items():
            # Use the counts_dir specified in data_mapper for this tag
            bc_path = tag.get_filtered_bcs_path()  # Uses tag.counts_dir by default
            
            if bc_path is not None and bc_path.exists():
                bcs = pd.read_csv(bc_path, header=None)[0].values
                bc_sets.append((tag_name, set(bcs)))
        
        if len(bc_sets) == 0:
            raise ValueError(f"No barcodes found for sample {self.id}")
        
        if use == 'union':
            result = set().union(*[s for _, s in bc_sets])
        elif use == 'intersection':
            result = set.intersection(*[s for _, s in bc_sets])
        elif use in self.tags:
            # Use barcodes from specific tag
            for tag_name, bc_set in bc_sets:
                if tag_name == use:
                    result = bc_set
                    break
            else:
                raise ValueError(f"Tag '{use}' not found for sample {self.id}")
        else:
            raise ValueError(f"Unknown use method: {use}. Expected 'union', 'intersection', or a tag name.")
        
        return np.array(list(result))
    
    def get_metrics(self) -> Optional[pd.DataFrame]:
        """Get metrics summary from the metrics_dir."""
        if not self.add_metrics or self.metrics_dir is None:
            return None
        
        metrics_list = []
        for tag_name, tag in self.tags.items():
            outdir = tag.get_outdir(self.metrics_dir)
            if outdir is None:
                continue
            
            if self.metrics_dir == 'cr':
                metrics_path = outdir.paths.get(outdir.sampletag, {}).get('metrics_summary')
                if metrics_path and metrics_path.exists():
                    metrics = pd.read_csv(metrics_path, thousands=',')
                    metrics['exp'] = self.exp_name
                    metrics['sample'] = self.id
                    metrics['tag'] = tag_name
                    metrics_list.append(metrics)
        
        if len(metrics_list) == 0:
            return None
        
        return pd.concat(metrics_list, ignore_index=True)
    
    def __str__(self) -> str:
        return f"Sample({self.id}, tags={list(self.tags.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Experiment:
    """
    Represents a single experiment containing one or more samples.
    
    Parameters
    ----------
    exp_name : str
        The experiment name (e.g., 'exp371').
    exp_rows : pd.DataFrame
        Subset of the data_mapper DataFrame for this experiment.
    path_to_exp : Path
        Path to the experiment directory.
        
    Attributes
    ----------
    name : str
        The experiment name.
    samples : dict
        Dictionary mapping sample IDs to Sample objects.
    """
    
    def __init__(self,
                 exp_name: str,
                 exp_rows: pd.DataFrame,
                 path_to_exp: Path) -> None:
        
        self.name = exp_name
        self.path = path_to_exp
        
        if not path_to_exp.exists():
            raise FileNotFoundError(f"Experiment directory not found: {path_to_exp}")
        
        # Build Sample objects
        self.samples = {}
        for sample_id in exp_rows['samp-id'].unique():
            sample_rows = exp_rows[exp_rows['samp-id'] == sample_id]
            self.samples[sample_id] = Sample(exp_name, sample_id, sample_rows, path_to_exp)
    
    def get_sample(self, sample_id: str) -> Optional[Sample]:
        """Get a Sample by ID."""
        return self.samples.get(sample_id)
    
    def get_all_metrics(self) -> Optional[pd.DataFrame]:
        """Get combined metrics from all samples."""
        metrics_list = []
        for sample in self.samples.values():
            metrics = sample.get_metrics()
            if metrics is not None:
                metrics_list.append(metrics)
        
        if len(metrics_list) == 0:
            return None
        
        return pd.concat(metrics_list, ignore_index=True)
    
    def validate_paths(self, error: bool = False) -> List[Path]:
        """Validate that all required paths exist."""
        missing = []
        for sample in self.samples.values():
            for tag in sample.tags.values():
                for counts_dir, outdir in tag.outdirs.items():
                    if not outdir.path.exists():
                        missing.append(outdir.path)
        
        if error and missing:
            raise FileNotFoundError(f"Missing paths: {missing}")
        
        return missing
    
    def __str__(self) -> str:
        return f"Experiment({self.name}, samples={list(self.samples.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Experiments:
    """
    Top-level class representing multiple experiments.
    
    This class validates the data_mapper DataFrame, integrates metadata,
    and provides access to all experiments, samples, and tags.
    
    Parameters
    ----------
    path_to_datadir : str or Path
        Path to the directory containing experiment subdirectories.
    data_mapper : pd.DataFrame
        DataFrame from data_mapper.tsv defining the data structure.
    meta : pd.DataFrame, optional
        Metadata DataFrame with sample information.
    add_cols : list of str, optional
        Columns from meta to add to .obs of the RNA modality.
        
    Attributes
    ----------
    exps : dict
        Dictionary mapping experiment names to Experiment objects.
    data_mapper : pd.DataFrame
        The validated data_mapper DataFrame.
    meta : pd.DataFrame or None
        The metadata DataFrame.
    add_cols : list of str or None
        Columns to add from metadata.
    """
    
    def __init__(self,
                 path_to_datadir: Union[str, Path],
                 data_mapper: pd.DataFrame,
                 meta: Optional[pd.DataFrame] = None,
                 add_cols: Optional[List[str]] = None) -> None:
        
        self.path_to_datadir = Path(path_to_datadir)
        
        if not self.path_to_datadir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.path_to_datadir}")
        
        # Validate and store data_mapper
        self.data_mapper = self._validate_data_mapper(data_mapper)
        
        # Validate and store metadata
        self.meta = meta
        self.add_cols = add_cols
        if meta is not None:
            self._validate_meta(meta, add_cols)
        
        # Build Experiment objects
        self.exps = {}
        for exp_name in self.data_mapper['exp'].unique():
            exp_rows = self.data_mapper[self.data_mapper['exp'] == exp_name]
            path_to_exp = self.path_to_datadir / exp_name
            self.exps[exp_name] = Experiment(exp_name, exp_rows, path_to_exp)
    
    def _validate_data_mapper(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the data_mapper DataFrame structure and contents."""
        
        # 1. Validate headers
        missing_cols = set(DATAMAPPER_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data_mapper: {missing_cols}")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert categorical columns
        cat_cols = ['exp', 'counts_dir', 'samp-id', 'tag', 'mod', 'gex_tag', 'use_bcs', 'metrics_dir']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Convert boolean columns
        bool_cols = ['use_mtx', 'add_metrics']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        
        # 2. Validate gex_tag uniqueness per (exp, samp-id)
        for (exp, sample), group in df.groupby(['exp', 'samp-id'], observed=True):
            gex_tags = group['gex_tag'].dropna().unique()
            if len(gex_tags) > 1:
                raise ValueError(
                    f"Multiple gex_tags found for ({exp}, {sample}): {gex_tags}. "
                    "Expected at most one gex_tag per sample."
                )
        
        # 3. Validate use_mtx vs counts_dir compatibility
        mtx_rows = df[df['use_mtx'] == True]
        invalid_mtx = mtx_rows[~mtx_rows['counts_dir'].isin(MTX_COUNTS_DIRS)]
        if len(invalid_mtx) > 0:
            invalid_dirs = invalid_mtx['counts_dir'].unique().tolist()
            raise ValueError(
                f"use_mtx=True requires counts_dir in {MTX_COUNTS_DIRS}, "
                f"but found: {invalid_dirs}"
            )
        
        # 4. Validate filter code groups (if exact match is set, suffixes must be null)
        for code in FILTER_CODE_GROUPS:
            suffix_cols = [f"{code}_{s}" for s in ['sw', 'ew', 'cn']]
            exact_set = df[code].notna()
            
            for suffix_col in suffix_cols:
                if suffix_col in df.columns:
                    conflict = exact_set & df[suffix_col].notna()
                    if conflict.any():
                        conflicting_rows = df[conflict].index.tolist()
                        raise ValueError(
                            f"Row(s) {conflicting_rows}: If '{code}' is set, "
                            f"'{suffix_col}' must be empty."
                        )
        
        # 5. Warn if no functional dependency between tag and mod
        try:
            get_functional_dependency(df, ('tag', 'mod'))
        except ValueError as e:
            warnings.warn(
                f"No functional dependency from 'tag' to 'mod': {e}. "
                "This may cause unexpected behavior."
            )
        
        # 6. Validate use_bcs values
        valid_use_bcs = {'union', 'intersection'}
        all_tags = set(df['tag'].dropna().unique())
        valid_use_bcs = valid_use_bcs | all_tags
        
        invalid_use_bcs = df['use_bcs'].dropna().unique()
        invalid_use_bcs = [v for v in invalid_use_bcs if v not in valid_use_bcs]
        if invalid_use_bcs:
            raise ValueError(
                f"Invalid use_bcs values: {invalid_use_bcs}. "
                f"Must be 'union', 'intersection', or a tag name from: {all_tags}"
            )
        
        return df
    
    def _validate_meta(self, meta: pd.DataFrame, add_cols: Optional[List[str]]) -> None:
        """Validate the metadata DataFrame."""
        
        # Check if 'exp' exists (might be in index)
        if meta.index.name == 'exp':
            meta_exp_col = meta.index
        elif 'exp' in meta.columns:
            meta_exp_col = meta['exp']
        else:
            raise ValueError("Metadata must have 'exp' as a column or index.")
        
        # Check if 'samp-id' column exists
        if 'samp-id' not in meta.columns:
            raise ValueError("Metadata must have 'samp-id' column.")
        
        # Check that all (exp, samp-id) combinations in data_mapper exist in meta
        dm_pairs = set(zip(self.data_mapper['exp'], self.data_mapper['samp-id']))
        
        if meta.index.name == 'exp':
            meta_pairs = set(zip(meta.index, meta['samp-id']))
        else:
            meta_pairs = set(zip(meta['exp'], meta['samp-id']))
        
        missing_pairs = dm_pairs - meta_pairs
        if missing_pairs:
            raise ValueError(
                f"The following (exp, samp-id) pairs from data_mapper are missing in meta: {missing_pairs}"
            )
        
        # Check that add_cols exist in meta
        if add_cols is not None:
            missing_cols = set(add_cols) - set(meta.columns)
            if missing_cols:
                raise ValueError(f"add_cols {missing_cols} not found in metadata columns.")
    
    def get_experiment(self, exp_name: str) -> Optional[Experiment]:
        """Get an Experiment by name."""
        return self.exps.get(exp_name)
    
    def get_sample(self, exp_name: str, sample_id: str) -> Optional[Sample]:
        """Get a Sample by experiment name and sample ID."""
        exp = self.get_experiment(exp_name)
        if exp is None:
            return None
        return exp.get_sample(sample_id)
    
    def get_all_metrics(self) -> Optional[pd.DataFrame]:
        """Get combined metrics from all experiments."""
        metrics_list = []
        for exp in self.exps.values():
            metrics = exp.get_all_metrics()
            if metrics is not None:
                metrics_list.append(metrics)
        
        if len(metrics_list) == 0:
            return None
        
        return pd.concat(metrics_list, ignore_index=True)
    
    def get_meta_for_sample(self, exp_name: str, sample_id: str) -> Optional[pd.Series]:
        """Get metadata for a specific sample."""
        if self.meta is None:
            return None
        
        if self.meta.index.name == 'exp':
            mask = (self.meta.index == exp_name) & (self.meta['samp-id'] == sample_id)
        else:
            mask = (self.meta['exp'] == exp_name) & (self.meta['samp-id'] == sample_id)
        
        result = self.meta[mask]
        if len(result) == 0:
            return None
        return result.iloc[0]
    
    def __str__(self) -> str:
        return f"Experiments(n={len(self.exps)}, exps={list(self.exps.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __iter__(self):
        """Iterate over experiments."""
        return iter(self.exps.values())
    
    def __len__(self) -> int:
        return len(self.exps)


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


def make_mudata(experiments: Experiments,
                verbosity: int = 0) -> md.MuData:
    """
    Create a MuData object from an Experiments object.
    
    This function orchestrates the loading of data from multiple experiments,
    samples, and tags, creating a concatenated MuData object with all
    specified modalities. It also integrates metrics and metadata.
    
    All loading parameters (which counts_dir to use, how to consolidate
    barcodes, etc.) are read from the data_mapper DataFrame that was used
    to create the Experiments object.
    
    Parameters
    ----------
    experiments : Experiments
        An Experiments object containing all experiment, sample, and tag
        information along with the data_mapper configuration.
    verbosity : int, default 0
        Internal verbosity level for progress output:
        - 0: Silent
        - 1: Print experiment/sample progress
        - 2: Print detailed loading info (tags, files)
        
    Returns
    -------
    mudata.MuData
        A concatenated MuData object containing all modalities across all
        samples and experiments. Includes:
        - Metrics in mdata.uns['metrics'] (if available)
        - Metadata columns (add_cols) in mdata['rna'].obs (if specified)
        
    Examples
    --------
    >>> import pandas as pd
    >>> data_mapper = pd.read_csv('data-mapper.tsv', sep='\\t')
    >>> meta = pd.read_csv('metadata.csv', index_col='exp')
    >>> exps = Experiments('/path/to/data', data_mapper, meta, add_cols=['rxn'])
    >>> mdata = make_mudata(exps)
    """
    
    def _vprint(msg: str, level: int = 1):
        """Print message if verbosity >= level."""
        if verbosity >= level:
            print(msg)
    
    mdatas = []
    all_metrics = []
    batch_counter = 0
    n_exps = len(experiments)
    
    # Track metadata to add to obs
    obs_metadata = []
    
    for exp_idx, exp in enumerate(experiments):
        _vprint(f"Processing {exp.name} ({exp_idx + 1}/{n_exps})", level=1)
        
        for sample_id, sample in exp.samples.items():
            _vprint(f"  Sample {sample_id}", level=1)
            
            # Get barcodes using the use_bcs specified in data_mapper
            bcs = sample.get_bcs(use=sample.tags[list(sample.tags.keys())[0]].use_bcs or 'union')
            
            # Collect metrics
            if sample.add_metrics:
                metrics = sample.get_metrics()
                if metrics is not None:
                    all_metrics.append(metrics)
            
            # Find the GEX tag for this sample
            gex_tag_name = sample.gex_tag
            if gex_tag_name is None:
                # Use first tag as GEX source
                gex_tag_name = list(sample.tags.keys())[0]
            
            gex_tag_obj = sample.tags[gex_tag_name]
            
            # Load the GEX tag's H5 once - it contains both GEX and the tag's modality data
            # counts_dir is determined by data_mapper (stored in tag.counts_dir)
            use_mtx = gex_tag_obj.use_mtx and gex_tag_obj.counts_dir in MTX_COUNTS_DIRS
            
            with QuietScanpyLoad(0):  # Always quiet scanpy
                h5_path = gex_tag_obj.get_raw_h5_path()  # Uses tag.counts_dir by default
                if h5_path is None:
                    raise FileNotFoundError(f"No H5 file found for {exp.name}/{sample_id}/{gex_tag_name}")
                
                _vprint(f"    Loading H5 from {gex_tag_name} ({gex_tag_obj.counts_dir}): {h5_path.name}", level=2)
                
                if use_mtx:
                    mtx_path = h5_path.with_suffix('')
                    gex_adata = sc.read_10x_mtx(mtx_path, gex_only=False)
                else:
                    gex_adata = sc.read_10x_h5(h5_path, gex_only=False)
                
                gex_adata.var_names_make_unique()
            
            # If loading from CellBender, extract CB-specific metadata (cell_probability, etc.)
            cb_metadata = None
            if gex_tag_obj.counts_dir == 'cb':
                try:
                    cb_metadata = extract_cb_h5(str(h5_path), extract_only='droplet_latents')
                    if len(list(cb_metadata.columns)) <= 3:
                        vprint_text = str(list(cb_metadata.columns))
                    else:
                        vprint_text = str(list(cb_metadata.columns[:3])) + f' and {len(cb_metadata.columns) - 3} more...'
                    _vprint(f"      Extracted CellBender metadata: {vprint_text}", level=2)
                except Exception as e:
                    warnings.warn(f"Failed to extract CellBender metadata from {h5_path}: {e}")
            
            # Build individual adatas per modality
            adatas_split = {}
            
            # Extract GEX (Gene Expression) -> 'rna' modality
            gex_vars = gex_adata.var_names[gex_adata.var['feature_types'] == 'Gene Expression']
            adatas_split['rna'] = get_ind_adata(gex_adata, bcs, gex_vars, batch_counter)
            _vprint(f"      Extracted rna: {len(gex_vars)} genes", level=2)
            
            # Add CellBender metadata to rna.obs if available
            if cb_metadata is not None and len(cb_metadata) > 0:
                # Match barcodes (CB metadata uses original barcodes, rna uses -batch suffix)
                # Extract barcode prefix from rna obs_names
                rna_bc_prefix = adatas_split['rna'].obs_names.str.rsplit('-', n=1).str[0]
                # CB metadata barcodes also have -1 suffix, strip it
                cb_bc_prefix = cb_metadata.index.str.rsplit('-', n=1).str[0]
                cb_metadata_reindexed = cb_metadata.copy()
                cb_metadata_reindexed.index = cb_bc_prefix
                
                # Flatten MultiIndex columns if present (from pd.concat with keys)
                if isinstance(cb_metadata_reindexed.columns, pd.MultiIndex):
                    cb_metadata_reindexed.columns = ['_'.join(map(str, col)).strip('_') 
                                                      for col in cb_metadata_reindexed.columns]
                
                # Join on matching prefixes
                for col in cb_metadata_reindexed.columns:
                    col_data = cb_metadata_reindexed[col]
                    # Prefix CB columns to avoid conflicts
                    obs_col_name = f"cb_{col}" if not col.startswith('cb_') else col
                    adatas_split['rna'].obs[obs_col_name] = rna_bc_prefix.map(
                        pd.Series(col_data.values, index=col_data.index).to_dict()
                    ).values
            
            # Also extract the gex_tag's own modality (e.g., 'sbc') from the same H5
            gex_tag_mod = gex_tag_obj.mod
            if gex_tag_mod != 'rna':  # Only if it's a different modality
                lib_tag_dict = {gex_tag_mod: gex_tag_obj.filter_criteria}
                gex_tag_vars = filter_var(gex_adata.var, lib_tag_dict, gex_tag_mod)
                
                if len(gex_tag_vars) > 0:
                    adatas_split[gex_tag_mod] = get_ind_adata(gex_adata, bcs, gex_tag_vars.index, batch_counter)
                    _vprint(f"      Extracted {gex_tag_mod}: {len(gex_tag_vars)} variables", level=2)
            
            # Process remaining tags (not the gex_tag) for their modalities
            for tag_name, tag in sample.tags.items():
                # Skip the gex_tag - we already processed it above
                if tag_name == gex_tag_name:
                    continue
                
                mod = tag.mod
                
                # Skip if modality already processed
                if mod in adatas_split:
                    continue
                
                # counts_dir is determined by data_mapper (stored in tag.counts_dir)
                tag_use_mtx = tag.use_mtx and tag.counts_dir in MTX_COUNTS_DIRS
                
                with QuietScanpyLoad(0):  # Always quiet scanpy
                    h5_path = tag.get_raw_h5_path()  # Uses tag.counts_dir by default
                    if h5_path is None:
                        warnings.warn(f"No H5 file found for {exp.name}/{sample_id}/{tag_name}, skipping")
                        continue
                    
                    _vprint(f"    Loading H5 from {tag_name} ({tag.counts_dir}): {h5_path.name}", level=2)
                    
                    if tag_use_mtx:
                        mtx_path = h5_path.with_suffix('')
                        tag_adata = sc.read_10x_mtx(mtx_path, gex_only=False)
                    else:
                        tag_adata = sc.read_10x_h5(h5_path, gex_only=False)
                    
                    tag_adata.var_names_make_unique()
                
                # Filter variables based on criteria (extract only the tag's modality, not GEX)
                lib_tag_dict = {mod: tag.filter_criteria}
                tag_vars = filter_var(tag_adata.var, lib_tag_dict, mod)
                
                if len(tag_vars) > 0:
                    adatas_split[mod] = get_ind_adata(tag_adata, bcs, tag_vars.index, batch_counter)
                    _vprint(f"      Extracted {mod}: {len(tag_vars)} variables", level=2)
            
            # Create MuData for this sample
            sample_mdata = md.MuData(adatas_split)
            
            _vprint(f"    Created MuData with modalities: {list(adatas_split.keys())}", level=2)
            
            # Store identifiers (no batch - user adds manually)
            sample_mdata.obs['exp'] = exp.name
            sample_mdata.obs['sample'] = sample_id
            
            # Convert to categorical
            for col in ['exp', 'sample']:
                sample_mdata.obs[col] = sample_mdata.obs[col].astype(str).astype('category')
            
            sample_mdata.push_obs()
            
            # Store metadata for later addition to RNA modality
            if experiments.meta is not None and experiments.add_cols:
                meta_row = experiments.get_meta_for_sample(exp.name, sample_id)
                if meta_row is not None:
                    for col in experiments.add_cols:
                        val = meta_row.get(col)
                        obs_metadata.append({
                            'batch': batch_counter,
                            'col': col,
                            'value': val
                        })
            
            mdatas.append(sample_mdata)
            batch_counter += 1
    
    # Concatenate all MuDatas
    if len(mdatas) == 1:
        mdata = mdatas[0]
    else:
        mdata = concat_mdatas(mdatas)
    
    # Add metadata columns to RNA modality
    if experiments.meta is not None and experiments.add_cols and 'rna' in mdata.mod:
        for col in experiments.add_cols:
            # Create column with NaN
            mdata['rna'].obs[col] = pd.NA
            
            # Fill in values by batch
            for meta_item in obs_metadata:
                if meta_item['col'] == col:
                    batch_mask = mdata['rna'].obs_names.str.endswith(f"-{meta_item['batch']}")
                    mdata['rna'].obs.loc[batch_mask, col] = meta_item['value']
            
            # Try to convert to categorical
            try:
                mdata['rna'].obs[col] = mdata['rna'].obs[col].astype('category')
            except (TypeError, ValueError):
                pass
    
    # Add metrics to uns
    if len(all_metrics) > 0:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        # Set multi-index on exp, sample, tag
        if all(col in combined_metrics.columns for col in ['exp', 'sample', 'tag']):
            combined_metrics = combined_metrics.set_index(['exp', 'sample', 'tag'])
        mdata.uns['metrics'] = combined_metrics
    
    _vprint(f"Created MuData with {mdata.n_obs} cells, modalities: {list(mdata.mod.keys())}", level=1)
    
    return mdata


def concat_mdatas(mdatas: List[md.MuData], 
                  concat_kwargs: Dict = None,
                  store_vars: bool = False) -> md.MuData:
    """
    Concatenate multiple mudata objects along the `obs` axis, preserving the 
    *union* of all modalities, as opposed to the default behavior of 
    MuData.concat(). This is done by concatenating each modality's `adata` 
    separately using scanpy.concat() and building a new MuData object.

    Parameters
    ----------
    mdatas : list of mudata.MuData
        List of MuData objects to concatenate.
    concat_kwargs : dict, optional
        Additional keyword arguments to pass to `scanpy.concat()`. If any keys
        overlap with the detected modalities, concat_kwargs will be understood as a
        dict of dicts with modality-specific kwargs.
    store_vars : bool, optional
        If True and `join=outer` is provided via `concat_kwargs`, stores the specific
        variables present in each adata (or only the adata for which there are concat_kwargs) 
        in `adata.uns['vars_dict']`. If a list-like is provided, it should be the same 
        length as `mdatas` and will be used as the keys in the `vars_dict`. Default is False.

    Returns
    -------
    mudata.MuData
        A single MuData object with concatenated modalities.
    """

    # Get the union of all mods in the mdata objects
    all_mods = set().union(*[mdata.mod.keys() for mdata in mdatas])
    concatenated = {}

    if concat_kwargs is None:
        concat_kwargs = {mod: {} for mod in all_mods}
    elif any([k in concat_kwargs for k in all_mods]):
        # Assume concat_kwargs is a dict of dicts with modality-specific kwargs
        for mod in all_mods:
            if mod not in concat_kwargs:
                concat_kwargs[mod] = {}
    else:
        # Use the same kwargs for all modalities
        concat_kwargs = {mod: concat_kwargs for mod in all_mods}
        
    for mod in all_mods:
        # Get all adata objects for this modality from all mdatas where this modality exists
        adatas_for_mod = [mdata[mod] for mdata in mdatas if mod in mdata.mod]

        # Make sure `axis` isn't in concat_kwargs, since we always want to concat along obs
        config = get_config(concat_kwargs[mod], {}, protected=['axis'])
        
        # Concatenate along the obs axis
        concatenated[mod] = sc.concat(adatas_for_mod, axis=0, **config)

        if config.get('join', None) == 'outer' and store_vars is not False:
            if store_vars is True:
                keys = [f"adata_{i}" for i in range(len(adatas_for_mod))]
            elif is_listlike(store_vars):
                if not len(store_vars) == len(adatas_for_mod):
                    raise ValueError("If providing a list-like for `store_vars`, it must match the number of adatas being concatenated.")
                keys = store_vars
                store_vars = True
            else:
                raise TypeError(f"`store_vars` must be a bool or list-like if `join='outer'` is used in `concat_kwargs`, not {type(store_vars)}.")
            if store_vars:
                # Store the specific variables present in each adata in .uns['vars_dict']
                vars_dict = {key: adata.var_names.tolist() for key, adata in zip(keys, adatas_for_mod)}
                concatenated[mod].uns['vars_dict'] = vars_dict

    # Assuming 'mudata' has a constructor that takes a dictionary where keys are modalities and 
    # values are the concatenated adata objects
    concatenated_mdata = md.MuData(concatenated)
    
    return concatenated_mdata


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

