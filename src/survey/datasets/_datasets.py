import scanpy as sc
from pathlib import Path

from ..singlecell.io import read_data
from ..helpers import SurveyPaths

HERE = Path(__file__).parent

# IMGS_PATH = HERE / "imgs/"

AVAILABLE_DATASETS = {
    'kidney': {
        'name': "wt_ms_kidney_5K",
        'desc': "Wild-type mouse kidney, chips 485 and 492, downsampled to ~5K cells"
    }
}

def kidney() -> sc.AnnData:
    """
    Load the wild-type mouse kidney 5K dataset.

    Returns
    -------
    mdata : sc.AnnData
        The AnnData object containing the dataset.
    paths : dict
        A dictionary with paths to related resources.
    """
    data_id = "kidney"

    spaths = SurveyPaths(HERE, data_id)
    mdata = read_data(spaths.get_h5_paths(last=True))
    
    if spaths.imgs and not spaths.imgs.exists():
        raise FileNotFoundError(f"Images path not found: {spaths.imgs}")

    paths = {'imgs': spaths.imgs} # Not now, but maybe just return spaths in the future

    return mdata, paths


def load(built_in=None, 
         data_dir=None, 
         data_id=None,
         name=None,
         last=False,
         normalize=None,
         log_transform=None):
    """
    Load a built-in dataset or provided dataset.

    Parameters
    ----------
    built_in : str, optional
        Name of the built-in dataset to load. If None, and data_dir and 
        data_id are also None, loads the default dataset.
    data_dir : str, Path, or None
        The directory containing data, supplied directly to SurveyPaths.
    data_id : str, optional
        The experiment or tissue ID, supplied to SurveyPaths.
    name : str, optional
        The name of the dataset to load, used with SurveyPaths to find the .h5mu file.
    last : bool, optional
        If True, load the most recent .h5mu file found with SurveyPaths. Default is False.
    normalize : bool, optional
        Whether to normalize the RNA data. Default is True.
    log_transform : bool, optional
        Whether to log-transform the RNA data. Default is True.
    
    Returns
    -------
    mdata : mudata.MuData
        The loaded MuData object.
    paths : dict
        A dictionary containing paths to relevant directories, e.g., {'imgs': Path to imgs directory}
    """
    if built_in is None and data_dir is None and data_id is None:
        built_in = list(AVAILABLE_DATASETS.keys())[0]
        use_built_in = True
    else:
        if built_in in AVAILABLE_DATASETS:
            use_built_in = True
        else:
            if (data_dir is None) != (data_id is None):
                raise ValueError("Both data_dir and data_id must be provided together, or both must be None to use a built-in dataset.")
            use_built_in = False
    
    if use_built_in:
        print(f"Loading built-in dataset {built_in}...")
        if built_in in AVAILABLE_DATASETS:
            # This assumes a function with the same name as the dataset key exists
            mdata, paths = globals()[built_in]()
        else:
            raise ValueError(f"Unknown built-in dataset: {built_in}. Available datasets are: {list(AVAILABLE_DATASETS.keys())}")
        if normalize is None:
            normalize = True
        if log_transform is None:
            log_transform = True
    else:
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir {data_dir} does not exist.")
        
        sdd = SurveyPaths(data_dir, data_id)
        mdata_path = sdd.get_h5_paths(name=name, last=last)
        if isinstance(mdata_path, list):
            raise ValueError(f"Multiple .h5mu files found: {', '.join([i.stem for i in mdata_path])}. Set a `name` or `last=True` to load a specific dataset.")
        imgs_dir = sdd.imgs
        
        pkl_path = mdata_path.with_suffix('.pkl')

        if not mdata_path.exists():
            raise FileNotFoundError(f"Data file does not exist at {mdata_path}.")
        if not pkl_path.exists():
            raise FileNotFoundError(f"Sidecar file does not exist at {pkl_path}. Cannot read Survey metadata.")
        if imgs_dir and not imgs_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist at {imgs_dir}.")
        
        mdata = read_data(mdata_path)
        paths = {'imgs': imgs_dir}
    
    
    if normalize:
        print("Normalizing RNA data...")
        mdata['rna'].layers['npc'] = sc.pp.normalize_total(mdata['rna'], target_sum=1e4, copy=True).X
    if log_transform:
        print("Log-transforming RNA data...")
        mdata['rna'].layers['npc-l1p'] = sc.pp.log1p(mdata['rna'], layer='npc', copy=True).layers['npc']
        del(mdata['rna'].layers['npc'])

    return mdata, paths