import scanpy as sc
from pathlib import Path

from ..singlecell.io import read_data

HERE = Path(__file__).parent

IMGS_PATH = HERE / "imgs/"

AVAILABLE_DATASETS = {
    "wt_ms_kidney_5K": "Wild-type mouse kidney, chips 485 and 492, downsampled to ~5K cells",
}


def wt_ms_kidney_5K() -> sc.AnnData:
    """
    Load the wild-type mouse kidney 5K dataset.

    Returns
    -------
    mdata : sc.AnnData
        The AnnData object containing the dataset.
    paths : dict
        A dictionary with paths to related resources.
    """
    relative_path = "kidney/wt_ms_kidney_5K.h5mu"

    mdata = read_data(HERE / relative_path)
    
    if not IMGS_PATH.exists():
        raise FileNotFoundError(f"Images path not found: {IMGS_PATH}")
    
    paths = {'imgs': IMGS_PATH}

    return mdata, paths


def load(built_in=None, data_dir=None, mdata_relative_path=None, imgs_relative_path="imgs/"):
    """
    Load a built-in dataset or provided dataset.

    Parameters
    ----------
    built_in : str, optional
        Name of the built-in dataset to load. If None, and data_dir and 
        mdata_relative_path are also None, loads the default dataset.
    data_dir : str, Path, or None
        The directory where the .h5mu file and associated files are located.
    mdata_relative_path : str or None
        The relative path to the .h5mu file within data_dir.
    imgs_relative_path : str, optional
        The relative path to the image directory within data_dir. 
        Defaults to "imgs/".
    
    Returns
    -------
    mdata : mudata.MuData
        The loaded MuData object.
    paths : dict
        A dictionary containing paths to relevant directories, e.g., {'imgs': Path to imgs directory}
    """
    if built_in is None:
        if (mdata_relative_path is None) and (data_dir is None):
            built_in = list(AVAILABLE_DATASETS.keys())[0]
            use_built_in = True
        elif (mdata_relative_path is not None) and (data_dir is not None):
            use_built_in = False
        else:
            raise ValueError("Either both data_dir and mdata_relative_path should be provided, or neither.")
    else:
        use_built_in = True

    if use_built_in:
        print(f"Loading built-in dataset {built_in}...")
        if built_in in AVAILABLE_DATASETS:
            # This assumes a function with the same name as the dataset key exists
            mdata, paths = globals()[built_in]()
        else:
            raise ValueError(f"Unknown built-in dataset: {built_in}. Available datasets are: {list(AVAILABLE_DATASETS.keys())}")
    else:
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(mdata_relative_path, str):
            mdata_relative_path = Path(mdata_relative_path)
        if not mdata_relative_path.suffix == '.h5mu':
            raise ValueError("mdata_relative_path should end with .h5mu")
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir {data_dir} does not exist.")
        
        mdata_path = data_dir / mdata_relative_path
        pkl_path = mdata_path.with_suffix('.pkl')
        imgs_dir = data_dir / imgs_relative_path

        if not mdata_path.exists():
            raise FileNotFoundError(f"Data file does not exist at {mdata_path}.")
        if not pkl_path.exists():
            raise FileNotFoundError(f"Sidecar file does not exist at {pkl_path}. Cannot read Survey metadata.")
        if not imgs_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist at {imgs_dir}.")
        
        mdata = read_data(mdata_path)
        paths = {'imgs': imgs_dir}
    
    print("Normalizing and log-transforming RNA data...")
    mdata['rna'].layers['npc'] = sc.pp.normalize_total(mdata['rna'], target_sum=1e4, copy=True).X
    mdata['rna'].layers['npc-l1p'] = sc.pp.log1p(mdata['rna'], layer='npc', copy=True).layers['npc']
    del(mdata['rna'].layers['npc'])

    return mdata, paths