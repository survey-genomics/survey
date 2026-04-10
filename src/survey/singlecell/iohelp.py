# Built-ins
from typing import Dict, Union

# Standard libs
import numpy as np
import pandas as pd

# Other 3rd-party libs
import h5py

PandasObj = Union[pd.Series, pd.DataFrame]

_CB_SKIP_DATASET_NAMES = {
    # CSR matrix internals
    "data", "indices", "indptr", "shape",
    # feature tables (not per-barcode)
    "features",
}

def _as_str_array(x: np.ndarray) -> np.ndarray:
    """Convert a numpy array of bytes/objects to str if needed."""
    if x.dtype.kind in ("S", "O"):  # bytes or python objects
        return x.astype(str)
    return x

def _series_on_full_barcodes(
    values: np.ndarray,
    full_barcodes: np.ndarray,
    name: str,
    latent_inds: np.ndarray | None = None,
) -> pd.Series:
    """
    Return a pd.Series indexed by full_barcodes.
    If latent_inds is provided, scatter values onto those indices, NaN elsewhere.
    """
    full_index = pd.Index(full_barcodes, name="barcode")

    # scalar
    if values.shape == ():
        return pd.Series([values.item()] * len(full_barcodes), index=full_index, name=name)

    values = np.asarray(values)

    if latent_inds is None:
        if values.shape[0] != len(full_barcodes):
            raise ValueError(f"{name}: cannot align length {values.shape[0]} to n_barcodes {len(full_barcodes)}")
        return pd.Series(values, index=full_index, name=name)

    # scatter (latent -> full)
    out = pd.Series(np.nan, index=full_index, name=name, dtype="float64")
    out.iloc[latent_inds] = values.astype(float, copy=False)
    return out

def _df_on_full_barcodes(
    values: np.ndarray,
    full_barcodes: np.ndarray,
    name: str,
    latent_inds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Return a pd.DataFrame indexed by full_barcodes for 2D arrays.
    If latent_inds is provided, scatter rows to those indices, NaN elsewhere.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"{name}: expected 2D array, got shape {values.shape}")

    full_index = pd.Index(full_barcodes, name="barcode")

    if latent_inds is None:
        if values.shape[0] != len(full_barcodes):
            raise ValueError(f"{name}: cannot align rows {values.shape[0]} to n_barcodes {len(full_barcodes)}")
        cols = [f"{name}__{i}" for i in range(values.shape[1])]
        return pd.DataFrame(values, index=full_index, columns=cols)

    # scatter rows
    cols = [f"{name}__{i}" for i in range(values.shape[1])]
    out = pd.DataFrame(np.nan, index=full_index, columns=cols)
    out.iloc[latent_inds, :] = values
    return out

def extract_cb_h5(cb_h5_path: str, extract_only='droplet_latents') -> Dict[str, PandasObj]:
    """
    Extract as many per-barcode datasets as possible from a CellBender-style H5
    into pandas objects indexed by matrix/barcodes.

    Returns dict mapping a key like:
      "droplet_latents/cell_probability" -> pd.Series
      "droplet_latents/gene_expression_encoding" -> pd.DataFrame
    """
    out: Dict[str, PandasObj] = {}

    with h5py.File(cb_h5_path, "r") as f:
        # Universal barcode list
        full_barcodes = _as_str_array(f["matrix/barcodes"][()])

        # Latent mapping (if present)
        latent_inds = None
        if "droplet_latents" in f and "barcode_indices_for_latents" in f["droplet_latents"]:
            latent_inds = np.asarray(f["droplet_latents/barcode_indices_for_latents"][()]).astype(np.int64)

        # Walk all datasets and try to align to barcodes
        def visitor(name: str, obj):
            if not isinstance(obj, h5py.Dataset):
                return

            # Skip clearly-non-barcode-level pieces
            base = name.split("/")[-1]
            if base in _CB_SKIP_DATASET_NAMES:
                return
            if name.startswith("matrix/") and base in _CB_SKIP_DATASET_NAMES:
                return

            arr = obj[()]

            # Decode strings but only if this is actually barcode-level (rare)
            # (Most barcode-level values are numeric.)
            if isinstance(arr, np.ndarray):
                arr = _as_str_array(arr)

            key = name  # keep full path as key, e.g. "droplet_latents/cell_probability"

            # Heuristics:
            # - droplet_latents/*: try latent scatter if 1D/2D and length matches n_latents
            # - otherwise: try direct align if first dim matches n_barcodes
            try:
                if isinstance(arr, np.ndarray) and arr.ndim == 1:
                    if latent_inds is not None and name.startswith("droplet_latents/") and arr.shape[0] == latent_inds.shape[0]:
                        out[key] = _series_on_full_barcodes(arr, full_barcodes, key, latent_inds=latent_inds)
                    elif arr.shape[0] == len(full_barcodes):
                        out[key] = _series_on_full_barcodes(arr, full_barcodes, key, latent_inds=None)
                    else:
                        # not barcode-level; ignore
                        return

                elif isinstance(arr, np.ndarray) and arr.ndim == 2:
                    if latent_inds is not None and name.startswith("droplet_latents/") and arr.shape[0] == latent_inds.shape[0]:
                        out[key] = _df_on_full_barcodes(arr, full_barcodes, key, latent_inds=latent_inds)
                    elif arr.shape[0] == len(full_barcodes):
                        out[key] = _df_on_full_barcodes(arr, full_barcodes, key, latent_inds=None)
                    else:
                        return

                elif np.isscalar(arr):
                    # scalar metadata: not barcode-level; skip by default
                    return

                else:
                    # higher-dim arrays etc: skip
                    return

            except Exception:
                # If anything is weird (dtypes, shapes), just skip it rather than failing the whole extraction.
                return

        f.visititems(visitor)
    
    if extract_only is not None:
        out = {k.split('/')[-1]: v for k, v in out.items() if k.startswith(extract_only)}
        out = pd.concat([v.dropna() for v in out.values()], keys=out.keys(), axis=1)

    return out