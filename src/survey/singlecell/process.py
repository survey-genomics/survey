# Built-ins
import warnings
from typing import Dict, Optional, List, Union, Tuple

# Standard libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.axes import Axes

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.singlecell.scutils import clr_normalize
from survey.genplot import loglog_hist


def make_filter_genes_plots(adata: sc.AnnData,
                            vlines: Dict[str, Optional[float]] = {'count': None, 'cells': None},
                            ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
    '''
    Generate plots of gene counts and cell counts per gene.

    This function creates two log-log histograms to visualize the distribution
    of total counts per gene and the number of cells expressing each gene. It is
    useful for deciding filtering thresholds.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    vlines : dict, optional
        A dictionary with keys 'count' and 'cells' specifying the x-coordinates
        for vertical lines to be drawn on the plots, typically to indicate
        filtering thresholds. Default is {'count': None, 'cells': None}.
    ranges : dict, optional
        A dictionary with keys 'count' and 'cells' to set the x-axis range for
        the histograms. If None, default ranges are used.
        Default: {'count': (1, 1e8), 'cells': (1, 1e5)}.

    Returns
    -------
    None
        This function displays plots and does not return any value.

    Examples
    --------
    >>> import scanpy as sc
    >>> import numpy as np
    >>> adata = sc.AnnData(np.random.poisson(1, (100, 2000)))
    >>> make_filter_genes_plots(adata, vlines={'count': 100, 'cells': 10})
    '''

    default_ranges = {'count':(1, 1e8), 'cells':(1, 1e5)}
    if ranges is None:
        ranges = default_ranges
    else:
        assert isinstance(ranges, dict), "Param `ranges` must be a dictionary."
        assert all([k in ('count', 'cells') for k in ranges]), "Param `ranges` must have keys 'count' and/or 'cells'."
        default_ranges.update(ranges)


    fig, axes = plt.subplots(1,2,figsize=(15,4))

    sc.pp.filter_genes(adata, min_counts=0)
    vals = sc.pp.filter_genes(adata, min_counts=0, inplace=False)[1] # total number of UMIs observed
    axes[0] = loglog_hist(vals, ranges['count'], 300, title='Counts per gene', ax=axes[0], vline=vlines['count']);
    axes[0].set_ylabel('Number of genes')
    axes[0].set_xlabel('Count per gene')

    sc.pp.filter_genes(adata, min_cells=0)
    vals = sc.pp.filter_genes(adata, min_cells=0, inplace=False)[1] # total number of UMIs observed
    axes[1] = loglog_hist(vals, ranges['cells'], 300, title='Cells per gene', ax=axes[1], vline=vlines['cells']);
    axes[1].set_ylabel('Number of genes')
    axes[1].set_xlabel('Cells per gene')
    plt.tight_layout()

    return


def pre_process(mdata: md.MuData,
                transforms: Dict[str, Dict[str, bool]],
                capture_norm: Optional[List[List[str]]] = None,
                filter_genes: Optional[Dict[str, int]] = None,
                total: Optional[List[Union[int, float]]] = None,
                add_params: bool = True) -> md.MuData:
    """
    Pre-process a MuData object by applying normalization and filtering.

    This function applies various transformations to specified modalities,
    optionally performs cross-modality count normalization, and filters genes.
    The results are stored in new layers of the `MuData` object.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object to pre-process.
    transforms : dict
        A dictionary where keys are modality names (e.g., 'rna', 'adt') and
        values are dictionaries of transformations to apply (e.g.,
        {'npc': True, 'l1p': True}).
    capture_norm : list of list of str, optional
        A list of modality groups. For each group, counts are summed across
        modalities before cell-wise normalization ('npc'). This is useful for
        technologies where modalities share a capture mechanism.
        Example: [['rna', 'adt'], ['hto']]
    filter_genes : dict, optional
        A dictionary specifying gene filtering criteria for the 'rna' modality.
        Keys can be 'count' (for `min_counts`) and/or 'cells' (for `min_cells`).
        Example: {'count': 10, 'cells': 3}
    total : list of numeric, optional
        Target sum for per-cell normalization, corresponding to each group in
        `capture_norm`. If None, defaults to 1 for each group (simple division
        by total counts).
    add_params : bool, optional
        If True, stores the parameters used for pre-processing in `mdata.uns`.

    Returns
    -------
    md.MuData
        The processed MuData object.

    Notes
    -----
    Accepted codes for `transforms`:
    - `npc`: Normalize Per Cell. Divides counts by the total counts for the
      cell within its `capture_norm` group.
    - `l1p`: Applies `log(x + 1)` transformation.
    - `clr`: Applies Centered Log-Ratio transformation.

    The function creates a new layer for each modality in `transforms`, named
    by concatenating the applied transformation codes (e.g., 'npc-l1p').

    Examples
    --------
    >>> capture_groups = [['rna', 'adt'], ['hto']]
    >>> transformations = {
    ...     'rna': {'npc': True, 'l1p': True},
    ...     'adt': {'npc': True, 'clr': True},
    ...     'hto': {'npc': True, 'clr': True}
    ... }
    >>> gene_filters = {'count': 10, 'cells': 3}
    >>> mdata = pre_process(mdata,
    ...                     transforms=transformations,
    ...                     capture_norm=capture_groups,
    ...                     filter_genes=gene_filters)
    """

    params = {
        'transforms': transforms,

        # list of lists breaks things, and need str keys, so we have to convert to dict with str
        'capture_norm': dict(zip(map(str,range(len(capture_norm))), capture_norm)), 

        'filter_genes': filter_genes,
        'total': total,
        }  

    if capture_norm is None:
        capture_norm = []
    if total is None:
        total = [1 for _ in capture_norm]

    # If any of the keys of transforms have `npc: True`, calculate capture_norm_vals
    if any([transforms[mod].get('npc', False) for mod in transforms]):
        capture_norm_vals = dict()
        for mod_set, t in zip(capture_norm, total):
            for mod in mod_set:
                mdata[mod].obs['n_counts'] = np.array(mdata[mod].X.sum(1)).flatten()

            countsum = pd.concat([mdata[mod].obs['n_counts'] for mod in mod_set], axis=1).fillna(0).sum(1)

            for mod in mod_set:
                capture_norm_vals[mod] = countsum[mdata[mod].obs_names]/t
    
    # For each modality, first normalize by total counts, then apply the transformation
    for mod in transforms:
        # Make a transformation code
        tcode = '-'.join([i for i in iter(transforms[mod]) if transforms[mod][i]])
        if transforms[mod].get('npc'):
            arr = mdata[mod].X / capture_norm_vals[mod].values.reshape(-1, 1)
            mdata[mod].layers[tcode] = arr.tocsr()
        if transforms[mod].get('l1p'):
            sc.pp.log1p(mdata[mod], layer=tcode)
        if transforms[mod].get('clr'):
            try:
                mdata[mod].layers[tcode] = mdata[mod].layers[tcode].toarray()
            except AttributeError:
                pass
            if len(mdata[mod].var) > 0:
                mdata[mod].layers[tcode] = clr_normalize(mdata[mod].layers[tcode])
    
    if filter_genes is not None:
        sc.pp.filter_genes(mdata['rna'], min_counts=filter_genes['count'])
        sc.pp.filter_genes(mdata['rna'], min_cells=filter_genes['cells'])

    if add_params:
        mdata.uns['pre_process_params'] = params
        
    return mdata


def dim_reduction(adata: sc.AnnData,
                  input_layer: str = 'X',
                  output_layer: Optional[str] = None,
                  log1p: bool = True,
                  scale: bool = True,
                  scale_max_value: Optional[float] = None,
                  scale_zero_center: bool = True,
                  combat: bool = True,
                  combat_key: str = 'batch',
                  pca: bool = True,
                  ncomps: int = 150,
                  copy: bool = True,
                  plot_graphs: bool = True) -> None:
    '''
    Perform a standard dimensionality reduction workflow on an AnnData object.

    This pipeline can include log-transformation, scaling, batch correction
    with ComBat, and Principal Component Analysis (PCA).

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    input_layer : str, optional
        Name of the layer in `adata` to use as input. Defaults to `adata.X`.
    output_layer : str, optional
        Name of the layer in `adata` to store the processed data. If None, the
        input layer is overwritten.
    log1p : bool, optional
        If True, apply `log(x + 1)` normalization.
    scale : bool, optional
        If True, scale data to unit variance.
    scale_max_value : float, optional
        Clip values after scaling to this maximum value.
    scale_zero_center : bool, optional
        If True, center data to mean 0.
    combat : bool, optional
        If True, apply ComBat batch correction. Requires `scale` to be True.
    combat_key : str, optional
        Key in `adata.obs` for batch information used by ComBat.
    pca : bool, optional
        If True, perform PCA.
    ncomps : int, optional
        Number of principal components to compute.
    copy : bool, optional
        If True, operates on a copy of the data. If False, `adata` is modified
        in-place, which can be more memory-efficient but overwrites `adata.X`.
    plot_graphs : bool, optional
        If True, display plots for data distribution and PCA variance ratio.

    Returns
    -------
    None
        This function modifies `adata` in-place.

    Raises
    ------
    ValueError
        If `combat` is True but `scale` is False.
    '''

    if combat and not scale:
        raise ValueError('Param `combat` cannot be True if `scale` is False.')

    params = {
        'input_layer': input_layer,
        'output_layer': output_layer,
        'log1p': log1p,
        'scale': scale,
        'scale_max_value': scale_max_value,
        'scale_zero_center': scale_zero_center,
        'combat': combat,
        'combat_key': combat_key,
        'pca': pca,
        'ncomps': ncomps,
        'copy': copy
        }

    # Since layers aren't keywords implemented in all functions, just make a new momentarily
    # Obviously, will be memory intensive, the lack of layer implementation is to blame
    if copy:
        adata_copy = adata.copy()
    else:
        adata_copy = adata
        warnings.warn('With param `copy` == False, `adata` will be modified in-place and '\
                      'raw counts will be transformed (i.e. adata.X will not contain raw counts).')

    if not input_layer == 'X':
        adata_copy.X = adata.layers[input_layer].copy()
    
    if log1p:
        print('Log transforming...')
        sc.pp.log1p(adata_copy)
    
    if scale:
        print('Scaling')
        sc.pp.scale(adata_copy, max_value=scale_max_value, zero_center=scale_zero_center)
    if combat:
        print('Running ComBat')
        sc.pp.combat(adata_copy, key=combat_key, inplace=True)
        print('Re-scaling')
        sc.pp.scale(adata_copy, max_value=scale_max_value, zero_center=scale_zero_center)

    if plot_graphs:
        if isinstance(adata_copy.X, np.ndarray):
            vals = adata_copy.X.flatten()
        else: # it's still a sparse matrix?
            vals = adata_copy.X.data
        plt.figure(figsize=(5, 3))
        plt.hist(vals, bins=100);
        plt.yscale('log')
        plt.ylabel('Number of values')
        plt.xlabel('Values')
        plt.title('Distribution of scaled values into PCA')
        plt.show()

    if pca:
        print('Running PCA')

        sc.pp.pca(adata_copy, n_comps=ncomps)
        if plot_graphs:
            sc.pl.pca_variance_ratio(adata_copy, log=True, n_pcs=ncomps)

    # Start rebuilding the adata

    adata.uns['dimred_params'] = params

    if output_layer is not None:
        adata.layers[output_layer] = adata_copy.X
    else:
        if input_layer == 'X':
            adata.X = adata_copy.X.copy()
        else:
            adata.layers[input_layer] = adata_copy.X.copy()

    try: # if log1p
        adata.uns['log1p'] = adata_copy.uns['log1p']
    except KeyError:
        pass

    try: # if pca
        adata.obsm['X_pca'] = adata_copy.obsm['X_pca']
        adata.varm['PCs'] = adata_copy.varm['PCs']
        adata.uns['pca'] = adata_copy.uns['pca']
        adata.var[['mean', 'std']] = adata_copy.var[['mean', 'std']].copy()
    except KeyError:
        pass
        
    return


def add_pmito(data: Union[sc.AnnData, md.MuData],
              org: str = 'mouse',
              plot: bool = True,
              name: str = '%mito') -> None:
    '''
    Calculate and add the percentage of mitochondrial counts to `.obs`.

    This function identifies mitochondrial genes based on a common prefix
    ('mt-' for mouse, 'MT-' for human) and calculates the percentage of
    total counts per cell that map to these genes.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The data object. If a MuData object is provided, the calculation is
        performed on the 'rna' modality.
    org : {'mouse', 'human'}, optional
        Organism, used to determine the mitochondrial gene prefix.
    plot : bool, optional
        If True, display a histogram of the mitochondrial percentages.
    name : str, optional
        The name of the column to add to `.obs`.

    Returns
    -------
    None
        This function modifies the data object in-place.

    Raises
    ------
    ValueError
        If an unsupported `org` is provided, or if `data` is a MuData object
        without an 'rna' modality.
    '''

    if isinstance(data, sc.AnnData):
        adata = data
    elif isinstance(data, md.MuData):
        if 'rna' not in data.mod:
            raise ValueError("If mudata is passed, `rna` mod must be present.")
        adata = data['rna']
        
    if org == 'mouse':
        sw = 'mt-'
    elif org == 'human':
        sw = 'MT-'
    else:
        raise ValueError("Only mouse and human genomes supported.")
    
    genelist = adata.var_names.tolist()
    mito_genes_names = [gn for gn in genelist if gn.startswith(sw)]
    mito_genes = [genelist.index(gn) for gn in mito_genes_names]

    adata.obs[name] = np.ravel(np.sum(adata[:, mito_genes].X, axis=1)) / np.ravel(np.sum(adata.X, axis=1))

    if plot:
        plt.figure(figsize=(5,3))
        plt.hist(adata.obs[name].values,bins=200,density=True)
        plt.xlabel('Percent mitochondrial')
        plt.ylabel('Density')
        plt.show()

    return

