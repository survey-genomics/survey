# Built-ins
import warnings
import numbers
from copy import deepcopy
import re
from typing import Optional, Union, Dict, Any, Tuple, List

# Standard libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.genutils import is_listlike, get_config
from survey.genplot import subplots
from survey.singlecell.datatypes import determine_data
from survey.singlecell.obs import get_obs_df
from survey.singlecell.meta import get_cat_dict
from survey.singlecell.decorate import decorate_scatter, get_pm


def get_plot_data(data: Union[sc.AnnData, md.MuData],
                  mod: Optional[str] = None,
                  color: Optional[str] = None,
                  basis: Optional[str] = None,
                  components: Optional[List[int]] = None,
                  size: Optional[Union[float, List[float]]] = None,
                  scale: Optional[float] = None,
                  layer: Optional[str] = None,
                  sort_order: bool = True) -> Tuple[pd.DataFrame, Dict, Optional[Dict]]:
    """
    Prepares a DataFrame for plotting from single-cell data.

    This function extracts coordinates, color, and size information from an
    AnnData or MuData object and organizes it into a single DataFrame suitable
    for scatter plot functions.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The input single-cell data object.
    mod : str, optional
        If `data` is a MuData object, specifies the modality to use for color
        and size information.
    color : str, optional
        The key for coloring points. Can be a column in `.obs`, a gene name,
        a named color, or an RGBA tuple.
    basis : str, optional
        The embedding to use for coordinates (e.g., 'umap', 'pca').
    components : list of int, optional
        The components of the embedding to use (e.g., `[0, 1]`).
    size : float or list of float, optional
        The size of the points. Can be a single value or a list of values.
    scale : float, optional
        A scaling factor to apply to the point sizes.
    layer : str, optional
        The data layer to use for expression-based coloring.
    sort_order : bool, optional
        If True, sorts the data for plotting (e.g., by color value).
        If False, shuffles the data. If None, maintains original order.

    Returns
    -------
    plot_df : pd.DataFrame
        A DataFrame with columns 'x1', 'x2', 'c' (color), and 's' (size).
    dtypes : dict
        A dictionary containing the determined data types for `data`, `color`,
        and `basis`.
    cdict : dict or None
        A color dictionary if the color is categorical, otherwise None.
    """

    def _get_plotdf(adata, basis, components):
        # Instantiate the plot_df with the points data
        X = adata.obsm[f'X_{basis}']
        if components is not None:
            X = X[:, components]

        plot_df = pd.DataFrame(data=X, index=adata.obs_names, columns=['x1', 'x2'])

        return plot_df
    

    def _add_color_plotdf(color_type, plot_df, adata, color, layer):
        # Add color data to the plot_df
        cdict = None

        if color_type == 'num':
            color_vector = adata.obs_vector(color, layer=layer)
            # Create a Series with the index from the color adata before reindexing
            color_series = pd.Series(color_vector, index=adata.obs_names)
            plot_df['c'] = color_series.reindex(plot_df.index)

        elif color_type == 'cat':
            # Store `color` because we may need it to sort later
            color_series = adata.obs[color].copy()
            plot_df['color'] = color_series.reindex(plot_df.index)
            cdict = get_cat_dict(adata, color, 'color')
            plot_df['c'] = plot_df['color'].map(cdict)

        elif color_type == 'named':
            plot_df['c'] = color

        elif color_type == 'rgba':
            plot_df['c'] = [tuple(color)]*len(plot_df)

        else:
            raise ValueError(f"Color type '{color}' not found.")
        
        return plot_df, cdict


    def _add_size_plotdf(color_type, plot_df, adata, size, scale):
        # Add size data to the plot_df
        if isinstance(size, numbers.Number):
            plot_df['s'] = size
        elif is_listlike(size):
            if len(size) != len(plot_df.index):
                raise ValueError(
                    "Size list must be the same length as the number of "
                    f"points in the data ({len(plot_df.index)})."
                    )
            plot_df['s'] = pd.Series(size, index=plot_df.index)
        else:
            if color_type == 'cat':
                # Size not implemented for now....
                #
                # # If color is categorical, try to also get size from metadata
                # # Will return None if not found
                # sdict = get_cat_dict(adata, color, 'size')
                # if sdict is not None:
                #     size_series = adata.obs[color].map(sdict)
                #     plot_df['s'] = size_series.reindex(plot_df.index)
                #
                pass
            if size is None and 's' not in plot_df.columns:
                # Still no size, set to 1
                plot_df['s'] = 1
            elif 's' not in plot_df.columns:
                raise ValueError(
                    "Size must be a number, a list-like object, or "
                    "stored in obs metadata adata.uns['meta'].")
        if scale is not None:
            plot_df['s'] *= scale
        return plot_df

    
    def _sort_plotdf(plot_df, sort_order, color_type):
        # Sort or shuffle data based on sort_order and color type
        if sort_order is None:
            # No sorting or shuffling, use the original order
            pass
        elif sort_order is True:
            if color_type == 'cat':
                # Sort based on categorical order
                order = np.argsort(plot_df['color'].cat.codes)
                plot_df = plot_df.iloc[order]
            elif color_type == 'num':
                # Sort numerically in ascending order
                plot_df = plot_df.sort_values(by='c', ascending=True)
            elif color_type == 'named':
                # Sorting need not happen, all points are the same color
                pass
            else:
                raise ValueError(f"Color type '{color_type}' not recognized.")
        elif sort_order is False:
            plot_df = plot_df.sample(frac=1)
        else:
            raise ValueError("sort_order must be True, False, or None.")
        return plot_df
    
    if scale is not None and not isinstance(scale, numbers.Number):
        raise TypeError("Scale must be a number.")

    dtypes = determine_data(data, mod=mod, color=color, basis=basis, error=True)

    if dtypes['data'] == 'adata':
        adata = data
        plot_df = _get_plotdf(adata, basis, components)

    else: # mdata
        if mod is not None:
            adata = data[mod]
        else:
            adata = data[dtypes['color']['mod']]
        plot_df = _get_plotdf(data[dtypes['basis']], basis, components)

    color_type = dtypes['color']['type']

    plot_df, cdict = _add_color_plotdf(color_type, plot_df, adata, color, layer)
    plot_df = _add_size_plotdf(color_type, plot_df, adata, size, scale)
    plot_df = _sort_plotdf(plot_df, sort_order, color_type)

    return plot_df, dtypes, cdict


def scatter(data: Union[sc.AnnData, md.MuData],
            color: Optional[str] = None,
            layer: Optional[str] = None,
            sort_order: bool = True,
            mod: Optional[str] = None,
            basis: str = 'umap',
            components: Optional[List[int]] = None,
            size: Optional[Union[float, List[float]]] = None,
            scale: Optional[float] = None,
            plot_data: Optional[Tuple[pd.DataFrame, Dict, Optional[Dict]]] = None,
            title: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            plot_label: bool = True,
            plot_label_params: Optional[Dict] = None,
            cbar: Optional[bool] = None,
            cbar_params: Optional[Dict] = None,
            legend: Optional[bool] = None,
            legend_params: Optional[Dict] = None,
            lims: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
            invert: bool = False,
            **kwargs) -> plt.Axes:
    """
    Creates a scatter plot from single-cell data, typically an embedding.

    This is a versatile plotting function that handles various data types,
    color schemes, and plot decorations like legends and colorbars.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The input data object.
    color : str, optional
        The key for coloring points.
    layer : str, optional
        The data layer to use for expression-based coloring.
    sort_order : bool, optional
        Whether to sort or shuffle points before plotting.
    mod : str, optional
        The modality to use if `data` is a MuData object.
    basis : str, optional
        The embedding to use (e.g., 'umap').
    components : list of int, optional
        The components of the embedding to plot.
    size : float or list, optional
        The size of the points.
    scale : float, optional
        A scaling factor for point sizes.
    plot_data : tuple, optional
        Pre-computed plot data from `get_plot_data` to bypass data extraction.
    title : str, optional
        The title for the plot.
    ax : plt.Axes, optional
        An existing Axes object to plot on.
    plot_label : bool, optional
        If True, adds a label indicating the color key to the plot.
    plot_label_params : dict, optional
        Parameters for customizing the plot label.
    cbar : bool, optional
        If True, displays a colorbar (for numerical colors).
    cbar_params : dict, optional
        Parameters for customizing the colorbar.
    legend : bool, optional
        If True, displays a legend (for categorical colors).
    legend_params : dict, optional
        Parameters for customizing the legend.
    lims : tuple, optional
        A tuple `((x_min, x_max), (y_min, y_max))` to set plot limits.
    invert : bool, optional
        If True, sets the plot background to black.
    **kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    plt.Axes
        The Axes object containing the plot.
    """

    def _run_scatter(ax, X, c, basis, s, lims, **kwargs):
        # Create a scatter plot
        if ax is None:
            fig, ax = plt.subplots()

        # Plot data
        scpc = ax.scatter(X[:, 0], X[:, 1], c=c, s=s, **kwargs)
        ax.set_xlabel(f'{basis.upper()}1')
        ax.set_ylabel(f'{basis.upper()}2')
        
        # Remove grid, ticks, and tick labels
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Adjust the lims
        if lims is not None:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])

        fig = ax.figure

        return fig, ax, scpc
    
    # Get ParamManagers
    plot_label_params = plot_label_params or {}
    cbar_params = cbar_params or {}
    legend_params = legend_params or {}
    plot_label_params.update({'invert': invert})
    cbar_params.update({'invert': invert})
    user_params = {'plot_label': plot_label_params, 'cbar': cbar_params, 'legend': legend_params}
    configs = {plot_type: get_pm(plot_type).get_params(user_params[plot_type]) for plot_type in user_params}

    if plot_data is not None: # in case its already been computed
        if not isinstance(plot_data, tuple) or len(plot_data) != 3:
            raise ValueError("plot_data must be a tuple of (plot_df, dtypes, cdict).")
        # Maybe add some more checks here?
        plot_df, dtypes, cdict = plot_data
    else:
        plot_df, dtypes, cdict = get_plot_data(data, mod, color, basis, components, size, scale, layer, sort_order)

    color_type = dtypes['color']['type']

    # Run the scatter plot
    X = plot_df[['x1', 'x2']].values
    c = plot_df['c'].values
    s = plot_df['s'].values

    fig, ax, scpc = _run_scatter(ax, X, c, basis, s, lims, **kwargs)

    
    # Decorate the scatter plot

    ## Add a colorbar if color values are provided and it's not categorical    
    if cbar is None:
        if color_type == 'num':
            cbar = True
        else:
            cbar = False
    elif cbar is True and color_type != 'num':
        warnings.warn("cbar ignored when color_type is not 'num'.")
        cbar = False
    if cbar:
        ax = decorate_scatter(ax, configs['cbar'], plot_type='cbar', fig=fig, scpc=scpc)
    
    ## Add a legend if color values are provided and categorical    
    if legend is None:
        if color_type == 'cat':
            legend = True
        else: 
            legend = False
    elif legend is True and color_type != 'cat':
        warnings.warn("legend ignored when color_type is not 'cat'.")
        legend = False
    if legend:
        ax = decorate_scatter(ax, configs['legend'], plot_type='legend', cdict=cdict)

    ## Add a plot_label if requested:
    if plot_label and color_type != 'named':
        # Silently ignore plot_label if color is named, since its default 
        # is True and plot label is not useful
        ax = decorate_scatter(ax, configs['plot_label'], plot_type='plot_label', label=color)

    # Set the title if provided
    if title:
        ax.set_title(title)

    # Invert if requested
    if invert:
        ax.set_facecolor('black')

    return ax


def labeled_scatter(data: Union[sc.AnnData, md.MuData],
                    color: str,
                    mod: Optional[str] = None,
                    basis: str = 'umap',
                    size: float = 1,
                    numbered: bool = False,
                    start: int = 1,
                    legend: bool = False,
                    global_adj: Optional[Tuple[float, float]] = None,
                    match_color: bool = True,
                    label_contrast: Optional[bool] = None,
                    scatter_params: Optional[Dict] = None,
                    legend_params: Optional[Dict] = None,
                    label_scatter_params: Optional[Dict] = None,
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Creates a scatter plot with labels for each categorical group.

    This function overlays text labels on the centroids of categorical clusters
    in an embedding plot.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The input data object.
    color : str
        The categorical key in `.obs` to use for coloring and labeling.
    mod : str, optional
        The modality to use if `data` is a MuData object.
    basis : str, optional
        The embedding to use.
    size : float, optional
        The size of the scatter points.
    numbered : bool, optional
        If True, replaces text labels with numbers and adds a numbered legend.
    start : int, optional
        The starting number for labels if `numbered` is True.
    legend : bool, optional
        If True, displays a legend.
    global_adj : tuple of (float, float), optional
        A tuple `(x_adj, y_adj)` to globally shift all labels.
    match_color : bool, optional
        If True, label text color matches the cluster color.
    label_contrast : bool, optional
        If True, adjusts label color for better contrast against the background.
    scatter_params : dict, optional
        Parameters passed to the underlying `scatter` function.
    legend_params : dict, optional
        Parameters for customizing the legend.
    label_scatter_params : dict, optional
        Parameters for customizing the text labels.
    ax : plt.Axes, optional
        An existing Axes object to plot on.

    Returns
    -------
    plt.Axes
        The Axes object containing the plot.
    """

    dtypes = determine_data(data, mod, color, error=True)

    if dtypes['data'] == 'adata':
        adata = data
    else: # mdata
        if mod is not None:
            adata = data[mod]
        else:
            adata = data[dtypes['color']['mod']]

    color_type = dtypes['color']['type']

    if color_type != 'cat':
        raise ValueError("labeled_scatter only works with categorical color.")

    # Get ParamManagers, update with the kwargs
    legend_params = legend_params or {}
    label_scatter_params = label_scatter_params or {}
    if not match_color and label_contrast:
        warnings.warn(
            "Parameter 'label_contrast' can only be set to True if 'match_color' is also True. "
            "Setting 'label_contrast' to False."
        )
        label_contrast = False
    if match_color and label_contrast is None:
        label_contrast = True
    label_scatter_params.update({
        'match_color': match_color,
        'label_contrast': label_contrast,
    })
    user_params = {'legend': legend_params, 'label_scatter': label_scatter_params}
    configs = {plot_type: get_pm(plot_type).get_params(user_params[plot_type]) for plot_type in user_params}

    # Validate other inputs
    ## Confirm no legend is requested from scatter,
    ## If user wants legend, we need to plot our own
    if scatter_params is not None and 'legend' in scatter_params:
        raise ValueError(
            "scatter_params should not contain 'legend'. To plot legend, pass legend=True directly."
        )
    default_config = {
        'legend': False, 
    }
    scatter_config = get_config(scatter_params, default_config, protected={'legend'})
    
    # Plot the underlying scatter
    if ax is None:
        fig, ax = subplots(1, fss=10, ar=1)

    ax = scatter(adata, color=color, size=size, ax=ax, basis=basis, **scatter_config)


    # Get color dict and centroids
    if match_color is True:
        # If match_color is True, we need the color dict
        cdict = get_cat_dict(adata, color, 'color')
    else:
        cdict = None
    centroids = get_obs_df(adata, obs_keys=[color], obsm_keys=['X_' + basis]).groupby(color, observed=False).median()

    # Adjust the positions of centroids
    if global_adj is not None:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xrange, yrange = xlim[1] - xlim[0], ylim[1] - ylim[0]
        centroids['X_%s_0' % basis] += global_adj[0]*xrange
        centroids['X_%s_1' % basis] += global_adj[1]*yrange

    # Adjust the labels and cdicts if numbered
    # They will be different for labeled points and legend if numbered
    cdict_labeled = cdict
    cdict_legend = cdict

    if numbered:
        new_index_legend = [' '.join([str(i), cat]) for i, cat in enumerate(centroids.index, start=start)]
        new_index_labeled = range(start, len(centroids.index) + start)
        if match_color is True:
            cdict_labeled = dict(zip(new_index_labeled, [cdict[i] for i in centroids.index]))
            cdict_legend = dict(zip(new_index_legend, [cdict[i] for i in centroids.index]))
        centroids.index = new_index_labeled

    # Plot the centroids with labels
    ax = decorate_scatter(ax, configs['label_scatter'], plot_type='label_scatter', positions=centroids, cdict=cdict_labeled)

    # Add legend if requested
    if legend:
        if cdict is None:
            raise ValueError(
                "cdict must be provided to plot legend. "
                "Set match_color=True to get cdict."
            )
        ax = decorate_scatter(ax, configs['legend'], plot_type='legend', cdict=cdict_legend)

    return ax


def highlight(data: Union[sc.AnnData, md.MuData],
              color: str,
              mod: Optional[str] = None,
              cats: Optional[Union[str, List[str], range]] = None,
              size: float = 1,
              bg_color: str = 'lightgray',
              bg_size: float = 1,
              ncols: Optional[int] = None,
              fss: Optional[float] = None,
              ar: Optional[float] = None,
              sep: bool = True,
              scatter_params: Optional[Dict] = None) -> List[plt.Axes]:
    """
    Highlights one or more categorical groups in a scatter plot.

    This function can either create separate plots for each highlighted group
    or overlay all highlighted groups on a single plot.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The input data object.
    color : str
        The categorical key in `.obs` to use.
    mod : str, optional
        The modality to use if `data` is a MuData object.
    cats : str, list, or range, optional
        The categories to highlight. If None, all categories are highlighted.
    size : float, optional
        The size of the highlighted points.
    bg_color : str, optional
        The color of the non-highlighted background points.
    bg_size : float, optional
        The size of the background points.
    ncols : int, optional
        Number of columns for the subplot grid if `sep=True`.
    fss : float, optional
        Figure size scaler for the subplot grid.
    ar : float, optional
        Aspect ratio for the subplot grid.
    sep : bool, optional
        If True, creates a separate subplot for each highlighted category.
        If False, highlights all categories on a single plot.
    scatter_params : dict, optional
        Parameters passed to the underlying `scatter` function.

    Returns
    -------
    list of plt.Axes
        A list of the Axes objects containing the plots.
    """
    
    def validate_cats(cats, adata_cats):
        # Determine categories to highlight
        if cats is None:
            cats = adata_cats
        elif isinstance(cats, range):
            cats = [i for i in adata_cats if int(i.split(',')[0]) in cats]
        else:
            if isinstance(cats, str):
                cats = [cats]
            if is_listlike(cats):
                found = False
                if all([i in adata_cats for i in cats]):
                    found = True
                if not found and any([isinstance(i, int) for i in cats]):
                    # Check to see if numbered clusters were passed
                    cats = [str(i) for i in cats]
                    if all([i in adata_cats for i in cats]):
                        found = True
                if not found:
                    # Last resort, check for the top parent cluster of any subclusters
                    matches = [re.match(r'^(\d+)(,\d+)+$', i) for i in adata_cats]
                    new_cats = list(cats)
                    already_removed = []
                    for match in matches:
                        if (match is not None) and (match.group(1) in cats):
                            new_cats.append(match.group(0))
                            if match.group(1) not in already_removed:
                                new_cats.remove(match.group(1))
                                already_removed.append(match.group(1))
                            
                    cats = new_cats
                    if all([i in adata_cats for i in cats]):
                        found = True
                if not found:
                    raise ValueError("Not all categories in 'cats' are present in the data.")
        
            else:
                raise TypeError("cats must be a list, range, or string.")
            
        return cats
    
    # Validate inputs
    if not isinstance(sep, bool):
        raise TypeError("sep must be a boolean.")

    dtypes = determine_data(data, mod, color, error=True)

    if dtypes['data'] == 'adata':
        adata = data

    else: # mdata
        if mod is not None:
            adata = data[mod]
        else:
            adata = data[dtypes['color']['mod']]

    color_type = dtypes['color']['type']

    if color_type != 'cat':
        raise ValueError("labeled_scatter only works with categorical color.")


    if scatter_params is None:
        scatter_params = {}
    elif sep is True and 'legend' in scatter_params:
        raise ValueError(
            "legend not supported for scatter_params."
        )   
    
    adata_cats = adata.obs[color].cat.categories
    cats = validate_cats(cats, adata_cats)

    if sep:
        fig, axes = subplots(len(cats), ncols=ncols, fss=fss, ar=ar)
        for cat, ax in zip(cats, np.ravel(axes)):
            show_bool = adata.obs[color] == cat
            
            # Plot background cells
            ax = scatter(adata, color=bg_color, size=bg_size, sort_order=False, ax=ax, plot_label=False, 
                    legend=False, **scatter_params)
            
            # Plot highlighted cells
            ax = scatter(adata[show_bool], color=color, size=size, ax=ax, linewidth=0.25, edgecolor=(0.3, 0.3, 0.3, 0.3),
                    legend=False, plot_label=False, **scatter_params)
            
            ax.set_title(cat)
            ax.set_ylabel('')
            ax.set_xlabel('')
    else:
        if 'ax' in scatter_params:
            if any([i is not None for i in (ncols, fss, ar)]):
                warnings.warn(
                    "scatter_params['ax'] is set, but ncols, fss, and ar are also provided. "
                    "These will be ignored."
                )
            ax = scatter_params['ax']
            del(scatter_params['ax'])
        else:
            fig, ax = subplots(1, ncols=ncols, fss=fss, ar=ar)

        bg_scatter_params = deepcopy(scatter_params)
        fg_scatter_params = deepcopy(scatter_params)

        if 'legend' not in fg_scatter_params:
            fg_scatter_params['legend'] = True # The default for scatter_params
        if 'legend_params' not in fg_scatter_params:
            fg_scatter_params['legend_params'] = {}
        if 'show_all_cats' in fg_scatter_params['legend_params']:
            warnings.warn("scatter_params['legend_params']['show_all_cats'] will be overridden by cats.")
        fg_scatter_params['legend_params']['show_all_cats'] = cats

        bg_scatter_params['legend'] = False

        
        # Plot background cells, scatter_params will only be fed to 
        ax = scatter(adata, color=bg_color, size=bg_size, sort_order=False, 
                     plot_label=False, ax=ax, **bg_scatter_params)
        
        # Plot highlighted cells
        for cat in cats:
            show_bool = adata.obs[color] == cat
            ax = scatter(adata[show_bool], color=color, size=size, 
                         plot_label=False, ax=ax, **fg_scatter_params)
        
        # only show title elipsis if the title is too long
        if len(cats) > 1 and len(', '.join(cats)) > 80:
            ax.set_title(', '.join(cats)[:80] + '...')
        else:
            ax.set_title(', '.join(cats))
        ax.set_ylabel('')
        ax.set_xlabel('')
        axes = [ax]
    
    return axes


def plot_features(data: Union[sc.AnnData, md.MuData],
                  features: Union[List[str], Tuple[str, ...], np.ndarray, pd.Series, Dict[str, Any]],
                  fss: float = 4,
                  ar: float = 1,
                  ncols: int = 4,
                  **kwargs) -> Union[Tuple[plt.Figure, np.ndarray], Dict[str, Tuple[plt.Figure, np.ndarray]]]:
    """
    Plots a grid of scatter plots for a set of features.

    This is a convenience function to quickly visualize the expression or
    value of multiple features across an embedding.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The input data object.
    features : list, tuple, np.ndarray, pd.Series, or dict
        The features to plot. Can be a simple list of feature names or a
        dictionary where keys are titles for sets of features.
    mod : str, optional
        The modality to use if `data` is a MuData object.
    fss : float, optional
        Figure size scaler for the subplot grid.
    ar : float, optional
        Aspect ratio for the subplots.
    ncols : int, optional
        Number of columns in the subplot grid.
    **kwargs
        Additional keyword arguments passed to `svc.pl.scatter`.

    Returns
    -------
    (plt.Figure, np.ndarray) or dict
        If `features` is a list, returns a tuple of the Figure and an array
        of Axes. If `features` is a dictionary, returns a dictionary mapping
        feature set names to (Figure, Axes) tuples.
    """

    def _plotter(feature_set):
        fig, axes = subplots(len(feature_set), ncols=ncols, fss=fss, ar=ar)
        for color, ax in zip(feature_set, np.ravel(axes)):
            ax = scatter(data, color=color, ax=ax, **kwargs)
        return (fig, axes)
    
    if isinstance(features, dict):
        figs_axes = dict()
        for feature_set_name in features:
            feature_set = features[feature_set_name]
            figs_axes[feature_set_name] = _plotter(feature_set)

    elif is_listlike(features):
        feature_set = features
        if isinstance(feature_set, pd.Series):
            feature_set = pd.Series.values
        figs_axes = _plotter(feature_set)
    else:
        raise ValueError('Features must be a dict or list-like')
    return figs_axes

