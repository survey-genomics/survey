# Built-ins
from pathlib import Path
from numbers import Number
from typing import Optional, Union, List, Tuple, Dict, Any, Callable

# Standard libs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Single-cell libs
import mudata as md

# Survey libs
from survey.singlecell.plotting import scatter, get_plot_data
from survey.singlecell.decorate import (
    decorate_scatter, get_add_legend_pm, 
    get_text_position_vals, get_add_plotlabel_pm
)
from survey.singlecell.datatypes import determine_data
from survey.spatial.core import validate_chipnums, validate_spatial_mdata
from survey.genutils import get_config, get_mask, is_listlike
from survey.genplot import subplots
from survey.singlecell.meta import get_cat_dict
from survey.spatial.segutils import get_seg_keys


def survey_plot(mdata: md.MuData,
                chipnum: int,
                color: Optional[str] = None,
                subset: Optional[Dict] = None,
                ax: Optional[plt.Axes] = None,
                fss: float = 10,
                borders: bool = False,
                walls: bool = False,
                wells: Optional[Union[Dict, pd.Series, Callable]] = None,
                thresh: Optional[Number] = None,
                layer: Optional[str] = None,
                cmap: Optional[mpl.colors.Colormap] = None,
                norm: Optional[mpl.colors.Normalize] = None,
                img: Optional[Tuple[Path, int]] = None,
                **kwargs: Any) -> plt.Axes:
    """
    Creates a spatial plot for a single chip from a Survey experiment.

    This function is the main entry point for visualizing spatial data. It can
    overlay cell scatter plots, well-based aggregations, tissue images, and
    array geometry.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing the spatial experiment data.
    chipnum : int
        The number of the chip to plot.
    color : str, optional
        The key for coloring cells in a scatter plot.
    subset : dict, optional
        A dictionary to subset the cells before plotting (e.g., `{'leiden': ['1', '2']}`).
    ax : plt.Axes, optional
        An existing Matplotlib Axes to plot on.
    fss : float, optional
        Figure size scale factor.
    borders : bool or color-like, optional
        If True, draws black borders around each well. Can also be a color.
    walls : bool or color-like, optional
        If True, fills the space between wells with a semi-transparent gray.
        Can also be a color.
    wells : dict, pd.Series, or callable, optional
        Determines how to color entire wells. Can be a mapping from well ID to
        color, or a function to aggregate cell data within each well.
    thresh : Number, optional
        A minimum number of cells required for a well to be colored when using `wells`.
    layer : str, optional
        The data layer to use for expression-based coloring.
    cmap : mpl.colors.Colormap, optional
        Colormap for numerical data (either for cells or wells).
    norm : mpl.colors.Normalize, optional
        A Matplotlib Normalize instance for scaling numerical data.
    img : tuple of (Path, int), optional
        A tuple containing the path to the image directory and the index of the
        image to display from the chip's image list.
    **kwargs
        Additional keyword arguments passed to the underlying `scatter` function.

    Returns
    -------
    plt.Axes
        The Axes object containing the plot.
    """
    
    def _add_img(ax, chip, img):
        
        img_arg_is_valid = isinstance(img, tuple) and len(img) == 2 and isinstance(img[0], Path) and isinstance(img[1], int)
        if not img_arg_is_valid:
            raise ValueError("Param `img` must be a tuple of (img_prefix: Path, idx: int).")
        
        tissue_img = chip.imgs[img[1]]
        img_full_path = img[0] / tissue_img.fn
        if not img_full_path.exists():
            raise ValueError(f"Image file {tissue_img.fn} not found in provided img_prefix directory.")
        img_extent = tissue_img.extent

        img = mpimg.imread(img_full_path)[::-1] # required because of how we set the ylim later
        height, width, _ = img.shape

        xrange = chip.array.lims['x'][1] - chip.array.lims['x'][0]
        yrange = chip.array.lims['y'][1] - chip.array.lims['y'][0]

        if img_extent == 'auto': # The image was adjusted in Photoshop be the exact right size and shape
            # Slight discrepancies between the way the Photoshop image appears when overlaid with the grid
            # generated in make_imgs.ipynb and the way it's plotted in Python, determined that we 
            # need to adjust the size by 0.5% through trial and error
            left, wscale, bottom, hscale = 0, (xrange*0.995)/width, 0, (yrange*0.995)/height
        else: # The image was adjusted in Python to get the right size and shape, should be a list of 4 floats
            # The image was plotted through trial and error in Python to get the right size and shape
            # Display the image on the axes, scaled up
            left, wscale, bottom, hscale = img_extent
        

        ax.imshow(img, extent=[left, width*wscale, -bottom, -height*hscale]) # negative required because of how we set y lim later

        return ax
    
    
    def _add_borders(borders, chip, ax):
        if borders is True:
            wall_border_color = (0, 0, 0, 0.3)
        elif mpl.colors.is_color_like(borders):
            wall_border_color = borders
        else:
            raise ValueError('Param `wall_borders` must be True or a valid color.')
        for vert in chip.array.verts.values():
            poly = mpl.patches.Polygon(vert, closed=True, facecolor=(1, 1, 1, 0), edgecolor=wall_border_color, linewidth=1)
            ax.add_patch(poly)
        return ax
    
    
    def _add_walls(walls, chip, ax):
        if walls is True:
            walls_color = (0.3, 0.3, 0.3, 0.3)
        elif mpl.colors.is_color_like(walls):
            walls_color = walls
        else:
            raise ValueError('Param `walls` must be True or a valid color.')
        for verts in chip.array.get_wall_verts():
            poly = mpl.patches.Polygon(verts, closed=True, facecolor=walls_color, edgecolor=None, linewidth=0)
            ax.add_patch(poly)
        return ax
    
    
    def _add_wells(masked_mdata, wells, thresh, chip, ax, cmap, norm, color, layer, lims, dtypes):

        center = [np.mean(v) for k, v in lims.items()]

        sizes = masked_mdata['xyz'].obs.groupby('id', observed=True).size()

        if thresh == 0:
            thresh = None
        if thresh is None:
            wells_show = sizes.index
        elif not isinstance(thresh, Number):
            raise ValueError("Param `thresh` must None or a number.")
        elif thresh > 0:
            wells_show = sizes[sizes >= thresh].index
            

        if isinstance(wells, dict):
            wells = pd.Series(wells)
        elif callable(wells):
            if dtypes['color']['type'] != 'num':
                raise ValueError("If `wells` is a callable function, `color` must be a numeric column.")
            plot_df, _, _ = get_plot_data(masked_mdata, color=color, basis='survey', layer=layer)
            wells = plot_df.join(masked_mdata['xyz'].obs[['id']]).groupby('id', observed=True)['c'].apply(wells)

        if isinstance(wells, pd.Series):
            if not set(wells_show).issubset(wells.index):
                raise ValueError(
                    "If `wells` is a dict, it must contain all well ids present in "
                    "mdata['xyz'].obs['id'].unique() as keys.")
            
            if thresh is None:
                pass
            else:
                wells = wells.loc[wells_show]

            if dtypes['color']['type'] == 'num':
                if norm is not None:
                    wells_norm = wells.apply(norm)
                    well_colors = wells_norm.apply(cmap)
                else:
                    well_colors = wells.apply(cmap)
            elif pd.Series(wells).apply(mpl.colors.is_color_like).all():
                well_colors = wells
            else:
                raise ValueError("If `wells` is a dict, all values must be valid "
                                 "colors or mappable to colors.")

                
            for id, fc in well_colors.items():
                poly = mpl.patches.Polygon(chip.array.verts[id], closed=True, facecolor=fc,
                                           edgecolor=None, linewidth=0)
                ax.add_patch(poly)
            
            if dtypes['color']['type'] == 'num':
                center = np.array([center, center]).T
                scpc = ax.scatter(*center, c=[wells.min(), wells.max()], cmap=cmap, norm=norm, s=0)
                config = {'label_size': 10, 'label_color': 'white'}
                ax = decorate_scatter(ax, config=config, plot_type='cbar', scpc=scpc, fig=ax.figure)
        else:
            raise ValueError('Param `wells` must be a callable function, a dict, or a pd.Series')
        return ax
    
    
    def _add_cells(masked_mdata, color, ax, cmap, norm, dtypes, layer, **kwargs):
        if dtypes['color']['type'] == 'cat':
            cats_show = masked_mdata[dtypes['color']['mod']].obs[color].unique()
        else:
            cats_show = None

        default_legend_params = {'loc': 'lower right', 'bbox_to_anchor': (1, 0), 'show_all_cats': cats_show}
        if 'legend_params' in kwargs:
            kwargs['legend_params'] = get_config(kwargs['legend_params'], default_legend_params)

        supplied_scatter_params = ['data', 'color', 'basis', 'ax', 'plot_data', 'layer']
        overlap_with_mpl_scatter = ['cmap', 'norm', '']

        protected_scatter_params = supplied_scatter_params + overlap_with_mpl_scatter
        scatter_params = get_config(kwargs, {}, protected=protected_scatter_params)

        mpl_scatter_params = {'cmap': cmap, 'norm': norm}
        
        ax = scatter(data=masked_mdata, color=color, basis='survey', ax=ax, plot_data=None, layer=layer, 
                            **scatter_params, **mpl_scatter_params)

        return ax

    # Validate inputs
    
    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    chipnums = validate_chipnums(chipset, chipnum)
    if len(chipnums) > 1:
        raise ValueError("Only one chip number is allowed.")
    chipnum = chipnums[0]

    ## If neither wells nor color is provided, set color to 'k'
    if wells is None and color is None:
        color = 'k'

    # Get the chipset, the specific chip, and axes limits for its Array
    chipset = mdata['xyz'].uns['survey']
    chip = chipset.chips[chipnum]
    lims = chip.array.lims

    # Get chip_key_prop column in mdata.obs
    chip_key_prop_col = None
    for i in mdata.obs.columns:
        if isinstance(i, str) and i.endswith(chipset.chip_key_prop):
            chip_key_prop_col = i
            break
    
    if chip_key_prop_col is None:
        # What if user changed the name of the chip_key_prop column?
        # Maybe if its not found, do a deep search (determine_data(color=chip_key_prop)?) to see if its
        # still in any modality, and if not, provide a separate error.
        raise ValueError(
            f"Could not find a column in mdata.obs that ends with '{chipset.chip_key_prop}'."
            f" Please make sure the chip_key_prop from chipset is consistent with the mdata and that"
            f" it has been pulled from any individual modalites to the global .obs."
        )

    # Subset the mdata to only the specified chipnum and any additional subset criteria
    subset = get_config(subset, {chip_key_prop_col: [chipnum]}, protected=chip_key_prop_col)
    mask = get_mask(mdata.obs, subset)

    if mask.sum() == 0:
        raise ValueError("The subset provided results in zero cells to plot. Please check the subset criteria.")

    masked_mdata = mdata[mask]

    # Get Axes object and set limits
    if ax is None:
        ar = (np.subtract(*lims['x'])/np.subtract(*lims['y']))
        fig, ax = subplots(1, ar=ar, fss=fss)

    ax.set_xlim(lims['x'])
    ax.set_ylim(lims['y'])
    ax.grid(False)

    if img is not None:
        ax = _add_img(ax, chip, img)

    if borders:
        ax = _add_borders(borders, chip, ax)

    if walls:
        ax = _add_walls(walls, chip, ax)

    # Determine data types for coloring, check cmap if numeric
    dtypes = determine_data(masked_mdata, color=color, basis='survey')

    if dtypes['color']['type'] == 'num':
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        elif not isinstance(cmap, mpl.colors.Colormap):
            raise ValueError("Param `cmap` must be a valid matplotlib colormap instance.")

    if wells:
        ax = _add_wells(masked_mdata, wells, thresh, chip, ax, cmap, norm, color, layer, lims, dtypes)
        # Just run _add_cells with an empty mdata to propogate other kwargs (e.g. invert)
        remasked_mdata = masked_mdata[[], :]
        if color is None:
            color = 'k'
        kwargs.update({'cbar': False})  # Disable colorbar when plotting wells
        ax = _add_cells(remasked_mdata, color, ax, cmap, norm, dtypes, layer=layer, **kwargs)
    else:
        ax = _add_cells(masked_mdata, color, ax, cmap, norm, dtypes, layer=layer, **kwargs)

    return ax


def plot_seg(mdata: md.MuData,
             group: str,
             chipnum: Optional[Union[int, List[int]]] = None,
             label_positions: Optional[Union[bool, List[str]]] = None,
             ax: Optional[plt.Axes] = None,
             **kwargs: Any) -> List[plt.Axes]:
    """
    Plots spatial segmentation data for one or more chips.

    This function visualizes pre-computed segmentation results, such as tissue
    regions, by coloring the wells of the array plot according to the
    segmentation group.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing the spatial data and segmentation results.
    group : str
        The segmentation key to plot (e.g., 'tissue_type'). This should exist
        as a column in `mdata['xyz'].uns['survey'].chips[chipnum].seg`.
    chipnum : int or list of int, optional
        The chip number(s) to plot. If None, all chips are plotted.
    label_positions : bool or list of str, optional
        Positions for placing chip number and group labels on the plot.
        If False, no labels are shown. If None, defaults to ['upper left', 'upper right'].
    ax : plt.Axes or list of plt.Axes, optional
        An existing Axes or list of Axes to plot on.
    **kwargs
        Additional keyword arguments passed to `survey_plot`.

    Returns
    -------
    list of plt.Axes
        A list of the Axes objects containing the plots.

    Raises
    ------
    ValueError
        If `group` is not a valid segmentation key or if `ax` is provided with
        an incorrect shape for the number of chips.
    """

    if label_positions is False:
        label_positions = []
    elif label_positions is None:
        label_positions = ['upper left', 'upper right']
    else:
        lp = label_positions
        valid_lp = is_listlike(lp) and all([isinstance(pos, str) for pos in lp]) and len(lp) == 2
        if not valid_lp:
            raise ValueError("Param `label_positions` must be a list of positions.")

    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    
    if chipnum is None:
        chipnums = list(chipset.chips.keys())
    else:
        chipnums = validate_chipnums(chipset, chipnum)

    if 'xyz:id' not in mdata.obs:
        raise ValueError(
            "Param `xyz:id` must be present in `mdata.obs`. "
            "Please run mdata.pull_obs('id', mods=['xyz']) to add it.")

    if is_listlike(group):
        raise ValueError("Only one group can be specified.")

    possible_groups = get_seg_keys(mdata)
    if group not in possible_groups:
        raise ValueError(f"Param `group` must be one of {possible_groups}.")

    if ax is None:
        fig, axes = subplots(len(chipnums), fss=10, as_seq=True)
    elif isinstance(ax, mpl.axes.Axes) and len(chipnums) == 1:
        axes = [ax]
    elif not is_listlike(ax) or len(ax) != len(chipnums):
        raise ValueError("Param `ax` must be None, a single Axes object for a single chip, or a list of Axes objects.")
    else:
        axes = ax

    # Setting these for now, maybe in the future make more customizable
    legend_params = {'loc': 'lower right', 'bbox_to_anchor': (1, 0), 'prop': {'size': 8, 'weight': 'bold'}}


    for chipnum, ax in zip(chipnums, axes):

        cdict = get_cat_dict(mdata['xyz'], group, 'color')
        chip = chipset.chips[chipnum]
        well_dict = chip.seg[group].dropna().map(cdict).to_dict()

        subset={'xyz:id': list(well_dict.keys())}
        ax = survey_plot(mdata, chipnum, wells=well_dict, subset=subset, ax=ax, **kwargs)

        for label, position in zip([chipnum, group], label_positions):
            pos, ha, va = get_text_position_vals(position)
            plot_label_params = get_add_plotlabel_pm().get_params({'pos': pos, 'ha': ha, 'va': va})
            ax = decorate_scatter(ax, config=plot_label_params, plot_type='plot_label', label=label)
        
        for label, position in zip([chipnum, group], label_positions):
            legend_params = get_add_legend_pm().get_params(legend_params)
            ax = decorate_scatter(ax, config=legend_params, plot_type='legend', cdict=cdict)

    return axes

