# Built-ins
from pathlib import Path
from numbers import Number
from typing import (
    Optional, Union, Dict, Callable, Any, List, Tuple
)
import warnings

# Standard libs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Single-cell libs
import mudata as md

# Survey libs
from survey.singlecell.plotting import scatter, get_plot_data, get_plotting_configs
from survey.singlecell.decorate import (
    decorate_scatter, get_add_legend_pm, 
    get_text_position_vals, get_add_plotlabel_pm
)
from survey.singlecell.datatypes import determine_data
from survey.spatial.core import validate_chipnums, validate_spatial_mdata
from survey.genutils import get_config, get_mask, is_listlike, normalize
from survey.genplot import subplots
from survey.singlecell.meta import get_cat_dict
from survey.spatial.segutils import get_seg_keys

AxisLimits = Tuple[Tuple[float, float], Tuple[float, float]]
Point = Tuple[float, float]
PointList = List[Point]
HoodMapCircleSize = Optional[Union[Number, Dict[int, Number]]]


def get_chip_mdata(mdata: md.MuData, 
                   chipnum: int, 
                   subset: Optional[Dict] = None) -> md.MuData:
    """
    Filters a MuData object to a specific chip and optional subset criteria.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing spatial data.
    chipnum : int
        The chip number to filter by.
    subset : dict, optional
        A dictionary of additional subsetting criteria, where keys are column
        names in `mdata.obs` and values are the values to keep.

    Returns
    -------
    tuple
        A tuple containing:
            - chip: The chip object corresponding to `chipnum`.
            - masked_mdata: The subsetted MuData object.

    Raises
    ------
    ValueError
        If the chip key property column is not found in `mdata.obs` or if the
        subset results in zero cells.
    """
    

    # Get chip parameters
    chipset = mdata['xyz'].uns['survey']
    chip = chipset.chips[chipnum]

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

    ## Quick check to ensure subset results in at least 1 cell
    if mask.sum() == 0:
        raise ValueError("The subset provided results in zero cells. Please check the subset criteria.")

    masked_mdata = mdata[mask]

    return chip, masked_mdata


def get_lerper(lims1: AxisLimits, 
               lims2: AxisLimits) -> Callable[[Point], Point]:
    """
    Creates a linear interpolator (lerper) function to map points from one coordinate system to another.

    Parameters
    ----------
    lims1 : AxisLimits
        The axis limits of the source coordinate system, as a tuple of ((xmin, xmax), (ymin, ymax)).
    lims2 : AxisLimits
        The axis limits of the target coordinate system, as a tuple of ((xmin, xmax), (ymin, ymax)).

    Returns
    -------
    Callable[[Point], Point]
        A function that takes a point (x, y) in the source coordinate system and
        returns the corresponding point in the target coordinate system.

    Raises
    ------
    ValueError
        If the span of the source coordinate system is zero.
    """
    
    # Unpack the limits for easier access
    (xmin1, xmax1), (ymin1, ymax1) = lims1
    (xmin2, xmax2), (ymin2, ymax2) = lims2

    # Calculate the span (range) of each axis
    x_span1 = xmax1 - xmin1
    y_span1 = ymax1 - ymin1
    x_span2 = xmax2 - xmin2
    y_span2 = ymax2 - ymin2

    # Check for division by zero
    if x_span1 == 0 or y_span1 == 0:
        raise ValueError("Input coordinate system limits cannot have zero span.")
    
    def lerper(point: Point) -> Point:
        x, y = point
        new_x = xmin2 + ((x - xmin1) / x_span1) * x_span2
        new_y = ymin2 + ((y - ymin1) / y_span1) * y_span2
        return (new_x, new_y)
    
    return lerper


def arrplot(mdata: md.MuData,
            chipnum: int,
            color: Optional[str] = None,
            subset: Optional[Dict] = None,
            ax: Optional[plt.Axes] = None,
            fss: int = 10,
            units: str = 'm',
            borders: bool = False,
            walls: bool = False,
            wells: Optional[Union[Dict, pd.Series, Callable]] = None,
            return_welldata: bool = False,
            thresh: Optional[Number] = None,
            layer: Optional[str] = None,
            cmap: Optional[mpl.colors.Colormap] = None,
            norm: Optional[mpl.colors.Normalize] = None,
            img: Optional[Tuple[Path, int]] = None,
            plot_label: bool = True,
            plot_label_params: Optional[Dict[str, Any]] = None,
            cbar: bool = True,
            cbar_params: Optional[Dict[str, Any]] = None,
            invert: bool = False) -> plt.Axes:
    """
    Plots array-based spatial features for a single chip.

    This function visualizes elements of the array itself, such as the wells,
    walls, and borders. It can also overlay a tissue image.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing spatial data.
    chipnum : int
        The chip number to plot.
    color : str, optional
        The column in `mdata.obs` or `mdata.obsm` to use for coloring wells.
    subset : dict, optional
        A dictionary of subsetting criteria.
    ax : plt.Axes, optional
        An existing Axes to plot on.
    fss : int, default 10
        Figure size scale for the plot, fed directly to `survey.genplot.subplots` if 
        `ax` is not provided.
    units : {'m', 'w'}, default 'm'
        The units for the plot coordinates. 'm' for metric (microns), 'w' for well units.
    borders : bool or color, default False
        If True, draws borders around the wells. Can also be a color string.
    walls : bool or color, default False
        If True, draws the walls of the array. Can also be a color string.
    wells : dict, pd.Series, or callable, optional
        How to color the wells. Can be a dictionary mapping well IDs to colors,
        a Series, or a function to apply to numeric `color` data.
    return_welldata : bool, default False
        If True, returns the computed well-level data that was plotted, along with the Axes.
    thresh : Number, optional
        A threshold for the number of cells required to show a well.
    layer : str, optional
        The layer in the modality to use for `color`.
    cmap : mpl.colors.Colormap, optional
        The colormap for numeric `color` data.
    norm : mpl.colors.Normalize, optional
        The normalization for numeric `color` data.
    img : tuple of (Path, int), optional
        A tuple containing the path to the image directory and the image index to display.
    plot_label : bool, default True
        If True, adds a label to the plot indicating the `color` variable.
    plot_label_params : dict, optional
        Parameters for the plot label.
    cbar : bool, default True
        If True, adds a colorbar to the plot for numeric `color` data.
    cbar_params : dict, optional
        Parameters for the colorbar.
    invert : bool, default False
        If True, inverts the plot colors (e.g., for a dark background).
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plt.Axes
        The Axes object containing the plot.
    """
    
    def _add_img(ax, chip, img, lims, units):
        
        img_arg_is_valid = isinstance(img, tuple) and len(img) == 2 and isinstance(img[0], Path) and isinstance(img[1], int)
        if not img_arg_is_valid:
            raise ValueError("Param `img` must be a tuple of (img_prefix: Path, idx: int).")
        
        tissue_img = chip.imgs[img[1]]
        img_full_path = img[0] / tissue_img.fn
        if not img_full_path.exists():
            raise ValueError(f"Image file {tissue_img.fn} not found in provided img_prefix directory.")
        img_extent = tissue_img.extent

        img = mpimg.imread(img_full_path)
        if units == 'm':
            img = img[::-1] # required because of how we set the ylim later
        elif units == 'w':
            pass
        else:
            raise ValueError("Invalid units. Only 'm' or 'w' is supported.")
        
        height, width, _ = img.shape

        xrange = lims['x'][1] - lims['x'][0]
        yrange = lims['y'][1] - lims['y'][0]

        if units == 'm':
            if img_extent == 'auto': # The image was adjusted in Photoshop be the exact right size and shape
                # Slight discrepancies between the way the Photoshop image appears when overlaid with the grid
                # generated in make_imgs.ipynb and the way it's plotted in Python, determined that we 
                # need to adjust the size by 0.5% through trial and error
                left, wscale, bottom, hscale = lims['x'][0], (xrange*0.995)/width, lims['y'][1], (yrange*0.995)/height
            else: # The image was adjusted in Python to get the right size and shape, should be a list of 4 floats
                # The image was plotted through trial and error in Python to get the right size and shape
                # Display the image on the axes, scaled up
                left, wscale, bottom, hscale = img_extent
            extent = [left, width*wscale, bottom, -height*hscale] # negative required because of how we set y lim 
        elif units == 'w':
            extent = [lims['x'][0], lims['x'][1], lims['y'][0], lims['y'][1]]
        else:
            raise ValueError("Invalid units. Only 'm' or 'w' is supported.")

        ax.imshow(img, extent=extent)

        return ax
    

    def _add_borders(borders, chip, ax, lerper):

        verts = {id: [lerper(p) for p in v] for id, v in chip.array.verts.items()}

        if borders is True:
            wall_border_color = (0, 0, 0, 0.3)
        elif mpl.colors.is_color_like(borders):
            wall_border_color = borders
        else:
            raise ValueError('Param `wall_borders` must be True or a valid color.')
        for vert in verts.values():
            poly = mpl.patches.Polygon(vert, closed=True, facecolor=(1, 1, 1, 0), edgecolor=wall_border_color, linewidth=1)
            ax.add_patch(poly)
        return ax
    

    def _add_walls(walls, chip, ax, lerper):
        if walls is True:
            walls_color = (0.3, 0.3, 0.3, 0.3)
        elif mpl.colors.is_color_like(walls):
            walls_color = walls
        else:
            raise ValueError('Param `walls` must be True or a valid color.')

        wall_verts = [[lerper(p) for p in v] for v in chip.array.get_wall_verts()]

        for verts in wall_verts:
            poly = mpl.patches.Polygon(verts, closed=True, facecolor=walls_color, edgecolor=None, linewidth=0)
            ax.add_patch(poly)
        return ax
    

    def _add_wells(masked_mdata, wells, thresh, chip, ax, cmap, norm, color, layer, lims, dtypes, cbar, plot_label, configs, lerper):

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

            if (dtypes['color']['type'] == 'num') or (color is None and pd.api.types.is_numeric_dtype(pd.Series(wells))):
                if norm is None:
                    norm = mpl.colors.Normalize(vmin=wells.min(), vmax=wells.max())
                well_colors = wells.apply(lambda x: cmap(norm(x)))

            elif pd.Series(wells).apply(mpl.colors.is_color_like).all():
                well_colors = wells
            else:
                raise ValueError("If `wells` is a dict, all values must be valid "
                                 "colors or mappable to colors.")
            
            verts = {id: [lerper(p) for p in v] for id, v in chip.array.verts.items() if id in wells.index}
                
            for id, fc in well_colors.items():
                poly = mpl.patches.Polygon(verts[id], closed=True, facecolor=fc,
                                           edgecolor=None, linewidth=0)
                ax.add_patch(poly)
            
            if cbar and dtypes['color']['type'] == 'num':
                center = np.array([center, center]).T
                scpc = ax.scatter(*center, c=[wells.min(), wells.max()], cmap=cmap, norm=norm, s=0)
                ax = decorate_scatter(ax, config=configs['cbar'], plot_type='cbar', scpc=scpc, fig=ax.figure)
            
            if plot_label:
                ax = decorate_scatter(ax, config=configs['plot_label'], plot_type='plot_label', label=color)
        else:
            raise ValueError('Param `wells` must be a callable function, a dict, or a pd.Series')
        return ax, wells
    
    # Get configs
    configs = get_plotting_configs(
        plot_type='scatter', # because wells are plotted like scatter
        plot_label_params=plot_label_params,
        cbar_params=cbar_params,
        invert=invert
    )

    basis = 'survey'

    # Validate inputs
    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    chipnums = validate_chipnums(chipset, chipnum)
    if len(chipnums) > 1:
        raise ValueError("Only one chip number is allowed.")
    chipnum = chipnums[0]

    chip, masked_mdata = get_chip_mdata(mdata, chipnum, subset=subset)

    if units == 'm':
        lims = chip.array.lims
        def lerper(p):
            """An identity function that returns the input unchanged."""
            return p
    elif units == 'w':
        xy = chip.array.wells[['x', 'y']]
        wall_dist = (chip.array.w/chip.array.pitch)
        half_well_dist = 0.5*(chip.array.s/chip.array.pitch)
        lims = dict(zip(['x', 'y'], zip(xy.min() - half_well_dist - wall_dist, xy.max() + half_well_dist + wall_dist)))
        lerper = get_lerper((chip.array.lims['x'], chip.array.lims['y']), (lims['x'], lims['y']))
    else:
        raise ValueError("Invalid units. Only 'm' or 'w' is supported.")

    # Get Axes object and set limits
    if ax is None:
        ar = (np.subtract(*lims['x'])/np.subtract(*lims['y']))
        fig, ax = subplots(1, fss=fss, ar=ar)

    ax.set_xlim(lims['x'])
    ax.set_ylim(lims['y'])
    ax.grid(False)

    # Determine data types for coloring, check cmap if numeric
    dtypes = determine_data(masked_mdata, color=color, basis=basis)

    # Avoid only checking cmap if color is None, because wells can be a dict of numbers
    # for which we'll still need a cmap to map to colors; this is only checked inside _add_wells
    # # if dtypes['color']['type'] == 'num':
    if cmap is None:
        cmap = plt.get_cmap('viridis')
    elif isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except ValueError:
            raise ValueError(f"Param `cmap` string '{cmap}' is not a valid matplotlib colormap name.")
    elif not isinstance(cmap, mpl.colors.Colormap):
        raise ValueError("Param `cmap` must be a valid matplotlib colormap instance.")

    # Add all the requested elements to the plot
    if img is not None:
        ax = _add_img(ax, chip, img, lims, units)

    if borders:
        ax = _add_borders(borders, chip, ax, lerper)

    if walls:
        ax = _add_walls(walls, chip, ax, lerper)

    if wells is not None:
        ax, welldata = _add_wells(masked_mdata, wells, thresh, chip, ax, 
                                  cmap, norm, color, layer, lims, dtypes, 
                                  cbar, plot_label, configs, lerper)

    # Clean up the plot, make similar to svc.pl.scatter
    if invert:
        ax.set_facecolor('black')

    ax.set_xlabel(f'{basis.upper()}1')
    ax.set_ylabel(f'{basis.upper()}2')
    
    # Remove grid, ticks, and tick labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if return_welldata:
        return ax, welldata
    else:
        return ax


def cellmap(mdata: md.MuData,
            chipnum: int,
            color: Optional[str] = None,
            subset: Optional[Dict] = None,
            ax: Optional[plt.Axes] = None,
            fss: int = 10,
            units: str = 'm',
            layer: Optional[str] = None,
            cmap: Optional[mpl.colors.Colormap] = None,
            norm: Optional[mpl.colors.Normalize] = None,
            **kwargs: Any) -> plt.Axes:
    """
    Plots cell-level data on a spatial array plot.

    This function is used to visualize individual cells on the chip, colored by
    specified features. It serves as a spatial equivalent of a scatter plot.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing spatial data.
    chipnum : int
        The chip number to plot.
    color : str, optional
        The column in `mdata.obs` or `mdata.obsm` to use for coloring cells.
    subset : dict, optional
        A dictionary of subsetting criteria.
    ax : plt.Axes, optional
        An existing Axes to plot on.
    fss : int, default 10
        Figure size scale for the plot, fed directly to `survey.genplot.subplots` if
        `ax` is not provided.
    units : {'m'}, default 'm'
        The units for the plot coordinates. Currently only 'm' (metric) is supported.
    layer : str, optional
        The layer in the modality to use for `color`.
    cmap : mpl.colors.Colormap, optional
        The colormap for numeric `color` data.
    norm : mpl.colors.Normalize, optional
        The normalization for numeric `color` data.
    **kwargs
        Additional keyword arguments passed to `survey.singlecell.plotting.scatter`.

    Returns
    -------
    plt.Axes
        The Axes object containing the plot.

    Raises
    ------
    NotImplementedError
        If `units` is not 'm'.
    """
    

    def _add_cells(masked_mdata, color, ax, cmap, norm, dtypes, basis, layer, **kwargs):
        if dtypes['color']['type'] == 'cat':
            cats_show = masked_mdata[dtypes['color']['mod']].obs[color].unique()
        else:
            cats_show = None

        default_legend_params = {'loc': 'lower right', 'bbox_to_anchor': (1, 0), 'show_all_cats': cats_show}
        if 'legend_params' in kwargs:
            kwargs['legend_params'] = get_config(kwargs['legend_params'], default_legend_params)

        supplied_scatter_params = ['data', 'color', 'basis', 'ax', 'plot_data', 'layer']
        # overlap_with_mpl_scatter = ['cmap', 'norm', ''] # what's with the empty string?
        overlap_with_mpl_scatter = ['cmap', 'norm'] 

        protected_scatter_params = supplied_scatter_params + overlap_with_mpl_scatter
        scatter_params = get_config(kwargs, {}, protected=protected_scatter_params)

        mpl_scatter_params = {'cmap': cmap, 'norm': norm}

        ax = scatter(data=masked_mdata, color=color, basis=basis, ax=ax, plot_data=None, layer=layer,
                     **scatter_params, **mpl_scatter_params)


        return ax
    
    basis = 'survey'

    # Validate inputs
    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    chipnums = validate_chipnums(chipset, chipnum)
    if len(chipnums) > 1:
        raise ValueError("Only one chip number is allowed.")
    chipnum = chipnums[0]
    lims = chipset.chips[chipnum].array.lims


    if units != 'm':
        # The only reason units=='w' is required is for piemap() charts; I really don't envision cells being
        # plotted on top of pie charts being a common use case, if ever, so will table this for now
        # Still choosing to leave the units param here for consistency with other spatial plotting functions
        raise NotImplementedError("Only units='m' is currently supported in cellmap().")
    
    chip, masked_mdata = get_chip_mdata(mdata, chipnum, subset=subset)
    
    # Get Axes object and set limits
    if ax is None:
        ar = (np.subtract(*lims['x'])/np.subtract(*lims['y']))
        fig, ax = subplots(1, fss=fss, ar=ar)

    ax.set_xlim(lims['x'])
    ax.set_ylim(lims['y'])
    ax.grid(False)

    # Determine data types for coloring, check cmap if numeric
    dtypes = determine_data(masked_mdata, color=color, basis=basis)

    if dtypes['color']['type'] == 'num':
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        elif isinstance(cmap, str):
            try:
                cmap = plt.get_cmap(cmap)
            except ValueError:
                raise ValueError(f"Param `cmap` string '{cmap}' is not a valid matplotlib colormap name.")
        elif not isinstance(cmap, mpl.colors.Colormap):
            raise ValueError("Param `cmap` must be a valid matplotlib colormap instance.")

    ax = _add_cells(masked_mdata, color, ax, cmap, norm, dtypes, basis, layer, **kwargs)
    
    return ax


def hoodmap(mdata: md.MuData,
            chipnum: int,
            color: Optional[str] = None,
            subset: Optional[Dict] = None,
            ax: Optional[plt.Axes] = None,
            fss: int = 10,
            units: str = 'w',
            invert: bool = False,
            size: HoodMapCircleSize = None,
            wedgeprops: Optional[Dict[str, Any]] = None,
            circleprops: Optional[Dict[str, Any]] = None,
            legend: bool = True,
            legend_params: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> plt.Axes:
    """
    Plots neighborhood-level data on a spatial array plot using pie charts or circles.

    This function visualizes aggregate information for each well (neighborhood),
    such as the composition of cell types within the well, represented as pie charts.
    The size of the pies or circles can be scaled by the number of cells in the well.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing spatial data.
    chipnum : int
        The chip number to plot.
    color : str, optional
        The categorical column in `mdata.obs` to use for pie chart segments.
        If None, only circles are plotted.
    subset : dict, optional
        A dictionary of subsetting criteria.
    ax : plt.Axes, optional
        An existing Axes to plot on.
    fss : int, default 10
        Figure size scale for the plot, fed directly to `survey.genplot.subplots` if 
        `ax` is not provided.
    units : {'w'}, default 'w'
        The units for the plot coordinates. Currently only 'w' (well units) is supported.
    invert : bool, default False
        If True, inverts the plot colors (e.g., for a dark background).
    size : float or dict, optional
        Controls the size of the pies/circles.
        - If a float, it's a fraction of the well size (0 to 1).
        - If a dict, it maps cell counts to sizes for dynamic sizing, e.g., `{10: 0.2, 100: 0.8}`.
        - If None, a default size is used.
    wedgeprops : dict, optional
        Properties for the pie chart wedges (see `matplotlib.patches.Wedge`).
    circleprops : dict, optional
        Properties for the circles drawn around/for each well (see `matplotlib.patches.Circle`).
    legend : bool, default True
        If True, adds a legend for the pie chart colors (if `color` is provided).
    legend_params : dict, optional
        Parameters for the legend (if `color` is provided).

    Returns
    -------
    plt.Axes
        The Axes object containing the plot.

    Raises
    ------
    NotImplementedError
        If `units` is not 'w'.
    ValueError
        If `color` is not a categorical column, or if `size` parameters are invalid.
    """
    
    def _check_size(size):
        if size <= 0 or size > 1:
            raise ValueError(
                "Provided sizes in `size` (as float or dict values) should be a "
                "fraction of the well size (i.e. must be > 0 and <= 1.)")
        return

    basis = 'survey'

    if invert:
        default_circle_color = 'white'
        default_circle_edge_color = 'white'
    else:
        default_circle_color = 'black'
        default_circle_edge_color = 'black'

    default_wedgeprops = {'edgecolor': default_circle_edge_color, 'linewidth': 0.5}

    # Validate inputs
    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    chipnums = validate_chipnums(chipset, chipnum)
    if len(chipnums) > 1:
        raise ValueError("Only one chip number is allowed.")
    chipnum = chipnums[0]

    chip, masked_mdata = get_chip_mdata(mdata, chipnum, subset=subset)

    if units != 'w':
        # Pies take too long to render in 'm' units, not sure why, something about the large coordinate values in 
        # `center`. Instead, we convert to 'w' units internally and make other functions accept it as well.
        # Eventually, this unit conversion should be built into the Array class (with get_lerper() function and
        # related `lims` code in the individual plotting functions). Still choosing to leave the units param here 
        # for consistency with other spatial plotting functions.
        raise NotImplementedError("Only units='w' is currently supported in hoodmap().")
    else:
        xy = chip.array.wells[['x', 'y']]
        wall_dist = (chip.array.w/chip.array.pitch)
        half_well_dist = 0.5*(chip.array.s/chip.array.pitch)
        lims = dict(zip(['x', 'y'], zip(xy.min() - half_well_dist - wall_dist, xy.max() + half_well_dist + wall_dist)))
    
    # Get Axes object and set limits
    if ax is None:
        ar = (np.subtract(*lims['x'])/np.subtract(*lims['y']))
        fig, ax = subplots(1, fss=fss, ar=ar)

    ax.set_xlim(lims['x'])
    ax.set_ylim(lims['y'])
    ax.grid(False)

    if color is None:
        # User wants to plot only circles showing neighborhood size without pie charts
        plot_pies = False
        if not isinstance(size, dict):
            raise ValueError(
                "If `color` is None, only neighborhood size is plotted. Therefore, "
                "`size` must be a dict of {int: float} pairs.")
        if wedgeprops is not None:
            warnings.warn(
                "Param `wedgeprops` is ignored when `color` is None.",
                UserWarning)
        if circleprops is None:
            circleprops = {'facecolor': default_circle_color, 'edgecolor': default_circle_edge_color, 'linewidth': 1}
    else:
        plot_pies = True

    df1 = chip.get_welldata()[['arr-x', 'arr-y']]
    if color is None:
        df2 = masked_mdata['xyz'].obs[['id']].groupby(['id'], observed=True).size().to_frame().rename(columns={'id': 'count'})
    else:
        df2 = masked_mdata['xyz'].obs[['id', color]].groupby(['id'], observed=True).value_counts().unstack().rename_axis(columns=None)
        cdict = get_cat_dict(mdata['xyz'], key=color, prop='color')

    max_size = (chip.array.s/chip.array.pitch)/2 # default radius is half the side length of the well

    if size is None:
        circlesize = max_size
        sizes = pd.Series(circlesize, index=df2.index) 
    else:
        if isinstance(size, Number):
            _check_size(size)
            circlesize = size * max_size
            sizes = pd.Series(circlesize, index=df2.index) 
        elif isinstance(size, dict):
            if len(size) != 2:
                raise ValueError("If `size` is a dict, it must contain exactly two entries.")
            if any(not isinstance(k, int) or k <= 0 for k in size.keys()):
                raise ValueError("If `size` is a dict, both keys must be an integer number of cells.")
            for k in size:
                _check_size(size[k])
            well_cell_counts = masked_mdata['xyz'].obs.groupby('id').size()
            numcells = tuple(size.keys())
            sizes = normalize(well_cell_counts, clip=numcells, lower=size[numcells[0]]*max_size, upper=size[numcells[1]]*max_size)
        else:
            raise ValueError("Param `size` must be None, a number, or a dict of {int: float} pairs.")
    
    df2['sizes'] = sizes

    piedf = pd.concat([df1, df2], axis=1, join='inner').set_index(['arr-x', 'arr-y', 'sizes'], append=True)
    

    if plot_pies:
        # Determine data types for coloring, check cmap if numeric
        dtypes = determine_data(masked_mdata, color=color, basis=basis)

        wedgeprops = get_config(wedgeprops, default_wedgeprops, protected=None)

        if dtypes['color']['type'] != 'cat':
            raise ValueError("Param `color` must be a categorical column for hoodmap().")
        
        for i, ((id, x, y, size), row) in enumerate(piedf.iterrows()):
            ax.pie(row.values, center=(x, y), radius=size, colors=[cdict[i] for i in row.index], wedgeprops=wedgeprops)

        if legend:
            legend_params = get_add_legend_pm().get_params(legend_params)
            ax = decorate_scatter(ax, config=legend_params, plot_type='legend', cdict=cdict)

    if circleprops is not None:
        for i, ((id, x, y, size), row) in enumerate(piedf.iterrows()):
            if plot_pies:
                # Make sure facecolor is none so pie chart is visible
                config = get_config(circleprops, {}, protected=['facecolor'])
                config.update(facecolor='none')
            else:
                # No pies, user can customize circle appearance as they like
                config = circleprops
            circle = mpl.patches.Circle((x, y), radius=size, **config)
            ax.add_patch(circle)
    
    # Plotting pies messes with the limits, so reset them
    ax.set_xlim(lims['x'])
    ax.set_ylim(lims['y'])
    
    # Clean up the plot, make similar to svc.pl.scatter
    if invert:
        ax.set_facecolor('black')

    ax.set_xlabel(f'{basis.upper()}1')
    ax.set_ylabel(f'{basis.upper()}2')
    
    # Remove grid, ticks, and tick labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return ax


def segplot(mdata: md.MuData,
            group: str,
            chipnum: Optional[Union[int, List[int]]] = None,
            label_positions: Optional[Union[bool, List[str]]] = None,
            ax: Optional[plt.Axes] = None,
            fss: int = 10,
            ar: int = 1,
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
    fss : int, default 10
        Figure size scale for the plots, fed directly to `survey.genplot.subplots` if
        `ax` is not provided.
    ar : float, optional
        Aspect ratio for the plots, fed directly to `survey.genplot.subplots` if
        `ax` is not provided. 
    **kwargs
        Additional keyword arguments passed to `arrplot`.

    Returns
    -------
    list of plt.Axes
        A list of the Axes objects containing the plots.

    Raises
    ------
    ValueError
        If `group` is not a valid segmentation key or if `ax` is provided with
        an incorrect shape for the number of chips.

    Notes
    -----
    Unlike other svp.plotting functions, this function calls `arrplot` internally
    to create the plots, as segmentation data is visualized by coloring wells. There is
    no need to separately plot array elements before calling this function, as the kwargs
    are passed directly to `arrplot`.
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
        fig, axes = subplots(len(chipnums), fss=fss, ar=ar, as_seq=True)
    elif isinstance(ax, mpl.axes.Axes) and len(chipnums) == 1:
        axes = [ax]
    elif not is_listlike(ax) or len(ax) < len(chipnums):
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
        ax = arrplot(mdata, chipnum, wells=well_dict, subset=subset, ax=ax, **kwargs)

        for label, position in zip([chipnum, group], label_positions):
            pos, ha, va = get_text_position_vals(position)
            plot_label_params = get_add_plotlabel_pm().get_params({'pos': pos, 'ha': ha, 'va': va})
            ax = decorate_scatter(ax, config=plot_label_params, plot_type='plot_label', label=label)
        
        for label, position in zip([chipnum, group], label_positions):
            legend_params = get_add_legend_pm().get_params(legend_params)
            ax = decorate_scatter(ax, config=legend_params, plot_type='legend', cdict=cdict)

    return axes

