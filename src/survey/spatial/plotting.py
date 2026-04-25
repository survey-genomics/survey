# Built-ins
from pathlib import Path
from numbers import Number
from typing import (
    Optional, Union, Dict, Callable, Any, List, Tuple
)
import warnings
import itertools as it
import uuid

# Standard libs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
import matplotlib.image as mpimg
from scipy.sparse import csr_matrix
from scipy.ndimage import convolve, gaussian_filter, binary_fill_holes
from multiprocessing import Pool

# Single-cell libs
import mudata as md

# Survey libs
from survey.singlecell.plotting import scatter, get_plot_data, get_plotting_configs, Ridge
from survey.singlecell.decorate import (
    decorate_scatter, get_add_legend_pm, get_pm,
    get_text_position_vals, get_add_plotlabel_pm
)
from survey.singlecell.obs import get_obs_df, transfer_obs
from survey.singlecell.datatypes import determine_data
from survey.spatial.core import validate_chipnums, validate_spatial_mdata
from survey.genutils import get_config, get_mask, is_listlike, normalize
from survey.genplot import subplots, create_alpha_cmap
from survey.singlecell.meta import get_cat_dict, reset_meta_keys, add_colors
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
            dilation: Optional[Number] = None,
            plot: bool = True,
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
    dilation : Number, optional
        A percentage to dilate (>100) or shrink (<100) the well squares.
    plot : bool, default True
        If True, perform plotting. Useful for obtaining well-level data without plotting.
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
    ax (if `plot` and not `return_welldata`)
    ax, welldata (if `plot` and `return_welldata`)
    welldata (if not `plot` and `return_welldata`)
    
    ax, (plt.Axes)
        The Axes object containing the plot.
    welldata (pd.DataFrame)
        The well-level data that was plotted.
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
    
    
    def _add_wells(masked_mdata, wells, thresh, chip, ax, cmap, norm, color, 
                   layer, lims, dtypes, cbar, plot_label, configs, lerper, dilation, plot):

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

            if dilation is not None:
                verts = resize_square_vertices(verts, dilation)
            
            if plot:
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

    if not plot and not return_welldata:
        raise ValueError("Specify at least one of `plot` or `return_welldata` as True.")
    
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

    # Handle units and lerper setup
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
    if plot and ax is None:
        ar = (np.subtract(*lims['x'])/np.subtract(*lims['y']))
        fig, ax = subplots(1, fss=fss, ar=ar)

    if plot:
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

    # Add all the requested elements to the plot if plotting is enabled
    if plot:
        if img is not None:
            ax = _add_img(ax, chip, img, lims, units)

        if borders:
            ax = _add_borders(borders, chip, ax, lerper)

        if walls:
            ax = _add_walls(walls, chip, ax, lerper)

    if wells is not None:
        ax, welldata = _add_wells(masked_mdata, wells, thresh, chip, ax, 
                                  cmap, norm, color, layer, lims, dtypes, 
                                  cbar, plot_label, configs, lerper, dilation, plot)
    if not plot:
        return welldata
    
    # Clean up the plot, make similar to svc.pl.scatter
    if invert:
        ax.set_facecolor('black')

    if plot:
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

        if 'legend_params' in kwargs:
            if 'pos' in kwargs['legend_params']:
                default_legend_params = {'pos': 'BR1', 'show_all_cats': cats_show}
            else:
                default_legend_params = {'loc': 'lower right', 'bbox_to_anchor': (1, 0), 'show_all_cats': cats_show}
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
    elif dtypes['color']['type'] is None:
        color = 'k'

    ax = _add_cells(masked_mdata, color, ax, cmap, norm, dtypes, basis, layer, **kwargs)
    
    return ax


def resize_square_vertices(verts_dict, scale_percent):
    """
    Resize square vertices by a given percentage while maintaining center position.
    
    Parameters
    ----------
    verts_dict : dict
        Dictionary where values are square vertices in order:
        [top_right, bottom_right, bottom_left, top_left]
        Each vertex should be a tuple/list of (x, y) coordinates
    scale_percent : float
        Percentage to scale each side (e.g., 110 = 110% of original size,
        90 = 90% of original size)
        
    Returns
    -------
    dict
        Dictionary with same keys but resized vertices
    """
    
    scale_factor = scale_percent / 100.0
    resized_dict = {}
    
    for key, verts in verts_dict.items():
        verts = np.array(verts)
        
        # Calculate center of the square
        center = verts.mean(axis=0)
        
        # Translate vertices to origin, scale, then translate back
        centered_verts = verts - center
        scaled_verts = centered_verts * scale_factor
        resized_verts = scaled_verts + center
        
        resized_dict[key] = resized_verts
    
    return resized_dict


class Reinforcement:
    """
    A reaction-diffusion model for generating organic-looking patterns.
    
    This class implements a reinforcement-based pattern formation algorithm
    using partial differential equations and diffusion processes.

    Based on the model described in MALHEIROS, FENSTERSEIFER, and WALTER (2020):
    https://mgmalheiros.github.io/research/leopard/leopard-2020-preprint.pdf
    https://github.com/CorentinDumery/cow-tex-generator/tree/main
    
    Parameters
    ----------
    shape : int or tuple of int, default=50
        Initial shape of the simulation grid. If int, creates a square grid.
    speed : int, default=100
        Simulation speed parameter affecting time step size.
    ini_c : float, default=3
        Initial concentration value for the grid.
    var_c : float, default=2
        Variance added to initial concentration (0 for uniform).
    scale : float, default=1
        Scaling factor for the Laplacian term in the model.
    seed : int, optional
        Random seed for reproducibility. If None, results are non-deterministic.
    
    Attributes
    ----------
    width : float
        Width parameter for the threshold calculation.
    scale : float
        Scaling factor for the diffusion term.
    shape : tuple of int
        Current shape of the simulation grid.
    delta_t : float
        Time step size for the simulation.
    c_reg : ndarray
        Concentration field array.
    lap_c : ndarray
        Laplacian of concentration field (workspace array).
    """
    def __init__(self, shape=50, speed=100, ini_c=3, var_c=2, scale=1, seed=None):
        self.width = 1
        self.scale = scale
        self.shape = shape
        self.delta_t = 0.01 * speed / 100
        
        # Only set the global seed if one is explicitly provided
        if seed is not None:
            np.random.seed(seed)

        if isinstance(self.shape, int): self.shape = (self.shape, self.shape)

        self.c_reg = np.full(self.shape, ini_c, dtype=float)
        if var_c != 0: 
            self.c_reg += np.random.random_sample(self.shape) * var_c

        self.lap_c = np.empty_like(self.c_reg)

    def reinforcement_model(self, g):
        """
        Execute one time step of the reinforcement diffusion model.
        
        Parameters
        ----------
        g : float
            Gamma parameter controlling reaction strength.
        """
        kernel_c = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]]) / 6
        threshold = 1
        mc = self.c_reg
        wrap = True

        if wrap: 
            convolve(mc, kernel_c, output=self.lap_c, mode='wrap')
        else:    
            convolve(mc, kernel_c, output=self.lap_c, mode='reflect')

        self.c_reg = mc + ((threshold - self.width - mc) * (threshold - mc) * (threshold + self.width - mc) * g + self.scale * self.lap_c) * self.delta_t

    def load_c(self, c):
        """
        Load a pre-existing concentration field into the model.
        
        Parameters
        ----------
        c : ndarray
            Concentration field array to load.
        """
        self.c_reg = c

    def run_simulation(self, start, stop, increasing_size=False):
        """
        Run the simulation for a specified number of iterations.
        
        Parameters
        ----------
        start : int
            Starting iteration number.
        stop : int
            Final iteration number (inclusive).
        increasing_size : bool, default=False
            If True, periodically grows the grid size during simulation.
        """
        gamma = 3 * np.sqrt(3) / (2 * self.width * self.width)
        
        for iteration in range(start + 1, stop + 1):
            if increasing_size: 
                if iteration % 50 == 1:
                    self.grow_one_row_c()
                    self.grow_one_col_c()

            if (self.c_reg).shape != self.shape:
                self.shape = self.c_reg.shape
                self.lap_c = np.empty_like(self.c_reg)
            
            self.reinforcement_model(gamma)

    def grow_one_row_c(self):
        """
        Add one row to the concentration field by duplicating random rows.
        
        This method inserts a new row by randomly selecting and duplicating
        existing rows, with bias toward avoiding the center region.
        """
        rows, cols = self.c_reg.shape
        new_c = np.zeros((rows + 1, cols))
        new_c[:rows,:] = self.c_reg[:,:]
        for col in range(0, cols):
            row = np.random.randint(0, rows)
            if 0.45 * rows <= row <= 0.55 * rows:
                if np.random.random() < 0.5:
                    row = np.random.randint(0, rows)
            new_c[(row+1):(rows+1), col] = self.c_reg[row:rows, col]
        self.c_reg = new_c

    def grow_one_col_c(self):
        """
        Add one column to the concentration field by duplicating random columns.
        
        This method inserts a new column by randomly selecting and duplicating
        existing columns at random positions.
        """
        rows, cols = self.c_reg.shape
        new_c = np.zeros((rows, cols + 1))
        new_c[:,:cols] = self.c_reg[:,:]        
        for row in range(0, rows):
            col = np.random.randint(0, cols)
            new_c[row, (col+1):(cols+1)] = self.c_reg[row, col:cols]
        self.c_reg = new_c

    def getC(self):
        """
        Get the current concentration field.
        
        Returns
        -------
        ndarray
            Current concentration field array.
        """
        return self.c_reg


def _process_single_pattern(args):
    """
    Helper function for parallel processing of cow patterns.
    
    Parameters
    ----------
    args : tuple
        Packed arguments containing:
        - idx : int
            Index of the pattern in the batch
        - props : tuple of float
            Proportion values for each color segment
        - shape : int or tuple
            Initial grid shape
        - speed : int
            Simulation speed parameter
        - ini_c : float
            Initial concentration value
        - var_c : float
            Concentration variance
        - scale : float
            Diffusion scaling factor
        - seed : int or None
            Base random seed
        - bisection_strength : float
            Strength of noise perturbation
        - rgb_palette : ndarray
            Array of RGB color tuples
        - total : int
            Total number of patterns being generated
    
    Returns
    -------
    rgb_image : ndarray
        RGB image array with shape (height, width, 3).
    indices : ndarray
        Integer array indicating segment indices for each pixel.
    """
    idx, props, shape, speed, ini_c, var_c, scale, seed, bisection_strength, rgb_palette, total, verbose = args
    
    if verbose:
        print(f"Starting pattern {idx + 1}/{total}...")
    
    pattern_seed = seed + idx if seed is not None else None
    
    # 1. Run the simulation for this pattern
    cow_tex = Reinforcement(shape=shape, speed=speed, ini_c=ini_c, var_c=var_c, scale=scale, seed=pattern_seed)
    
    # Standard burn-in and growth phases
    cow_tex.run_simulation(0, 100)
    cow_tex.run_simulation(0, 1000, increasing_size=True)

    raw_c = cow_tex.getC()
    
    # Smooth results to remove simulation artifacts
    pattern = gaussian_filter(raw_c, sigma=1.1)

    # 2. Prepare the Perturbation Map (The "Bisection" Logic)
    if bisection_strength > 0:
        noise = np.random.random(pattern.shape)
        smooth_noise = gaussian_filter(noise, sigma=pattern.shape[0] / 20) 
        
        # Normalize both signals
        smooth_noise = (smooth_noise - smooth_noise.mean()) / smooth_noise.std()
        pattern_norm = (pattern - pattern.mean()) / pattern.std()
        
        mixed_signal = pattern_norm + (smooth_noise * bisection_strength)
    else:
        mixed_signal = pattern

    # 3. Verify proportions
    if not np.isclose(sum(props), 1.0):
        props = np.array(props) / np.sum(props)
    
    # Calculate percentiles based on cumulative proportions
    cum_props = np.clip(np.cumsum(props)[:-1], 0, 1)
    threshold_values = np.percentile(mixed_signal, cum_props * 100)
    
    # Digitize creates an integer map (0, 1, 2...) based on thresholds
    indices = np.digitize(mixed_signal, threshold_values)
    
    # Map integer indices to RGB colors
    rgb_image = rgb_palette[indices]
    
    if verbose:
        print(f"Pattern {idx + 1}/{total} completed.")
    
    return rgb_image, indices


def generate_cow_patterns(
    shape=500,
    proportion_sets=[(0.5, 0.5)],
    colors=['black', 'white'],
    bisection_strength=1.2,
    seed=None,
    speed=200, 
    scale=5,
    ini_c=0, 
    var_c=2,
    return_segments=False,
    n_proc=1,
    verbose=False
):
    """
    Generate reaction-diffusion patterns segmented into specified color proportions.
    
    This function creates organic-looking patterns using a reaction-diffusion model,
    then segments them into regions with specified color proportions.
    
    Parameters
    ----------
    shape : int or tuple of int, default=500
        Initial shape of the simulation grid.
    proportion_sets : list of tuple of float, default=[(0.5, 0.5)]
        List where each tuple represents desired proportions for each color.
        Example: [(0.5, 0.5), (0.3, 0.3, 0.4)] creates two patterns.
    colors : list of str, default=['black', 'white']
        Color names to use for segments. Must be at least as long as the
        longest tuple in proportion_sets. Accepts matplotlib color names.
    bisection_strength : float, default=1.2
        Controls pattern irregularity. 0.0 produces concentric patterns,
        values >0.5 create irregular bisecting/patchwork patterns.
    seed : int, optional
        Base random seed for reproducibility. If None, results are random.
        Each pattern uses seed + index for independent randomness.
    speed : int, default=200
        Simulation speed parameter affecting convergence rate.
    scale : float, default=5
        Scaling factor for diffusion term in the model.
    ini_c : float, default=0
        Initial concentration value for the simulation grid.
    var_c : float, default=2
        Variance in initial concentration values.
    return_segments : bool, default=False
        If True, also return integer segment maps alongside RGB images.
    n_proc : int, default=1
        Number of processes for parallel execution. Use 1 for sequential.
    verbose : bool, default=True
        If True, print progress messages.
    
    Returns
    -------
    output_arrays : list of ndarray
        List of RGB numpy arrays (images), one per tuple in proportion_sets.
        Each array has shape (height, width, 3) with values in [0, 1].
    segment_maps : list of ndarray, optional
        Returned only if return_segments=True. List of integer arrays
        indicating segment indices for each pixel.
    
    Examples
    --------
    >>> patterns = generate_cow_patterns(
    ...     shape=200,
    ...     proportion_sets=[(0.6, 0.4), (0.3, 0.3, 0.4)],
    ...     colors=['red', 'blue', 'green'],
    ...     seed=42
    ... )
    >>> len(patterns)
    2
    """
    
    # Pre-convert color names to RGB tuples (0-1 floats)
    rgb_palette = np.array([mpl.colors.to_rgb(c) for c in colors])
    
    total = len(proportion_sets)
    if verbose:
        print(f"Generating {total} cow pattern(s) using {n_proc} process(es)...")

    # Prepare arguments for parallel processing
    task_args = [
        (idx, props, shape, speed, ini_c, var_c, scale, seed, bisection_strength, rgb_palette, total, verbose)
        for idx, props in enumerate(proportion_sets)
    ]
    
    if n_proc > 1:
        with Pool(processes=n_proc) as pool:
            results = pool.map(_process_single_pattern, task_args)
    else:
        results = [_process_single_pattern(arg) for arg in task_args]
    
    if verbose:
        print(f"All {total} pattern(s) completed successfully")
    
    # Unpack results
    output_arrays = [r[0] for r in results]
    segment_maps = [r[1] for r in results]
    
    if return_segments:
        return output_arrays, segment_maps
    else:
        return output_arrays


def consolidate_proportions(series_list, max_num=10, min_prop=0.01, reset_index=True):
    """
    Consolidate categories in proportion series based on max count and minimum proportion.
    
    Parameters:
    -----------
    series_list : list of pd.Series
        List of Series where each represents proportions (values should sum to 1)
    max_num : int
        Maximum number of categories to keep before consolidating into "Other"
    min_prop : float
        Minimum proportion threshold; categories below this are consolidated into "Other"
    reset_index : bool
        If True, use integer index (0, 1, 2, ...) for rows. If False, preserve series names.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with consolidated categories as columns, transposed so each row represents one series
    """
    consolidated_list = []
    
    for i, series in enumerate(series_list):
        # Sort by proportion in descending order
        sorted_series = series.sort_values(ascending=False)
        
        # Identify categories to keep based on max_num
        top_categories = sorted_series.iloc[:max_num]
        
        # Further filter by min_prop
        keep_mask = top_categories >= min_prop
        kept_categories = top_categories[keep_mask]
        
        # Calculate "Other" proportion
        other_prop = 1.0 - kept_categories.sum()
        
        # Create consolidated series
        if other_prop > 0:
            consolidated = pd.concat([
                kept_categories,
                pd.Series({'Other': other_prop})
            ])
        else:
            consolidated = kept_categories
        
        # Set series name based on reset_index parameter
        if reset_index:
            consolidated.name = i
        else:
            consolidated.name = series.name if series.name is not None else i
        
        consolidated_list.append(consolidated)
    
    # Concatenate into DataFrame, filling missing categories with 0
    df = pd.concat(consolidated_list, axis=1).fillna(0)
    
    # Transpose so rows represent series and columns represent categories
    return df.T


def hoodmap(mdata: md.MuData,
            chipnum: int,
            color: Optional[str] = None,
            subset: Optional[Dict] = None,
            ax: Optional[plt.Axes] = None,
            fss: int = 10,
            units: str = 'w',
            invert: bool = False,
            cow: bool = False,
            cow_maxnum: int = 5,
            cow_mincells: int = None,
            cow_minprop: float = 0.1,
            cow_cellmap_kwargs: Optional[Dict[str, Any]] = None,
            dilation: Optional[Number] = None,
            n_proc: int = 1,
            size: HoodMapCircleSize = None,
            wedgeprops: Optional[Dict[str, Any]] = None,
            circleprops: Optional[Dict[str, Any]] = None,
            legend: bool = True,
            legend_params: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> plt.Axes:
    """
    Plots neighborhood-level data on a spatial array plot using pie charts, circles,
    or cowprint style.

    This function visualizes aggregate information for each well (neighborhood),
    such as the composition of cell types within the well, represented as pie charts or
    cowprint style. If no color is provided, only circles are plotted. For pies or circles, 
    the size of the pies or circles can be scaled by the number of cells in the well.

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
    units : {'w', 'm'}, default 'w'
        The units for the plot coordinates. 'w' for well units, 'm' for metric (microns).
        Note: For pie/circle mode, only 'w' is supported. For cow mode, both are supported.
    invert : bool, default False
        If True, inverts the plot colors (e.g., for a dark background).
    cow : bool, default False
        If True, use a "cowprint" style for coloring wells. All other parameters related to 
        pie charts and circles are ignored.
    cow_mincells : int, default 5
        Minimum number of cells in a well to apply cowprint coloring. Wells with fewer cells are
        colored using cellmap.
    cow_maxnum : int, default 5
        Maximum number of categories to display in cowprint mode. Other categories are 
        consolidated into "Other".
    cow_minprop : float, default 0.1
        Minimum proportion of the most abundant category to color a well in cowprint mode.
    cow_cellmap_kwargs: dict, optional
        Additional keyword arguments to pass to cellmap() for wells that don't meet cowprint criteria.
    dilation : Number, optional
        A percentage to dilate (>100) or shrink (<100) the well squares.
    n_proc : int, default 1
        Number of processes to use for generating cow patterns in parallel.
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
        If `units` is not 'w' for pie/circle mode.
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

    check_size = False # may remove the size checking code later if we decide to allow sizes > 1

    if invert:
        default_circle_color = 'white'
        default_circle_edge_color = 'white'
    else:
        default_circle_color = 'black'
        default_circle_edge_color = 'black'

    default_wedgeprops = {'edgecolor': default_circle_edge_color, 'linewidth': 0.5}

    if cow_mincells is not None:
        if not isinstance(cow_mincells, int) or cow_mincells < 0:
            raise ValueError("Param `cow_mincells` must be a non-negative integer.")

    # Validate inputs
    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    chipnums = validate_chipnums(chipset, chipnum)
    if len(chipnums) > 1:
        raise ValueError("Only one chip number is allowed.")
    chipnum = chipnums[0]

    chip, masked_mdata = get_chip_mdata(mdata, chipnum, subset=subset)

    # Handle units and lerper setup
    if cow:
        # Cow mode supports both 'w' and 'm' units
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
    else:
        # Pie/circle mode only supports 'w' units
        if units != 'w':
            raise NotImplementedError("Only units='w' is currently supported in hoodmap() for pie/circle mode.")
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
        if cow:
            raise ValueError("Param `cow` cannot be True when `color` is None.")
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

    # Check for "Other" category conflict
    if cow and color is not None:
        dtypes = determine_data(masked_mdata, color=color, basis=basis)
        if dtypes['color']['type'] != 'cat':
            raise ValueError("Param `color` must be a categorical column for hoodmap() with cow=True.")
        
        color_mod = dtypes['color']['mod']
        if 'Other' in masked_mdata[color_mod].obs[color].cat.categories:
            raise ValueError(
                "The color column contains a category named 'Other', which conflicts with "
                "the consolidation process in cow mode. Please rename this category before using cow=True.")

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
            if check_size:
                _check_size(size)
            circlesize = size * max_size
            sizes = pd.Series(circlesize, index=df2.index) 
        elif isinstance(size, dict):
            if len(size) != 2:
                raise ValueError("If `size` is a dict, it must contain exactly two entries.")
            if any(not isinstance(k, int) or k <= 0 for k in size.keys()):
                raise ValueError("If `size` is a dict, both keys must be an integer number of cells.")
            for k in size:
                if check_size:
                    _check_size(size[k])
            well_cell_counts = masked_mdata['xyz'].obs.groupby('id').size()
            numcells = tuple(size.keys())
            sizes = normalize(well_cell_counts, clip=numcells, lower=size[numcells[0]]*max_size, upper=size[numcells[1]]*max_size)
        else:
            raise ValueError("Param `size` must be None, a number, or a dict of {int: float} pairs.")
    
    df2['sizes'] = sizes

    piedf = pd.concat([df1, df2], axis=1, join='inner').set_index(['arr-x', 'arr-y', 'sizes'], append=True)


    if cow:
        # Cowprint mode
        # 1. Calculate proportions for each well
        proportion_series_list = []
        well_ids = []

        if cow_mincells is not None:
            cellmap_ids = piedf[piedf.sum(1) < cow_mincells].index.get_level_values('id')
            piedf = piedf[~piedf.index.get_level_values('id').isin(cellmap_ids)]
        
        for id in piedf.index.get_level_values(0).unique():
            # Reset index to access data as columns
            well_data = piedf.loc[id].reset_index()
            
            # Get the first row (they should all be identical for a given id)
            well_data = well_data.iloc[0]
            
            # Extract proportions (all columns except 'arr-x', 'arr-y', and 'sizes')
            proportion_cols = [col for col in well_data.index if col not in ['arr-x', 'arr-y', 'sizes']]
            proportions = well_data[proportion_cols]
            proportions = proportions / proportions.sum()
            
            proportion_series_list.append(proportions)
            well_ids.append(id)

        # 2. Consolidate proportions
        consolidated_df = consolidate_proportions(
            proportion_series_list, 
            max_num=cow_maxnum, 
            min_prop=cow_minprop,
            reset_index=False
        )
        consolidated_df.index = well_ids
        
        # 3. Prepare color palette including "Other"
        colors_list = [cdict[cat] for cat in consolidated_df.columns if cat in cdict]
        if 'Other' in consolidated_df.columns:
            # Use a neutral gray for "Other"
            if invert:
                colors_list.append('lightgray')
            else:
                colors_list.append('darkgray')
        
        # 4. Generate cowprint patterns
        proportion_sets = [tuple(row.values) for _, row in consolidated_df.iterrows()]
        
        # Determine pattern size based on well size
        if units == 'm':
            pattern_size = int(chip.array.s)  # pixels roughly matching microns
        else:
            pattern_size = 100  # default size for well units
        
        cowprint_images = generate_cow_patterns(
            shape=pattern_size,
            proportion_sets=proportion_sets,
            colors=colors_list,
            bisection_strength=1.2,
            return_segments=False,
            n_proc=n_proc,
        )

        verts = chip.array.verts

        if dilation is not None:
            verts = resize_square_vertices(verts, dilation)
        
        # 5. Plot cells and cowprints onto wells
        verts_dict = {id: [lerper(p) for p in v] for id, v in verts.items() if id in well_ids}

        if cow_mincells is not None:

            default_config = {
                'invert': invert, 'legend': False, 'units': units, 
                'color': color, 'subset': subset, 'ax': ax, 'chipnum': chipnum}

            cellmap_config = get_config(cow_cellmap_kwargs, default_config, protected=tuple(default_config.keys()))

            ax = cellmap(masked_mdata, **cellmap_config)
            
        for idx, (id, cowprint) in enumerate(zip(well_ids, cowprint_images)):
            if id not in verts_dict:
                continue
                
            verts = np.array(verts_dict[id])
            
            # Calculate bounding box for the well
            x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
            y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
            
            # Display the cowprint image within the well bounds
            ax.imshow(
                cowprint,
                extent=[x_min, x_max, y_min, y_max],
                aspect='auto',
                interpolation='bilinear',
                origin='lower'
            )
        
        # Add legend for cowprint mode
        if legend:
            legend_cdict = {cat: cdict[cat] if cat in cdict else ('lightgray' if invert else 'darkgray') 
                           for cat in consolidated_df.columns}
            legend_params_config = get_add_legend_pm().get_params(legend_params)
            ax = decorate_scatter(ax, config=legend_params_config, plot_type='legend', cdict=legend_cdict)

    elif plot_pies:
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

    if circleprops is not None and not cow:
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
            plot_label_params: Optional[Dict[str, Any]] = None,
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
    plot_label_params : dict, optional
        Parameters for the plot labels, passed to `decorate_scatter`.
    ax : plt.Axes or list of plt.Axes, optional
        An existing Axes or list of Axes to plot on.
    fss : int, default 10
        Figure size scale for the plots, fed directly to `survey.genplot.subplots` if
        `ax` is not provided.
    ar : float, optional
        Aspect ratio for the plots, fed directly to `survey.genplot.subplots` if
        `ax` is not provided. 
    **kwargs
        Additional keyword arguments (not including plot_label_params) passed to `arrplot`.

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
        ax = arrplot(mdata, chipnum=chipnum, wells=well_dict, subset=subset, ax=ax, **kwargs)

        for label, position in zip([chipnum, group], label_positions):
            pos, ha, va = get_text_position_vals(position)
            default_config = {'pos': pos, 'ha': ha, 'va': va}
            plot_label_config = get_config(plot_label_params, default_config, protected={'pos', 'ha', 'va'})
            input_plot_label_params = get_add_plotlabel_pm().get_params(plot_label_config)
            ax = decorate_scatter(ax, config=input_plot_label_params, plot_type='plot_label', label=label)
        
        for label, position in zip([chipnum, group], label_positions):
            legend_params = get_add_legend_pm().get_params(legend_params)
            ax = decorate_scatter(ax, config=legend_params, plot_type='legend', cdict=cdict)

    return axes


class MultiFeatureArrayPlot:

    def __init__(self, mdata, chipnum, features, mods=None, layers=None, colors=None, order=None):
        """
        Initialize the MultiFeatureArrayPlot object for multi-channel spatial visualization.

        Parameters
        ----------
        mdata : mu.MuData
            A multimodal data object containing spatial and feature information.
        chipnum : int or str
            The specific chip identifier within the 'xyz' modality to be plotted.
        features : str or list of str
            The feature names (e.g., gene names) to be extracted and visualized.
        mods : str or list of str, optional
            The modality/modalities from which to extract each feature. If None,
            defaults to 'rna' for all features.
        layers : str or list of str, optional
            The specific data layer to use for each feature (e.g., 'counts', 'log1p').
            If None, uses the first available layer or 'raw'.
        colors : list of color specifications, optional
            List of colors to use for each feature. Can be any matplotlib-compatible
            color format (named colors, hex, RGB tuples, etc.). If None, defaults to
            ['red', 'green', 'blue'] for up to 3 features, extended with additional
            distinct colors for more features.
        order : str, optional
            Deprecated parameter for backward compatibility. If provided with colors=None,
            reorders the default RGB colors (e.g., 'rgb', 'bgr'). Ignored if colors is provided.
        """
        
        self.mdata = mdata
        self.chipnum = chipnum
        self.features = features
        self.mods = mods
        self.layers = layers
        self._input_colors = colors
        self._input_order = order.lower() if order is not None else None
        self.validate_inputs()

        self.feature_data = []

        for feature, mod, layer in zip(self.features, self.mods, self.layers):
            if feature not in self.mdata[mod].var_names:
                raise ValueError(f"Feature '{feature}' not found in modality '{mod}'.")
            if layer != 'raw' and layer not in self.mdata[mod].layers:
                raise ValueError(f"Layer '{layer}' not found in modality '{mod}'.")
            self.feature_data.append(get_obs_df(self.mdata[mod], features=feature, layer=layer))
        self.feature_data = pd.concat(self.feature_data, axis=1)
        self.feature_data = self.feature_data.copy()

        # Set up colors and determine RGB compatibility
        self._setup_colors()


    def _setup_colors(self):
        """
        Set up color mappings and determine if colors are RGB-compatible for additive plotting.
        
        Sets the following attributes:
        - self.colors: dict mapping color identifiers to RGB tuples
        - self.cmaps: list of LinearSegmentedColormaps for each feature
        - self.is_rgb_compatible: bool indicating if plot_additive can be used
        """
        default_rgb = {
            'r': (1, 0, 0),
            'g': (0, 1, 0),
            'b': (0, 0, 1),
        }
        
        # Extended palette for more than 3 features
        default_extended = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 
                           'orange', 'purple', 'pink', 'lime', 'navy', 'teal']
        
        if self._input_colors is None:
            # Use default colors
            if self._input_order is not None:
                # Backward compatibility: use order parameter
                if len(self.features) > 3:
                    raise ValueError("The 'order' parameter only works with up to 3 features. Use 'colors' parameter for more features.")
                color_specs = [default_rgb[c] for c in self._input_order]
                self.is_rgb_compatible = set(self._input_order) <= {'r', 'g', 'b'}
            else:
                # Default: RGB for first 3, then extended palette
                num_features = len(self.features)
                if num_features <= len(default_extended):
                    color_specs = default_extended[:num_features]
                else:
                    raise ValueError(f"Maximum {len(default_extended)} features supported with default colors. Please provide custom colors.")
                self.is_rgb_compatible = num_features <= 3
        else:
            # Custom colors provided
            if len(self._input_colors) != len(self.features):
                raise ValueError(f"Number of colors ({len(self._input_colors)}) must match number of features ({len(self.features)}).")
            color_specs = self._input_colors
            # Check if all colors are pure RGB
            self.is_rgb_compatible = self._check_rgb_compatibility(color_specs)
        
        # Convert all color specs to RGB tuples
        self.color_values = [mpl.colors.to_rgb(c) for c in color_specs]
        
        # Create dictionary for feature-to-color mapping
        self.colors = {feature: color for feature, color in zip(self.features, self.color_values)}
        
        # Create colormaps (white to color)
        self.cmaps = [LinearSegmentedColormap.from_list(f'white_to_{i}', ['white', color]) 
                      for i, color in enumerate(self.color_values)]


    def _check_rgb_compatibility(self, color_specs):
        """
        Check if provided colors are exactly pure red, green, and/or blue.
        
        Parameters
        ----------
        color_specs : list
            List of color specifications
            
        Returns
        -------
        bool
            True if all colors are pure R, G, or B and there are at most 3 features
        """
        if len(color_specs) > 3:
            return False
        
        rgb_tuples = [mpl.colors.to_rgb(c) for c in color_specs]
        pure_rgb = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        
        for rgb in rgb_tuples:
            # Check if color matches any pure RGB (with small tolerance for float comparison)
            if not any(all(abs(rgb[i] - pure[i]) < 1e-6 for i in range(3)) for pure in pure_rgb):
                return False
        
        return True

    def validate_inputs(self):
        """
        Validate the integrity of input parameters and data structure.

        Ensures that the MuData object contains the required 'xyz' modality, 
        that features exist in their respective modalities/layers, and that 
        the requested color ordering is valid.

        Raises
        ------
        ValueError
            If mdata is not a MuData object, if the 'xyz' modality or 'survey' 
            metadata is missing, or if feature/modality/layer lengths are mismatched.
        """
        if not isinstance(self.mdata, md.MuData):
            raise ValueError("mdata must be a MuData object.")

        if 'xyz' not in self.mdata.mod:
            raise ValueError("Modality 'xyz' must be present in mdata for array plotting.")
        
        if 'survey' not in self.mdata['xyz'].uns:
            raise ValueError("Modality 'xyz' must contain a 'survey' key in `.uns`.")

        if self.chipnum not in self.mdata['xyz'].uns['survey'].chips:
            raise ValueError(f"Chip number {self.chipnum} not found in modality 'xyz'.")

        if isinstance(self.features, str):
            self.features = [self.features]
        elif not isinstance(self.features, (list, tuple)):
            raise ValueError("features must be a string or a list/tuple of strings.")
        else:
            self.features = list(self.features)

        # No maximum feature limit anymore
        
        if self.mods is None:
            self.mods = ['rna'] * len(self.features)
        elif isinstance(self.mods, str):
            self.mods = [self.mods] * len(self.features)
        elif isinstance(self.mods, (list, tuple)):
            if len(self.mods) != len(self.features):
                raise ValueError("Length of mods must match length of features.")
            else:
                self.mods = list(self.mods)

        for feature, mod in zip(self.features, self.mods):
            if feature in self.mdata['xyz'].obs.columns and mod != 'xyz':
                raise ValueError(f"Feature '{feature}' exists in 'xyz' obs; please drop this column to plot the feature from {mod}.")
        
        if self.layers is None:
            layers = []
            for mod in self.mods:
                if len(self.mdata[mod].layers) == 0:
                    layers.append('raw')
                else:
                    layers.append(list(self.mdata[mod].layers.keys())[0])
            self.layers = layers
        elif isinstance(self.layers, str):
            self.layers = [self.layers] * len(self.features)
        elif isinstance(self.layers, (list, tuple)):
            if len(self.layers) != len(self.features):
                raise ValueError("Length of layers must match length of features.")
            else:
                self.layers = list(self.layers)
        
        # Validate order parameter (for backward compatibility)
        if self._input_order is not None:
            if not isinstance(self._input_order, str):
                raise ValueError("order must be a string ('rgb' or any permutation).")
            elif set(self._input_order) != set('rgb'):
                raise ValueError("order must be a permutation of 'r', 'g', 'b'.")
            if self._input_colors is not None:
                warnings.warn("Both 'colors' and 'order' parameters provided. 'order' will be ignored.", UserWarning)

    def get_configs(self, wells, kwargs, protected):
        """
        Generate individual configuration dictionaries for each feature.

        Parameters
        ----------
        wells : list, dict, or str
            Well identifiers to be plotted. Can be a single set applied to all 
            features or a mapping of features to specific wells.
        kwargs : dict
            Additional keyword arguments for the plotting function.
        protected : set
            A set of parameter names that are managed internally and cannot 
            be overridden via kwargs.

        Returns
        -------
        dict
            A dictionary where keys are feature names and values are 
            dictionaries of parameters for `svp.pl.arrplot`.
        """

        if 'wells' not in protected:
            raise ValueError("The 'wells' parameter must be protected and cannot be overridden in arrplot.")
        
        arrplot_configs = {feature: {} for feature in self.features}
        for k, v in kwargs.items():
            if k in protected:
                raise ValueError(f"Cannot override protected parameter '{k}' in arrplot.")
            if isinstance(v, dict) and all([i in self.features for i in v.keys()]):
                for feature in self.features:
                    arrplot_configs[feature][k] = v[feature]
            else:
                for feature in self.features:
                    arrplot_configs[feature][k] = v
        
        if isinstance(wells, dict):
            for feature in self.features:
                if feature not in wells:
                    raise ValueError(f"When providing wells as a dictionary, it must contain an entry for each feature. Missing: '{feature}'")
                arrplot_configs[feature]['wells'] = wells[feature]
        else:
            for feature in self.features:
                arrplot_configs[feature]['wells'] = wells

        return arrplot_configs


    def visualize_dists(self, wells, return_welldata=False, **kwargs):
        """
        Generate ridge plots to visualize the distribution of feature intensities.

        Useful for assessing normalization and contrast before merging channels.

        Parameters
        ----------
        wells : list, dict, or str
            Well identifiers to include in the distribution analysis.
        return_welldata : bool, default False
            If True, returns the processed well data intensities.
        **kwargs : dict
            Additional arguments passed to the underlying `arrplot` call.

        Returns
        -------
        dict or None
            If `return_welldata` is True, returns a dictionary mapping features 
            to intensity DataFrames; otherwise returns None.
        """

        protected = {'mdata', 'chipnum', 'feature', 'wells', 'plot', 'return_welldata'}
        arrplot_configs = self.get_configs(wells, kwargs, protected=protected)

        self.mdata['xyz'].obs = self.mdata['xyz'].obs.join(self.feature_data, how='left')
        try:
            welldatas = {}

            feature = self.features[0]
            welldatas[feature] = arrplot(self.mdata, chipnum=self.chipnum, color=feature, plot=False, return_welldata=True, **arrplot_configs[feature])
            plt.close() # bug in arrplot, fixed on instance repo, needs to be pushed
            for idx, feature in enumerate(self.features[1:], start=1):
                welldatas[feature] = arrplot(self.mdata, chipnum=self.chipnum, color=feature, plot=False, return_welldata=True, **arrplot_configs[feature])
                plt.close()
            palette = {feature: color for feature, color in zip(self.features, self.color_values)}
            df = pd.DataFrame(welldatas).melt(var_name='feature')

            ridge = Ridge(df, x='value', y='feature')
            ridge.add_ridge_data(hist=True, kde=False, bins=50, order=self.features)
            fig, axes = ridge.plot(colors=[palette[c] for c in ridge.order], legend=True)
            fig.text(0.08, 0.55, 'Number of Wells', va='center', rotation='vertical')

        except Exception as e:
            raise e
        finally:
            self.mdata['xyz'].obs.drop(columns=self.features, inplace=True)
        if return_welldata:
            return welldatas
        else:
            return


    def visualize_maps(self, wells, ar=1.1, fss=10, alpha_cmap_kwargs=None, **kwargs):
        """
        Plot individual spatial heatmaps for each feature in a grid.



        Parameters
        ----------
        wells : list, dict, or str
            Well identifiers to be plotted.
        ar : float, default 1.1
            Aspect ratio for the subplots.
        fss : int, default 10
            Figure size scaling factor.
        alpha_cmap_kwargs : dict, optional
            Arguments passed to `create_alpha_cmap` to handle transparency.
        **kwargs : dict
            Additional arguments passed to the underlying `arrplot` call.
        """

        alpha_cmap_config = get_config(alpha_cmap_kwargs, {'scale_alpha': True})
        protected = {'mdata', 'chipnum', 'feature', 'wells', 'cmap', 'ax', 'cbar', 'plot_label'}
        arrplot_configs = self.get_configs(wells, kwargs, protected=protected)

        fig, axes = subplots(len(self.features), ar=ar, fss=fss, as_seq=True)

        self.mdata['xyz'].obs = self.mdata['xyz'].obs.join(self.feature_data, how='left')
        try:
            for ax, cmap, feature in zip(axes.flat, self.cmaps, self.features):
                cmap = create_alpha_cmap(cmap, **alpha_cmap_config)
                ax = arrplot(self.mdata, chipnum=self.chipnum, color=feature, cmap=cmap, cbar=True, plot_label=True, ax=ax, **arrplot_configs[feature])
        except Exception as e:
            raise e
        finally:
            self.mdata['xyz'].obs.drop(columns=self.features, inplace=True)


    def merge_color_dictionaries(self, welldicts, baseline=0):
        """
        Merge one, two, or three dictionaries into a single RGB color map.
        
        Maps intensity values to colors by creating gradients from a baseline color
        to pure R, G, B, or their combinations (yellow, magenta, cyan, or grayscale).
        Designed for plotting on white backgrounds where higher values → darker colors.
        
        Parameters
        ----------
        welldicts : list of dict
            List of 1-3 dictionaries mapping IDs to intensity values.
            - welldicts[0]: Red channel intensities
            - welldicts[1]: Blue channel intensities (if provided)
            - welldicts[2]: Green channel intensities (if provided)
            Values should be pre-normalized/digitized to desired range.
        baseline : float, default=0
            Starting saturation of the colormap (0-1).
            - 0: gradients start at white
            - 1: gradients start at full color
            - 0.25: gradients start at 25% saturated color
        
        Returns
        -------
        dict
            Dictionary mapping IDs to RGB tuples with values in range [0, 1].
            
        Notes
        -----
        Color mapping logic based on active channels:
        - R only: white/baseline → red
        - G only: white/baseline → green  
        - B only: white/baseline → blue
        - R+G: white/baseline → yellow
        - R+B: white/baseline → magenta
        - G+B: white/baseline → cyan
        - R+G+B: white/baseline → black (grayscale)
        
        Intensity is calculated as the sum of values across all provided channels,
        normalized by the number of active channels. Higher intensity values produce
        darker colors (approaching the target color or black for grayscale).
        """
        # Validate input
        if not isinstance(welldicts, list) or len(welldicts) == 0 or len(welldicts) > 3:
            raise ValueError("welldicts must be a list of 1-3 dictionaries")
        
        # Pad with empty dicts if fewer than 3 provided
        while len(welldicts) < 3:
            welldicts.append({})
        
        r_dict, b_dict, g_dict = welldicts[0], welldicts[1], welldicts[2]
        
        # Get all unique IDs
        all_ids = set(r_dict.keys()) | set(b_dict.keys()) | set(g_dict.keys())
        
        # Determine which channels are available
        r_available = len(r_dict) > 0
        g_available = len(g_dict) > 0
        b_available = len(b_dict) > 0
        
        # Count active channels
        num_active_channels = sum([r_available, g_available, b_available])
        
        if num_active_channels == 0:
            raise ValueError("At least one dictionary must contain data")
        
        color_map = {}
        
        for id_ in all_ids:
            # Get intensity values (default to 0 if ID not in dict)
            r_val = r_dict.get(id_, 0)
            g_val = g_dict.get(id_, 0)
            b_val = b_dict.get(id_, 0)
            
            # Calculate total intensity (sum of values from active channels)
            total_intensity = 0
            if r_available and r_val > 0:
                total_intensity += r_val
            if g_available and g_val > 0:
                total_intensity += g_val
            if b_available and b_val > 0:
                total_intensity += b_val
            
            # Normalize intensity to 0-1 range
            # Use max possible value from all active channels
            max_vals = []
            if r_available:
                max_vals.append(max(r_dict.values()) if r_dict else 0)
            if g_available:
                max_vals.append(max(g_dict.values()) if g_dict else 0)
            if b_available:
                max_vals.append(max(b_dict.values()) if b_dict else 0)
            
            max_sum = sum(max_vals)
            intensity = total_intensity / max_sum if max_sum > 0 else 0
            
            # Determine which channels are active for this ID
            r_active = r_available and r_val > 0
            g_active = g_available and g_val > 0
            b_active = b_available and b_val > 0
            
            # Determine target color based on active channels (additive mixing)
            if r_active and g_active and b_active:
                # All three: grayscale (white → black)
                target_color = (0.0, 0.0, 0.0)  # Black
            elif r_active and g_active:
                # R+G: yellow
                target_color = (1.0, 1.0, 0.0)
            elif r_active and b_active:
                # R+B: magenta
                target_color = (1.0, 0.0, 1.0)
            elif g_active and b_active:
                # G+B: cyan
                target_color = (0.0, 1.0, 1.0)
            elif r_active:
                # Only R: red
                target_color = (1.0, 0.0, 0.0)
            elif g_active:
                # Only G: green
                target_color = (0.0, 1.0, 0.0)
            elif b_active:
                # Only B: blue
                target_color = (0.0, 0.0, 1.0)
            else:
                # None active: white
                target_color = (1.0, 1.0, 1.0)
            
            # Calculate baseline color (starting point)
            # Baseline shifts from white (1,1,1) toward target color
            baseline_color = (
                1.0 - baseline * (1.0 - target_color[0]),
                1.0 - baseline * (1.0 - target_color[1]),
                1.0 - baseline * (1.0 - target_color[2])
            )
            
            # Linear interpolation from baseline to target based on intensity
            # Higher intensity → move toward target (darker/more saturated)
            r = baseline_color[0] - intensity * (baseline_color[0] - target_color[0])
            g = baseline_color[1] - intensity * (baseline_color[1] - target_color[1])
            b = baseline_color[2] - intensity * (baseline_color[2] - target_color[2])
            
            color_map[id_] = (r, g, b)
        
        return color_map


    def plot_alpha(self, wells, alpha_cmap_kwargs=None, legend=True, **kwargs):
        """
        Overlay multiple features using alpha-blended colormaps.

        Each feature is plotted on the same axes with a custom transparency 
        gradient, allowing visual overlap of up to three channels.



        Parameters
        ----------
        wells : list, dict, or str
            Well identifiers to be plotted.
        alpha_cmap_kwargs : dict, optional
            Configuration for the transparency scaling (e.g., `scale_alpha`).
        legend : bool, default True
            Whether to display the legend.
        **kwargs : dict
            Additional arguments passed to the underlying `arrplot` call.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the overlaid plots and custom legend if requested.
        """

        alpha_cmap_config = get_config(alpha_cmap_kwargs, {'scale_alpha': True})
        protected = {'mdata', 'chipnum', 'feature', 'wells', 'cmap', 'ax', 'cbar', 'plot_label'}
        arrplot_configs = self.get_configs(wells, kwargs, protected=protected)

        self.mdata['xyz'].obs = self.mdata['xyz'].obs.join(self.feature_data, how='left')
        try:
            cmap = create_alpha_cmap(self.cmaps[0], **alpha_cmap_config)
            feature = self.features[0]
            ax = arrplot(self.mdata, chipnum=self.chipnum, color=feature, cmap=cmap, cbar=False, plot_label=False, **arrplot_configs[feature])
            for idx, feature in enumerate(self.features[1:], start=1):
                cmap = create_alpha_cmap(self.cmaps[idx], **alpha_cmap_config)
                ax = arrplot(self.mdata, chipnum=self.chipnum, color=feature, cmap=cmap, cbar=False, plot_label=False, ax=ax, **arrplot_configs[feature])
            palette = {feature: color for feature, color in zip(self.features, self.color_values)}
            if legend:
                config = get_pm('legend').get_params()
                ax = decorate_scatter(ax, plot_type='legend', config=config, cdict=palette)
        except Exception as e:
            raise e
        finally:
            self.mdata['xyz'].obs.drop(columns=self.features, inplace=True)
        
        return ax


    def plot_additive(self, wells, norms=None, baseline=0, **kwargs):
        """
        Plot features using additive color mixing (RGB logic).

        Normalizes intensities and merges them into a single RGB dictionary for 
        plotting. This method only works when features are assigned to pure red,
        green, and/or blue colors.

        Parameters
        ----------
        wells : list, dict, or str
            Well identifiers to be plotted.
        norms : list, tuple, or dict, optional
            Normalization bounds (min, max). If a tuple, applies to all features. 
            If a dict, maps feature names to specific (min, max) tuples.
        baseline : float, default 0
            The background saturation level (0 is white).
        **kwargs : dict
            Additional arguments passed to the underlying `arrplot` call.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the combined additive color plot.
            
        Raises
        ------
        ValueError
            If the assigned colors are not RGB-compatible (i.e., not pure red,
            green, and/or blue).
        """
        
        # Check RGB compatibility
        if not self.is_rgb_compatible:
            raise ValueError(
                "plot_additive() only works with pure red (1,0,0), green (0,1,0), and blue (0,0,1) colors. "
                "Current colors are not RGB-compatible. Use plot_alpha() instead or reinitialize with "
                "RGB-compatible colors (e.g., colors=['red', 'green', 'blue'])."
            )
        
        if len(self.features) > 3:
            raise ValueError("plot_additive() only supports up to 3 features.")
        
        for kwarg in kwargs:
            if kwarg in {'cbar', 'plot_label'} and kwargs[kwarg]:
                raise ValueError(f"The '{kwarg}' parameter is not supported for plot_additive().")
            elif kwarg in {'norm'}:
                raise ValueError(f"The '{kwarg}' parameter is not supported for plot_additive(), please supply norms directly using the 'norms' parameter.")

        if norms is not None:
            if isinstance(norms, (list, tuple)):
                if len(norms) != 2:
                    raise ValueError("If norms is a list/tuple, it must be of the form (min1, max1), which will be applied to all features.")
                norms = {feature: norms for feature in self.features}
            elif isinstance(norms, dict):
                for feature in norms:
                    if feature not in self.features:
                        raise ValueError(f"norms dictionary contains an invalid feature '{feature}'. Must match features provided.")
                    if not isinstance(norms[feature], (list, tuple)) or len(norms[feature]) != 2:
                        raise ValueError("Each entry in norms must be a tuple/list of (min, max).")
                    not_in_norms = set(self.features) - set(norms.keys())
                    for feature in not_in_norms:
                        norms.update({feature: [None, None]})

        
        welldatas = self.visualize_dists(wells, return_welldata=True, **kwargs)
        plt.close()
        if norms is not None:
            for feature in self.features:
                welldatas[feature] = normalize(welldatas[feature], clip=(norms[feature][0], norms[feature][1]))
        color_map = self.merge_color_dictionaries([welldatas[f].to_dict() for f in welldatas], baseline=baseline)
        ax = arrplot(self.mdata, chipnum=self.chipnum, wells=color_map, cbar=False, plot_label=False, **kwargs)

        return ax


class DiffusionVisualizer:

    def __init__(self, mdata, chipnum, chip_key_prop='rna:chip-num', layer=None):
        """
        Initialize the DiffusionVisualizer to analyze barcode spatial diffusion.

        Parameters
        ----------
        mdata : MuData
            Multimodal data object containing spatial and sequencing information.
            Must contain an 'xyz' modality with survey information.
        chipnum : int or str
            The identifier for the specific chip to visualize.
        chip_key_prop : str, default 'rna:chip-num'
            The column name in `mdata.obs` used to filter observations by `chipnum`.
        layer : str, optional
            The specific layer in the modality anndata objects to use for counts. 
            If None, uses `.X`.

        Raises
        ------
        ValueError
            If 'xyz' modality or 'survey' key is missing.
            If `chipnum` is not found in the survey.
            If `chip_key_prop` is missing from `mdata.obs`.
            If required modalities for the chip's barcode types are missing.
            If the chip layout does not have exactly 2 spatial indices.
        """

        if 'xyz' not in mdata.mod:
            raise ValueError("mdata must have 'xyz' modality for spatial plotting.")
        if 'survey' not in mdata['xyz'].uns:
            raise ValueError("mdata['xyz'] must have 'survey' key.")
        
        if chipnum not in mdata['xyz'].uns['survey'].chips:
            raise ValueError(f"Chip number {chipnum} not found in mdata['xyz'].uns['survey'].chips.")
        
        if chip_key_prop not in mdata.obs:
            raise ValueError(f"\
                             The `mdata.obs` must contain '{chip_key_prop}' column for subsetting by chip number.\
                             Please add it using mdata.update from any modality that contains the chip number information.\
                                ")
        
        chip = mdata['xyz'].uns['survey'].chips[chipnum]
        bctypes = chip.layout.bctypes

        for bctype in bctypes:
            if bctype not in mdata.mod:
                raise ValueError(f"mdata must contain modality '{bctype}' for chip {chipnum}.")
    
        submdata = mdata[mdata.obs[chip_key_prop] == chipnum]

        if submdata.n_obs == 0:
            raise ValueError(f"No observations found for chip {chipnum}. Please check parameters.")

        num_spidxes = len(chip.layout.coords)

        if num_spidxes != 2:
            raise ValueError(f"Chip {chipnum} has {num_spidxes} spatial indices, but this visualizer is designed only for 2.")

        spbcmaps = self.get_spbcmaps(submdata, chip, num_spidxes, bctypes, layer)
        toplocs = self.get_top_locs(chip, num_spidxes, spbcmaps)

        self.mdata = mdata
        self.submdata = submdata
        self.chip = chip
        self.chip_key_prop = chip_key_prop
        self.layer = layer

        self.spbcmaps = spbcmaps
        self.toplocs = toplocs

        self.diffarr = None

    def __repr__(self):
        return f"DiffusionVisualizer for chip {self.chipnum} ({self.chip.layout.format}, {'-'.join(self.bctypes)}, ncells={self.submdata.n_obs})"
    
    def __str__(self):
        return self.__repr__()
    
    @property
    def chipnum(self):
        """
        Get the current chip identifier.

        Returns
        -------
        int or str
            The chip number.
        """
        return self.chip.num

    @property
    def num_spidxes(self):
        """
        Get the number of spatial indices for the current chip.

        Returns
        -------
        int
            Number of coordinate axes (e.g., 2 for 2D layouts).
        """
        return len(self.chip.layout.coords)

    @property
    def bctypes(self):
        """
        Get the barcode types associated with the current chip.

        Returns
        -------
        list of str
            List of modality keys representing barcode types.
        """
        return self.chip.layout.bctypes
    
    @property
    def center(self):
        """
        Get the dimensions/center reference of the chip layout.

        Returns
        -------
        tuple of int
            The shape (rows, cols) of the chip's design array.
        """
        return self.chip.layout.da.shape[:2]
    
    def get_ranges(self):
        """
        Get the spatial coordinate ranges for the chip layout.

        Returns
        -------
        list of range
            List of range objects for each spatial dimension, centered around zero.
        """
        rangerows = range(-self.chip.layout.df.shape[0] + 1, self.chip.layout.df.shape[0])
        rangecols = range(-self.chip.layout.df.shape[1] + 1, self.chip.layout.df.shape[1])
        return rangerows, rangecols
    
    def get_spbcmaps(self, submdata, chip, num_spidxes, bctypes, layer=None):
        """
        Map spatial barcodes to their count matrices for native and permuted states.

        Parameters
        ----------
        submdata : MuData
            Subset of the master MuData containing only observations for the current chip.
        chip : Chip
            The chip object containing layout and mapper information.
        num_spidxes : int
            The number of spatial indices to process.
        bctypes : list of str
            The barcode modalities to extract.
        layer : str, optional
            The AnnData layer to extract counts from. Defaults to `.X`.

        Returns
        -------
        dict
            A nested dictionary with keys 'n' (native) and 'p' (permuted).
            Each contains indexed dictionaries with count matrices ('x'), 
            names ('s'), and mapped values ('v').
        """
        spbcmaps = {'n': {}, 'p': {}} # native, permuted

        for bctype, i in zip(bctypes, range(num_spidxes)):
            spbcmaps['n'][i] = {}
            subsbc_view = submdata[bctype][:, chip.layout.mappers[i].index]

            if layer is None:
                spbcmaps['n'][i]['x'] = subsbc_view.X.astype(int).copy()
            else:
                spbcmaps['n'][i]['x'] = subsbc_view.layers[layer].astype(int).copy()

            spbcmaps['n'][i]['x'].eliminate_zeros()
            spbcmaps['n'][i]['s'] = subsbc_view.var_names.copy()
            spbcmaps['n'][i]['v'] = subsbc_view.var_names.map(chip.layout.mappers[i].to_dict())

            spbcmaps['p'][i] = {}

            arr = spbcmaps['n'][i]['x'].toarray().flatten()
            np.random.shuffle(arr)
            arr = arr.reshape(spbcmaps['n'][i]['x'].shape)
            spbcmaps['p'][i]['x'] = csr_matrix(arr)
            spbcmaps['p'][i]['s'] = spbcmaps['n'][i]['s'].copy()
            spbcmaps['p'][i]['v'] = spbcmaps['n'][i]['v'].copy()

        return spbcmaps
    
    
    def get_top_locs(self, chip, num_spidxes, spbcmaps):
        """
        Identify the spatial locations of the most frequent barcodes (top barcodes).

        Parameters
        ----------
        chip : Chip
            The chip object containing layout and coordinate metadata.
        num_spidxes : int
            The number of spatial indices.
        spbcmaps : dict
            The mapping dictionary generated by `get_spbcmaps`.

        Returns
        -------
        dict
            A dictionary with keys 'n' and 'p' containing lists of 
            coordinate tuples representing the "top" spatial location 
            for each observation. Returns -1 if no data is present.
        """
        top_bcids = {'n': {}, 'p': {}} # native, permuted
        toplocs_bcid = {'n': {}, 'p': {}} # native, permuted
        toplocs = {'n': [], 'p': []} # native, permuted

        for i in range(num_spidxes):
            top_bcids['n'][i] = []
            for row in spbcmaps['n'][i]['x']:
                if row.data.size > 0:
                    nzidxes = row.nonzero()[1]
                    bcids = spbcmaps['n'][i]['s'][nzidxes]
                    top_bcid = bcids[np.argsort(row.data)[-1]]
                else:
                    top_bcid = np.nan
                top_bcids['n'][i].append(top_bcid)

        for i in range(num_spidxes):
            top_bcids['p'][i] = []
            for row in spbcmaps['p'][i]['x']:
                if row.data.size > 0:
                    nzidxes = row.nonzero()[1]
                    bcids = spbcmaps['p'][i]['s'][nzidxes]
                    top_bcid = bcids[np.argsort(row.data)[-1]]
                else:
                    top_bcid = np.nan
                top_bcids['p'][i].append(top_bcid)
                
        toplocs_bcid['n'] = list('-'.join(map(str, t)) for t in zip(*(top_bcids['n'][i] for i in top_bcids['n'].keys())))
        toplocs_bcid['p'] = list('-'.join(map(str, t)) for t in zip(*(top_bcids['p'][i] for i in top_bcids['p'].keys())))

        for toploc_bcid in toplocs_bcid['n']:
            if 'nan' in toploc_bcid:
                toplocs['n'].append(-1)
            else:
                toplocs['n'].append(tuple(chip.layout.df_stacked.loc[toploc_bcid]))

        for toploc_bcid in toplocs_bcid['p']:
            if 'nan' in toploc_bcid:
                toplocs['p'].append(-1)
            else:
                toplocs['p'].append(tuple(chip.layout.df_stacked.loc[toploc_bcid]))
        
        return toplocs


    def add_diffarr(self):
        """
        Calculate the spatial diffusion arrays for native and permuted data.

        This method populates `self.diffarr` with:
            - 'n': Cumulative spatial distribution of barcodes relative to the top barcode.
            - 'p': Randomly shuffled spatial distribution (control).
            - 'd': The difference ('n' - 'p'), representing non-random diffusion.

        Returns
        -------
        None
        """

        arr_template = np.zeros((self.center[0]*2-1, self.center[1]*2-1), dtype=int)

        diffarr = {'n': arr_template.copy(), 'p': arr_template.copy()} # native, permuted

        print("Calculating diffusion arrays for native data...", end=' ')
        for row0, row1, loc in zip(self.spbcmaps['n'][0]['x'], self.spbcmaps['n'][1]['x'], self.toplocs['n']):
            if loc == -1:
                continue
            nzidxes = row0.nonzero()[1]
            bcids = self.spbcmaps['n'][0]['v'][nzidxes]
            # top_bcid = bcids[np.argsort(row0.data)[-1]]
            for idx, bcid in enumerate(bcids):
                row_range, col_range = slice(self.center[0]-loc[0], 2*self.center[0]-loc[0]), slice(self.center[1]-loc[1], 2*self.center[1]-loc[1])
                diffarr['n'][row_range, col_range] += np.where(self.chip.layout.da[:, :, 0] == bcid, row0.data[idx], 0)
            
            nzidxes = row1.nonzero()[1]
            bcids = self.spbcmaps['n'][1]['v'][nzidxes]
            # top_bcid = bcids[np.argsort(row1.data)[-1]]
            for idx, bcid in enumerate(bcids):
                row_range, col_range = slice(self.center[0]-loc[0], 2*self.center[0]-loc[0]), slice(self.center[1]-loc[1], 2*self.center[1]-loc[1])
                diffarr['n'][row_range, col_range] += np.where(self.chip.layout.da[:, :, 1] == bcid, row1.data[idx], 0)
        print("Done.")

        print("Calculating diffusion arrays for permuted data...", end=' ')
        for row0, row1, loc in zip(self.spbcmaps['p'][0]['x'], self.spbcmaps['p'][1]['x'], self.toplocs['p']):
            if loc == -1:
                continue
            nzidxes = row0.nonzero()[1]
            bcids = self.spbcmaps['p'][0]['v'][nzidxes]
            # top_bcid = bcids[np.argsort(row0.data)[-1]]
            for idx, bcid in enumerate(bcids):
                row_range, col_range = slice(self.center[0]-loc[0], 2*self.center[0]-loc[0]), slice(self.center[1]-loc[1], 2*self.center[1]-loc[1])
                diffarr['p'][row_range, col_range] += np.where(self.chip.layout.da[:, :, 0] == bcid, row0.data[idx], 0)
            
            nzidxes = row1.nonzero()[1]
            bcids = self.spbcmaps['p'][1]['v'][nzidxes]
            # top_bcid = bcids[np.argsort(row1.data)[-1]]
            for idx, bcid in enumerate(bcids):
                row_range, col_range = slice(self.center[0]-loc[0], 2*self.center[0]-loc[0]), slice(self.center[1]-loc[1], 2*self.center[1]-loc[1])
                diffarr['p'][row_range, col_range] += np.where(self.chip.layout.da[:, :, 1] == bcid, row1.data[idx], 0)
        print("Done.")
        
        diffarr['d'] = np.abs(diffarr['n'] - diffarr['p'])

        self.diffarr = diffarr
        return


    def plot_diffarrs(self, norm=None, fss=6, plotarrs=None):
        """
        Plot the native, permuted, and difference diffusion arrays.

        Parameters
        ----------
        norm : matplotlib.colors.Normalize or list thereof, optional
            Normalization to apply to the heatmaps. Can be a single 
            normalizer or a list of three for [native, permuted, diff].
        fss : int, default 6
            Figure size scaling factor.
        plotarrs : list of str, optional
            List of which arrays to plot. Can contain any of 'n' (native),
            'p' (permuted), 'd' (difference). If None, plots all three.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.
        axes : numpy.ndarray of matplotlib.axes.Axes
            The array of subplot axes.
        """
        ar = self.chip.array.arr_shape[0]/self.chip.array.arr_shape[1]
        name_mapper = {'n': 'Native', 'p': 'Permuted', 'd': 'Difference'}

        if plotarrs is not None:
            if not all(plotarr in ('n', 'p', 'd') for plotarr in plotarrs):
                raise ValueError("plotarrs must be a list containing any of 'n', 'p', 'd'.")
            arr_names = list(plotarrs)
        else:
            arr_names = ['n', 'p', 'd']
        
        fig, axes = subplots(len(arr_names), ar=ar, fss=fss, as_seq=True)
        if norm is None:
            norms = [None]*len(arr_names)
        elif isinstance(norm, (list, tuple)) and len(norm) == len(arr_names):
            norms = norm
        else: # norm must be a normalizer
            norms = [norm]*len(arr_names)

        for ax, arr_name, norm in zip(axes, arr_names, norms):
            arr = self.diffarr[arr_name]
            ax.imshow(arr, cmap='viridis', norm=norm)
            ax.grid()
            ax.set_title(f"Diffusion map: {name_mapper[arr_name]}")
        
        return fig, axes
    
    
    def plot_marginal_diffs(self, fss=3, ylabel_yoffset=-0.1):
        """
        Plot marginal distributions of barcode diffusion along spatial axes.

        Parameters
        ----------
        fss : int, default 3
            Figure size scaling factor.
        ylabel_yoffset : float, default -0.1
            Vertical offset for the y-axis label to improve spacing.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.
        axes : numpy.ndarray of matplotlib.axes.Axes
            The array of subplot axes (one for each spatial dimension).

        Raises
        ------
        ValueError
            If the chip layout format is not 'rowcol'.
        """

        if self.chip.layout.format != 'rowcol':
            raise ValueError("Marginal diffs plot is only implemented for 'rowcol' format chips.")

        label_mapper = {
            'row': 'Row',
            'col': 'Column'
        }

        diffarr_normz_sums = self.diffarr['d'].sum(0), self.diffarr['d'].sum(1)
        
        overallsum = self.diffarr['d'].sum()

        fig, axes = subplots(2, ncols=1, ar=3, fss=fss)

        rangerows, rangecols = self.get_ranges()
        if self.chip.layout.coords[0] == 'row' and self.chip.layout.coords[1] == 'col':
            range0, range1 = rangecols, rangerows
        else:
            range0, range1 = rangerows, rangecols
        axes[0].bar(range0, diffarr_normz_sums[0]/overallsum)
        axes[1].bar(range1, diffarr_normz_sums[1]/overallsum)

        label0, label1 = label_mapper[self.chip.layout.coords[1]], label_mapper[self.chip.layout.coords[0]]

        axes[0].set_xlabel(f'{label0} offset from top {label0.lower()} barcode')
        axes[1].set_xlabel(f'{label1} offset from top {label1.lower()} barcode')

        axes[0].set_ylabel('Relative frequency of non-top barcodes', y=ylabel_yoffset, horizontalalignment='center')
        # skip adjusting ylabel for second plot since they share the same y-axis meaning

        plt.tight_layout()
        
        return fig, axes


def get_vertices(mdata, chipnum, method='printed', xy=None, context=0):
    """
    Get the vertices of a rectangular region on a chip.

    Parameters
    ----------
    mdata : AnnData
        Annotated data matrix.
    chipnum : int
        Chip identifier.
    method : str, optional
        Method to determine the vertices. Must be either 'printed' or 'ids'.
        Default is 'printed'.
    xy : tuple of int, optional
        Coordinates (arr-x, arr-y) to use when method is 'ids'. Default is None.
    context : int, optional
        Context size to use when method is 'ids'. Default is 0.

    Returns
    -------
    ids : list
        List of IDs of the vertices.
    extent : list of pd.Series
        Extent of the vertices.
    verts : np.ndarray
        Array of vertices coordinates.
    lims : np.ndarray
        Limits of the vertices.

    Raises
    ------
    ValueError
        If `method` is not 'printed' or 'ids'.
        If `xy` is not None when `method` is 'printed'.

    Notes
    -----
    This function calculates the vertices of a rectangular region on a chip
    based on the specified method. If `method` is 'printed', the vertices are
    determined based on the printed spots on the chip. If `method` is 'ids',
    the vertices are determined based on the specified coordinates and context.

    Examples
    --------
    >>> ids, extent, verts, lims = get_vertices(mdata, 'chip1', method='printed')
    >>> ids, extent, verts, lims = get_vertices(mdata, 'chip1', method='ids', xy=(10, 10), context=2)
    """

    # Validate inputs
    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    chipnums = validate_chipnums(chipset, chipnum)
    if len(chipnums) > 1:
        raise ValueError("Only one chip number is allowed.")
    chipnum = chipnums[0]

    chip = chipset.chips[chipnum]
    chipmap = chip.get_welldata().reset_index()
    
    if method == 'printed':
        if xy is not None:
            raise ValueError('`xy` must be None when `method` is "printed"')
        ids = []
        extent = chipmap[['arr-x', 'arr-y']][~chipmap['barcode'].isna()].min(0), chipmap[['arr-x', 'arr-y']][~chipmap['barcode'].isna()].max(0)
        for x, y in [extent[0].values, (extent[0]['arr-x'], extent[1]['arr-y']), extent[1].values, (extent[1]['arr-x'], extent[0]['arr-y'])]:
            ids.append(chipmap.set_index(['arr-x', 'arr-y']).loc[(x, y), 'id'])
    elif method == 'ids':
        x, y = xy
        ids = chipmap[chipmap['arr-x'].isin(range(x - context, x + context + 1)) & chipmap['arr-y'].isin(range(y - context, y + context + 1))]['id'].values
        extent = chipmap[['arr-x', 'arr-y']][chipmap['id'].isin(ids)].agg(['min', 'max']).values
        extent = [pd.Series(i, index=['arr-x', 'arr-y']) for i in extent]
    else:
        raise ValueError('`method` must be "printed" or "ids"')

    pre_verts = np.concatenate([chip.array.verts[id] for id in ids])

    lims = np.vstack([pre_verts.min(0), pre_verts.max(0)]).T
    verts = np.array(list(it.product(*lims)))

    # Put in order that makes a Patch rectangle
    verts = verts[[0, 1, 3, 2]]

    return ids, extent, verts, lims


def get_well_outline_verts(mdata, chipnum):
    '''
    Get the outline vertices of the well array for a given chip number.

    Parameters
    ----------
    mdata : MuData
        The MuData object containing the spatial data.
    chipnum : int
        The chip number to extract outline vertices for.

    Returns
    -------
    outline_verts : np.ndarray
        An array of (x, y) coordinates representing the outline vertices of the well array,
        returned as a flat list of (x, y) tuples of length (nrows + 1)*(ncols + 1), in 
        column major order. 
    '''

    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    chipnums = validate_chipnums(chipset, chipnum)
    if len(chipnums) > 1:
        raise ValueError("Only one chip number is allowed.")
    chipnum = chipnums[0]

    chipset = mdata['xyz'].uns['survey']
    array = chipset.chips[chipnum].array

    # Get_wall_verts() returns horizontal walls first, then vertical walls
    wall_verts = array.get_wall_verts()
    nrows, ncols = array.arr_shape

    yverts, xverts = wall_verts[:nrows + 1], wall_verts[nrows + 1:]
    if not len(xverts) == ncols + 1:
        raise ValueError("Number of xverts does not match number of columns + 1")
    outline_verts = np.array(list(it.product([(i[0][0] + i[-1][0])//2 for i in xverts], [(i[0][1] + i[1][1])//2 for i in yverts])))
    
    return outline_verts


def get_niche_outlines(mdata, chipnum, niches, vertex_list, color=None, **kwargs):
    """
    Generates Matplotlib Polygons outlining contiguous "niches" of wells.

    Parameters:
    -----------
    mdata : MuData
        The MuData object containing the spatial data.
    chipnum : int
        The chip number to extract niches for.
    niches : dict
        Key: niche ID (str or int)
        Value: List of (row, col) tuples representing wells in the niche.
    vertex_list : list of tuples
        List of (x, y) coordinates for each vertex in the well array,
        ordered in column-major order.
    kwargs : dict
        Additional keyword arguments to pass to the Polygon constructor.

    Returns:
    --------
    dict
        Key: niche ID
        Value: List of Matplotlib Polygon objects outlining the niche.

    """
    default_color = 'black'

    if color is not None:
        if 'edgecolor' in kwargs:
            warnings.warn("Both 'color' and 'edgecolor' (in kwargs) params provided. Param 'color' will take precedence for edge colors.")
        if isinstance(color, dict):
            invalid_colors = []
            for seg_id, seg_color in color.items():
                if not mpl.colors.is_color_like(seg_color):
                    invalid_colors.append((seg_id, seg_color))
            if invalid_colors:
                error_msg = "Invalid colors for niches:\n"
                error_msg += "\n".join([f"niche ID: {seg_id}, Color: {seg_color}" for seg_id, seg_color in invalid_colors])
                raise ValueError(error_msg)
        elif mpl.colors.is_color_like(color):
            color = {seg_id: color for seg_id in niches.keys()}
        else:
            raise ValueError(f"Invalid color: {color}")
    else:
        if 'edgecolor' in kwargs:
            color = {seg_id: kwargs['edgecolor'] for seg_id in niches.keys()}
            kwargs.pop('edgecolor')  # Remove edgecolor from kwargs since we're using it in color dict
        else:
            color = {seg_id: default_color for seg_id in niches.keys()}
        

    chipset = mdata['xyz'].uns['survey']
    array = chipset.chips[chipnum].array
    grid_shape = array.arr_shape

    nrows, ncols = grid_shape
    stride = nrows + 1
    
    output_polygons = {}

    for seg_id, well_indices in niches.items():
        # 1. Identify all exterior edges
        edges = set()
        
        for r, c in well_indices:
            # Calculate vertex indices
            tl = c * stride + r
            bl = c * stride + (r + 1)
            tr = (c + 1) * stride + r
            br = (c + 1) * stride + (r + 1)
            
            # Counter-Clockwise edges
            well_edges = [
                (tl, bl), (bl, br), (br, tr), (tr, tl)
            ]
            
            for edge in well_edges:
                reverse_edge = (edge[1], edge[0])
                if reverse_edge in edges:
                    edges.remove(reverse_edge)
                else:
                    edges.add(edge)
        
        # 2. Stitch edges into loops
        # Use a dict of lists to handle vertices with multiple outgoing edges
        # (e.g., touching corners in a checkerboard pattern)
        adjacency = {}
        for u, v in edges:
            if u not in adjacency:
                adjacency[u] = []
            adjacency[u].append(v)
        
        loops = []
        
        while adjacency:
            # Start a new loop from an arbitrary remaining edge
            start_node = next(iter(adjacency))
            current_node = start_node
            
            loop_coords = []
            
            while True:
                loop_coords.append(vertex_list[current_node])
                
                # Get valid neighbors for the current node
                if current_node not in adjacency:
                    # This implies a broken path or non-closed geometry logic error,
                    # but strictly shouldn't happen in valid grid topology.
                    break
                    
                neighbors = adjacency[current_node]
                
                # Pop one outgoing edge. 
                # If there are multiple (touching corners), we just pick the last one added.
                next_node = neighbors.pop()
                
                # Clean up empty keys to prevent infinite loops in outer while
                if not neighbors:
                    del adjacency[current_node]
                
                current_node = next_node
                
                # If we return to start, the loop is closed
                if current_node == start_node:
                    break
            
            loops.append(loop_coords)

        # 3. Convert loops to Polygons
        poly_patches = []
        for loop in loops:
            poly = Polygon(loop, closed=True, edgecolor=color.get(seg_id, default_color), **kwargs)
            poly_patches.append(poly)
            
        output_polygons[seg_id] = poly_patches

    return output_polygons


def plot_niches(mdata, chipnum, niches, mod, subset=None, 
                  size=1.0, niche_col_name='spatial_niche', chipnum_col_name='rna:chip-num', 
                  plot_pies=True, plot_pies_kwargs=None,
                  plot_outlines=False, outline_fill_holes=True, outlines_kwargs=None,
                  legend=True,
                  plot_array=True, plot_array_kwargs=None,
                  plot_background=True, plot_background_kwargs=None):
    '''
    Plots spatial niches on a chip, with options for pie charts and outlines.

    Parameters
    ----------
    mdata : MuData
        The MuData object containing the spatial data.
    chipnum : int
        The chip number to plot niches for.
    niches : list
        List of niche identifiers to plot.
    mod : str, optional
        The modality in mdata where the spatial niche information is stored. With niche_col_name, will 
        be the name used for column that's transfered to/identified in the `xyz` modality.
    subset : dict, optional
        A dictionary specifying subsets of the data to plot, in the format {mod: {col_name: [values]}}. 
        Default is None (no subsetting).
    size : float, optional
        The size of the plotted points. Default is 1.0.
    niche_col_name : str, optional
        The name of the column in mdata.obs that contains spatial niche identifiers. 
        Default is 'spatial_niche'.
    chipnum_col_name : str, optional
        The name of the column in mdata.obs that contains chip number identifiers. 
        Default is 'rna:chip-num'.
    plot_pies : bool, optional
        Whether to plot pie charts representing niche composition. Default is True.
    plot_pies_kwargs : dict, optional
        Additional keyword arguments to pass to the pie chart plotting function. Default is None.
    plot_outlines : bool, optional
        Whether to plot outlines around niches. Default is False.
    outline_fill_holes : bool, optional
        Whether to fill holes in the niche outlines. Default is True.
    outlines_kwargs : dict, optional
        Additional keyword arguments to pass to the outline plotting function. Default is None.
    plot_array : bool, optional
        Whether to plot the well array. Default is True.
    plot_array_kwargs : dict, optional
        Additional keyword arguments to pass to the array plotting function. Default is None.
    plot_background : bool, optional
        Whether to plot the background hoodmap. Default is True.
    plot_background_kwargs : dict, optional
        Additional keyword arguments to pass to the background plotting function. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the plot.
    ax : matplotlib.axes.Axes
        The Matplotlib axes object containing the plot.

    Notes
    -----
    This function subsets the data for the specified chip number and creates a copy in memory. Make sure
    that there is enough memory to hold this copy, especially if the original mdata is large. 


    TODO:
    - Avoid creating copy in memory using try/except/finally block to attempt plot and clean up pulled columns.
    '''

    if 'legend' in plot_pies_kwargs and plot_pies_kwargs['legend'] != legend:
        warnings.warn("plot_pies_kwargs contains 'legend' set to False, but 'legend' is set to True and takes precedence.")
    

    none_id = str(uuid.uuid4())[:8]
    other_id = str(uuid.uuid4())[:8]
    filter_col_name = str(uuid.uuid4())[:8]

    submdata = mdata[mdata.obs[chipnum_col_name] == chipnum].copy()

    transfer_col_name = f'{mod}.{niche_col_name}'

    if transfer_col_name not in submdata['xyz'].obs.columns:
        transfer_obs(submdata, mods=(mod, 'xyz'), columns=[niche_col_name], overwrite=True)

    all_niches = submdata[mod].obs[niche_col_name].cat.categories   

    mapper = dict(it.product(all_niches.difference(niches), (other_id,)))
    mapper.update(dict(zip(niches, niches)))

    submdata['xyz'].obs[filter_col_name] = submdata['xyz'].obs[transfer_col_name].map(mapper).fillna(none_id).astype('category')

    submdata.pull_obs(mods='xyz', columns=[filter_col_name])

    reset_meta_keys(submdata['xyz'], [transfer_col_name, filter_col_name])
    for key in [transfer_col_name, filter_col_name]:
        add_colors(submdata['xyz'], key, overwrite=True)
    
    if plot_outlines:
        chip = submdata['xyz'].uns['survey'].chips[chipnum]
        outline_verts = get_well_outline_verts(mdata, chipnum=chipnum)
        welldata = chip.get_welldata()
        color_dict = get_cat_dict(submdata['xyz'], filter_col_name, prop='color')
        if outline_fill_holes:
            niches2locs = submdata['xyz'].obs[['arr-row', 'arr-col', transfer_col_name]].dropna().drop_duplicates().reset_index(drop=True)
            niches2locs = niches2locs.groupby(transfer_col_name, observed=True)[['arr-row', 'arr-col']].apply(lambda df: df.values.tolist()).to_dict()
            
            zero_arr = np.zeros(chip.array.arr_shape, dtype=int)
            niches2wells = {}
            welldata_locs = welldata.reset_index().set_index(['arr-row', 'arr-col'])['id']
            for spniche, locs in niches2locs.items():
                niche_arr = zero_arr.copy()
                for r, c in locs:
                    niche_arr[r, c] = 1
                # Fill holes
                niche_arr = binary_fill_holes(niche_arr)
                # Get well indices for the niche
                well_indices = list(map(tuple,np.argwhere(niche_arr)))
                niches2wells[spniche] = welldata_locs.loc[well_indices].values.tolist()

        else:
            niches2wells = submdata['xyz'].obs[['id', transfer_col_name]].dropna().drop_duplicates().reset_index(drop=True)
            niches2wells = niches2wells.groupby(transfer_col_name, observed=True)['id'].apply(list).to_dict()

        xy = chip.array.wells[['x', 'y']]
        wall_dist = (chip.array.w/chip.array.pitch)
        half_well_dist = 0.5*(chip.array.s/chip.array.pitch)
        lims = dict(zip(['x', 'y'], zip(xy.min() - half_well_dist - wall_dist, xy.max() + half_well_dist + wall_dist)))
        lerper = get_lerper((chip.array.lims['x'], chip.array.lims['y']), (lims['x'], lims['y']))
        
        niche_locs = {spniche: welldata.loc[niches2wells[spniche]][['arr-row', 'arr-col']].values for spniche in niches}
        outline_verts_lerped = [lerper(p) for p in outline_verts]

        protected_outlines_kwargs = {'chipnum', 'niches', 'vertex_list', 'fill'}
        default_config = {'fill': False, 'color': color_dict, 'linewidth': 1}
        outlines_kwargs = get_config(outlines_kwargs, default_config, protected=protected_outlines_kwargs)
        niche_outlines = get_niche_outlines(mdata, chipnum=chipnum, niches=niche_locs, 
                                               vertex_list=outline_verts_lerped, **outlines_kwargs)

    fig, ax = subplots(1, fss=10)
    if plot_array:
        default_config = {'walls': True, 'units': 'w'}
        config = get_config(plot_array_kwargs, default_config, protected={'chip-num', 'units', 'ax'})
        ax = arrplot(submdata, chipnum=chipnum, ax=ax, **config)
    if plot_background:
        circleprops={'facecolor': 'darkgray'}
        default_config = {'units': 'w', 'subset': subset, 'circleprops': circleprops, 'color': None, 'size': {1: size, 2: size}}
        config = get_config(plot_background_kwargs, default_config, protected={'chip-num', 'units', 'ax'})
        ax = hoodmap(submdata, chipnum=chipnum, ax=ax, **config)

    if plot_pies or plot_outlines:
        pulled_filter_col_name = f'xyz:{filter_col_name}'
        submdata_exclude = submdata[~submdata.obs[pulled_filter_col_name].isin([other_id, none_id])]
        if legend:
            show_all_cats = submdata_exclude.obs[pulled_filter_col_name].cat.categories.difference([other_id, none_id])
            default_legend_params = {'pos': 'TR2', 'show_all_cats': show_all_cats}

    if plot_pies:
        default_config={'legend_params': default_legend_params, 'units': 'w', 'subset': subset, 'size': size}
        if legend:
            default_config['legend'] = True
        config = get_config(plot_pies_kwargs, default_config, protected={'size', 'chip-num', 'units', 'color', 'ax'})
        ax = hoodmap(submdata_exclude, chipnum=chipnum, color=filter_col_name, ax=ax, **config)

    if plot_outlines:
        for spniche, polygons in niche_outlines.items():
            for poly in polygons:
                ax.add_patch(poly)
        if not plot_pies and legend:
            config = get_add_legend_pm().get_params(default_legend_params)
            decorate_scatter(ax, config=config, plot_type='legend', cdict=color_dict)

    return fig, ax