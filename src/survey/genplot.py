# Built-ins
from pathlib import Path
import re
from typing import (
    Optional, Tuple, List, Any, Union, Dict
)
import colorsys
from numbers import Number

# Standard libs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Survey libs
from survey.genutils import make_logspace, get_config

inverted_rcparams = {
    # Figure
    "figure.facecolor": (0.0, 0.0, 0.0, 0.0),  # transparent figure background
    "figure.edgecolor": "white",

    # Axes
    "axes.facecolor": (0.0, 0.0, 0.0, 0.0),  # transparent axes background
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",

    # Ticks
    "xtick.color": "white",
    "ytick.color": "white",

    # Text
    "text.color": "white",

    # Grid
    "grid.color": "gray",
    "grid.alpha": 0.5,

    # Legend
    "legend.facecolor": (0.0, 0.0, 0.0, 0.0), # transparent legend background
    "legend.edgecolor": "white",
    "legend.labelcolor": "white",
    
    # Saving figures
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0), # transparent background when saving
    "savefig.edgecolor": (0.0, 0.0, 0.0, 0.0),
    "savefig.transparent": True,
}


def loglog_hist(vals: np.ndarray,
                binminmax: Tuple[Number, Number],
                numbins: int = 100,
                vline: Optional[Number] = None,
                title: Optional[str] = None,
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a log-log histogram of a list-like of values.

    Parameters
    ----------
    vals : np.ndarray
        An array of values to be binned and plotted.
    binminmax : tuple of (Number, Number)
        A tuple specifying the minimum and maximum range for the histogram bins.
        The minimum value must be greater than 0 for a log scale.
    numbins : int, optional
        The number of bins to create for the histogram.
    vline : Number, optional
        If provided, draws a vertical line at this x-coordinate.
    title : str, optional
        The title for the plot's axes.
    ax : plt.Axes, optional
        An existing Matplotlib Axes object to plot on. If None, a new one is created.

    Returns
    -------
    plt.Axes
        The Axes object containing the log-log histogram.
    """ 
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        
    ax.hist(vals, bins=make_logspace(binminmax[0], binminmax[1], numbins))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(which='both', alpha=0.5)
    ax.set_title(title)
    
    if vline is not None:
        ylim = ax.get_ylim()
        ax.vlines(vline, ylim[0], ylim[1], color='k')
        
    return ax


def subplots(nplots: int,
             ncols: Optional[int] = None,
             cols: Optional[int] = None,
             ar: Optional[float] = None,
             fss: Optional[float] = None,
             split_each: Optional[Tuple[int, int]] = None,
             hrs: Optional[List[float]] = None,
             wrs: Optional[List[float]] = None,
             as_seq: bool = False,
             **kwargs: Any) -> Tuple[mpl.figure.Figure, Union[mpl.axes.Axes, np.ndarray]]:
    """Create a matplotlib figure and axes objects.

    Creates a grid of subplots with a specified number of plots,
    maximum columns, and aspect ratio.

    Parameters
    ----------
    nplots : int
        The total number of subplots.
    ncols : int, optional
        The maximum number of columns in the grid, by default 4.
    cols : Optional[int], optional
        Alias for ncols, by default None.
    ar : float, optional
        The aspect ratio (width:height) of each subplot, by default 1.
    fss : Optional[float], optional
        The scaling factor for the figure size, by default 4.
    split_each : Optional[Tuple[int, int]], optional
        If provided, each subplot will be split into a grid of the given shape, by default None.
    hrs : Optional[List[float]], optional
        The height ratios for the split subplots, by default None.
    wrs : Optional[List[float]], optional
        The width ratios for the split subplots, by default None.
    as_seq : bool, optional
        If nplots=1 and split_each is None, return an ndarray of Axes objects instead
        of a single object.
    **kwargs : Any
        Keyword arguments to pass to `matplotlib.pyplot.subplots`.

    Returns
    -------
    Tuple[mpl.figure.Figure, Union[mpl.axes.Axes, np.ndarray]]
        The created Figure object and the array of created Axes objects.
        Extra axes are deleted if `nplots % ncols != 0`.

    Notes
    -----
    If `nplots % ncols != 0`, the extra axes in the last row are deleted.
    """

    if cols is not None:
        ncols = cols

    if ncols is None:
        ncols = 4
    if ar is None:
        ar = 1.0
    if fss is None:
        fss = 4.0

    if nplots == 1:
        fig, axes = plt.subplots(
            1, 1, figsize=(1 * ar * fss, 1 * fss), **kwargs
        )
    else:
        # Calculate the number of rows needed
        nrows = np.ceil(nplots / ncols).astype(int)

        # Create the figure and axes
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * ar * fss, nrows * fss),
            **kwargs,
        )

        if (
            nplots % ncols != 0
        ):  # If nplots % ncols != 0, delete the extra axes
            for ax in axes.flatten()[nplots:]:
                fig.delaxes(ax)

    if split_each is not None:
        if isinstance(axes, mpl.axes.Axes):
            # If only one subplot, convert to a 2D array with one element
            axes = np.array([[axes]])
        # Split each subplot into a grid
        orig_shape = axes.shape
        axes = np.ravel(axes)

        for i in range(len(axes)):
            # Store the sharex, sharey, then remove the original subplot
            # sharex, sharey = axes[i].sharex, axes[i].sharey

            try:  # In the event that it was removed because of nplots % cols != 0
                axes[i].remove()
            except KeyError:
                continue

            # Create a new grid of subplots
            gs = mpl.gridspec.GridSpecFromSubplotSpec(
                *split_each,
                subplot_spec=axes[i].get_subplotspec(),
                height_ratios=hrs,
                width_ratios=wrs,
            )

            # Create the subplots
            axes[i] = np.array(
                [fig.add_subplot(gs[j]) for j in range(np.prod(split_each))]
            ).reshape(split_each)

        axes = axes.reshape(orig_shape)

    if nplots == 1 and split_each is None and as_seq:
        # axes is an Axes object, but user wants a sequence
        axes = np.array([axes])
    return fig, axes


def get_inset_ax_params(pos: str,
                        ax: plt.Axes,
                        width: float,
                        height: float) -> Tuple[List[float], float, Tuple[str, ...], Tuple[bool, ...]]:
    """
    Calculates parameters for creating an inset axes within a parent axes.

    Parameters
    ----------
    pos : str
        The corner position for the inset axes ('lower left', 'upper right', etc.).
    ax : plt.Axes
        The parent axes object.
    width : float
        The width of the inset axes, as a fraction of the parent axes' width.
    height : float
        The height of the inset axes, as a fraction of the parent axes' height.

    Returns
    -------
    tuple
        A tuple containing:
        - `rect`: A list `[left, bottom, width, height]` for the new axes.
        - `yadjust_pos`: A float for adjusting y-tick positions.
        - `yk`: A tuple of y-tick parameter keys.
        - `yv`: A tuple of y-tick parameter boolean values.

    Raises
    ------
    ValueError
        If `pos` is not a valid corner position.
    """
    # yk, yv = y-tick parameters keys, values
    yk = "labelleft", "labelright", "left", "right"

    if pos == 'lower left':
        rect = [ax.get_position().x0, ax.get_position().y0, width, height]
        yadjust_pos = 1
        yv = (False, True, False, True)
    elif pos == 'lower right':
        rect = [ax.get_position().x1 - width, ax.get_position().y0, width, height]
        yadjust_pos = 1
        yv = (True, False, True, False)
    elif pos == 'upper left':
        rect = [ax.get_position().x0, ax.get_position().y1 - height, width, height]
        yadjust_pos = 0.95
        yv = (False, True, False, False)
    elif pos == 'upper right':
        rect = [ax.get_position().x1 - width, ax.get_position().y1 - height, width, height]
        yadjust_pos = 0.95
        yv = (True, False, False, False)
    else:
        raise ValueError(
            "Invalid pos. Choose one of: 'lower left', "
            "'lower right', 'upper left', or 'upper right'."
        )
    return rect, yadjust_pos, yk, yv


def cbar_in_axes(fig: mpl.figure.Figure,
                 ax: mpl.axes.Axes,
                 pos: Optional[str] = None,
                 shape: Optional[Tuple[float, float]] = None,
                 cax: Optional[mpl.axes.Axes] = None) -> Tuple[mpl.axes.Axes, float, Dict[str, bool]]:
    """
    Places a colorbar as an inset inside a Matplotlib axes.

    Parameters
    ----------
    fig : mpl.figure.Figure
        The figure object.
    ax : mpl.axes.Axes
        The axes object to place the colorbar in.
    pos : str, optional
        Position of the colorbar. One of 'lower left', 'lower right',
        'upper left', or 'upper right'. Defaults to 'lower right'.
    shape : tuple of (float, float), optional
        The (width, height) of the colorbar as a fraction of the parent axes'
        dimensions. Defaults to (0.04, 0.12).
    cax : mpl.axes.Axes, optional
        An existing axes to use for the colorbar. If None, a new one is created.

    Returns
    -------
    tuple
        A tuple containing:
        - The colorbar axes object.
        - The y-adjustment position for ticks.
        - A dictionary of tick parameters.

    Raises
    ------
    ValueError
        If `pos` is invalid.
    TypeError
        If `cax` is not a valid Axes object.
    """

    shape = (0.04, 0.12) if shape is None else np.array(shape)

    ax_size = (ax.get_position().x1 - ax.get_position().x0,
               ax.get_position().y1 - ax.get_position().y0)
    
    cbar_w = shape[0] * ax_size[0]
    cbar_h = shape[1] * ax_size[1]
    
    pos = "lower right" if pos is None else pos

    rect, yadjust_pos, yk, yv = get_inset_ax_params(pos, ax, cbar_w, cbar_h)

    if cax is None:
        cax = fig.add_axes(rect)
    elif isinstance(cax, mpl.axes.Axes):
        cax.set_position(rect)
    else:
        raise TypeError("cax must be a matplotlib Axes object or None.")
    
    tick_params = dict(zip(yk, yv))

    return cax, yadjust_pos, tick_params


def adj_light(color: Union[str, Tuple[float, ...]], amount: float = 0.5) -> Tuple[float, float, float]:
    """
    Lightens a given color.

    The function adjusts the luminosity of the color in HLS space.

    Parameters
    ----------
    color : str or tuple
        The input color. Can be a Matplotlib color string, hex string, or RGB tuple.
    amount : float, optional
        The amount to lighten the color. 0.0 gives the original color, 1.0 gives white.

    Returns
    -------
    tuple
        The lightened color as an RGB tuple.
    """
    
    try:
        c = mpl.colors.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_contrast_color(hex_color: str, lum_thresh: int = 186) -> str:
    """
    Calculates a high-contrast color (black or white) for a given hex color.

    This is useful for determining a legible text color to overlay on a
    colored background.

    Parameters
    ----------
    hex_color : str
        The background color as a hexadecimal string (e.g., '#FFFFFF').
    lum_thresh : int, optional
        The luminance threshold (0-255) for switching between black and white text.

    Returns
    -------
    str
        '#000000' (black) or '#FFFFFF' (white).
    """
    r, g, b = np.array(mpl.colors.hex2color(hex_color))*255
    rgbsum = (r*0.299 + g*0.587 + b*0.114)
    return '#000000' if rgbsum > lum_thresh else '#FFFFFF'


def natural_sort_key(text: str) -> List[Union[str, int]]:
    """
    Creates a key for natural sorting (e.g., 'item1', 'item2', 'item10').

    Parameters
    ----------
    text : str
        The string to be converted into a sortable key.

    Returns
    -------
    list
        A list of mixed strings and integers for sorting.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]


def create_gif_from_pngs(png_dir: Union[str, Path],
                         output_gif: Union[str, Path],
                         duration: int = 200,
                         loop: int = 0) -> None:
    """
    Creates an animated GIF from a directory of PNG files.

    The PNG files are sorted naturally before being combined into the GIF.

    Parameters
    ----------
    png_dir : str or Path
        The directory containing the PNG files.
    output_gif : str or Path
        The path for the output GIF file.
    duration : int, optional
        The duration (in milliseconds) for each frame.
    loop : int, optional
        The number of times the GIF should loop (0 means infinite).
    """
    png_files = sorted(Path(png_dir).glob('*.png'), key=natural_sort_key)
    images = [Image.open(f) for f in png_files]
    
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )


def id_axes(ax: mpl.axes.Axes, 
            lim: tuple[float, float] | None = None, 
            tix: list | None = None) -> mpl.axes.Axes:
    """Make x and y axes identical.

    Creates identical x and y axes with the same range, tick locations,
    and tick labels, while ensuring all data remains visible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    lim : tuple[float, float], optional
        A 2-tuple of (min_limit, max_limit) to use for both axes. If None,
        limits are computed from the existing axes limits to include all data.
    tix : list, optional
        A list of tick locations to apply to both axes. If None, the ticks
        from the axis with more ticks are used.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
    """
    if lim is None:
        lim = min(min(ax.get_xlim()), min(ax.get_ylim())), max(max(ax.get_xlim()), max(ax.get_ylim()))

    ax.set_xlim(lim)
    ax.set_ylim(lim)

    if tix is None:
        tix = ax.get_xticks() if len(ax.get_xticks()) > len(ax.get_yticks()) else ax.get_yticks()
    
    tix = np.asarray(tix)
    ax.set_xticks(ticks=tix)
    ax.set_yticks(ticks=tix)

    # Create integer labels for whole numbers, float labels otherwise
    is_float = (tix % 1).astype(bool)
    tix_labels = np.where(is_float, tix.astype(str), tix.astype(int).astype(str))
    
    ax.set_xticklabels(labels=tix_labels)
    ax.set_yticklabels(labels=tix_labels)
    
    return ax


def boxstrip(df, x, y, hue=None, ax=None, box_kws=None, strip_kws=None, common_kws=None):
    """Create a boxplot overlaid with a stripplot.

    This function uses seaborn to create a boxplot and a stripplot on the
    same axes. It provides a convenient way to visualize the distribution
    of a numerical variable across different categories.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the plotting data.
    x : str
        The name of the column in `df` to be used for the x-axis.
    y : str
        The name of the column in `df` to be used for the y-axis.
    hue : str, optional
        The name of the column in `df` for color encoding. Defaults to None.
    ax : mpl.axes.Axes, optional
        The matplotlib axes on which to draw the plot. If None, a new figure
        and axes are created. Defaults to None.
    box_kws : dict, optional
        Keyword arguments to pass to `seaborn.boxplot`.
    strip_kws : dict, optional
        Keyword arguments to pass to `seaborn.stripplot`.
    common_kws : dict, optional
        Keyword arguments common to both plots. Will override
        any conflicting keys in `box_kws` and `strip_kws`.

    Returns
    -------
    mpl.axes.Axes
        The matplotlib axes object containing the plot.
    """

    if ax is None:
        fig, ax = subplots(1, ar=3, fss=6)
    if hue == x:
        dodge = False
    else:
        dodge = True

    if box_kws is None:
        box_kws = {}
    if strip_kws is None:
        strip_kws = {}

    if common_kws:
        box_kws.update(common_kws)
        strip_kws.update(common_kws)
    # print(box_kws)

    default_box_kws = {'dodge': dodge, 'showfliers': False, 'saturation': 1.0}
    default_strip_kws = {'dodge': dodge, 'size' : 5, 'jitter' : 0.1, 'linewidth': 0.5, 'legend': False}

    box_kws = get_config(box_kws, default_box_kws)
    strip_kws = get_config(strip_kws, default_strip_kws)

    ax = sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, **box_kws)
    ax = sns.stripplot(data=df, x=x, y=y, hue=hue, ax=ax, **strip_kws)

    return ax


def set_tick_params(ax: mpl.axes.Axes, 
                    **kwargs: Any) -> mpl.axes.Axes:
    """Set tick parameters with added support for text alignment.

    This function is a wrapper around `matplotlib.axes.Axes.tick_params`
    that adds support for horizontal ('ha') and vertical ('va') alignment
    of tick labels, which is not natively supported by `tick_params`.

    Parameters
    ----------
    ax : mpl.axes.Axes
        The matplotlib axes object to modify.
    **kwargs : Any
        Keyword arguments. These include standard `ax.tick_params` arguments
        plus 'ha'/'horizontalalignment' and 'va'/'verticalalignment'.
        The `axis` kwarg ('x', 'y', or 'both') determines which tick labels
        are affected by alignment settings.

    Returns
    -------
    mpl.axes.Axes
        The modified matplotlib axes object.

    Notes
    -----
    Alignment keywords ('ha', 'horizontalalignment', 'va', 'verticalalignment')
    are extracted and applied separately using `ax.set_xticklabels` and
    `ax.set_yticklabels`. The remaining keywords are passed directly to
    `ax.tick_params`.
    """
    label_kws = {}
    # These are common Text properties that can be passed to set_ticklabels
    label_prop_keys = [
        'alpha', 'backgroundcolor', 'bbox', 'color', 'fontfamily', 'fontname',
        'fontproperties', 'fontsize', 'fontstyle', 'fontweight', 'ha',
        'horizontalalignment', 'label', 'linespacing', 'ma', 'multialignment',
        'name', 'position', 'rotation', 'rotation_mode', 'size', 'style',
        'transform', 'va', 'verticalalignment', 'visible', 'wrap', 'zorder'
    ]
    
    # Separate label properties from other tick_params properties
    for key in list(kwargs.keys()):
        # `labelsize` in tick_params corresponds to `size` in set_ticklabels
        if key == 'labelsize':
            label_kws['size'] = kwargs.pop(key)
        elif key in label_prop_keys:
            label_kws[key] = kwargs.pop(key)

    ax.tick_params(**kwargs)

    if label_kws:
        # Normalize 'ha' and 'va' keys
        if 'horizontalalignment' in label_kws:
            label_kws['ha'] = label_kws.pop('horizontalalignment')
        if 'verticalalignment' in label_kws:
            label_kws['va'] = label_kws.pop('verticalalignment')

        axis = kwargs.get('axis', 'both')
        if axis in ['x', 'both']:
            # To prevent UserWarning about FixedLocator, we get and set ticks
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), **label_kws)
        if axis in ['y', 'both']:
            # To prevent UserWarning about FixedLocator, we get and set ticks
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticklabels(), **label_kws)

    return ax




