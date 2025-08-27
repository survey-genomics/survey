# Built-ins
from pathlib import Path
import re
from typing import (
    Optional, Tuple, List, Any, Union, Dict
)

# Standard libs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Special libs
import colorsys
from PIL import Image


# Survey libs
from .genutils import make_logspace


def loglog_hist(
        vals, 
        binminmax, 
        numbins=100, 
        vline=None, 
        title=None, 
        ax=None):
    '''
    Plot a log-lob histogram of a list-like of values.
    
    `vals`: list-like of values, fed directly to ax.hist()
    `binminmax`: 2-tuple of x axis bins minimum (should be > 0, since log-bins) and maximum
    `numbins`: number of bins for the x axis of the histogram
    `vline`: (optional) x location of a black vertical line
    `title`: (optional) title for the Axes
    `ax`: (optional) matplotlib.pyplot.Axes on which to plot the histogram
    
    returns: matplotlib.pyplot.Axes with plotted loglog histogram
    ''' 
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        
    ax.hist(vals, bins=make_logspace(binminmax[0], binminmax[1], numbins))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(which='both', alpha=0.5)
    ax.set_title(title)
    
    if vline is None:
        ylim = ax.get_ylim()
        ax.vlines(vline, ylim[0], ylim[1], color='k')
        
    return ax


def subplots(
    nplots: int,
    ncols: int = None,
    cols: Optional[int] = None,
    ar: float = None,
    fss: Optional[float] = None,
    split_each: Optional[Tuple[int, int]] = None,
    hrs: Optional[List[float]] = None,
    wrs: Optional[List[float]] = None,
    as_seq: bool = False,
    **kwargs: Any,
) -> Tuple[mpl.figure.Figure, Union[mpl.axes.Axes, np.ndarray]]:
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


def get_inset_ax_params(pos, ax, width, height):
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


def cbar_in_axes(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    pos: Optional[str] = None,
    shape: Optional[Tuple[float, float]] = None,
    cax: Optional[mpl.axes.Axes] = None,
) -> Tuple[mpl.axes.Axes, float, Dict[str, bool]]:
    """Place a colorbar inside a matplotlib axes.

    Parameters
    ----------
    fig : mpl.figure.Figure
        The figure object.
    ax : mpl.axes.Axes
        The axes object to place the colorbar in.
    pos : str, optional
        Position of the colorbar. One of 'lower left', 'lower right',
        'upper left', or 'upper right'. Default is 'lower right'.
    cbar_size : float, optional
        Size of the colorbar. Default is 0.1.
    cbar_aspect : float, optional
        Aspect ratio of the colorbar. Default is 0.2.
    cax : mpl.axes.Axes, optional
        An existing axes for the colorbar. If None, a new one is created.

    Returns
    -------
    Tuple[mpl.axes.Axes, float, Dict[str, bool]]
        A tuple containing:
        - The colorbar axes.
        - The y-adjustment position.
        - A dictionary of tick parameters.

    Raises
    ------
    ValueError
        If `pos` is invalid.
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


def adj_light(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    
    try:
        c = mpl.colors.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_contrast_color(hex, lum_thresh=186):

    """
    Calculate a contrast color for a given hexadecimal color.

    This function calculates the luminance of the input color and returns either white or black,
    depending on whether or not the luminance is greater than or less the defined threhsold.

    Parameters
    ----------
    hex : str
        The hexadecimal color string (e.g., "#FFFFFF" for white).
    lum_thresh : int, optional
        The luminance threshold for determining the contrast color. Default is 186.

    Returns
    -------
    str
        The contrasting color as a hexadecimal string. Returns "#000000" for black if the luminance 
        of the input color is greater than the threshold, and "#FFFFFF" for white otherwise.
    """
    r, g, b = np.array(mpl.colors.hex2color(hex))*255
    rgbsum = (r*0.299 + g*0.587 + b*0.114)
    return '#000000' if rgbsum > lum_thresh else '#FFFFFF'


def natural_sort_key(text):
    """Convert a string to a list of mixed strings and integers for natural sorting"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]


def create_gif_from_pngs(png_dir, output_gif, duration=200, loop=0):
    """
    Create GIF from PNG files in a directory.
    
    Parameters:
    - png_dir: Path to directory containing PNG files
    - output_gif: Output GIF file path
    - duration: Duration between frames in milliseconds
    - loop: Number of loops (0 = infinite)
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

