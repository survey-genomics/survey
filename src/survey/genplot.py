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
from PIL import Image

# Survey libs
from survey.genutils import make_logspace


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

