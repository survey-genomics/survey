# Built-ins
from copy import deepcopy
from typing import Dict, Tuple, Any

# Standard libs
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Survey libs
from survey.genutils import ParamManager
from survey.genplot import cbar_in_axes, get_contrast_color
    

MARGIN = 0.02

CORNER_POSITIONS = {
    "upper left": (MARGIN, 1 - MARGIN),
    "upper right": (1 - MARGIN, 1 - MARGIN),
    "lower left": (MARGIN, MARGIN),
    "lower right": (1 - MARGIN, MARGIN)
}

# Determine horizontal and vertical alignment based on the corner
HA_VALUES = {
    "upper left": "left",
    "upper right": "right",
    "lower left": "left",
    "lower right": "right"
}

VA_VALUES = {
    "upper left": "top",
    "upper right": "top",
    "lower left": "bottom",
    "lower right": "bottom"
}

def get_text_position_vals(label_pos: str) -> Tuple[Tuple[float, float], str, str]:
    """
    Retrieves coordinates and alignment for a predefined corner position.

    Parameters
    ----------
    label_pos : str
        The corner position name (e.g., 'upper left').

    Returns
    -------
    tuple
        A tuple containing:
        - (x, y) coordinates for the position.
        - Horizontal alignment string ('left' or 'right').
        - Vertical alignment string ('top' or 'bottom').

    Raises
    ------
    ValueError
        If `label_pos` is not a valid corner position.
    """

    if label_pos not in CORNER_POSITIONS:
        raise ValueError(
            f"Invalid label position: {label_pos}. "
            f"Must be one of {list(CORNER_POSITIONS.keys())}."
            )
    
    pos = CORNER_POSITIONS[label_pos]
    ha = HA_VALUES[label_pos]
    va = VA_VALUES[label_pos]
    return pos, ha, va


def get_add_plotlabel_pm() -> ParamManager:
    """
    Gets a ParamManager for adding a text label to a plot corner.

    Returns
    -------
    ParamManager
        A `ParamManager` instance pre-configured for `matplotlib.axes.Axes.text`
        to be used as a plot label.
    """

    def _pos_setter(v):
        if v is not None:
            return dict(zip(['x', 'y'], v))
        return {}

    def _invert_setter(v):
        if v is True:
            return {'color': 'white'}
        elif v is False:
            return {'color': 'black'}
        return {}

    pos, ha, va = get_text_position_vals('upper right')

    # param, value, type, prop, setter, error
    defaults = [
        ['pos', pos, 'm', False, _pos_setter, []],
        ['size', 12, 'd', None, None, []],
        ['color', 'black', 'd', None, None, []],
        ['invert', None, 'm', False, _invert_setter, []],
        ['ha', ha, 'd', None, None, []],
        ['va', va, 'd', None, None, []],
    ]

    func = mpl.axes.Axes.text
    
    error_on = {
        's': "Parameter 's' will be auto-set to the label for plot labels.",
        'transform': "Parameter 'transform' will be auto-set to ax.transAxes for plot labels."
        }
    
    pm = ParamManager(defaults, func=func, error_on=error_on)

    return pm


def get_add_label_scatter_pm() -> ParamManager:
    """
    Gets a ParamManager for adding labels to scatter plot centroids.

    Returns
    -------
    ParamManager
        A `ParamManager` instance pre-configured for `matplotlib.axes.Axes.text`
        to be used for labeling scatter plot clusters.
    """

    def _label_contrast_error(label_contrast, bbox):
        if label_contrast is True and bbox.pop('facecolor', None) is not None:
            return True
        return False

    label_contrast_error_dict = {
        'on': _label_contrast_error,
        'message': "Parameter 'label_contrast' will be auto-set to True for plot labels."
        }
    
    def _match_color_setter(v):
        if v is True:
            return {'color': None}
        else:
            return {}

    # param, value, type, prop, setter, error
    defaults = [
        ['fontsize', 12, 'd', None, None, []],
        ['fontweight', 'bold', 'd', None, None, []],
        ['bbox', {'boxstyle': 'round', 'facecolor': 'white'}, 'd', None, None, []],
        ['ha', 'center', 'd', None, None, []],
        ['va', 'center', 'd', None, None, []],
        ['label_contrast', None, 'a', None, None, [('bbox', label_contrast_error_dict)]],
        ['match_color', None, 'm', True, _match_color_setter, []],
    ]

    func = mpl.axes.Axes.text
    
    error_on = {
        's': "Parameter 's' will be auto-set to the data labels for labeled scatter.",
        'x': "Parameter 'x' will be auto-set to the centroid x-coordinate of each data label for labeled scatter.",
        'y': "Parameter 'y' will be auto-set to the centroid y-coordinate of each data label for labeled scatter.",
        'transform': "Parameter 'transform' must not be supplied for labeled scatter."
        }
    
    pm = ParamManager(defaults, func=func, error_on=error_on)

    return pm


def get_add_cbar_pm() -> ParamManager:
    """
    Gets a ParamManager for adding a colorbar to a plot.

    Returns
    -------
    ParamManager
        A `ParamManager` instance configured for creating a colorbar with
        `survey.genplot.cbar_in_axes`.
    """

    def _invert_setter(v):
        if v is True:
            return {'label_color': 'white'}
        elif v is False:
            return {'label_color': 'black'}
        return {}

    # param, value, type, prop, setter, error
    defaults = [
        ['pos', 'lower right', 'a', None, None, []],
        ['shape', (0.04, 0.12), 'a', None, None, []],
        ['label_size', 12, 'a', None, None, []],
        ['label_color', 'black', 'a', None, None, []],
        ['invert', None, 'm', False, _invert_setter, []],
    ]

    func = None
    
    error_on = {}
    
    pm = ParamManager(defaults, func=func, error_on=error_on)

    return pm


def get_add_legend_pm() -> ParamManager:
    """
    Gets a ParamManager for adding a legend to a plot.

    Returns
    -------
    ParamManager
        A `ParamManager` instance pre-configured for `matplotlib.axes.Axes.legend`.
    """

    def _marker_setter(v):
        if v is False:
            return_dict ={
                'handletextpad': 0,
                'handlelength': 0,
            }
        elif v is True:
            return_dict = {
                'handletextpad': None,
                'handlelength': None,
            }
        else:
            return {}
        return return_dict

    # param, value, type, prop, setter, error
    defaults = [
        ['show_all_cats', None, 'a', None, None, []],
        ['show_marker', False, 'm', True, _marker_setter, []],
        ['labelcolor', 'linecolor', 'd', None, None, []],
        ['prop', {'weight': 'bold', 'size': 12}, 'd', None, None, []],
        ['loc', 'upper left', 'd', None, None, []],
        ['bbox_to_anchor', CORNER_POSITIONS['upper left'], 'd', None, None, []]
    ]

    func = mpl.axes.Axes.legend
    
    error_on = {
        'fontsize': "Avoid using 'fontsize' in the legend parameters. Use 'prop={'size': <size>}' instead."
        }
    
    pm = ParamManager(defaults, func=func, error_on=error_on)

    return pm


def get_pm(plot_type: str) -> ParamManager:
    """
    Get the ParamManager for a specific plot decoration type.

    This function acts as a factory, returning a pre-configured `ParamManager`
    for different types of plot decorations.

    Parameters
    ----------
    plot_type : {'plot_label', 'label_scatter', 'cbar', 'legend'}
        The type of plot decoration.

    Returns
    -------
    ParamManager
        The corresponding `ParamManager` instance.

    Raises
    ------
    ValueError
        If `plot_type` is not a valid decoration type.
    """
    
    pm_dict = {
        'plot_label': get_add_plotlabel_pm(),
        'label_scatter': get_add_label_scatter_pm(),
        'cbar': get_add_cbar_pm(),
        'legend': get_add_legend_pm()
    }
    
    if plot_type not in pm_dict:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be one of {list(pm_dict.keys())}.")
    
    return pm_dict[plot_type]


def decorate_scatter(ax: plt.Axes,
                     config: Dict,
                     plot_type: str,
                     **kwargs: Any) -> plt.Axes:
    """
    Applies a specific decoration to a scatter plot's Axes object.

    This function dispatches to a specific decoration function based on `plot_type`.

    Parameters
    ----------
    ax : plt.Axes
        The Matplotlib Axes object to decorate.
    config : dict
        A dictionary of configuration parameters for the decoration.
    plot_type : {'plot_label', 'label_scatter', 'cbar', 'legend'}
        The type of decoration to apply.
    **kwargs
        Additional arguments required by the specific decoration function, such as
        `label`, `positions`, `cdict`, `fig`, or `scpc`.

    Returns
    -------
    plt.Axes
        The decorated Axes object.

    Raises
    ------
    ValueError
        If `plot_type` is invalid or if required `kwargs` or `config` keys
        are missing for the specified `plot_type`.
    """

    def _add_plot_label(ax, label):
        # Define the position and size of the plot label based on the provided keyword
        ax.text(s=label, transform=ax.transAxes, **config)
        return ax
    
    def _add_label_scatter(ax, positions, cdict=None):
        label_contrast =  config.pop('label_contrast')
        match_color = config.pop('match_color')

        if not isinstance(positions, pd.DataFrame):
            raise ValueError("Parameter 'positions' must be a DataFrame.")
        
        # Confirm that cdict.keys and positions.index match
        if cdict is not None:
            if not all([cat in cdict for cat in positions.index]):
                raise ValueError("If cdict provided, values in positions index must be present in cdict keys.")

        for label, (x, y) in positions.iterrows():
            label_config = deepcopy(config)
            if match_color:
                label_config.update({'color': cdict[label]})
            if label_contrast:
                facecolor_dict = {'facecolor': get_contrast_color(cdict[label])}
                if 'bbox' in label_config:
                    label_config['bbox'].update(facecolor_dict)
                else:
                    label_config['bbox'] = facecolor_dict
            ax.text(x, y, s=label, **label_config)

        return ax

    def _add_cbar(ax, fig, scpc):
        # Define the position and size of the colorbar based on the provided keyword
        
        label_size = config.pop('label_size')
        label_color = config.pop('label_color')

        cax, yadjust_pos, ytick_params = cbar_in_axes(fig, ax, cax=None, **config)

        colorbar = plt.colorbar(scpc, cax=cax)

        cax.set_yticks([colorbar.vmax*yadjust_pos], labels=["{:.2f}".format(colorbar.vmax)], 
                       size=label_size, va='top', color=label_color)
        
        cax.tick_params(axis='y', which='both', size=0, **ytick_params)

        return ax
    
    def _add_legend(ax, cdict):
        if config['show_all_cats'] is not None:
            cdict = {cat: cdict[cat] for cat in config['show_all_cats'] if cat in cdict}
        config.pop('show_all_cats', None)

        if config['show_marker'] is True:
            show_marker = True
        else:
            show_marker = False
        config.pop('show_marker', None)

        legend_handles = [mpl.patches.Patch(color=color, label=label) for label, color in cdict.items()]

        leg = ax.legend(handles=legend_handles, **config)

        if show_marker is False:
            for handle in leg.legend_handles:
                # Make the handle's patch invisible
                handle.set_visible(False)
        
        return ax
    
    plot_types = {'plot_label', 'label_scatter', 'cbar', 'legend'}

    necessary_config_keys = {
        'plot_label': [],
        'label_scatter': ['label_contrast', 'match_color'], # cdict is optional
        'cbar': ['label_size', 'label_color'],
        'legend': ['show_marker', 'show_all_cats']
    }

    necessary_kwargs_keys = {
        'plot_label': ['label'],
        'label_scatter': ['positions'],
        'cbar': ['fig', 'scpc'],
        'legend': ['cdict'],
    }

    decorators = {
        'plot_label': _add_plot_label,
        'label_scatter': _add_label_scatter,
        'cbar': _add_cbar,
        'legend': _add_legend
    }

    # For internal consistency, ensure that all keys in the dictionaries match the plot_types
    for decorator_dict in [necessary_config_keys, necessary_kwargs_keys, decorators]:
        assert set(decorator_dict.keys()) == plot_types
    
    if plot_type not in necessary_config_keys:
        raise ValueError("plot_type must be one of: "
                         f"{', '.join(necessary_config_keys.keys())}.")

    if any([i not in kwargs for i in necessary_kwargs_keys[plot_type]]):
        raise ValueError(f"with {plot_type}, kwargs must contain {necessary_kwargs_keys[plot_type]}.")
    
    if any([i not in config for i in necessary_config_keys[plot_type]]):
        raise ValueError(f"with {plot_type}, config must contain {necessary_config_keys[plot_type]}.")

    ax = decorators[plot_type](ax, **kwargs)

    return ax