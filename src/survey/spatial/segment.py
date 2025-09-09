# Built-ins
import itertools as it
import logging
from typing import List, Optional, Tuple
from pathlib import Path
import warnings

# Standard libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import TextBox, CheckButtons, RadioButtons
from scipy.spatial import distance
from skimage import morphology

# Single-cell libs
import scanpy as sc
import mudata as md

# Survey libs
from survey.singlecell.io import read_data, write_data
from survey.spatial.plotting import survey_plot
from survey.spatial.core import validate_spatial_mdata, validate_chipnums, ChipSet
from survey.genutils import pklop


###
# TODO:
# [X] Fix single-click selection: 
#       Currently, a user must click and drag to select wells. A single, stationary click does not register. 
#       The `on_press` and `on_release` event handlers should be modified to correctly capture and process a 
#       single-click event on a well.
# [X] Implement re-coloring of selected wells: 
#       If a well is already assigned to an ROI (e.g., 'ROI A') and the user clicks on it with a different 
#       ROI active (e.g., 'ROI B'), the well should be reassigned to 'ROI B' and its color updated. The current
#       logic only allows filling unselected wells or erasing selected ones.
# [ ] Refactor erase functionality: 
#       The current 'erase' mode is tied to the fill logic. A dedicated 'Erase Mode' button should be implemented.
#       When active, clicking or dragging over any selected well should remove it from its ROI, regardless of the 
#       currently selected ROI name.
# [X] Implement flood fill: 
#       Add a 'Flood Fill' mode. When a user clicks inside an un-segmented region, the tool should identify all 
#       contiguous, unselected wells and assign them to the currently active ROI. This will likely involve using 
#       `skimage.morphology.flood_fill` on a binary representation of the well grid.
# [X] Implement flood erase: 
#       Add a 'Flood Erase' mode. When a user clicks on a colored well, the tool should erase all contiguous wells
#       belonging to the *same* ROI. This prevents accidental erasure of adjacent, but distinct, ROIs.
# [ ] Add variable brush sizes: 
#       Introduce a UI element (like a slider or a set of buttons) to allow the user to change the brush size. For 
#       example, sizes could be 1x1, 3x3, and 5x5 wells, allowing for faster selection of larger areas.
# [ ] Add transparency slider for ROI overlay: 
#       Implement a slider widget to control the alpha (transparency) of the colored polygons that represent ROIs. 
#       This will allow the user to see the underlying data (e.g., cell colors) more clearly.
###


md.set_options(pull_on_update=False) # Avoid the pull_on_update warning

logger = logging.getLogger(__name__)

# Keys in mdata.uns to pickle when saving the h5mu file
UNS_TO_PICKLE = ['meta', 'clustering', 'annot', 'survey']


# --- Interactive Segmentation App ---

class InteractiveImage:
    """
    An interactive GUI for segmenting spatial transcriptomics data.

    This class creates a Matplotlib-based interface for manually annotating
    regions of interest (ROIs) on a spatial plot. Users can draw, erase, and
    label ROIs, which are then stored for downstream analysis.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing the spatial data.
    chipnum : int
        The chip number to segment.
    color : str, optional
        The key in `.obs` to use for coloring individual cells.
    imgdir : str, optional
        The directory containing the tissue image to be used as a background.
    size : float, optional
        The size of the plotted cells.
    fss : float, optional
        The figure size scale factor.
    group : str, optional
        The name of the segmentation group to be created or edited.
    delete : bool, optional
        If True, deletes any existing segmentation data for the specified `group`.

    Attributes
    ----------
    plot_ax : plt.Axes
        The main axes for the spatial plot.
    fig : plt.Figure
        The Matplotlib figure object.
    selected_ids : pd.DataFrame
        A DataFrame tracking the selected wells, their polygon objects, and ROI assignments.
    rois : pd.DataFrame
        A DataFrame storing the names and colors of defined ROIs.
    """
    def __init__(self,
                 mdata: md.MuData,
                 chipnum: int,
                 color: str = 'leiden',
                 imgdir: Optional[str] = None,
                 size: float = 10,
                 fss: float = 15,
                 group: str = 'tissue',
                 delete: bool = False) -> None:

        borders = (0, 1, 0, 0.2)
        img_arg = (Path(imgdir), 0)

        self.plot_ax = survey_plot(mdata, 
                                   chipnum=chipnum, 
                                   color=color, 
                                   size=size, 
                                   borders=borders, 
                                   fss=fss,
                                   img=img_arg,
                                   sort_order=False, 
                                   linewidth=1, 
                                   edgecolor=(0, 0, 0, 0.2)
                                   )
        
        chipset = mdata['xyz'].uns['survey']
        chip = chipset.chips[chipnum]
        seg = chip.seg

        self.arr = chip.array

        self.fig = self.plot_ax.get_figure()
        # Get the current size of the figure
        fig_width, fig_height = self.fig.get_size_inches()

        # Increase the width of the figure by a certain amount
        increase_width_by = 3  # Adjust this value as needed
        self.fig.set_size_inches(fig_width + increase_width_by, fig_height)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.dragging = False
        self.current_ys = []
        self.current_xs = []
        self.selected_ids = pd.DataFrame(columns=['poly', 'roi'])
        self.current_roi_name = ''
        self.rois = pd.DataFrame(columns=['name', 'color'])
        self.nums = it.count()
        self.roi_cdict = {}
        self.colors = [(*mpl.colors.hex2color(sc.pl.palettes.default_20[i]), 0.5) for i in range(20)]
        self.currently_segmenting = False
        self.click_mode = 'fill'
        self.click_type = 'individual'

        survey_plot_position = self.plot_ax.get_position()

        ## ROI Name Text Box Input

        # Set the width and height of the text box
        tb_ax_width = survey_plot_position.x1 - survey_plot_position.x0  # Adjust as needed
        tb_ax_height = 0.075  # Adjust as needed

        # Calculate the left and bottom coordinates for the text box
        tb_ax_left = survey_plot_position.x0
        tb_ax_bottom = 0.05

        # Create a new axes where the text box will be
        text_box_ax = plt.axes([tb_ax_left, tb_ax_bottom, tb_ax_width, tb_ax_height])

        self.text_box = TextBox(text_box_ax, 'ROI Name:')

        # Connect the text box to the on_text_submitted method
        self.text_box.on_submit(self.on_text_submitted)

        ## Defining Settings Boxes

        # buffer = 0.03
        buffer = 0
        settings_left = survey_plot_position.x1 + buffer

        ## Clicking Mode RadioButtons

        # Set the width and height of the radio
        radio_ax_width = 0.1  # Adjust as needed
        radio_ax_height = 0.1  # Adjust as needed

        # Calculate the left and bottom coordinates for the radio
        radio_ax_left = settings_left
        radio_ax_bottom = survey_plot_position.y1 - radio_ax_height

        # Create a new axes instance for the radio
        self.radio_ax = plt.axes([radio_ax_left, radio_ax_bottom, radio_ax_width, radio_ax_height])  # Adjust the position and size as needed
        
        # Set the limits to (0, 1) for both axes
        # self.radio_ax.set_xlim(0, 1)
        # self.radio_ax.set_ylim(0, 1)
        # self.radio_ax.set_xticks([])
        # self.radio_ax.set_yticks([])

        self.radio_ax.axis('off')  # Hide the axes

        self.radio = RadioButtons(self.radio_ax, ['Fill Individual', 'Erase Individual', 'Flood Fill', 'Flood Erase']) 

        # Connect the radio to a callback function
        self.radio.on_clicked(self.radio_callback)

        ## Clicking Mode AutoReset CheckButtons

        # Set the width and height of the checkbox
        checkbox_ax_width = 0.1  # Adjust as needed
        checkbox_ax_height = 0.1  # Adjust as needed

        # Calculate the left and bottom coordinates for the checkbox
        checkbox_ax_left = settings_left
        checkbox_ax_bottom = survey_plot_position.y1 - radio_ax_height - checkbox_ax_height

        # Create a new axes instance for the checkbox
        self.checkbox_ax = plt.axes([checkbox_ax_left, checkbox_ax_bottom, checkbox_ax_width, checkbox_ax_height])  # Adjust the position and size as needed

        self.checkbox_ax.axis('off')  # Hide the axes

        self.checkbox = CheckButtons(self.checkbox_ax, ['Auto-reset Flood'], [True]) 

        # # Connect the checkbox to a callback function
        # self.checkbox.on_clicked(self.checkbox_callback)
        
        ## ROI Legend

        # Set the width and height of the ROI legend
        roi_ax_width = 0.2  # Adjust as needed
        roi_ax_height = 0.3  # Adjust as needed
        
        # Calculate the left and bottom coordinates for the ROI legend
        roi_ax_left = settings_left + 0.01
        roi_ax_bottom = survey_plot_position.y1 - radio_ax_height - checkbox_ax_height - roi_ax_height

        # Create a new axes instance for the ROI legend
        self.roi_display_ax = plt.axes([roi_ax_left, roi_ax_bottom, roi_ax_width, roi_ax_height])  # Adjust the position and size as needed

        # self.roi_display_ax.set_title('ROIs')
        # self.roi_display_ax.spines['left'].set_visible(False)
        # self.roi_display_ax.spines['right'].set_visible(False)
        # self.roi_display_ax.spines['bottom'].set_visible(False)
        
        # Set the limits to (0, 1) for both axes
        # self.roi_display_ax.set_xlim(0, 1)
        # self.roi_display_ax.set_ylim(0, 1)
        # self.roi_display_ax.set_xticks([])
        # self.roi_display_ax.set_yticks([])

        self.roi_display_ax.axis('off')  # Hide the axes

        ## Pre-populate ROIs if group already exists

        if group in seg.columns and not delete:
            for roi_name in seg[group].cat.categories:
                self.current_roi_name = roi_name
                color = self.get_next_color()
                self.rois.loc[next(self.nums)] = [roi_name, color]
                self.update_cdict()
                current_ids = seg[seg[group] == roi_name].index
                self.fill_selected(current_ids)
            
            self.update_roi_display()
            self.current_roi_name = ''
        
        # TODO: 
        # Currently pre-population is pretty lightweight but with more functionality, we might consider storing the 
        # state of the app (with all its attributes, besides ax objects) and reload it?

    def update_cdict(self) -> None:
        """Updates the internal ROI name to color dictionary."""
        self.roi_cdict = self.rois.set_index('name')['color'].to_dict()

    def update_roi_display(self) -> None:
        """Refreshes the ROI legend display on the GUI."""
        # Clear the previous ROI display
        self.roi_display_ax.clear()
        
        # Display the ROI names with their colors
        # print(self.roi_cdict)
        for i, (name, color) in enumerate(self.roi_cdict.items()):
            ypos = 1 - (i / len(self.colors))
            self.roi_display_ax.text(0, ypos, name, color=(*color[:3], 1), va='top', ha='left')

        self.roi_display_ax.axis('off')

    def get_next_color(self) -> Tuple[float, float, float, float]:
        """
        Gets the next available color from the default palette.

        Returns
        -------
        tuple
            An RGBA color tuple.
        """
        used_colors = set(tuple(x) for x in self.rois['color'])
        # print(len([i for i in self.colors if i not in self.rois['color']]))
        return [i for i in self.colors if tuple(i) not in used_colors][0]
    
    def radio_callback(self, label: str) -> None:
        """
        Handles clicks on the mode selection (RadioButtons).

        Parameters
        ----------
        label : str
            The label of the selected radio button.
        """

        if label == 'Flood Fill':
            self.click_mode = 'fill'
            self.click_type = 'flood'
        elif label == 'Erase Individual':
            self.click_mode = 'erase'
            self.click_type = 'individual'
        elif label == 'Flood Erase':
            self.click_mode = 'erase'
            self.click_type = 'flood'
        else: # label == 'Fill Individual':
            self.click_mode = 'fill'
            self.click_type = 'individual'

    def on_text_submitted(self, text: str) -> None:
        """
        Handles ROI name submission from the TextBox widget.

        Parameters
        ----------
        text : str
            The text entered by the user.
        """
        # if self.default_roi_name:
        #     self.default_roi_name = False
        #     text = self.current_roi_name
        
        if text != '' and text not in self.rois['name'].values:
            color = self.get_next_color()
            self.rois.loc[next(self.nums)] = [text, color]
        
        self.update_cdict()
        self.update_roi_display()
        self.current_roi_name = text
    
    def closest(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Finds the closest well IDs to a set of (x, y) coordinates.

        Parameters
        ----------
        xs : np.ndarray
            An array of x-coordinates.
        ys : np.ndarray
            An array of y-coordinates.

        Returns
        -------
        np.ndarray
            An array of unique well IDs closest to the input coordinates.
        """
        # Create an array of (x, y) coordinates for the wells
        well_coords = self.arr.wells[['center_x', 'center_y']].values

        # Create an array of (x, y) coordinates for the input points
        points = np.column_stack((xs, ys))

        # Calculate the Euclidean distance from each well to each point
        distances = distance.cdist(well_coords, points, 'euclidean')

        # Find the index of the well with the smallest distance for each point
        closest_well_indices = np.argmin(distances, axis=0)

        # Get the unique set of closest well ids
        closest_well_ids = np.unique(self.arr.wells.index[closest_well_indices])

        return closest_well_ids
    
    def fill_selected(self, current_ids: np.ndarray) -> None:
        """
        Fills the specified wells with the current ROI color.

        Parameters
        ----------
        current_ids : np.ndarray
            An array of well IDs to fill.
        """

        diff_ids = np.setdiff1d(current_ids, self.selected_ids.index)

        for id in diff_ids:
            poly = mpl.patches.Polygon(self.arr.verts[id], closed=True, facecolor=self.roi_cdict[self.current_roi_name])
            self.plot_ax.add_patch(poly)
            self.selected_ids.loc[id, ['poly', 'roi']] = [poly, self.current_roi_name]
    
    def erase_selected(self, current_ids: np.ndarray) -> None:
        """
        Erases the segmentation from the specified wells.

        Parameters
        ----------
        current_ids : np.ndarray
            An array of well IDs to erase.
        """
        int_ids = np.intersect1d(current_ids, self.selected_ids.index)

        for id in int_ids:
            self.selected_ids.loc[id, 'poly'].remove()
            self.selected_ids.drop(index=[id], inplace=True)

    def on_press(self, event: mpl.backend_bases.MouseEvent) -> None:
        """
        Handles the 'button_press_event' from Matplotlib.

        Parameters
        ----------
        event : mpl.backend_bases.MouseEvent
            The event object.
        """
        if event.inaxes == self.plot_ax and self.current_roi_name != '':
            self.currently_segmenting = True
            # print("Pressed:", event.xdata, event.ydata)
            if event.xdata is not None and event.ydata is not None:
                self.current_xs.append(int(event.xdata))
                self.current_ys.append(int(event.ydata))
        else:
            self.currently_segmenting = False
        # elif self.click_mode == 'erase':
        #     ###
        # elif self.click_mode == 'flood_fill':

    def on_drag(self, event: mpl.backend_bases.MouseEvent) -> None:
        """
        Handles the 'motion_notify_event' (dragging) from Matplotlib.

        Parameters
        ----------
        event : mpl.backend_bases.MouseEvent
            The event object.
        """
        if self.currently_segmenting:
            if event.xdata is not None and event.ydata is not None:
                row, col = int(event.ydata), int(event.xdata)
                self.current_ys.append(row)
                self.current_xs.append(col)

    def on_release(self, event: mpl.backend_bases.MouseEvent) -> None:
        """
        Handles the 'button_release_event' from Matplotlib.

        This method finalizes a selection action (click or drag), identifies
        the affected wells, and applies the fill or erase operation.

        Parameters
        ----------
        event : mpl.backend_bases.MouseEvent
            The event object.
        """
        if self.currently_segmenting:
            
            # Get the closest well ids to the current xs and ys
            current_ids = self.closest(np.array(self.current_xs), np.array(self.current_ys))

            if self.click_type == 'flood':
                flood_seed = tuple(self.arr.wells.loc[current_ids[0], ['row', 'col']].astype(int))

                fillarr = np.ones(self.arr.arr_shape)*-1 # Can't use zeros because the first roi is number 0 (see self.nums = it.count() in __init__)
                # new_current_ids = list()

                for num, roi_name in self.rois['name'].items():
                    rows, cols = self.arr.wells.loc[self.selected_ids.index[self.selected_ids['roi'] == roi_name], ['row', 'col']].values.T

                    # For erasing, we only want to remove squares of the same color/roi
                    if self.click_mode == 'erase':
                        fillarr[rows, cols] = num
                    # For filling, we want to consider all borders bounding the region, including those from different colors/rois
                    else: # if self.click_mode == 'fill
                        fillarr[rows, cols] = -2
                    
                current_indices = np.argwhere(morphology.flood_fill(fillarr, flood_seed, -3, connectivity=1) == -3)
                # new_current_ids.append(self.arr.wells.reset_index(drop=False).set_index(['row', 'col']).loc[list(map(tuple,current_indices)), 'id'].values)
                # current_ids = np.concatenate(new_current_ids)

                current_ids = self.arr.wells.reset_index(drop=False).set_index(['row', 'col']).loc[list(map(tuple,current_indices)), 'id'].values
                

                auto_reset_status = self.checkbox.get_status()
                if auto_reset_status[0]: # First index because only one checkbox
                    if self.click_mode == 'fill':
                        self.radio.set_active(0) # 'Fill Individual' is index 0
                    else: # self.click_mode == 'erase':
                        self.radio.set_active(1) # 'Erase Individual' is index 1
            else: # self.click_type == 'individual'
                pass

            if self.click_mode == 'fill':
                self.fill_selected(current_ids)

            elif self.click_mode == 'erase':
                self.erase_selected(current_ids)
            
            dropped = [i for i in self.rois['name'].values if i not in self.selected_ids['roi'].unique()]
            if len(dropped) > 0:
                self.rois = self.rois[~self.rois['name'].isin(dropped)]
                self.rois.sort_index(inplace=True) # should be the order in which they were added
                self.rois['color'] = self.colors[:len(self.rois)]
                self.update_cdict()
                for id in self.selected_ids.index:
                    new_fc = self.roi_cdict[self.selected_ids.loc[id, 'roi']]
                    self.selected_ids.loc[id, 'poly'].set_facecolor(new_fc)
                self.update_roi_display()
                self.text_box.set_val('')

            # Refresh the displayed image
            self.fig.canvas.draw_idle()

            # Clear the current xs and ys
            self.current_xs = []
            self.current_ys = []
            self.currently_segmenting = False

    def show(self) -> None:
        """Displays the interactive plot window."""
        plt.show()

# --- Apply Segmentation ---

def apply_segmentation_results(mdata: md.MuData,
                               chipnum: int,
                               selected_ids: pd.DataFrame,
                               group: str,
                               rois: pd.DataFrame) -> None:
    """
    Applies segmentation results from an interactive session to the MuData object.

    This function takes the ROIs defined in the GUI and saves them to the
    appropriate `Chip` object's `.seg` DataFrame within `mdata`.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object to be modified.
    chipnum : int
        The chip number for which the segmentation was performed.
    selected_ids : pd.DataFrame
        The DataFrame from `InteractiveImage.selected_ids` containing the
        well-to-ROI assignments.
    group : str
        The name of the segmentation group (column name in `.seg`).
    rois : pd.DataFrame
        The DataFrame from `InteractiveImage.rois` defining the ROI names and colors.
    """
    if selected_ids['roi'].isna().all():
        logger.info(f"No segmentation was performed for chip {chipnum}. Skipping.")
        return

    logger.info(f"Applying segmentation results for chip {chipnum}.")
    
    # Prepare the results DataFrame
    selected_ids['roi'] = selected_ids['roi'].astype('category').cat.reorder_categories(rois['name'])
    selected_ids.rename(columns={'roi': group}, inplace=True)
    
    # Get the specific chip object
    chip = mdata['xyz'].uns['survey'].chips[chipnum]
    
    # Safely remove the old segmentation group if it exists
    if group in chip.seg.columns:
        chip.seg.drop(columns=group, inplace=True)
        
    # Join the new segmentation results
    chip.seg = chip.seg.join(selected_ids[group])


def transfer_segmentation(mdata: md.MuData,
                          path_to_saved_mdata: str,
                          chipnums: Optional[List[int]] = None,
                          groups: Optional[List[str]] = None) -> None:
    """
    Transfers segmentation data from a saved MuData file to the current one.

    This function reads the ChipSet object from the sidecar pickle file of a
    saved MuData object and copies specified segmentation groups to the
    corresponding chips in the provided `mdata` object.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object to transfer segmentation into.
    path_to_saved_mdata : str
        Path to the source `.h5mu` file from which to load segmentation.
    chipnums : list of int, optional
        A list of chip numbers to process. If None, all chips with segmentation
        in the source file are considered.
    groups : list of str, optional
        A list of segmentation group names to transfer. If None, all groups
        found on the source chips are transferred.

    Returns
    -------
    None, modifies the `mdata` object in place.
    """
    # logger.info(f"Transferring segmentation from {path_to_saved_mdata}")

    # 1. Validate the target mdata object
    validate_spatial_mdata(mdata)
    target_chipset = mdata['xyz'].uns['survey']

    # 2. Load the ChipSet from the source pickle file
    pickle_path = Path(path_to_saved_mdata).with_suffix('.pkl')
    if not pickle_path.exists():
        raise FileNotFoundError(f"Sidecar pickle file not found at {pickle_path}. "
                                "Cannot transfer segmentation without it.")

    saved_uns_data = pklop(pickle_path)

    if 'xyz' not in saved_uns_data or 'survey' not in saved_uns_data['xyz']:
        raise ValueError("Saved data pickle does not contain 'xyz' modality with a 'survey' ChipSet.")

    source_chipset = saved_uns_data['xyz']['survey']
    if not isinstance(source_chipset, ChipSet):
        raise TypeError("The 'survey' object in the saved data is not a valid ChipSet.")

    # 3. Determine which chips and groups to process
    source_chip_keys = source_chipset.chips.keys()
    chips_to_process = chipnums if chipnums is not None else source_chip_keys
    
    # 4. Iterate and transfer segmentation
    for chip_num in chips_to_process:
        if chip_num not in source_chip_keys:
            warnings.warn(f"Chip {chip_num} not found in source file. Skipping.")
            continue
        if chip_num not in target_chipset.chips:
            warnings.warn(f"Chip {chip_num} not found in target mdata. Skipping.")
            continue

        source_chip = source_chipset.chips[chip_num]
        target_chip = target_chipset.chips[chip_num]
        
        source_groups = source_chip.seg.columns
        groups_to_process = groups if groups is not None else source_groups

        for group in groups_to_process:
            if group not in source_groups:

                warnings.warn(f"Group '{group}' not found for chip {chip_num} in source. Skipping.")
                continue
            
            # Drop the column if it already exists in the target
            if group in target_chip.seg.columns:
                target_chip.seg.drop(columns=group, inplace=True)
            
            # Join the segmentation data
            seg_to_add = source_chip.seg[[group]].dropna()
            target_chip.seg = target_chip.seg.join(seg_to_add)

            # Ensure the new column is categorical
            if not isinstance(target_chip.seg[group], pd.CategoricalDtype):
                target_chip.seg[group] = target_chip.seg[group].astype('category')


# --- Public API Function ---

def run_segmentation(file_name: str,
                     chipnums: Optional[List[int]] = None,
                     color: str = "leiden",
                     size: float = 10.0,
                     group: str = "tissue",
                     delete: bool = False,
                     imgdir: Optional[str] = None) -> None:
    """
    Launches the interactive segmentation GUI for one or more chips.

    This is the main public function to start the segmentation workflow. It
    reads a MuData object, launches the `InteractiveImage` GUI for each
    specified chip, and saves the results back to the file upon completion.

    Parameters
    ----------
    file_name : str
        Path to the input `.h5mu` file.
    chipnums : list of int, optional
        A list of chip numbers to process. If None, all chips in the dataset
        are processed sequentially.
    color : str, optional
        The column in `.obs` to use for coloring cells in the background.
    size : float, optional
        The size of the plotted cells.
    group : str, optional
        The name for the segmentation group being created or edited.
    delete : bool, optional
        If True, any existing segmentation data in the specified `group`
        column will be deleted before starting the session.
    imgdir : str, optional
        The directory containing the tissue image files.
    """
        
    logger.info(f"Reading data from {file_name}...")
    mdata = read_data(file_name)

    validate_spatial_mdata(mdata)
    chipset = mdata['xyz'].uns['survey']
    if chipnums is None:
        chipnums = list(chipset.chips.keys())
    else:
        chipnums = validate_chipnums(chipset, chipnums)
    
    for chipnum in chipnums:
        logger.info(f"Starting interactive segmentation for chip {chipnum}...")
        
        # Instantiate and run the interactive GUI component
        app = InteractiveImage(mdata, chipnum=chipnum, color=color, imgdir=imgdir, size=size, group=group, delete=delete)
        app.show() # This is a blocking call; code waits until the user closes the window.

        logger.info(f"Interactive session for chip {chipnum} finished.")
        
        # Apply the results gathered from the interactive session
        apply_segmentation_results(
            mdata=mdata,
            chipnum=chipnum,
            selected_ids=app.selected_ids,
            group=group,
            rois=app.rois
        )

    # Determine the output file path
    
    logger.info(f"Writing updated data to {file_name}...")
    write_data(mdata, file_name, uns_to_pickle=UNS_TO_PICKLE, overwrite=True)
    logger.info("Segmentation complete.")

