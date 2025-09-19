# Built-ins
import re
from typing import (
    Union, List, Dict, Tuple, Optional, Any
)
import warnings
import itertools as it
from numbers import Number, Integral
from collections import deque
from pathlib import Path
from functools import reduce

# Standard libs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Single-cell libs
import mudata as md

# Survey libs
from survey.genutils import is_listlike



def are_contiguous_connected(arr: np.ndarray) -> bool:
    """
    Checks if all '1's in a binary NumPy array form a single connected component.

    Connectivity is defined by adjacency (up, down, left, right). This function
    is useful for validating grid-based layouts.

    Parameters
    ----------
    arr : np.ndarray
        The binary NumPy array to check. Should contain only 0s and 1s.

    Returns
    -------
    bool
        True if all '1's form a single contiguous block, False otherwise.
    
    Raises
    ------
    ValueError
        If the input array is not binary.
    """
    if not np.all(np.isin(arr, [0, 1])):
        raise ValueError("Input array must be binary (contain only 0s and 1s).")

    total_ones = arr.sum()

    if total_ones == 0:
        return True  # Trivially contiguous

    # Find the coordinates of the first '1' to start the search
    start_coords_arr = np.argwhere(arr)
    if start_coords_arr.size == 0:
         return True # Should be covered by total_ones check, but safe
    
    start_node = tuple(start_coords_arr[0])

    # Perform a search (BFS) to find all connected '1's
    q = deque([start_node])
    visited = {start_node}
    
    while q:
        r, c = q.popleft()

        # Check neighbors (up, down, left, right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            # Check bounds
            if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1]:
                # If neighbor is a '1' and not visited, add to queue
                if arr[nr, nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))

    # If the number of visited cells equals the total number of '1's,
    # they form a single contiguous block.
    return len(visited) == total_ones


def is_contiguous_block_rectangular(arr: np.ndarray) -> bool:
    """
    Checks if a single contiguous block of '1's in a binary array is rectangular.

    This function assumes the input array contains only one contiguous block of '1's
    and determines if that block forms a solid rectangle with no holes.

    Parameters
    ----------
    arr : np.ndarray
        The binary NumPy array to check. Should contain only 0s and 1s.

    Returns
    -------
    bool
        True if the block of '1's forms a solid rectangle, False otherwise.
    """
    # Find the coordinates of all '1's
    rows, cols = np.where(arr == 1)

    # If there are no '1's, it's trivially rectangular.
    if len(rows) == 0:
        return True

    # Determine the bounding box of the '1's
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Calculate the area of the bounding box
    bounding_box_area = (max_row - min_row + 1) * (max_col - min_col + 1)

    # The shape is a solid rectangle if the number of '1's
    # is equal to the area of its bounding box.
    return arr.sum() == bounding_box_area


class Array:
    """
    Represents the physical properties of a single-cell analysis array.

    This class generates and stores the geometry of a regular tessellation of
    wells (e.g., square or hexagonal grid), including well vertices, centers,
    and overall dimensions.

    Parameters
    ----------
    n : int
        The number of sides of the polygon for each well (e.g., 4 for square).
    s : float
        The size of the polygon (side length for squares, short diagonal for hex).
    w : float
        The width of the walls between adjacent wells.
    arr_shape : tuple of (int, int)
        The number of wells to tile in the (row, column) directions.
    version : any
        An identifier for this specific array version.
    flat_top : bool, optional
        Whether to use a flat-top orientation for the wells. Default is True.
    origin : {'corner', 'center'}, optional
        The origin (0, 0) of the array coordinate system. 'corner' places it
        at the bottom-left, 'center' at the geometric center. Default is 'corner'.

    Attributes
    ----------
    n : int
        Number of sides of the well polygons.
    s : float
        Size of the well polygons.
    w : float
        Width of the walls between wells.
    pitch : float
        The center-to-center distance between adjacent wells (s + w).
    arr_shape : tuple
        The (rows, columns) shape of the array.
    flat_top : bool
        Orientation of the wells.
    origin : str
        The coordinate system origin.
    version : any
        The identifier for the array version.
    wells : pd.DataFrame
        A DataFrame containing coordinates and IDs for each well.
    verts : dict
        A dictionary mapping well ID to their vertex coordinates.
    space : dict
        A dictionary with spacing and limit information.
    lims : dict
        A dictionary with 'x' and 'y' limits of the array.
    """
    def __init__(self,
                 n: int,
                 s: float,
                 w: float,
                 arr_shape: Tuple[int, int],
                 version: Any,
                 flat_top: bool = True,
                 origin: str = 'corner') -> None:

        if len(arr_shape) != 2:
            raise ValueError(f"arr_shape must be a 2-list-like of int, not {arr_shape}.")

        space, verts_arr = self.get_regular_tessellation(n, s, w, arr_shape, flat_top)

        id = it.count()
        wells = pd.DataFrame(columns=['row', 'col', # 'dep', 
                                    'x', 'y', # 'z', 
                                    'center_x', 'center_y', #'center_z'
                                    ], dtype=int).rename_axis('id', axis=0)
        verts = dict()

        if origin == 'center':
            center = np.array([0, 0])
        elif origin == 'corner':
            xmin, ymin = verts_arr.min(0)
            center = np.array([-xmin, ymin]) # they are negative because verts_arr is by default around (0, 0)
            verts_arr = verts_arr - np.array([xmin, -ymin])
            space['xlim'] -= np.array([xmin, xmin])
            space['ylim'] += np.array([ymin, ymin])
        else:
            raise ValueError(f'Param `origin` must be "center" or "corner", not {origin}')
        
        ydir = space['ydir']
        xdir = space['xdir']

        for col in range(arr_shape[1]):
            col_top_center = center
            col_top_verts_arr = verts_arr
            
            for row in range(arr_shape[0]):
                next_id = next(id)
                wells.loc[next_id] = [row, col, # 0, # # x, y z
                                    col, arr_shape[0] - row - 1, # 0, # # x, y z
                                    *center, # 0 # # x, y z
                                    ] 
                verts[next_id] = verts_arr.copy()

                center = center - np.array([ydir['xspace'], ydir['yspace']])  
                verts_arr = verts_arr - np.array([ydir['xspace'], ydir['yspace']])

            center = col_top_center + np.array([xdir['xspace'], xdir['yspace']*space['xswitch']])
            verts_arr = col_top_verts_arr + np.array([xdir['xspace'], xdir['yspace']*space['xswitch']])

            space['xswitch'] *= -1
        
        wells[['row', 'col', 'x', 'y']] = wells[['row', 'col', 'x', 'y']].astype(int)
        
        # Add the x and y lims of the resulting plot to be used in downstream methods
        lims = {'x': tuple(space['xlim']), 'y': tuple(space['ylim'])}

        self._n = n
        self._s = s
        self._w = w
        self._arr_shape = tuple(arr_shape)
        self._flat_top = flat_top
        self._origin = origin # 'corner' or 'center'
        self._version = version

        self._wells = wells
        self._verts = verts
        self._space = space

        self._lims = lims

    @property
    def n(self):
        return self._n

    @property
    def s(self):
        return self._s

    @property
    def w(self):
        return self._w
    
    @property
    def pitch(self):
        return self._w + self._s

    @property
    def arr_shape(self):
        return self._arr_shape

    @property
    def flat_top(self):
        return self._flat_top

    @property
    def origin(self):
        return self._origin

    @property
    def version(self):
        return self._version

    @property
    def wells(self):
        return self._wells.copy()  # Return copy to prevent modification

    @property
    def verts(self):
        return self._verts.copy()  # Return copy to prevent modification

    @property
    def space(self):
        return self._space.copy()  # Return copy to prevent modification

    @property
    def lims(self):
        return self._lims.copy()

    def __str__(self):
        # Define a string representation of the object
        return f"Array ({self.version}, n={self.n}, s={self.s}, w={self.w}, shape={self.arr_shape})"


    def __repr__(self):
            # Define a string representation of the object
            return self.__str__()


    def poly_verts(self,
                   n: int,
                   c: Tuple[float, float] = (0, 0),
                   r: float = 1,
                   rot: float = 0) -> np.ndarray:
        """
        Calculate the vertices of a regular polygon.

        Parameters
        ----------
        n : int
            The number of sides of the polygon.
        c : tuple, optional
            The (x, y) coordinates of the center of the polygon.
        r : float, optional
            The radius of the circumscribing circle of the polygon.
        rot : float, optional
            The rotation angle of the polygon in degrees.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (n, 2) containing the (x, y) coordinates
            of the polygon's vertices.
        """
        rot_rad = rot*(np.pi/180)
        points = list()
        angle = (2*np.pi)/n
        for i in range(n):
            points.append((c[0] + r * np.sin(i * angle + rot_rad), c[1] + r * np.cos(i * angle + rot_rad)))

        return np.array(points)  


    def get_regular_tessellation(self,
                                 n: int,
                                 s: float,
                                 w: float,
                                 arr_shape: Tuple[int, int],
                                 flat_top: bool = True) -> Tuple[Dict, np.ndarray]:
        """
        Calculates the geometric parameters for a regular tessellation.

        Parameters
        ----------
        n : int
            The number of sides of the polygons.
        s : float
            The size of the polygons.
        w : float
            The wall width between polygons.
        arr_shape : tuple
            The (rows, columns) shape of the array.
        flat_top : bool, optional
            Whether the polygons should have a flat top.

        Returns
        -------
        space : dict
            A dictionary containing spacing and limit information.
        verts_arr : np.ndarray
            The vertices of a single polygon at the origin.
        """

        def calc_max_lim(shapeval):
            return (shapeval - 0.5) * (s + w) + w / 2
        
        r2 = 2**0.5
        r3 = 3**0.5
        space = dict()

        if n == 4:
            if flat_top:
                verts_arr = self.poly_verts(n, r=(s/2)*r2, rot=(90/2))
                xspace_xdir = s + w
                yspace_xdir = 0
                xspace_ydir = 0
                yspace_ydir = s + w

                xswitch = 0

                xlim = np.array([-((s/2) + w), calc_max_lim(arr_shape[1])])
                ylim = np.array([-calc_max_lim(arr_shape[0]), (s/2) + w])
            else:
                raise NotImplementedError
            #     poly = self.poly_verts(n, r=s*r2, rot=0)
        elif n == 6:
            if flat_top:
                verts_arr = self.poly_verts(n, r=s/r3, rot=(180/6))
                xspace_xdir = (3/2)*(s/r3) + w*(r3/2)
                yspace_xdir = (s/2) + (w/2)
                xspace_ydir = 0
                yspace_ydir = s + w

                xswitch = -1
                
                xlim = np.array([-(s/r3 + w), (arr_shape[1]-1)*xspace_xdir + (s/r3 + w)])
                ylim =  np.array([-(arr_shape[0]*(s + w) + w/2), (yspace_ydir+w)/2])

            else:
                raise NotImplementedError
                # poly = self.poly_verts(n, r=s/r3, rot=0)
        elif n == 3: # the last regular tessellation of the plane
            raise NotImplementedError
        
        space['xdir'] = dict()
        space['xdir']['xspace'] = xspace_xdir
        space['xdir']['yspace'] = yspace_xdir

        space['ydir'] = dict()
        space['ydir']['xspace'] = xspace_ydir
        space['ydir']['yspace'] = yspace_ydir

        space['xswitch'] = xswitch
        space['xlim'] = xlim
        space['ylim'] = ylim

        return space, verts_arr


    def plot(self,
             fss: Optional[float] = None,
             ax: Optional[plt.Axes] = None,
             color: str = 'k',
             cmap: str = 'gray',
             **kwargs: Any) -> plt.Axes:
        """
        Plots the array's well tessellation.

        Parameters
        ----------
        fss : float, optional
            Figure size scale factor.
        ax : plt.Axes, optional
            An existing Matplotlib Axes to plot on.
        color : str, optional
            Coloring scheme for the wells. Can be 'multi' (random colors),
            'id' (sequential colormap), or a valid Matplotlib color string.
        cmap : str, optional
            Colormap to use if `color='id'`.
        **kwargs
            Additional keyword arguments passed to `matplotlib.patches.Polygon`.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """
        if ax:
            pass
        else:
            if fss is None:
                fss = 10
            fig, ax = plt.subplots(figsize=(fss, fss))
        if color == 'multi':
            colors = np.random.randint(0, 255, size=(len(self.verts), 3))/255
        elif color == 'id':
            cmap = plt.cm.get_cmap(cmap)
            colors = [cmap(i / (len(self.verts) - 1)) for i in range(len(self.verts))]
        elif mpl.colors.is_color_like(color):
            colors = [color]*len(self.verts)
        else:
            raise ValueError(f"color must be 'multi', 'id', or a matplotlib color, not {color}.")
        for (id, c) in zip(self.verts, colors):
            ax.add_patch(mpl.patches.Polygon(self.verts[id], closed=True, facecolor=c, **kwargs))
        ax.set_xlim(self.space['xlim'])
        ax.set_ylim(self.space['ylim'])
        ax.set_aspect('equal')
        ax.grid(False)
        return ax

    def get_wall_verts(self) -> List[np.ndarray]:
        """
        Create vertices for walls (the negative space between rectangular wells).
        
        This method assumes n=4 (squares) with a flat_top=True orientation.
        
        Returns
        -------
        list
            A list of numpy arrays, where each array represents the 4x2 vertices
            of a single wall rectangle.
            
        Raises
        ------
        ValueError
            If the array is not configured for rectangular wells with flat tops.
        """
        if self.n != 4:
            raise ValueError("Wall patches are only supported for square wells (n=4).")
        if not self.flat_top:
            raise ValueError("Wall patches are only supported for flat_top=True orientation.")
        
        wall_verts = []
        wells_df = self.wells
        n_rows, n_cols = self.arr_shape
        
        # Get the full span of the array from the pre-calculated limits
        x_left_lim, x_right_lim = self.lims['x']
        y_bottom_lim, y_top_lim = self.lims['y']
        
        # 1. Create horizontal walls (n_rows + 1 walls)
        # -----------------------------------------------
        for i in range(n_rows + 1):
            if i == 0:  # Top bounding wall
                # Get a reference well from the first row
                ref_well_id = wells_df[wells_df['row'] == 0].index[0]
                # The bottom of this wall aligns with the top of the wells
                y_bottom_wall = self.verts[ref_well_id][0][1]
                y_top_wall = y_bottom_wall + self.w
            elif i == n_rows:  # Bottom bounding wall
                # Get a reference well from the last row
                ref_well_id = wells_df[wells_df['row'] == n_rows - 1].index[0]
                # The top of this wall aligns with the bottom of the wells
                y_top_wall = self.verts[ref_well_id][1][1]
                y_bottom_wall = y_top_wall - self.w
            else:  # Internal horizontal walls (between rows)
                # This wall is between row (i-1) and row i
                top_well_id = wells_df[wells_df['row'] == i - 1].index[0]
                bottom_well_id = wells_df[wells_df['row'] == i].index[0]
                # The wall fills the space between the two rows
                y_top_wall = self.verts[top_well_id][1][1]      # Bottom edge of top well
                y_bottom_wall = self.verts[bottom_well_id][0][1] # Top edge of bottom well

            # Define the wall vertices (TR, BR, BL, TL order)
            h_wall = np.array([
                [x_right_lim, y_top_wall],
                [x_right_lim, y_bottom_wall],
                [x_left_lim, y_bottom_wall],
                [x_left_lim, y_top_wall]
            ])
            wall_verts.append(h_wall)

        # 2. Create vertical walls (n_cols + 1 walls)
        # ---------------------------------------------
        for j in range(n_cols + 1):
            if j == 0:  # Left bounding wall
                ref_well_id = wells_df[wells_df['col'] == 0].index[0]
                # The right side of this wall aligns with the left side of the wells
                x_right_wall = self.verts[ref_well_id][3][0]
                x_left_wall = x_right_wall - self.w
            elif j == n_cols:  # Right bounding wall
                ref_well_id = wells_df[wells_df['col'] == n_cols - 1].index[0]
                # The left side of this wall aligns with the right side of the wells
                x_left_wall = self.verts[ref_well_id][0][0]
                x_right_wall = x_left_wall + self.w
            else:  # Internal vertical walls (between columns)
                # This wall is between col (j-1) and col j
                left_well_id = wells_df[wells_df['col'] == j - 1].index[0]
                right_well_id = wells_df[wells_df['col'] == j].index[0]
                # The wall fills the space between the two columns
                x_left_wall = self.verts[left_well_id][0][0]  # Right edge of left well
                x_right_wall = self.verts[right_well_id][3][0] # Left edge of right well
            
            # Define the wall vertices (TR, BR, BL, TL order)
            v_wall = np.array([
                [x_right_wall, y_top_lim],
                [x_right_wall, y_bottom_lim],
                [x_left_wall, y_bottom_lim],
                [x_left_wall, y_top_lim]
            ])
            wall_verts.append(v_wall)
            
        return wall_verts


class Layout:
    """
    Represents the barcode layout of a single-cell analysis chip.

    This class processes a DataFrame representing the physical layout of barcodes,
    validates its format, and transforms it into a numerical representation for
    analysis. It also detects the encoding scheme (e.g., row/column, zipcode)
    and barcode types.

    Parameters
    ----------
    layout_df : pd.DataFrame
        A DataFrame where the index and columns are integer coordinates and
        values are hyphen-separated barcode strings.
    id : any
        An identifier for this layout.
    bctype_mapper : list of tuple, optional
        A list of (name, function) tuples used to classify barcode parts.
        The function should take a barcode string and return True if it matches
        the type.

    Attributes
    ----------
    id : any
        The layout identifier.
    df : pd.DataFrame
        The original layout DataFrame.
    df_stacked : pd.DataFrame
        A stacked version of the layout DataFrame with columns 'x', 'y', 'barcode'.
    da : np.ndarray
        A 3D integer-encoded representation of the layout.
    format : str
        The detected encoding format ('rowcol', 'zipcode', 'random').
    coords : dict
        A dictionary describing the role of each layer in the encoding.
    mappers : dict
        A dictionary mapping string barcode parts to integer codes for each layer.
    bctypes : list of str
        A list of detected barcode types for each layer.
    all_bcs : np.ndarray
        An array of all unique barcode parts in the layout.
    """

    # This keys match the name of the modality in the mdata object
    # since it will be used to split the layout by combinatorial index
    DEFAULT_BCTYPE_MAPPER = [
        ('sbc', lambda x: x.startswith('sbc')),
        ('hto', lambda x: x.startswith('tsb'))
    ]


    def __init__(self,
                 layout_df: pd.DataFrame,
                 id: Any,
                 bctype_mapper: Optional[List[Tuple[str, callable]]] = None) -> None:

        if bctype_mapper is None:
            bctype_mapper = self.DEFAULT_BCTYPE_MAPPER

        if any(['-' in i[0] for i in bctype_mapper]):
            raise ValueError("bctype_mapper keys must not contain hyphens '-'.")
        if any([re.match(r'^sp\d+$', i[0]) for i in bctype_mapper]):
            raise ValueError("bctype_mapper values must not match the pattern 'sp<digit>'. ")

        layout_df = self._clean_and_validate_format(layout_df)

        layout_df_stacked = pd.DataFrame([(*i, j) for i, j in layout_df.stack().items()], columns=['x', 'y', 'barcode']).set_index('barcode')

        # Make the columns and index 0-indexed
        layout_df.index = layout_df.index - 1
        layout_df.columns = layout_df.columns - 1

        # Convert the DataFrame to a 3D "digitized array" (da)
        da, mappers = self._transform_layout_to_integer_matrix(layout_df)

        bctypes = self._determine_barcode_types(mappers, bctype_mapper)
        encoded_layout = self._detect_encoding_layout(da)
        format = encoded_layout[0]
        coords = dict(enumerate(encoded_layout[1]))

        self._id = id
        self._df = layout_df
        self._da = da
        self._format = format
        self._coords = coords
        self._mappers = mappers
        self._bctypes = bctypes
        self._dfstacked = layout_df_stacked
    
     # Read-only properties
    
    
    @property
    def id(self):
        return self._id
    
    @property
    def df(self):
        return self._df.copy()  # Return copy to prevent modification
    
    @property
    def df_stacked(self):
        return self._dfstacked.copy()  # Return copy to prevent modification
    
    @property
    def da(self):
        return self._da.copy()
    
    @property
    def format(self):
        return self._format
    
    @property
    def coords(self):
        return self._coords.copy()
    
    @property
    def mappers(self):
        return {k: v.copy() for k, v in self._mappers.items()}
    
    @property
    def bctypes(self):
        return self._bctypes

    @property
    def all_bcs(self):
        return reduce(np.union1d, [self.mappers[i].index for i in self.mappers])


    def _clean_and_validate_format(self, layout_df):

        pretext_message = "Layout is not well-formed: "
        
        def is_valid_element(x):
            """Checks a single element."""
            # The element is valid if it is a NaN value...
            is_nan = pd.isna(x)
            # ...or if it is a string that contains a hyphen.
            is_hyphenated_string = isinstance(x, str) and '-' in x
            return is_nan or is_hyphenated_string

        valid_elements = layout_df.map(is_valid_element).values

        if not valid_elements.all():
            raise ValueError(pretext_message + "invalid elements at " + str(np.argwhere(~valid_elements)))

        try:
            layout_df.columns = layout_df.columns.astype(int)
        except ValueError:
            raise ValueError(pretext_message + "column names must be integers.")

        if not (layout_df.columns == np.arange(1, layout_df.shape[1] + 1)).all():
            raise ValueError(pretext_message + "column names must be sequential integers starting at 1.")
        
        try:
            layout_df.index = layout_df.index.astype(int)
        except ValueError:
            raise ValueError(pretext_message + "row names must be integers.")
        
        if not (layout_df.index == np.arange(1, layout_df.shape[0] + 1)).all():
            raise ValueError(pretext_message + "row names must be sequential integers starting at 1.")
        
        barcoded_wells = ~layout_df.isna().values

        if not are_contiguous_connected(barcoded_wells):
            raise ValueError(pretext_message + "all barcoded wells must form a single contiguous block.")
        else:
            if not is_contiguous_block_rectangular(barcoded_wells):
                warnings.warn(pretext_message + "the barcoded wells do not form a solid rectangle. Please double-check the layout if this is unexpected.")

        return layout_df
    

    def _transform_layout_to_integer_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, pd.Series]]:
        """
        Converts a DataFrame of hyphenated strings into a 3D integer matrix.

        Each string element is split by '-', creating 'k' layers. Each layer is
        then independently mapped to a set of unique integers.

        Parameters
        ----------
        df : pd.DataFrame
            An m x n DataFrame where each element is either NaN or a string
            containing hyphens. It is assumed all strings split into the same
            number of parts.

        Returns
        -------
        Tuple[np.ndarray, Dict[int, pd.Series]]
            - A 3D NumPy array of shape (m, n, k) with integer codes.
            NaNs from the original DataFrame are represented by -1.
            - A dictionary of mappers. The keys are the layer index (0 to k-1),
            and the values are pandas Series where the index contains the
            original string parts and the values are their corresponding integer codes.
        
        Raises
        ------
        ValueError
            If the DataFrame is not empty but contains no non-NaN values, or if
            the hyphenated strings do not split into a consistent number of parts.
        """
        m, n = df.shape
        if m == 0 or n == 0:
            return np.empty((m, n, 0), dtype=int), {}

        # Create a DataFrame of lists by splitting the strings
        # NaNs will remain NaNs
        split_df = df.map(lambda x: x.split('-') if pd.notna(x) else np.nan, na_action='ignore')

        # Find the first non-NaN list to determine k (number of layers)
        first_list = split_df.stack().dropna().iloc[0] if not split_df.stack().dropna().empty else None
        if first_list is None:
            # Handle case where DataFrame has shape but is all NaNs
            return np.full((m, n, 0), -1, dtype=int), {}
        
        k = len(first_list)

        layers_as_strings: List[np.ndarray] = []
        for i in range(k):
            # Extract the i-th element from each list, keeping NaNs
            layer_i = split_df.map(lambda x: x[i] if isinstance(x, list) else np.nan, na_action='ignore')
            layers_as_strings.append(layer_i.values)

        mappers = {}
        result_layers = []

        for i, string_layer in enumerate(layers_as_strings):
            # Find unique values in the current layer to create a mapper
            unique_vals = pd.unique(string_layer[~pd.isna(string_layer)])
            
            # Create the integer mapping and the Series for the output
            mapping = {val: j for j, val in enumerate(unique_vals)}
            mappers[i] = pd.Series(mapping, name=f"sp{i + 1}").sort_index()

            # Apply the mapping
            # Vectorized mapping is faster than .map for numpy arrays
            integer_layer = np.full(string_layer.shape, -1, dtype=int)
            for str_val, int_val in mapping.items():
                integer_layer[string_layer == str_val] = int_val
            
            result_layers.append(integer_layer)

        # Stack the integer layers along the third dimension
        final_matrix = np.stack(result_layers, axis=-1)

        return final_matrix, mappers
    

    def _detect_encoding_layout(self, arr: np.ndarray) -> Tuple[str, Tuple[Optional[str], Optional[str]]]:
        """
        Analyzes a 3D NumPy array to determine its 2D spatial encoding scheme,
        ignoring any locations marked with -1.

        The function first validates that every non-'-1' (layer_0, layer_1) code pair 
        is unique. It then auto-detects one of three possible layouts:

        1.  'rowcol': Layer 0 encodes rows and Layer 1 encodes columns (or vice-versa).
        2.  'zipcode': Layer 0 defines contiguous regions ("zipcodes"), and Layer 1
            provides a unique address within each of those regions.
        3.  'random': The codes are unique but show no discernible spatial pattern.

        Args:
            arr: A 3D NumPy array of shape (n, m, 2) and integer dtype.
                -1 values are treated as ignored/non-encoded locations.

        Returns:
            A tuple containing:
                - A string: "rowcol", "zipcode", or "random".
                - A tuple describing the role of each layer, e.g., ('rows', 'cols').

        Raises:
            ValueError: If the input array does not have a depth of 2, is not of
                        integer type, or if the location codes are not unique.
        """
        # --- Input Validation ---
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError("Input array must be 3D with a depth of 2 (shape (n, m, 2)).")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("Input array dtype must be integer.")

        layer0 = arr[:, :, 0]
        layer1 = arr[:, :, 1]
        
        # Create a mask for valid locations (where codes are not -1)
        valid_mask = layer0 != -1

        # --- Uniqueness Check (on valid locations only) ---
        valid_codes = arr[valid_mask]
        
        if valid_codes.size > 0:
            unique_codes = np.unique(valid_codes, axis=0)
            if unique_codes.shape[0] != valid_codes.shape[0]:
                raise ValueError("Location codes are not unique across the grid.")

        # --- 1. Check for "rowcol" layout ---
        rows_const_in_l0 = all(np.unique(row[row != -1]).size <= 1 for row in layer0)
        cols_const_in_l1 = all(np.unique(col[col != -1]).size <= 1 for col in layer1.T)
        
        if rows_const_in_l0 and cols_const_in_l1:
            return ("rowcol", ("row", "col"))

        cols_const_in_l0 = all(np.unique(col[col != -1]).size <= 1 for col in layer0.T)
        rows_const_in_l1 = all(np.unique(row[row != -1]).size <= 1 for row in layer1)

        if cols_const_in_l0 and rows_const_in_l1:
            return ("rowcol", ("col", "row"))
                
        # --- 2. Check for "zipcode" layout ---
        zipcodes = np.unique(layer0[valid_mask])
        
        if zipcodes.size > 0:
            all_regions_are_contiguous = True
            for zc in zipcodes:
                region_mask = (layer0 == zc).astype(int)
                if not are_contiguous_connected(region_mask):
                    all_regions_are_contiguous = False
                    break
            
            if all_regions_are_contiguous:
                return ("zipcode", ("region", "address"))

        # --- 3. Default to "random" ---
        return ("random", (None, None))


    def _determine_barcode_types(self, mappers, bctype_mapper):
        """
        Analyze barcode types in layout mappers and return a '-'-delimited string
        describing the barcode composition for each spatial position.
        
        Parameters
        ----------
        mappers : dict
            Dictionary from Layout.mappers with integer keys and pd.Series values
            where Series.index contains barcode strings
        bctype_mapper : list of tuples
            List of (type_name, function) tuples for barcode type detection
            
        Returns
        -------
        str
            '-'-delimited string describing barcode types at each spatial position
        """
        result_parts = []
        
        for sp_idx in sorted(mappers.keys()):
            barcodes = mappers[sp_idx].index
            
            # Apply all detection functions to create boolean matrix
            type_results = {}
            for type_name, detect_func in bctype_mapper:
                type_results[type_name] = barcodes.map(detect_func)
            
            # Convert to DataFrame for easier analysis
            type_df = pd.DataFrame(type_results)
            
            # Analyze results for this spatial position
            type_summary = type_df.all(axis=0)  # Which types have ALL barcodes matching
            type_any = type_df.any(axis=0)      # Which types have ANY barcodes matching
            # type_partial = type_any & ~type_summary  # Which types have SOME but not all
            
            # Determine the result string for this position
            all_types = type_summary[type_summary].index.tolist()
            any_types = type_any[type_any].index.tolist()
            # partial_types = type_partial[type_partial].index.tolist()
            
            if len(all_types) == 1 and len(any_types) == 1:
                # Single type, all barcodes match
                result_parts.append(all_types[0])
            elif len(all_types) == 0 and len(any_types) == 1:
                # Single type, partial match
                result_parts.append(f"partial({any_types[0]})")
            elif len(any_types) > 1:
                # Multiple types detected
                mixed_str = ",".join(sorted(any_types))
                result_parts.append(f"mixed({mixed_str})")
            else:
                # No types detected
                result_parts.append("unknown")
        
        return result_parts

    @classmethod
    def from_csv(cls, fname, **kwargs):
        """
        Create a SurveyLayout object from a CSV file.

        Parameters
        ----------
        fname : str or Path
            Path to the CSV file containing the layout.

        Returns
        -------
        SurveyLayout
            An instance of SurveyLayout with the layout data loaded.
        """

        pretext_message = "When coming from CSV, "

        layout_df = pd.read_csv(fname, index_col=0, header=0)

        if not layout_df.index.isna().any() or not all([i % 2 == 1 for i in np.argwhere(layout_df.index.isna()).flatten()]):
            raise ValueError(pretext_message + "every other row should be empty with NaNs (i.e. from merged cells in Google Sheets).")
        
        layout_df = layout_df.loc[~layout_df.index.isna()]

        return cls(layout_df, **kwargs)


    def __str__(self):
        return f"Layout {self.id} ({self.format} of {'-'.join(self.bctypes)}) with shape {self.df.shape}"


    def __repr__(self):
        return self.__str__()


class TissueImage:
    """
    Represents a tissue image associated with a spatial experiment.

    Parameters
    ----------
    fn : str
        The filename of the image (e.g., 'tissue_image.png').
    extent : str or tuple
        The spatial extent of the image `(left, right, bottom, top)`.
        If 'auto', the extent can be determined later.
    """
    
    def __init__(self,
                 fn: str,
                 extent: Union[str, Tuple[int, int, int, int]]) -> None:
        error_msg = "Param `fn` must be a string representing the filename"
        if not isinstance(fn, str):
            raise TypeError(error_msg + ", not " + str(type(fn)) + ".")
        fn = Path(fn)
        if not fn.suffix:
            raise ValueError(error_msg + ", including its suffix (e.g. .png, .jpg).")
        if len(fn.parts) > 1:
            raise ValueError(error_msg + ", excluding its parent directories.")
        
        if isinstance(extent, str):
            if extent != 'auto':
                raise ValueError("Param `extent` must be 'auto' or a tuple of 4 numbers.")
        elif isinstance(extent, tuple):
            if len(extent) != 4 or not all(isinstance(i, Number) for i in extent):
                raise ValueError("Param `extent` must be a tuple of 4 numbers.")
        
        self.fn = Path(fn)
        self.extent = extent


    def __str__(self):
        return f"TissueImage {self.fn} with extent {self.extent}"
    

    def __repr__(self):
        return self.__str__()


class Chip:
    """
    Represents a single experimental chip, combining a physical array and a barcode layout.

    Parameters
    ----------
    num : int
        The chip number or identifier.
    array : Array
        An `Array` object describing the physical grid of wells.
    layout : Layout
        A `Layout` object describing the barcode arrangement.
    offset : tuple of (int, int), optional
        The (row, column) offset of the layout within the array.
    imgs : list of TissueImage, optional
        A list of `TissueImage` objects associated with this chip.

    Attributes
    ----------
    num : int
        The chip number.
    array : Array
        The associated `Array` object.
    layout : Layout
        The associated `Layout` object.
    offset : tuple
        The (row, column) offset.
    seg : pd.DataFrame
        A DataFrame to store segmentation information, indexed by well ID.
    imgs : list of TissueImage
        Associated tissue images.
    """

    def __init__(self,
                 num: int,
                 array: Array,
                 layout: Layout,
                 offset: Optional[Tuple[int, int]] = None,
                 imgs: Optional[List[TissueImage]] = None) -> None:

        if not isinstance(num, int) or num < 0:
            raise TypeError("Param `num` (i.e. chip number) must be a non-negative integer.")
        if not isinstance(array, Array):
            raise TypeError("Param `array` must be an instance of Array.")
        if not isinstance(layout, Layout):
            raise TypeError("Param `layout` must be an instance of Layout.")
        
        if offset is None:
            offset = (0, 0)
        elif not isinstance(offset, tuple) or len(offset) != 2:
            raise TypeError("Param `offset` must be a tuple of length 2.")
        elif not all([(isinstance(i, int) and i >= 0) for i in offset]):
            raise TypeError("Param `offset` must be a tuple of non-negative integers.")

        for i, j, k in zip(array.arr_shape, layout.df.shape, offset):
            if i < j + k:
                raise ValueError(f"Array shape {array.arr_shape} is smaller than layout shape {layout.df.shape} with offset {offset}.")

        self.num = num
        self.array = array
        self.layout = layout
        self.offset = offset

        self.seg = pd.DataFrame(index=self.array.wells.index.copy())
        if imgs is not None:
            if not isinstance(imgs, list) or not all([isinstance(i, TissueImage) for i in imgs]):
                raise TypeError("Param `imgs` must be a list of TissueImage instances.")
            self.imgs = imgs
        else:
            self.imgs = []


    def get_welldata(self) -> pd.DataFrame:
        """
        Generates a comprehensive DataFrame for all wells on the chip.

        This method combines information from the `Array` and `Layout` objects,
        including physical coordinates, layout coordinates, barcode information,
        and segmentation data, into a single DataFrame indexed by well ID.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing detailed data for each well.
        """
        
        wells = self.array.wells.copy()
        layout_df = self.layout.df.copy()

        # Rename the columns so it's clear where they're from
        wells = wells.rename(columns=dict([(i, 'arr-' + i) for i in wells.columns]))

        # Adjust the layout DataFrame to account for the offset
        layout_df.index = layout_df.index + self.offset[0]
        layout_df.columns = layout_df.columns + self.offset[1]

        # Join on the row and column indices, then reset the index back to 'id'
        # Note: I perform a similar stacking operation layout_df.stack() as in the Layout class (see lyt.df_stacked)
        # but I'm redoing it here with the offset applied
        wells_rowcol = wells.reset_index(drop=False).set_index(['arr-row', 'arr-col'])
        layout_rowcol = pd.DataFrame([(*i, j) for i, j in layout_df.stack().items()], columns=['arr-row', 'arr-col', 'barcode']).set_index(['arr-row', 'arr-col'])
        welldata = wells_rowcol.join(layout_rowcol)
        welldata = welldata.reset_index(drop=False).set_index('id')

        # Add the split barcodes
        bcsplit = welldata['barcode'].str.split('-', expand=True).rename(columns=lambda x: f'sp{x}')
        welldata = welldata.join(bcsplit)

        # Add the layout coordinates
        layout_row = welldata['arr-row'] - self.offset[0]
        welldata['lyt-row'] = np.where((layout_row < 0) | (layout_row > layout_df.shape[0]), pd.NA, layout_row)
        layout_col = welldata['arr-col'] - self.offset[1]
        welldata['lyt-col'] = np.where((layout_col < 0) | (layout_col > layout_df.shape[1]), pd.NA, layout_col)

        # Add the internal integer barcode IDs
        for i in self.layout.mappers:
            mapped_values = welldata[f'sp{i}'].map(self.layout.mappers[i])
            welldata[f'int-bc-id{i}'] = mapped_values.astype('Int64')  # This handles NaN automatically

        # Add segmentation info
        welldata = welldata.join(self.seg)


        return welldata


    def __str__(self):
        return f"Chip {self.num} ({self.array.version}, {self.layout.id})"
    
    
    def __repr__(self):
        return self.__str__()


class ChipSet:
    """
    Manages a collection of `Chip` objects for a spatial experiment.

    This class serves as a factory and container for all the chips used in an
    experiment, handling the creation of `Array` and `Layout` objects from
    provided metadata and file paths.

    Parameters
    ----------
    chip_meta : pd.DataFrame
        A DataFrame defining the properties of each chip. Must be indexed by
        chip number and have columns 'version', 'lyt', 'offset', 'img', 'extent'.
    array_params : dict
        A dictionary mapping array `version` names to their parameter dictionaries.
    lyts_dir : str or Path
        The directory containing layout CSV files.
    imgs_dir : str or Path, optional
        The directory containing tissue image files.

    Attributes
    ----------
    arrays : dict
        A dictionary of `Array` objects, keyed by version.
    layouts : dict
        A dictionary of `Layout` objects, keyed by layout ID.
    chips : dict
        A dictionary of `Chip` objects, keyed by chip number.
    """

    def __init__(self,
                 chip_meta: pd.DataFrame,
                 array_params: Dict[str, Dict],
                 lyts_dir: Union[str, Path],
                 imgs_dir: Optional[Union[str, Path]] = None) -> None:

        chip_meta, array_params = self._validate_chip_array(chip_meta, array_params)

        lyts_dir = self._validate_lyts_dir(lyts_dir, chip_meta)

        if imgs_dir is not None:
            imgs_dir = self._validate_imgs_dir(imgs_dir)

        self._imgs_dir = imgs_dir


        arrays = {
            version: Array(version=version, **params) for version, params in array_params.items()
        }
        layouts = {
            lyt: Layout.from_csv(lyts_dir / (lyt + '.csv'), id=lyt) for lyt in chip_meta['lyt']
        }
        chips = {
            num: Chip(num, arrays[version], layouts[lyt], offset, imgs=[TissueImage(img, extent)]) 
            for num, (version, lyt, offset, img, extent) in chip_meta.iterrows()
        }

        self.arrays = arrays
        self.array_params = array_params
        self.layouts = layouts
        self.chips = chips

        self._chip_meta = chip_meta
        self._chip_key_prop = None # Gets set later when added to mdata
    

    def _validate_chip_array(self, chip_meta: pd.DataFrame, array_params: dict):
        expected_columns = ['version', 'lyt', 'offset', 'img', 'extent']
        expected_index_name = 'num'

        if not isinstance(chip_meta, pd.DataFrame):
            raise TypeError("Param `chip_meta` must be a DataFrame.")
        if not all([isinstance(i, int) for i in chip_meta.index]):
            raise TypeError("Param `chip_meta` must have integer index (chip numbers).")
        if not all([i in chip_meta.columns for i in expected_columns]):
            raise ValueError(f"Param `chip_meta` must have columns {expected_columns}, not {chip_meta.columns.tolist()}.")
        chip_meta = chip_meta[expected_columns]
        if not chip_meta.index.is_unique:
            raise ValueError("Param `chip_meta` must have unique index (chip numbers).")
        if chip_meta.index.name != expected_index_name:
            raise ValueError(f"Param `chip_meta` must have index name '{expected_index_name}', not '{chip_meta.index.name}'.")
        if not all([i in array_params.keys() for i in chip_meta['version']]):
            raise ValueError("All chip versions in `chip_meta` must have corresponding array parameters in `array_params`.")
        if not isinstance(array_params, dict) or not all(isinstance(v, dict) for v in array_params.values()):
            raise TypeError("Param `array_params` must be a dict of dicts mapping chip versions to their array parameters.")
        
        return chip_meta, array_params
        

    def _validate_imgs_dir(self, imgs_dir: Union[str, Path]):

        if not isinstance(imgs_dir, (str, Path)):
            raise TypeError("imgs_dir must be a string or Path object.")
        imgs_dir = Path(imgs_dir)
        if not imgs_dir.exists():
            raise FileNotFoundError(f"Provided imgs_dir {imgs_dir} does not exist.")
        if not imgs_dir.is_dir():
            raise TypeError("imgs_dir must be a directory.")
        
        # Image files will be checked for existence later, when chips are created
        return imgs_dir


    def _validate_lyts_dir(self, lyts_dir: Union[str, Path], chip_meta: pd.DataFrame):

        lyt_col = 'lyt'
        lyt_suffix = '.csv'

        if lyt_col not in chip_meta.columns:
            raise ValueError(f"Param `chip_meta` must have a column '{lyt_col}' with layout names.")

        if not isinstance(lyts_dir, (str, Path)):
            raise TypeError("lyts_dir must be a string or Path object.")
        lyts_dir = Path(lyts_dir)
        if not lyts_dir.exists():
            raise FileNotFoundError(f"Provided lyts_dir {lyts_dir} does not exist.")
        if not lyts_dir.is_dir():
            raise TypeError("lyts_dir must be a directory.")
        
        # Check if all layouts exist in the directory
        not_found = []
        for lyt in chip_meta[lyt_col].unique():
            lyt_file = lyts_dir / (lyt + lyt_suffix)
            if not lyt_file.exists():
                not_found.append(lyt_file)
        if not_found:
            raise FileNotFoundError(f"Layout files not found in {lyts_dir} for layouts: {not_found}")
        
        return lyts_dir


    def set_imgs_dir(self, imgs_dir: Union[str, Path]):

        imgs_dir = self._check_imgs_dir(imgs_dir)
        self.imgs_dir = imgs_dir

    @property
    def bctypes(self):
        """
        Returns a list of unique barcode types across all chips in the ChipSet.
        """
        bctypes_sets = [set(self.chips[num].layout.bctypes) for num in self.chips]
        return set([i for j in bctypes_sets for i in j])
    
    @property
    def bclens(self):
        """
        Returns a list of unique barcode lengths across all chips in the ChipSet.
        """
        bclens_sets = [len(self.chips[num].layout.bctypes) for num in self.chips]
        return list(set(bclens_sets))

    @property
    def imgs_dir(self):
        """
        Returns the images directory.
        """
        return self._imgs_dir

    @property
    def chip_meta(self):
        """
        Returns the chip metadata DataFrame.
        """
        return self._chip_meta.copy()

    @property
    def chip_key_prop(self):
        """
        Returns the chip key property.
        """
        return self._chip_key_prop

    def __str__(self):
        return f"ChipSet with {len(self.chips)} chips ({len(self.arrays)} array(s), {len(self.layouts)} layout(s))"


    def __repr__(self):
        return self.__str__()


def validate_spatial_mdata(mdata: md.MuData) -> None:
    """
    Validates that a MuData object is correctly structured for spatial analysis.

    Checks for the 'xyz' modality and a valid `ChipSet` object in `mdata['xyz'].uns`.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object to validate.

    Raises
    ------
    ValueError
        If the MuData object is missing the required components.
    """
    # Check mdata is MuData object
    if not isinstance(mdata, md.MuData):
        raise ValueError('Param `mdata` must be a MuData object.')

    # Check 'xyz' modality exists
    if 'xyz' not in mdata.mod:
        raise ValueError("Modality 'xyz' not found in mdata.")

    # Check 'survey' exists in mdata['xyz'].uns and is of type svp.core.ChipSet
    if 'survey' not in mdata['xyz'].uns or not isinstance(mdata['xyz'].uns['survey'], ChipSet):
        raise ValueError("mdata['xyz'].uns must contain a 'survey' key with a svp.core.ChipSet object.")
    
    return


def validate_chipnums(chipset: ChipSet,
                      chipnum: Union[int, List[int]]) -> List[int]:
    """
    Validates that chip numbers exist within a ChipSet.

    Parameters
    ----------
    chipset : ChipSet
        The ChipSet object to check against.
    chipnum : int or list of int
        The chip number(s) to validate.

    Returns
    -------
    list of int
        A standardized list of the validated chip numbers.

    Raises
    ------
    ValueError
        If any of the provided chip numbers are not found in the `chipset`.
    TypeError
        If `chipnum` is not an integer or a list of integers.
    """
    if not is_listlike(chipnum):
        chipnums = [chipnum]
    else:
        chipnums = chipnum

    if not all(isinstance(i, Integral) for i in chipnums):
        raise ValueError("Param `chipnum` must be integer or list-like of integers.")

    chips_not_found = set(chipnums).difference(set(chipset.chips.keys()))

    if chips_not_found:
        raise ValueError(f"Chip numbers not found: {chips_not_found}")
    
    return chipnums