# Built-ins
import os
from typing import (
    Union, List, Optional, Dict, Tuple, Any
)
import warnings

# Standard libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

# Single-cell libs
import scanpy as sc
import mudata as md

# Other 3rd party libs
from anytree import Node, RenderTree
from anytree.exporter import DictExporter
from anytree.importer import DictImporter

# Survey libs
from survey.genplot import subplots
from survey.singlecell.scutils import QuietScanpyLoad
from survey.genutils import is_listlike, get_functional_dependency, get_mask
from survey import singlecell as svc
from survey.singlecell.meta import (
    add_colors, meta_exists, reset_meta_keys
)


class Clustering:
    """
    A class for performing and visualizing stateful, hierarchical clustering
    on single-cell data. The history of operations is stored in a tree
    and can be reverted.

    This class provides an interactive workflow to iteratively subcluster cell
    populations, track the relationships in a hierarchy, and manage different
    clustering states.

    Parameters
    ----------
    data : sc.AnnData or md.MuData
        The single-cell data object.
    resolution : float, optional
        The resolution parameter for the initial Leiden clustering.
    run_initial : bool, optional
        If True, runs an initial clustering upon instantiation.
    mod : str, optional
        If `data` is a MuData object, specifies the modality to use.
        Defaults to 'rna'.
    cluster_key : str, optional
        The key in `.obs` where clustering results are stored.
    **kwargs
        Additional arguments passed to `sc.tl.leiden`.

    Attributes
    ----------
    adata : sc.AnnData
        The AnnData object being used for clustering.
    meta : dict
        A reference to `adata.uns['meta']`.
    cluster_key : str
        The base key for storing cluster labels.
    sub_cluster_key : str
        The temporary key for storing subcluster labels.
    current_cluster_key : str
        The key (`cluster_key` or `sub_cluster_key`) of the currently active clustering.
    cluster_root : anytree.Node
        The root of the clustering hierarchy tree.
    state_counter : int
        The ID of the current state in the history.
    state_history : list
        A list of dictionaries, each representing a state change.
    """

    def __init__(self,
                 data: Union[sc.AnnData, md.MuData],
                 resolution: float = 0.2,
                 run_initial: bool = True,
                 mod: Optional[str] = None,
                 cluster_key: str = 'leiden',
                 **kwargs) -> None:

        # --- Standard Setup ---
        if mod is None:
            mod = 'rna'
        self.mod = mod

        if isinstance(data, md.MuData):
            self.adata = data[mod]
        elif isinstance(data, sc.AnnData):
            self.adata = data
        
        meta_exists(self.adata)
        self.meta = self.adata.uns['meta']
        
        self.cluster_key = cluster_key
        self.sub_cluster_key = 'sub_' + cluster_key
        self.dge = {}
        
        # --- State and History Setup ---
        self.current_cluster_key = None
        self.cluster_root = Node("ROOT")
        self.state_counter = -1
        self.state_history = []
        
        # --- Attributes for pending operations ---
        self.pending_split_parent = None
        self.pending_split_children = None
        
        if run_initial:
            self.initial_clustering(resolution=resolution,
                                    **kwargs)
      

    def initial_clustering(self,
                           resolution: float = 0.2,
                           **kwargs) -> None:
        """
        Runs the initial clustering and establishes the base state (State 0).

        This performs a Leiden clustering on the entire dataset and sets up the
        initial state of the clustering history.

        Parameters
        ----------
        resolution : float, optional
            Resolution parameter for `sc.tl.leiden`.
        **kwargs
            Additional arguments passed to `sc.tl.leiden`.
        """
        sc.tl.leiden(self.adata, resolution=resolution, key_added=self.cluster_key, flavor="igraph", n_iterations=2, **kwargs)
        
        self.state_counter += 1
        state_entry = {
            'state_id': self.state_counter,
            'action': 'initial',
            'obs_after': self.adata.obs[self.cluster_key].copy(),
            'update_map': None,
            'parent_of_split': None
        }
        self.state_history.append(state_entry)
        
        self._rebuild_tree()
        self._update_colors()

        self.current_cluster_key = self.cluster_key

        # if show:
        #     self.show(color=self.cluster_key, **kwargs)


    def subcluster(self,
                   cluster: Union[str, int],
                   resolution: float = 0.2,
                   **kwargs) -> 'Clustering':
        """
        Performs subclustering on a specific cluster.

        This is a temporary operation that creates a new set of labels in
        `self.sub_cluster_key`. The change is not permanent until `.update()`
        is called.

        Parameters
        ----------
        cluster : str or int
            The label of the cluster to subcluster.
        resolution : float, optional
            Resolution parameter for the subcluster Leiden run.
        **kwargs
            Additional arguments passed to `sc.tl.leiden`.

        Returns
        -------
        Clustering
            Returns self to allow for method chaining.
        """
        if isinstance(cluster, (list, tuple)):
            raise ValueError("cluster must be a single cluster label (str or int), not a list-like object.")
        
        cluster_str = str(cluster)
        
        # Perform subclustering on a temporary key
        sc.tl.leiden(self.adata, resolution=resolution, restrict_to=(self.cluster_key, [cluster_str]), 
                     key_added=self.sub_cluster_key, flavor="igraph", n_iterations=2)

        new_clusters = [i for i in self.adata.obs[self.sub_cluster_key].cat.categories if i.startswith(cluster_str + ',')]
        print(f"Cluster {cluster_str} split into {len(new_clusters)} subclusters.")

        # Store the details of this split, pending an update
        self.pending_split_parent = cluster_str
        self.pending_split_children = new_clusters
        
        self._update_colors(key=self.sub_cluster_key)

        # if show:
        #     self.show(color=self.sub_cluster_key, **kwargs)

        self.current_cluster_key = self.sub_cluster_key
        
        return self


    def update(self) -> None:
        """
        Commits the last sub-clustering action, creating a new state.

        This makes the results of the last `subcluster` call permanent by
        updating the main `cluster_key`, creating a new state in the history,
        and rebuilding the hierarchy tree.
        """
        if self.pending_split_parent is None:
            print("No sub-clustering operation to update.")
            return

        # --- HISTORY PRUNING LOGIC ---
        # If we have reverted to a past state, any "future" states are now invalid.
        # Truncate the history list to the current state before adding a new one.
        if self.state_counter < len(self.state_history) - 1:
            print(f"Overwriting history from state {self.state_counter + 1} onwards.")
            self.state_history = self.state_history[:self.state_counter + 1]
        
        # Create a new mapping for ALL current clusters to a new integer range
        cats = self.adata.obs[self.sub_cluster_key].cat.categories
        mapper = {cat: str(i) for i, cat in enumerate(cats)}

        # Apply the mapping to create the new permanent clusters
        self.adata.obs[self.cluster_key] = self.adata.obs[self.sub_cluster_key].map(mapper).astype('category')
        self.adata.obs.drop(columns=[self.sub_cluster_key], inplace=True)
        
        # Create and record the new state
        self.state_counter += 1
        state_entry = {
            'state_id': self.state_counter,
            'action': 'update',
            'obs_after': self.adata.obs[self.cluster_key].copy(),
            'update_map': mapper.copy(),
            'parent_of_split': self.pending_split_parent
        }
        self.state_history.append(state_entry)

        # Rebuild the tree to reflect the new state and update colors
        self._rebuild_tree()
        self._update_colors()
        
        # Reset pending operation
        self.pending_split_parent = None
        self.pending_split_children = None

        self.current_cluster_key = self.cluster_key
        

    def _rebuild_tree(self) -> None:
        """
        Internal method to construct the cluster tree from the state history.
        
        It clears the existing tree and rebuilds it from scratch by processing
        the `state_history` up to the `self.state_counter`.
        """
        self.cluster_root.children = []
        node_map = {}

        # Start with state 0 (initial clustering)
        initial_clusters = self.state_history[0]['obs_after'].cat.categories
        for c in initial_clusters:
            node_map[str(c)] = Node(str(c), parent=self.cluster_root)
            
        # Apply subsequent state changes
        for i in range(1, self.state_counter + 1):
            state = self.state_history[i]
            parent_label = state['parent_of_split']
            update_map = state['update_map']
            
            # Find the parent node that was split
            parent_node = node_map.get(parent_label)
            if parent_node:
                # Get the children from the mapping
                children = [k for k, v in update_map.items() if k.startswith(parent_label + ',')]
                
                # Attach children to the parent
                parent_node.name = f".{parent_label}" # Mark as parent
                for child_label in children:
                    Node(child_label, parent=parent_node)
                del node_map[parent_label]

            # Rename all leaf nodes according to the update_map
            new_node_map = {}
            for leaf in self.cluster_root.leaves:
                old_name = leaf.name
                if old_name in update_map:
                    new_name = update_map[old_name]
                    leaf.name = new_name
                    new_node_map[new_name] = leaf
            node_map = new_node_map
            

    def _update_colors(self, key: Optional[str] = None) -> None:
        """
        Internal method to reset and add colors for a given key.
        
        This ensures that the colors in `adata.uns['meta']` are synchronized
        with the clusters in `adata.obs`.

        Parameters
        ----------
        key : str, optional
            The key in `.obs` to update colors for. If None, uses the main
            `self.cluster_key`.
        """
        if key is None:
            key = self.cluster_key

        reset_meta_keys(self.adata, keys=key)
        add_colors(self.adata, key, by_size=False)
            

    def show(self,
             mode: str = 'labeled',
             clusts: Optional[List[Union[str, int]]] = None,
             color: Optional[str] = None,
             subset: Optional[Dict] = None,
             fss: float = 8,
             ar: float = 1,
             **kwargs) -> None:
        """
        Displays clustering results using various plotting modes.

        Parameters
        ----------
        mode : {'labeled', 'highlight', 'dual'}, optional
            The plotting mode:
            - 'labeled': A single scatter plot with cluster labels.
            - 'highlight': Highlights specified clusters.
            - 'dual': Shows a standard scatter plot and a labeled one side-by-side.
        clusts : list, optional
            A list of cluster IDs to highlight when `mode='highlight'`. If None,
            highlights the results of the last `subcluster` operation.
        color : str, optional
            The key in `.obs` to color by. If None, uses the current active key.
        subset : dict, optional
            A dictionary of subsetting criteria, where keys are column
            names in `self.adata.obs` and values are the values to keep.
        fss : float, optional
            Figure size scaler.
        ar : float, optional
            Aspect ratio of the plot.
        **kwargs
            Additional arguments passed to the underlying plotting functions.
        """
        if color is None:
            color = self.current_cluster_key

        if subset is not None:
            mask = get_mask(self.adata.obs, subset)
            data = self.adata[mask]
        else:
            data = self.adata

        if mode == 'labeled':
            fig, ax = subplots(1, fss=fss, ar=ar)
            ax = svc.pl.labeled_scatter(data=data, color=color, ax=ax, **kwargs)
        elif mode == 'highlight':
            if clusts is None and self.pending_split_children is not None:
                clusts = self.pending_split_children
            ax = svc.pl.highlight(data=data, color=color, cats=clusts, ar=ar, fss=fss, **kwargs)
        elif mode == 'dual':
            fig, ax = subplots(2, fss=fss, ar=ar)
            if not all([k in ['scatter', 'labeled_scatter'] for k in kwargs.keys()]):
                scatter_kwargs = {}
                labeled_kwargs = kwargs
            else:
                scatter_kwargs = kwargs['scatter']
                labeled_kwargs = kwargs['labeled_scatter']
            ax[0] = svc.pl.scatter(data=data, color=color, ax=ax[0], legend=False, **scatter_kwargs)
            ax[1] = svc.pl.labeled_scatter(data=data, numbered=False, color=color, ax=ax[1], legend=True, **labeled_kwargs)
        else:
            raise ValueError("Invalid mode. Choose either 'labeled' or 'highlight'.")


    def get_tree_as_dict(self) -> Dict:
        """
        Exports the current cluster hierarchy tree to a serializable dictionary.
        
        This dictionary can be stored in an AnnData/MuData `.uns` attribute.
        
        Returns
        -------
        dict
            A dictionary containing the tree structure and current state.
        """
        exporter = DictExporter()
        tree_data = exporter.export(self.cluster_root)
        return {
            'tree_data': tree_data,
            'current_state': self.state_counter
        }


    def show_tree(self) -> None:
        """Prints the current cluster hierarchy from the instance."""
        print("--- Cluster Hierarchy ---")
        for pre, _, node in RenderTree(self.cluster_root):
            print(f"{pre}{node.name}")
        print(f"-------------------------\nCurrent State: {self.state_counter}")


    def store_tree(self) -> None:
        """
        Stores the current cluster hierarchy to `adata.uns['clustering']`.
        """
        tree_dict = self.get_tree_as_dict()
        self.adata.uns['clustering'] = tree_dict

    @staticmethod
    def show_tree_from_dict(tree_dict: Dict) -> None:
        """
        Prints a cluster hierarchy from a saved dictionary.

        This is a static utility method. Call it directly on the class:
        `Clustering.show_tree_from_dict(my_tree_dict)`
        
        Parameters
        ----------
        tree_dict : dict
            A dictionary created by the `get_tree_as_dict()` method.
        """

        tree_data = tree_dict.get('tree_data')
        if tree_data is None:
            raise ValueError("Dictionary must contain a 'tree_data' key.")
        
        importer = DictImporter()
        root_node = importer.import_(tree_data)
        current_state = tree_dict.get('current_state', 'N/A')

        print("--- Cluster Hierarchy (from dictionary) ---")
        for pre, _, node in RenderTree(root_node):
            print(f"{pre}{node.name}")
        print(f"-------------------------------------------\nState: {current_state}")


    def revert(self, state_id: int) -> None:
        """
        Reverts the clustering to a specific state.
        
        Parameters
        ----------
        state_id : int
            The state to revert to (from 0 to current state).
        """
        if not (0 <= state_id < len(self.state_history)):
            raise ValueError(f"State ID must be between 0 and {len(self.state_history) - 1}.")
            
        self.state_counter = state_id
        
        # Restore the AnnData object's cluster labels
        target_obs = self.state_history[state_id]['obs_after']
        self.adata.obs[self.cluster_key] = target_obs.copy().astype('category')
        
        # Rebuild the tree and colors to match the reverted state
        self._rebuild_tree()
        self._update_colors()
        
        # Clear any pending operations as they are no longer valid
        self.pending_split_parent = None
        self.pending_split_children = None
        
        print(f"Reverted to state {state_id}.")


    def undo(self) -> None:
        """Reverts to the previous state."""
        if self.state_counter > 0:
            self.revert(self.state_counter - 1)
        else:
            print("Already at the initial state.")


    def redo(self) -> None:
        """Moves forward to the next state if available."""
        if self.state_counter < len(self.state_history) - 1:
            self.revert(self.state_counter + 1)
        else:
            print("Already at the latest state.")

    
def get_dge(adata: sc.AnnData,
            gbg: bool = False) -> pd.DataFrame:
    """
    Extracts and formats differential gene expression (DGE) results.

    This function parses the results from `sc.tl.rank_genes_groups` stored in
    `adata.uns` and returns them as a tidy DataFrame.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object after running `sc.tl.rank_genes_groups`.
    gbg : bool, optional
        If True, pivots the DataFrame to have genes as rows and groups/stats
        as a multi-level column index (grouped-by-gene format). If False
        (default), returns a long-format DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the DGE results.
    """
    
    dge_data = pd.DataFrame()
    groups = adata.uns['rank_genes_groups']['scores'].dtype.names
    n_genes = adata.uns['rank_genes_groups']['scores'].shape[0]
    
    # Check which test was used
    if 'logreg' in adata.uns['rank_genes_groups']['params']['method']:
        for i in ['scores', 'names']:
            dge_data[i] = np.array(adata.uns['rank_genes_groups'][i].tolist()).flatten()
    elif 't-test' in adata.uns['rank_genes_groups']['params']['method'] or 't-test_overestim_var' in adata.uns['rank_genes_groups']['params']['method']:
        for i in ['scores', 'names', 'logfoldchanges', 'pvals', 'pvals_adj']:
            dge_data[i] = np.array(adata.uns['rank_genes_groups'][i].tolist()).flatten()
    else:
        raise ValueError('Unsupported test was used for rank_genes_groups().')
    dge_data['group'] = list(groups)*n_genes
    dge_data = pd.concat([dge_data[dge_data['group'] == group].sort_values(by='scores', ascending=False) for group in groups], axis=0)

    if gbg:
        dge_data['rank'] = dge_data.groupby('group')['scores'].rank(ascending=False, method='dense')
        dge_data = dge_data.set_index('rank').pivot(columns='group').sort_index(axis=1, level='group')
        dge_data.columns = dge_data.columns.reorder_levels([1, 0])
        dge_data.columns.rename(['group', 'stat'], inplace=True)

    return dge_data


def rank_genes(adata: sc.AnnData,
               g: Union[str, List[str]],
               r: Union[str, List[str]] = 'rest',
               gb: str = 'leiden',
               method: str = 't',
               n_genes: int = 20,
               ignore_warnings: bool = True,
               verbosity: int = 0) -> pd.DataFrame:
    """
    Performs differential gene expression analysis between specified groups.

    A wrapper around `scanpy.tl.rank_genes_groups` that simplifies comparing
    one or more groups against a reference set.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object.
    g : str or list of str
        The group(s) of interest.
    r : str or list of str, optional
        The reference group(s). If 'rest' (default), compares against all
        other groups.
    gb : str, optional
        The key in `adata.obs` containing the group labels.
    method : {'t', 'l'}, optional
        The statistical method to use: 't' for t-test, 'l' for logistic regression.
    n_genes : int, optional
        The number of top genes to store for each group.
    ignore_warnings : bool, optional
        If True, suppresses warnings during the DGE test.
    verbosity : int, optional
        Controls scanpy's verbosity level.

    Returns
    -------
    pd.DataFrame
        A tidy DataFrame of the DGE results, processed by `get_dge`.
    """

    if not is_listlike(g):
        g = [g]
    if not is_listlike(r):
        r = [r]

    if r == ['rest']:
        r = 'rest'
        gb_rank = gb
    else:
        mapper = dict()
        for clust in g:
            mapper[clust] = 'g'
        for clust in r:
            mapper[clust] = 'r'
        if adata.obs[gb].dtype.name == 'category':
            adata.obs[gb] = adata.obs[gb].cat.add_categories(['none'])
        adata.obs['sctools_grouped_rank'] = adata.obs[gb].map(mapper).fillna('none')
        gb_rank = 'sctools_grouped_rank'
        g = ['g']
        r = 'r'

    with QuietScanpyLoad(verbosity):
        if ignore_warnings:
            warnings.simplefilter('ignore')
        if method == 't':
            sc.tl.rank_genes_groups(adata, groupby=gb_rank, groups=g, reference=r, n_genes=n_genes)
        elif method == 'l':
            sc.tl.rank_genes_groups(adata, groupby=gb_rank, groups=g, reference=r, n_genes=n_genes, method='logreg')
        if ignore_warnings:
            warnings.simplefilter('default')

    if 'sctools_grouped_rank' in adata.obs.columns:
        adata.obs.drop(columns='sctools_grouped_rank', inplace=True)
    if adata.obs[gb].dtype.name == 'category':
        adata.obs[gb] = adata.obs[gb].cat.remove_unused_categories()

    return get_dge(adata)


def check_groups(adata: sc.AnnData,
                 groups: Union[str, List[List[str]]],
                 key: str = 'leiden') -> Tuple[List[List[str]], pd.Series, pd.Index]:
    """
    Validates and standardizes group definitions for comparison.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object.
    groups : str or list of lists
        The groups to check. If a string, it's treated as `[[group], ['rest']]`.
        If a list, it must be a list of two lists of group labels.
    key : str, optional
        The key in `adata.obs` containing group labels.

    Returns
    -------
    groups : list of lists
        The standardized list of two lists of group labels.
    vcounts : pd.Series
        The value counts of all groups in `adata.obs[key]`.
    possible_groups : pd.Index
        An index of all unique group labels.
    """

    vcounts = adata.obs[key].value_counts()
    possible_groups = vcounts.index

    # Validate input `groups` parameter
    if is_listlike(groups):
        # Ensure `groups` is a list of two disjoint lists
        assert len(groups) == 2 and all([is_listlike(i) for i in groups]), \
            'Param `groups`, if list, must be a list of length 2 of list-likes.'
        assert len(np.intersect1d(groups[0], groups[1])) == 0, 'Param `groups` must be disjoint.'
    else:
        groups = [[groups], ['rest']]

    # Handle 'rest' as the complement of the first group
    if groups[1] == ['rest']:
        groups = [groups[0], list(set(possible_groups).difference(groups[0]))]
    return groups, vcounts, possible_groups


def spec_expr(adata: sc.AnnData,
              groups: Union[str, List[List[str]]],
              p: float,
              key: str = 'leiden',
              layer: str = 'npc-l1p',
              p_of: Optional[Union[str, List[str]]] = None,
              ratio: str = 'percent') -> pd.Series:
    """
    Compute a specific expression score based on a ratio between two groups.

    This score can be the ratio of the mean expression in expressing cells
    or the ratio of the percentage of expressing cells.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    groups : str or list of lists
        The groups to compare. E.g., `[['CD4 T-cells'], ['CD8 T-cells']]`.
        The first list is the numerator, the second is the denominator.
        The denominator can be 'rest' or ['rest'] to compare against all
        other groups.
    p : float
        The minimum percentage of expressing cells (0 to 1) required in the
        `p_of` group(s) for a gene to be considered in the analysis.
    key : str, optional
        The key in `adata.obs` that specifies the cell groupings. Defaults to 'leiden'.
    layer : str, optional
        The data layer to use for expression values. Defaults to 'npc-l1p'.
    p_of : str or list of str, optional
        Logic for applying the percentage threshold `p`, which is always based on
        the numerator group(s) (`groups[0]`).
        - If `groups[0]` contains a single group, this is ignored and the filter
          is based on that single group.
        - If `groups[0]` has multiple subgroups:
            - `None` (default): Genes must be expressed in > `p` percent of cells
              when all subgroups in `groups[0]` are combined and treated as one.
            - `'any'`: Genes must be expressed in > `p` percent of cells in at
              least ONE of the subgroups in `groups[0]`.
            - `list-like`: A subset of groups from `groups[0]`. Genes must be
              expressed in > `p` percent of cells when the specified subgroups
              are combined and treated as one.
    ratio : {'mean', 'percent'}, optional
        The type of ratio to compute. Defaults to 'mean'.
        - 'mean': Ratio of the mean expression in positive cells (>0 counts).
        - 'percent': Ratio of the percentage of expressing cells.

    Returns
    -------
    pd.Series
        A Series of specific expression scores (numerator/denominator), sorted.

    Raises
    ------
    ValueError
        If invalid groups, `p_of`, or `ratio` are provided.

    Notes
    -----
    In general: 
         - to find genes with broader expression across all groups but 
         slightly elevated expression in groups[0], use `ratio='mean'` and 
         higher p (e.g. 0.5).
         - to find genes with low but highly-specific expression in groups[0], 
         use `ratio='percent'` and lower p (e.g. 0.2).

    """
    # This function is a slightly modified copy of the original `spec_expr`
    # It is used to check if pre-computed percentage data needs updating.
    def check_p_expr(adata: sc.AnnData,
                     key: str,
                     mode: str) -> Union[Tuple[bool, pd.Series], None]:
        vcounts = adata.obs[key].value_counts()
        vcounts_str = vcounts.copy()
        vcounts_str.index = vcounts_str.index.values.astype(str)
        # This dictionary holds the actual info to be stored/validated
        p_expr_info = {'groups': vcounts_str.index.values, 'vals': vcounts_str.values}

        if mode == 'validate':
            # Check if the key exists and has the correct structure.
            if key not in adata.uns['p_expr'] or 'groups' not in adata.uns['p_expr'][key]:
                print('Changes detected in p_expr (or first run for this key).')
                return True, vcounts

            stored_info = adata.uns['p_expr'][key]
            if len(stored_info['groups']) != len(p_expr_info['groups']):
                print('Changes detected in p_expr.')
                return True, vcounts
            
            groups_same = (stored_info['groups'] == p_expr_info['groups']).all()
            vals_same = (stored_info['vals'] == p_expr_info['vals']).all()

            if groups_same and vals_same:
                return False, vcounts
            else:
                print('Changes detected in p_expr.')
                return True, vcounts
        elif mode == 'update':
            # Store the info directly under the key
            adata.uns['p_expr'][key] = p_expr_info
            return
        else:
            raise ValueError("Mode must be 'validate' or 'update'.")

    # This is a simplified version of scanpy's internal _check_groups
    def check_groups(adata: sc.AnnData, groups: Union[str, List[List[str]]], key: str):
        if not isinstance(groups, list) or (len(groups) > 0 and not isinstance(groups[0], list)):
             groups = [[groups], [g for g in adata.obs[key].cat.categories if g != groups] ]
        
        vcounts = adata.obs[key].value_counts()
        possible_groups = vcounts.index.tolist()
        return groups, vcounts, possible_groups

    if ratio not in ['mean', 'percent']:
        raise ValueError("`ratio` must be either 'mean' or 'percent'.")

    groups, vcounts, possible_groups = check_groups(adata, groups, key)

    # Handle the special 'rest' case for the denominator group.
    if len(groups) == 2 and (groups[1] == ['rest'] or groups[1] == 'rest'):
        print("Denominator group 'rest' detected. Using all groups not in the numerator.")
        numerator_groups = groups[0]
        denominator_groups = [g for g in possible_groups if g not in numerator_groups]
        groups[1] = denominator_groups

    # Flatten groups and ensure all provided groups are valid
    groups_flat = [i for j in groups for i in j]
    if not all(i in possible_groups for i in groups_flat):
        invalid_group = next(i for i in groups_flat if i not in possible_groups)
        raise ValueError(f'Provided group "{invalid_group}" not in adata.obs["{key}"].')

    # === Step 1: Calculate Percentage Expression for Gene Filtering ===
    add_p_expr = False
    if 'p_expr' not in adata.uns:
        adata.uns['p_expr'] = dict()
        add_p_expr = True

    if key not in adata.uns['p_expr']:
        add_p_expr = True
    else:
        if f'P_expr_{key}' not in adata.varm:
            raise ValueError(
                'Data for p_expr found in adata.uns but not in adata.varm. '
                'Please remove p_expr from adata.uns and re-run.')
        add_p_expr, vcounts = check_p_expr(adata, key, mode='validate')

    X = adata.layers[layer]
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    if add_p_expr:
        print('Computing percentages for gene filtering...')
        p_expr_list = []
        for g in possible_groups:
            gbool = (adata.obs[key] == g).values
            gX = X[gbool].tocsc()
            # Handle case where a group might be empty
            if gX.shape[0] == 0:
                arr = np.zeros(gX.shape[1])
            else:
                arr = gX.getnnz(axis=0) / gX.shape[0]
            p_expr_list.append(arr)
        adata.varm[f'P_expr_{key}'] = np.vstack(p_expr_list).T
        check_p_expr(adata, key, mode='update')

    p_expr_df = pd.DataFrame(adata.varm[f'P_expr_{key}'], index=adata.var_names, columns=possible_groups).T

    # === Step 1b: Filter Genes Based on Expression Percentage (New Logic) ===
    passing_genes_mask = None
    target_groups = groups[0]

    if len(target_groups) == 1:
        passing_genes_mask = (p_expr_df.loc[target_groups[0]] > p).values
    else:
        if p_of is None:
            print(f"Filtering based on combined expression percentage across: {target_groups}")
            combined_mask = adata.obs[key].isin(target_groups).values
            X_subset = X[combined_mask]
            p_expr_combined = X_subset.getnnz(axis=0) / X_subset.shape[0]
            passing_genes_mask = p_expr_combined > p
        elif p_of == 'any':
            print(f"Filtering based on ANY subgroup passing threshold in: {target_groups}")
            passing_genes_mask = (p_expr_df.loc[target_groups] > p).any(axis=0).values
        elif is_listlike(p_of):
            if not np.all(np.isin(p_of, target_groups)):
                raise ValueError(f'If `p_of` is a list, it must be a subset of the numerator groups: {target_groups}')
            print(f"Filtering based on combined expression percentage across specified subgroups: {p_of}")
            combined_mask = adata.obs[key].isin(p_of).values
            X_subset = X[combined_mask]
            p_expr_combined = X_subset.getnnz(axis=0) / X_subset.shape[0]
            passing_genes_mask = p_expr_combined > p
        else:
            raise TypeError("`p_of` must be None, 'any', or a list-like object.")

    passing_genes = p_expr_df.columns[passing_genes_mask]
    if len(passing_genes) == 0:
        print("Warning: No genes passed the filtering criteria.")
        return pd.Series(dtype=np.float64)

    # === Step 2: Prepare DataFrame for Ratio Calculation ===
    sub_df = None
    if ratio == 'mean':
        print('Computing mean expression of positive cells...')
        mean_expr_list = []
        for g in possible_groups:
            gbool = (adata.obs[key] == g).values
            gX = X[gbool].tocsc()
            sums = np.array(gX.sum(axis=0)).flatten()
            nnz = gX.getnnz(axis=0)
            means = np.divide(sums, nnz, out=np.zeros_like(sums, dtype=float), where=nnz!=0)
            mean_expr_list.append(means)
        
        mean_expr_df = pd.DataFrame(np.vstack(mean_expr_list), index=possible_groups, columns=adata.var_names)
        sub_df = mean_expr_df[passing_genes]

    elif ratio == 'percent':
        print('Using percentage of expressing cells for ratio...')
        sub_df = p_expr_df[passing_genes]

    # === Step 3: Compute the Ratio ===
    div = dict(zip(['num', 'den'], [0, 0]))
    for group, val in zip(groups, ['num', 'den']):
        if len(group) == 1:
            div[val] = sub_df.loc[group[0]]
        else:
            weights = vcounts.loc[group].values
            div[val] = sub_df.loc[group].apply(lambda x: np.average(x, weights=weights), axis=0)
    
    epsilon = 1e-9
    result = div['num'] / (div['den'] + epsilon)

    return result.sort_values(ascending=False)


class Annotate:
    """Manages annotation transfer from a reference dataframe to an AnnData object.

    This class provides a semi-automated workflow to map cell type annotations
    from a reference `pandas.DataFrame` onto the clusters of a `scanpy.AnnData`
    object. It assumes the reference annotations follow a hierarchical structure
    (e.g., Level 1: Immune, Level 2: T-cell, Level 3: CD4+ T-cell).

    The workflow involves three main steps:
    1.  **Initialization**: The object is created, which automatically builds the
        annotation hierarchy, aligns the reference and target datasets by a
        shared identifier, and calculates the mapping rates between each cluster
        and the available annotations.
    2.  **Inspection**: Helper methods like `show_tree()` and `print_low_map_rates()`
        are used to understand the annotation structure and identify clusters
        that cannot be automatically annotated with high confidence.
    3.  **Application**: The `apply()` method is called with a confidence
        threshold and a manual mapping for low-confidence clusters. This adds
        the new, complete annotation columns to `adata.obs`.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix to which annotations will be added.
    df : pd.DataFrame
        A dataframe containing the reference annotations. It must contain the
        `id_var` column and the columns specified in `annot_vars`.
    annot_vars : List[str]
        A list of column names in `df` that represent the annotation
        hierarchy, ordered from the most general to the most granular.
    id_var : str, optional
        The name of the column present in both `adata.obs` and `df` used to
        align cells between the two datasets. Defaults to the reaction id "rxn".
    cluster_key : str, optional
        The column name in `adata.obs` that contains the cluster labels
        (e.g., 'leiden', 'louvain'). Defaults to 'leiden'.

    Attributes
    ----------
    adata : sc.AnnData
        The `AnnData` object being annotated.
    annot_vars : List[str]
        The list of hierarchical annotation column names.
    hierarchy_root : anytree.Node
        The root node of the annotation hierarchy tree built from `df`.
    leaf_id_map : Dict[str, int]
        A dictionary mapping the most granular annotation labels (leaves) to
        unique integer IDs.
    mapping : Dict[str, ClusterMap]
        A dictionary where keys are cluster labels and values are `ClusterMap`
        objects containing mapping statistics for that cluster.

    Examples
    --------
    >>> # Assume `adata` is a clustered AnnData object and `ref_df` is a DataFrame
    >>> # with annotations.
    >>> # ref_df might look like:
    >>> #           rxn   ct1          ct2
    >>> # ACGT-1-0  r1    Immune       T-cell
    >>> # TGCA-1-0  r2    Immune       B-cell
    >>>
    >>> # 1. Initialize the Annotate object
    >>> annotator = Annotate(
    ...     adata=adata,
    ...     df=ref_df,
    ...     annot_vars=['ct1', 'ct2'],
    ...     id_var='rxn',
    ...     cluster_key='leiden'
    ... )
    
    >>> # 2. Inspect the hierarchy and mapping rates
    >>> annotator.show_tree()
    --- Detected Annotation Hierarchy ---
    ROOT
    └── Immune
        ├── T-cell (ID: 0)
        └── B-cell (ID: 1)
    -----------------------------------

    >>> # Identify clusters that need manual annotation (e.g., max proportion < 75%)
    >>> annotator.print_low_map_rates(map_thresh=0.75)
    Cluster 5:
    T-cell (0): 0.60, B-cell (1): 0.35

    >>> # 3. Apply the annotations
    >>> # For cluster '5', we decide to call it 'T-cell' using its leaf ID.
    >>> # For cluster '8', we define a completely new annotation path.
    >>> cluster_assignments = {
    ...     '5': 0,  # Assign using leaf ID for T-cell
    ...     '8': ['Stromal', 'Fibroblast']
    ... }
    >>> annotator.apply(threshold=0.75, cluster_map=cluster_assignments)

    >>> # 4. Verify the results
    >>> print(adata.obs[['leiden', 'ct1', 'ct2']].head())
    """
    class ClusterMap:
        """A simple data class to hold mapping results for a single cluster."""
        def __init__(self, cluster, prop_new, vcounts):
            self.cluster = cluster
            self.prop_new = prop_new
            self.vcounts = vcounts

        def __repr__(self):
            return f"ClusterMap({self.cluster})"

    def __init__(
        self,
        adata: sc.AnnData,
        df: pd.DataFrame,
        annot_vars: List[str],
        id_var: str = "rxn",
        cluster_key: str = 'leiden',
        verbose=False
    ):
        # --- Input Validation ---
        if not isinstance(adata, sc.AnnData):
            raise TypeError("`adata` must be a `scanpy.AnnData` object.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a `pandas.DataFrame` object.")
        if not is_listlike(annot_vars) or not all(isinstance(i, str) for i in annot_vars):
            raise TypeError("`annot_vars` must be a list-like object of strings.")
        if not isinstance(id_var, str):
            raise TypeError("`id_var` must be a string.")
        if not isinstance(cluster_key, str):
            raise TypeError("`cluster_key` must be a string.")
        
        # svc.meta.is_key_categorical(adata, cluster_key, error=True)
        clusters = adata.obs[cluster_key].cat.categories

        # --- Compatibility Checks ---
        if id_var not in df.columns:
            raise ValueError(f"The `id_var` '{id_var}' is not a column in the provided DataFrame `df`.")
        if id_var not in adata.obs.columns:
            raise ValueError(f"The `id_var` '{id_var}' is not a column in `adata.obs`.")

        missing_annot_vars = [col for col in annot_vars if col not in df.columns]
        if missing_annot_vars:
            raise ValueError(f"The following `annot_vars` are not in the provided DataFrame `df`: {missing_annot_vars}")

        if cluster_key not in adata.obs.columns:
            raise ValueError(f"The `cluster_key` '{cluster_key}' is not a column in `adata.obs`.")

        # --- Store attributes ---
        self.adata = adata
        self.obs = adata.obs.copy()
        self.df = df.copy()
        self.annot_vars = annot_vars
        self.id_var = id_var
        self.cluster_key = cluster_key
        self.verbose = verbose

        self.clusters = clusters
        self.hierarchy_root = None
        self.leaf_id_map = None
        self.mapping = {}
        

        # --- Build Hierarchy ---
        self._build_hierarchy_tree()

        # --- Align Cell Barcodes ---
        self.align_cell_barcodes(self.adata, self.df, self.id_var, verbose=self.verbose)

        # --- Create Cluster Mapping ---
        self._add_mapping()
        
        if self.verbose:
            print("Annotate class initialized successfully.")

    def _build_hierarchy_tree(self):
        """Builds an anytree representation of the annotation hierarchy.

        This internal method validates that the hierarchy defined by `annot_vars`
        is a valid tree structure (i.e., each child has only one parent) and
        then constructs the tree. It correctly handles cases where nodes at
        different levels share the same name.
        """
        if self.verbose:
            print("Detecting and validating hierarchy...")
        hierarchy_df = self.df[self.annot_vars].drop_duplicates().dropna(how='all')

        leaf_labels = self.df[self.annot_vars[-1]].dropna().unique()
        self.leaf_id_map = {label: i for i, label in enumerate(leaf_labels)}

        if len(self.annot_vars) < 2:
            warnings.warn("Warning: Less than 2 `annot_vars` provided. No hierarchy to build.")
            self.hierarchy_root = Node("ROOT")
            if len(self.annot_vars) == 1:
                for label in hierarchy_df[self.annot_vars[0]].unique():
                    Node(label, parent=self.hierarchy_root, uid=self.leaf_id_map.get(label))
            return

        for i in range(1, len(self.annot_vars)):
            parent_col = self.annot_vars[i-1]
            child_col = self.annot_vars[i]
            inconsistency_check = hierarchy_df.dropna(subset=[child_col]).groupby(child_col, observed=True)[parent_col].nunique()
            if inconsistency_check.max() > 1:
                offenders = inconsistency_check[inconsistency_check > 1].index.tolist()
                raise ValueError(
                    f"Hierarchy is invalid. Labels in '{child_col}' map to multiple "
                    f"labels in '{parent_col}'.\nOffending labels: {offenders[:5]}"
                )

        self.hierarchy_root = Node("ROOT")
        node_map = {(): self.hierarchy_root} 

        for _, row in hierarchy_df.iterrows():
            path_prefix = ()
            for label in row:
                if pd.isna(label):
                    break
                current_path = path_prefix + (label,)
                if current_path not in node_map:
                    parent_node = node_map[path_prefix]
                    node = Node(label, parent=parent_node)
                    node_map[current_path] = node
                path_prefix = current_path
        
        for leaf in self.hierarchy_root.leaves:
            if leaf.name in self.leaf_id_map:
                leaf.uid = self.leaf_id_map[leaf.name]
        
        if self.verbose:
            print("Hierarchy tree built successfully.")

    @staticmethod
    def align_cell_barcodes(adata, df, id_var, verbose=False):
        """Aligns cell barcodes between `adata.obs` and the reference `df`.

        This internal method handles cases where cell barcodes may require a
        batch identifier to be unique. It modifies `self.df` in place to ensure
        its index aligns with `self.adata.obs.index`.
        """
        if verbose:
            print("Aligning cell barcodes...")

        adata.obs['detected_batch'] = adata.obs.index.str.split('-').str[1]

        try:
            id_to_batch_map = get_functional_dependency(adata.obs, (id_var, 'detected_batch'))
        except ValueError as e:
            raise ValueError(f"Could not create a unique mapping from '{id_var}' to 'detected_batch' in adata.obs. {e}")

        df['correct_batch'] = df[id_var].map(id_to_batch_map)
        df.dropna(inplace=True)

        nucleotides = df.index.str.split('-').str[0]
        new_index = nucleotides + '-' + df['correct_batch'].astype(str)

        df.index = new_index
        df.drop(columns=['correct_batch', id_var], inplace=True)

        if verbose:
            print("Cell barcodes aligned.")

    def _add_mapping(self):
        """Calculates annotation proportions for each cluster.

        This internal method iterates through each cluster in `adata`, finds the
        common cells with the reference `df`, and calculates the proportion of
        each granular annotation within those common cells. Results are stored in
        `self.mapping`.
        """
        if self.verbose:
            print("Mapping clusters to annotations...")
        self.mapping = {}
        granular_annot_col = self.annot_vars[-1]

        for cluster in self.clusters:
            cluster_cells = self.obs.index[self.obs[self.cluster_key] == cluster]
            common_cells = cluster_cells.intersection(self.df.index)
            
            if len(cluster_cells) > 0:
                prop_new = 1 - (len(common_cells) / len(cluster_cells))
            else:
                prop_new = 0.0

            if not common_cells.empty:
                vcounts = self.df.loc[common_cells, granular_annot_col].value_counts(normalize=True)
            else:
                vcounts = pd.Series(dtype=float)

            vcounts = vcounts[vcounts > 0]
            self.mapping[cluster] = self.ClusterMap(
                cluster=cluster,
                prop_new=prop_new,
                vcounts=vcounts
            )
        if self.verbose:
            print("Cluster mapping complete.")

    def get_max_map_rates(self) -> Dict[str, float]:
        """Gets the maximum mapping rate for each cluster.

        Returns
        -------
        Dict[str, float]
            A dictionary where keys are cluster labels and values are the
            highest proportion of any single annotation within that cluster.
        """
        max_map_rates = {}
        for cluster in self.clusters:
            cluster_map = self.mapping[cluster]
            max_rate = cluster_map.vcounts.max() if not cluster_map.vcounts.empty else 0
            max_map_rates[cluster] = max_rate
        return max_map_rates

    def get_high_prop_new_rates(self, thresh: float = 0.75) -> pd.Series:
        """Gets cluster proportions for clusters whose prop_new is above a confidence threshold.

        Parameters
        ----------
        thresh : float, optional
            The confidence threshold. Any cluster whose prop_new is greater than this
            value will be included. Defaults to 0.75.

        Returns
        -------
        pd.Series
            A multi-indexed Series showing the annotation proportions for all
            high-confidence clusters.
        """

        high_prop_new_rates = {i: self.mapping[i].prop_new for i in self.clusters if self.mapping[i].prop_new >= thresh}

        return pd.Series(high_prop_new_rates)

    def get_low_map_rates(self, thresh: float = 0.75) -> pd.Series:
        """Gets annotation proportions for clusters below a confidence threshold.

        Parameters
        ----------
        thresh : float, optional
            The confidence threshold. Any cluster whose top-matching annotation
            has a proportion less than this value will be included.
            Defaults to 0.75.

        Returns
        -------
        pd.Series
            A multi-indexed Series showing the annotation proportions for all
            low-confidence clusters.
        """
        low_map_rates = []
        low_map_rate_clusters = []
        for cluster in self.clusters:
            cluster_map = self.mapping[cluster]
            max_rate = cluster_map.vcounts.max() if not cluster_map.vcounts.empty else 0.0
            if max_rate < thresh:
                if cluster_map.vcounts.empty:
                    low_map_rates.append(pd.Series(0, index=['N/A'], dtype=float))
                else:
                    low_map_rates.append(cluster_map.vcounts)
                low_map_rate_clusters.append(cluster)
        
        if not low_map_rates:
            return pd.Series(0, index=['N/A'], dtype=float)
            
        low_map_rates = pd.concat(low_map_rates, axis=0, keys=low_map_rate_clusters)
        return low_map_rates

    def print_low_map_rates(self, map_thresh: float = 0.75, print_thresh: float = 0.05):
        """Prints a summary of clusters with low mapping confidence.

        This is a helper method for interactively identifying which clusters
        will require manual annotation in the `apply` step.

        Parameters
        ----------
        map_thresh : float, optional
            The confidence threshold to identify low-mapping clusters.
            Defaults to 0.75.
        print_thresh : float, optional
            The minimum proportion for an annotation to be printed.
            Defaults to 0.05.
        """
        low_map_rates = self.get_low_map_rates(map_thresh)
        if low_map_rates.empty:
            print(f"No clusters found with max mapping rate below {map_thresh}.")
        else:
            for cluster in low_map_rates.index.get_level_values(0).unique():
                print(f"Cluster {cluster}:")
                proportions = low_map_rates.loc[cluster]
                print_str = ', '.join([
                    f"{ct} ({self.leaf_id_map.get(ct, 'N/A')}): {prop:.2f}"
                    for ct, prop in proportions.items() if prop >= print_thresh
                ])
                print(print_str)
                print()

    def get_top_mapper(self, clusts, map_thresh: float = 0.75):
        """Gets the top-mapping annotation for clusters below a confidence threshold.

        This is a helper method for getting a mapper to supply to `apply()`, mapping
        the specific clusters in `clusts` to their top annotation.

        Parameters
        ----------
        clusts : List[str]
            A list of cluster labels to map, a subset of those with low mapping rates
            based on `map_thresh`.
        map_thresh : float, optional
            The confidence threshold to identify low-mapping clusters.
            Defaults to 0.75.
        """
        top_mapper = {}
        low_map_rates = self.get_low_map_rates(map_thresh)
        if low_map_rates.empty:
            print(f"No clusters found with max mapping rate below {map_thresh}.")
        else:
            for cluster in low_map_rates.index.get_level_values(0).unique():
                if cluster not in clusts:
                    continue
                proportions = low_map_rates.loc[cluster]
                top_ct = proportions.idxmax()
                top_mapper[cluster] = self.leaf_id_map.get(top_ct, 'N/A')
        return top_mapper

    def apply(self, threshold: float, cluster_map: Dict[str, Any] = None):
        """Applies annotations to `adata.obs` based on mapping rates.

        This method performs the final annotation step. For each cluster, if the
        proportion of its most common annotation is at or above the `threshold`,
        it is automatically assigned that annotation's full hierarchy. For all
        clusters falling below the threshold, a manual assignment must be
        provided in the `cluster_map` dictionary.

        This method modifies `self.adata.obs` in place.

        Parameters
        ----------
        threshold : float
            The minimum proportion for a cluster to be automatically annotated.
            Must be between 0 and 1.
        cluster_map : Dict[str, Any], optional
            A dictionary to manually assign annotations to clusters that fall
            below the `threshold`. Keys are cluster labels. Values can be:
            - An integer (a leaf ID from `self.leaf_id_map`).
            - A list of strings representing the full annotation hierarchy.
            Defaults to None.

        Raises
        ------
        ValueError
            If any cluster falls below the `threshold` and is not provided in
            the `cluster_map`.
        """
        if cluster_map is None:
            cluster_map = {}

        unmapped_clusters = []
        for cluster in self.clusters:
            max_rate = self.mapping[cluster].vcounts.max() if not self.mapping[cluster].vcounts.empty else 0
            if max_rate < threshold and cluster not in cluster_map:
                unmapped_clusters.append(cluster)

        if unmapped_clusters:
            raise ValueError(
                f"All clusters must be accounted for. The following clusters have a max "
                f"mapping rate below the threshold ({threshold}) and are not in the "
                f"provided `cluster_map` dictionary: {unmapped_clusters}"
            )

        id_to_leaf_label = {v: k for k, v in self.leaf_id_map.items()}
        leaf_nodes = {node.name: node for node in self.hierarchy_root.leaves}
        new_annots_df = pd.DataFrame(index=self.adata.obs.index, columns=self.annot_vars)

        if self.verbose:
            print("Applying annotations...")
        for cluster in self.clusters:
            cluster_cells = self.obs.index[self.obs[self.cluster_key] == cluster]
            max_rate = self.mapping[cluster].vcounts.max() if not self.mapping[cluster].vcounts.empty else 0
            annotation_path = None

            if max_rate >= threshold:
                top_annot_label = self.mapping[cluster].vcounts.idxmax()
                if top_annot_label in leaf_nodes:
                    node = leaf_nodes[top_annot_label]
                    annotation_path = [n.name for n in node.path[1:]]
            else:
                manual_map = cluster_map[cluster]
                if isinstance(manual_map, int):
                    if manual_map not in id_to_leaf_label:
                        raise ValueError(f"Leaf ID {manual_map} for cluster '{cluster}' not found.")
                    leaf_label = id_to_leaf_label[manual_map]
                    if leaf_label not in leaf_nodes:
                        raise ValueError(f"Leaf label '{leaf_label}' for cluster '{cluster}' not found in hierarchy.")
                    node = leaf_nodes[leaf_label]
                    annotation_path = [n.name for n in node.path[1:]]
                elif is_listlike(manual_map):
                    if len(manual_map) != len(self.annot_vars):
                        raise ValueError(f"Provided path for cluster '{cluster}' has length {len(manual_map)}, expected {len(self.annot_vars)}.")
                    annotation_path = manual_map
                else:
                    raise TypeError(f"Unsupported map type for cluster '{cluster}': {type(manual_map)}.")

            if annotation_path:
                padded_path = annotation_path + [pd.NA] * (len(self.annot_vars) - len(annotation_path))
                new_annots_df.loc[cluster_cells, self.annot_vars] = padded_path

        for col in self.annot_vars:
            self.adata.obs[col] = new_annots_df[col]
            self.adata.obs[col] = self.adata.obs[col].astype('category')
        
        if self.verbose:
            print("Annotations successfully applied to `adata.obs`.")

    def show_tree(self):
        """Prints a visual representation of the detected annotation hierarchy.
        
        Leaf nodes with assigned IDs will be marked.
        """
        if not self.hierarchy_root:
            print("Hierarchy tree has not been built.")
            return
            
        print("--- Detected Annotation Hierarchy ---")
        for pre, _, node in RenderTree(self.hierarchy_root):
            if hasattr(node, 'uid') and node.uid is not None:
                print(f"{pre}{node.name} (ID: {node.uid})")
            else:
                print(f"{pre}{node.name}")
        print("-----------------------------------")

    @staticmethod
    def annotation(adata: sc.AnnData, 
                   path_to_annots: str, 
                   key: str = 'leiden', 
                   overwrite: bool = False, 
                   export: bool = False, 
                   check_genes=False):
        """
        Annotate clusters or export annotations using a TSV file.

        This function can either import cell type definitions from a file into `adata.uns`
        and apply them, or export an existing annotation set from `adata.uns` to a file.

        Parameters
        ----------
        adata : sc.AnnData
            The AnnData object to modify.
        path_to_annots : str
            Path to the TSV annotation file.
        key : str
            The key in `adata.obs` (e.g., 'leiden') and `adata.uns['annot']` to use.
        overwrite : bool
            If True, allows overwriting existing files or `adata.uns` entries.
        export : bool
            If True, exports annotations; if False (default), imports them.
        check_genes : bool
            If True, checks if all marker genes are present in `adata.var_names`.
        """
        if export:
            # --- Export Logic ---
            if key not in adata.uns.get('annot', {}):
                raise ValueError(f"Key '{key}' not in `adata.uns['annot']`. Cannot export.")
            if os.path.exists(path_to_annots) and not overwrite:
                raise ValueError(f"File '{path_to_annots}' exists. Use `overwrite=True`.")

            annotations = adata.uns['annot'][key]
            mappers = annotations.get('mappers', {})
            
            # Use the index from the first mapper as the base
            if not mappers:
                 raise ValueError("No mappers found in annotations to export.")
            base_index = next(iter(mappers.values())).keys()
            
            df_export = pd.DataFrame(index=pd.Index(base_index, name=key))

            for ct_col, mapper_dict in mappers.items():
                df_export[ct_col] = df_export.index.map(mapper_dict)
            
            # Helper to join lists into strings for TSV
            def joiner(x):
                return ', '.join(map(str, x)) if isinstance(x, list) else x

            df_export['Marker Genes'] = df_export.index.map(lambda i: joiner(annotations.get('markers', {}).get(i, '')))
            df_export['Sources'] = df_export.index.map(lambda i: joiner(annotations.get('sources', {}).get(i, '')))
            df_export['Details'] = df_export.index.map(annotations.get('details', {}))
            
            df_export.to_csv(path_to_annots, sep='\t')
            print(f"Annotations exported to '{path_to_annots}'")

        else:
            # --- Import Logic ---
            if 'annot' not in adata.uns:
                adata.uns['annot'] = {}
            if key in adata.uns['annot'] and not overwrite:
                raise ValueError(f"Key '{key}' exists in `adata.uns['annot']`. Use `overwrite=True`.")
            
            df = pd.read_csv(path_to_annots, sep='\t', header=0, index_col=0)
            df.index = df.index.astype(str)

            # Parse mappers, markers, and other details
            mappers = {col: df[col].dropna().to_dict() for col in df.columns if col.startswith('ct')}
            def parser(x):
                return [i.strip() for i in x.strip(' ,').split(',')] if isinstance(x, str) else []
            
            markers = df['Marker Genes'].fillna('').apply(parser).to_dict()
            sources = df['Sources'].fillna('').apply(parser).to_dict()
            details = df['Details'].fillna('').to_dict()

            # Validate marker genes
            if check_genes:
                unique_genes = {gene for gene_list in markers.values() for gene in gene_list}
                missing_genes = unique_genes - set(adata.var_names)
                missing_genes.discard('')  # Remove empty strings if any
                if missing_genes:
                    raise ValueError(f"Genes not in `adata.var_names`: {', '.join(list(missing_genes)[:5])}...")
            
            # Store in adata.uns
            adata.uns['annot'][key] = {
                'mappers': mappers,
                'markers': markers,
                'sources': sources,
                'details': details,
            }

            # Apply annotations to adata.obs
            for ct_col, mapper in mappers.items():
                adata.obs[ct_col] = adata.obs[key].astype(str).map(mapper).astype('category')
            print(f"Annotations for key '{key}' imported and applied.")

