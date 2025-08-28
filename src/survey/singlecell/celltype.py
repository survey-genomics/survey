# Built-ins
import os
from typing import Union, List, Optional, Dict, Tuple
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
from survey.genutils import is_listlike
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
        fss : float, optional
            Figure size scaler.
        ar : float, optional
            Aspect ratio of the plot.
        **kwargs
            Additional arguments passed to the underlying plotting functions.
        """
        if color is None:
            color = self.current_cluster_key

        if mode == 'labeled':
            fig, ax = subplots(1, fss=fss, ar=ar)
            ax = svc.pl.labeled_scatter(data=self.adata, color=color, ax=ax, **kwargs)
        elif mode == 'highlight':
            if clusts is None and self.pending_split_children is not None:
                clusts = self.pending_split_children
            ax = svc.pl.highlight(data=self.adata, color=color, cats=clusts, ar=ar, fss=fss, **kwargs)
        elif mode == 'dual':
            fig, ax = subplots(2, fss=fss, ar=ar)
            ax[0] = svc.pl.scatter(data=self.adata, color=color, ax=ax[0], legend=False)
            ax[1] = svc.pl.labeled_scatter(data=self.adata, numbered=False, color=color, ax=ax[1], legend=True, **kwargs)
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
              p_of: Optional[Union[str, List[str]]] = None) -> pd.Series:
    """
    Compute a specific expression score based on the ratio of mean expression
    in expressing cells between two groups.

    This score is the ratio of the mean expression of a gene in expressing cells
    (>0 counts) of a target group to the mean expression in expressing cells of
    a reference group.

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

    Returns
    -------
    pd.Series
        A Series of specific mean expression scores (numerator/denominator), sorted.

    Raises
    ------
    ValueError
        If invalid groups are provided or if `p_of` is invalid.
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

    groups, vcounts, possible_groups = check_groups(adata, groups, key)

    # Handle the special 'rest' case for the denominator group.
    if len(groups) == 2 and (groups[1] == ['rest'] or groups[1] == 'rest'):
        print("Denominator group 'rest' detected. Using all groups not in the numerator.")
        numerator_groups = groups[0]
        denominator_groups = [g for g in possible_groups if g not in numerator_groups]
        groups[1] = denominator_groups

    # Validate groups
    if not all([is_listlike(g) for g in groups]):
        raise ValueError("All groups must be list-like.")

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
        # Case 1: Only one group in the numerator. Filter is based on this single group.
        passing_genes_mask = (p_expr_df.loc[target_groups[0]] > p).values
    else:
        # Case 2: Multiple groups in the numerator. Logic depends on `p_of`.
        if p_of is None:
            # 2a: Calculate combined percentage across ALL subgroups in groups[0]
            print(f"Filtering based on combined expression percentage across: {target_groups}")
            combined_mask = adata.obs[key].isin(target_groups).values
            X_subset = X[combined_mask]
            p_expr_combined = X_subset.getnnz(axis=0) / X_subset.shape[0]
            passing_genes_mask = p_expr_combined > p
        elif p_of == 'any':
            # 2b: Gene passes if ANY subgroup in groups[0] meets the threshold
            print(f"Filtering based on ANY subgroup passing threshold in: {target_groups}")
            passing_genes_mask = (p_expr_df.loc[target_groups] > p).any(axis=0).values
        elif is_listlike(p_of):
            # 2c: `p_of` is a specific list of subgroups within groups[0]
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

    # === Step 2: Calculate Mean Expression of Positive Cells ===
    print('Computing mean expression of positive cells...')
    mean_expr_list = []
    for g in possible_groups:
        gbool = (adata.obs[key] == g).values
        gX = X[gbool].tocsc()
        sums = np.array(gX.sum(axis=0)).flatten()
        nnz = gX.getnnz(axis=0)
        means = np.divide(sums, nnz, out=np.zeros_like(sums, dtype=float), where=nnz!=0)
        mean_expr_list.append(means)

    mean_expr_df = pd.DataFrame(np.vstack(mean_expr_list), index=possible_groups, columns=adata.var_names).T
    sub_mean_df = mean_expr_df.loc[passing_genes].T

    # === Step 3: Compute the Ratio ===
    div = dict(zip(['num', 'den'], [0, 0]))
    for group, val in zip(groups, ['num', 'den']):
        if len(group) == 1:
            div[val] = sub_mean_df.loc[group[0]]
        else:
            weights = vcounts.loc[group].values
            div[val] = sub_mean_df.loc[group].apply(lambda x: np.average(x, weights=weights), axis=0)
    
    epsilon = 1e-9
    result = div['num'] / (div['den'] + epsilon)

    return result.sort_values(ascending=False)



class Annotate:
    """
    Manages annotation transfer, hierarchy recovery, and visualization for single-cell data.

    This class provides a streamlined workflow for applying existing cell type annotations
    to a new clustering of the same dataset. It handles mapping, corrects for hierarchical
    inconsistencies that may arise from more granular clustering, and offers tools for
    visualization and final application of the new labels.
    """

    def __init__(
        self,
        adata_or_mdata: Union[sc.AnnData, md.MuData],
        annot_cols: List[str],
        col: str,
        modality_key: Optional[str] = None,
        prefix_prev: str = 'previous',
        prefix_new: str = 'new',
        reset: bool = True,
        verbose: bool = False
    ):
        """
        Initializes the Annotate object.

        Parameters
        ----------
        adata_or_mdata : Union[sc.AnnData, md.MuData]
            The AnnData or MuData object. If MuData, `modality_key` is required.
        annot_cols : List[str]
            Original annotation columns, ordered from lowest to highest granularity
            (e.g., ['ct1', 'ct2', 'ct3']).
        col : str
            The column with new clustering labels (e.g., 'leiden').
        modality_key : Optional[str]
            If using a MuData object, the key for the target AnnData modality.
        prefix_prev : str
            Prefix for renaming original annotation columns (e.g., 'previous_ct1').
        prefix_new : str
            Prefix for newly created annotation columns (e.g., 'new_ct1').
        reset : bool
            If True, removes any existing columns with the defined prefixes.
        verbose : bool
            If True, enables detailed print statements.
        """
        # --- Input Validation and Data Handling ---
        if isinstance(adata_or_mdata, sc.AnnData):
            self.adata = adata_or_mdata
        elif isinstance(adata_or_mdata, md.MuData):
            if modality_key is None:
                raise ValueError("If a MuData object is provided, 'modality_key' must be specified.")
            if modality_key not in adata_or_mdata.mod:
                raise ValueError(f"Modality key '{modality_key}' not in MuData keys: {list(adata_or_mdata.mod.keys())}")
            self.adata = adata_or_mdata[modality_key]
        else:
            raise TypeError("Input 'adata_or_mdata' must be an AnnData or MuData object.")

        if not isinstance(annot_cols, list) or not annot_cols:
            raise ValueError("'annot_cols' must be a non-empty list.")
        if not isinstance(col, str) or not col:
            raise ValueError("'col' must be a non-empty string.")
        if not isinstance(prefix_prev, str):
            raise TypeError("'prefix_prev' must be a string.")
        if not isinstance(prefix_new, str):
            raise TypeError("'prefix_new' must be a string.")

        self.annot_cols = annot_cols
        self.col = col
        self.prefix_prev = prefix_prev
        self.prefix_new = prefix_new
        self.verbose = verbose

        # --- Initialize results attributes ---
        self.proportions_df: Optional[pd.DataFrame] = None
        self.hierarchy_df: Optional[pd.DataFrame] = None
        self.hierarchy_lookup: Optional[pd.DataFrame] = None
        self.inconsistent_clusters: Optional[List[Union[str, int]]] = None

        self._print("Annotate initialized.")
        self._print(f"Target AnnData: {f'MuData modality {repr(modality_key)}' if modality_key else 'Provided AnnData'}")
        self._print(f"Clustering column: '{self.col}'")
        self._print(f"Annotation columns: {self.annot_cols}")

        if reset:
            self._reset_columns()

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _print(self, *args, **kwargs):
        """Prints only if self.verbose is True."""
        if self.verbose:
            print(*args, **kwargs)

    def _get_prev_col_name(self, col_name: str) -> str:
        """Constructs the 'previous' column name."""
        return f"{self.prefix_prev}_{col_name}"

    def _get_new_col_name(self, col_name: str) -> str:
        """Constructs the 'new' column name."""
        return f"{self.prefix_new}_{col_name}"

    def _reset_columns(self):
        """Removes existing prefixed columns from `adata.obs`."""
        self._print("Resetting columns...")
        prev_cols_to_drop = [self._get_prev_col_name(c) for c in self.annot_cols if self._get_prev_col_name(c) in self.adata.obs]
        if prev_cols_to_drop:
            self._print(f"  Dropping previous annotation columns: {prev_cols_to_drop}")
            self.adata.obs.drop(columns=prev_cols_to_drop, inplace=True)

        new_cols_to_drop = [self._get_new_col_name(c) for c in self.annot_cols if self._get_new_col_name(c) in self.adata.obs]
        if new_cols_to_drop:
            self._print(f"  Dropping new annotation columns: {new_cols_to_drop}")
            self.adata.obs.drop(columns=new_cols_to_drop, inplace=True)
            
    # --------------------------------------------------------------------------
    # Core Methods
    # --------------------------------------------------------------------------

    def annotate(self, thresh: float = 0.8, fill: str = 'no_match') -> pd.DataFrame:
        """
        Automatically annotates new clusters based on previous annotations.

        This method calculates the proportion of cells in each new cluster that
        map to a previous annotation. If the proportion exceeds `thresh`, the new
        cluster is assigned that label.

        Parameters
        ----------
        thresh : float
            Minimum proportion (0.0 to 1.0) for a label to be assigned.
        fill : str
            Value for clusters that do not meet the threshold.

        Returns
        -------
        pd.DataFrame
            A DataFrame with detailed mapping proportions and metrics, also
            stored in `self.proportions_df`.
        """
        self._print("\n--- Starting Annotation ---")
        self._validate_annotate_inputs(thresh)

        # 1. Rename original columns to 'previous_*'
        renamed_map = self._rename_original_columns()

        # 2. Pre-calculate total sizes of each previous label for efficiency
        prev_label_sizes = {
            prev_col: self.adata.obs[prev_col].value_counts()
            for prev_col in renamed_map.values()
        }

        # 3. Perform the core mapping logic
        proportion_details, new_cluster_ids = self._calculate_mapping(renamed_map, prev_label_sizes, thresh, fill)

        # 4. Format results into a final DataFrame
        self.proportions_df = self._create_proportions_df(proportion_details, new_cluster_ids)
        self._print("--- Annotation Finished ---\n")
        return self.proportions_df

    def recover(self, include: List[Union[str, int]] = [], fill: str = 'no_match') -> List[Union[str, int]]:
        """
        Corrects hierarchical inconsistencies in new annotations.

        This method identifies new annotations that violate the hierarchy established
        by the previous labels (e.g., a subtype is assigned to the wrong parent type).
        It can then force-correct specified clusters.

        Parameters
        ----------
        include : List[Union[str, int]]
            A list of cluster IDs to force-correct based on the hierarchy.
        fill : str
            The value for unassigned annotations, which are ignored.

        Returns
        -------
        List[Union[str, int]]
            A list of cluster IDs found to be inconsistent *before* correction.
            Also stored in `self.inconsistent_clusters`.
        """
        self._print("\n--- 🔍 Starting Hierarchy Recovery ---")
        self._validate_recover_inputs(include)

        # 1. Build and validate the hierarchy from the previous annotations
        self._build_and_validate_hierarchy()

        # 2. Identify clusters with inconsistent new annotations
        self.inconsistent_clusters = self._identify_inconsistent_clusters(fill)
        self._print(f"  Found {len(self.inconsistent_clusters)} inconsistent clusters.")

        # 3. Correct the specified clusters
        if include:
            self._correct_included_clusters(include)
        
        self._print("--- Hierarchy Recovery Finished ---\n")
        return self.inconsistent_clusters

    def plot_heatmap(self, figsize: tuple = (10, 10), cmap: str = 'Blues', show_cbar: bool = False, nan_annot: str = "") -> Optional[plt.Axes]:
        """
        Generates a heatmap of inverse proportions for sub-threshold clusters.

        This visualizes potential annotations for clusters that were not confidently
        mapped. The color shows the inverse proportion ($|N \\cap L| / |L|$), helping
        to identify if a small cluster is a pure subset of a larger original cell type.

        Parameters
        ----------
        figsize : tuple
            Figure size for the plot.
        cmap : str
            Colormap for the heatmap.
        show_cbar : bool
            Whether to display the color bar.
        nan_annot : str
            String to display for empty annotations.

        Returns
        -------
        Optional[plt.Axes]
            The Matplotlib Axes object for the heatmap, or None if no data is plotted.
        """
        self._print("\n--- Generating Heatmap ---")
        if self.proportions_df is None:
            raise AttributeError("The 'annotate' method must be run first.")

        ax = self._plot_unmapped_heatmap(
            proportions_df=self.proportions_df,
            figsize=figsize,
            cmap=cmap,
            show_cbar=show_cbar,
            nan_annot=nan_annot
        )
        self._print("--- Heatmap Generation Finished ---\n")

        if ax is not None:
            ax.grid(False)
        return ax

    def apply_annotations(self, keep_previous: bool = False) -> None:
        """
        Finalizes annotations by renaming 'new_*' columns to their original names.

        This is the final step, making the new annotations the primary ones in `adata.obs`.

        Parameters
        ----------
        keep_previous : bool
            If False (default), the `previous_*` columns are deleted.
        """
        self._print("\n--- Applying Final Annotations ---")
        if not isinstance(keep_previous, bool):
            raise TypeError("'keep_previous' must be a boolean.")

        for col_name in self.annot_cols:
            prev_name = self._get_prev_col_name(col_name)
            new_name = self._get_new_col_name(col_name)

            # Rename 'new_*' to original name
            if new_name in self.adata.obs.columns:
                if col_name in self.adata.obs.columns and col_name != new_name:
                    warnings.warn(f"'{col_name}' exists and will be overwritten by '{new_name}'.", UserWarning)
                self.adata.obs.rename(columns={new_name: col_name}, inplace=True)
                self._print(f"  Renamed '{new_name}' to '{col_name}'.")
            else:
                warnings.warn(f"Column '{new_name}' not found. Cannot apply annotation for '{col_name}'.", UserWarning)

            # Optionally delete 'previous_*' column
            if not keep_previous and prev_name in self.adata.obs.columns:
                self._print(f"  Deleting '{prev_name}'.")
                del self.adata.obs[prev_name]
        
        self._print("--- Annotation Application Finished ---\n")

    # --------------------------------------------------------------------------
    # Private Refactored Helper Functions
    # --------------------------------------------------------------------------

    def _validate_annotate_inputs(self, thresh: float):
        """Validates inputs for the annotate method."""
        if self.col not in self.adata.obs.columns:
            raise ValueError(f"Clustering column '{self.col}' not found in `adata.obs`.")
        missing_cols = [c for c in self.annot_cols if c not in self.adata.obs.columns]
        if missing_cols:
            raise ValueError(f"Original annotation columns not found: {missing_cols}")
        if not (0.0 <= thresh <= 1.0):
            raise ValueError("'thresh' must be between 0.0 and 1.0.")
        self._print("  Input validation passed.")

    def _rename_original_columns(self) -> Dict[str, str]:
        """Renames original annotation columns with the 'previous' prefix."""
        self._print("  Renaming original annotation columns...")
        renamed_map = {}
        for old_col in self.annot_cols:
            prev_col_name = self._get_prev_col_name(old_col)
            if prev_col_name in self.adata.obs.columns and old_col != prev_col_name:
                warnings.warn(f"Target '{prev_col_name}' exists and will be overwritten.", UserWarning)
            self.adata.obs.rename(columns={old_col: prev_col_name}, inplace=True)
            renamed_map[old_col] = prev_col_name
        return renamed_map
    
    def _calculate_mapping(self, renamed_map: dict, prev_label_sizes: dict, thresh: float, fill: str) -> Tuple[list, pd.Index]:
        """Performs the core annotation mapping logic."""
        self._print(f"  Mapping clusters in '{self.col}'...")
        is_cat = isinstance(self.adata.obs[self.col].dtype, pd.CategoricalDtype)
        cluster_ids = self.adata.obs[self.col].cat.categories if is_cat else pd.Index(np.sort(self.adata.obs[self.col].unique()))
        cluster_sizes = self.adata.obs.groupby(self.col, observed=is_cat).size()

        if cluster_sizes.empty:
            warnings.warn("No clusters found.", UserWarning)
            return [], pd.Index([])

        proportion_details = []
        for orig_col, prev_col in renamed_map.items():
            new_annot_col = self._get_new_col_name(orig_col)
            counts = self.adata.obs.dropna(subset=[prev_col]).groupby([self.col, prev_col], observed=is_cat).size()
            
            final_mapping = {}
            for cid in cluster_ids:
                best_label, prop, inv_prop = np.nan, 0.0, np.nan
                assigned_label = fill
                
                if cid in counts.index.get_level_values(0):
                    group_counts = counts.loc[cid]
                    if not group_counts.empty:
                        best_label = group_counts.idxmax()
                        best_count = group_counts.max()
                        total_cluster_size = cluster_sizes.get(cid, 0)
                        
                        if total_cluster_size > 0:
                            prop = best_count / total_cluster_size
                            if prop >= thresh:
                                assigned_label = best_label
                            else: # Only calculate inverse for sub-threshold
                                total_label_size = prev_label_sizes.get(prev_col, {}).get(best_label, 0)
                                if total_label_size > 0:
                                    inv_prop = best_count / total_label_size
                
                proportion_details.append({
                    'cluster_id': cid, 'annotation_level': orig_col, 
                    'best_previous_label': best_label, 'proportion': prop, 
                    'inverse_proportion': inv_prop
                })
                final_mapping[cid] = assigned_label

            self.adata.obs[new_annot_col] = self.adata.obs[self.col].map(final_mapping).astype('category')
        
        return proportion_details, cluster_ids

    def _create_proportions_df(self, details: list, cluster_ids: pd.Index) -> pd.DataFrame:
        """Pivots and formats the final proportions DataFrame."""
        self._print("  Creating proportions DataFrame...")
        if not details:
            warnings.warn("No proportion details generated.", UserWarning)
            return pd.DataFrame(index=pd.Index(cluster_ids, name=self.col))

        long_df = pd.DataFrame(details)
        try:
            pivoted_df = long_df.pivot(
                index='cluster_id', columns='annotation_level',
                values=['best_previous_label', 'proportion', 'inverse_proportion']
            )
            pivoted_df = pivoted_df.reindex(cluster_ids)
            pivoted_df.index = pivoted_df.index.astype(self.adata.obs[self.col].dtype)
            pivoted_df.index.name = self.col
            pivoted_df.columns = pivoted_df.columns.rename(['metric', 'annotation_level'])
            return pivoted_df.sort_index()
        except Exception as e:
            warnings.warn(f"Could not pivot DataFrame, returning long format. Error: {e}", UserWarning)
            return long_df.set_index('cluster_id')

    def _validate_recover_inputs(self, include: list):
        """Validates inputs for the recover method."""
        if self.proportions_df is None:
            raise AttributeError("'annotate' must be run first.")
        if not isinstance(include, list):
            raise TypeError("'include' must be a list.")
        
        prev_cols = [self._get_prev_col_name(c) for c in self.annot_cols]
        new_cols = [self._get_new_col_name(c) for c in self.annot_cols]
        missing_cols = [c for c in prev_cols + new_cols if c not in self.adata.obs]
        if missing_cols:
            raise ValueError(f"Required columns missing from `adata.obs`: {missing_cols}")

        valid_cids = set(self.adata.obs[self.col].unique())
        invalid_ids = [c for c in include if c not in valid_cids]
        if invalid_ids:
            raise ValueError(f"Invalid cluster IDs in 'include': {invalid_ids}")

    def _build_and_validate_hierarchy(self):
        """Builds a lookup table from the previous annotations and validates its hierarchy."""
        self._print("  Detecting and validating hierarchy...")
        prev_cols = [self._get_prev_col_name(c) for c in self.annot_cols]
        self.hierarchy_df = self.adata.obs[prev_cols].drop_duplicates().dropna(how='all')
        if self.hierarchy_df.empty:
            raise ValueError("No hierarchy structure found in 'previous_*' columns.")
        
        # Validate hierarchy
        for i in range(len(self.annot_cols) - 1):
            higher_level, lower_level = self.annot_cols[i], self.annot_cols[i+1]
            prev_higher = self._get_prev_col_name(higher_level)
            prev_lower = self._get_prev_col_name(lower_level)
            
            # Check if any lower-level label maps to more than one higher-level label
            is_lower_cat = isinstance(self.hierarchy_df[prev_lower], pd.CategoricalDtype)
            inconsistency_check = self.hierarchy_df.groupby(prev_lower, observed=is_lower_cat)[prev_higher].nunique()
            if inconsistency_check.max() > 1:
                offenders = inconsistency_check[inconsistency_check > 1].index.tolist()
                raise ValueError(f"Hierarchy invalid: '{lower_level}' maps to multiple '{higher_level}'. Offenders: {offenders[:3]}")

        # Create lookup table using the highest resolution annotation as the key
        highest_res_col = prev_cols[-1]
        self.hierarchy_lookup = self.hierarchy_df.drop_duplicates(subset=highest_res_col).set_index(highest_res_col)
        self._print("  Hierarchy validation passed.")

    def _identify_inconsistent_clusters(self, fill: str) -> List[Union[str, int]]:
        """Identifies clusters whose new annotations violate the established hierarchy."""
        self._print("  Identifying inconsistent clusters...")
        new_cols = [self._get_new_col_name(c) for c in self.annot_cols]
        inconsistent_list = []
        
        # Get unique annotations for each cluster
        cluster_annots = self.adata.obs[[self.col] + new_cols].drop_duplicates()
        
        for _, row in cluster_annots.iterrows():
            cid = row[self.col]
            assigned_annots = row[new_cols]
            
            if (assigned_annots == fill).any():
                continue

            highest_assigned = assigned_annots.iloc[-1]
            try:
                # Get the expected parent annotations from the hierarchy
                expected_parents = self.hierarchy_lookup.loc[highest_assigned]
                
                # Check if the assigned parent annotations match the expected ones
                for i in range(len(self.annot_cols) - 1):
                    assigned_parent = assigned_annots.iloc[i]
                    expected_parent = expected_parents.iloc[i]
                    if assigned_parent != expected_parent:
                        inconsistent_list.append(cid)
                        break # Move to next cluster once an inconsistency is found
            except KeyError:
                warnings.warn(f"Cluster {cid}: Assigned label '{highest_assigned}' not in hierarchy lookup.", UserWarning)
        
        return inconsistent_list

    def _correct_included_clusters(self, include: list):
        """Force-corrects the annotations for a given list of cluster IDs."""
        self._print(f"  Correcting {len(include)} specified clusters...")
        new_cols = [self._get_new_col_name(c) for c in self.annot_cols]
        corrected_count = 0

        for cid in include:
            try:
                best_label = self.proportions_df.loc[cid, ('best_previous_label', self.annot_cols[-1])]
                if pd.isna(best_label):
                    warnings.warn(f"Cluster {cid}: NaN best label, cannot correct.", UserWarning)
                    continue

                # Get the correct full hierarchy for the best label
                correct_hierarchy = self.hierarchy_lookup.loc[best_label]
                
                # Apply the correction to adata.obs
                mask = self.adata.obs[self.col] == cid
                for i, level_name in enumerate(self.annot_cols):
                    new_col_name = new_cols[i]
                    # The highest level's label is `best_label`, parents are from lookup
                    correct_label = best_label if i == len(self.annot_cols) - 1 else correct_hierarchy.iloc[i]

                    # Ensure the category exists before assignment
                    if not isinstance(self.adata.obs[new_col_name], pd.CategoricalDtype):
                        self.adata.obs[new_col_name] = self.adata.obs[new_col_name].astype('category')
                    if correct_label not in self.adata.obs[new_col_name].cat.categories:
                        self.adata.obs[new_col_name] = self.adata.obs[new_col_name].cat.add_categories([correct_label])
                    
                    self.adata.obs.loc[mask, new_col_name] = correct_label
                corrected_count += 1
            except KeyError:
                warnings.warn(f"Cluster {cid}: Best label '{best_label}' not in hierarchy. Skipping correction.", UserWarning)
            except Exception as e:
                warnings.warn(f"Error correcting cluster {cid}: {e}", UserWarning)
        
        self._print(f"  Finished correction: {corrected_count} cluster(s) updated.")

    def _plot_unmapped_heatmap(self, proportions_df, **kwargs) -> Optional[plt.Axes]:
        """Internal plotting logic for the heatmap."""
        try:
            # Filter for clusters where inverse proportion was calculated for at least one level
            idx = pd.IndexSlice
            sub_df = proportions_df.loc[proportions_df.loc[:, idx['inverse_proportion', :]].notna().any(axis=1)]

            if sub_df.empty:
                warnings.warn("No sub-threshold clusters with inverse proportions to plot.", UserWarning)
                return None

            heatmap_data = sub_df.loc[:, idx['inverse_proportion', :]].astype(float)
            annot_labels = pd.DataFrame(index=sub_df.index, columns=heatmap_data.columns, dtype=str)
            
            # Create annotation strings: "Label (Inv. Prop)"
            levels = heatmap_data.columns.get_level_values('annotation_level').unique()
            for level in levels:
                labels = sub_df[('best_previous_label', level)]
                inv_props = sub_df[('inverse_proportion', level)]
                annot_labels[('inverse_proportion', level)] = [
                    f"{lbl} ({ip:.2f})" if pd.notna(ip) else str(lbl) if pd.notna(lbl) else ""
                    for lbl, ip in zip(labels, inv_props)
                ]

            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
            sns.heatmap(
                heatmap_data,
                annot=annot_labels,
                fmt='s',
                cmap=kwargs.get('cmap', 'Blues'),
                cbar=kwargs.get('show_cbar', False),
                cbar_kws={'label': 'Inverse Proportion (|N∩L|/|L|)'},
                linewidths=0.5,
                linecolor='black',
                xticklabels=levels,
                ax=ax
            )
            ax.set_title('Potential Annotation Mapping (Inverse Proportion)')
            ax.set_ylabel('New Clusters (Sub-Threshold)')
            ax.set_xlabel('Annotation Levels')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            fig.tight_layout()
            return ax

        except Exception as e:
            warnings.warn(f"Heatmap generation failed: {e}", UserWarning)
            return None

    @staticmethod
    def annotation(adata: sc.AnnData, path_to_annots: str, key: str = 'leiden', overwrite: bool = False, export: bool = False):
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

