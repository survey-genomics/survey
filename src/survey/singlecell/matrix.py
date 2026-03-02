'''
The detools module for functions aiding in analysis of DE results.
'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist


# KMeans and Hierarchical Clustering Convenience Functions
def get_dists(df, ks, metric='euclidean', method='ward', clustering_type='k'):
    """
    Get distortions with varying `k` for k-means or hierarchical clustering.

    This function computes the within-cluster sum of squares (WCSS) for different
    numbers of clusters obtained by:
    - k-means clustering (if `clustering_type` is 'kmeans')
    - Cutting the hierarchical clustering dendrogram (if `clustering_type` is 'hierarchical')

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame on which to perform clustering, with observations
        as rows and variables as columns.
    ks: 1D ndarray
        Values for `k` (number of clusters) to test.
    metric: str, optional
        Distance metric to use for linkage calculation (hierarchical) or
        cdist (kmeans). Default is 'euclidean'.
    method: str, optional
        Linkage method to use for hierarchical clustering. Default is 'ward'.
        Only used if `clustering_type` is 'hierarchical'.
    clustering_type: str, optional
        Type of clustering to perform. Either 'k' for kmeans or 'h' for hierarchical.
        Default is 'k'.

    Returns
    -------
    list
        List of distortions (WCSS) for each value of `k`.
    """

    dists = []

    ks = ks

    if clustering_type == 'k':
        for k in ks:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(df)
            dists.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, metric), axis=1)) / df.shape[0])

    elif clustering_type == 'h':
        # Calculate the linkage matrix
        Z = linkage(df, metric=metric, method=method)

        for k in ks:
            # Get cluster labels for the current k
            labels = fcluster(Z, k, criterion='maxclust')

            # Calculate WCSS for the current k
            wcss = 0
            for cluster_id in range(1, k + 1):
                cluster_indices = np.where(labels == cluster_id)
                cluster_points = df.iloc[cluster_indices]
                cluster_mean = cluster_points.mean(axis=0)
                wcss += ((cluster_points - cluster_mean) ** 2).sum().sum()

            dists.append(wcss)

    else:
        raise ValueError("Invalid clustering_type. Choose either 'kmeans' or 'hierarchical'.")

    return dists


def plot_dists(dists_data, ks, slopes_plot='bar', slopes_bar_params=None,
               slopes_line_params=None, return_slopes=False, smooth=None, ci_percentiles=(2.5, 97.5)):
    '''
    Plot the distortions and slope between each k. Handles single or multiple trial data.

    Parameters
    ----------
    dists_data : 1D or 2D array-like
        If 1D: List/array of distortions from a single run.
        If 2D: Array of distortions where rows are trials and columns correspond to ks.
    ks : 1D array-like
        Values for k tested.
    slopes_plot : str, optional
        Type of plot for slopes: 'bar' or 'line'. Default is 'bar'.
    slopes_bar_params : dict, optional
        Parameters for the slopes bar plot.
    slopes_line_params : dict, optional
        Parameters for the slopes line plot (if slopes_plot='line').
    return_slopes : bool, optional
        If True, return the calculated (median if multiple trials) slopes. Default is False.
    smooth : list-like or None, optional
        If provided, smooths the median distortion and slope plots.
        Format: [window_size, function, *args]. Default is None.
    ci_percentiles : tuple, optional
        Tuple defining the lower and upper percentiles for the confidence interval
        when multiple trials are provided. Default is (2.5, 97.5) for 95% CI.

    Returns
    -------
    ax : list
        List of 2 matplotlib Axes objects (distortion plot, slopes plot).
    slopes : np.ndarray (optional)
        The calculated slopes (median if multiple trials), returned if return_slopes=True.
    '''
    assert slopes_plot in ['bar', 'line'], "slopes_plot must be 'bar' or 'line'"
    ks = np.array(ks)
    dists_data = np.array(dists_data)
    assert dists_data.shape[-1] == len(ks), "Last dimension of dists_data must match length of ks"

    multiple_trials = dists_data.ndim == 2 and dists_data.shape[0] > 1

    if multiple_trials:
        # Calculate median and confidence intervals for distortions
        median_dists = np.median(dists_data, axis=0)
        lower_ci_dists = np.percentile(dists_data, ci_percentiles[0], axis=0)
        upper_ci_dists = np.percentile(dists_data, ci_percentiles[1], axis=0)

        # Calculate slopes for EACH trial
        all_slopes = []
        run = np.diff(ks) # Calculate run once (difference between consecutive k values)
        if np.any(run <= 0):
            print("Warning: ks must be strictly increasing for slope calculation.")
            # Use index diff if ks are unevenly spaced but increasing
            run = np.ones_like(run) if np.all(run > 0) else np.diff(np.arange(len(ks)))

        max_dist_per_trial = np.max(dists_data, axis=1, keepdims=True)
        # Avoid division by zero if max distortion is zero for a trial (unlikely but possible)
        max_dist_per_trial[max_dist_per_trial == 0] = 1

        for i in range(dists_data.shape[0]):
            trial_dists = dists_data[i, :]
            rise = np.diff(trial_dists) # Difference between consecutive distortions
            # Slope = -rise/run, normalized by max distortion of that trial
            slopes_trial = (-rise / run) / max_dist_per_trial[i]
            all_slopes.append(slopes_trial)

        all_slopes_np = np.array(all_slopes)

        # Calculate median and confidence intervals for slopes
        median_slopes = np.median(all_slopes_np, axis=0)
        lower_ci_slopes = np.percentile(all_slopes_np, ci_percentiles[0], axis=0)
        upper_ci_slopes = np.percentile(all_slopes_np, ci_percentiles[1], axis=0)

        # Data to plot is the median
        plot_dists = median_dists
        plot_slopes = median_slopes

    else:
        # Single trial data
        plot_dists = dists_data.flatten() # Ensure it's 1D

        # Calculate slopes for the single trial
        run = np.diff(ks)
        if np.any(run <= 0):
            print("Warning: ks must be strictly increasing for slope calculation.")
            run = np.ones_like(run) if np.all(run > 0) else np.diff(np.arange(len(ks)))

        rise = np.diff(plot_dists)
        max_dist = np.max(plot_dists) if np.max(plot_dists) > 0 else 1
        plot_slopes = (-rise / run) / max_dist


    # Apply smoothing if requested (applied to median if multiple trials)
    if smooth is not None:
        assert len(smooth) >= 2, '`smooth` must be a list-like with at least two elements: [window, function, *args].'
        window = smooth[0]
        if len(smooth) > 2:
            # Apply function with arguments
            def func(x):
                return smooth[1](x, *smooth[2:])
        else:
            # Apply function without extra arguments
            def func(x):
                return smooth[1](x)

        # Smooth distortions (or median distortions)
        plot_dists_series = pd.Series(plot_dists)
        plot_dists = plot_dists_series.rolling(window=window, center=True, min_periods=1).apply(func, raw=True).bfill().ffill().values

        # Smooth slopes (or median slopes)
        plot_slopes_series = pd.Series(plot_slopes)
        plot_slopes = plot_slopes_series.rolling(window=window, center=True, min_periods=1).apply(func, raw=True).bfill().ffill().values


    # --- Plotting ---
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True) # Share x-axis

    # Plot Distortions
    # Normalize for plotting consistency (shift min to near 1)
    min_plot_dist = np.min(plot_dists)
    ax[0].plot(ks, plot_dists - min_plot_dist + 1, 'bx-', label='Distortion (Median)' if multiple_trials else 'Distortion')
    if multiple_trials:
        # Normalize CI based on the median's min value before plotting
        lower_ci_dists_norm = lower_ci_dists - min_plot_dist + 1
        upper_ci_dists_norm = upper_ci_dists - min_plot_dist + 1
        ax[0].fill_between(ks, lower_ci_dists_norm, upper_ci_dists_norm, color='blue', alpha=0.2, label=f'{100-(2*ci_percentiles[0]):.0f}% CI')
    ax[0].set_ylabel('Normalized Distortion')
    ax[0].set_title('Elbow Method for Optimal k')
    ax[0].grid(True, linestyle='--', alpha=0.6)
    if multiple_trials:
        ax[0].legend()

    # Plot Slopes
    ks_slopes = ks[1:] # Slopes correspond to k values starting from the second one

    # _slopes_bar_params = {'width': 0.9 * np.mean(np.diff(ks)) if len(ks)>1 else 1, # Auto-adjust width

    _slopes_bar_params = {'width': [ks[i] - ks[i-1] for i in range(1, len(ks))], 
                          'linewidth': 1, 'edgecolor': 'k'}

    if slopes_bar_params is not None:
        _slopes_bar_params.update(slopes_bar_params)

    _slopes_line_params = {'color': 'k', 'marker': 'o', 'linestyle': '-'}
    if slopes_line_params is not None:
        _slopes_line_params.update(slopes_line_params)


    if slopes_plot == 'bar':
        ax[1].bar(ks_slopes, plot_slopes, **_slopes_bar_params, label='Slope (Median)' if multiple_trials else 'Slope')
        # Optional: Add CI as error bars for bar plot (can be visually noisy)
        if multiple_trials:
            # Calculate error relative to median
            # yerr_lower = median_slopes - lower_ci_slopes
            # yerr_upper = upper_ci_slopes - median_slopes
            # ax[1].errorbar(ks_slopes, plot_slopes, yerr=[yerr_lower, yerr_upper], fmt='none', ecolor='gray', capsize=5, label=f'{100-(2*ci_percentiles[0]):.0f}% CI')
            pass # Decided against plotting slope CI on bar for clarity, but kept code commented

    elif slopes_plot == 'line':
        # If smoothing was applied via 'smooth' param, we use plot_slopes directly
        # If specific 'enhanced_roll' type smoothing was intended *only* for line plot slopes:
        # This part might need adjustment depending on what `enhanced_roll` did.
        # Assuming the generic smoothing handles it, or no smoothing if smooth=None.
        # window = _slopes_line_params.pop('window', 5) # Remove window if present
        # s = pd.Series(plot_slopes)
        # s_smooth = enhanced_roll(s, window) # Use the placeholder or your actual function
        # ax[1].plot(ks_slopes, s_smooth.values, **_slopes_line_params, label='Smoothed Slope (Median)' if multiple_trials else 'Smoothed Slope')

        # Plotting non-smoothed line slopes:
        ax[1].plot(ks_slopes, plot_slopes, **_slopes_line_params, label='Slope (Median)' if multiple_trials else 'Slope')

        if multiple_trials:
            # Plot CI band for line plot
            ax[1].fill_between(ks_slopes, lower_ci_slopes, upper_ci_slopes, color='gray', alpha=0.3, label=f'{100-(2*ci_percentiles[0]):.0f}% CI')


    ax[1].set_xlabel('Number of clusters (k)')
    ax[1].set_ylabel('Normalized Slope')
    ax[1].set_title('Rate of Change in Distortion')
    ax[1].grid(True, linestyle='--', alpha=0.6)
    ax[1].legend()
    # Set x-axis ticks explicitly to match ks
    ax[1].set_xticks(ks)
    plt.tight_layout()


    if return_slopes:
        return ax, plot_slopes
    else:
        return ax


def distortion_plot(df, ks, metric='euclidean', trials=1, return_slopes=False, **kwargs):
    '''
    Calculate and plot the distortions with varying k for clustering.
    Handles multiple trials for K-means to show median and confidence intervals.

    Figure produces two plots: (1) distortions vs k, and (2) slopes between k.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for clustering (observations as rows, features as columns).
    ks : 1D array-like
        Values for k to test; typically produced with np.arange() or np.linspace().
    metric : str, optional
        Distance metric ('euclidean', 'cityblock', etc.). Default is 'euclidean'.
    trials : int, optional
        Number of times to run K-means clustering for each k. If > 1, median and CI
        are plotted. Ignored if clustering_type is 'h'. Default is 1.
    return_slopes : bool, optional
        If True, return the calculated slopes (median if multiple trials).
    **kwargs : dict
        Additional keyword arguments passed directly to `plot_dists`.

    Returns
    -------
    dists_results : np.ndarray
        If trials=1, a 1D array of distortions.
        If trials>1, a 2D array (trials x ks) of distortions.
    ax : list
        List of 2 matplotlib Axes objects.
    '''
    ks = np.array(ks) # Ensure ks is an array

    if trials <= 1:
        # Single trial for K-means
        dists_results = get_dists(df, ks, metric=metric, clustering_type='k')
        plot_dists_out = plot_dists(dists_results, ks, return_slopes=return_slopes, **kwargs)
    else:
        # Multiple trials for K-means
        print(f"Running {trials} trials for K-means...")
        all_dists = []
        for _ in range(trials):
            # Vary random state per trial if base is provided
            dists = get_dists(df, ks, metric=metric, clustering_type='k') # Disable inner progress bar
            all_dists.append(dists)

        dists_results = np.array(all_dists) # Shape: (trials, len(ks))
        # Pass the 2D array to plot_dists
        plot_dists_out = plot_dists(dists_results, ks, return_slopes=return_slopes, **kwargs)

    if return_slopes:
        ax = plot_dists_out[0]
        slopes = plot_dists_out[1]
        return dists_results, slopes, ax
    else:
        ax = plot_dists_out
        return dists_results, ax


def get_ordering(Z, labels, return_idxs=False):
    '''
    Get the ordering of a linkage matrix. 
    [Source](https://stackoverflow.com/questions/12572436/calculate-ordering-of-dendrogram-leaves)
    
    `Z`: linkage matrix from scipy.cluster.hierarchy.linkage()
    `labels`: np.ndarray, pd.Index, or pd.Series with labels; typically
              the index of the clustered axis
    `return_idxs`: return the indices that reorder the labels instead of the
                   re-ordered labels
                   
    returns: re-ordered `labels` or indices of re-ordered labels, depending 
             on `return_idxs`
             
    '''
    
    n = len(Z) + 1
    cache = dict()
    for k in range(len(Z)):
        c1, c2 = int(Z[k][0]), int(Z[k][1])
        c1 = [c1] if c1 < n else cache.pop(c1)
        c2 = [c2] if c2 < n else cache.pop(c2)
        cache[n+k] = c1 + c2
    ordering = cache[2*len(Z)]
    if return_idxs is False:
        return np.array(labels)[ordering]
    elif return_idxs is True:
        return ordering
    else:
        raise ValueError


class DFClust():
    """
    A class used to perform hierarchical and kmeans clustering on a DataFrame.

    ...

    Attributes
    ----------
    df : DataFrame
        The DataFrame to be clustered.
    hier_params : dict, optional
        Parameters for hierarchical clustering (default is {'method':'ward', 'metric': 'euclidean', 
        'optimal_ordering':True}).
    kmeans_params : dict, optional
        Parameters for KMeans clustering (default is {'random_state':0, 'n_init':'auto'}).
    res : dict
        A dictionary to store the results of the clustering.

    Methods
    -------
    run(which, k, hier=True, kmeans=True, returnZ=True):
        Performs hierarchical and/or kmeans clustering on the DataFrame.
    get_labels(clust, order_by=None, axis=None):
        Returns the clustered index and its labels according to the algorithms that were run.
    """
    def __init__(self, df, hier_params=None, kmeans_params=None):
        """
        Constructs all the necessary attributes for the DFClust object.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to be clustered.
        hier_params : dict, optional
            Parameters for hierarchical clustering (default is {'method':'ward', 'metric': 'euclidean', 
            'optimal_ordering':True}).
        kmeans_params : dict, optional
            Parameters for KMeans clustering (default is {'random_state':0, 'n_init':'auto'}).
        """
        self.df = df
        self.res = {}

        if hier_params is None:
            self.hier_params = {'method':'ward', 'metric': 'euclidean', 'optimal_ordering':True}
        else:
            self.hier_params = hier_params
        if kmeans_params is None:
            self.kmeans_params = {'random_state':0, 'n_init':'auto'}
        else:
            self.kmeans_params = kmeans_params

    def __str__(self):
        # Define a string representation of the object
        if self.res == {}:
            return f"DFClust(df_shape={self.df.shape}, res=None)"
        else:
            keys1 = list(self.res.keys())
            keys2 = [tuple(self.res[k].keys()) for k in keys1]
            res_str = dict(zip(keys1, keys2))
        return f"DFClust(df_shape={self.df.shape}, res={res_str})"
    
    def __repr__(self):
            # Define a string representation of the object
            return self.__str__()

    def run(self, which, k, hier=True, kmeans=True, returnZ=True):
        """
        Performs hierarchical and/or kmeans clustering on the DataFrame.

        Parameters
        ----------
        which : str
            Specifies on which axis to perform clustering ('r' for rows, 'c' for columns, 'b' for both).
        k : int
            The number of clusters.
        hier : bool, optional
            Whether to perform hierarchical clustering (default is True).
        kmeans : bool, optional
            Whether to perform KMeans clustering (default is True).
        returnZ : bool, optional
            Whether to return the linkage matrix from hierarchical clustering (default is True).
        """
        
        def get_kmeans(df, k, **kwargs):
            kmeans = KMeans(n_clusters=k, **kwargs).fit(df)
            s = pd.Series(kmeans.labels_, index=df.index, name='k')
            return s
        
        def get_hier(df, k, returnZ, **kwargs):
            Z = linkage(df,  **kwargs)
            ordering = get_ordering(Z, df.index)
            s = pd.Series(fcluster(Z, k, criterion='maxclust') - 1, index=ordering, name='h')
            if returnZ:
                return Z, s
            else:
                return s
        
        def run_axis(df, _which, k, hier, kmeans, hier_params, kmeans_params):
            self.res[_which] = dict()
            if hier:
                hier_clust = get_hier(df, k, returnZ, **hier_params)
                if returnZ:
                    self.res[_which]['Z'] = hier_clust[0]
                    self.res[_which]['h'] = hier_clust[1]
                else:
                    self.res[_which]['h'] = hier_clust
            if kmeans:
                kmeans_clust = get_kmeans(df, k, **kmeans_params)
                self.res[_which]['k'] = kmeans_clust

        if which == 'r':
            df_run = self.df
            run_axis(df_run, which, k, hier, kmeans, self.hier_params, self.kmeans_params)
        elif which == 'c':
            df_run = self.df.T
            run_axis(df_run, which, k, hier, kmeans, self.hier_params, self.kmeans_params)
        elif which == 'b':
            for (axis, df_run) in ['r', 'c']:
                run_axis(df_run, axis, k, hier, kmeans, self.hier_params, self.kmeans_params)

        
    def get_labels(self, clust, order_by=None, axis=None):
        """
        Returns the clustered index and its labels according to the algorithms that were run.

        Parameters
        ----------
        clust : str
            Specifies which clustering algorithm's results to return ('k' for KMeans, 'h' for hierarchical).
        order_by : str, optional
            Specifies the order in which to return the results ('k' for increasing KMeans cluster, 'h' for increasing 
            hierarchical cluster, 'Z' for the order returned from the linkage function).
        axis : str, optional
            Specifies which axis's results to return ('r' for rows, 'c' for columns, 'b' for both).
        """

        # if order_by:
            # assert clust != order_by, 'Params `clust` and `order_by` cannot be the same.'
        if self.res == {}:
            raise ValueError('No clustering results found. Run DFClust.run() first.')
        
        if len(self.res.keys()) == 1:
            only_axis = list(self.res.keys())[0]
            if axis and axis != only_axis:
                print(f'Only one axis found, ignoring `axis` param and using {only_axis}.')
            axis = only_axis

        def get_labels_axis(axis):
            concat1 = self.res[axis][clust]
            if order_by == clust:
                s = concat1.sort_values(ascending=True)
            else:
                if order_by == 'Z':
                    concat2 = pd.Series(range(self.df.shape[0]), index=get_ordering(self.res[axis]['Z'], self.df.index), name='Z')
                else:
                    concat2 = self.res[axis][order_by]
                s = pd.concat([concat1, concat2], axis=1).sort_values(by=order_by, ascending=True)[clust]
            return s

        if order_by is None:
            if axis:
                labels_res = self.res[axis][clust]
            else:
                labels_res = dict()
                for axis in self.res.keys():
                    labels_res[axis] = self.res[axis][clust]
            return labels_res
        else:
            if axis in ['r', 'c']:
                labels_res = get_labels_axis(axis)
            elif (axis is None) or (axis == 'b'):
                labels_res = dict()
                for axis in self.res.keys():
                    labels_res[axis] = get_labels_axis(axis)
        return labels_res
    