import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# -----------------------------
# Utilities
# -----------------------------
def _corr_to_dist(corr: pd.DataFrame) -> np.ndarray:
    """
    Convert correlation matrix to HRP distance matrix:
      d_ij = sqrt(0.5 * (1 - corr_ij))
    Returns a square numpy array with zeros on the diagonal.
    """
    c = np.clip(corr.values.astype(float), -1.0, 1.0)
    d = np.sqrt(0.5 * (1.0 - c))
    np.fill_diagonal(d, 0.0)
    return d

def _cluster_order_from_children(children, n_leaves) -> list:
    """
    Reconstruct dendrogram leaf order (seriation) from AgglomerativeClustering.children_.
    Leaves are 0..n_leaves-1; internal nodes are n_leaves..(2*n_leaves-2).
    We do a deterministic left->right depth-first traversal.
    """
    # Build a dict: node_id -> (left_child, right_child)
    tree = {n_leaves + i: (children[i, 0], children[i, 1]) for i in range(children.shape[0])}

    def _dfs(node):
        if node < n_leaves:  # leaf
            return [node]
        left, right = tree[node]
        return _dfs(left) + _dfs(right)

    root = n_leaves + children.shape[0] - 1
    return _dfs(root)

def _get_cluster_var(cov: pd.DataFrame, tickers: list) -> float:
    """
    HRP cluster variance using inverse-variance weights within cluster.
    """
    if len(tickers) == 1:
        # Single asset variance
        return float(cov.loc[tickers[0], tickers[0]])
    sub = cov.loc[tickers, tickers].values
    ivp = 1.0 / np.diag(sub)
    w = ivp / ivp.sum()
    return float(w @ sub @ w)

# -----------------------------
# Main HRP using scikit-learn clustering
# -----------------------------
def get_hrp_allocation(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """
    HRP weights via scikit-learn AgglomerativeClustering (single linkage, precomputed distances).
    Args:
        cov  : pd.DataFrame covariance matrix indexed by tickers
        corr : pd.DataFrame correlation matrix indexed by tickers
    Returns:
        pd.Series of weights indexed by tickers, summing to 1.0
    """
    tickers = list(cov.index)
    n = len(tickers)
    if n != corr.shape[0]:
        raise ValueError("cov and corr must have the same tickers/order")

    # 1) Distances from correlations (same as SciPy version)
    dist = _corr_to_dist(corr)

    # 2) Hierarchical clustering with sklearn (single linkage, precomputed metric)
    #    Use 'metric' for newer sklearn; fall back to 'affinity' for backward compatibility.
    try:
        model = AgglomerativeClustering(
            metric='precomputed', linkage='single',
            distance_threshold=0.0, n_clusters=None
        )
    except TypeError:
        # Older sklearn versions use 'affinity' instead of 'metric'
        model = AgglomerativeClustering(
            affinity='precomputed', linkage='single',
            distance_threshold=0.0, n_clusters=None
        )
    model.fit(dist)

    # 3) Quasi-diagonalization: derive leaf order from children_
    order = _cluster_order_from_children(model.children_, n_leaves=n)
    sorted_tickers = [tickers[i] for i in order]

    # 4) Recursive bisection: allocate by cluster variance (identical to standard HRP)
    weights = pd.Series(1.0, index=sorted_tickers, dtype=float)

    clusters = [sorted_tickers]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            cluster_left = cluster[:split]
            cluster_right = cluster[split:]

            var_left = _get_cluster_var(cov, cluster_left)
            var_right = _get_cluster_var(cov, cluster_right)

            # Allocation: the lower-variance cluster gets higher weight
            alpha = 1.0 - var_left / (var_left + var_right)
            weights.loc[cluster_left] *= alpha
            weights.loc[cluster_right] *= (1.0 - alpha)

            next_clusters.extend([cluster_left, cluster_right])
        clusters = next_clusters

    # 5) Normalize to 1.0 and return sorted by original ticker order (if you prefer)
    weights = weights.reindex(tickers)  # back to original order expected elsewhere
    weights /= weights.sum()
    return weights
