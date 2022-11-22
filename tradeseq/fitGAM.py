from anndata import AnnData
import numpy as np
import warnings
import pandas as pd
from typing import Any, Sized, Tuple, Union, Literal, Optional, Sequence, TYPE_CHECKING
from scipy.sparse import issparse


def _get_pseudotime(adata: AnnData, key: str, n_lineages: int) -> np.ndarray:
    attrs = ["obsm", "obs"] if n_lineages > 1 else ["obs", "obsm"]
    for attr in attrs:
        try:
            pseudotime = np.asarray(getattr(adata, attr)[key])
            if pseudotime.ndim == 1:
                return np.repeat(pseudotime[:, None], n_lineages, axis=1)
            if pseudotime.shape != (adata.n_obs, n_lineages):
                raise ValueError("TODO: invalid pseudotime/lineage shape.")
            return pseudotime
        except KeyError:
            pass

    raise KeyError("Unable to find pseudotime")


def _get_lineage(adata: AnnData, key: str) -> Tuple[np.ndarray, np.ndarray]:

    try:
        data = adata.obsm[key]
    except KeyError:
        raise KeyError("Invalid key for cell to lineage assignment")

    if isinstance(data, pd.DataFrame):
        data, names =  np.asarray(data), np.array([str(c) for c in data.columns])
    elif hasattr(data, "names"):
        data, names = np.asarray(data), np.array([str(c) for c in data.names])
    else:
        data = np.asarray(data)
        names = np.array([str(i) for i in range(data.shape[1])])

    if data.ndim != 2 or data.shape[0] != adata.n_obs:
        raise("Invalid cell weight shape (has to be n_obs x n_lineages")

    return data, names



def find_knots(adata: AnnData, key: str, n_lineages: int, n_knots: int) -> np.ndarray:
    """
    Calculates knot locations at quantiles of pseudotime values (of all lineages).
    If possible, end points of lineages are used as knots.
    Knot locations are stored in adata.uns[tradeSeq.knots] and returned.

    Parameters
    ----------
    adata: AnnData
        AnnData object which contains pseudotime values for every lineage in adata.obsm[key] or adata.obs[key]
    key: str
        key for pseudotime values
    n_lineages: int
        number of lineages in adata
    n_knots: int
        number of knots that should be found

    Returns
    --------
    np.ndarray
        A np.ndarray of length n_knots with the found knot locations
    """
    pseudotimes = _get_pseudotime(adata, key, n_lineages)
    n_pseudotimes = pseudotimes.size
    quantiles = np.linspace(0.0, 1, n_knots)
    knots = np.quantile(pseudotimes, quantiles)
    if np.unique(knots).size != n_knots:
        # duplicates in quantiles
        # try to fix it by considering only the first (and longest) lineage
        knots = np.quantile(pseudotimes[:, 0], quantiles)
        if np.unique(knots).size != n_knots:
            _, unique_ind = np.unique(knots, return_index=True)
            for i in range(n_knots):
                if i not in unique_ind and i != n_knots -1:
                    knots[i] = (knots[i-1] + knots[i+1])/2

        if np.unique(knots).size != n_knots:
            # if there are still duplicates, get equally spaced knots
            knots = np.linspace(0.0, np.amax(pseudotimes), n_knots)

    # try to add end points of all lineages to knots
    end_points = np.amax(pseudotimes, axis=0)

    def find_closest(end_point):
        return np.argmin(np.abs(knots - end_point))

    replace_ids = np.array([find_closest(end_point) for end_point in end_points])
    if np.unique(replace_ids).size < np.unique(end_points).size:
        warnings.warn("Impossible to place a knot at all endpoints. Increase the number of knots to avoid this issue",
                      RuntimeWarning)
    knots[replace_ids] = end_points
    adata.uns["tradeSeq"]["knots"] = knots
    return knots


def assign_cells(adata: AnnData, weights_key: str):
    """
        Assigns every cell randomly to one lineage with probabilities based on the supplied lineage weights.

        Parameters
        ----------
        adata: AnnData
            AnnData object which contains lineage weights for every cell and lineage in adata.obsm[weights_key]
        weights_key: str
            key for weights

        Returns
        --------
        np.ndarray
            A n_cells x n_lineage np.ndarray where each row contains exactly one 1 (the assigned lineage)
            and 0 everywhere else
        """
    cell_weights, lineage_names = _get_lineage(adata, weights_key)
    if (cell_weights < 0).any() or (np.sum(cell_weights, axis=1) == 0).any():
        raise ValueError("Cell weights have to be non-negative and cells need to have at least one positive cell weight")

    def sample_lineage(cell_weights_row):
        return np.random.multinomial(1, cell_weights_row/np.sum(cell_weights_row))

    return np.apply_along_axis(sample_lineage, 1, cell_weights)







