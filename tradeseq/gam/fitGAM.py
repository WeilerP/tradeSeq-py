from anndata import AnnData
import numpy as np
import warnings
import pandas as pd
from typing import Tuple, Optional
from scipy.sparse import issparse
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter


class GAM:
    def __init__(self, adata: AnnData):
        self._adata = adata
        self._model = None
        self._genes = None
        self._knots: Optional[np.ndarray] = None

    def predict(self):
        pass  # TODO: add method for predicting with fitted GAMs

    def _get_pseudotime(self, time_key: str, n_lineages: int) -> np.ndarray:
        """
        Retrieves pseudotime from ``self._adata``

        Parameters
        ----------
        time_key:
            Key for pseudotime values,
            ``self._adata`` has to contain pseudotime values for every lineage
            in ``adata.obsm[time_key]`` or ``adata.obs[time_key]`
        n_lineages:
            Number of lineages

        Returns
        --------
        Array of shape (``self._adata.n_obs``,``_lineages``) containing pseudotime values for all lineages
        """

        if time_key in self._adata.obs.columns:
            pseudotime = self._adata.obs[time_key].values
        elif time_key in self._adata.obsm.keys:
            pseudotime = np.array(self._adata.obsm[time_key])
        else:
            raise KeyError(f"Invalid key {time_key} for pseudotimes."
                           f"The key {time_key} must be either in adata.obs or adata.obsm")

        if pseudotime.ndim == 1:
            return np.repeat(pseudotime[:, None], n_lineages, axis=1)

        if pseudotime.shape != (self._adata.n_obs, n_lineages):
            raise ValueError("Invalid pseudotime shape.\n"
                             f"Expected shape ({self._adata.n_obs}, {n_lineages}) or ({self._adata.n_obs},)\n"
                             f"Actual shape: {pseudotime.shape}")

        return pseudotime

    def _get_lineage(self, lineage_key: str) -> Tuple[np.ndarray, list]:
        """
        Retrieves cell to lineage weights from ``self._adata``

        Parameters
        ----------
        lineage_key
            Key for cell to lineage weights,
            ``self._adata`` has to contain a weights object of shape (``self._adata.n_obs``, n_lineages)
            in ``adata.obsm[lineage_key]``

        Returns
        -------
            Tuple of cell to lineage weights as a (``self._adata.n_obs``, n_lineages) array and lineage names
        """

        try:
            data = self._adata.obsm[lineage_key]
        except KeyError:
            raise KeyError(f"Invalid key {lineage_key} for cell to lineage assignment."
                           f"The key {lineage_key} must be present in adata.obsm")

        if isinstance(data, pd.DataFrame):
            weights = data.values
            names = data.columns.astype(str).to_list()
        elif hasattr(data, "names"):
            weights = np.asarray(data)
            names = [str(c) for c in data.names]
        else:
            weights = np.asarray(data)
            names = np.array([str(i) for i in range(data.shape[1])])

        if weights.ndim != 2 or weights.shape[0] != self._adata.n_obs:
            raise (
                f"Invalid cell weight shape.\n"
                f"Expected shape: ({self._adata.n_obs}, n_lineages) or ({self._adata.n_obs},).\n"
                f"Actual shape: {data.shape}")

        return weights, names

    def _get_knots(self, time_key: str, n_lineages: int, n_knots: int) -> np.ndarray:
        """
        Calculates knot locations at quantiles of pseudotime values (of all lineages).

        If possible, end points of lineages are used as knots.
        Knot locations are stored in ``adata.uns[tradeSeq.knots]`` and returned.

        Parameters
        ----------
        time_key:
            Key for pseudotime values,
            ``self._adata`` has to contain pseudotime values for every lineage
            in ``adata.obsm[time_key]`` or ``adata.obs[time_key]``
        n_lineages:
            Number of lineages in
        n_knots:
            Number of knots that should be found

        Returns
        -------
        A np.ndarray of length ``n_knots`` with the found knot locations
        """
        pseudotimes = self._get_pseudotime(time_key, n_lineages)
        quantiles = np.linspace(0.0, 1, n_knots)
        knots = np.quantile(pseudotimes, quantiles)
        if np.unique(knots).size != n_knots:
            # duplicates in quantiles
            # try to fix it by considering only the first (and longest) lineage
            knots = np.quantile(pseudotimes[:, 0], quantiles)
            if np.unique(knots).size != n_knots:
                # there are still duplicates
                # try to fix it by replacing duplicate knots with mean of previous and next knot
                _, unique_ids = np.unique(knots, return_index=True)
                for knot_id in range(n_knots - 1):
                    if knot_id not in unique_ids:
                        knots[knot_id] = (knots[knot_id - 1] + knots[knot_id + 1]) / 2

            if np.unique(knots).size != n_knots:
                # if there are still duplicates, get equally spaced knots
                knots = np.linspace(0.0, pseudotimes.max(), n_knots)

        # try to add end points of all lineages to knots
        end_points = pseudotimes.max(axis=0)

        def find_closest(end_point):
            return np.argmin(np.abs(knots - end_point))

        replace_ids = np.array([find_closest(end_point) for end_point in end_points])
        if np.unique(replace_ids).size < np.unique(end_points).size:
            warnings.warn(
                "Impossible to place a knot at all endpoints. Increase the number of knots to avoid this issue",
                RuntimeWarning)
        knots[replace_ids] = end_points
        self._knots = knots
        return knots

    def _assign_cells_to_lineages(self, weights_key: str):
        """Assigns every cell randomly to one lineage with probabilities based on the supplied lineage weights.

        Parameters
        ----------
        adata
            AnnData object which contains lineage weights for every cell and lineage in ``adata.obsm[weights_key]``
        weights_key
            Key for weights

        Returns
        -------
            A ``n_cells`` x ``n_lineage`` np.ndarray where each row contains exactly one 1 (the assigned lineage)
            and 0 everywhere else
        """
        cell_weights, _ = self._get_lineage(weights_key)
        if not _check_cell_weights(cell_weights):
            raise ValueError(
                "Cell weights have to be non-negative and cells need to have at least one positive cell weight")

        def sample_lineage(cell_weights_row):
            return np.random.multinomial(1, cell_weights_row / np.sum(cell_weights_row))

        return np.apply_along_axis(sample_lineage, 1, cell_weights)  # TODO: compare with jit

    def _get_counts(
            self,
            layer_key: Optional[str] = None,
            use_raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves gene expression counts from ``self._adata``.

        Parameters
        ----------
        layer_key:
            Key for the layer from which to retrieve the counts in ``self._adata`` If None, ``self._adata.X`` is used.
        use_raw:
            Boolean indicating whether self._adata.raw should be used if existing.

        Returns
        -------
            A ``n_cell`` x ``n_lineage`` dense np.ndarry containing gene counts for every cell and a list containing the
            gene names.
        """
        # TODO: maybe add support for gene subsets?
        if use_raw and self._adata.raw is None:
            use_raw = False  # TODO(warn)

        if use_raw:
            data = self._adata.raw
        else:
            data = self._adata
        genes = list([f"{name}" for name in self._adata.var_names])

        if layer_key is None:
            counts = data.X
        elif layer_key in self._adata.layers:
            counts = data.layers[layer_key]
        else:
            raise KeyError(f"Impossible to find counts. No layer with key {layer_key} exists.")

        # TODO(michalk): warn of too many genes
        return (counts.A if issparse(counts) else counts), genes

    def fit(self, layer_key: str, weights_key: str, pseudotime_key: str, offset_key: str, genes,
                n_jobs: Optional[int],
                family: str = "nb", n_knots: int = 6, verbose: bool = True):

        w_sample = self._assign_cells_to_lineages(weights_key)
        n_lineages = w_sample.shape[1]
        knots = self._get_knots(pseudotime_key, n_lineages, n_knots)
        pseudotimes = self._get_pseudotime(pseudotime_key, n_lineages)

        np_cv_rules = default_converter + numpy2ri.converter
        with localconverter(np_cv_rules):
            ro.globalenv["pseudotimes"] = pseudotimes
            ro.globalenv["w_sample"] = w_sample

        # define pseudotimes t{ii} in R for every lineage
        ro.r('''
        for (ii in seq_len(ncol(pseudotimes))) {
            assign(paste0("t",ii), pseudotimes[,ii])
        }
        ''')

        # define lineage indicators l{ii} in R for every lineage
        ro.r('''
        for (ii in seq_len(ncol(pseudotimes))) {
            assign(paste0("l", ii), 1 * (w_sample[, ii] == 1))
        }
        ''')

        right_side = "+".join(
            [f"s(t{i}, by=l{i}, bs='cr', id=1, k=n_knots)" for i in range(1,n_lineages+1)])  # TODO: add offset
        smooth_form = "y ~ " + right_side

        mgcv = importr("mgcv")

        def fit_gam_for_gene(y):
            ro.globalenv["n_knots"] = n_knots
            ro.globalenv["y"] = ro.vectors.FloatVector(y)
            knots_l = [float(i) for i in knots]
            gam = mgcv.gam(ro.Formula(smooth_form), family=family, knots=knots_l)
            return gam

        use_raw = False
        counts, gene_names = self._get_counts(layer_key, use_raw)
        gams = []
        for gene_count in counts.T:
            gams.append(fit_gam_for_gene(gene_count))
        self._model = gams
        return gams


def _check_cell_weights(cell_weights: np.ndarray) -> bool:
    """
    Checks if all cell weights are non-negative and if every cell has at least one positive cell weight

    Parameters
    __________
    cell_weights:
        Array of shape ``n_cells`` x ``_lineages`` containing cell to lineage weights.

    Returns
    ________
        Boolean indicating whether all cell weights are non-negative and if every cell has at least one positive cell weight
    """
    return (cell_weights >= 0).all() and (np.sum(cell_weights, axis=1) > 0).all()













