from typing import List, Tuple, Union, Optional
import warnings

from conorm import tmm_norm_factors
from anndata import AnnData
from scipy.sparse import issparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tradeseq.gam import _backend


class GAM:
    """Class for fitting and working with GAMs for gene expression data.

    First, the fit method should be called to fit GAMs for (a subset of) genes.
    Then, one can make predictions with and plot the fitted GAMs.
    """

    # TODO: keys as parameters
    def __init__(
        self,
        adata: AnnData,
        n_lineages: int,
        time_key: str,
        weights_key: str,
        offset_key: Optional[str] = None,
        layer_key: Optional[str] = None,
    ):
        """Initialize GAM class.

        Parameters
        ----------
        adata
            AnnData object containing the gene counts, the cell to lineage weights, and the pseudotimes.
        n_lineages
            Number of lineages.
        time_key
            Key for pseudotime values,
            ``adata`` has to contain pseudotime values for every lineage
            in ``adata.obsm[time_key]`` or ``adata.obs[time_key]``.
        weights_key
            Key for cell to lineage weights. ``adata`` has to contain a weights object of
            shape (``adata.n_obs``, n_lineages) in ``adata.obsm[weights_key]``.
        offset_key
            Key for a cell specific offset that accounts for differences in sequencing depth. ``adata`` has to contain
            an offset object of shape (``adata.n_obs``,) in the column `òffset_key`` in ``adata.obs``.
        layer_key
            Key for the layer from which to retrieve the counts in ``adata`` If ``None``, ``adata.X`` is
            used.
        """
        self._adata: AnnData = adata
        self._n_lineages = n_lineages

        self._model: Optional[List[_backend.GAM]] = None
        self._genes = None

        self._lineage_names: Optional[List[str]] = None

        self._knots: Optional[np.ndarray] = None

        self._lineage_assignment: Optional[np.ndarray] = None
        self._offset: Optional[np.ndarray] = None  # TODO: Not sure if necessary or sufficient to just save average

        self._time_key = time_key
        self._weights_key = weights_key
        self._offset_key = offset_key
        self._layer_key = layer_key

    # TODO: change so that list of gene_ids or gene_names are accepted
    def predict(self, gene_id: int, lineage_assignment: np.ndarray, pseudotimes, log_scale: bool = False) -> np.ndarray:
        """Predict gene count for new data according to fitted GAM.

        Parameters
        ----------
        gene_id
            Index of the gene for which prediction is made.
        lineage_assignment
            A ``n_predictions`` x ``n_lineage`` np.ndarray where each row contains exactly one 1 (the assigned lineage)
            and 0 everywhere else. TODO: maybe easier to just have a list with lineage indices for every data point
        pseudotimes:
            A ``n_prediction`` x ``n_lineage`` np.ndarray containing the pseudotime values for every lineage.
            Note that only the pseudotimes of the corresponding lineage are considered.
            TODO: probably easier to just have list of pseudotime values
        log_scale:
            Should predictions be returned in log_scale (this is not log1p-scale!).

        Returns
        -------
             A np.ndarray of shape (``n_predictions``,) containing the predicted counts.
        """
        if self._model is None:
            raise RuntimeError("No GAM fitted. The fit method has to be called first.")

        # offsets are just mean offsets of fitted data
        n_predictions = lineage_assignment.shape[0]
        offsets = np.repeat(self._offset.mean(), n_predictions)

        return self._model[gene_id].predict(lineage_assignment, pseudotimes, offsets, log_scale)

    def plot(
        self,
        gene_id: int,
        lineage_id: Optional[Union[List[int], int]] = None,
        resolution: int = 200,
        knot_locations: bool = True,
        log_scale=False,
        x_label: str = "pseudotime",
        y_label: str = "gene expression",
    ):
        """Plot gene counts and fitted smoothers.

        Parameters
        ----------
        gene_id
            Index of the gene that should be plotted.
        lineage_id
            Indices of plotted lineages. Can be a list or an int if only a single lineage should be plotted.
            If None, all lineages are plotted.
        resolution
            Number of points that are used to plot the smoother.
        knot_locations
            Boolean indicating whether knot locations should be plotted as dashed vertical lines.
        log_scale
            Boolean indicating whether counts and smoothers should be plotted in log1p scale.
        x_label
            Label for x-axis.
        y_label
            label for y-axis
        """
        n_lineages = self._n_lineages
        if lineage_id is None:
            lineage_id = list(range(n_lineages))
        if isinstance(lineage_id, int):
            lineage_id = [lineage_id]

        times_fitted = []
        counts_fitted = []
        for id in lineage_id:
            cell_mask = self._lineage_assignment[:, id] == 1
            times_fitted.append(self._get_pseudotime()[cell_mask, id])
            counts_fitted.append(self._get_counts()[0][cell_mask, gene_id])

        times_pred = []
        counts_pred = []
        for id in lineage_id:
            equally_spaced = np.linspace(times_fitted[id].min(), times_fitted[id].max(), resolution)
            times_pred.append(equally_spaced)
            # create matrix with pseudotimes for every lineage (needed for prediction)
            times = np.zeros((resolution, n_lineages))
            times[:, id] = times_pred[-1]

            lineage_pred = np.zeros((resolution, n_lineages))
            lineage_pred[:, id] = 1

            counts_pred.append(self.predict(gene_id, lineage_pred, times, log_scale=False))

        for times, counts in zip(times_fitted, counts_fitted):
            if log_scale:
                counts = np.log1p(counts)
            plt.scatter(times, counts)

        for times, counts, id in zip(times_pred, counts_pred, lineage_id):
            if log_scale:
                counts = np.log1p(counts)
            plt.plot(times, counts, label=f"lineage {self._lineage_names[id]}")

        # Plot knot locations
        if knot_locations:
            y_max = max([max([pred.max() for pred in counts_pred]), max([fitted.max() for fitted in counts_fitted])])
            plt.vlines(self._knots, 0, y_max, linestyle="dashed", colors="k")
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        plt.legend()
        plt.show()

    def _get_pseudotime(self) -> np.ndarray:
        """Retrieve pseudotime from ``self._adata``.

        Returns
        --------
        Array of shape (``self._adata.n_obs``, ``n_lineages``) containing pseudotime values for all lineages.
        """
        time_key = self._time_key
        n_lineages = self._n_lineages
        if time_key in self._adata.obs.columns:
            pseudotime = self._adata.obs[time_key].values
        elif time_key in self._adata.obsm.keys():
            pseudotime = np.array(self._adata.obsm[time_key])
        else:
            raise KeyError(
                f"Invalid key {time_key} for pseudotimes."
                f"The key `{time_key}` must be either in `adata.obs` or `adata.obsm`."
            )

        if pseudotime.ndim == 1:
            return np.repeat(pseudotime[:, None], n_lineages, axis=1)

        if pseudotime.shape != (self._adata.n_obs, n_lineages):
            raise ValueError(
                "Invalid pseudotime shape.\n"
                f"Expected shape ({self._adata.n_obs}, {n_lineages}) or ({self._adata.n_obs},)\n"
                f"Actual shape: {pseudotime.shape}"
            )

        if pseudotime.max(axis=0).argmax() != 0:
            warnings.warn(
                "First lineage is not the longest lineage (i.e. the lineage with the greatest pseudotime value).",
                RuntimeWarning,
            )

        return pseudotime

    def _get_lineage(self) -> Tuple[np.ndarray, list]:
        """Retrieve cell to lineage weights from ``self._adata``.

        Returns
        -------
            Tuple of cell to lineage weights as a (``self._adata.n_obs``, ``n_lineages``) array and lineage names.
        """
        weights_key = self._weights_key
        try:
            data = self._adata.obsm[weights_key]
        except KeyError:
            raise KeyError(
                f"Invalid key {weights_key} for cell to lineage assignment."
                f"The key `{weights_key}` must be present in `adata.obsm`."
            )

        # TODO: use single dispatch
        if isinstance(data, pd.DataFrame):
            weights = data.values
            names = data.columns.astype(str).to_list()
        elif hasattr(data, "names"):
            weights = np.asarray(data)
            names = [str(c) for c in data.names]
        else:
            weights = np.asarray(data)
            names = [str(i) for i in range(data.shape[1])]

        if weights.ndim != 2 or weights.shape[0] != self._adata.n_obs:
            raise (
                f"Invalid cell weight shape.\n"
                f"Expected shape: ({self._adata.n_obs}, n_lineages).\n"
                f"Actual shape: {data.shape}."
            )

        return weights, names

    def _get_knots(self, n_knots: int) -> np.ndarray:
        """Calculate knot locations at quantiles of pseudotime values (of all lineages).

        If possible, end points of lineages are used as knots.
        Knot locations are stored in ``adata.uns[tradeSeq.knots]`` and returned.

        Parameters
        ----------
        n_knots:
            Number of knots that should be found.

        Returns
        -------
        A np.ndarray of length ``n_knots`` with the found knot locations.
        """
        pseudotimes = self._get_pseudotime()
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

        def get_closest_knot(end_point):
            return np.argmin(np.abs(knots - end_point))

        replace_ids = np.array([get_closest_knot(end_point) for end_point in end_points])
        if np.unique(replace_ids).size < np.unique(end_points).size:
            warnings.warn(
                "Impossible to place a knot at all endpoints. Increase the number of knots to avoid this issue.",
                RuntimeWarning,
            )
        knots[replace_ids] = end_points
        self._knots = knots
        return knots

    # TODO: Compare runtime with jit
    def _assign_cells_to_lineages(self) -> Tuple[np.ndarray, list]:
        """Assign every cell randomly to one lineage with probabilities based on the supplied lineage weights.

        Returns
        -------
            A ``n_cells`` x ``n_lineage`` np.ndarray where each row contains exactly one 1 (the assigned lineage)
            and 0 everywhere else
        """
        cell_weights, lineage_names = self._get_lineage()
        if not _check_cell_weights(cell_weights):
            raise ValueError(
                "Cell weights have to be non-negative and cells need to have at least one positive cell weight"
            )

        def sample_lineage(cell_weights_row):
            return np.random.multinomial(1, cell_weights_row / np.sum(cell_weights_row))

        return np.apply_along_axis(sample_lineage, 1, cell_weights), lineage_names

    def _get_counts(
        self,
        use_raw: bool = False,
    ) -> Tuple[np.ndarray, list]:
        """Retrieve gene expression counts from ``self._adata``.

        Parameters
        ----------
        use_raw
            Boolean indicating whether ``self._adata.raw`` should be used if existing.

        Returns
        -------
            A ``n_cell`` x ``n_lineage`` dense np.ndarry containing gene counts for every cell and a list containing the
            gene names.
        """
        # TODO: maybe add support for gene subsets?
        layer_key = self._layer_key
        if use_raw and self._adata.raw is None:
            use_raw = False  # TODO(warn)

        if use_raw:
            data = self._adata.raw
        else:
            data = self._adata
        genes = self._adata.var_names.astype(str).to_list()

        if layer_key is None:
            counts = data.X
        elif layer_key in self._adata.layers:
            counts = data.layers[layer_key]
        else:
            raise KeyError(f"Impossible to find counts. No layer with key {layer_key} exists.")

        # TODO(michalk): warn of too many genes
        return (counts.A if issparse(counts) else counts), genes

    def _get_offset(self):
        """Get cell-specific offset to account for differences in sequencing depth from ``self._adata``.

        Returns
        -------
            A np.ndarray of shape (``adata.n_obs``,) containing the cell-specific offsets.
        """
        if self._offset_key in self._adata.obs.columns:
            offset = self._adata.obs[self._offset_key].values
        else:
            raise KeyError(
                f"Invalid key {self._offset_key} for cell offset."
                f"The key `{self._offset_key}` must be present in `adata.obs`."
            )
        return offset

    # TODO: Parallelize
    # TODO: Add possibility to add weights
    def fit(
        self,
        genes=None,
        n_jobs: Optional[int] = None,
        family: str = "nb",
        n_knots: int = 6,
    ):
        """Fit generalized additive model for every selected gene.

        The GAMs all share the same knot locations for the spline functions. For every lineage and gene a single spline
        function is fitted. The cells are assigned to lineages randomly according to the given cell weights.

        Parameters
        ----------
        genes
            TODO
        n_jobs
            TODO
        family
            Family of probability distributions that is used for fitting the GAM. Defaults to the negative binomial.
            distributions. Can be any family available in mgcv.gam.
        n_knots
            Number of knots that are used for the splines in the GAM.
        """
        self._lineage_assignment, self._lineage_names = self._assign_cells_to_lineages()
        n_lineages = self._lineage_assignment.shape[1]
        self._knots = self._get_knots(n_knots)
        pseudotimes = self._get_pseudotime()

        use_raw = False
        counts, _ = self._get_counts(use_raw)

        if self._offset_key is None:
            self._offset = _calculate_offset(counts)
        else:
            self._offset = self._get_offset()

        right_side = "+".join([f"s(t{i}, by=l{i}, bs='cr', id=1, k=n_knots)" for i in range(1, n_lineages + 1)])
        right_side += "+ offset(offset)"
        smooth_form = "y ~ " + right_side

        backend = _backend.GAMFitting(
            pseudotimes, self._lineage_assignment, self._offset, self._knots, smooth_form, family
        )
        gams = []
        for gene_count in counts.T:
            gams.append(backend.fit(y=gene_count))
        self._model = gams


def _check_cell_weights(cell_weights: np.ndarray) -> bool:
    """Check if all cell weights are non-negative and if every cell has at least one positive cell weight.

    Parameters
    __________
    cell_weights:
        Array of shape ``n_cells`` x ``_lineages`` containing cell to lineage weights.

    Returns
    ________
        Boolean indicating whether all cell weights are non-negative and if every cell has at least one positive
        cell weight.
    """
    return (cell_weights >= 0).all() and (np.sum(cell_weights, axis=1) > 0).all()


def _calculate_offset(counts: np.ndarray) -> np.ndarray:
    """Calculate library size normalization offsets.

    To calculate the offset values TMM normalization is used.

    Parameters
    ----------
    counts: A ``n_cell`` x ``n_lineage`` np.ndarray containing gene counts for every cell

    Returns
    -------
    A np.ndarray of shape (``n_cell``,) containing an offset for each cell
    """
    norm_factors = tmm_norm_factors(counts.T)
    library_size = counts.sum(axis=1) * norm_factors.flatten()
    offset = np.log(library_size)
    if (offset == 0).any():
        # TODO: I do not really understand why this is done
        warnings.warn("Some calculated offsets are 0, offsetting these to 1.", RuntimeWarning)
        offset[offset == 0] = 0  # TODO: this seems like a obvious typo in tradeSeq
    return offset
