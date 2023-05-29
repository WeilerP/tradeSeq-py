import warnings
from typing import List, Optional, Tuple, Union

from conorm import tmm_norm_factors

import numpy as np
import pandas as pd
from scipy.sparse import issparse

import matplotlib.pyplot as plt
import seaborn as sns

from anndata import AnnData

from tradeseq.gam import _backend


class GAM:
    """Class for fitting and working with GAMs for gene expression data.

    First, the fit method should be called to fit GAMs for (a subset of) genes.
    Then, one can make predictions with and plot the fitted GAMs.
    """

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
        self._genes: List[str] = None

        self.lineage_names: Optional[List[str]] = None

        self._knots: Optional[np.ndarray] = None

        self._lineage_assignment: Optional[np.ndarray] = None
        self._offset: Optional[
            np.ndarray
        ] = None  # TODO: Not sure if necessary or sufficient to just save average

        self._time_key = time_key
        self._weights_key = weights_key
        self._offset_key = offset_key
        self._layer_key = layer_key

    # TODO: change so that list of gene_ids or gene_names are accepted
    def predict(
        self,
        var_id: int,
        lineage_assignment: np.ndarray,
        pseudotimes: np.ndarray,
        log_scale: bool = False,
    ) -> np.ndarray:
        """Predict gene count for new data according to fitted GAM.

        Parameters
        ----------
        var_id
            Index of the gene for which prediction is made.
        lineage_assignment
            A (``n_predictions``,) np.ndarray where each integer entry indicates the lineage index for the prediction point.
        pseudotimes
            A (``n_predictions``,) np.ndarray where each entry is the pseudotime value for the prediction point.
        log_scale
            Should predictions be returned in log_scale (this is not log1p-scale!).

        Returns
        -------
        An np.ndarray of shape (``n_predictions``,) containing the predicted counts.
        """
        self.check_is_fitted()

        if (
            lineage_assignment.shape != pseudotimes.shape
            or lineage_assignment.ndim != 1
        ):
            raise ValueError(
                "The arguments lineage_assignment and pseudotimes should have the same length and have to "
                "be one dimensional."
            )

        n_predictions = lineage_assignment.shape[0]

        pseudotimes = np.repeat(pseudotimes[:, np.newaxis], self._n_lineages, axis=1)
        lineage_indicator = _indices_to_indicator_matrix(
            lineage_assignment, self._n_lineages
        )

        # offsets are just mean offsets of fitted data
        offsets = np.repeat(self._offset.mean(), n_predictions)

        if log_scale:
            return_type = "link"
        else:
            return_type = "response"

        return self._model[var_id].predict(
            lineage_indicator, pseudotimes, offsets, return_type
        )

    def get_lpmatrix(
        self, var_id: int, lineage_assignment: np.ndarray, pseudotimes: np.ndarray
    ) -> np.ndarray:
        """Return linear predictor matrix of the GAM for the given gene with the given parameters.

        Parameters
        ----------
        var_id
            Index of the gene for which the lpmatrix is returned.
        lineage_assignment
            A (``n_predictions``,) np.ndarray where each integer entry indicates the lineage index for the prediction point.
        pseudotimes
            A (``n_predictions``,) np.ndarray where each entry is the pseudotime value for the prediction point.

        Returns
        -------
        A two dimensional np.ndarray, the linear predictor matrix.
        """
        self.check_is_fitted()

        if (
            lineage_assignment.shape != pseudotimes.shape
            or lineage_assignment.ndim != 1
        ):
            raise ValueError(
                "The arguments lineage_assignment and pseudotimes should have the same length and have to "
                "be one dimensional."
            )

        n_predictions = lineage_assignment.shape[0]

        pseudotimes = np.repeat(pseudotimes[:, np.newaxis], self._n_lineages, axis=1)
        lineage_indicator = _indices_to_indicator_matrix(
            lineage_assignment, self._n_lineages
        )

        # offsets are just mean offsets of fitted data
        offsets = np.repeat(self._offset.mean(), n_predictions)

        return self._model[var_id].predict(
            lineage_indicator, pseudotimes, offsets, "lpmatrix"
        )

    def get_covariance(self, var_id: int) -> np.ndarray:
        """Return covariance matrix of the parameters fitted for the GAM for the given gene.

        Parameters
        ----------
        var_id
            Index of the gene for which the covariance matrix of the parameters of the GAM are returned.

        Returns
        -------
        A (``n_parameters``,``n_parameters``) np.ndarray, the covariance matrix.
        """
        self.check_is_fitted()
        self._model[var_id].check_fitted()

        return self._model[var_id].covariance_matrix

    def get_aic(self) -> List[float]:
        """Get Akaike information criterion (AIC) for each fitted GAM.

        Returns
        -------
        List of AICs: For GAMs that could not be fitted NaN is returned.
        """
        self.check_is_fitted()

        return [np.nan if not model.fitted else model.aic for model in self._model]

    def get_fitted_indices(self) -> List[int]:
        """Find indices of genes for which fitting of the corresponding GAM worked.

        Returns
        -------
        List of indices of genes for which fitting worked.
        """
        self.check_is_fitted()
        return [ind for ind, gam in enumerate(self._model) if gam.fitted]

    def plot(
        self,
        var_id: int,
        lineage_id: Optional[Union[List[int], int]] = None,
        resolution: int = 200,
        knot_locations: bool = True,
        log_scale: bool = False,
        sample: float = 1,
        x_label: str = "pseudotime",
        y_label: str = "gene expression",
        **kwargs,
    ):
        """Plot gene counts and fitted smoothers.

        Parameters
        ----------
        var_id
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
        sample
            Float between 0 (no observations) and 1 (all observations) determining the fraction of observations that should be plotted.
            Smaller values can speed up the plotting process.
        x_label
            Label for x-axis.
        y_label
            Label for y-axis.
        kwargs
            Additional arguments passed to the pyplot scatter function
        """
        n_lineages = self._n_lineages
        if lineage_id is None:
            lineage_id = list(range(n_lineages))
        if isinstance(lineage_id, int):
            lineage_id = [lineage_id]

        times_fitted = []
        counts_fitted = []
        for id in lineage_id:
            obs_mask = self._lineage_assignment[:, id] == 1
            times_fitted.append(self._get_pseudotime()[obs_mask, id])
            counts_fitted.append(self._get_counts()[0][obs_mask, var_id])

        times_pred = []
        counts_pred = []
        for id, time_fitted in zip(lineage_id, times_fitted):
            equally_spaced = np.linspace(
                time_fitted.min(), time_fitted.max(), resolution
            )
            times_pred.append(equally_spaced)

            lineage_pred = (
                np.zeros(resolution, dtype=int) + id
            )  # assign every prediction point to lineage with lineage id: id

            counts_pred.append(
                self.predict(var_id, lineage_pred, equally_spaced, log_scale=False)
            )

        for times, counts in zip(times_fitted, counts_fitted):
            if log_scale:
                counts = np.log1p(counts)
            obs_mask = np.random.choice(
                times.shape[0], size=int(times.shape[0] * sample), replace=False
            )

            if "s" not in kwargs:
                # set default marker size to 5
                kwargs["s"] = 5
            plt.scatter(times[obs_mask], counts[obs_mask], **kwargs)

        for times, counts, id in zip(times_pred, counts_pred, lineage_id):
            if log_scale:
                counts = np.log1p(counts)
            plt.plot(times, counts, label=f"lineage {self.lineage_names[id]}")

        # Plot knot locations
        if knot_locations:
            y_max = max(
                [
                    max([pred.max() for pred in counts_pred]),
                    max([fitted.max() for fitted in counts_fitted]),
                ]
            )
            if log_scale:
                y_max = np.log1p(y_max)
            plt.vlines(
                self._knots, 0, y_max, linestyle="dashed", colors="k", linewidth=0.5
            )
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        plt.legend()
        plt.show()

    def check_is_fitted(self):
        """Check whether GAMs have already been fitted. If not raises RunTimeError."""
        if self._model is None:
            raise RuntimeError("No GAM fitted. The fit method has to be called first.")

    def _get_pseudotime(self) -> np.ndarray:
        """Retrieve pseudotime from ``self._adata``.

        Returns
        -------
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

    def _get_pseudotimes_per_lineage(self) -> List[np.ndarray]:
        """Get the pseudotime values per lineage.

        Returns
        -------
        A list with ``n_lineage`` many elements: each a np.ndarray with the pseudotime values of cells assigned to this
        lineage.
        """
        pseudotimes = self._get_pseudotime()
        # only consider pseudotimes of the lineage the cell is assigned to
        lineage_pseudotimes = [
            pseudotimes[
                self._lineage_assignment[:, lineage_id].astype(bool), lineage_id
            ]
            for lineage_id in range(self._n_lineages)
        ]
        return lineage_pseudotimes

    def _get_knots(self, n_knots: int) -> np.ndarray:
        """Calculate knot locations at quantiles of pseudotime values (of all lineages).

        If possible, end points of lineages are used as knots.
        Knot locations are returned and stored in ``self._knots``.

        Parameters
        ----------
        n_knots:
            Number of knots that should be found.

        Returns
        -------
        A np.ndarray of length ``n_knots`` with the found knot locations.
        """
        pseudotimes = self._get_pseudotime()
        lineage_pseudotimes = self._get_pseudotimes_per_lineage()
        quantiles = np.linspace(0.0, 1, n_knots)
        knots = np.quantile(np.concatenate(lineage_pseudotimes), quantiles)
        if np.unique(knots).size != n_knots:
            # duplicates in quantiles
            # try to fix it by considering only the first (and longest) lineage
            knots = np.quantile(lineage_pseudotimes[0], quantiles)
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
        end_points = [times.max() for times in lineage_pseudotimes]

        def get_closest_knot(end_point):
            return np.argmin(np.abs(knots - end_point))

        replace_ids = np.array(
            [get_closest_knot(end_point) for end_point in end_points]
        )
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
            A ``n_obs`` x ``n_lineage`` np.ndarray where each row contains exactly one 1 (the assigned lineage)
            and 0 everywhere else and a list of lineage names.
        """
        obs_weights, lineage_names = self._get_lineage()
        if not _check_obs_weights(obs_weights):
            raise ValueError(
                "Cell weights have to be non-negative and cells need to have at least one positive cell weight"
            )

        def sample_lineage(obs_weights_row):
            return np.random.multinomial(1, obs_weights_row / np.sum(obs_weights_row))

        lineage_assignment = np.apply_along_axis(sample_lineage, 1, obs_weights)

        if any(lineage_assignment.sum(axis=0) == 0):
            lineage_name = np.array(lineage_names)[lineage_assignment.sum(axis=0) == 0][
                0
            ]
            raise RuntimeError(
                f"No cell was randomly assigned to lineage {lineage_name}. Delete this lineage, "
                f"increase the weights for this lineage or just try again."
            )
        return lineage_assignment, lineage_names

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
            raise KeyError(
                f"Impossible to find counts. No layer with key {layer_key} exists."
            )

        # TODO(michalk): warn of too many genes
        return (counts.A if issparse(counts) else counts), genes

    def _get_offset(self):
        """Get cell-specific offset to account for differences in sequencing depth from ``self._adata``.

        Returns
        -------
        An np.ndarray of shape (``adata.n_obs``,) containing the cell-specific offsets.
        """
        if self._offset_key in self._adata.obs.columns:
            offset = self._adata.obs[self._offset_key].values
        else:
            raise KeyError(
                f"Invalid key {self._offset_key} for cell offset."
                f"The key `{self._offset_key}` must be present in `adata.obs`."
            )
        return offset

    # TODO: Add possibility to add weights
    def fit(self, family: str = "nb", n_knots: int = 6, n_jobs: int = 1):
        """Fit generalized additive model for every selected gene.

        The GAMs all share the same knot locations for the spline functions. For every lineage and gene a single spline
        function is fitted. The cells are assigned to lineages randomly according to the given cell weights.

        Parameters
        ----------
        family
            Family of probability distributions that is used for fitting the GAM. Defaults to the negative binomial.
            distributions. Can be any family available in mgcv.gam.
        n_knots
            Number of knots that are used for the splines in the GAM.
        n_jobs
            Number of jobs that are used for fitting. If n_jobs > 2, the R library biocParallel is used for fitting the
            GAMs in parallel.
        """
        self._lineage_assignment, self.lineage_names = self._assign_cells_to_lineages()
        n_lineages = self._lineage_assignment.shape[1]
        self._knots = self._get_knots(n_knots)
        pseudotimes = self._get_pseudotime()

        use_raw = False
        counts, self._genes = self._get_counts(use_raw)

        if self._offset_key is None:
            self._offset = _calculate_offset(counts)
        else:
            self._offset = self._get_offset()

        right_side = "+".join(
            [
                f"s(t{i}, by=l{i}, bs='cr', id=1, k=n_knots)"
                for i in range(1, n_lineages + 1)
            ]
        )
        right_side += "+ offset(offset)"
        smooth_form = "y ~ " + right_side

        self._model = _backend.fit(
            counts,
            pseudotimes,
            self._lineage_assignment,
            self._offset,
            self._knots,
            smooth_form,
            family,
            n_jobs,
        )

    def evaluate_n_knots(
        self,
        n_knots_options: List[int],
        family: str = "nb",
        n_vars: int = 500,
        n_jobs: int = 1,
        plot: bool = True,
    ) -> pd.DataFrame:
        """Evaluate different choices for number of knots.

        Parameters
        ----------
        n_knots_options
            List of different options for number of knots (usual choices for are between 3 and 10).
        family
            Family of probability distributions that is used for fitting the GAM. Defaults to the negative binomial
            distributions. Can be any family available in mgcv.gam.
        n_vars
            Number of randomly sampled genes that are used for the evaluation.
        n_jobs
            Number of jobs that are used for fitting. If n_jobs > 2, the R library biocParallel is used for fitting the
            GAMs in parallel.
        plot
            Boolean indicating whether plots evaluating the different choices for number of knots should be shown.

        Returns
        -------
        Pandas DataFrame containing AIC of the sampled genes for the different choices for n_knots and the mean AIC,
        the relative mean AIC and the number of knots that have the optimal AIC for this value of n_knots.
        """
        if any(n_knots < 3 for n_knots in n_knots_options):
            raise RuntimeError(
                "Cannot fit with fewer than 3 knots, please increase the number of knots."
            )

        aic = []
        var_ind_sample = np.random.randint(0, self._adata.n_vars, size=(n_vars,))
        gam = GAM(
            self._adata[:, var_ind_sample],
            self._n_lineages,
            self._time_key,
            self._weights_key,
            self._offset_key,
            self._layer_key,
        )

        for n_knots in n_knots_options:
            gam.fit(family, n_knots, n_jobs)
            aic.append(gam.get_aic())

        var_names = self._adata.var_names[var_ind_sample]
        result = pd.DataFrame(aic, index=n_knots_options, columns=var_names)
        result["Number of knots"] = n_knots_options
        result["Mean AIC"] = result[var_names].mean(axis=1)
        result["Mean Relative AIC"] = (
            result[var_names] / result[var_names].iloc[0]
        ).mean(axis=1)
        result["Number of Genes with optimal n_knots"] = (
            result[var_names] == result[var_names].min(axis=0)
        ).sum(axis=1)

        if plot:
            fig, axs = plt.subplots(ncols=4)

            sns.boxplot(
                data=(result[var_names] - result[var_names].mean(axis=0)).T, ax=axs[0]
            )
            axs[0].set_xlabel("Number of knots")
            axs[0].set_ylabel("Deviation from gene-wise average AIC")

            sns.scatterplot(data=result, x="Number of knots", y="Mean AIC", ax=axs[1])
            sns.lineplot(data=result, x="Number of knots", y="Mean AIC", ax=axs[1])

            sns.scatterplot(
                data=result, x="Number of knots", y="Mean Relative AIC", ax=axs[2]
            )
            sns.lineplot(
                data=result, x="Number of knots", y="Mean Relative AIC", ax=axs[2]
            )

            sns.scatterplot(
                data=result,
                x="Number of knots",
                y="Number of Genes with optimal n_knots",
                ax=axs[3],
            )
            sns.lineplot(
                data=result,
                x="Number of knots",
                y="Number of Genes with optimal n_knots",
                ax=axs[3],
            )

            fig.tight_layout(pad=3.0)
            plt.show()

        return result


def _indices_to_indicator_matrix(indices: np.ndarray, n_indices: int):
    """Compute indicator matrice from indices.

    Parameter
    ---------
    indices:
        One-dimensional np.ndarray of indices (assumed to be in [0,``n_indice``[ ).
    n_indices:
        Number of indices (maximum index value +1)

    Returns
    -------
    A (``len(indices)``, ``n_indices``) indicator matrix.
    """
    return (indices.reshape(-1, 1) == list(range(n_indices))).astype(int)


def _check_obs_weights(obs_weights: np.ndarray) -> bool:
    """Check if all cell weights are non-negative and if every cell has at least one positive cell weight.

    Parameters
    ----------
    __________
    obs_weights:
        Array of shape ``n_obs`` x ``n_lineages`` containing cell to lineage weights.

    Returns
    -------
    ________
        Boolean indicating whether all cell weights are non-negative and if every cell has at least one positive
        cell weight.
    """
    return (obs_weights >= 0).all() and (np.sum(obs_weights, axis=1) > 0).all()


def _calculate_offset(counts: np.ndarray) -> np.ndarray:
    """Calculate library size normalization offsets.

    To calculate the offset values TMM normalization is used.

    Parameters
    ----------
    counts
        A ``n_cell`` x ``n_lineage`` np.ndarray containing gene counts for every cell

    Returns
    -------
    An np.ndarray of shape (``n_cell``,) containing an offset for each cell.
    """
    norm_factors = tmm_norm_factors(counts.T)
    library_size = counts.sum(axis=1) * norm_factors.flatten()
    offset = np.log(library_size)
    if (offset == 0).any():
        # TODO: I do not really understand why this is done
        warnings.warn(
            "Some calculated offsets are 0, offsetting these to 1.", RuntimeWarning
        )
        offset[offset == 0] = 0  # TODO: this seems like a obvious typo in tradeSeq
    if np.isnan(offset).any():
        warnings.warn(
            "Some calculated offsets are NaN, offsetting these to 1.", RuntimeWarning
        )
        offset[np.isnan(offset)] = 1
    return offset
