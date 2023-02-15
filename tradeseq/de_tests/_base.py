from abc import ABC, abstractmethod
from typing import List, Union

from scipy.stats import chi2

import numpy as np
import pandas as pd

from tradeseq.gam import GAM


class WithinLineageTest(ABC):
    """Abstract base class for a within lineage differential expression test."""

    def __init__(self, model: GAM):
        """
        Initialize WithinLineageTest class.

        Parameters
        ----------
        model
            Fitted GAM class.
        """
        self._model = model

    @abstractmethod
    def __call__(self, **kwargs):
        """Perform the WithinLineageTest."""

    def _get_start_pseudotime(self) -> List[float]:
        """
        Get pseudotime values for the start of every lineage.

        Returns
        -------
        A list containing the minimum pseudotime for every lineage.
        """
        pseudotimes_per_lineage = self._model._get_pseudotimes_per_lineage()
        return [pseudotimes.min() for pseudotimes in pseudotimes_per_lineage]

    def _get_end_pseudotime(self) -> List[float]:
        """
        Get pseudotime values for the end of every lineage.

        Returns
        -------
        A list containing the maximum pseudotime for every lineage.
        """
        pseudotimes_per_lineage = self._model._get_pseudotimes_per_lineage()
        return [pseudotimes.max() for pseudotimes in pseudotimes_per_lineage]

    def _test(
        self,
        pseudotimes_a: List[np.ndarray],
        pseudotimes_b: List[np.ndarray],
        lineages: Union[List[int], np.ndarray[int]],
        lineage_test: bool = False,
        global_test: bool = True,
    ):
        """
        Perform Wald tests for all genes comparing the predictions for pseudotime_a with pseudotime_b given an assigment to lineages.

        Parameters
        ----------
        pseudotime_a
            A list of ``len(lineages)`` many np.ndarray of shape (``n_predictions``,) specifying pseudotime values per tested lineage.
        pseudotime_b
            A list of ``len(lineages)`` many np.ndarray of shape (``n_predictions``,) specifying pseudotime values per tested lineage.
        lineages
            A np.ndarray or list of integers specifying the lineage indices for which tests should be performed.
        lineage_test
            Boolean indicating whether a test should be performed per lineage (independent of other lineages).
        global_test
            Boolean indicating whether a global_test should be performed (across all lineages).

        Returns
        -------
        A Pandas DataFrame containing the Wald statistic, the degrees of freedom and the p-value for each gene.
        """
        result = {}
        for gene_id, gene_name in enumerate(self._model._genes):
            # TODO: parallelize
            predictions = []
            lpmatrices = []
            sigma = self._model.get_covariance(gene_id)

            for pseudotime_a, pseudotime_b, lineage in zip(pseudotimes_a, pseudotimes_b, lineages):
                lineage_array = np.repeat(lineage, len(pseudotime_a))

                pred_a = self._model.predict(gene_id, lineage_array, pseudotime_a, log_scale=True)
                pred_b = self._model.predict(gene_id, lineage_array, pseudotime_b, log_scale=True)

                predictions.append(pred_a - pred_b)

                lpmatrix_a = self._model.get_lpmatrix(gene_id, lineage_array, pseudotime_a)
                lpmatrix_b = self._model.get_lpmatrix(gene_id, lineage_array, pseudotime_b)

                lpmatrices.append(lpmatrix_a - lpmatrix_b)

                if lineage_test:
                    lineage_name = self._model.lineage_names[lineage]
                    result[f"{gene_name} in lineage {lineage_name}"] = _wald_test(
                        pred_a - pred_b, lpmatrix_a - lpmatrix_b, sigma
                    )

            if global_test:
                pred = np.concatenate(predictions)
                lpmatrix = np.concatenate(lpmatrices, axis=0)
                result[f"{gene_name} globally"] = _wald_test(pred, lpmatrix, sigma)

        return pd.DataFrame.from_dict(
            result, orient="index", columns=["wald statistic", "degrees of freedom", "p value"]
        )


# TODO: add different methods to compute the inverse
def _wald_test(prediction: np.ndarray, contrast: np.ndarray, sigma: np.ndarray):
    """Perform a Wald test for the null hypothesis: contrast * fitted_parameters = 0.

    Computes Wald statistics: prediction (contrast sigma contrast^T)^(-1) prediction^T and the corresponding p-value.

    Parameter
    ---------
    prediction
        A (`n_prediction`,) np.ndarray typically containing the difference between two predictions.
    contrast
        A (`n_prediction`, `n_params`) np.ndarray typically containing the difference between the linear predictor matrices of the
        same predictions as above.
    sigma
        The covariance matrix for the fitted parameters given as an (`n_params`, `n_params`) np.ndarray.

    Returns
    -------
    A tuple containing the Wald statistic, the degrees of freedom and the p value.
    """
    sigma_inv = np.linalg.pinv(contrast @ sigma @ contrast.T)
    wald = prediction @ sigma_inv @ prediction.T
    if wald < 0:
        wald = 0

    df = prediction.shape[0]
    pval = 1 - chi2.cdf(wald, df)
    return wald, df, pval
