from abc import ABC, abstractmethod
from typing import List, Union, Literal

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

    def _test(
        self,
        pseudotimes_a: List[np.ndarray],
        pseudotimes_b: List[np.ndarray],
        lineages: Union[List[int], np.ndarray[int]],
        lineage_test: bool = False,
        global_test: bool = True,
    ):
        """
        Perform wald tests for all genes comparing the predictions for pseudotime_a with pseudotime_b given an assigment to lineages.

        Parameters
        ----------
        pseudotime_a
            A list of ``len(lineages)`` many np.ndarray of shape (``n_predictions``,) specifying pseudotime values per tested lineage.
        pseudotime_b
            A list of ``len(lineages)`` many np.ndarray of shape (``n_predictions``,) specifying pseudotime values per tested lineage.
        lineages
            A np.ndarray or list of integers specifying the lineage indices for which tests should be performed.

        Returns
        -------
        A pandas data frame containing the wald statistic, the degrees of freedom and the p value for each gene.
        """
        result = {}
        for gene_id, name in enumerate(self._model._genes):
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
                    result[f"{name} in lineage {lineage_name}"] = wald_test(
                        pred_a - pred_b, lpmatrix_a - lpmatrix_b, sigma
                    )

            if global_test:
                pred = np.concatenate(predictions)
                lpmatrix = np.concatenate(lpmatrices, axis=0)
                result[f"{name} globally"] = wald_test(pred, lpmatrix, sigma)

        return pd.DataFrame.from_dict(
            result, orient="index", columns=["wald statistic", "degrees of freedom", "p value"]
        )


def wald_test(
    prediction: np.ndarray, contrast: np.ndarray, sigma: np.ndarray, inverse: Literal["cholesky", "eigen"] = "eigen"
):
    """Perform a wald test for the null hypothesis: contrast * fitted_parameters = 0.

    Computes wald statistics: prediction (contrast sigma contrast^T)^(-1) prediction^T and the corresponding p value.

    Parameter
    ---------
    prediction
        A (`n_prediction`,) np.ndarray typically containing the difference between two predictions.
    contrast
        A (`n_prediction`, `n_params`) np.ndarray typically containing the difference between the lp matrices of the
        same predictions as above.
    sigma
        A (`n_params`, `n_params`) np.ndarray, the covariance matrix for the fitted parameters.

    Returns
    -------
    A tuple containing the wald statistic, the degrees of freedom and the p value.
    """
    sigma_inv = np.linalg.pinv(contrast @ sigma @ contrast.T)
    wald = prediction @ sigma_inv @ prediction.T
    if wald < 0:
        wald = 0

    df = prediction.shape[0]
    pval = 1 - chi2.cdf(wald, df)
    return wald, df, pval
