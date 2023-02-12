from abc import ABC, abstractmethod
from typing import Literal

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

    def _test(self, pseudotime_a, pseudotime_b, lineage_assignment):
        """
        Perform wald tests for all genes comparing the predictions for pseudotime_a with pseudotime_b given an assigment to lineages.

        Parameters
        ----------
        pseudotime_a
            A (``n_predictions``,) np.ndarray specifying pseudotime values.
        pseudotime_b
            A (``n_predictions``,) np.ndarray specifying pseudotime values.
        lineage_assignment
            A (``n_predictions``,) np.ndarray where each integer entry indicates the lineage index for the prediction
            point (both for pseudotime_a and pseudotime_b).

        Returns
        -------
        A pandas data frame containing the wald statistic, the degrees of freedom and the p value for each gene.
        """
        result = {}
        for gene_id, name in enumerate(self._model._genes):
            pred_a = self._model.predict(gene_id, lineage_assignment, pseudotime_a, log_scale=True)
            pred_b = self._model.predict(gene_id, lineage_assignment, pseudotime_b, log_scale=True)

            lpmatrix_a = self._model.get_lpmatrix(gene_id, lineage_assignment, pseudotime_a)
            lpmatrix_b = self._model.get_lpmatrix(gene_id, lineage_assignment, pseudotime_b)

            sigma = self._model.get_covariance(gene_id)

            result[name] = wald_test(pred_a - pred_b, lpmatrix_a - lpmatrix_b, sigma)

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
    sigma_inv = np.linalg.inv(contrast @ sigma @ contrast.T)
    wald = prediction @ sigma_inv @ prediction.T
    if wald < 0:
        wald = 0

    df = prediction.shape[0]
    pval = 1 - chi2.cdf(wald, df)
    return wald, df, pval
