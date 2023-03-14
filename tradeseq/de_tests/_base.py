from abc import ABC, abstractmethod
from itertools import combinations
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from tradeseq.gam import GAM


class DifferentialExpressionTest(ABC):
    """Abstract base class for a DifferntialExpressionTest."""

    def __init__(self, model: GAM):
        """Initialize WithinLineageTest class.

        Parameters
        ----------
        model
            Fitted GAM class.
        """
        self._model = model

    @abstractmethod
    def __call__(self, **kwargs):
        """Perform the DifferntialExpressionTest."""

    def _get_start_pseudotime(self) -> List[float]:
        """Get pseudotime values for the start of every lineage.

        Returns
        -------
        A list containing the minimum pseudotime for every lineage.
        """
        pseudotimes_per_lineage = self._model._get_pseudotimes_per_lineage()
        return [pseudotimes.min() for pseudotimes in pseudotimes_per_lineage]

    def _get_end_pseudotime(self) -> List[float]:
        """Get pseudotime values for the end of every lineage.

        Returns
        -------
        A list containing the maximum pseudotime for every lineage.
        """
        pseudotimes_per_lineage = self._model._get_pseudotimes_per_lineage()
        return [pseudotimes.max() for pseudotimes in pseudotimes_per_lineage]


class WithinLineageTest(DifferentialExpressionTest):
    """Abstract class for a within lineage differential expression test."""

    def _test(
        self,
        pseudotimes_a: List[np.ndarray],
        pseudotimes_b: List[np.ndarray],
        lineages: Union[List[int], np.ndarray],
        lineage_test: bool = False,
        global_test: bool = True,
        l2fc: float = 0,
    ):
        """Perform Wald tests for all genes comparing the predictions for pseudotime_a with pseudotime_b given an assigment to lineages.

        Parameters
        ----------
        pseudotimes_a
            A list of ``len(lineages)`` many np.ndarray of shape (``n_predictions``,) specifying pseudotime values per tested lineage.
        pseudotimes_b
            A list of ``len(lineages)`` many np.ndarray of shape (``n_predictions``,) specifying pseudotime values per tested lineage.
        lineages
            A np.ndarray or list of integers specifying the lineage indices for which tests should be performed.
        lineage_test
            Boolean indicating whether a test should be performed per lineage (independent of other lineages).
        global_test
            Boolean indicating whether a global_test should be performed (across all lineages).

        Returns
        -------
        A Pandas DataFrame containing the Wald statistic, the degrees of freedom, the p-value and the mean log fold change
        for each gene for each lineage (if ``lineage_test=True``) and/or globally.
        """
        result = {}
        lineage_ids = np.array(
            [
                lineage_ind
                for lineage_ind in lineages
                for n_prediction in range(len(pseudotimes_a[lineage_ind]))
            ]
        )
        pseudotimes_a = np.concatenate(pseudotimes_a)
        pseudotimes_b = np.concatenate(pseudotimes_b)

        for gene_id, gene_name in enumerate(self._model._genes):
            sigma = self._model.get_covariance(gene_id)

            pred_a = self._model.predict(
                gene_id, lineage_ids, pseudotimes_a, log_scale=True
            )
            pred_b = self._model.predict(
                gene_id, lineage_ids, pseudotimes_b, log_scale=True
            )
            pred_diff = pred_a - pred_b
            pred_fc = pred_diff.copy()
            _fold_change_cutoff(pred_diff, l2fc)
            lpmatrix_a = self._model.get_lpmatrix(gene_id, lineage_ids, pseudotimes_a)
            lpmatrix_b = self._model.get_lpmatrix(gene_id, lineage_ids, pseudotimes_b)
            lpmatrix_diff = lpmatrix_a - lpmatrix_b

            if lineage_test:
                for lineage in lineages:
                    lineage_name = self._model.lineage_names[lineage]
                    fc = np.mean(pred_fc[lineage_ids == lineage])
                    wald_stat, df, p_value = _wald_test(
                        pred_diff[lineage_ids == lineage],
                        lpmatrix_diff[lineage_ids == lineage],
                        sigma,
                    )
                    result[f"{gene_name} in lineage {lineage_name}"] = (
                        wald_stat,
                        df,
                        p_value,
                        fc,
                    )

            if global_test:
                fc = np.mean(pred_fc)
                wald_stat, df, p_value = _wald_test(pred_diff, lpmatrix_diff, sigma)
                result[f"{gene_name} globally"] = wald_stat, df, p_value, fc

        return pd.DataFrame.from_dict(
            result,
            orient="index",
            columns=[
                "wald statistic",
                "degrees of freedom",
                "p value",
                "log fold change",
            ],
        )


class BetweenLineageTest(DifferentialExpressionTest):
    """Abstract class for a between lineage differential expression test."""

    def _test(
        self,
        pseudotimes: List[np.ndarray],
        lineages: Union[List[int], np.ndarray],
        pairwise_test: bool = False,
        global_test: bool = True,
    ):
        """Perform Wald tests for all genes comparing the predictions for the given pseudotimes between the lineages.

        Parameters
        ----------
        pseudotimes
            A list of ``len(lineages)`` many np.ndarray of shape (``n_predictions``,) specifying pseudotime values per tested lineage.
        lineages
            A np.ndarray or list of integers specifying the lineage indices for which tests should be performed.
        pairwise_test
            Boolean indicating whether a test should be performed for all pairs of lineages (independent of other lineages).
        global_test
            Boolean indicating whether a global_test should be performed (across all lineages).

        Returns
        -------
        A Pandas DataFrame containing the Wald statistic, the degrees of freedom and the p-value
        for each gene for each pair of lineages (if ``pairwise_test``) and/or globally (if ``global_test``).
        """
        result = {}
        lineage_ids = np.array(
            [
                lineage_ind
                for lineage_ind in lineages
                for n_prediction in range(len(pseudotimes[lineage_ind]))
            ]
        )
        pseudotimes = np.concatenate(pseudotimes)

        for gene_id, gene_name in enumerate(self._model._genes):
            sigma = self._model.get_covariance(gene_id)

            predictions = self._model.predict(
                gene_id, lineage_ids, pseudotimes, log_scale=True
            )
            lpmatrix = self._model.get_lpmatrix(gene_id, lineage_ids, pseudotimes)

            predictions_comb = [
                predictions[lineage_ids == lineage_a]
                - predictions[lineage_ids == lineage_b]
                for (lineage_a, lineage_b) in combinations(lineages, 2)
            ]
            lpmatrices_comb = [
                lpmatrix[lineage_ids == lineage_a] - lpmatrix[lineage_ids == lineage_b]
                for (lineage_a, lineage_b) in combinations(lineages, 2)
            ]

            if pairwise_test:
                for prediction_diff, lpmatrix_diff, (lineage_a, lineage_b) in zip(
                    predictions_comb, lpmatrices_comb, combinations(lineages, 2)
                ):
                    result[
                        f"{gene_name} between lineages {self._model.lineage_names[lineage_a]} and {self._model.lineage_names[lineage_b]}"
                    ] = _wald_test(
                        prediction_diff,
                        lpmatrix_diff,
                        sigma,
                    )

            if global_test:
                pred = np.concatenate(predictions_comb)
                lpmatrix = np.concatenate(lpmatrices_comb, axis=0)
                result[f"{gene_name} globally"] = _wald_test(pred, lpmatrix, sigma)

        return pd.DataFrame.from_dict(
            result,
            orient="index",
            columns=["wald statistic", "degrees of freedom", "p value"],
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


def _fold_change_cutoff(a: np.ndarray, l2fc: float = 0):
    a[abs(a) < l2fc] = 0
