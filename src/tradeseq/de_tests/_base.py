from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.linalg.lapack import dtrtri
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
        l2fc
            Log2 fold change cut off.

        Returns
        -------
        A (multi-index) Pandas DataFrame containing the Wald statistic, the degrees of freedom, the p-value and the mean log fold change
        for each gene for each lineage (if ``lineage_test=True``) and/or globally.
        """
        result = defaultdict(dict)
        lineage_ids = np.array(
            [
                lineage_ind
                for lineage_ind in lineages
                for n_prediction in range(len(pseudotimes_a[lineage_ind]))
            ]
        )
        pseudotimes_a = np.concatenate(pseudotimes_a)
        pseudotimes_b = np.concatenate(pseudotimes_b)

        for var_id in self._model.get_fitted_indices():
            var_name = self._model._genes[var_id]
            sigma = self._model.get_covariance(var_id)

            pred_a = self._model.predict(
                var_id, lineage_ids, pseudotimes_a, log_scale=True
            )
            pred_b = self._model.predict(
                var_id, lineage_ids, pseudotimes_b, log_scale=True
            )
            pred_diff = pred_a - pred_b
            pred_fold_change = pred_diff.copy() * np.log2(np.e)  # change basis to log2
            _fold_change_cutoff(pred_diff, l2fc)

            lpmatrix_a = self._model.get_lpmatrix(var_id, lineage_ids, pseudotimes_a)
            lpmatrix_b = self._model.get_lpmatrix(var_id, lineage_ids, pseudotimes_b)
            lpmatrix_diff = lpmatrix_a - lpmatrix_b

            if lineage_test:
                for lineage in lineages:
                    lineage_name = self._model.lineage_names[lineage]
                    mean_fold_change = np.mean(pred_fold_change[lineage_ids == lineage])
                    wald_stat, df, p_value = _wald_test(
                        pred_diff[lineage_ids == lineage],
                        lpmatrix_diff[lineage_ids == lineage],
                        sigma,
                    )
                    result[lineage_name][var_name] = (
                        wald_stat,
                        df,
                        p_value,
                        mean_fold_change,
                    )

            if global_test:
                global_fold_change = np.mean(pred_fold_change)
                wald_stat, df, p_value = _wald_test(pred_diff, lpmatrix_diff, sigma)
                result["globally"][var_name] = (
                    wald_stat,
                    df,
                    p_value,
                    global_fold_change,
                )

        return _create_multi_index_data_frame(
            result,
            ["wald statistic", "degrees of freedom", "p value", "log fold change"],
        )


class BetweenLineageTest(DifferentialExpressionTest):
    """Abstract class for a between lineage differential expression test."""

    def _test(
        self,
        pseudotimes: List[np.ndarray],
        lineages: Union[List[int], np.ndarray],
        pairwise_test: bool = False,
        global_test: bool = True,
        l2fc: float = 0,
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
        l2fc
            Log2 fold change cut off.

        Returns
        -------
        A (multi-index) Pandas DataFrame containing the Wald statistic, the degrees of freedom, the p-value and the mean log fold change
        for each gene for each pair of lineages (if ``pairwise_test``) and/or globally (if ``global_test``).
        """
        result = defaultdict(dict)
        lineage_ids = np.array(
            [
                lineage_ind
                for lineage_ind in lineages
                for n_prediction in range(len(pseudotimes[lineage_ind]))
            ]
        )
        pseudotimes = np.concatenate(pseudotimes)

        for var_id in self._model.get_fitted_indices():
            var_name = self._model._genes[var_id]
            sigma = self._model.get_covariance(var_id)

            predictions = self._model.predict(
                var_id, lineage_ids, pseudotimes, log_scale=True
            )
            lpmatrix = self._model.get_lpmatrix(var_id, lineage_ids, pseudotimes)

            predictions_comb = [
                predictions[lineage_ids == lineage_a]
                - predictions[lineage_ids == lineage_b]
                for (lineage_a, lineage_b) in combinations(lineages, 2)
            ]

            fold_changes = [
                np.mean(pred) * np.log2(np.e) for pred in predictions_comb
            ]  # change basis to log2 fold changes

            # apply fold change cut off
            for pred in predictions_comb:
                _fold_change_cutoff(pred, l2fc)

            lpmatrices_comb = [
                lpmatrix[lineage_ids == lineage_a] - lpmatrix[lineage_ids == lineage_b]
                for (lineage_a, lineage_b) in combinations(lineages, 2)
            ]

            if pairwise_test:
                for (
                    prediction_diff,
                    lpmatrix_diff,
                    (lineage_a, lineage_b),
                    fold_change,
                ) in zip(
                    predictions_comb,
                    lpmatrices_comb,
                    combinations(lineages, 2),
                    fold_changes,
                ):
                    wald_stat, df, p_value = _wald_test(
                        prediction_diff, lpmatrix_diff, sigma
                    )
                    result[
                        f"between {self._model.lineage_names[lineage_a]} and {self._model.lineage_names[lineage_b]}"
                    ][var_name] = (wald_stat, df, p_value, fold_change)

            if global_test:
                pred = np.concatenate(predictions_comb)
                lpmatrix = np.concatenate(lpmatrices_comb, axis=0)
                wald_stat, df, p_value = _wald_test(pred, lpmatrix, sigma)
                global_fold_change = np.mean(fold_changes)
                result["globally"][var_name] = (
                    wald_stat,
                    df,
                    p_value,
                    global_fold_change,
                )

        return _create_multi_index_data_frame(
            result,
            ["wald statistic", "degrees of freedom", "p value", "log fold change"],
        )


def _create_multi_index_data_frame(df_dict: dict, columns):
    result_dfs = {}
    for test_type in df_dict.keys():
        result_dfs[test_type] = pd.DataFrame.from_dict(
            df_dict[test_type], orient="index", columns=columns
        )
    return pd.concat(result_dfs.values(), axis=1, keys=result_dfs.keys())


def _wald_test(
    prediction: np.ndarray,
    contrast: np.ndarray,
    sigma: np.ndarray,
    inverse: Literal["pinv", "QR", "inv"] = "QR",
):
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
    pivot = _linearly_independent_rows(contrast)
    contrast = contrast[pivot]  # reduce to linearly independent rows
    prediction = prediction[pivot]

    if inverse == "QR":
        q, r = qr(contrast @ sigma @ contrast.T)
        r_inv, _ = dtrtri(r, lower=0)
        sigma_inv = r_inv @ q.T
    elif inverse == "pinv":
        sigma_inv = np.linalg.pinv(contrast @ sigma @ contrast.T)
    elif inverse == "inv":
        sigma_inv = np.linalg.inv(contrast @ sigma @ contrast.T)

    wald = prediction @ sigma_inv @ prediction.T
    if wald < 0:
        wald = 0

    df = prediction.shape[0]
    pval = 1 - chi2.cdf(wald, df)
    return wald, df, pval


def _fold_change_cutoff(a: np.ndarray, l2fc: float = 0):
    # change basis to natural logarithm
    fold_change_cutoff = l2fc / np.log2(np.e)
    a[abs(a) < fold_change_cutoff] = 0


def _linearly_independent_rows(a: np.ndarray) -> np.ndarray:
    q, r, pivot = qr(a.T, pivoting=True)
    linearly_independent = np.abs(np.diag(r)) >= 1e-9
    pivot = pivot[: len(linearly_independent)][linearly_independent]
    return pivot
