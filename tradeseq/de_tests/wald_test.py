from typing import Literal

from scipy.stats import chi2

import numpy as np


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
        A (`n_parmas`, `n_params`) np.ndarray, the covariance matrix for the fitted parameters.

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
