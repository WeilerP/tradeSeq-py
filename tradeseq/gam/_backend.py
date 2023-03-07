from typing import List, Literal

from rpy2.robjects import numpy2ri, pandas2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

import numpy as np
import pandas as pd


class GAM:
    """GAM backend class encapsulating R gam object."""

    def __init__(self, gam):
        """Initialize GAM object.

        Parmaeters
        ----------
        gam
            rpy2 representation of fitted mgcv GAM object.
        """
        self._gam = gam
        self.covariance_matrix: np.ndarray = _get_covariance_matrix(gam)
        self.aic = _get_aic(gam)[0]

    def predict(
        self,
        lineage_assignment: np.ndarray,
        pseudotimes: np.ndarray,
        offsets: np.ndarray,
        return_type: Literal["response", "link", "lpmatrix"],
    ):
        """Predict gene count for new data.

        Parameters
        ----------
        lineage_assignment
            A ``n_predictions`` x ``n_lineage`` np.ndarray where each row contains exactly one 1 (the assigned lineage)
            and 0 everywhere else. TODO: maybe easier to just have a list with lineage indices for every data point
        pseudotimes
            A ``n_prediction`` x ``n_lineage`` np.ndarray containing the pseudotime values for every lineage.
            Note that only the pseudotimes of the corresponding lineage are considered.
            TODO: probably easier to just have list of pseudotime values
        offsets
            An np.ndarray of shape (``n_prediction``,) containing offsets for each prediciton point.
        return_type
            Should predictions be returned in log_scale ("link"), linear scale ("response") or as a linear predictor
            matrix ("lpmatrix")

        Returns
        -------
        A np.ndarray of shape (``n_predictions``,) containing the predicted counts if return_type is "link" or "log".
        A np.ndarray of shape (``n_predictions``,``n_variables``), the linear predictor matrix if return_type is
        "lpmatrix".
        """
        stats = importr("stats")  # TODO: import only once

        n_lineages = lineage_assignment.shape[1]
        lineage_assignment = pd.DataFrame(
            data=lineage_assignment, columns=[f"l{lineage_id}" for lineage_id in range(1, n_lineages + 1)]
        )
        pseudotimes = pd.DataFrame(
            data=pseudotimes, columns=[f"t{lineage_id}" for lineage_id in range(1, n_lineages + 1)]
        )
        offsets = pd.DataFrame(data=offsets, columns=["offset"])

        parameters = pd.concat([lineage_assignment, pseudotimes, offsets], axis=1)

        with localconverter(default_converter + pandas2ri.converter):
            prediction = stats.predict(self._gam, parameters, type=return_type)
        return prediction


def _get_covariance_matrix(gam) -> np.ndarray:
    np_cv_rules = default_converter + numpy2ri.converter + pandas2ri.converter
    with localconverter(np_cv_rules):
        ro.globalenv["gam"] = gam
        covariance = ro.r("gam$Vp")
    return covariance


def _get_aic(gam) -> int:
    np_cv_rules = default_converter + numpy2ri.converter + pandas2ri.converter
    with localconverter(np_cv_rules):
        ro.globalenv["gam"] = gam
        aic = ro.r("gam$aic")
    return aic


def _assign_pseudotimes(pseudotimes: np.ndarray):
    """Assign pseudotimes in R.

    Assign pseudotimes for every lineage to variables t_{lineage_id} in R.

    Parameters
    ----------
    pseudotimes
        A ``n_cells`` x ``n_lineage`` np.ndarray containing pseudotimes for every cell and lineage.
    """
    np_cv_rules = default_converter + numpy2ri.converter
    with localconverter(np_cv_rules):
        ro.globalenv["pseudotimes"] = pseudotimes
    ro.r(
        """
        for (lineage_id in seq_len(ncol(pseudotimes))) {
            assign(paste0("t",lineage_id), pseudotimes[,lineage_id])
        }
        """
    )


def _assign_lineages(w_sample: np.ndarray):
    """Assign lineage assingments in R.

    Assign lineage indicators for every lineage to variables l_{lineage_id} in R.

    Parameters
    ----------
    w_sample
        A ``n_cells`` x ``n_lineage`` np.ndarray where each row contains exactly one `1` (the assigned lineage).
        and `0` everywhere else.
    """
    np_cv_rules = default_converter + numpy2ri.converter
    with localconverter(np_cv_rules):
        ro.globalenv["w_sample"] = w_sample
    ro.r(
        """
        for (lineage_id in seq_len(ncol(w_sample))) {
            assign(paste0("l", lineage_id), 1 * (w_sample[, lineage_id] == 1))
        }
        """
    )


def _assign_offset(offset: np.ndarray):
    """Assign offset to variable offset in R.

    Parameters
    ----------
    offset
        An np.ndarray of shape (``n_cells``,) containing cell specific offsets accounting for different library sizes.
    """
    np_cv_rules = default_converter + numpy2ri.converter
    with localconverter(np_cv_rules):
        ro.globalenv["offset"] = offset


def fit(
    counts: np.ndarray,
    pseudotimes: np.ndarray,
    w_sample: np.ndarray,
    offset: np.ndarray,
    knots: np.ndarray,
    smooth_form: str,
    family: str,
    n_jobs: int,
) -> List[GAM]:
    """Fit GAMs for all genes using the R library mgcv.

    Parameters
    ----------
    counts
        A ``n_cell`` x ``n_lineage`` dense np.ndarry containing gene counts for every cell.
    pseudotimes
        A ``n_cells`` x ``n_lineage`` np.ndarray containing pseudotimes for every cell and lineage.
    w_sample
        A ``n_cells`` x ``n_lineage`` np.ndarray where each row contains exactly one `1` (the assigned lineage).
        and `0` everywhere else.
    offset
        An np.ndarray of shape (``n_cells``,) containing cell specific offsets accounting for different library
        sizes.
    knots
        Location of knots used for fitting the splines in the GAM.
    smooth_form
        Smooth form of the fitted GAM. Can be any formula accepted by mgcv.gam where y stands for gene expression
        data, and l_{lineage_id}, t_{lineage_id} are variables for the lineage-assignment and the pseudotime values,
        respectively.
    family
        Family of probability distributions that is used for fitting the GAM. Defaults to the negative binomial
        distributions. Can be any family available in mgcv.gam.
        TODO: change type hint to Literal
    n_jobs
        Number of jobs used for fitting the GAM. For n_jobs >= 2, the R library biocParallel is used.
        TODO: maybe support for -1

    Returns
    -------
        List of fitted GAM objects.
    """
    importr("mgcv")

    ro.globalenv["counts"] = numpy2ri.converter.py2rpy(counts)
    _assign_pseudotimes(pseudotimes)
    _assign_lineages(w_sample)
    _assign_offset(offset)
    ro.globalenv["n_knots"] = len(knots)
    ro.globalenv["smooth"] = ro.Formula(smooth_form)
    ro.globalenv["knots"] = default_converter.py2rpy(knots.astype(float).tolist())
    ro.globalenv["family"] = default_converter.py2rpy(family)
    # TODO: error handling while fitting
    ro.globalenv["fit"] = ro.r(
        """
        function(i){
            data = list(y = counts[,i])
            res <- mgcv::gam(smooth, family=family, knots =knots, data = data)
            return(res)
        }
        """
    )
    if n_jobs > 1:
        bioc = importr("BiocParallel")
        param = bioc.MulticoreParam(worker=n_jobs, progressbar=True)
        with localconverter(numpy2ri.converter + default_converter):
            res = bioc.bplapply(list(range(1, counts.shape[1] + 1)), ro.globalenv["fit"], BPPARAM=param)
    else:
        base = importr("base")
        res = base.lapply(list(range(1, counts.shape[1] + 1)), ro.globalenv["fit"])
    print(f"Finished fitting {counts.shape[1]} GAMs")
    return [GAM(gam) for gam in res]
