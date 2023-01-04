from typing import List

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
        self._stats = importr("stats")

    def predict(self, lineage_assignment: np.ndarray, pseudotimes: np.ndarray, offsets: np.ndarray, log_scale: bool):
        """Predict gene count for new data.

        Parameters
        ----------
        lineage_assignment
            A ``n_predictions`` x ``n_lineage`` np.ndarray where each row contains exactly one 1 (the assigned lineage)
            and 0 everywhere else. TODO: maybe easier to just have a list with lineage indices for every data point
        pseudotimes:
            A ``n_prediction`` x ``n_lineage`` np.ndarray containing the pseudotime values for every lineage.
            Note that only the pseudotimes of the corresponding lineage are considered.
            TODO: probably easier to just have list of pseudotime values
        offsets:
            An np.ndarray of shape (``n_prediction``,) containing offsets for each prediciton point.
        log_scale:
            Should predictions be returned in log_scale (this is not log1p-scale!).

        Returns
        -------
        An np.ndarray of shape (``n_predictions``,) containing the predicted counts.
        """
        n_lineages = lineage_assignment.shape[1]
        lineage_assignment = pd.DataFrame(
            data=lineage_assignment, columns=[f"l{lineage_id}" for lineage_id in range(1, n_lineages + 1)]
        )
        pseudotimes = pd.DataFrame(
            data=pseudotimes, columns=[f"t{lineage_id}" for lineage_id in range(1, n_lineages + 1)]
        )
        offsets = pd.DataFrame(data=offsets, columns=["offset"])

        parameters = pd.concat([lineage_assignment, pseudotimes, offsets], axis=1)

        if log_scale:
            return_type = "link"
        else:
            return_type = "response"

        with localconverter(default_converter + pandas2ri.converter):
            prediction = self._stats.predict(self._gam, parameters, type=return_type)
        return prediction


class GAMFitting:
    """Backend class used for fitting GAMs in R for multiple genes."""

    # TODO: go directly to mgcv.gam by clicking
    # TODO: separate R environments ?

    def __init__(
        self,
        pseudotimes: np.ndarray,
        w_sample: np.ndarray,
        offset: np.ndarray,
        knots: np.ndarray,
        smooth_form: str,
        family: str,
    ):
        """Initialize class and assign pseudotime and cell_weight values to corresponding variables in R.

        Parameters
        ----------
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
        """
        _assign_pseudotimes(pseudotimes)
        _assign_lineages(w_sample)
        _assign_offset(offset)
        self._knots: List[float] = knots.astype(float).tolist()  # Convert to list to make conversion to R easier
        self._smooth_form = smooth_form
        self._family = family
        self._mgcv = importr("mgcv")

    def fit(self, y: np.ndarray) -> GAM:
        """Fit GAM for a single gene.

        Parameters
        ----------
        y
            A np.ndarray of shape (``n_cells``,) containing gene expression data for a single gene.

        Returns
        -------
            Fitted GAM object.
        """
        ro.globalenv["n_knots"] = len(self._knots)
        ro.globalenv["y"] = ro.vectors.FloatVector(y)
        gam = self._mgcv.gam(ro.Formula(self._smooth_form), family=self._family, knots=self._knots)
        return GAM(gam)


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
