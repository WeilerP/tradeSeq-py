from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

import numpy as np


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

    def predict(self):
        """TODO."""


class GAM_Fitting:
    """Backend class used for fitting GAMs in R for multiple genes."""

    def __init__(self, pseudotimes: np.ndarray, w_sample: np.ndarray, knots: np.ndarray, smooth_form: str, family: str):
        """Initialize class and assing pseudotime and cell_weight values to corresponding variales in R.

        Parameters
        ----------
        pseudotimes
            A ``n_cells`` x ``n_lineage`` np.ndarray containing pseudotimes for every cell and lineage.
        w_sample
            A ``n_cells`` x ``n_lineage`` np.ndarray where each row contains exactly one `1` (the assigned lineage).
            and `0` everywhere else.
        knots
            Location of knots used for fitting the splines in the GAM.
        smooth_form
            Smooth form of the fitted GAM. Can be any formula accepted by mgcv.gam where y stands for gene expression
            data, and l_{lineage_id}, t_{lineage_id} are variables for the lineage-assignment and the pseudotime values,
            respectively.
        family
            Family of probability distributions that is used for fitting the GAM. Defaults to the negative binomial
            distributions. Can be any family available in ``mgcv.gam``.
        """
        _assign_pseudotimes(pseudotimes)
        _assign_lineages(w_sample)
        self._knots = [float(knot) for knot in knots]  # Convert to list to make conversion to R easier
        self._smooth_form = smooth_form
        self._family = family
        self._mgcv = importr("mgcv")

    def fit_gam(self, y: np.ndarray) -> GAM:
        """Fit GAM for a single gene.

        Parameters
        ----------
        y
            A np.ndarray of shape (``n_cells``,) containing gene expression data for a single gene.

        Returns
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
