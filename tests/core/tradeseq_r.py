import numpy as np

import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter

from tradeseq.gam._backend import GAM


class TradeseqR:
    """
    Bridge to tradeseq R implementation.
    """

    def __init__(
        self,
        counts: np.ndarray,
        pseudotime: np.ndarray,
        obs_weights: np.ndarray,
        n_knots: int,
    ):
        np_cv_rules = default_converter + numpy2ri.converter
        with localconverter(np_cv_rules):
            ro.globalenv["counts"] = counts.T
            ro.globalenv["pseudotime"] = pseudotime
            ro.globalenv["cellWeights"] = obs_weights
            ro.globalenv["n_knots"] = n_knots

        ro.r(
            """
            library(tradeSeq)
            library(SingleCellExperiment)
            sce <- fitGAM(counts = counts, pseudotime = pseudotime, cellWeights = cellWeights, nknots = n_knots, verbose = FALSE)
            gam <- fitGAM(counts = counts, pseudotime = pseudotime, cellWeights = cellWeights, nknots = n_knots, verbose = FALSE, sce = FALSE)
            """
        )

    def get_knots(self):
        np_cv_rules = default_converter + numpy2ri.converter
        ro.r(
            """
            knots <- metadata(sce)$tradeSeq$knots
            """
        )
        with localconverter(np_cv_rules):
            knots = ro.r("knots")
        return knots

    def get_offset(self):
        np_cv_rules = default_converter + numpy2ri.converter
        ro.r(
            """
            offset <- colData(sce)$tradeSeq$dm$`offset(offset)`
            """
        )
        with localconverter(np_cv_rules):
            offset = ro.r("offset")
        return offset

    def get_coefficients(self, ind: int):
        np_cv_rules = default_converter + numpy2ri.converter
        with localconverter(np_cv_rules):
            ro.globalenv["ind"] = ind
            coefficients = ro.r("gam[[ind]]$coefficients")
        return coefficients

    def get_gam(self, ind: int):
        return GAM(ro.globalenv["gam"][ind])
