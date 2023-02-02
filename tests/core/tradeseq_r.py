from rpy2.robjects import numpy2ri, pandas2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

import numpy as np
import pandas as pd

from tradeseq.gam._backend import GAM


class TradeseqR:
    def __init__(self, counts: np.ndarray, pseudotime: np.ndarray, cell_weights: np.ndarray, n_knots: int):
        np_cv_rules = default_converter + numpy2ri.converter
        with localconverter(np_cv_rules):
            ro.globalenv["counts"] = counts.T
            ro.globalenv["pseudotime"] = pseudotime
            ro.globalenv["cellWeights"] = cell_weights
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

    def get_prediction(self, ind: int, lineage_assignment, pseudotimes, offsets, log_scale):
        importr("stats")

        n_lineages = lineage_assignment.shape[1]
        lineage_assignment = pd.DataFrame(
            data=lineage_assignment, columns=[f"l{lineage_id}" for lineage_id in range(1, n_lineages + 1)]
        )
        pseudotimes = pd.DataFrame(
            data=pseudotimes, columns=[f"t{lineage_id}" for lineage_id in range(1, n_lineages + 1)]
        )
        offsets = pd.DataFrame(data=offsets, columns=["offset"])
        U = pd.DataFrame(data=np.ones(pseudotimes.shape[0]), columns=["U"])

        parameters = pd.concat([lineage_assignment, pseudotimes, offsets, U], axis=1)

        if log_scale:
            return_type = "link"
        else:
            return_type = "response"

        with localconverter(pandas2ri.converter + default_converter):
            ro.globalenv["params"] = parameters
            ro.globalenv["ind"] = ind + 1
            ro.globalenv["return_type"] = return_type

        ro.r(
            """
            library(stats)
            res <- stats::predict(gam[[ind]], params, type=return_type)
            """
        )
        with localconverter(numpy2ri.converter + default_converter):
            prediction = ro.globalenv["res"]

        return prediction

    def get_gam(self, ind: int):
        return GAM(ro.globalenv["gam"][ind])
