from hypothesis import given, settings, strategies as st
import anndata as ad

import numpy as np

from tradeseq.gam._gam import GAM, _calculate_offset
from tests.core.test_base import get_gam
from tests.core.tradeseq_r import TradeseqR


class TestGAMFitting:
    def test_tradeseq_r(self):
        n_knots = 6
        adata = ad.read("../_data/tradeseqTutorialData.h5ad")  # Data is modified such that it has deterministic weights
        gam = GAM(adata, 2, "pseudotime", "lineage")
        cell_weights, _ = gam._get_lineage()
        pseudotimes = gam._get_pseudotime()
        counts, _ = gam._get_counts()
        gam._assign_cells_to_lineages()
        knots = gam._get_knots(n_knots)
        offset = _calculate_offset(counts)
        tradeseq = TradeseqR(counts, pseudotimes, cell_weights, n_knots)
        knots_tradeseq = tradeseq.get_knots()
        offset_tradeseq = tradeseq.get_offset()
        assert np.allclose(knots, knots_tradeseq)
        assert np.allclose(offset, offset_tradeseq, rtol=0.1)  # There is a small difference between the offsets...

        prediction_tradeseq = tradeseq.get_prediction(0, cell_weights, pseudotimes, offset, log_scale=False)
        gam.fit(n_knots=n_knots)
        prediction = gam._model[0].predict(cell_weights, pseudotimes, offset, log_scale=False)
        assert np.allclose(prediction, prediction_tradeseq)

    @given(
        gam=get_gam(n_vars=2, min_obs=60, max_obs=100, n_lineages=2),
        constant=st.integers(min_value=0, max_value=10),
        n_knots=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=5, deadline=50000)
    def test_constant(self, gam: GAM, constant: int, n_knots: int):
        gam._adata.X = np.zeros((gam._adata.n_obs, gam._adata.n_vars), dtype=int) + constant
        gam._adata.obs[gam._time_key] = np.linspace(0, 5, gam._adata.n_obs)
        del gam._adata.obsm[gam._time_key]
        weights = np.ones((gam._adata.n_obs, gam._n_lineages))
        gam._adata.obsm[gam._weights_key] = weights
        gam.fit(n_knots=n_knots)
        prediction_n = 50
        lineage_assignment = np.zeros((prediction_n,), dtype=int)
        pseudotime = np.linspace(0.0, 1, prediction_n)
        prediction = gam.predict(gene_id=0, lineage_assignment=lineage_assignment, pseudotimes=pseudotime)
        assert np.allclose(prediction, constant)
