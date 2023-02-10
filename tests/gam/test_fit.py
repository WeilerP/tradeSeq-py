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

        # TODO: compare predictions with tradeseq

    @given(
        gam=get_gam(n_vars=2, min_obs=60, max_obs=100, n_lineages=2),
        constant=st.integers(min_value=0, max_value=10),
        n_knots=st.integers(min_value=2, max_value=4),
        n_jobs=st.integers(min_value=1, max_value=2),
    )
    @settings(max_examples=10, deadline=50000)
    def test_constant(self, gam: GAM, constant: int, n_knots: int, n_jobs: int):
        gam._adata.X = np.zeros((gam._adata.n_obs, gam._adata.n_vars), dtype=int) + constant
        gam._adata.obs[gam._time_key] = np.linspace(0, 5, gam._adata.n_obs)
        del gam._adata.obsm[gam._time_key]
        weights = np.ones((gam._adata.n_obs, gam._n_lineages))
        gam._adata.obsm[gam._weights_key] = weights
        gam.fit(n_knots=n_knots, n_jobs=n_jobs)
        prediction_n = 50
        lineage_assignment = np.zeros((prediction_n,), dtype=int)
        pseudotime = np.linspace(0.0, 1, prediction_n)
        prediction = gam.predict(gene_id=0, lineage_assignment=lineage_assignment, pseudotimes=pseudotime)
        assert np.allclose(prediction, constant)

    @given(
        gam=get_gam(n_vars=2, min_obs=60, max_obs=100, n_lineages=2),
        n_knots=st.integers(min_value=4, max_value=4),
    )
    @settings(max_examples=10, deadline=50000)
    def test_linear(self, gam: GAM, n_knots: int):
        gam._adata.X = np.exp(np.repeat(np.linspace(1, 5, gam._adata.n_obs)[:, np.newaxis], gam._adata.n_vars, axis=1))
        gam._adata.obs[gam._time_key] = np.linspace(1, 5, gam._adata.n_obs)
        del gam._adata.obsm[gam._time_key]
        weights = np.ones((gam._adata.n_obs, gam._n_lineages))
        gam._adata.obsm[gam._weights_key] = weights
        gam._adata.obs["offset"] = np.ones(gam._adata.n_obs)
        gam._offset_key = "offset"
        gam.fit(n_knots=n_knots)

        prediction_n = 50
        lineage_assignment = np.zeros((prediction_n,), dtype=int)
        pseudotime = np.linspace(1.0, 2, prediction_n)
        prediction = gam.predict(
            gene_id=0, lineage_assignment=lineage_assignment, pseudotimes=pseudotime, log_scale=True
        )
        assert np.allclose(prediction, pseudotime)