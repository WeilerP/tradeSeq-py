from hypothesis import given, settings
from hypothesis import strategies as st

import numpy as np

from tests.core.test_base import get_gam
from tradeseq.de_tests._diff_end_test import DiffEndTest
from tradeseq.gam import GAM


class TestDiffEnd:
    @given(
        gam=get_gam(n_vars=2, min_obs=60, max_obs=100, n_lineages=3),
        constant=st.integers(min_value=0, max_value=10),
        n_knots=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=1, deadline=50000)
    def test_equal(self, gam: GAM, constant: int, n_knots: int):
        gam._adata.X = (
            np.zeros((gam._adata.n_obs, gam._adata.n_vars), dtype=int) + constant
        )
        gam._adata.obs[gam._time_key] = np.linspace(0, 5, gam._adata.n_obs)
        del gam._adata.obsm[gam._time_key]
        weights = np.ones((gam._adata.n_obs, gam._n_lineages))
        gam._adata.obsm[gam._weights_key] = weights
        gam.fit(n_knots=n_knots)

        result = DiffEndTest(gam)(pairwise_test=True, global_test=True)

        np.testing.assert_allclose(result["p value"], 1)

    @given(
        gam=get_gam(n_vars=2, min_obs=60, max_obs=100, n_lineages=3),
        difference=st.integers(10, 100),
        n_knots=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=10, deadline=50000)
    def test_different(self, gam: GAM, difference: float, n_knots: int):
        n_lineages = 3
        n_obs = gam._adata.n_obs
        interval = n_obs // n_lineages
        gam._adata.X = np.ones((n_obs, gam._adata.n_vars), dtype=int)
        gam._adata.X[interval : 2 * interval, :] += difference
        gam._adata.X[2 * interval :, :] += 2 * difference
        # pseudotime values should not matter
        gam._adata.obs[gam._time_key] = np.random.uniform(0.0, 1.0, n_obs)
        del gam._adata.obsm[gam._time_key]

        weights = np.zeros((gam._adata.n_obs, gam._n_lineages))
        weights[:interval, 0] = 1
        weights[interval : 2 * interval, 1] = 1
        weights[2 * interval :, 2] = 1
        gam._adata.obsm[gam._weights_key] = weights

        gam._adata.obs["offset"] = np.zeros(gam._adata.n_obs)
        gam._offset_key = "offset"
        gam.fit(n_knots=n_knots)

        result = DiffEndTest(gam)(pairwise_test=True, global_test=True)

        np.testing.assert_allclose(result["p value"], 0, atol=1e-4)

    # TODO: test with one lineage constantly at 0 (failed for me for some reason)
