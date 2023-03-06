from typing import Literal

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core.test_base import get_gam
from tradeseq.de_tests._association_test import AssociationTest
from tradeseq.gam import GAM


class TestAssociation:
    @given(
        gam=get_gam(n_vars=2, min_obs=60, max_obs=100, n_lineages=2),
        constant=st.integers(min_value=0, max_value=10),
        n_knots=st.integers(min_value=2, max_value=4),
        contrast_type=st.sampled_from(["start", "end", "consecutive"]),
    )
    @settings(max_examples=10, deadline=50000)
    def test_constant(
        self,
        gam: GAM,
        constant: int,
        n_knots: int,
        contrast_type: Literal["start", "end", "consecutive"],
    ):
        gam._adata.X = (
            np.zeros((gam._adata.n_obs, gam._adata.n_vars), dtype=int) + constant
        )
        gam._adata.obs[gam._time_key] = np.linspace(0, 5, gam._adata.n_obs)
        del gam._adata.obsm[gam._time_key]
        weights = np.ones((gam._adata.n_obs, gam._n_lineages))
        gam._adata.obsm[gam._weights_key] = weights
        gam.fit(n_knots=n_knots)

        result = AssociationTest(gam)(
            n_points=n_knots * 2,
            contrast_type=contrast_type,
            lineage_test=True,
            global_test=True,
        )

        np.testing.assert_allclose(result["p value"], 1)

    @given(
        gam=get_gam(n_vars=2, min_obs=60, max_obs=100, n_lineages=2),
        difference=st.floats(10, 100),
        n_knots=st.integers(min_value=2, max_value=4),
        contrast_type=st.sampled_from(["start", "end", "consecutive"]),
    )
    @settings(max_examples=10, deadline=50000)
    def test_linear(
        self,
        gam: GAM,
        difference: float,
        n_knots: int,
        contrast_type: Literal["start", "end", "consecutive"],
    ):
        gam._adata.X = np.repeat(
            np.linspace(0, difference, gam._adata.n_obs)[:, np.newaxis],
            gam._adata.n_vars,
            axis=1,
        )
        gam._adata.obs[gam._time_key] = np.linspace(0, 5, gam._adata.n_obs)
        del gam._adata.obsm[gam._time_key]
        weights = np.ones((gam._adata.n_obs, gam._n_lineages))
        gam._adata.obsm[gam._weights_key] = weights
        gam._adata.obs["offset"] = np.zeros(gam._adata.n_obs)
        gam._offset_key = "offset"
        gam.fit(n_knots=n_knots)

        result = AssociationTest(gam)(
            n_points=n_knots * 2,
            contrast_type=contrast_type,
            lineage_test=True,
            global_test=True,
        )

        np.testing.assert_allclose(result["p value"], 0, atol=1e-5)
