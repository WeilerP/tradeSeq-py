from hypothesis import given, assume, settings, strategies as st
from hypothesis.extra.numpy import arrays
import pytest

import numpy as np

from tradeseq.gam._gam import GAM
from tests.core.test_base import get_gam, MAX_INT_VALUE


class TestGetPseudotime:
    @given(gam=get_gam())
    def test_invalid_shape(self, gam):
        gam._n_lineages = gam._n_lineages + 1
        with pytest.raises(ValueError):
            gam._get_pseudotime()

    @given(gam=get_gam(min_lineages=2))
    def test_first_lineage_longest(self, gam):
        gam._adata.obsm[gam._time_key][0, 1] = MAX_INT_VALUE + 1  # make sure that second lineage is longest
        with pytest.warns(RuntimeWarning):
            gam._get_pseudotime()

    @given(gam=get_gam())
    @settings(max_examples=50, deadline=1000)
    def test_shape(self, gam: GAM):
        pseudotime = gam._get_pseudotime()
        assert isinstance(pseudotime, np.ndarray)
        assert pseudotime.shape == (gam._adata.n_obs, gam._n_lineages)

    @given(gam=get_gam(), data=st.data())
    @settings(max_examples=50, deadline=1000)
    def test_single_lineage(self, gam: GAM, data):
        del gam._adata.obsm[gam._time_key]
        gam._adata.obs[gam._time_key] = data.draw(arrays(np.float64, (gam._adata.n_obs,)))
        pseudotime = gam._get_pseudotime()
        assert isinstance(pseudotime, np.ndarray)
        assert pseudotime.shape == (gam._adata.n_obs, gam._n_lineages)


class TestGetLineage:
    @given(gam=get_gam())
    @settings(max_examples=50, deadline=1000)
    def test_shape(self, gam: GAM):
        weights, names = gam._get_lineage()

        assert isinstance(weights, np.ndarray)
        assert isinstance(names, list)

        assert weights.shape == (gam._adata.n_obs, gam._n_lineages)
        assert len(names) == gam._n_lineages

    @given(gam=get_gam(), weights_key2=st.text())
    def test_invalid_key(self, gam, weights_key2):
        assume(weights_key2 not in gam._adata.obsm.keys())
        gam._weights_key = weights_key2
        with pytest.raises(KeyError):
            gam._get_lineage()


class TestGetKnots:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @given(gam=get_gam(), n_knots=st.integers(min_value=2, max_value=20))
    def test_number_of_knots(self, gam: GAM, n_knots: int):
        gam._lineage_assignment, _ = gam._assign_cells_to_lineages()
        assume((gam._lineage_assignment[:, 0] == 1).any())  # at least one cell is assigned to first lineage
        knots = gam._get_knots(n_knots)

        assert isinstance(knots, np.ndarray)
        assert knots.shape == (n_knots,)
        assert np.unique(knots).size == n_knots


class TestCellAssignment:
    @given(gam=get_gam())
    def test_zero_weights(self, gam: GAM):
        gam._adata.obsm[gam._weights_key][0, :] = 0

        with pytest.raises(ValueError):
            gam._assign_cells_to_lineages()

    @given(gam=get_gam())
    def test_negative_weights(self, gam: GAM):
        gam._adata.obsm[gam._weights_key][0, 0] = -1

        with pytest.raises(ValueError):
            gam._assign_cells_to_lineages()

    @given(gam=get_gam())
    @settings(max_examples=50, deadline=1000)
    def test_assignment(self, gam: GAM):
        assign, lineage_names = gam._assign_cells_to_lineages()

        assert isinstance(assign, np.ndarray)
        assert assign.shape == (gam._adata.n_obs, gam._n_lineages)
        assert assign.sum() == gam._adata.n_obs

        assert isinstance(lineage_names, list)
        assert len(lineage_names) == gam._n_lineages


class TestGetCounts:
    @given(gam=get_gam())
    @settings(max_examples=50, deadline=1000)
    def test_without_layer(self, gam: GAM):
        counts, genes = gam._get_counts()

        assert isinstance(counts, np.ndarray)
        assert counts.shape == (gam._adata.n_obs, gam._adata.n_vars)
        assert isinstance(genes, list)
        assert len(genes) == gam._adata.n_vars

    @given(gam=get_gam(use_layer=True))
    @settings(max_examples=50, deadline=1000)
    def test_with_layer(self, gam):
        counts, genes = gam._get_counts()

        assert isinstance(counts, np.ndarray)
        assert counts.shape == (gam._adata.n_obs, gam._adata.n_vars)
        assert isinstance(genes, list)
        assert len(genes) == gam._adata.n_vars
