from anndata import AnnData
from hypothesis import given, assume, settings, strategies as st
from scipy.sparse import csr_matrix
from hypothesis.extra.numpy import arrays
import pytest

import numpy as np

from tradeseq.gam.GAM import GAM


def adata_random(n_cells=100, n_genes=2000, n_lineages=2, weights_key="lineages", time_key="pseudotime"):
    counts = csr_matrix(np.random.poisson(1, size=(n_cells, n_genes)), dtype=np.float32)
    adata = AnnData(counts)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    adata.obsm[time_key] = np.random.uniform(0, 1, size=(n_cells, n_lineages))
    adata.obsm[time_key][0, 0] = 1  # make sure that first lineage is longest
    adata.obsm[weights_key] = np.random.uniform(0, 1, size=(n_cells, n_lineages))
    return adata


class TestGetPseudotime:
    @given(time_key=st.text(), n_lineages=st.integers(min_value=1, max_value=20))
    def test_invalid_shape(self, time_key: str, n_lineages: int):
        adata = adata_random(n_lineages=n_lineages + 1, time_key=time_key)
        gam = GAM(adata)
        with pytest.raises(ValueError):
            gam._get_pseudotime(time_key, n_lineages)

    @given(time_key=st.text(), n_lineages=st.integers(min_value=2, max_value=20))
    def test_first_lineage_longest(self, time_key: str, n_lineages: int):
        adata = adata_random(n_lineages=n_lineages, time_key=time_key)
        adata.obsm[time_key][0, 1] = 2  # make sure that second lineage is longest
        gam = GAM(adata)
        with pytest.warns(RuntimeWarning):
            gam._get_pseudotime(time_key, n_lineages)

    @given(
        time_key=st.text(),
        n_lineages=st.integers(min_value=1, max_value=20),
        n_cells=st.integers(min_value=1, max_value=2000),
    )
    @settings(max_examples=50, deadline=1000)
    def test_shape(self, time_key: str, n_lineages: int, n_cells: int):
        adata = adata_random(n_cells=n_cells, n_lineages=n_lineages, time_key=time_key)
        gam = GAM(adata)
        pseudotime = gam._get_pseudotime(time_key, n_lineages)
        assert isinstance(pseudotime, np.ndarray)
        assert pseudotime.shape == (n_cells, n_lineages)

    @given(
        data=st.data(),
        time_key=st.text(),
        n_lineages=st.integers(min_value=1, max_value=20),
        n_cells=st.integers(min_value=1, max_value=2000),
    )
    @settings(max_examples=50, deadline=1000)
    def test_single_lineage(self, data, time_key: str, n_cells: int, n_lineages: int):
        adata = adata_random(n_cells=n_cells, time_key="pseudotime")
        del adata.obsm["pseudotime"]
        adata.obs[time_key] = data.draw(arrays(np.float64, (n_cells,)))
        gam = GAM(adata)
        pseudotime = gam._get_pseudotime(time_key, n_lineages)
        assert isinstance(pseudotime, np.ndarray)
        assert pseudotime.shape == (n_cells, n_lineages)


class TestGetLineage:
    @given(
        weights_key=st.text(),
        n_lineages=st.integers(min_value=1, max_value=20),
        n_cells=st.integers(min_value=1, max_value=2000),
    )
    @settings(max_examples=50, deadline=1000)
    def test_shape(self, weights_key: str, n_lineages: int, n_cells: int):
        adata = adata_random(n_cells=n_cells, n_lineages=n_lineages, weights_key=weights_key)
        gam = GAM(adata)
        weights, names = gam._get_lineage(weights_key)

        assert isinstance(weights, np.ndarray)
        assert isinstance(names, list)

        assert weights.shape == (n_cells, n_lineages)
        assert len(names) == n_lineages

    @given(weights_key1=st.text(), weights_key2=st.text())
    def test_invalid_key(self, weights_key1, weights_key2):
        assume(weights_key1 != weights_key2)
        adata = adata_random(weights_key=weights_key1)
        gam = GAM(adata)
        with pytest.raises(KeyError):
            gam._get_lineage(weights_key2)


class TestGetKnots:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @given(
        n_knots=st.integers(min_value=2, max_value=30),
        n_lineages=st.integers(min_value=1, max_value=20),
        time_key=st.text(),
    )
    def test_number_of_knots(self, n_knots, n_lineages, time_key):
        adata = adata_random(n_lineages=n_lineages, time_key=time_key)
        gam = GAM(adata)
        knots = gam._get_knots(time_key, n_lineages, n_knots)

        assert isinstance(knots, np.ndarray)
        assert knots.shape == (n_knots,)
        assert np.unique(knots).size == n_knots


class TestCellAssignment:
    def test_zero_weights(self):
        weights_key = "weights"
        adata = adata_random(weights_key=weights_key)
        adata.obsm[weights_key][0, :] = 0
        gam = GAM(adata)

        with pytest.raises(ValueError):
            gam._assign_cells_to_lineages(weights_key)

    def test_negative_weights(self):
        weights_key = "weights"
        adata = adata_random(weights_key=weights_key)
        adata.obsm[weights_key][0, 0] = -1
        gam = GAM(adata)

        with pytest.raises(ValueError):
            gam._assign_cells_to_lineages(weights_key)

    @given(n_lineages=st.integers(min_value=1, max_value=20), n_cells=st.integers(min_value=1, max_value=2000))
    @settings(max_examples=50, deadline=1000)
    def test_assignment(self, n_lineages, n_cells):
        weights_key = "weights"
        adata = adata_random(weights_key=weights_key, n_lineages=n_lineages, n_cells=n_cells)
        gam = GAM(adata)

        assign = gam._assign_cells_to_lineages(weights_key)

        assert isinstance(assign, np.ndarray)
        assert assign.shape == (n_cells, n_lineages)
        assert assign.sum() == n_cells


class TestGetCounts:
    @given(n_genes=st.integers(min_value=1, max_value=2000), n_cells=st.integers(min_value=1, max_value=2000))
    @settings(max_examples=50, deadline=1000)
    def test_without_layer(self, n_cells, n_genes):
        adata = adata_random(n_genes=n_genes, n_cells=n_cells)
        gam = GAM(adata)

        counts, genes = gam._get_counts()

        assert isinstance(counts, np.ndarray)
        assert counts.shape == (n_cells, n_genes)
        assert isinstance(genes, list)
        assert len(genes) == n_genes

    @given(n_genes=st.integers(min_value=1, max_value=2000), n_cells=st.integers(min_value=1, max_value=2000))
    @settings(max_examples=50, deadline=1000)
    def test_with_layer(self, n_cells, n_genes):
        layer_key = "l"
        adata = adata_random(n_genes=n_genes, n_cells=n_cells)
        adata.layers[layer_key] = adata.X
        gam = GAM(adata)

        counts, genes = gam._get_counts(layer_key=layer_key)

        assert isinstance(counts, np.ndarray)
        assert counts.shape == (n_cells, n_genes)
        assert isinstance(genes, list)
        assert len(genes) == n_genes
