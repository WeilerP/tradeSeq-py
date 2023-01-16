from typing import Optional

from anndata import AnnData
from scipy.sparse import csr_matrix
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st

import numpy as np

from tradeseq.gam import GAM

MAX_INT_VALUE = 100


@st.composite
def get_adata(
    draw,
    n_obs: Optional[int] = None,
    n_vars: Optional[int] = None,
    n_lineages: Optional[int] = None,
    min_obs: Optional[int] = 1,
    max_obs: Optional[int] = 100,
    min_vars: Optional[int] = 1,
    max_vars: Optional[int] = 100,
    min_lineages: Optional[int] = 1,
    max_lineages: Optional[int] = 20,
    weights_key: Optional[str] = None,
    time_key: Optional[str] = None,
    use_layer: bool = False,
    layer_key: Optional[str] = None,
    use_offset: bool = False,
    offset_key: Optional[str] = None,
    deterministic_weights: bool = False,
    sparse_matrix: bool = False,
) -> AnnData:
    """Generate an AnnData object with cell to lineage weights, pseudotimes and counts.

    Arguments
    ---------
    TODO

    Returns
    -------
    AnnData
        Generated :class:`~anndata.AnnData` object.
    """
    if n_obs is None:
        n_obs = draw(st.integers(min_value=min_obs, max_value=max_obs))
    if n_vars is None:
        n_vars = draw(st.integers(min_value=min_vars, max_value=max_vars))
    if n_lineages is None:
        n_lineages = draw(st.integers(min_value=min_lineages, max_value=max_lineages))

    if weights_key is None:
        weights_key = draw(st.text())
    if time_key is None:
        time_key = draw(st.text())
        # Make sure that time_key and weights_key are different
        if time_key == weights_key:
            time_key += "_t"
    if use_layer and layer_key is None:
        layer_key = draw(st.text())

    if use_offset and offset_key is None:
        offset_key = draw(st.text())

    counts = np.random.poisson(1, size=(n_obs, n_vars))

    if deterministic_weights:
        # every cell is assigned to exactly one lineage
        lineage_ids = draw(
            arrays(dtype=int, elements=st.integers(min_value=0, max_value=n_lineages - 1), shape=(n_obs,))
        )
        weights = np.zeros(shape=(n_obs, n_lineages), dtype=int)
        weights[range(n_obs), lineage_ids] = 1
    else:
        weights = draw(
            arrays(
                dtype=float, elements=st.floats(min_value=0, max_value=10, exclude_min=True), shape=(n_obs, n_lineages)
            )
        )

    pseudotimes = np.random.uniform(0, MAX_INT_VALUE - 1, size=(n_obs, n_lineages))

    if sparse_matrix:
        counts = csr_matrix(counts)

    adata = AnnData(counts)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]

    adata.obsm[time_key] = pseudotimes
    adata.obsm[time_key][0, 0] = MAX_INT_VALUE  # make sure that first lineage is longest

    adata.obsm[weights_key] = weights

    if use_layer:
        adata.layers[layer_key] = counts

    if use_offset:
        offset = draw(arrays(dtype=float, elements=st.floats(min_value=0, max_value=MAX_INT_VALUE), shape=(n_obs,)))
        adata.obs[offset_key] = offset

    return adata


@st.composite
def get_gam(
    draw,
    n_obs: Optional[int] = None,
    n_vars: Optional[int] = None,
    n_lineages: Optional[int] = None,
    min_obs: Optional[int] = 1,
    max_obs: Optional[int] = 100,
    min_vars: Optional[int] = 1,
    max_vars: Optional[int] = 100,
    min_lineages: Optional[int] = 1,
    max_lineages: Optional[int] = 20,
    weights_key: Optional[str] = None,
    time_key: Optional[str] = None,
    use_layer: bool = False,
    layer_key: Optional[str] = None,
    use_offset: bool = False,
    offset_key: Optional[str] = None,
    deterministic_weights: bool = False,
    sparse_matrix: bool = False,
) -> GAM:

    if n_lineages is None:
        n_lineages = draw(st.integers(min_value=min_lineages, max_value=max_lineages))

    if weights_key is None:
        weights_key = draw(st.text())
    if time_key is None:
        time_key = draw(st.text())
        # Make sure that time_key and weights_key are different
        if time_key == weights_key:
            time_key += "_t"

    if layer_key is None and use_layer:
        layer_key = draw(st.text())
    if not use_layer:
        layer_key = None

    if offset_key is None and use_offset:
        offset_key = draw(st.text())
    if not use_offset:
        offset_key = None

    adata = draw(
        get_adata(
            n_obs,
            n_vars,
            n_lineages,
            min_obs,
            max_obs,
            min_vars,
            max_vars,
            min_lineages,
            max_lineages,
            weights_key,
            time_key,
            use_layer,
            layer_key,
            use_offset,
            offset_key,
            deterministic_weights,
            sparse_matrix,
        )
    )

    return GAM(adata, n_lineages, time_key, weights_key, offset_key, layer_key)
