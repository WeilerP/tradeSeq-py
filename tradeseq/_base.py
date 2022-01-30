from io import StringIO
from abc import ABC, abstractmethod
from typing import Any, Sized, Tuple, Union, Literal, Optional, Sequence, TYPE_CHECKING
from contextlib import redirect_stderr
import sys
import multiprocessing

from anndata import AnnData
from scipy.sparse import issparse
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro

import numpy as np
import pandas as pd

from tradeseq._backend._base import TradeSeqTest, _load_library

_PARALLEL = importr("BiocParallel")


class TestABC(ABC):
    def __init__(self, adata: AnnData):
        self._adata = adata

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> "TestABC":
        pass

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @property
    def adata(self) -> AnnData:
        return self._adata


class Test(TestABC, ABC):
    def __init__(self, adata: AnnData):
        super().__init__(adata)
        self._model = None
        self._genes: Optional[np.ndarray] = None
        self._lineages: Optional[np.ndarray] = None

    def fit(
        self,
        layer: Optional[str] = None,
        lineage_key: str = "lineages",
        pseudotime_key: str = "pseudotime",
        use_raw: Optional[bool] = None,
        genes: Optional[Union[str, Sequence[str]]] = None,
        # TODO(michalk8): Gamma and inverse.gaussian?
        family: Literal["nb", "gaussian", "poisson", "binomial"] = "nb",
        n_knots: int = 6,
        offset: Optional[np.ndarray] = None,
        n_workers: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "TestABC":
        library, _ = _load_library()

        if offset is not None:
            offset = np.asarray(offset).ravel()
            if offset.shape != (self.adata.n_obs,):
                raise ValueError("TODO")
            kwargs["offset"] = (
                ro.vectors.FloatVector(offset) if np.issubdtype(offset.dtype, float) else ro.vectors.IntVector(offset)
            )

        bpparam = _PARALLEL.bpparam()
        if n_workers in (None, 1):
            kwargs["parallel"] = False
        else:
            kwargs["parallel"] = True
            bpparam.slots[".xData"]["workers"] = _get_n_workers(n_workers)
            bpparam.slots[".xData"]["progressbar"] = verbose
        kwargs["BPPARAM"] = bpparam
        kwargs.pop("genes", None)

        if family == "nb" and use_raw is None:
            use_raw = True

        counts, self._genes = self._get_counts(layer, genes=genes, use_raw=use_raw)
        lineages, self._lineages = self._get_lineage(lineage_key)
        pseudotime = self._get_pseudotime(pseudotime_key, n_lineages=lineages.shape[1])

        with redirect_stderr(sys.stderr if verbose else StringIO()):
            with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
                try:
                    self._model = library.fitGAM(
                        counts,
                        cellWeights=lineages,
                        pseudotime=pseudotime,
                        nknots=n_knots,
                        family=family,
                        sce=True,
                        verbose=False,
                        **kwargs,
                    )
                except RRuntimeError as e:
                    raise RuntimeError(str(e)) from None

        return self

    def predict(self, test: Literal["start_end"] = "start_end", *args: Any, **kwargs: Any) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("TODO")

        test = TradeSeqTest.create(test, model=self._model)
        df = test(*args, **kwargs).set_index(self._genes)
        mapper = {
            (f"{k}lineage{i + 1}" if k == "logFC" else f"{k}_lineage{i + 1}"): f"{k}_{lin}"
            for k in ("waldStat", "df", "pvalue", "logFC")
            for i, lin in enumerate(self._lineages)
        }
        return df.rename(mapper, axis=1, errors="ignore")

    def _get_counts(
        self,
        layer: Optional[str] = None,
        genes: Optional[Union[str, Sequence[str]]] = None,
        use_raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if use_raw and self.adata.raw is None:
            use_raw = False  # TODO(warn)

        subset = self._get_gene_subset(genes, normalize=use_raw)
        if use_raw:
            adata = self.adata.raw
            adata = adata if genes is None else adata[:, adata.var_names.isin(subset)]
        else:
            adata = self.adata if subset is None else self.adata[:, subset]
        subset = np.array([str(n) for n in adata.var_names])

        if layer is None:
            data = adata.X
        elif layer in adata.layers:
            data = adata.layers[layer]
        else:
            raise KeyError("TODO")

        # TODO(michalk): warn of too many genes
        return (data.A if issparse(data) else data).T, subset

    def _get_gene_subset(
        self, genes: Optional[Union[str, Sized]] = None, normalize: bool = False
    ) -> Optional[np.ndarray]:
        if genes is None:
            return None

        if isinstance(genes, str):
            genes = self.adata.var[genes]
        genes = np.asarray(genes)
        if not len(genes):
            raise ValueError("TODO")

        if not normalize:
            return genes
        if TYPE_CHECKING:
            assert isinstance(genes, np.ndarray)

        # TODO(michalk8): check
        if np.issubdtype(genes.dtype, bool):
            return self.adata.var_names[genes]
        if np.issubdtype(genes.dtype, int):
            return self.adata.var_names[genes]
        return genes

    def _get_lineage(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        data = self.adata.obsm[key]
        if isinstance(data, pd.DataFrame):
            return np.asarray(data), np.array([str(c) for c in data.columns])
        if hasattr(data, "names"):
            return np.asarray(data), np.array([str(c) for c in data.names])

        data = np.asarray(data)
        names = np.array([str(i) for i in range(data.shape[1])])

        return data, names

    def _get_pseudotime(self, key: str, n_lineages: int) -> np.ndarray:
        attrs = ["obsm", "obs"] if n_lineages > 1 else ["obs", "obsm"]
        for attr in attrs:
            try:
                pseudotime = np.asarray(getattr(self.adata, attr)[key])
                if pseudotime.ndim == 1:
                    return np.repeat(pseudotime[:, None], n_lineages, axis=1)
                if pseudotime.shape != (self.adata.n_obs, n_lineages):
                    raise ValueError("TODO")
                return pseudotime
            except KeyError:
                pass

        raise KeyError("TODO")

    def _format_params(self) -> str:
        if self._model is None:
            n_genes, n_lineages = None, None
        else:
            n_genes, n_lineages = len(self._genes), len(self._lineages)

        return f"genes={n_genes}, lineages={n_lineages}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params()}]"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params()}]"


def _get_n_workers(n: Optional[int]) -> int:
    if n is None or n == 1:
        return 1
    if n == 0:
        raise ValueError("Number of workers cannot be `0`.")
    if n < 0:
        return multiprocessing.cpu_count() + 1 + n

    return n
