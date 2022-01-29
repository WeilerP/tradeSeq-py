from abc import ABC, abstractmethod
from typing import Any

from anndata import AnnData


class TestABC(ABC):
    def __init__(self, adata: AnnData):
        self._adata = adata

    @abstractmethod
    def _fit(self, *args: Any, **kwargs: Any) -> "TestABC":
        pass

    @property
    def adata(self) -> AnnData:
        return self._adata
