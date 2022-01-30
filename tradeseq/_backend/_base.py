from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Type,
    Tuple,
    Union,
    Literal,
    Callable,
    Optional,
    TYPE_CHECKING,
)

from rpy2.robjects import pandas2ri
from packaging.version import parse, Version, LegacyVersion
from rpy2.robjects.packages import importr, InstalledSTPackage, PackageNotInstalledError
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro

import pandas as pd

__all__ = ("TradeSeqTest", "register")


_LIBRARY: Optional[InstalledSTPackage] = None
_VERSION: Optional[Union[Version, LegacyVersion]] = None


class TradeSeqTest(ABC):
    _REGISTRY: Dict[str, Type["TradeSeqTest"]] = {}

    def __init__(self, model: ro.RS4):
        self._model = model

    @abstractmethod
    def _call(self, **kwargs: Any) -> pd.DataFrame:
        pass

    def __call__(self, **kwargs: Any) -> pd.DataFrame:
        with localconverter(ro.default_converter + pandas2ri.converter):
            try:
                return self._call(**kwargs)
            except RRuntimeError as e:
                raise RuntimeError("TODO: calling test failed.") from e

    @classmethod
    def create(cls, name: Literal["start_end"], *, model: ro.RS4) -> "TradeSeqTest":
        clazz = cls._REGISTRY.get(name, None)
        if clazz is None:
            raise ValueError(f"TODO: invalid test `{name}`, valid are {sorted(cls._REGISTRY.keys())}")

        return clazz(model)


def register(key: str) -> Callable[[Type[TradeSeqTest]], Type[TradeSeqTest]]:
    def wrapper(clazz: Type[TradeSeqTest]) -> Type[TradeSeqTest]:
        if not issubclass(clazz, TradeSeqTest):
            raise TypeError(clazz)
        TradeSeqTest._REGISTRY[key] = clazz
        return clazz

    return wrapper


def _load_library(*, force_reload: bool = False) -> Tuple[InstalledSTPackage, Union[Version, LegacyVersion]]:
    global _LIBRARY, _VERSION

    if _LIBRARY is not None and _VERSION is not None and not force_reload:
        return _LIBRARY, _VERSION

    try:
        utils = importr("utils")
        _LIBRARY = importr("tradeSeq")
        _VERSION = parse(".".join(str(n) for n in utils.packageVersion("tradeSeq")))
        if TYPE_CHECKING:
            assert isinstance(_LIBRARY, InstalledSTPackage)
            assert isinstance(_VERSION, (Version, LegacyVersion))

        return _LIBRARY, _VERSION
    except RRuntimeError as e:
        if "there is no package called ‘tradeSeq`" not in str(e):
            raise
        raise ImportError("TODO: unable to import `tradeSeq`") from None
    except PackageNotInstalledError:
        raise ImportError("TODO: unable to import `tradeSeq`") from None
