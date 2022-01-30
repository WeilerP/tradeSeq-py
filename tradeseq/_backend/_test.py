from abc import ABC
from typing import Any, Literal, Optional, Sequence

from packaging.version import parse
import rpy2.robjects as ro

import pandas as pd

from tradeseq._backend._base import register, TradeSeqTest, _load_library


@register("start_end")
class StartVsEndTest(TradeSeqTest):
    def _call(
        self,
        glob: bool = True,
        lineages: bool = False,
        pseudotime: Optional[Sequence[float]] = None,
        l2fc: float = 0,
        **__: Any,
    ) -> pd.DataFrame:
        if pseudotime is None:
            pseudotime = ro.NULL

        library, _ = _load_library()
        return library.startVsEndTest(
            self._model, lineages=lineages, pseudotimeValues=pseudotime, l2fc=l2fc, **{"global": glob}
        )


@register("association")
class AssociationTest(TradeSeqTest):
    def _call(
        self,
        glob: bool = True,
        lineages: bool = False,
        l2fc: float = 0,
        contrast_type: Literal["start", "end", "consecutive"] = "start",
        n_points: Optional[int] = None,
        inverse: Literal["eigen", "Chol", "QR", "generalized"] = "Chol",
        **__: Any,
    ) -> pd.DataFrame:
        if contrast_type not in ("start", "end", "consecutive"):
            raise ValueError("TODO")

        library, version = _load_library()

        kwargs = {"global": glob, "nPoints": n_points, "contrastType": contrast_type, "inverse": inverse}
        if n_points is None:
            kwargs.pop("nPoints")
        if version <= parse("1.5.0"):
            kwargs.pop("contrastType")
            kwargs.pop("inverse")

        return library.associationTest(self._model, lineages=lineages, l2fc=l2fc, **kwargs)


class InterLineageTest(TradeSeqTest, ABC):
    def __init__(self, model: ro.RS4):
        super().__init__(model)
        # TODO(michalk8): check if more than 1 lineage


@register("diff_end")
class DiffEndTest(InterLineageTest):
    def _call(self, glob: bool = True, pairwise: bool = False, l2fc: float = 0, **__: Any) -> pd.DataFrame:
        library, _ = _load_library()
        return library.diffEndTest(self._model, l2fc=l2fc, pairwise=pairwise, **{"global": glob})


@register("pattern")
class PatternTest(InterLineageTest):
    def _call(
        self,
        glob: bool = True,
        pairwise: bool = False,
        l2fc: float = 0,
        n_points: Optional[int] = None,
        eigen_thresh: float = 1e-2,
        **__: Any,
    ) -> pd.DataFrame:
        library, _ = _load_library()

        kwargs = {"global": glob, "nPoints": n_points}
        if n_points is None:
            kwargs.pop("nPoints")

        return library.patternTest(
            self._model,
            l2fc=l2fc,
            pairwise=pairwise,
            eigenThresh=eigen_thresh,
            **kwargs,
        )


@register("early_de")
class EarlyDETest(InterLineageTest):
    def _call(
        self,
        glob: bool = True,
        pairwise: bool = False,
        l2fc: float = 0,
        n_points: Optional[int] = None,
        eigen_thresh: float = 1e-2,
        **__: Any,
    ) -> pd.DataFrame:
        library, _ = _load_library()

        kwargs = {"global": glob, "nPoints": n_points}
        if n_points is None:
            kwargs.pop("nPoints")

        return library.earlyDETest(
            self._model,
            l2fc=l2fc,
            pairwise=pairwise,
            eigenThresh=eigen_thresh,
            **kwargs,
        )
