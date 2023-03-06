from abc import ABC
from typing import Any, Literal, Optional, Sequence

import pandas as pd
import rpy2.robjects as ro
from packaging.version import parse

from tradeseq._backend._base import TradeSeqTest, _load_library, register


# TODO: Add docstrings
@register("start_end")
class StartVsEndTest(TradeSeqTest):
    """TODO."""

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
            self._model,
            lineages=lineages,
            pseudotimeValues=pseudotime,
            l2fc=l2fc,
            **{"global": glob},
        )


# TODO: Add docstrings
@register("association")
class AssociationTest(TradeSeqTest):
    """TODO."""

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
        library, version = _load_library()

        kwargs = {
            "global": glob,
            "nPoints": n_points,
            "contrastType": contrast_type,
            "inverse": inverse,
        }
        if n_points is None:
            kwargs.pop("nPoints")
        if version <= parse("1.5.0"):  # TODO(michalk8): check this
            kwargs.pop("contrastType")
            kwargs.pop("inverse")
        elif contrast_type not in ("start", "end", "consecutive"):
            raise ValueError("TODO: invalid contrast type")

        return library.associationTest(
            self._model, lineages=lineages, l2fc=l2fc, **kwargs
        )


# TODO: Add docstrings
class InterLineageTest(TradeSeqTest, ABC):
    """TODO."""

    def __init__(self, model: ro.RS4):
        """TODO."""
        super().__init__(model)
        # TODO(michalk8): check if more than 1 lineage


# TODO: Add docstrings
@register("diff_end")
class DiffEndTest(InterLineageTest):
    """TODO."""

    def _call(
        self, glob: bool = True, pairwise: bool = False, l2fc: float = 0, **__: Any
    ) -> pd.DataFrame:
        library, _ = _load_library()
        return library.diffEndTest(
            self._model, l2fc=l2fc, pairwise=pairwise, **{"global": glob}
        )


# TODO: Add docstrings
@register("pattern")
class PatternTest(InterLineageTest):
    """TODO."""

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


# TODO: Add docstrings
@register("early_de")
class EarlyDETest(InterLineageTest):
    """TODO."""

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
