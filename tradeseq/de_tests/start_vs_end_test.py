from typing import Union

import numpy as np

from tradeseq.de_tests._base import WithinLineageTest


class StartVsEndTest(WithinLineageTest):
    """The startVsEndTest compares whether a given gene is expressed differently at the start and the end of a lineage (or at two given pseudotime points)."""

    def __call__(
        self,
        start: Union[None, float, np.ndarry] = None,
        end: Union[None, float, np.ndarry] = None,
        global_test: bool = True,
    ):
        """
        Perform StartVsEndTest.

        Parameters
        ----------
        start
            Start pseudotime value (that is compared against end).
            Can be a float (same start pseudotime value for every lineage), a (``n_lineages``,) np.ndarray (per lineage
            start pseudotime values) or None (then the minimum pseudotime value for every lineage is taken).
        end
            End pseudotime value (that is compared against start).
            Can be a float (same end pseudotime value for every lineage), a (``n_lineages``,) np.ndarray (per lineage
            end pseudotime values) or None (then the maximum pseudotime value for every lineage is taken).
        global_test
            Boolean indicating wheter a global_test should be performed.

        Returns
        -------
        A pandas data frame containing the wald statistic, the degrees of freedom and the p value for each gene.
        """
        pseudotimes_per_lineage = self._model._get_pseudotimes_per_lineage()
        if start is None:
            start = [pseudotimes.min() for pseudotimes in pseudotimes_per_lineage]
        if end is None:
            end = [pseudotimes.max() for pseudotimes in pseudotimes_per_lineage]

        n_lineages = self._model._n_lineages
        pseudotimes_start = np.zeros(n_lineages) + start
        pseudotimes_end = np.zeros(n_lineages) + end
        lineage_assignment = np.arange(n_lineages)

        return self._test(pseudotimes_start, pseudotimes_end, lineage_assignment)
