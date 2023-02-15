from typing import Union

import numpy as np
import pandas as pd

from tradeseq.de_tests._base import WithinLineageTest


class StartVsEndTest(WithinLineageTest):
    """Test to assess if a given gene is differently expressed at two given points of a lineage.

    The startVsEndTest compares whether a given gene is expressed differently at the start and the end of a lineage
    (or at two given pseudotime points).
    """

    def __call__(
        self,
        start: Union[None, float, np.ndarray] = None,
        end: Union[None, float, np.ndarray] = None,
        lineage_test: bool = False,
        global_test: bool = True,
    ) -> pd.DataFrame:
        """
        Perform StartVsEndTest.

        Parameters
        ----------
        start
            Start pseudotime value (that is compared against end).
            Can be a float (same start pseudotime value for every lineage), a (``n_lineages``,) np.ndarray (per lineage
            start pseudotime values) or ``None`` (then the minimum pseudotime value for every lineage is taken).
        end
            End pseudotime value (that is compared against start).
            Can be a float (same end pseudotime value for every lineage), a (``n_lineages``,) np.ndarray (per lineage
            end pseudotime values) or ``None`` (then the maximum pseudotime value for every lineage is taken).
        lineage_test
            Boolean indicating whether a test should be performed per lineage (independent of other lineages).
        global_test
            Boolean indicating whether a global_test should be performed (across all lineages).

        Returns
        -------
        A Pandas DataFrame containing the Wald statistic, the degrees of freedom and the p-value
        for each gene for each lineage (if ``lineage_test=True``) and/or globally.
        """
        pseudotimes_per_lineage = self._model._get_pseudotimes_per_lineage()
        if start is None:
            start = [pseudotimes.min() for pseudotimes in pseudotimes_per_lineage]
        if end is None:
            end = [pseudotimes.max() for pseudotimes in pseudotimes_per_lineage]

        n_lineages = self._model._n_lineages
        pseudotimes_start = np.split(np.zeros(n_lineages) + start, n_lineages)
        pseudotimes_end = np.split(np.zeros(n_lineages) + end, n_lineages)
        lineage_assignment = np.arange(n_lineages)

        return self._test(pseudotimes_start, pseudotimes_end, lineage_assignment, lineage_test, global_test)
