import numpy as np

from tradeseq.de_tests._base import BetweenLineageTest


class DiffEndTest(BetweenLineageTest):
    """The DiffEndTest checks whether a gene is expressed differently at the end of two lineages."""

    def __call__(
        self, pairwise_test: bool = False, global_test: bool = True, l2fc: float = 0
    ):
        """Perform DiffEndTest.

        Parameters
        ----------
        pairwise_test
            Boolean indicating whether a test should be performed for all pairs of lineages (independent of other lineages).
        global_test
            Boolean indicating whether a global_test should be performed (across all lineages).
        l2fc
            Log2 fold change cut off.

        Returns
        -------
        A (multi index) Pandas DataFrame containing the Wald statistic, the degrees of freedom, the p-value and the mean log2 fold change
        for each gene for each pair of lineages (if ``pairwise_test=True``) and/or globally (if ``global_test=True``).
        """
        end_pseudotimes = [np.array([end]) for end in self._get_end_pseudotime()]
        lineages = np.arange(self._model._n_lineages)

        return self._test(end_pseudotimes, lineages, pairwise_test, global_test, l2fc)
