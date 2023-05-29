from typing import Optional, Union

import numpy as np

from tradeseq.de_tests._base import BetweenLineageTest


class PatternTest(BetweenLineageTest):
    """The PatternTest checks whether two lineages follow the same pattern over time."""

    def __call__(
        self,
        start_times: Optional[Union[np.ndarray, float]] = None,
        end_times: Optional[Union[np.ndarray, float]] = None,
        n_points: int = 6,
        pairwise_test: bool = False,
        global_test: bool = True,
        l2fc: float = 0,
    ):
        """Perform generalized PatternTest.

        Corresponds to the ``PatternTest`` in tradeseq for ``start_times=None`` and ``end_times=None``.
        Corresponds to the ``earlyDeTest`` in tradeseq if ``start_time`` and ``end_time`` are given as a float (in
        tradeseq of a knot location).

        Parameters
        ----------
        start_times
            Pseudo time starting from which lineage should be compared to other lineages. Either a np.ndarray of shape
            (``_lineages``,) specifying individual start times for all lineages or a float specifying the same start
            time for all lineages.
        end_times
            End pseudotime for comparison. Either a np.ndarray of shape (``_lineages``,)
            specifying individual end times for all lineages or a float specifying the same end time for all lineages.
        n_points
            Number of equally spaced time points between start_times and end_times at which to compare the lineages.
            Note that if ``end_time - start_time`` is not the same for every lineage, the lineage length is effectively
            normalized.
        pairwise_test
            Boolean indicating whether a test should be performed for all pairs of lineages (independent of other
            lineages).
        global_test
            Boolean indicating whether a global_test should be performed (across all lineages).
        l2fc
            Log2 fold change cut off.

        Returns
        -------
        A (multi-index) Pandas ``DataFrame`` containing the Wald statistic, the degrees of freedom, the p-value and the mean log2 fold change
        for each gene for each pair of lineages (if ``pairwise_test=True``) and/or globally (if ``global_test=True``).
        """
        n_lineages = self._model._n_lineages

        if start_times is None:
            start_times = np.array(self._get_start_pseudotime())
        if end_times is None:
            end_times = np.array(self._get_end_pseudotime())

        if isinstance(start_times, float):
            start_times = np.zeros((n_lineages,)) + start_times
        if isinstance(end_times, float):
            end_times = np.zeros((n_lineages,)) + end_times

        pseudotimes = np.split(
            np.linspace(start_times, end_times, num=n_points, axis=1), n_lineages
        )
        pseudotimes = [pseudotime.flatten() for pseudotime in pseudotimes]
        lineage_assigment = np.arange(0, n_lineages)

        return self._test(
            pseudotimes, lineage_assigment, pairwise_test, global_test, l2fc
        )
