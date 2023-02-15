from typing import Literal

import numpy as np
import pandas as pd

from tradeseq.de_tests._base import WithinLineageTest


class AssociationTest(WithinLineageTest):
    """The AssociationTest checks whether the expression of a gene varies with pseudotime."""

    def __call__(
        self, n_points: int, contrast_type: Literal["start", "end", "consecutive"], lineage_test, global_test
    ) -> pd.DataFrame:
        """
        Perform AssociationTest.

        Parameters
        ----------
        n_points
            Number of equally spaced points that should be compared.
        contrast_type
            Specifies how to compare time points:
            * start:
                Compare equally spaced pseudotimes between start and end of the lineage all to the start pseudotime of the lineage.
            * end:
                Compare equally spaced pseudotimes between start and end of the lineage all to the start pseudotime of the lineage.
            * consecutive:
                Compare all pairs of consecutive timepoints in the list of equally spaced pseudotimes between start and end of a lineage.
                Not recommended as the differences can be very small.
        lineage_test
            Boolean indicating whether a test should be performed per lineage (independent of other lineages).
        global_test
            Boolean indicating whether a global_test should be performed (across all lineages).

        Returns
        -------
        A Pandas DataFrame containing the Wald statistic, the degrees of freedom and the p-value
        for each gene for each lineage (if ``lineage_test=True``) and/or globally.
        """
        start = self._get_start_pseudotime()
        end = self._get_end_pseudotime()

        pseudotimes_a = []
        pseudotimes_b = []
        lineages = np.arange(self._model._n_lineages)

        if contrast_type == "start":
            for lineage_id in lineages:
                pseudotimes_a.append(np.repeat(start[lineage_id], n_points))
                pseudotimes_b.append(np.linspace(start=start[lineage_id], stop=end[lineage_id], num=n_points))

        elif contrast_type == "end":
            for lineage_id in lineages:
                pseudotimes_a.append(np.repeat(end[lineage_id], n_points))
                pseudotimes_b.append(np.linspace(start=start[lineage_id], stop=end[lineage_id], num=n_points))

        elif contrast_type == "consecutive":
            for lineage_id in lineages:
                equally_spaced = np.linspace(start=start[lineage_id], stop=end[lineage_id], num=n_points)
                pseudotimes_a.append(equally_spaced[:-1])
                pseudotimes_b.append(equally_spaced[1:])

        else:
            raise ValueError(
                f"Contrast type cannot be {contrast_type}. Contrast type has to be 'start', 'stop' or 'consecutive'"
            )

        return self._test(pseudotimes_a, pseudotimes_b, lineages, lineage_test, global_test)
