import numpy as np
from scipy.stats import wasserstein_distance

from ..features import FourierSeries
from .distance_metric import DistanceMetric
import numpy as np


class EMD(DistanceMetric):
    name: str = "EMD"

    def compute_distance(
        self,
        series_1: FourierSeries,
        series_2: FourierSeries,
    ) -> float:
        """Calculate the Earth Mover's Distance (EMD) between two FourierSeries.

        Args:
            series_1 (FourierSeries): First FourierSeries.
            series_2 (FourierSeries): Second FourierSeries.

        Returns:
            float: The Earth Mover's Distance between series_1 and series_2.
        """
        if series_1.magnitude.empty or series_2.magnitude.empty:
            raise ValueError("Magnitude data is missing in one or both FourierSeries.")

        magnitude_1 = series_1.magnitude
        magnitude_2 = series_2.magnitude

        emd_distance = 0.0
        for col in magnitude_1.columns:
            emd_distance += wasserstein_distance(magnitude_1[col], magnitude_2[col])

        return emd_distance
