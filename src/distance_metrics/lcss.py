from typing import Union

import tslearn.metrics

from ..features import AngleSeries, JointSeries
from .distance_metric import DistanceMetric


class LCSS(DistanceMetric):
    name: str = "LCSS"

    def __init__(self, epsilon: float = 1_000):
        """Initialize an LCSS distance metric calculator.

        Args:
            epsilon (float, optional): A parameter controlling the matching tolerance. Default is 1000.
        """
        super().__init__()
        self.epsilon: float = epsilon

    def compute_distance(
        self,
        series_1: Union[AngleSeries, JointSeries],
        series_2: Union[AngleSeries, JointSeries],
    ) -> float:
        """Calculate the LCSS distance between two time series.

        Args:
            series_1 (Union[AngleSeries, JointSeries]): First time series.
            series_2 (Union[AngleSeries, JointSeries]): Second time series.

        Returns:
            float: The Longest Common Subsequence distance (LCSS) between series_1 and series_2, normalized between 0 and 1.
        """
        similarity = tslearn.metrics.lcss(series_1, series_2, self.epsilon)
        distance = 1 - similarity

        return distance
