from typing import Union

import tslearn.metrics

from ..features import AngleSeries, JointSeries
from .distance_metric import DistanceMetric


class DTW(DistanceMetric):
    name: str = "DTW"

    def __init__(self):
        """Initialize an DTW distance metric calculator."""
        super().__init__()

    def compute_distance(
        self,
        series_1: Union[AngleSeries, JointSeries],
        series_2: Union[AngleSeries, JointSeries],
    ) -> float:
        """Calculate the DTW distance between two time series.

        Args:
            series_1 (Union[AngleSeries, JointSeries]): First time series.
            series_2 (Union[AngleSeries, JointSeries]): Second time series.

        Returns:
            float: The Dynamic Time Warping distance (DTW) between series_1 and series_2, normalized between 0 and 1.
        """
        similarity = tslearn.metrics.dtw(series_1, series_2)
        distance = 1 - similarity

        return distance
