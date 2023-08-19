from typing import Union

import numpy as np

from ..features import AngleSeries, FourierSeries, JointSeries
from .distance_metric import DistanceMetric


class L1(DistanceMetric):
    name: str = "L1"

    def compute_distance(
        self,
        series_1: Union[AngleSeries, JointSeries, FourierSeries],
        series_2: Union[AngleSeries, JointSeries, FourierSeries],
    ) -> float:
        """Calculate the Manhattan distance (L1 norm) between two time series.

        Args:
            series_1 (Union[AngleSeries, JointSeries, FourierSeries]): First time series.
            series_2 (Union[AngleSeries, JointSeries, FourierSeries]): Second time series.

        Returns:
            float: The Manhattan distance (L1) between series_1 and series_2.
        """
        if isinstance(series_1, FourierSeries):
            distance = np.linalg.norm(series_1.magnitude - series_2.magnitude, ord=1)
        else:
            distance = np.linalg.norm(series_1 - series_2, ord=1)
        return distance
