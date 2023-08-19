from typing import Union

from ..features import AngleSeries, FourierSeries, JointSeries


class DistanceMetric:
    def __init__(self):
        self.name = None

    def compute_distance(
        self,
        series_1: Union[AngleSeries, JointSeries, FourierSeries],
        series_2: Union[AngleSeries, JointSeries, FourierSeries],
    ) -> float:
        """Calculate the distance between two time series.

        This is a base class method and should be overridden by subclasses.

        Args:
            series_1 (Union[AngleSeries, JointSeries, FourierSeries]): First time series (or Fourier transform).
            series_2 (Union[AngleSeries, JointSeries, FourierSeries]): Second time series (or Fourier transform).

        Returns:
            float: The calculated distance between the two time series.
        """
        raise NotImplementedError("Subclasses must implement compute_distance")
