import logging

import numpy as np
import pandas as pd

from .compute_angles import AngleSeries

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class FourierSeries(pd.DataFrame):
    """Class for calculating the Fourier transform of angle time series."""

    def __init__(self, angle_series: AngleSeries, dc_offset: bool = False):
        """Initialize FourierSeries with the provided AngleSeries.

        Args:
            angle_series (AngleSeries): A series of joint angles.
            dc_offset (bool): Whether to subtract the DC value from the signals or not.
        """
        super().__init__()
        self.__dict__["_angle_series"]: AngleSeries = angle_series
        self.__dict__["_dc_offset"]: AngleSeries = dc_offset
        self.__dict__["magnitude"]: pd.DataFrame = pd.DataFrame()
        self.__dict__["phase"]: pd.DataFrame = pd.DataFrame()
        self._fourier_transform()

    def _fourier_transform(self) -> None:
        """Calculate the Fourier transform of angle time series and update self DataFrame."""
        logging.info("Computing Fourier transform of angle time series")
        angle_data = self._angle_series.to_numpy()
        if self._dc_offset:
            self._dc_offset = angle_data.mean(axis=0)
            angle_data -= self._dc_offset
        freqs = np.fft.fftfreq(len(angle_data), d=1 / self._angle_series._joint_series.fps)
        transformed_data = np.fft.fft(angle_data, axis=0)

        index = pd.MultiIndex.from_tuples(
            self._angle_series.columns, names=["first_joint", "mid_joint", "end_joint"]
        )
        columns = freqs
        transformed_df = pd.DataFrame(
            data=transformed_data.T, index=index, columns=columns, dtype="complex64"
        ).transpose()
        transformed_df = transformed_df[transformed_df.index > 0]

        self.index = transformed_df.index
        self.index.name = "frequency"
        self.loc[:, transformed_df.columns] = transformed_df.values
        self.columns = transformed_df.columns

        return None

    @property
    def magnitude(self) -> pd.DataFrame:
        """Retrieve the magnitude of the Fourier transform for all frequencies."""
        return pd.DataFrame(np.abs(self), index=self.index, columns=self.columns, dtype="float32")

    @property
    def phase(self) -> pd.DataFrame:
        """Retrieve the phase of the Fourier transform for all frequencies."""
        return pd.DataFrame(
            np.angle(self, deg=True), index=self.index, columns=self.columns, dtype="float32"
        )
