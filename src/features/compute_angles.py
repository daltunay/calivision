import logging
import os
from typing import Optional

import pandas as pd
import yaml
from tsmoothie.smoother import LowessSmoother

from ..utils import calculate_angle
from .extract_joints import JointSeries

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Import the list of angle joints to be computed
yaml_file_path = os.path.join(os.path.dirname(__file__), "joints_data.yaml")

with open(yaml_file_path, "r") as f:
    joints_data = yaml.safe_load(f)

ANGLE_LIST = [
    (
        joint_angle["first"],
        joint_angle["mid"],
        joint_angle["end"],
    )
    for joint_angle in joints_data
]


class AngleSeries(pd.DataFrame):
    """Class for managing time series of joint angles."""

    def __init__(self, joint_series: JointSeries):
        """Initialize AngleSeries with the provided JointSeries.

        Args:
            joint_series (JointSeries): A series of joint coordinates.
        """
        super().__init__()
        self.__dict__["_joint_series"]: JointSeries = joint_series
        self._compute_angles()

    def _compute_angles(self) -> None:
        """Extract angle values from the joint coordinates time series and update self
        DataFrame."""
        logging.info("Calculating angles from joint coordinates")
        angle_data = {}

        # Group joint series data by timestamp and calculate angles
        for timestamp in self._joint_series.index:
            angle_values = {
                angle_joints: calculate_angle(
                    *[
                        self._joint_series.loc[timestamp, joint][["x", "y", "z"]].tolist()
                        for joint in angle_joints
                    ]
                )
                for angle_joints in ANGLE_LIST
            }

            angle_data[timestamp] = angle_values

        angles_df = pd.DataFrame(
            angle_data,
            index=pd.MultiIndex.from_tuples(
                ANGLE_LIST, names=["first_joint", "mid_joint", "end_joint"]
            ),
            columns=self._joint_series.index,
        ).T

        self.index = angles_df.index
        self.loc[:, angles_df.columns] = angles_df.values
        self.columns = angles_df.columns
        return None

    def smooth(
        self, smooth_fraction: float = 0.1, inplace: bool = False
    ) -> Optional["AngleSeries"]:
        """Applies smoothing to the angle values.

        Args:
            smooth_fraction (float, optional): Smoothing factor. Default is 0.1.
        """
        if smooth_fraction == 0:
            return None if inplace else self

        logging.info(f"Smoothing angle values ({smooth_fraction=})")

        to_smooth = self.loc[:, :]

        smoother = LowessSmoother(smooth_fraction=smooth_fraction)
        smoothed_data = smoother.smooth(to_smooth.T.to_numpy()).smooth_data.transpose()

        if not inplace:
            return smoothed_data
        self.loc[:, :] = smoothed_data
        return None
