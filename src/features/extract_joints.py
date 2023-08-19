import logging
from typing import List, Optional

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.framework.formats.landmark_pb2 import LandmarkList
from tsmoothie.smoother import LowessSmoother

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class JointSeries(pd.DataFrame):
    """Represents a time series of joint coordinates (x, y, z position values) over time."""

    def __init__(self, landmarks_series: List[LandmarkList], fps: int):
        """Initializes a JointSeries object.

        Args:
            landmarks_series (List[LandmarkList]): List of LandmarkList objects containing joint landmarks over time.
            fps (int): Frames per second of the video.
        """
        super().__init__()
        self.__dict__["_landmarks_series"]: List[LandmarkList] = landmarks_series
        self.__dict__["fps"]: int = fps
        self._extract_joints()

    def _extract_joints(self) -> None:
        """Extracts the (x, y, z) coordinates of each joint from the landmarks time series."""
        joint_data = []
        index_values = []
        all_landmark_refs = list(mp.solutions.pose.PoseLandmark)

        logging.info("Extracting joint coordinates from landmarks")
        # Iterate through each frame's landmarks
        for frame_index, landmarks in enumerate(self._landmarks_series):
            frame_data = []

            for landmark_ref in all_landmark_refs:
                landmark = landmarks.landmark[landmark_ref]
                frame_data.extend(
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility,
                    ]
                )

            # Create a tuple for the index
            index = np.round(frame_index / self.fps, 4)
            index_values.append(index)
            joint_data.append(frame_data)

        # Create a DataFrame from the collected data
        columns = pd.MultiIndex.from_product(
            [
                [landmark_ref._name_ for landmark_ref in all_landmark_refs],
                ["x", "y", "z", "visibility"],
            ],
            names=["joint", "coordinate"],
        )
        joints_df = pd.DataFrame(
            data=joint_data, index=index_values, columns=columns, dtype="float32"
        )
        joints_df.index.name = "timestamp"

        self.index = joints_df.index
        self.loc[:, joints_df.columns] = joints_df.values
        self.columns = joints_df.columns
        return None

    def smooth(
        self,
        smooth_fraction: float = 0.1,
        inplace: bool = False,
    ) -> Optional["JointSeries"]:
        """Applies smoothing to the joint coordinates.

        Args:
            smooth_fraction (float, optional): Smoothing factor. Default is 0.1.
            inplace (bool, optional): If True, applies smoothing in place. If False, returns a new smoothed JointSeries. Default is False.

        Returns:
            Optional[JointSeries]: If inplace is False, returns a new JointSeries with smoothed data.
        """
        if smooth_fraction == 0:
            return None if inplace else self

        logging.info(f"Smoothing joint coordinates ({smooth_fraction=})")

        to_smooth = self.loc[:, (slice(None), ("x", "y", "z"))]

        smoother = LowessSmoother(smooth_fraction=smooth_fraction)
        smoothed_data = smoother.smooth(to_smooth.T.to_numpy()).smooth_data.transpose()

        if not inplace:
            return smoothed_data
        self.loc[:, (slice(None), ("x", "y", "z"))] = smoothed_data
        return None
