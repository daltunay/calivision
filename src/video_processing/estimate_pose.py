import logging
from typing import Literal, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import LandmarkList

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class PoseEstimator:
    """Estimates poses from images using the Mediapipe library."""

    def __init__(
        self,
        model_complexity: Literal[0, 1, 2] = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the PoseEstimator object.

        Args:
            model_complexity (Literal[0, 1, 2]): Complexity of the Mediapipe pose estimation model.
            min_detection_confidence (float): Minimum confidence value for detection.
            min_tracking_confidence (float): Minimum confidence value for tracking.
        """
        logging.info(f"Initializing PoseEstimator: {model_complexity=}")
        self.model_complexity: Literal[0, 1, 2] = model_complexity
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.pose_estimator: mp.python.solutions.pose.Pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            smooth_landmarks=True,
        )

    def _estimate_pose(self, _image: np.ndarray) -> Tuple[LandmarkList, LandmarkList]:
        """Estimate the pose from a single image.

        Args:
            _image (np.ndarray): Image array with RGB format, shape (height, width, 3).

        Returns:
            Tuple[LandmarkList, LandmarkList]: Pose estimation result landmarks.
        """
        results = self.pose_estimator.process(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))
        _base_landmarks, _world_landmarks = (
            results.pose_landmarks,
            results.pose_world_landmarks,
        )
        return _base_landmarks, _world_landmarks
