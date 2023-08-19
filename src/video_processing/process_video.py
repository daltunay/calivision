import logging
import os
from typing import List, Optional, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import LandmarkList
from tqdm import tqdm

from .estimate_pose import PoseEstimator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class VideoProcessor:
    """Processes videos and extracts pose landmarks."""

    def __init__(
        self,
        pose_estimator: PoseEstimator,
        path: Optional[str] = None,
        webcam: Optional[int] = None,
    ) -> None:
        """Initialize the VideoProcessor object.

        Args:
            pose_estimator (PoseEstimator): Pose estimator object.
            path (Optional[str]): Path to a local video. Leave as None if using webcam input.
            webcam (Optional[int]): Webcam source (0 or 1). Leave as None if using a video path.
        """
        self.pose_estimator: PoseEstimator = pose_estimator
        self.path: Optional[str] = path
        self.webcam: Optional[int] = webcam
        self.frame_count: Union[int, None] = None
        self.fps: Union[int, None] = None
        self.landmarks_series: List[LandmarkList] = []

    def process_video(
        self, show: bool = False, width: Optional[int] = None, height: Optional[int] = None
    ) -> None:
        """Extract the landmarks of each frame of the video.

        Args:
            show (bool): Whether to show the video and landmarks in an output window or not.
            width (Optional[int]): Desired width for resizing. If None, aspect ratio is maintained.
            height (Optional[int]): Desired height for resizing. If None, aspect ratio is maintained.
        """
        # Manage the path and webcam inputs
        if self.path is not None and self.webcam is not None:
            logging.error("Both 'path' and 'webcam' cannot be specified at the same time")
            raise ValueError
        elif self.path is not None:
            logging.info(f"Initializing VideoProcessor for input: {self.path}")
            path = self.path
        elif self.webcam is not None:
            logging.info(f"Initializing VideoProcessor for input: webcam (source {self.webcam})")
            path = self.webcam
            show = True
        else:
            logging.error("Either 'path' or 'webcam' must be specified")
            raise ValueError

        self.landmarks_series = []

        # Video capture (local or webcam)
        cap = cv2.VideoCapture(path)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Get the original width and height of the video
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the new dimensions for resizing while maintaining the aspect ratio
        if width is not None and height is None:
            new_width = width
            new_height = int((new_width / original_width) * original_height)
        elif height is not None and width is None:
            new_height = height
            new_width = int((new_height / original_height) * original_width)
        else:
            new_width = width
            new_height = height

        # Progress bar
        source_desc = (
            os.path.basename(self.path) if isinstance(path, str) else f"webcam (source {path})"
        )
        pbar = tqdm(
            total=self.frame_count if isinstance(self.path, str) else None,
            desc=f"[IN PROGRESS] Body pose estimation: {source_desc}",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

        success, frame = cap.read()  # Read first frame

        # Loop over all frames
        while success:
            pbar.update(1)

            # Resize the frame if desired
            if new_width is not None and new_height is not None:
                frame = cv2.resize(frame, (new_width, new_height))

            # Process image
            (
                _base_landmarks,
                _normalized_world_landmarks,
            ) = self.pose_estimator._estimate_pose(_image=frame)
            self.landmarks_series.append(_normalized_world_landmarks)

            # Show frame and landmarks
            if show:
                self._show_landmarks(_image=frame, _landmarks=_base_landmarks)
                self._show_time(_image=frame, _time=len(self.landmarks_series) / self.fps)
                cv2.imshow(
                    "Mediapipe Feed",
                    frame,
                )

                # Manual exit
                if cv2.waitKey(1) & 0xFF == ord("\x1b"):  # Press Esc
                    self.show = False
                    cv2.destroyAllWindows()
                    break

            success, frame = cap.read()  # Read next frame

        # Exit
        pbar.set_description(f"[DONE] Body pose estimation: {source_desc}")
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

        self.landmarks_series = [landmarks for landmarks in self.landmarks_series if landmarks]
        self.frame_count = len(self.landmarks_series)

        if self.webcam:
            logging.info(
                f"Body pose estimation completed ({self.frame_count} frames, "
                f"{len(self.landmarks_series) / self.fps:.2f}s)."
            )
        else:
            logging.info(
                f"Body pose estimation completed ({len(self.landmarks_series)} / {self.frame_count} frames, "
                f"{len(self.landmarks_series) / self.fps:.2f} / {self.frame_count / self.fps:.2f}s)."
            )
        return None

    @staticmethod
    def _show_landmarks(
        _image: np.ndarray,
        _landmarks: LandmarkList,
    ) -> None:
        """Draw landmarks on a given input image.

        Args:
            _image (np.ndarray): Image array with RGB format, shape (height, width, 3).
            _landmarks (LandmarkList): Pose estimation result landmarks.
        """
        mp.solutions.drawing_utils.draw_landmarks(
            image=_image,
            landmark_list=_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2
            ),
        )

    @staticmethod
    def _show_time(_image: np.ndarray, _time: float) -> None:
        """Show a time value on a given input image.

        Args:
            _image (np.ndarray): Image array with RGB format, shape (height, width, 3).
            _time (float): Time in seconds.
        """
        cv2.putText(
            img=_image,
            text=f"t={_time:.3f}s",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
