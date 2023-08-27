import logging
from typing import List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import LandmarkList
from tqdm import tqdm

from ..utils import calculate_new_dimensions
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
        skip_frame: Optional[int] = 0,
        flask: bool = False,
    ) -> None:
        """Initialize the VideoProcessor object.

        Args:
            pose_estimator (PoseEstimator): Pose estimator object.
            path (Optional[str]): Path to a local video. Leave as None if using webcam input.
            webcam (Optional[int]): Webcam source (0 or 1). Leave as None if using a video path.
            flask (bool): Whether this is run inside a Flask app or not.
            skip_frame (int): number of frames to skip between two consecutive processed frames.
        """
        self.pose_estimator: PoseEstimator = pose_estimator
        self.path: Optional[str] = path
        self.webcam: Optional[int] = webcam
        self.skip_frame: Optional[int] = skip_frame
        self.flask: bool = flask

        self.frame_count: Union[int, None] = None
        self.fps: Union[int, None] = None

        self.normalized_world_landmarks_series: List[LandmarkList] = []
        self.base_landmarks_series: List[LandmarkList] = []
        self.processed_frames: List[np.ndarray] = []
        self.annotated_processed_frames: List[np.ndarray] = []

        self._terminated = False

    def _process_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[LandmarkList], List[LandmarkList]]:
        """Process a single frame and estimate pose landmarks.

        Args:
            frame (np.ndarray): Input image frame.

        Returns:
            Tuple[LandmarkList, LandmarkList]: Base landmarks and normalized world landmarks.
        """
        (base_landmarks, normalized_world_landmarks) = self.pose_estimator._estimate_pose(
            _image=frame
        )
        self.processed_frames.append(frame)
        return base_landmarks, normalized_world_landmarks

    def process_video(
        self,
        show: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Extract landmarks from each frame of the video and optionally show them.

        Args:
            show (bool): Whether to show the video and landmarks in an output window or not.
            width (Optional[int]): Desired width for resizing.
            height (Optional[int]): Desired height for resizing.
        """
        self._terminated = False
        # Manage the path and webcam inputs
        if self.path and self.webcam:
            raise ValueError("Both 'path' and 'webcam' cannot be specified at the same time")
        (self.input_type, self.source) = (
            ("path", self.path) if self.path else ("webcam", self.webcam)
        )
        logging.info(
            f"Initializing VideoProcessor for input: {self.input_type} (source {self.source})"
        )

        # Video capture (local or webcam)
        self.cap = cv2.VideoCapture(self.source)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Resize video
        original_width, original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        new_width, new_height = calculate_new_dimensions(
            original_width, original_height, width, height
        )

        # Initialization
        success, frame = self.cap.read()  # Read first frame
        self.pbar = self._initialize_progress_bar()

        # Loop over all frames
        while success:
            self.pbar.update(1)

            # Process frame
            frame = cv2.resize(frame, (new_width, new_height))
            base_landmarks, normalized_world_landmarks = self._process_frame(frame)
            if base_landmarks:
                self.normalized_world_landmarks_series.append(normalized_world_landmarks)
                self.base_landmarks_series.append(base_landmarks)

            if show:
                # Annotate frames with landmarks and time
                annotated_image = self.annotate_image(frame, base_landmarks)
                self.annotated_processed_frames.append(annotated_image)

                # Show frame
                if self.flask:
                    yield annotated_image
                else:
                    cv2.imshow(
                        "Mediapipe Feed",
                        annotated_image,
                    )

                # Manual exit
                if cv2.waitKey(1) & 0xFF == ord("\x1b"):  # Press Esc
                    show = False
                    cv2.destroyAllWindows()
                    break

            success, frame = self.cap.read()  # Read next frame

        # Cleanup
        self._terminate()
        return None

    def _initialize_progress_bar(self):
        pbar = tqdm(
            total=self.frame_count if self.input_type == "path" else None,
            desc=f"[IN PROGRESS] Body pose estimation: {self.input_type=}, {self.source=}",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
        return pbar

    def _terminate(self):
        cv2.waitKey(1)
        self.pbar.set_description(
            f"[DONE] Body pose estimation: {self.input_type=}, {self.source=}"
        )
        self.pbar.close()
        self.cap.release()
        cv2.destroyAllWindows()
        if not self._terminated:
            self.frame_count = len(self.processed_frames)
            # Retain only frames that were not skipped
            self.normalized_world_landmarks_series = self.normalized_world_landmarks_series[
                :: self.skip_frame + 1
            ]
            skipped_count = self.frame_count - len(self.normalized_world_landmarks_series)
            logging.info(
                f"Body pose estimation completed ({self.frame_count} frames of which {skipped_count} skipped, {self.frame_count / self.fps:.2f}s)."
            )
        self._terminated = True

    @staticmethod
    def _show_landmarks(image: np.ndarray, landmarks: LandmarkList) -> None:
        """Draw pose landmarks on an input image.

        Args:
            image (np.ndarray): Image array with RGB format, shape (height, width, 3).
            landmarks (LandmarkList): Pose estimation result landmarks.
        """
        annotated_image = image.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 0, 255), thickness=2, circle_radius=5
            ),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 0, 0), thickness=3
            ),
        )
        return annotated_image

    @staticmethod
    def _show_time(image: np.ndarray, time: float) -> None:
        """Display a time value on an input image.

        Args:
            image (np.ndarray): Image array with RGB format, shape (height, width, 3).
            time (float): Time in seconds.
        """
        annotated_image = image.copy()

        # Create a white rectangle as the background for the timestamp with a black border
        rect_color = (255, 255, 255)  # White background
        border_color = (0, 0, 0)  # Black border
        text = f"t={time:.3f}s"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        rect_width = text_size[0] + 20
        rect_height = text_size[1] + 20
        cv2.rectangle(
            img=annotated_image,
            pt1=(0, 0),
            pt2=(rect_width, rect_height),
            color=rect_color,
            thickness=cv2.FILLED,
        )
        cv2.rectangle(
            img=annotated_image,
            pt1=(0, 0),
            pt2=(rect_width, rect_height),
            color=border_color,
            thickness=2,
        )

        # Put the timestamp text on the white background
        cv2.putText(
            img=annotated_image,
            text=text,
            org=(10, text_size[1] + 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return annotated_image

    def annotate_image(self, image, base_landmarks):
        """Annotate a single image."""
        # Similar to process_frame, but only return annotated frame
        annotated_frame = self._show_landmarks(image, base_landmarks)
        annotated_frame = self._show_time(annotated_frame, len(self.processed_frames) / self.fps)
        return annotated_frame
