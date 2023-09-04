import argparse
import contextlib
import json
import logging
import os
from typing import Dict, List, Optional

from ..features import AngleSeries, FourierSeries, JointSeries
from ..video_processing import PoseEstimator, VideoProcessor

DATASET_FOLDER = "dataset"
INFO_SUBFOLDER = "info"
PROCESSED_SUBFOLDER = "processed"


class DatasetProcessor:
    def __init__(
        self,
        smooth_fraction_joints: float,
        smooth_fraction_angles: float,
        model_complexity: int,
        show: bool,
    ):
        """Initialize the DatasetProcessor.

        Args:
            smooth_fraction_joints (float): Smoothing fraction for joint series.
            smooth_fraction_angles (float): Smoothing fraction for angle series.
            model_complexity (int): Pose estimator model complexity.
        """
        self.dataset_folder: str = DATASET_FOLDER
        self.info_folder: str = INFO_SUBFOLDER
        self.processed_subfolder: str = PROCESSED_SUBFOLDER
        self.smooth_fraction_joints: float = smooth_fraction_joints
        self.smooth_fraction_angles: float = smooth_fraction_angles
        self.model_complexity: int = model_complexity
        self.show: bool = show

        # Initialize the PoseEstimator with specified model complexity and confidence thresholds
        self.pose_estimator: PoseEstimator = PoseEstimator(
            model_complexity=self.model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def load_dataset_info(self, dataset_name: str) -> Dict[str, List[str]]:
        """Load dataset information from JSON file.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            dict: Dictionary containing dataset information.
        """
        dataset_info_path: str = os.path.join(
            self.dataset_folder, self.info_folder, f"{dataset_name}.json"
        )
        with open(dataset_info_path, "r") as f:
            dataset_info: Dict[str, List[str]] = json.load(f)
        return dataset_info

    def process_and_save_dataset(self, dataset_name: str) -> None:
        """Process and save video series for a given dataset.

        Args:
            dataset_name (str): Name of the dataset.
        """
        dataset_info: Dict[str, List[str]] = self.load_dataset_info(dataset_name)
        for label, videos in dataset_info.items():
            for video_name in videos:
                video_path: str = os.path.join(
                    self.dataset_folder, "raw", dataset_name, label, video_name
                )

                with contextlib.suppress(Exception):
                    self.process_and_save_video(dataset_name, label, video_name, video_path)

    def process_and_save_video(
        self, dataset_name: str, label: str, video_name: str, video_path: str
    ) -> None:
        """Process video, compute components, and save dataframes.

        Args:
            dataset_name (str): Name of the dataset.
            label (str): Label of the video.
            video_name (str): Name of the video.
            video_path (str): Path to the video file.
        """
        video_processor = VideoProcessor(
            path=video_path,
            pose_estimator=self.pose_estimator,
        )

        for _ in video_processor.process_video(show=self.show):
            pass

        joint_series = JointSeries(
            landmarks_series=video_processor.normalized_world_landmarks_series,
            fps=video_processor.fps,
        )
        joint_series.smooth(smooth_fraction=self.smooth_fraction_joints, inplace=True)
        self.save_dataframe(
            dataframe=joint_series,
            dataset_name=dataset_name,
            smooth_fraction=self.smooth_fraction_joints,
            label=label,
            video_name=video_name,
            feature_type="joints",
        )

        angle_series = AngleSeries(joint_series=joint_series)
        angle_series.smooth(smooth_fraction=self.smooth_fraction_angles, inplace=True)
        self.save_dataframe(
            dataframe=angle_series,
            dataset_name=dataset_name,
            smooth_fraction=self.smooth_fraction_angles,
            label=label,
            video_name=video_name,
            feature_type="angles",
        )

        fourier_series = FourierSeries(angle_series=angle_series)
        self.save_dataframe(
            dataframe=fourier_series,
            dataset_name=dataset_name,
            smooth_fraction=None,
            label=label,
            video_name=video_name,
            feature_type="fourier",
        )

    def save_dataframe(
        self,
        dataframe,
        dataset_name: str,
        smooth_fraction: Optional[float],
        label: str,
        video_name: str,
        feature_type: str,
    ) -> None:
        """Save DataFrame to file.

        Args:
            dataframe: DataFrame to be saved.
            dataset_name (str): Name of the dataset.
            smooth_fraction (float): Smoothing fraction used.
            label (str): Label of the video.
            video_name (str): Name of the video.
            feature_type (str): Name of the folder.
        """
        folder_path: str = os.path.join(
            self.dataset_folder, self.processed_subfolder, dataset_name, label, feature_type
        )
        os.makedirs(folder_path, exist_ok=True)
        if smooth_fraction is not None:
            file_suffix = f"smooth_{int(smooth_fraction * 100):02d}"
        else:
            file_suffix = ""
        file_path: str = os.path.join(
            folder_path, f"{os.path.splitext(video_name)[0]}_{feature_type}_{file_suffix}.pkl"
        )
        dataframe.to_pickle(file_path)
        logging.info(f"Saving computed {feature_type} data to: {file_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Process and save video series")
    parser.add_argument(
        "--smooth_fraction_joints", type=float, default=0.1, help="Smoothing fraction for joints"
    )
    parser.add_argument(
        "--smooth_fraction_angles", type=float, default=0.1, help="Smoothing fraction for angles"
    )
    parser.add_argument(
        "--model_complexity", type=int, default=1, help="Pose Estimator model complexity"
    )
    parser.add_argument(
        "-show", action="store_true", help="Show video processing and pose estimation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset inside dataset/raw/ (e.g., 'UCF50')",
    )
    args = parser.parse_args()

    processor: DatasetProcessor = DatasetProcessor(
        args.smooth_fraction_joints, args.smooth_fraction_angles, args.model_complexity, args.show
    )

    processor.process_and_save_dataset(args.dataset)


if __name__ == "__main__":
    main()
