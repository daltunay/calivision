import json
import logging
from io import BytesIO

import cv2
import plotly
from flask import Response, render_template

from src.features import AngleSeries, FourierSeries, JointSeries
from src.video_processing.process_video import PoseEstimator, VideoProcessor
from src.visualization import (
    plot_angle_evolution,
    plot_angle_heatmap,
    plot_fourier_magnitude,
    plot_fourier_phase,
    plot_joint_series,
)


class PoseEstimationApp:
    """Class to manage the body pose estimation application."""

    def __init__(self):
        """Initialize the app instance with necessary attributes."""

        self.pose_estimator = None
        self.video_processor = None
        self.landmarks_series = None
        self.fps = None
        self.joint_series = None
        self.angle_series = None
        self.fourier_series = None
        self.pose_estimation_active = False

    def start_estimation(
        self, min_detection_confidence, min_tracking_confidence, model_complexity
    ):
        """Start the pose estimation process with given parameters.

        Args:
            min_detection_confidence (float): Minimum detection confidence threshold.
            min_tracking_confidence (float): Minimum tracking confidence threshold.
            model_complexity (int): Model complexity level.
        """
        if not self.pose_estimation_active:
            self.joint_series = None  # reset current data
            self.pose_estimator = PoseEstimator(
                model_complexity, min_detection_confidence, min_tracking_confidence
            )
            self.video_processor = VideoProcessor(self.pose_estimator, webcam=0, flask=True)
            self.start_estimation_flag = True
            self.pose_estimation_active = True

    def generate_frames(self):
        """Generates and yields frames during the pose estimation"""
        if not self.pose_estimation_active:
            return

        for annotated_frame in self.video_processor.process_video(show=True, width=1000):
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                break
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    def terminate_estimation(self):
        """Terminate the pose estimation process."""
        if self.pose_estimation_active:
            if self.video_processor is not None:
                self.video_processor._terminate()
                self.landmarks_series, self.fps = (
                    self.video_processor.normalized_world_landmarks_series,
                    self.video_processor.fps,
                )
            self.video_processor = None
            self.pose_estimation_active = False

    def process_data(self):
        """Process data series to compute joint, angle, and Fourier series data."""

        if self.landmarks_series is None:
            return "Pose estimation data not available."

        if self.joint_series is None:
            self.joint_series = JointSeries(landmarks_series=self.landmarks_series, fps=self.fps)
            self.joint_series.smooth(smooth_fraction=0.1, inplace=True)

            self.angle_series = AngleSeries(joint_series=self.joint_series)
            self.angle_series.smooth(smooth_fraction=0.1, inplace=True)

            self.fourier_series = FourierSeries(angle_series=self.angle_series, dc_offset=False)

    def visualize_joints(self):
        """Visualize the joint data."""
        if self.joint_series is None:
            return "Joint data not available."

        joint_series_plot = json.dumps(
            plot_joint_series(self.joint_series, visibility_threshold=0.5),
            cls=plotly.utils.PlotlyJSONEncoder,
        )
        return render_template("visualize_joints.html", joint_series_plot=joint_series_plot)

    def visualize_angles(self):
        """Visualize the angle data."""
        if self.angle_series is None:
            return "Angle data not available."
        angle_evolution_plot = json.dumps(
            plot_angle_evolution(self.angle_series), cls=plotly.utils.PlotlyJSONEncoder
        )
        angle_heatmap_plot = json.dumps(
            plot_angle_heatmap(self.angle_series), cls=plotly.utils.PlotlyJSONEncoder
        )
        return render_template(
            "visualize_angles.html",
            angle_evolution_plot=angle_evolution_plot,
            angle_heatmap_plot=angle_heatmap_plot,
        )

    def visualize_fourier(self):
        """Visualize the fourier data."""
        if self.fourier_series is None:
            return "Fourier data not available."
        fourier_magnitude_plot = json.dumps(
            plot_fourier_magnitude(self.fourier_series), cls=plotly.utils.PlotlyJSONEncoder
        )
        fourier_phase_plot = json.dumps(
            plot_fourier_phase(self.fourier_series), cls=plotly.utils.PlotlyJSONEncoder
        )
        return render_template(
            "visualize_fourier.html",
            fourier_magnitude_plot=fourier_magnitude_plot,
            fourier_phase_plot=fourier_phase_plot,
        )

    def export_joints(self):
        """Function to export joint data as Excel."""
        if self.joint_series is None:
            return "Joint data not available."

        excel_stream = BytesIO()
        self.joint_series.to_excel(index=True, excel_writer=excel_stream, engine="xlsxwriter")

        response = Response(
            excel_stream.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-disposition": "attachment; filename=joints_data.xlsx"},
        )

        return response

    def export_angles(self):
        """Function to export angle data as Excel."""
        if self.angle_series is None:
            return "Angle data not available."

        excel_stream = BytesIO()
        self.angle_series.to_excel(index=True, excel_writer=excel_stream, engine="xlsxwriter")

        response = Response(
            excel_stream.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-disposition": "attachment; filename=angles_data.xlsx"},
        )

        return response

    def export_fourier_magnitude(self):
        """Function to export fourier magnitude data as Excel."""
        if self.fourier_series is None:
            return "Fourier data not available."

        excel_stream = BytesIO()
        self.fourier_series.magnitude.to_excel(
            index=True, excel_writer=excel_stream, engine="xlsxwriter"
        )

        response = Response(
            excel_stream.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-disposition": "attachment; filename=fourier_magnitude_data.xlsx"},
        )

        return response

    def export_fourier_phase(self):
        """Function to export fourier phase data as Excel."""
        if self.fourier_series is None:
            return "Fourier data not available."

        excel_stream = BytesIO()
        self.fourier_series.phase.to_excel(
            index=True, excel_writer=excel_stream, engine="xlsxwriter"
        )

        response = Response(
            excel_stream.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-disposition": "attachment; filename=fourier_phase_data.xlsx"},
        )

        return response
