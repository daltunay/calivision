import json

import cv2
import plotly
from flask import Flask, Response, redirect, render_template, request, url_for

from src.features import AngleSeries, FourierSeries, JointSeries
from src.video_processing.process_video import PoseEstimator, VideoProcessor
from src.visualization import (
    plot_angle_evolution,
    plot_angle_heatmap,
    plot_fourier_magnitude,
    plot_fourier_phase,
    plot_joint_series,
)

app = Flask(__name__)


class PoseEstimationApp:
    """Class to manage the body pose estimation application."""

    def __init__(self):
        """Initialize the app instance with necessary attributes."""

        self.pose_estimator = None
        self.video_processor = None
        self.start_estimation_flag = False
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
        self.pose_estimator = PoseEstimator(
            model_complexity, min_detection_confidence, min_tracking_confidence
        )
        self.video_processor = VideoProcessor(self.pose_estimator, webcam=0, flask=True)
        self.start_estimation_flag = True
        self.pose_estimation_active = True

    def terminate_estimation(self):
        """Terminate the pose estimation process."""

        if self.video_processor is not None:
            self.video_processor._terminate()
            self.landmarks_series, self.fps = (
                self.video_processor.normalized_world_landmarks_series,
                self.video_processor.fps,
            )
        self.video_processor = None
        self.start_estimation_flag = False
        self.pose_estimation_active = False

    def process_data(self):
        """Process data series to compute joint, angle, and Fourier series data."""

        if self.landmarks_series is None:
            return "Pose estimation data not available."

        self.joint_series = JointSeries(landmarks_series=self.landmarks_series, fps=self.fps)
        self.joint_series.smooth(smooth_fraction=0.1, inplace=True)

        self.angle_series = AngleSeries(joint_series=self.joint_series)
        self.angle_series.smooth(smooth_fraction=0.1, inplace=True)

        self.fourier_series = FourierSeries(angle_series=self.angle_series)

    def visualize_joints(self):
        """Visualize the joint data."""
        if self.joint_series is None:
            return "Joint series plot data not available."

        joint_series_plot = json.dumps(
            plot_joint_series(self.joint_series, visibility_threshold=0.5),
            cls=plotly.utils.PlotlyJSONEncoder,
        )
        return render_template("visualize_joints.html", joint_series_plot=joint_series_plot)

    def visualize_angles(self):
        """Visualize the angle data."""
        if self.angle_series is None:
            return "Angle visualization data not available."
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
            return "Fourier visualization data not available."
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

    def generate_frames(self):
        if self.video_processor is None or not self.start_estimation_flag:
            return

        for annotated_frame in self.video_processor.process_video(show=True, width=1000):
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                break
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


app_instance = PoseEstimationApp()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if app_instance.start_estimation_flag:
            app_instance.terminate_estimation()
        else:
            min_detection_confidence = float(request.form["min_detection_confidence"])
            min_tracking_confidence = float(request.form["min_tracking_confidence"])
            model_complexity = int(request.form["model_complexity"])

            app_instance.start_estimation(
                min_detection_confidence, min_tracking_confidence, model_complexity
            )

    if app_instance.start_estimation_flag:
        start_button_text = "Terminate Pose Estimation"
        process_button_disabled = "disabled" if app_instance.pose_estimation_active else ""
    else:
        start_button_text = "Start Pose Estimation"
        process_button_disabled = ""

    return render_template(
        "index.html",
        start_button_text=start_button_text,
        process_button_disabled=process_button_disabled,
    )


@app.route("/terminate", methods=["POST"])
def terminate():
    """Route to terminate pose estimation and redirect back to the homepage."""

    app_instance.terminate_estimation()
    return redirect(url_for("index"))


@app.route("/process_data", methods=["GET"])
def process_data():
    """Route to process data and render data processing template."""

    app_instance.process_data()
    return render_template("process_data.html")


@app.route("/visualize_joints")
def visualize_joints():
    """Route to visualize joint series data."""

    return app_instance.visualize_joints()


@app.route("/visualize_angles")
def visualize_angles():
    """Route to visualize angle series data."""

    return app_instance.visualize_angles()


@app.route("/visualize_fourier")
def visualize_fourier():
    """Route to visualize Fourier series data."""

    return app_instance.visualize_fourier()


@app.route("/video_feed")
def video_feed():
    """Route to provide video feed with annotated frames."""

    return Response(
        app_instance.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
