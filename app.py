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
    def __init__(self):
        self.pose_estimator = None
        self.video_processor = None
        self.start_estimation_flag = False
        self.landmarks_series = None
        self.fps = None
        self.joint_series = None
        self.angle_series = None
        self.fourier_series = None

    def start_estimation(self, min_detection_confidence, min_tracking_confidence, model_complexity):
        self.pose_estimator = PoseEstimator(
            model_complexity, min_detection_confidence, min_tracking_confidence
        )
        self.video_processor = VideoProcessor(self.pose_estimator, webcam=0, flask=True)
        self.start_estimation_flag = True

    def terminate_estimation(self):
        if self.video_processor is not None:
            self.video_processor._terminate()
            self.landmarks_series, self.fps = (
                self.video_processor.normalized_world_landmarks_series,
                self.video_processor.fps,
            )
        self.video_processor = None
        self.start_estimation_flag = False

    def process_data_series(self):
        if self.landmarks_series is not None:
            self.joint_series = JointSeries(landmarks_series=self.landmarks_series, fps=self.fps)
            self.joint_series.smooth(smooth_fraction=0.1, inplace=True)
            self.joint_series_plot = json.dumps(
                plot_joint_series(self.joint_series, visibility_threshold=0.5),
                cls=plotly.utils.PlotlyJSONEncoder,
            )

            self.angle_series = AngleSeries(joint_series=self.joint_series)
            self.angle_series.smooth(smooth_fraction=0.1, inplace=True)

            self.fourier_series = FourierSeries(angle_series=self.angle_series)

    def visualize_joints(self):
        if self.joint_series_plot is not None:
            return render_template("visualize_joints.html", joint_series_plot=self.joint_series_plot)
        else:
            return "Joint series plot data not available."

    def visualize_angles(self):
        if self.angle_series is not None:
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
        else:
            return "Angle visualization data not available."

    def visualize_fourier(self):
        if self.fourier_series is not None:
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
        else:
            return "Fourier visualization data not available."

    def generate_frames(self):
        if self.video_processor is None or not self.start_estimation_flag:
            return

        for annotated_frame in self.video_processor.process_video(show=True):
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                break
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

app_instance = PoseEstimationApp()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        min_detection_confidence = float(request.form["min_detection_confidence"])
        min_tracking_confidence = float(request.form["min_tracking_confidence"])
        model_complexity = int(request.form["model_complexity"])
        
        app_instance.start_estimation(min_detection_confidence, min_tracking_confidence, model_complexity)

    return render_template("index.html")

@app.route("/terminate", methods=["POST"])
def terminate():
    app_instance.terminate_estimation()
    return redirect(url_for("index"))

@app.route("/process_data_series", methods=["GET"])
def process_data_series():
    app_instance.process_data_series()
    return render_template("process_data_series.html")

@app.route("/visualize_joints")
def visualize_joints():
    return app_instance.visualize_joints()

@app.route("/visualize_angles")
def visualize_angles():
    return app_instance.visualize_angles()

@app.route("/visualize_fourier")
def visualize_fourier():
    return app_instance.visualize_fourier()

@app.route("/video_feed")
def video_feed():
    return Response(app_instance.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
