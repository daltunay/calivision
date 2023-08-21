import cv2
from flask import Flask, Response, render_template, request, url_for, redirect, send_file
from src.video_processing.process_video import PoseEstimator, VideoProcessor
from src.features import AngleSeries, FourierSeries, JointSeries
from src.visualization import (
    plot_joint_series,
    plot_angle_evolution,
    plot_angle_heatmap,
    plot_fourier_magnitude,
    plot_fourier_phase,
)
import json
import plotly

app = Flask(__name__)

# Global variables
pose_estimator = None
video_processor = None
start_estimation = False
landmarks_series = None
fps = None
joint_series = None
angle_series = None
fourier_series = None

# Section: Routes for Pose Estimation and Data Processing


@app.route("/", methods=["GET", "POST"])
def index():
    global pose_estimator, video_processor, start_estimation

    if request.method == "POST":
        # Get user input for PoseEstimator
        min_detection_confidence = float(request.form["min_detection_confidence"])
        min_tracking_confidence = float(request.form["min_tracking_confidence"])

        # Get user input for VideoProcessor
        model_complexity = int(request.form["model_complexity"])

        # Initialize PoseEstimator and VideoProcessor
        pose_estimator = PoseEstimator(
            model_complexity, min_detection_confidence, min_tracking_confidence
        )
        video_processor = VideoProcessor(pose_estimator, webcam=0, flask=True)

        # Start pose estimation
        start_estimation = True

    return render_template("index.html")


@app.route("/terminate", methods=["POST"])
def terminate():
    global video_processor, start_estimation, landmarks_series, fps

    if video_processor is not None:
        video_processor._terminate()
        landmarks_series, fps = (
            video_processor.normalized_world_landmarks_series,
            video_processor.fps,
        )
    video_processor = None
    start_estimation = False

    return redirect(url_for("index"))


@app.route("/visualize_joints")
def visualize_joints():
    global joint_series_plot

    if joint_series_plot is not None:
        return render_template("visualize_joints.html", joint_series_plot=joint_series_plot)
    else:
        return "Joint series plot data not available."


@app.route("/visualize_angles")
def visualize_angles():
    global angle_evolution_plot, angle_heatmap_plot

    if angle_evolution_plot is not None and angle_heatmap_plot is not None:
        return render_template(
            "visualize_angles.html",
            angle_evolution_plot=angle_evolution_plot,
            angle_heatmap_plot=angle_heatmap_plot,
        )
    else:
        return "Angle visualization data not available."


@app.route("/visualize_fourier")
def show_visualize_fourier():
    global fourier_magnitude_plot, fourier_phase_plot

    if fourier_magnitude_plot is not None and fourier_phase_plot is not None:
        return render_template(
            "visualize_fourier.html",
            fourier_magnitude_plot=fourier_magnitude_plot,
            fourier_phase_plot=fourier_phase_plot,
        )
    else:
        return "Fourier visualization data not available."


@app.route("/process_data_series", methods=["GET"])
def process_data_series():
    global landmarks_series, fps, joint_series, joint_series_plot, angle_series, angle_evolution_plot, angle_heatmap_plot, fourier_series, fourier_magnitude_plot, fourier_phase_plot

    if landmarks_series is not None:
        # Process the data and generate Plotly figures
        joint_series = JointSeries(landmarks_series=landmarks_series, fps=fps)
        joint_series.smooth(smooth_fraction=0.1, inplace=True)
        joint_series_plot = json.dumps(
            plot_joint_series(joint_series, visibility_threshold=0.5),
            cls=plotly.utils.PlotlyJSONEncoder,
        )

        angle_series = AngleSeries(joint_series=joint_series)
        angle_series.smooth(smooth_fraction=0.1, inplace=True)
        angle_evolution_plot = json.dumps(
            plot_angle_evolution(angle_series), cls=plotly.utils.PlotlyJSONEncoder
        )
        angle_heatmap_plot = json.dumps(
            plot_angle_heatmap(angle_series), cls=plotly.utils.PlotlyJSONEncoder
        )

        fourier_series = FourierSeries(angle_series=angle_series)
        fourier_magnitude_plot = json.dumps(
            plot_fourier_magnitude(fourier_series), cls=plotly.utils.PlotlyJSONEncoder
        )
        fourier_phase_plot = json.dumps(
            plot_fourier_phase(fourier_series), cls=plotly.utils.PlotlyJSONEncoder
        )

        return render_template("explore_processed_data.html", 
                               joint_series_plot=joint_series_plot,
                               angle_evolution_plot=angle_evolution_plot,
                               angle_heatmap_plot=angle_heatmap_plot,
                               fourier_magnitude_plot=fourier_magnitude_plot,
                               fourier_phase_plot=fourier_phase_plot)
    else:
        return "Landmarks series data not available."


@app.route("/explore_processed_data", methods=["GET"])
def explore_processed_data():
    return render_template("explore_processed_data.html")


def generate_frames():
    global video_processor, start_estimation

    if video_processor is None or not start_estimation:
        return

    for annotated_frame in video_processor.process_video(show=True):
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            break
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
