import logging
from typing import Dict

from flask import Blueprint, Response, current_app, request
from src.distance_metrics import DTW, EMD, L1, L2, LCSS, DistanceMetric

from ..action_prediction import ActionRecognitionApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

data_routes = Blueprint("data_routes", __name__)


@data_routes.route("/visualize_joints")
def visualize_joints():
    """Route to visualize joint series data."""

    return current_app.pose_estimation_app_instance.visualize_joints()


@data_routes.route("/visualize_angles")
def visualize_angles():
    """Route to visualize angle series data."""

    return current_app.pose_estimation_app_instance.visualize_angles()


@data_routes.route("/visualize_fourier")
def visualize_fourier():
    """Route to visualize Fourier series data."""

    return current_app.pose_estimation_app_instance.visualize_fourier()


@data_routes.route("/export_joints")
def export_joints():
    """Route to export joint data as Excel."""
    return current_app.pose_estimation_app_instance.export_joints()


@data_routes.route("/export_angles")
def export_angles():
    """Route to export angle data as Excel."""
    return current_app.pose_estimation_app_instance.export_angles()


@data_routes.route("/export_fourier_magnitude")
def export_fourier_magnitude():
    """Route to export Fourier data as Excel."""
    return current_app.pose_estimation_app_instance.export_fourier_magnitude()


@data_routes.route("/export_fourier_phase")
def export_fourier_phase():
    """Route to export Fourier data as Excel."""
    return current_app.pose_estimation_app_instance.export_fourier_phase()


@data_routes.route("/action_recognition", methods=["POST"])
def action_recognition():
    DISTANCE_METRICS: Dict[str, DistanceMetric] = {
        "l1": L1,
        "l2": L2,
        "dtw": DTW,
        "lcss": LCSS,
        "emd": EMD,
    }

    """Route to predict the bodyweight exercise."""
    model_type = str(request.form["model_type"])
    feature_type = str(request.form["feature_type"])

    action_recognition_app_instance = ActionRecognitionApp(
        model_name=f"UCF101_{model_type}_{feature_type}_model.pth"
    )

    current_app.action_recognition_app_instance = action_recognition_app_instance
    if model_type == "knn":
        k = int(request.form["k"])
        metric = str(request.form["metric"])
        metric = DISTANCE_METRICS[metric]
        weights = str(request.form["weights"])
        current_app.action_recognition_app_instance.model.k = k
        current_app.action_recognition_app_instance.model.feature_type = feature_type
        current_app.action_recognition_app_instance.model.metric = metric()
        current_app.action_recognition_app_instance.model.weights = weights

    action_recognition_app_instance = ActionRecognitionApp(
        model_name=f"UCF101_{model_type}_{feature_type}_model.pth"
    )
    current_app.action_recognition_app_instance = action_recognition_app_instance

    if feature_type == "joints":
        X = current_app.pose_estimation_app_instance.joint_series
    elif feature_type == "angles":
        X = current_app.pose_estimation_app_instance.angle_series
    elif feature_type == "fourier":
        X = current_app.pose_estimation_app_instance.fourier_series

    current_app.action_recognition_app_instance.predict(X)
    return current_app.action_recognition_app_instance.visualize_predictions()
