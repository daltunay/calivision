from flask import Blueprint, current_app

data_routes = Blueprint("data_routes", __name__)


@data_routes.route("/visualize_joints")
def visualize_joints():
    """Route to visualize joint series data."""

    return current_app.app_instance.visualize_joints()


@data_routes.route("/visualize_angles")
def visualize_angles():
    """Route to visualize angle series data."""

    return current_app.app_instance.visualize_angles()


@data_routes.route("/visualize_fourier")
def visualize_fourier():
    """Route to visualize Fourier series data."""

    return current_app.app_instance.visualize_fourier()
