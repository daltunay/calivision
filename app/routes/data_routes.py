from flask import Blueprint, Response, current_app

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


@data_routes.route("/export_joints")
def export_joints():
    """Route to export joint data as Excel."""
    return current_app.app_instance.export_joints()


@data_routes.route("/export_angles")
def export_angles():
    """Route to export angle data as Excel."""
    return current_app.app_instance.export_angles()


@data_routes.route("/export_fourier_magnitude")
def export_fourier_magnitude():
    """Route to export Fourier data as Excel."""
    return current_app.app_instance.export_fourier_magnitude()

@data_routes.route("/export_fourier_phase")
def export_fourier_phase():
    """Route to export Fourier data as Excel."""
    return current_app.app_instance.export_fourier_phase()
