from flask import Blueprint, Response, current_app, redirect, render_template, request, url_for

visualization_routes = Blueprint("visualization_routes", __name__)


@visualization_routes.route("/visualize_joints")
def visualize_joints():
    """Route to visualize joint series data."""

    return current_app.app_instance.visualize_joints()


@visualization_routes.route("/visualize_angles")
def visualize_angles():
    """Route to visualize angle series data."""

    return current_app.app_instance.visualize_angles()


@visualization_routes.route("/visualize_fourier")
def visualize_fourier():
    """Route to visualize Fourier series data."""

    return current_app.app_instance.visualize_fourier()
