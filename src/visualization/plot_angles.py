import logging
from typing import Tuple

import plotly.graph_objects as go

from ..features import AngleSeries
from ..utils import format_joint_name

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def plot_angle_evolution(angle_frame: AngleSeries) -> go.Figure:
    """Plot the evolution of body angles over time using an interactive 2D line plot.

    Args:
        angle_frame (AngleSeries): Multi-column DataFrame of angles to plot. Each column represents a combination of joint angles.

    Returns:
        go.Figure: An interactive 2D line plot displaying the angle series.
    """
    logging.info("Plotting angle time series")
    fig = go.Figure()

    # Add traces for each angle combination
    for column in angle_frame.columns:
        (first, mid, end), formatted_joint_name = format_joint_name(column)

        # Define hover template for tooltips
        hover_template = (
            f"first: {first}<br>"
            f"mid: {mid}<br>"
            f"end: {end}<br>"
            "time: %{x:.2f}s<br>"
            "angle: %{y:.2f}°"
        )

        # Add a scatter trace for the angle series
        fig.add_trace(
            go.Scatter(
                x=angle_frame.index,
                y=angle_frame[column],
                mode="lines",
                name=formatted_joint_name,
                hovertemplate=hover_template,
            )
        )

    # Update layout of the figure
    fig.update_layout(
        width=1280,
        height=720,
        title="Evolution of Body Angles Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Angle (°)",
        legend_title="Body Angles",
        hovermode="closest",
    )

    return fig


def plot_angle_heatmap(angle_frame: AngleSeries) -> go.Figure:
    """Plot a heatmap of body angles over time.

    Args:
        angle_frame (AngleSeries): Multi-column DataFrame of angles to plot. Each column represents a combination of joint angles.

    Returns:
        go.Figure: A heatmap displaying the angle series.
    """
    logging.info("Plotting angle heatmap")
    # Create a heatmap figure
    fig = go.Figure()

    # Convert the column names to a more readable format
    formatted_joint_names = [format_joint_name(column)[1] for column in angle_frame.columns]

    # Define hover template for tooltips
    hover_template = "body angle: %{y}<br>" "time: %{x:.2f}s<br>" "angle: %{z:.2f}°"

    # Create the heatmap
    heatmap = go.Heatmap(
        x=angle_frame.index,
        y=formatted_joint_names,
        z=angle_frame.values.T,
        colorscale="magma",
        hoverongaps=False,
        colorbar={"title": "Angle (°)"},
        zmin=0,
        zmax=180.0,
        hovertemplate=hover_template,
        name="",
    )

    # Add the heatmap trace to the figure
    fig.add_trace(heatmap)

    # Update layout of the figure
    fig.update_layout(
        width=1280,
        height=720,
        title="Heatmap of Body Angles Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Body Angles",
        hovermode="closest",
        xaxis=dict(dtick=1),  # Set tick interval for better x-axis readability
    )

    return fig
