import logging

import plotly.graph_objects as go

from ..features import AngleSeries
from ..utils import format_joint_name

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def visible_angles_only(angle_frame, visibility_threshold):
    visibility_frame = (
        angle_frame._joint_series.loc[:, (slice(None), "visibility")]
        .droplevel(level=1, axis=1)
        .applymap(lambda visibility: visibility > visibility_threshold)
    )

    # Remove angles with non-visible joints
    for timestamp in angle_frame.index:
        for col in angle_frame.columns:
            first_joint, mid_joint, end_joint = col  # Extract the joint keys from the column

            # Check if any of the joints involved have False visibility at the timestamp
            if (
                not visibility_frame.loc[timestamp, first_joint]
                or not visibility_frame.loc[timestamp, mid_joint]
                or not visibility_frame.loc[timestamp, end_joint]
            ):
                # Set the angle value to NaN for this timestamp and column
                angle_frame.loc[timestamp, col] = None
    return angle_frame


def plot_angle_evolution(angle_frame: AngleSeries, visibility_threshold: float = 0.5) -> go.Figure:
    """Plot the evolution of body angles over time using an interactive 2D line plot.

    Args:
        angle_frame (AngleSeries): Multi-column DataFrame of angles to plot. Each column represents a combination of joint angles.
        visibility_threshold (float): Threshold to determine which joints are considered visible or not.

    Returns:
        go.Figure: An interactive 2D line plot displaying the angle series.
    """
    logging.info("Plotting angle time series")
    fig = go.Figure()

    to_plot = visible_angles_only(angle_frame, visibility_threshold).dropna(axis=1, how="all")

    # Add traces for each angle combination
    for column in to_plot.columns:
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
                x=to_plot.index,
                y=to_plot[column],
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


def plot_angle_heatmap(angle_frame: AngleSeries, visibility_threshold: float = 0.5) -> go.Figure:
    """Plot a heatmap of body angles over time.

    Args:
        angle_frame (AngleSeries): Multi-column DataFrame of angles to plot. Each column represents a combination of joint angles.
        visibility_threshold (float): Threshold to determine which joints are considered visible or not.

    Returns:
        go.Figure: A heatmap displaying the angle series.
    """
    logging.info("Plotting angle heatmap")
    # Create a heatmap figure
    fig = go.Figure()

    to_plot = visible_angles_only(angle_frame, visibility_threshold).dropna(axis=1, how="all")

    # Convert the column names to a more readable format
    formatted_joint_names = [format_joint_name(column)[1] for column in to_plot.columns]

    # Define hover template for tooltips
    hover_template = "body angle: %{y}<br>" "time: %{x:.2f}s<br>" "angle: %{z:.2f}°"

    # Create the heatmap
    heatmap = go.Heatmap(
        x=to_plot.index,
        y=formatted_joint_names,
        z=to_plot.values.T,
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
