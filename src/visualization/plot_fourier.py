import logging

import numpy as np
import plotly.graph_objects as go

from ..utils import format_joint_name

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def visible_angles_only(
    fourier_frame, visibility_threshold: float = 0.5, visibility_percentage_threshold: float = 0.5
):
    visibility_frame = (
        fourier_frame._angle_series._joint_series.loc[:, (slice(None), "visibility")]
        .droplevel(level=1, axis=1)
        .applymap(lambda visibility: visibility > visibility_threshold)
    )

    # Calculate the minimum number of visible timestamps required
    min_visible_timestamps = int(fourier_frame.index.size * visibility_percentage_threshold)

    # List to store the columns with all visible joints
    visible_columns = []

    for col in fourier_frame.columns:
        first_joint, mid_joint, end_joint = col  # Extract the joint keys from the column

        # Count the number of visible timestamps for each joint
        visible_timestamps_first = visibility_frame[first_joint].sum()
        visible_timestamps_mid = visibility_frame[mid_joint].sum()
        visible_timestamps_end = visibility_frame[end_joint].sum()

        # Check if all joints have the required number of visible timestamps
        if (
            visible_timestamps_first >= min_visible_timestamps
            and visible_timestamps_mid >= min_visible_timestamps
            and visible_timestamps_end >= min_visible_timestamps
        ):
            visible_columns.append(col)

    return visible_columns


def plot_fourier_magnitude(
    fourier_frame, visibility_threshold: float = 0.5, visibility_percentage_threshold: float = 0.5
):
    """Plot Fourier magnitude data as an interactive bar plot.

    Args:
        fourier_frame (FourierSeries): FourierSeries frame.
            Columns represent joint combinations, and the index represents frequency.

    Returns:
        go.Figure: Interactive bar plot of Fourier magnitude data.
    """
    logging.info("Plotting Fourier magnitude bar plot")
    fig = go.Figure()

    visible_angles = visible_angles_only(
        fourier_frame, visibility_threshold, visibility_percentage_threshold
    )
    magnitude_frame = fourier_frame.magnitude.loc[:, visible_angles]

    for column in magnitude_frame.columns:
        (first, mid, end), formatted_joint_name = format_joint_name(column)
        fig.add_trace(
            go.Bar(
                x=magnitude_frame.index,
                y=magnitude_frame[column],
                name=formatted_joint_name,
                hovertemplate=(
                    f"first: {first}<br>"
                    f"mid: {mid}<br>"
                    f"end: {end}<br>"
                    "frequency: %{x:.2e}<br>"
                    "magnitude: %{y:.2f}<extra></extra>"
                ),
                text=formatted_joint_name,
            )
        )

    fig.update_layout(
        width=1280,
        height=720,
        title="Fourier Frequency Domain : Magnitude",
        xaxis_title="Frequency (s⁻¹)",
        yaxis_title="Magnitude (°)",
        legend_title="Body Angles",
        hovermode="closest",
    )

    return fig


def plot_fourier_phase(
    fourier_frame, visibility_threshold: float = 0.5, visibility_percentage_threshold: float = 0.5
):
    """Plot Fourier phase data as an interactive bar plot.

    Args:
        fourier_frame (FourierSeries): FourierSeries frame.
            Columns represent joint combinations, and the index represents frequency.

    Returns:
        go.Figure: Interactive bar plot of Fourier phase data.
    """
    logging.info("Plotting Fourier phase bar plot")
    fig = go.Figure()

    visible_angles = visible_angles_only(
        fourier_frame, visibility_threshold, visibility_percentage_threshold
    )
    phase_frame = fourier_frame.phase.loc[:, visible_angles]

    for column in phase_frame.columns:
        (first, mid, end), formatted_joint_name = format_joint_name(column)
        fig.add_trace(
            go.Bar(
                x=phase_frame.index,
                y=phase_frame[column],
                name=formatted_joint_name,
                hovertemplate=(
                    f"first: {first}<br>"
                    f"mid: {mid}<br>"
                    f"end: {end}<br>"
                    "frequency: %{x:.2e}<br>"
                    "phase: %{y:.2f}°<extra></extra>"
                ),
                text=formatted_joint_name,
            )
        )

    fig.update_layout(
        width=1280,
        height=720,
        title="Fourier Frequency Domain : Phase",
        xaxis_title="Frequency (s⁻¹)",
        yaxis_title="Phase (°)",
        legend_title="Body Angles",
        hovermode="closest",
    )

    return fig
