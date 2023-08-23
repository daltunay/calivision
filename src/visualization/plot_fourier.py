import logging

import plotly.graph_objects as go

from ..utils import format_joint_name

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def plot_fourier_magnitude(fourier_frame):
    """Plot Fourier magnitude data as an interactive bar plot.

    Args:
        fourier_frame (FourierSeries): FourierSeries frame.
            Columns represent joint combinations, and the index represents frequency.

    Returns:
        go.Figure: Interactive bar plot of Fourier magnitude data.
    """
    logging.info("Plotting Fourier magnitude bar plot")
    fig = go.Figure()

    for column in fourier_frame.magnitude.columns:
        (first, mid, end), formatted_joint_name = format_joint_name(column)
        fig.add_trace(
            go.Bar(
                x=fourier_frame.magnitude.index,
                y=fourier_frame.magnitude[column],
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


def plot_fourier_phase(fourier_frame):
    """Plot Fourier phase data as an interactive bar plot.

    Args:
        fourier_frame (FourierSeries): FourierSeries frame.
            Columns represent joint combinations, and the index represents frequency.

    Returns:
        go.Figure: Interactive bar plot of Fourier phase data.
    """
    logging.info("Plotting Fourier phase bar plot")
    fig = go.Figure()

    for column in fourier_frame.phase.columns:
        (first, mid, end), formatted_joint_name = format_joint_name(column)
        fig.add_trace(
            go.Bar(
                x=fourier_frame.phase.index,
                y=fourier_frame.phase[column],
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
