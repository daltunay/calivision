import logging

import mediapipe as mp
import plotly.graph_objects as go

from ..features import JointSeries

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def plot_joint_series(
    joints_frame: JointSeries,
    visibility_threshold: float = 0.5,
) -> go.Figure:
    """Plot landmarks in 3D and visualize their connections.

    Args:
        joints_frame (JointSeries): Multi-index DataFrame of landmarks to plot.
        visibility_threshold (float, optional): Visibility threshold. Defaults to 0.5.

    Returns:
        go.Figure: Interactive 3D figure displaying the landmarks and connections.
    """
    logging.info("Plotting interactive 3D landmarks")

    # Create figure
    fig = go.Figure()

    # Extract the timestamps and initialize the visible status
    timestamps = joints_frame.index.unique(level="timestamp")
    num_timestamps = len(timestamps)
    visible_status = [False] * num_timestamps

    for i, timestamp in enumerate(timestamps):
        # Set the visibility status to True for the first frame
        if i == 0:
            visible_status[i] = True

        landmarks_subset = (
            joints_frame.loc[timestamp]
            .reset_index(name="value")
            .pivot(index="joint", columns="coordinate", values="value")
        )
        landmarks_subset.index.name = "joint"

        # Filter landmarks based on visibility thresholds
        landmarks_subset = landmarks_subset.loc[
            landmarks_subset["visibility"] >= visibility_threshold
        ]

        # Replace 'y' values with '-z' and 'z' values with '-y'
        y_values = landmarks_subset["y"].values
        z_values = landmarks_subset["z"].values
        landmarks_subset["y"] = -z_values
        landmarks_subset["z"] = -y_values

        # Add scatter trace for landmarks
        scatter_trace = go.Scatter3d(
            x=landmarks_subset["x"].values,
            y=landmarks_subset["y"].values,
            z=landmarks_subset["z"].values,
            mode="markers",
            text=landmarks_subset.index,
            hoverinfo="text",
            customdata=landmarks_subset[["x", "y", "z", "visibility"]],
            hovertemplate=(
                "joint: %{text}<br>"
                "x: %{customdata[0]:.4f}<br>"
                "y: %{customdata[1]:.4f}<br>"
                "z: %{customdata[2]:.4f}<br>"
                "visibility: %{customdata[3]:.2%}"
                "<extra></extra>"
            ),
            marker={"color": "red"},
            showlegend=False,
            visible=visible_status[i],
        )
        fig.add_trace(scatter_trace)

        # Create a list to store connections for plotting
        connections = {axis: [] for axis in ("x", "y", "z")}
        # Create a mapping of Mediapipe PoseLandmark values to their names
        MAPPING = {
            landmark_ref._value_: landmark_ref._name_
            for landmark_ref in mp.solutions.pose.PoseLandmark
        }
        for connection in mp.solutions.pose.POSE_CONNECTIONS:
            start_joint, end_joint = MAPPING[connection[0]], MAPPING[connection[1]]
            if all(joint in landmarks_subset.index for joint in (start_joint, end_joint)):
                start_landmark = landmarks_subset.loc[start_joint, ["x", "y", "z"]]
                end_landmark = landmarks_subset.loc[end_joint, ["x", "y", "z"]]
                for axis in ("x", "y", "z"):
                    connections[axis].extend([start_landmark[axis], end_landmark[axis], None])

        connection_trace = go.Scatter3d(
            x=connections["x"],
            y=connections["y"],
            z=connections["z"],
            mode="lines",
            line={"color": "black", "width": 5},
            name="connections",
            hoverinfo="none",
            showlegend=False,
            visible=visible_status[i],
        )
        fig.add_trace(connection_trace)

        # Update visible status for this frame
        visible_status[i] = True

    # Create steps for the slider
    steps = [
        {
            "method": "update",
            "args": [
                {"visible": [False] * (num_timestamps * 2)},
                {"title": f"Body landmarks at t={timestamps[i]}s (frame {i + 1})"},
            ],
            "label": f"{timestamps[i]:.2f}s",
        }
        for i in range(num_timestamps)
    ]

    # Toggle scatter and connections for the selected frame
    for i in range(num_timestamps):
        steps[i]["args"][0]["visible"][i * 2 : (i * 2) + 2] = [True, True]

    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "time: "},
            "pad": {"l": 50, "r": 50, "t": 50, "b": 10},
            "steps": steps,
        }
    ]

    # Update layout of the figure with sliders
    fig.update_layout(
        width=1280,
        height=720,
        sliders=sliders,
        title="Body landmarks at t=0s (frame 1)",
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        scene={
            "aspectmode": "cube",  # Set aspect mode to manual
            "xaxis": {"range": [-1, 1], "title": "x-axis"},
            "yaxis": {"range": [-1, 1], "title": "y-axis"},
            "zaxis": {"range": [-1, 1], "title": "z-axis"},
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 0.5}},
        },
    )

    return fig
