from typing import Iterable, Tuple

import numpy as np


def calculate_angle(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    """Calculate the angle between three points.

    Args:
        a (Iterable[float]): The first point as an iterable of floats.
        b (Iterable[float]): The second point as an iterable of floats.
        c (Iterable[float]): The third point as an iterable of floats.

    Returns:
        float: The angle between the three points in degrees.
    """
    # Convert input points to numpy arrays for numerical calculations
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # Calculate vectors ba and bc
    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle between vectors ba and bc
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Calculate the angle in radians
    angle_rad = np.arccos(cos_angle)

    # Convert the angle from radians to degrees and return
    return np.degrees(angle_rad)


def format_joint_name(column: Tuple[str]) -> str:
    """Format a joint name for display purposes.

    Args:
        column (tuple): Tuple representing a joint name.

    Returns:
        str: Formatted joint name.
    """
    first, mid, end = column
    joint_name = f"({first}, {mid}, {end})"
    formatted_joint_name = joint_name.replace("LEFT_", "l_").replace("RIGHT_", "r_").lower()
    return (first, mid, end), formatted_joint_name


def calculate_new_dimensions(original_width, original_height, width, height):
    if width is None and height is None:
        new_width, new_height = original_width, original_height
    elif width is not None and height is None:
        new_width = width
        new_height = int((new_width / original_width) * original_height)
    elif height is not None and width is None:
        new_height = height
        new_width = int((new_height / original_height) * original_width)
    else:
        # Calculate dimensions based on aspect ratio
        aspect_ratio = original_width / original_height
        if width / height > aspect_ratio:
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            new_height = height
            new_width = int(height * aspect_ratio)
    return new_width, new_height
