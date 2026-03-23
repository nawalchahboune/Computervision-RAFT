"""
visualization.py

Visualization utilities for dense optical flow fields.

This module provides standard tools to visualize optical flow magnitude
and direction, and to overlay motion information on image frames.
"""

import cv2
import numpy as np
import os


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """
    Convert a dense optical flow field to an HSV visualization.

    Args:
        flow: optical flow (H, W, 2)

    Returns:
        BGR image representing flow direction and magnitude
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2      # direction
    hsv[..., 1] = 255                        # saturation
    hsv[..., 2] = cv2.normalize(
        mag, None, 0, 255, cv2.NORM_MINMAX
    )

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def overlay_flow(
    frame: np.ndarray,
    flow: np.ndarray,
    alpha: float = 0.7
) -> np.ndarray:
    """
    Overlay optical flow visualization on an image frame.

    Args:
        frame: original frame (H, W, 3)
        flow: optical flow (H, W, 2)
        alpha: blending factor

    Returns:
        blended image
    """
    flow_vis = flow_to_hsv(flow)
    blended = cv2.addWeighted(frame, alpha, flow_vis, 1 - alpha, 0)
    return blended


def draw_flow_vectors(
    frame: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """
    Draw sparse optical flow vectors on an image.

    Args:
        frame: original frame (H, W, 3)
        flow: optical flow (H, W, 2)
        step: sampling step for vectors
        color: vector color (BGR)

    Returns:
        image with flow vectors drawn
    """
    h, w = frame.shape[:2]
    output = frame.copy()

    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            end_point = (int(x + fx), int(y + fy))
            cv2.arrowedLine(
                output,
                (x, y),
                end_point,
                color,
                1,
                tipLength=0.3
            )

    return output


def save_flow_visualization(
    flow: np.ndarray,
    frame: np.ndarray,
    output_dir: str,
    index: int
) -> None:
    """
    Save flow visualizations to disk.

    Args:
        flow: optical flow (H, W, 2)
        frame: corresponding video frame
        output_dir: directory where images are saved
        index: frame index
    """
    os.makedirs(output_dir, exist_ok=True)

    hsv_img = flow_to_hsv(flow)
    overlay_img = overlay_flow(frame, flow)
    vector_img = draw_flow_vectors(frame, flow)

    cv2.imwrite(
        os.path.join(output_dir, f"flow_hsv_{index:04d}.png"),
        hsv_img
    )
    cv2.imwrite(
        os.path.join(output_dir, f"flow_overlay_{index:04d}.png"),
        overlay_img
    )
    cv2.imwrite(
        os.path.join(output_dir, f"flow_vectors_{index:04d}.png"),
        vector_img
    )
