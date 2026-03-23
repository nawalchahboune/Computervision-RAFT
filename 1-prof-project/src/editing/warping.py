"""
Warping utilities based on optical flow.

This module provides functions to warp an image (or a video frame)
using a dense optical flow field.

Conventions:
- Images are NumPy arrays of shape (H, W, C) or (H, W)
- Optical flow is a NumPy array of shape (H, W, 2)
  where flow[y, x] = (dx, dy)
"""

import numpy as np
import cv2


def warp_image(image, flow, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT):
    """
    Warp an image using a dense optical flow field.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, C)
    flow : np.ndarray
        Optical flow of shape (H, W, 2), where flow[..., 0] is dx
        and flow[..., 1] is dy
    interpolation : int, optional
        OpenCV interpolation flag (default: cv2.INTER_LINEAR)
    border_mode : int, optional
        OpenCV border mode (default: cv2.BORDER_REFLECT)

    Returns
    -------
    warped : np.ndarray
        Warped image, same shape as input image
    """

    if image.ndim not in [2, 3]:
        raise ValueError("image must have shape (H, W) or (H, W, C)")

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape (H, W, 2)")

    h, w = flow.shape[:2]

    # Create pixel coordinate grid
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Compute remapping coordinates
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    # Apply remapping
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=border_mode,
    )

    return warped


def warp_mask(mask, flow):
    """
    Warp a binary or soft mask using nearest-neighbor interpolation.

    Output mask is float32 in [0, 1] with shape (H, W, 1),
    ready for alpha blending with RGB images.
    """

    warped = warp_image(
        mask,
        flow,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
    )

    # Convert to float in [0, 1]
    warped = warped.astype(np.float32)
    if warped.max() > 1.0:
        warped /= 255.0

    # Ensure channel dimension for broadcasting
    if warped.ndim == 2:
        warped = warped[..., None]  # (H, W, 1)

    return warped

def warp_sequence(frames, flows):
    """
    Warp a sequence of frames using a sequence of optical flows.

    frames[t] is warped using flows[t] (typically flow from t -> t+1).

    Parameters
    ----------
    frames : list or np.ndarray
        List/array of frames, each of shape (H, W, C)
    flows : list or np.ndarray
        List/array of flows, each of shape (H, W, 2)

    Returns
    -------
    warped_frames : list
        List of warped frames
    """

    if len(frames) != len(flows):
        raise ValueError("frames and flows must have the same length")

    warped_frames = []
    for frame, flow in zip(frames, flows):
        warped_frames.append(warp_image(frame, flow))

    return warped_frames
