"""
io.py

Input / output utilities for image sequences.

This module handles loading and saving video frames stored as image files.
"""

import os
import cv2
from typing import List


def load_frames(folder: str) -> List:
    """
    Load a sequence of frames from a directory.

    Frames must be named frame_XXX.png (e.g., frame_000.png).

    Args:
        folder: path to the directory containing frames

    Returns:
        List of images as numpy arrays (H, W, 3)
    """
    assert os.path.isdir(folder), f"Folder not found: {folder}"

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    )

    assert len(files) > 0, f"No PNG files found in {folder}"

    frames = []
    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        assert img is not None, f"Failed to load image: {path}"
        frames.append(img)

    return frames


def save_frames(frames: List, folder: str) -> None:
    """
    Save a sequence of frames to a directory.

    Frames are saved as frame_XXX.png.

    Args:
        frames: list of images (H, W, 3)
        folder: output directory
    """
    os.makedirs(folder, exist_ok=True)

    for i, frame in enumerate(frames):
        filename = f"frame_{i:03d}.png"
        path = os.path.join(folder, filename)

        success = cv2.imwrite(path, frame)
        assert success, f"Failed to save image: {path}"
