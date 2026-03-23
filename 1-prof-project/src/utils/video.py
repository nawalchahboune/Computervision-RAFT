"""
video.py

Video utilities for converting image sequences to video files.
"""

import os
import cv2
from typing import List


def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: int = 25,
    codec: str = "mp4v"
) -> None:
    """
    Convert a directory of frames into a video.

    Args:
        frames_dir: directory containing frame_XXX.png images
        output_path: path to output video file
        fps: frames per second
        codec: fourcc codec (default: mp4v)
    """
    assert os.path.isdir(frames_dir), f"Frames directory not found: {frames_dir}"

    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(".png")
    )

    assert len(frame_files) > 0, f"No frames found in {frames_dir}"

    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    assert first_frame is not None, "Failed to read first frame"

    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    assert writer.isOpened(), "Failed to open video writer"

    for fname in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        assert frame is not None, f"Failed to read frame: {fname}"

        if frame.shape[:2] != (height, width):
            raise ValueError("Frame size mismatch in video sequence")

        writer.write(frame)

    writer.release()
