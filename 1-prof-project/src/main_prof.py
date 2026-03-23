"""
main.py

Reference-based optical flow logo propagation.

Pipeline:
1. Load video frames (frame_000.png ... frame_XXX.png)
2. Export original (non-edited) video
3. Select a reference frame
4. Load edited reference frame and logo mask
5. For each frame:
   - compute optical flow from current frame to reference frame
   - warp logo and mask from reference to current frame
   - recompose final frame
6. Save propagated frames and export edited video
"""

import os
import cv2
import numpy as np

from motion.optical_flow import compute_optical_flow_pair
from editing.warping import warp_image, warp_mask
from utils.io import load_frames, save_frames
from utils.video import frames_to_video


# =====================================================
# Configuration
# =====================================================

# Paths
INPUT_FRAMES_DIR = "data/input_frames"
EDIT_DIR = "edit"
OUTPUT_DIR = "output"

PROPAGATED_FRAMES_DIR = os.path.join(OUTPUT_DIR, "propagated_frames")

ORIGINAL_VIDEO_PATH = os.path.join(OUTPUT_DIR, "original_video.mp4")
EDITED_VIDEO_PATH = os.path.join(OUTPUT_DIR, "final_video.mp4")

# Reference frame index
# Must correspond to:
#   data/input_frames/frame_XXX.png
#   edit/modified_XXX.png
#   edit/logomask_XXX.png
REF_INDEX = 0

# Optical flow
FLOW_METHOD = "dis"      # "dis" (recommended) or "farneback"

# Video
FPS = 25


# =====================================================
# Main pipeline
# =====================================================

def main():
    print("=== Reference-Based Optical Flow Video Editing ===")

    # -------------------------------------------------
    # 1. Load input frames
    # -------------------------------------------------
    print("[1] Loading input frames...")
    frames = load_frames(INPUT_FRAMES_DIR)
    n_frames = len(frames)

    assert n_frames > 0, "No input frames found."
    assert 0 <= REF_INDEX < n_frames, "Invalid reference frame index."

    print(f"    Loaded {n_frames} frames")
    print(f"    Reference frame index: {REF_INDEX}")

    reference_frame = frames[REF_INDEX]

    # -------------------------------------------------
    # 2. Export original (non-edited) video
    # -------------------------------------------------
    print("[2] Exporting original video (no editing)...")
    frames_to_video(
        INPUT_FRAMES_DIR,
        ORIGINAL_VIDEO_PATH,
        fps=FPS
    )

    # -------------------------------------------------
    # 3. Load edited reference frame and logo mask
    # -------------------------------------------------
    print("[3] Loading edited reference frame and logo mask...")

    edited_path = os.path.join(
        EDIT_DIR, f"modified_{REF_INDEX:03d}.png"
    )
    mask_path = os.path.join(
        EDIT_DIR, f"logomask_{REF_INDEX:03d}.png"
    )

    edited_ref = cv2.imread(edited_path)
    assert edited_ref is not None, f"Could not load {edited_path}"

    logo_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert logo_mask is not None, f"Could not load {mask_path}"

    # Normalize mask to [0, 1] and add channel dimension
    logo_mask = (logo_mask.astype(np.float32) / 255.0)[..., None]

    # Extract logo content from edited reference frame
    logo_ref = edited_ref * logo_mask

    print("    Edited reference and mask loaded")

    # -------------------------------------------------
    # 4. Propagate logo using reference-based optical flow
    # -------------------------------------------------
    print("[4] Propagating logo through the video...")

    propagated_frames = []

    for t, frame in enumerate(frames):
        print(f"    Processing frame {t:03d}")

        if t == REF_INDEX:
            composed = (
                logo_mask * logo_ref +
                (1.0 - logo_mask) * frame
            )
            propagated_frames.append(composed.astype(np.uint8))
            continue

        # Compute optical flow: frame_t -> reference_frame
        flow_t_to_ref = compute_optical_flow_pair(
            frame,
            reference_frame,
            FLOW_METHOD
        )

        # Warp logo and mask from reference to current frame
        logo_t = warp_image(logo_ref, flow_t_to_ref)
        mask_t = warp_mask(logo_mask, flow_t_to_ref)

        # Recompose final frame
        composed = (
            mask_t * logo_t +
            (1.0 - mask_t) * frame
        )

        propagated_frames.append(composed.astype(np.uint8))

    # -------------------------------------------------
    # 5. Save propagated frames
    # -------------------------------------------------
    print("[5] Saving propagated frames...")
    save_frames(propagated_frames, PROPAGATED_FRAMES_DIR)

    # -------------------------------------------------
    # 6. Export edited video
    # -------------------------------------------------
    print("[6] Exporting edited video...")
    frames_to_video(
        PROPAGATED_FRAMES_DIR,
        EDITED_VIDEO_PATH,
        fps=FPS
    )

    print("=== Done ===")
    print(f"Results saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
