"""
optical_flow.py

Dense optical flow estimation using OpenCV.

Supported methods:
- "farneback" : baseline optical flow
- "dis"       : Dense Inverse Search (recommended)

Optical flow convention:
------------------------
Given two images I_src and I_dst, the flow F is defined such that:
    I_src(x) corresponds to I_dst(x + F(x))
"""

import cv2
import numpy as np


def compute_optical_flow_pair(img_src, img_dst, method="dis"):
    """
    Compute dense optical flow from img_src to img_dst.

    Args:
        img_src: source image (H, W, 3), uint8
        img_dst: destination image (H, W, 3), uint8
        method: "farneback" or "dis"

    Returns:
        flow: (H, W, 2) float32 displacement field
    """
    if method == "farneback":
        return _flow_farneback(img_src, img_dst)

    if method == "dis":
        return _flow_dis(img_src, img_dst)

    raise ValueError(f"Unknown optical flow method: {method}")


# =====================================================
# Farnebäck Optical Flow (baseline)
# =====================================================

def _flow_farneback(img_src, img_dst):
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_src,
        gray_dst,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=15,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    return flow.astype(np.float32)


# =====================================================
# DIS Optical Flow (recommended)
# =====================================================

def _flow_dis(img_src, img_dst):
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    dis = cv2.DISOpticalFlow_create(
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    )

    flow = dis.calc(gray_src, gray_dst, None)
    return flow.astype(np.float32)
