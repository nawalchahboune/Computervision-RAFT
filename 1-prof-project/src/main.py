import os
import cv2
import numpy as np
import torch

# On ne garde que les imports utiles pour le flux et la vidéo
from motion.optical_flow import compute_optical_flow_pair
from utils.io import load_frames, save_frames
from utils.video import frames_to_video

# =====================================================
# Configuration
# =====================================================

INPUT_FRAMES_DIR = "../data/input_frames"
OUTPUT_DIR = "../output"
FLOW_VIS_DIR = "../output/flow_visualization"
FLOW_VIDEO_PATH = os.path.join(OUTPUT_DIR, "optical_flow_video.mp4")

FLOW_METHOD = "dis"      # "dis" ou "farneback" (en attendant FlowNet !)
FPS = 25

# =====================================================
# Fonction de Visualisation (Vecteurs -> Couleurs)
# =====================================================


def flow2rgb(flow_map, max_value=None):
 
    """
    Convert optical flow to RGB visualization using a color wheel.
    
    Args:
        flow_map: Optical flow map (2 x H x W) as a PyTorch tensor or NumPy array.
        max_value: Maximum flow value for normalization. If None, it will be computed from the flow map.
    
    Returns:
        RGB visualization of the optical flow as a NumPy array (H x W x 3).
    """
    print(f"Flow shape: {flow_map.shape}")
    assert flow_map.ndim == 3 and flow_map.shape[0] == 2, \
        f"flow2rgb attend (2,H,W) mais reçoit {flow_map.shape}"

    if isinstance(flow_map, torch.Tensor):
        flow_map_np = flow_map.detach().cpu().numpy()
    else:
        flow_map_np = flow_map

    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float("nan")
    rgb_map = np.ones((3, h, w)).astype(np.float32)

    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max() + 1e-5)

    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]

    rgb_map = rgb_map.clip(0, 1)
    rgb_map = (rgb_map * 255).astype(np.uint8)
    rgb_map = np.transpose(rgb_map, (1, 2, 0))  # Convert to H x W x 3
    return rgb_map
   
   
   
def flow_to_color(flow):
    """
    Convertit un champ de flux optique (x, y) en image RGB (HSV).
    - La direction du mouvement définit la couleur (Hue).
    - La magnitude du mouvement définit l'intensité (Value).
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255 # Saturation au maximum

    # Extraction de la magnitude et de l'angle des vecteurs
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Conversion de l'angle en Teinte (Hue) et Magnitude en Valeur
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Retour en BGR (le format par défaut d'OpenCV)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

# =====================================================
# Main pipeline
# =====================================================

def main():
    print("=== Optical Flow Computation & Visualization ===")

    # 1. Load input frames
    print("[1] Loading input frames...")
    frames = load_frames(INPUT_FRAMES_DIR)
    n_frames = len(frames)

    assert n_frames > 1, "Il faut au moins 2 images pour calculer un mouvement."
    print(f"    Loaded {n_frames} frames")

    # 2. Compute Flow sequentially
    print("[2] Computing optical flow sequentially...")
    vis_frames = []

    for t in range(n_frames - 1):
        print(f"    Processing flow between frame {t:03d} and {t+1:03d}")
        frame1 = frames[t]
        frame2 = frames[t+1]

        # Calcul du flux optique : frame1 -> frame2
        flow = compute_optical_flow_pair(frame1, frame2, FLOW_METHOD)
        # CORRECTION OBLIGATOIRE - une seule fois ici
        if flow.ndim == 3 and flow.shape[-1] == 2:  # (H,W,2) → (2,H,W)
            flow = np.transpose(flow, (2, 0, 1))
            print(f"    Converted to CHW: {flow.shape}")

        flow_color_img = flow2rgb(flow)  # maintenant c'est (2,H,W) → OK !
        # flow_color_img = flow_to_color(flow)
        vis_frames.append(flow_color_img)

    # Note: On ajoute une image noire à la fin pour avoir le même nombre de frames
    # car on a calculé n-1 flux pour n images.
    vis_frames.append(np.zeros_like(vis_frames[-1]))

    # 3. Save propagated frames
    print("[3] Saving flow visualization frames...")
    os.makedirs(FLOW_VIS_DIR, exist_ok=True)
    save_frames(vis_frames, FLOW_VIS_DIR)

    # # 4. Export video
    # print("[4] Exporting flow video...")
    # frames_to_video(FLOW_VIS_DIR, FLOW_VIDEO_PATH, fps=FPS)

    # print("=== Done ===")
    # print(f"Results saved in: {os.path.abspath(OUTPUT_DIR)}")
    # print("FLOW_VIS_DIR:", os.path.abspath(FLOW_VIS_DIR))
    # print("Fichiers présents:", os.listdir(FLOW_VIS_DIR) if os.path.exists(FLOW_VIS_DIR) else "DOSSIER N'EXISTE PAS")


if __name__ == "__main__":
    main()