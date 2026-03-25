import sys
sys.path.append('core')
import os
import cv2
import numpy as np
import torch
import glob
from raft import RAFT
from utils.utils import InputPadder
from PIL import Image

DEVICE = 'cpu'
ITERS  = 128

# ──────────────────────────────────────────────────────────────
# MÉTHODE 1a — PAIRWISE  (Approach 1 : single-point lookup)
# Flow calculé entre In-1 → In à chaque step.
# Le centre du masque est mis à jour par lecture du vecteur de flow
# au pixel le plus proche du centre courant.
# ✓ Rapide  ✗ Drift accumule sur longues séquences
# ──────────────────────────────────────────────────────────────

def warp_point(cx, cy, flow):
    """Déplace (cx,cy) par lecture du vecteur flow au pixel le plus proche."""
    h, w = flow.shape[:2]
    xi = int(np.clip(round(cx), 0, w - 1))
    yi = int(np.clip(round(cy), 0, h - 1))
    dx, dy = flow[yi, xi]
    return cx + dx, cy + dy


def main(model_path, frames_path, mask_ref_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialisation RAFT
    args = {'model': model_path, 'small': False, 'mixed_precision': False,
            'alternate_corr': False, 'dropout': 0.0}
    class Map(dict):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw); self.__dict__ = self
    args_obj = Map(args)
    model = torch.nn.DataParallel(RAFT(args_obj))
    model.load_state_dict(torch.load(args_obj.model, map_location=DEVICE))
    model = model.module.to(DEVICE).eval()

    # 2. Chargement des frames
    images_paths = sorted(glob.glob(os.path.join(frames_path, "*.png")))
    if not images_paths:
        images_paths = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
    if not images_paths:
        raise ValueError(f"Aucune image trouvée dans {frames_path}")

    # Frame de référence déduite depuis le nom du masque
    ref_name = os.path.splitext(os.path.basename(mask_ref_path))[0]
    ref_path = next((p for p in images_paths if ref_name in os.path.basename(p)), None)
    if ref_path is None:
        raise ValueError(f"Frame de référence '{ref_name}' introuvable dans {frames_path}")
    REF_INDEX = images_paths.index(ref_path)

    # Image éditée et masque de référence
    edited_ref = np.array(Image.open('test-data-mask/ref.png')).astype(np.float32)
    if edited_ref.shape[2] == 4:
        edited_ref = edited_ref[:, :, :3]

    mask_ref = cv2.imread(mask_ref_path, cv2.IMREAD_GRAYSCALE)
    _, mask_ref = cv2.threshold(mask_ref, 1, 255, cv2.THRESH_BINARY)
    mask_ref_float = mask_ref.astype(np.float32) / 255.0
    h, w = mask_ref.shape

    # Centre initial du masque
    ys, xs = np.where(mask_ref > 0)
    cx, cy = float(xs.mean()), float(ys.mean())
    print(f"Centre initial du masque : ({cx:.1f}, {cy:.1f})")

    # Première frame pour RAFT pairwise
    prev_torch = torch.from_numpy(
        np.array(Image.open(images_paths[0]))).permute(2,0,1).float()[None].to(DEVICE)
    padder   = InputPadder(prev_torch.shape)
    prev_pad = padder.pad(prev_torch)[0]

    identity_x, identity_y = np.meshgrid(np.arange(w), np.arange(h))

    # Masque courant (warpé au fil des frames)
    current_mask_float = mask_ref_float.copy()

    print(f"[Pairwise-A1] Propagation sur {len(images_paths)} frames...")

    with torch.no_grad():
        for t, im_path in enumerate(images_paths):

            curr_torch = torch.from_numpy(
                np.array(Image.open(im_path))).permute(2,0,1).float()[None].to(DEVICE)
            curr_pad = padder.pad(curr_torch)[0]

            if t == 0:
                binary_mask   = mask_ref.copy()
                edited_warped = edited_ref.copy()
            else:
                # Flow In-1 → In
                _, flow_up = model(prev_pad, curr_pad, iters=ITERS, test_mode=True)
                flow = padder.unpad(flow_up)[0].permute(1,2,0).cpu().numpy()

                # Mise à jour du centre (single-point)
                cx, cy = warp_point(cx, cy, flow)

                # Warp du masque via le flow courant
                map_x = (identity_x + flow[:,:,0]).astype(np.float32)
                map_y = (identity_y + flow[:,:,1]).astype(np.float32)
                current_mask_float = cv2.remap(current_mask_float, map_x, map_y, cv2.INTER_LINEAR)

                _, binary_mask = cv2.threshold(
                    (current_mask_float * 255).astype(np.uint8), 180, 255, cv2.THRESH_BINARY)

                # Plus grande composante connexe
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                if num_labels > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    clean = np.zeros_like(binary_mask)
                    clean[labels == largest] = 255
                    binary_mask = clean

                edited_warped = cv2.remap(edited_ref, map_x, map_y, cv2.INTER_LINEAR)
                prev_pad = curr_pad

            # Composition finale
            frame_orig  = np.array(Image.open(im_path)).astype(np.float32)
            mask_alpha  = (binary_mask / 255.0)[..., None]
            frame_final = mask_alpha * edited_warped + (1 - mask_alpha) * frame_orig

            cv2.imwrite(os.path.join(output_dir, f"mask_{t:04d}.png"), binary_mask)
            cv2.imwrite(os.path.join(output_dir, f"edited_{t:04d}.png"), frame_final.astype(np.uint8))
            print(f"Frame {t:04d}  centre=({cx:.1f},{cy:.1f})")


if __name__ == '__main__':
    main('models/raft-things.pth', 'test-data',
         'test-data-mask/00000.png', 'output/pairwise_A1')