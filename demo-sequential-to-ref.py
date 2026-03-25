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
from scipy.interpolate import RegularGridInterpolator

DEVICE = 'cpu'
ITERS  = 128

def main(model_path, frames_path, mask_ref_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Initialisation RAFT
    args = {'model': model_path, 'small': False, 'mixed_precision': False,
            'alternate_corr': False, 'dropout': 0.0}
    class Map(dict):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw); self.__dict__ = self
    args_obj = Map(args)
    model = torch.nn.DataParallel(RAFT(args_obj))
    model.load_state_dict(torch.load(args_obj.model, map_location=DEVICE))
    model = model.module.to(DEVICE).eval()

    # Chargement des frames
    images_paths = sorted(glob.glob(os.path.join(frames_path, "*.png")))
    if not images_paths:
        images_paths = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
    if not images_paths:
        raise ValueError(f"Aucune image trouvée dans {frames_path}")

    ref_name = os.path.splitext(os.path.basename(mask_ref_path))[0]
    ref_path = next((p for p in images_paths if ref_name in os.path.basename(p)), None)
    if ref_path is None:
        raise ValueError(f"Frame de référence '{ref_name}' introuvable dans {frames_path}")
    REF_INDEX = images_paths.index(ref_path)

    edited_ref = np.array(Image.open('test-data-mask/ref.png')).astype(np.float32)
    if edited_ref.shape[2] == 4:
        edited_ref = edited_ref[:, :, :3]

    mask_ref = cv2.imread(mask_ref_path, cv2.IMREAD_GRAYSCALE)
    _, mask_ref = cv2.threshold(mask_ref, 1, 255, cv2.THRESH_BINARY)
    mask_ref_float = mask_ref.astype(np.float32) / 255.0
    h, w = mask_ref.shape

    # Centre initial
    ys, xs = np.where(mask_ref > 0)
    cx, cy = float(xs.mean()), float(ys.mean())

    ys_grid = np.arange(h)
    xs_grid = np.arange(w)
    identity_x, identity_y = np.meshgrid(np.arange(w), np.arange(h))

    # Flow cumulatif backward D initialisé à 0
    D = np.zeros((h, w, 2), dtype=np.float32)

    prev_torch = torch.from_numpy(
        np.array(Image.open(images_paths[0]))).permute(2,0,1).float()[None].to(DEVICE)
    padder   = InputPadder(prev_torch.shape)
    prev_pad = padder.pad(prev_torch)[0]

    print(f"[Sequential-to-ref] Propagation sur {len(images_paths)} frames...")

    with torch.no_grad():
        for t, im_path in enumerate(images_paths):

            curr_torch = torch.from_numpy(
                np.array(Image.open(im_path))).permute(2,0,1).float()[None].to(DEVICE)
            curr_pad = padder.pad(curr_torch)[0]

            if t == 0:
                binary_mask   = mask_ref.copy()
                edited_warped = edited_ref.copy()
            else:
                # Backward flow
                _, flow_bwd_up = model(curr_pad, prev_pad, iters=ITERS, test_mode=True)
                u_backward = padder.unpad(flow_bwd_up)[0].permute(1,2,0).cpu().numpy()

                # Interpolateurs pour u_x et u_y
                interp_x = RegularGridInterpolator(
                    (ys_grid, xs_grid), u_backward[:,:,0],
                    method='linear', bounds_error=False, fill_value=0.0)
                interp_y = RegularGridInterpolator(
                    (ys_grid, xs_grid), u_backward[:,:,1],
                    method='linear', bounds_error=False, fill_value=0.0)

                # Mise à jour du centre
                pt  = np.array([[cy, cx]])
                cx -= interp_x(pt)[0]
                cy -= interp_y(pt)[0]

                # Composition backward du flow cumulatif :
                map_x_bwd = (identity_x + u_backward[:,:,0]).astype(np.float32)
                map_y_bwd = (identity_y + u_backward[:,:,1]).astype(np.float32)

                D_prev_warped_x = cv2.remap(D[:,:,0], map_x_bwd, map_y_bwd, cv2.INTER_LINEAR)
                D_prev_warped_y = cv2.remap(D[:,:,1], map_x_bwd, map_y_bwd, cv2.INTER_LINEAR)

                D[:,:,0] = u_backward[:,:,0] + D_prev_warped_x
                D[:,:,1] = u_backward[:,:,1] + D_prev_warped_y

                # Warp du masque depuis I0 via D cumulatif
                # D est backward (In→I0), donc on warp le masque ref vers It
                map_x_fwd = (identity_x - D[:,:,0]).astype(np.float32)
                map_y_fwd = (identity_y - D[:,:,1]).astype(np.float32)

                warped_mask = cv2.remap(mask_ref_float, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)

                _, binary_mask = cv2.threshold(
                    (warped_mask * 255).astype(np.uint8), 180, 255, cv2.THRESH_BINARY)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                if num_labels > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    clean = np.zeros_like(binary_mask)
                    clean[labels == largest] = 255
                    binary_mask = clean

                edited_warped = cv2.remap(edited_ref, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)
                prev_pad = curr_pad

            frame_orig  = np.array(Image.open(im_path)).astype(np.float32)
            mask_alpha  = (binary_mask / 255.0)[..., None]
            frame_final = mask_alpha * edited_warped + (1 - mask_alpha) * frame_orig

            cv2.imwrite(os.path.join(output_dir, f"mask_{t:04d}.png"), binary_mask)
            cv2.imwrite(os.path.join(output_dir, f"edited_{t:04d}.png"), frame_final.astype(np.uint8))
            print(f"Frame {t:04d}  centre=({cx:.1f},{cy:.1f})")


if __name__ == '__main__':
    main('models/raft-things.pth', 'test-data',
         'test-data-mask/00000.png', 'output/sequential_to_ref')