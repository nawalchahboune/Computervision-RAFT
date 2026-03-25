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

DEVICE = 'cpu'  # Change en 'cuda' si tu as une carte NVIDIA
# ITERS = 96 
ITERS = 128

def main(model_path, frames_path, mask_ref_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialisation RAFT
    args = {'model': model_path, 'small': False, 'mixed_precision': False, 'alternate_corr': False, 'dropout': 0.0}
    class Map(dict):
        def __init__(self, *args, **kwargs):
            super(Map, self).__init__(*args, **kwargs)
            self.__dict__ = self
    args_obj = Map(args)
    model = torch.nn.DataParallel(RAFT(args_obj))
    model.load_state_dict(torch.load(args_obj.model, map_location=DEVICE))
    model = model.module.to(DEVICE).eval()

    # 2. Chargement des ressources
    images_paths = sorted(glob.glob(os.path.join(frames_path, "*.png")))
    if len(images_paths) == 0:
        images_paths = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
    if len(images_paths) == 0:
        raise ValueError(f"Aucune image trouvée dans {frames_path} (ni .png ni .jpg)")

    # Déduire le nom de la frame de référence à partir du masque
    ref_name = os.path.splitext(os.path.basename(mask_ref_path))[0]  # ex: '00027'
    ref_path = None
    for p in images_paths:
        if ref_name in os.path.basename(p):
            ref_path = p
            break
    if ref_path is None:
        raise ValueError(f"Aucune image correspondant à {ref_name} dans {frames_path}")
    REF_INDEX = images_paths.index(ref_path)

    # Charger l'image éditée
    edited_ref = np.array(Image.open('test-data-mask/ref.png')).astype(np.float32)
    if edited_ref.shape[2] == 4:
        edited_ref = edited_ref[:, :, :3]

    # Charger le masque de référence (noir et blanc)
    mask_ref = cv2.imread(mask_ref_path, cv2.IMREAD_GRAYSCALE)
    _, mask_ref = cv2.threshold(mask_ref, 1, 255, cv2.THRESH_BINARY)
    mask_ref_float = mask_ref.astype(np.float32) / 255.0
    h, w = mask_ref.shape

    # Préparer la frame de référence pour RAFT
    img_ref_torch = torch.from_numpy(np.array(Image.open(images_paths[REF_INDEX]))).permute(2,0,1).float()[None].to(DEVICE)
    padder = InputPadder(img_ref_torch.shape)
    img_ref_pad = padder.pad(img_ref_torch)[0]

    identity_x, identity_y = np.meshgrid(np.arange(w), np.arange(h))

    print(f"Démarrage de la propagation avec Forward-Backward Check sur {len(images_paths)} frames...")

    with torch.no_grad():
        for t, im_path in enumerate(images_paths):

            # Charger frame actuelle
            img_t = torch.from_numpy(np.array(Image.open(im_path))).permute(2,0,1).float()[None].to(DEVICE)
            img_t_pad = padder.pad(img_t)[0]

            # Calcul du Flow (Forward & Backward)
            # Flow: Frame_T -> Frame_Ref
            _, flow_fwd_up = model(img_t_pad, img_ref_pad, iters=ITERS, test_mode=True)
            flow_fwd = padder.unpad(flow_fwd_up)[0].permute(1, 2, 0).cpu().numpy()

            # Flow: Frame_Ref -> Frame_T (pour la cohérence)
            _, flow_bwd_up = model(img_ref_pad, img_t_pad, iters=ITERS, test_mode=True)
            flow_bwd = padder.unpad(flow_bwd_up)[0].permute(1, 2, 0).cpu().numpy()

            # Lissage du Flow 
            flow_fwd[:,:,0] = cv2.GaussianBlur(flow_fwd[:,:,0], (15, 15), 0)
            flow_fwd[:,:,1] = cv2.GaussianBlur(flow_fwd[:,:,1], (15, 15), 0)

            # Forward-Backward Consistency Check
            map_x_fwd = (identity_x + flow_fwd[:,:,0]).astype(np.float32)
            map_y_fwd = (identity_y + flow_fwd[:,:,1]).astype(np.float32)
            
            # Warp du flow backward pour le comparer au point de départ
            flow_bwd_warped_x = cv2.remap(flow_bwd[:,:,0], map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)
            flow_bwd_warped_y = cv2.remap(flow_bwd[:,:,1], map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)
            
            # Calcul de l'erreur de retour (L2 distance)
            err_x = flow_fwd[:,:,0] + flow_bwd_warped_x
            err_y = flow_fwd[:,:,1] + flow_bwd_warped_y
            err_dist = np.sqrt(err_x**2 + err_y**2)
            
            # Créer un masque de confiance (Pixels qui bougent de façon logique)
            confidence_mask = (err_dist < 3.0).astype(np.float32) # Seuil de 2 pixels d'erreur max

            # Warping et Nettoyage
            warped_mask = cv2.remap(mask_ref_float, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)
            
            # Appliquer la confiance et le seuil
            combined_mask = warped_mask * confidence_mask
            _, binary_mask = cv2.threshold((combined_mask * 255).astype(np.uint8), 180, 255, cv2.THRESH_BINARY)

            # Ne garder que la plus grande composante
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                final_mask = np.zeros_like(binary_mask)
                final_mask[labels == largest_label] = 255
                binary_mask = final_mask

            # Composition Finale 
            edited_ref_warped = cv2.remap(edited_ref, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)
            frame_orig = np.array(Image.open(im_path)).astype(np.float32)
            
            mask_alpha = (binary_mask / 255.0)[..., None]
            frame_final = mask_alpha * edited_ref_warped + (1 - mask_alpha) * frame_orig

            # Sauvegarde
            cv2.imwrite(os.path.join(output_dir, f"mask_{t:04d}.png"), binary_mask)
            cv2.imwrite(os.path.join(output_dir, f"edited_{t:04d}.png"), frame_final.astype(np.uint8))
            print(f"Frame {t} traitée (Erreur moyenne flow: {np.mean(err_dist):.2f})")

if __name__ == '__main__':
    main('models/raft-things.pth', 'test-data', 'test-data-mask/00000.png', 'output/propagation-paragliding')