import cv2
import numpy as np
import glob
import os

# === Configuration des chemins ===
# Ton masque initial parfait (Frame 0)
mask_init_path = 'mask_camel_half_res/00000.png' 
# Le dossier où RAFT a sauvegardé les .npy (T_n -> T_0)
flow_dir = 'output_viz'
# Dossier de sortie
output_dir = 'propagated_masks_camel_T0-Tn_half_res'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Charger le masque initial (Source Frame 0)
mask_initial = cv2.imread(mask_init_path, cv2.IMREAD_GRAYSCALE)
if mask_initial is None:
    raise ValueError(f"Impossible de charger le masque initial : {mask_init_path}")
mask_initial = (mask_initial > 0).astype(np.uint8) * 255
h, w = mask_initial.shape[:2]

# 2. Créer la grille de coordonnées (Identity Map)
# Cette grille représente les positions (x, y) de la frame ACTUELLE
identity_x, identity_y = np.meshgrid(np.arange(w), np.arange(h))
identity_x = identity_x.astype(np.float32)
identity_y = identity_y.astype(np.float32)

# 3. Récupérer les fichiers flow (.npy)
flow_files = sorted(glob.glob(os.path.join(flow_dir, 'flow_*.npy')))

print(f"Début de la propagation pour {len(flow_files)} frames...")

for idx, f_file in enumerate(flow_files):
    # Charger le flow calculé par RAFT (Shape: 2, H, W)
    # Ce flow dit : "Pour le pixel à (x,y), son ancienne position à T0 était +dx, +dy"
    flow = np.load(f_file) 
    
    # Calculer la carte de remappage (Mapping vers T0)
    # On ajoute le déplacement au coordonnées actuelles pour trouver le pixel source
    map_x = identity_x + flow[0, :, :]
    map_y = identity_y + flow[1, :, :]
    
    # === REMAP DIRECT ===
    # On va chercher les pixels du masque Frame 0 
    # pour remplir la frame actuelle Frame IDX
    warped_mask = cv2.remap(mask_initial, 
                            map_x, 
                            map_y, 
                            interpolation=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=0)
    
    # Binarisation (pour éviter le flou sur les bords après interpolation)
    _, binary_mask = cv2.threshold(warped_mask, 128, 255, cv2.THRESH_BINARY)
    
    # Sauvegarde
    out_name = os.path.join(output_dir, f'mask_{idx:04d}.png')
    cv2.imwrite(out_name, binary_mask)

    if idx % 10 == 0:
        print(f"Frame {idx:04d} traitée...")

print(f"Terminé ! Les masques sont disponibles dans : {output_dir}")