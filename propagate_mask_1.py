import numpy as np
import numpy as np
import cv2
import glob
import os
# Ajout pour affichage
import matplotlib.pyplot as plt
# Chemins


# === Chemins ===
flow_dir = 'output_viz'  # dossier où sont les .npy
# Utiliser le masque raquette binaire
mask_path = 'mask/raquette_masks/00000_raquette.png'  # adapte le nom si besoin
output_dir = 'propagated_masks'
os.makedirs(output_dir, exist_ok=True)
# Charger le masque initial


# === Charger le masque initial (binaire raquette) ===
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # binaire: 0 ou 255
print(f"Type du masque: {type(mask)}, shape: {mask.shape}, dtype: {mask.dtype}")

# Affichage du masque initial
plt.figure("Masque raquette initiale")
if mask is not None:
    plt.imshow(mask, cmap='gray')
    plt.title(f"Masque raquette initial ({mask_path})")
    plt.axis('off')
    plt.show()
else:
    print("Le masque n'a pas été chargé correctement !")

#cv2.imwrite(os.path.join(output_dir, 'mask_0000.png'), mask)
# Charger les flows

# === Charger les flows ===
flow_files = sorted(glob.glob(os.path.join(flow_dir, 'flow_*.npy')))

# === Boucle de propagation désactivée pour debug du masque initial ===
for idx, flow_file in enumerate(flow_files):
   flow = np.load(flow_file)  # shape: (2, H, W)
   flow = np.transpose(flow, (1, 2, 0))  # shape: (H, W, 2)

   h, w = mask.shape[:2]
   coords = np.meshgrid(np.arange(w), np.arange(h))
   coords = np.stack(coords, axis=-1).astype(np.float32)  # (H, W, 2)
   map_xy = coords + flow

   # Propagation du masque
   new_mask = cv2.remap(mask, map_xy[...,0], map_xy[...,1], interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
   cv2.imwrite(os.path.join(output_dir, f'mask_{idx+1:04d}.png'), new_mask)
   mask = new_mask  # propagation séquentielle
print("Propagation terminée.")

