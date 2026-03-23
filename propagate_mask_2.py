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

# Au lieu de mask = new_mask (itératif), on accumule le flow ou on utilise un masque flottant
mask_float = mask.astype(np.float32) 

for idx, flow_file in enumerate(flow_files):
    flow = np.load(flow_file)
    flow = np.transpose(flow, (1, 2, 0))

    h, w = mask.shape[:2]
    # Création de la carte de remappage
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32) + flow[..., 0]
    map_y = map_y.astype(np.float32) + flow[..., 1]

    # 1. Utiliser INTER_LINEAR pour préserver la forme
    # 2. Utiliser un masque flottant pour éviter les arrondis successifs
    new_mask = cv2.remap(mask_float, map_x, map_y, 
                         interpolation=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT)
    
    # Post-process : Seuillage pour nettoyer le bruit et rebinaryser
    _, mask_binary = cv2.threshold(new_mask, 128, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(os.path.join(output_dir, f'mask_{idx+1:04d}.png'), mask_binary)
    
    # On continue avec le masque float pour la prochaine itération
    mask_float = new_mask