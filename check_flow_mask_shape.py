import cv2
import numpy as np
import glob
import os

# Chemins à adapter si besoin
mask_init_path = 'mask/raquette_masks/00000_raquette.png'
flow_dir = 'output_viz'

# Charger le masque initial
mask_initial = cv2.imread(mask_init_path, cv2.IMREAD_GRAYSCALE)
if mask_initial is None:
    raise ValueError(f"Impossible de charger le masque initial : {mask_init_path}")
h, w = mask_initial.shape[:2]

# Vérification de la cohérence des tailles
flow_files = sorted(glob.glob(os.path.join(flow_dir, 'flow_*.npy')))
for f_file in flow_files:
    flow = np.load(f_file)
    if flow.shape[1] != h or flow.shape[2] != w:
        print(f"Flow {f_file} shape {flow.shape} ne correspond pas au masque shape {(h, w)}")
    else:
        print(f"Flow {f_file} OK : {flow.shape} == {(2, h, w)}")
print("Vérification terminée.")
