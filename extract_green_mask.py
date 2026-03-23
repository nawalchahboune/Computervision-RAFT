
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Tolérance autour de [0, 128, 0]
lower_green = np.array([0, 120, 0])
upper_green = np.array([10, 135, 10])

mask_dir = 'mask'
output_dir = os.path.join(mask_dir, 'raquette_masks')
os.makedirs(output_dir, exist_ok=True)

mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

for mask_path in mask_files:
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Erreur: masque non trouvé pour {mask_path}!")
        continue

    mask_green = cv2.inRange(mask, lower_green, upper_green)

    # Affichage pour la première image seulement
    if mask_path == mask_files[0]:
        plt.figure("Masque raquette (vert)")
        plt.imshow(mask_green, cmap='gray')
        plt.title(f"Masque binaire (raquette verte)\n{os.path.basename(mask_path)}")
        plt.axis('off')
        plt.show()

    out_name = os.path.splitext(os.path.basename(mask_path))[0] + '_raquette.png'
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, mask_green)
    print(f"Masque raquette extrait et sauvegardé sous {out_path}")