
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Tolérance autour de [0, 0, 255] (RED en BGR)
lower_red = np.array([0, 0, 100])
upper_red = np.array([50, 50, 255])

mask_dir = 'test-data-mask'  # adapte le chemin si besoin
output_dir = os.path.join(mask_dir, 'red_masks')
os.makedirs(output_dir, exist_ok=True)

mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

for mask_path in mask_files:
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Erreur: masque non trouvé pour {mask_path}!")
        continue

    mask_red = cv2.inRange(mask, lower_red, upper_red)

    # Affichage pour la première image seulement
    if mask_path == mask_files[0]:
        plt.figure("Masque objet (rouge)")
        plt.imshow(mask_red, cmap='gray')
        plt.title(f"Masque binaire (objet rouge)\n{os.path.basename(mask_path)}")
        plt.axis('off')
        plt.show()

    out_name = os.path.splitext(os.path.basename(mask_path))[0] + '_red.png'
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, mask_red)
    print(f"Masque rouge extrait et sauvegardé sous {out_path}")