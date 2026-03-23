import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger le masque couleur
mask_path = 'mask/00000.png'
mask = cv2.imread(mask_path)

if mask is None:
    print("Erreur: masque non trouvé!")
    exit(1)

# Afficher l'image pour sélectionner un pixel
plt.figure("Clique sur la raquette (vert)")
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.title("Clique sur la raquette pour voir la couleur BGR")
plt.axis('off')

coords = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        bgr = mask[y, x]
        print(f"Coordonnées (x={x}, y={y}) - BGR: {bgr}")
        coords.append((x, y, bgr))
        plt.scatter([x], [y], c='yellow', s=40)
        plt.draw()

cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()

if coords:
    print("Pixels sélectionnés (x, y, BGR):")
    for c in coords:
        print(c)
else:
    print("Aucun pixel sélectionné.")
