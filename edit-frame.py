import cv2
import numpy as np

# Charger la frame originale (par exemple, frame_000.png)
frame = cv2.imread('test-data/00000.jpg')

# Charger le masque du ballon (blanc sur le ballon, noir ailleurs)
mask = cv2.imread('test-data-mask/00000.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

# Créer une image rouge de la même taille
red = np.zeros_like(frame)
red[:, :] = (0, 0, 255)  # BGR pour rouge pur

# Appliquer le masque (zone blanche = ballon)
mask_3c = cv2.merge([mask, mask, mask]) // 255  # [0,1] sur 3 canaux
frame_colored = frame * (1 - mask_3c) + red * mask_3c

cv2.imwrite('test-data-mask/ref.png', frame_colored.astype(np.uint8))