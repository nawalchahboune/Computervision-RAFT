import os
import cv2

folder = "camel-frames"  # adapte le chemin si besoin

sizes = set()
for fname in sorted(os.listdir(folder)):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img = cv2.imread(os.path.join(folder, fname))
        if img is not None:
            sizes.add(img.shape)
        else:
            print(f"Erreur de lecture: {fname}")

print("Dimensions trouvées:")
for s in sizes:
    print(s)