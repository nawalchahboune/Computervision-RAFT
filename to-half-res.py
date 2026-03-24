import cv2
import os
import glob

output_dir  = 'test-data'
input_dir = 'test-data'
os.makedirs(output_dir, exist_ok=True)
print(f"le dossier d'input est : {input_dir} et contient {len(glob.glob(os.path.join(input_dir, '*')))} images.")
print(f"Conversion des images de {input_dir} vers {output_dir}...")
for img_path in glob.glob(os.path.join(input_dir, '*')):
    print(f"Traitement de {img_path}...")
    img = cv2.imread(img_path)
    print(f"Lecture: {img is not None}")
    if img is None:
        continue
    h, w = img.shape[:2]
    img_half = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    saved = cv2.imwrite(out_path, img_half)
    print(f"Écriture: {saved} -> {out_path}")
print("Conversion terminée.")