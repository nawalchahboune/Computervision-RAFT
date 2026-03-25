import cv2
import os
from pathlib import Path
import glob

print("=" * 60)
print("CONVERSION DES FRAMES EN VIDEO")
print("=" * 60)

# *** CONFIGURATION ***
FRAMES_DIR = "output/sequential_to_ref"
OUTPUT_VIDEO = "output-videos/sequential_from_to.mp4"
FPS = 15 # Frames par seconde
# ***

# Trouver toutes les images PNG/JPG commençant par "edited_"
frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "edited_*.png"))) + \
              sorted(glob.glob(os.path.join(FRAMES_DIR, "edited_*.jpg")))

if not frame_files:
    print(f" Erreur: aucune image trouvée dans '{FRAMES_DIR}'")
    exit(1)

print(f"\n✓ {len(frame_files)} frames trouvées")

# Charger la première frame pour obtenir les dimensions
first_frame = cv2.imread(frame_files[0])
if first_frame is None:
    print(f" Erreur: impossible de charger {frame_files[0]}")
    exit(1)

h, w = first_frame.shape[:2]
print(f"  Résolution: {w}×{h}")
print(f"  Durée: {len(frame_files)/FPS:.1f}s ({len(frame_files)} frames @ {FPS} fps)")

# Créer le VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec H.264
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

if not out.isOpened():
    print(f" Erreur: impossible de créer le fichier vidéo")
    exit(1)

# Écrire chaque frame dans la vidéo
print(f"\n Conversion en cours...")
for i, frame_path in enumerate(frame_files):
    frame = cv2.imread(frame_path)
    if frame is not None:
        # Adapter la taille si nécessaire
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        out.write(frame)
        
        if (i + 1) % 10 == 0:
            print(f"  ✓ {i+1}/{len(frame_files)} frames traitées")
    else:
        print(f"Impossible de charger: {frame_path}")

out.release()

print(f"\n Vidéo créée: {OUTPUT_VIDEO}")
print(f"   Fichier: {os.path.getsize(OUTPUT_VIDEO) / (1024*1024):.1f} MB")
print(f"\n Pour changer le dossier source, modifiez FRAMES_DIR au début du script")
