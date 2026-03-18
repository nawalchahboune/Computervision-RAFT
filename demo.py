import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cpu'  # Utilisation du CPU uniquement

# DEVICE = 'cuda'
DEVICE = 'cpu'
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # return img[None].to(DEVICE)  # GPU code commenté
    return img[None]  # CPU uniquement


def viz(img, flo):
    print("in viz")
    img = img[0].permute(1,2,0).numpy()  # .cpu() inutile sur CPU
    flo = flo[0].permute(1,2,0).numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # Sauvegarder l'image au lieu d'afficher
    output_path = 'output_viz'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    idx = len(os.listdir(output_path))
    out_file = os.path.join(output_path, f'viz_{idx:04d}.png')
    # OpenCV attend BGR, donc conversion
    cv2.imwrite(out_file, (img_flo[:, :, [2,1,0]]).astype(np.uint8))
    print(f"Image sauvegardée : {out_file}")
    
    
def demo(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    # model.to(DEVICE)  # GPU code commenté
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
