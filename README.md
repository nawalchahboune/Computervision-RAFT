# Computer Vision Project - RAFT for dense optical flow estimation

This repository uses on the official RAFT implementation from:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020<br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## The pipline:

Beyond the original RAFT code, this repository contains our project work focused on
video editing by optical-flow-based propagation.

Our main contributions are:

1. Multiple propagation strategies implemented as separate experiments:
	- direct reference warping with consistency filtering (`demo_direct.py`),
	- pairwise propagation (`demo-pairwise.py`),
	- pairwise robust median-center propagation (`demo-pairwise-median.py`),
	- sequential cumulative flow from reference (`demo-sequential-from-ref.py`),
	- sequential backward composition to reference (`demo-sequential-to-ref.py`).
2. Mask and data preparation tools:
	- mask color inspection (`inspect_mask_color.py`),
	- red/green mask extraction (`extract_red_mask.py`, `extract_green_mask.py`),
	- image resizing helper (`to-half-res.py`).
3. Debugging and validation utilities:
	- image dimension checks (`check_dim_images.py`),
	- flow/mask shape consistency checks (`check_flow_mask_shape.py`),
	- frame sequence to video conversion (`frames_to_video.py`).
4. Reproducible test assets in `test-data` and `test-data-mask` to run and compare methods.

## Project Context

This work was developed as part of the UE Computer Vision course at IMT Atlantique,
with an emphasis on long-term dense motion estimation and edit propagation.

## Setup

### RAFT Base Environment

The original RAFT code was tested with PyTorch 1.6 and CUDA 10.1.

```bash
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

### Additional Project Dependencies

For the educational pipeline in `1-prof-project`:

```bash
pip install -r 1-prof-project/requirements.txt
```

## Model Weights

Pretrained models can be downloaded with:

```bash
./download_models.sh
```

Or manually from Google Drive:
[RAFT pretrained models](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

## Quick Start

### 1. Run one of our propagation experiments

```bash
python demo_direct.py
python demo-pairwise.py
python demo-pairwise-median.py
python demo-sequential-from-ref.py
python demo-sequential-to-ref.py
```

### 2. Run the educational reference-based pipeline

```bash
cd 1-prof-project/src
python main.py
```

## Repository Layout (Main Additions)

- `1-prof-project/src/editing`: warping functions for image/mask propagation.
- `1-prof-project/src/motion`: optical flow wrappers and visualization.
- `1-prof-project/src/utils`: frame I/O and video utilities.
- `demo_*.py` and `demo-*.py`: alternative propagation experiment scripts.
- `test-data`, `test-data-mask`: sample frames and masks.

## Notes

- Some scripts use hardcoded paths for fast experimentation. Update paths as needed.
- Most experiment scripts are currently configured for CPU (`DEVICE = 'cpu'`), but can be adapted to CUDA.
