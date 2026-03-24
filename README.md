# Computer vision project
# RAFT

This repository contains the source code for the paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements

The code has been tested with PyTorch 1.6 and Cuda 10.1.

```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## Demos

Pretrained models can be downloaded by running

```Shell
./download_models.sh
```

or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

You can demo a trained model on a sequence of frames

python demo.py --model=models/raft-things.pth --path=tennis-frames
