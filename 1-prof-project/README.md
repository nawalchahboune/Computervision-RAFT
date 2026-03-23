# Long-term dense motion estimation for video editing

This repository contains a Python implementation of an **optical flow**–based **video editing** pipeline.

The objective is to propagate a **manual edit applied to a reference frame** across a video sequence
using dense and long-term motion estimation.

## Context

This project was developed in the context of the **UE Computer Vision** course, at **IMT Atlantique**. It is intended as an **educational baseline**, not as a production-ready system.

## Objectives

The goals of this project are to:

- Estimate **dense optical flow** between video frames
- Integrate motion over time to obtain **long-term correspondences**
- Propagate a **manual image edit** consistently across a video sequence
- Analyze typical limitations such as drift, occlusions, and temporal artifacts

## Pipeline overview

1. Load a sequence of input frames  
2. Estimate dense optical flow between frames  
3. Obtain long-term displacement fields  
4. Warp the edited reference frame through time  
5. Generate an edited output video  

## Disclaimer

This code is intentionally simplified and does not handle all real-world challenges robustly. Students are encouraged to identify **failure cases**, analyze **limitations**, propose and implement **improvements**.

## Project structure

```text
project/
|-- README.md
|-- requirements.txt
|
|-- data/
|   |-- input_frames/
|   |   |-- frame_000.png
|   |   |-- frame_001.png
|   |   `-- ...
|
|-- edit/
|   |-- modified_000.png            (edited externally, e.g. with GIMP)
|   |-- logoimta.png
|   |-- logomask_000.png
|
|-- output/
|   |-- flow_visualization/
|   |-- propagated_frames/
|   |-- final_video.mp4
|
|   |-- motion/
|   |   |-- optical_flow.py         (dense optical flow estimation)
|   |   |-- visualization.py
|   |
|   |-- utils/
|   |   |-- io.py                   (frame I/O utilities)
|   |   |-- video.py                (frame <-> video conversion)
|   |
|   `-- main.py                     (pipeline)