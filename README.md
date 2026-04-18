# Physiological_STMap

This repository contains the codebase for generating Spatial-Temporal Maps (STmaps) from facial videos and training neural networks for remote photoplethysmography (rPPG) and physiological signal estimation.

## Features

- **STmap Generation**: 
  - Extracts frames from `.avi` or `.mp4` videos.
  - Uses `face_alignment` for facial landmark detection and tracking to crop processing bounding boxes.
  - Generates YUV-based STmaps evaluating regional (or global) spatial pixel variations horizontally, stacked temporally along the x-axis.
  - Specialized pipeline scripts implemented for widely-used rPPG datasets including **UBFC**, **PURE**, and **PHYS**.

- **Model Architectures**:
  - Implements state-of-the-art vision models and custom neural pipelines located in the `model/` directory (e.g., `STNet`, `Transfuser`, `STEM`, `ResNet` variants).

- **Training Pipeline**:
  - `pretrain.py`, `train.py`, `train2.py`, `finetuning.py` for flexible end-to-end learning phases.
  - `dataloader.py` for effective batching.
  - Configurable loss functions and `eval.py` scripts for benchmarking models.

## Structure Overview

* `STmap/`: Core algorithms and dataset-specific scripts mapping spatial pixels to generated `_stmap_yuv.png` maps.
* `STmap_lmks/`: Face landmarks caching and processing helper utilities.
* `model/`: Neural architecture definitions, multi-head attention systems (SelfAttn, CrossAttn).
* `evaluation/`: Scripts focusing solely on validation data parsing and dataset inference.
* `utils/`: Includes training metrics and custom loss definitions.

## Getting Started

1. Set your python environment and make sure prerequisites are met (`torch`, `opencv-python`, `face_alignment`, etc).
2. Configure video paths inside the respective STmap scripts to match where you stored the raw video data.
3. Once STmaps are constructed, feed them into `train.py` for training the models.
