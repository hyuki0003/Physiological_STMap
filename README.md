# Physiological STmap

> Official implementation of **"생리학적 혈류 방향 기반 시공간 특징맵과 교차 주의 집중을 이용한 비접촉 원격 심박수 추정"**
> (*Remote Heart Rate Estimation via Physiology-guided Spatio-Temporal Feature Maps and Cross-Attention*) — 2025 IEIE Summer Conference.
>
> 📄 [Paper (DBpia)](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE12332223) · 이동혁, 최영석 (광운대학교)

Code for generating **physiology-guided Spatio-Temporal maps (STmaps)** from facial videos and training neural networks for **remote photoplethysmography (rPPG)** and heart rate estimation.

---

## Overview

Remote photoplethysmography (rPPG) is a non-contact technique that recovers physiological signals (heart rate, pulse waveform) from the subtle chromatic variations present in facial video. Conventional STmaps capture temporal dynamics well, but their **spatial layout typically ignores physiology** — facial regions are ordered by pixel index or raw bounding-box position, not by how blood actually flows through the face.

This work addresses that gap with two ideas:

1. **Physiology-guided STmap.** Facial ROIs are aligned and arranged along the real direction of blood flow (**chin → forehead**), so the spatial axis of the STmap carries a physiologically meaningful, per-subject-consistent ordering in addition to the temporal axis.
2. **Cross-STmap Attention.** A 2D CNN with a cross-attention module that fuses complementary information from STmaps computed in **different color spaces (RGB and YUV)**.

Evaluated on **PURE** and **UBFC-rPPG**, the proposed approach outperforms conventional STmap-based baselines on heart rate estimation.

---

## Preprocessing pipeline

The preprocessing class `BasePreprocess` runs in three stages. The first two are what make the downstream STmap *physiology-guided*.

### 1 · Landmark-anchored face alignment

- 68-point 2D landmarks are detected per frame using [`face_alignment`](https://github.com/1adrianb/face-alignment) on GPU.
- Missing / failed detections are recovered by **cubic B-spline interpolation** (`scipy.interpolate.splrep` / `splev`) independently over each of the 136 landmark coordinates, followed by an **edge-padded moving-average** smoother to suppress jitter.
- Failure safeguards — a video is discarded when:
  - more than **9 consecutive** frames fail detection, or
  - more than **5 %** of all frames have no detectable face.
- Each frame is then **affine-warped** using three anchor landmarks:

  | Source landmark | Index | Destination (128 × 128) |
  |---|---|---|
  | Left outer eye corner | `lmk[36]` | `(20, 48)` |
  | Right outer eye corner | `lmk[45]` | `(108, 48)` |
  | Chin tip | `lmk[8]` | `(64, 128)` |

After this step the chin sits at a fixed pixel row, the eye line sits at another fixed row, and the forehead occupies the top band — **across every subject and every frame**. This is what lets the STmap's vertical axis carry a consistent anatomical meaning along the blood-flow direction, instead of being a per-video arbitrary ordering.

### 2 · BVP synchronization

Ground-truth PPG / BVP signals are resampled to the video's frame rate and saved to `preprocess_data_path/vts_bvps/`. The `_sync` hook is dataset-specific and overridden per subclass (UBFC, PURE, PHYS).

### 3 · STmap generation

The aligned 128 × 128 face is partitioned into an 8 × 8 grid of 16 × 16 patches (`m = 16`). For each patch the channel-wise spatial mean is taken, with per-frame z-normalization to suppress momentary baseline drift. Stacking over `T` frames yields an STmap tensor of shape `(C, H/m, W/m, T) = (C, 8, 8, T)`.

The pipeline produces three variants in parallel:

| Output folder | Shape | Color space |
|---|---|---|
| `stmap/rgb/` | `(3, 8, 8, T)` | RGB, per-frame z-normalized |
| `stmap/yuv/` | `(3, 8, 8, T)` | YUV (BT.709), per-frame z-normalized |
| `stmap/both/` | `(6, 8, 8, T)` | RGB ⊕ YUV (channel-concatenated) |

The `both` variant is the input for the **Cross-STmap Attention** module, which is designed to fuse RGB and YUV representations.

---

## Repository structure

```
├── STmap/        # BasePreprocess + dataset-specific subclasses
│                 # (UBFC, PURE, PHYS pipelines; STmap generation)
├── STmap_lmks/   # Facial landmark caching and processing helpers
├── model/        # Network architectures and attention modules
│                 #   - STNet, Transfuser, STEM, ResNet variants
│                 #   - SelfAttn, CrossAttn (Cross-STmap Attention)
├── evaluation/   # Validation parsing and dataset inference scripts
└── utils/        # Training metrics and custom loss definitions
```

---

## Preprocessing parameters

| Arg | Default | Description |
|---|---|---|
| `img_size` | `128` | Aligned face resolution (square). |
| `m` | `16` | Patch size for the STmap grid → produces an `img_size/m × img_size/m` grid. |
| `fs` | `30.0` | Video frame rate (Hz). |
| `fl`, `fh` | `0.4`, `2.5` Hz | Band-pass cutoffs for BVP (≈ 24 – 150 bpm). |
| `order` | `8` | Butterworth filter order. |
| `device` | `cuda:2` | GPU for landmark detection. |
| `do_stmap` | `True` | If False, only produces aligned videos. |
| `stmap_type` | `2` | `0` = RGB, `1` = YUV, `2` = both (6-channel). |

---

## Getting started

### 1 · Environment

```bash
pip install torch torchvision opencv-python face-alignment numpy scipy tqdm
```

Tested with PyTorch ≥ 1.13 and CUDA 11.x.

### 2 · Preprocess a dataset

Subclass `BasePreprocess` and override the two dataset-specific hooks:

```python
from STmap.base import BasePreprocess

class UBFCPreprocess(BasePreprocess):
    def _get_video_metadata(self, data_path):
        # return a list of {"index": str, "path": str, ...}
        ...

    def _sync(self, preprocess_data_path, raw_data_path):
        # resample / align the dataset's ground-truth BVPs to video timestamps
        ...

pre = UBFCPreprocess(
    raw_data_path        = '/data/UBFC-rPPG',
    preprocess_data_path = '/data/UBFC-rPPG/processed',
    stmap_type           = 2,      # RGB + YUV
    device               = 'cuda:0',
)

# pre.x -> list of STmap .npy paths
# pre.y -> list of synchronized BVP .npy paths
# pre.abnormal_files -> discarded / unreadable videos
```

The processed directory is organized as:

```
preprocess_data_path/
├── align/           # 128×128 affine-aligned .mp4 videos
├── vts_bvps/        # synchronized BVP .npy files
└── stmap/
    ├── rgb/         # (3, 8, 8, T) RGB STmaps
    ├── yuv/         # (3, 8, 8, T) YUV STmaps
    └── both/        # (6, 8, 8, T) concatenated RGB+YUV STmaps
```

### 3 · Train

```bash
python train.py        # main supervised training
python pretrain.py     # optional pretraining phase
python finetuning.py   # dataset-specific finetuning
```

### 4 · Evaluate

```bash
python eval.py
```

---

## Datasets

- [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
- [PURE](https://www.tu-ilmenau.de/neurob/data-sets-code/pulse/)
- PHYS (multimodal physiological dataset)

Each dataset has a dedicated `BasePreprocess` subclass under `STmap/`.

---

## Citation

If this repository is useful for your research, please cite:

```bibtex
@inproceedings{lee2025physiological,
  title        = {생리학적 혈류 방향 기반 시공간 특징맵과 교차 주의 집중을 이용한 비접촉 원격 심박수 추정},
  author       = {이동혁 and 최영석},
  booktitle    = {2025년도 대한전자공학회 하계학술대회 논문집},
  pages        = {3471--3475},
  year         = {2025},
  organization = {대한전자공학회}
}
```

---

## Contact

Kwangwoon University — Dept. of Electronic Engineering.
For questions, issues, or collaboration, please open a GitHub issue.
