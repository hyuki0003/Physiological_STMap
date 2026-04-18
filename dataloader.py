import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import math
import torchvision.transforms as transforms
from utils import metrics
import scipy.signal as signal

class Data_DG(Dataset):
    def __init__(self, version, channels, root_dir, dataName, STMap1, STMap2, frames_num, step, frames_overlap,
                 step_overlap):
        self.version = version
        self.channels = channels
        self.root_dir = root_dir
        self.dataName = dataName
        self.STMap_Name1 = STMap1
        self.STMap_Name2 = STMap2
        self.frames_num = frames_num
        self.step = step
        self.frames_overlap = frames_overlap
        self.step_overlap = step_overlap
        self.samples = []

        self._load_data()

        print(f"Dataset initialized with {len(self.samples)} samples.")

    def _load_data(self):
        for subject_dir in os.listdir(self.root_dir):
            subject_path = os.path.join(self.root_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            stmap1, stmap2 = self._load_stmaps(subject_path)
            bvp = self._load_bvp(subject_path)

            if stmap1 is None or (self.channels == 6 and stmap2 is None):
                continue

            if self.channels == 3:
                stmap = stmap1
            elif self.channels == 6:
                stmap = np.concatenate((stmap1, stmap2), axis=2)
            else:
                raise ValueError(f"Unsupported number of channels: {self.channels}")

            min_length = min(stmap.shape[1], len(bvp))
            if min_length < self.frames_num:
                continue

            self._generate_samples(stmap, bvp, min_length, subject_id=subject_dir)

    def _generate_samples(self, stmap, bvp, min_length, subject_id):
        length = (min_length // self.frames_num) * self.frames_num
        stmap = stmap[:, :length, :]
        bvp = bvp[:length]

        step_size1 = self.frames_num - self.frames_overlap
        step_size2 = self.step - self.step_overlap

        for i in range(0, length - self.frames_num + 1, max(1, step_size1)):
            for j in range(0, stmap.shape[0] - self.step + 1, max(1, step_size2)):
                stmap_sample = stmap[j:j + self.step, i:i + self.frames_num]
                bvp_sample = bvp[i:i + self.frames_num]

                stmap_sample = self._normalize_stmap(stmap_sample)
                bvp_sample = self._process_bvp(bvp_sample)

                if stmap_sample.shape[1] == self.frames_num and len(bvp_sample) == self.frames_num:
                    self.samples.append((stmap_sample, bvp_sample, subject_id))

    def _load_stmaps(self, subject_path):
        if self.dataName == "VV":
            subject_name = os.path.basename(subject_path)
            stmap_path1 = os.path.join(subject_path, f"{subject_name}_processed_rgb.png")
            stmap_path2 = os.path.join(subject_path, f"{subject_name}_processed_yuv.png")
        else:
            stmap_path1 = os.path.join(subject_path, self.STMap_Name1)
            stmap_path2 = os.path.join(subject_path, self.STMap_Name2)

        if os.path.exists(stmap_path1) and os.path.exists(stmap_path2):
            stmap1 = cv2.imread(stmap_path1)
            stmap2 = cv2.imread(stmap_path2)
            return stmap1, stmap2
        else:
            print(f"⚠️ Skipping {subject_path} due to missing STMap files")
            return None, None

    def _load_bvp(self, subject_path):
        bvp_path_PURE = os.path.join(subject_path, 'bvp.csv')
        bvp_path_UBFC = os.path.join(subject_path, 'ground_truth.txt')

        if self.dataName == "UBFC" and os.path.exists(bvp_path_UBFC):
            with open(bvp_path_UBFC, 'r') as f:
                lines = f.readlines()
                bvp_values = lines[0].split()
                return np.array(bvp_values, dtype=np.float32).flatten()

        elif self.dataName == "PURE" and os.path.exists(bvp_path_PURE):
            bvp = pd.read_csv(bvp_path_PURE)
            return np.array(bvp['BVP'], dtype=np.float32).flatten()

        elif self.dataName == "VV":
            bvp_path = os.path.join(subject_path, 'bvp.csv')
            if os.path.exists(bvp_path):
                bvp = pd.read_csv(bvp_path)
                return np.array(bvp['bvp'], dtype=np.float32).flatten()

        print(f"⚠️ Skipping {subject_path} due to missing BVP file")
        return None

    def _normalize_stmap(self, stmap_sample):

        # stmap_sample = stmap_sample.astype(np.float32)
        # H, T, C = stmap_sample.shape
        # for h in range(H):
        #     for c in range(C):
        #         signal_1d = stmap_sample[h, :, c]
        #         signal_1d = metrics.butter_bandpass_filter(signal_1d, 0.5, 3.0, fs=30, order=4)
        #         signal_1d = metrics.standardized_label(signal_1d)
        #         signal_1d = (signal_1d - signal_1d.min()) / (signal_1d.max() - signal_1d.min() + 1e-8)  # MinMax
        #
        #         stmap_sample[h, :, c] = signal_1d

        return stmap_sample

    def _process_bvp(self, bvp_sample):
        bvp_sample = metrics.butter_bandpass_filter(bvp_sample, lowcut=0.5, highcut=3.0, fs=30, order=4)
        bvp_sample = metrics.standardized_label(bvp_sample)
        return bvp_sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stmap, bvp, subject_id = self.samples[idx]

        stmap1 = torch.tensor(stmap[:, :, :3], dtype=torch.float32).permute(2, 0, 1)  # RGB
        stmap2 = torch.tensor(stmap[:, :, 3:], dtype=torch.float32).permute(2, 0, 1)  # YUV
        bvp = torch.tensor(bvp.copy(), dtype=torch.float32)

        return stmap1, stmap2, bvp, subject_id
