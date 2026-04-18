import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from utils.metrics import metrics

class SingleSubjectData(Dataset):
    def __init__(self, subject_dir, version, channels, STMap1, STMap2, frames_num, height, dataName):
        self.subject_dir = subject_dir
        self.version = version
        self.channels = channels
        self.STMap_Name1 = STMap1
        self.STMap_Name2 = STMap2
        self.frames_num = frames_num
        self.height = height
        self.dataName = dataName
        self.samples = []

        self._load_data()

        print(f"Dataset initialized with {len(self.samples)} samples for subject {os.path.basename(self.subject_dir)}.")

    def _load_data(self):
        """STMap 및 BVP 데이터를 로드하고 샘플을 생성합니다."""
        stmap1, stmap2 = self._load_stmaps()
        if stmap1 is None or stmap2 is None:
            return

        if self.channels == 3:
            stmap = stmap1
        elif self.channels == 6:
            stmap = np.concatenate((stmap1, stmap2), axis=2)
        else:
            raise ValueError(f"Unsupported number of channels: {self.channels}")

        bvp = self._load_bvp()
        if bvp is None:
            return

        min_length = min(stmap.shape[1], len(bvp))
        if min_length < self.frames_num:
            print(f"Skipping subject {os.path.basename(self.subject_dir)} due to insufficient data length.")
            return

        self._generate_samples(stmap, bvp, min_length)

    def _load_stmaps(self):
        """STMap1과 STMap2 이미지를 로드합니다."""
        stmap_path1 = os.path.join(self.subject_dir, self.STMap_Name1)
        stmap_path2 = os.path.join(self.subject_dir, self.STMap_Name2)

        if os.path.exists(stmap_path1) and os.path.exists(stmap_path2):
            stmap1 = cv2.imread(stmap_path1)
            stmap2 = cv2.imread(stmap_path2)
            return stmap1, stmap2
        else:
            print(f"Skipping subject {os.path.basename(self.subject_dir)} due to missing STMap files.")
            return None, None

    def _load_bvp(self):
        """BVP 데이터를 로드합니다."""
        bvp_path_PURE = os.path.join(self.subject_dir, 'bvp.csv')
        bvp_path_UBFC = os.path.join(self.subject_dir, 'ground_truth.txt')

        if self.dataName == 'PURE' and os.path.exists(bvp_path_PURE):
            bvp = pd.read_csv(bvp_path_PURE)
            return np.array(bvp['BVP'], dtype=np.float32).flatten()

        elif self.dataName == 'UBFC' and os.path.exists(bvp_path_UBFC):
            with open(bvp_path_UBFC, 'r') as f:
                lines = f.readlines()
                bvp_values = lines[0].split()
                return np.array(bvp_values, dtype=np.float32).flatten()

        print(f"Skipping subject {os.path.basename(self.subject_dir)} due to missing BVP file.")
        return None

    def _generate_samples(self, stmap, bvp, min_length):
        """주어진 STMap과 BVP 데이터를 기반으로 샘플을 생성합니다."""
        length = (min_length // self.frames_num) * self.frames_num
        stmap = stmap[:self.height, :length]
        bvp = bvp[:length]

        num_samples = length // self.frames_num
        for i in range(num_samples):
            stmap_sample = stmap[:, i * self.frames_num:(i + 1) * self.frames_num]
            bvp_sample = bvp[i * self.frames_num:(i + 1) * self.frames_num]

            stmap_sample = self._normalize_stmap(stmap_sample)
            bvp_sample = self._process_bvp(bvp_sample)

            if stmap_sample.shape[1] == self.frames_num and len(bvp_sample) == self.frames_num:
                self.samples.append((stmap_sample, bvp_sample))

    def _normalize_stmap(self, stmap_sample):
        """STMap 데이터를 정규화합니다."""
        for c in range(stmap_sample.shape[2]):
            for r in range(stmap_sample.shape[0]):
                filtered_data = metrics.butter_bandpass_filter(stmap_sample[r, :, c], lowcut=0.75, highcut=2.5, fs=30, order=5)
                min_val = np.min(filtered_data)
                max_val = np.max(filtered_data)
                stmap_sample[r, :, c] = 255 * (
                    (stmap_sample[r, :, c] - min_val) / (0.00001 + max_val - min_val))
        return stmap_sample

    def _process_bvp(self, bvp_sample):
        """BVP 데이터를 필터링 및 정규화합니다."""
        bvp_sample = metrics.butter_bandpass_filter(bvp_sample, lowcut=0.75, highcut=2.5, fs=30, order=5)
        return metrics.diff_normalize_label(bvp_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stmap, bvp = self.samples[idx]
        stmap = torch.tensor(stmap, dtype=torch.float32).permute(2, 0, 1)  # (channels, height, width)
        bvp = torch.tensor(bvp, dtype=torch.float32)

        return stmap, bvp
