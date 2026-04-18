import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt

class Data_DG_SubjectWise(Dataset):
    def __init__(self, root_dir, STMap1, STMap2, frames_num, step, subject_ids, mode='train', overlap=True):
        self.root_dir = root_dir
        self.STMap1 = STMap1
        self.STMap2 = STMap2
        self.frames_num = frames_num
        self.step = step
        self.subject_ids = subject_ids
        self.mode = mode
        self.overlap = overlap
        self.samples = []
        self.subjectwise_bvp = defaultdict(list)
        self._load_data()

    def _load_data(self):
        for subject_dir in sorted(os.listdir(self.root_dir)):
            if subject_dir not in self.subject_ids:
                continue
            subject_path = os.path.join(self.root_dir, subject_dir)
            stmap1 = cv2.imread(os.path.join(subject_path, self.STMap1))
            stmap2 = cv2.imread(os.path.join(subject_path, self.STMap2))
            if stmap1 is None or stmap2 is None:
                continue

            bvp_path = os.path.join(subject_path, 'bvp.csv')
            if not os.path.exists(bvp_path):
                continue
            bvp = pd.read_csv(bvp_path)['BVP'].to_numpy(dtype=np.float32)

            stmap = np.concatenate((stmap1, stmap2), axis=2)
            min_length = min(stmap.shape[1], len(bvp))
            stmap = stmap[:, :min_length, :]
            bvp = bvp[:min_length]

            stride = self.step if not self.overlap else self.step // 2
            for i in range(0, min_length - self.frames_num + 1, stride):
                s = stmap[:, i:i + self.frames_num]
                b = bvp[i:i + self.frames_num]
                if s.shape[1] == self.frames_num:
                    self.samples.append((s, b, subject_dir))
                    if self.mode == 'test':
                        self.subjectwise_bvp[subject_dir].append(b)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stmap, bvp, sid = self.samples[idx]
        stmap1 = torch.tensor(stmap[:, :, :3], dtype=torch.float32).permute(2, 0, 1)
        stmap2 = torch.tensor(stmap[:, :, 3:], dtype=torch.float32).permute(2, 0, 1)
        bvp = torch.tensor(bvp.copy(), dtype=torch.float32)
        return stmap1, stmap2, bvp, sid

    def get_subjectwise_bvp(self):
        result = {}
        for sid, segments in self.subjectwise_bvp.items():
            result[sid] = np.concatenate(segments)
        return result

def subject_split(root_dir, test_ratio=0.2, val_ratio=0.2):
    subjects = sorted([s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))])
    total = len(subjects)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    test = subjects[-n_test:]
    val = subjects[-(n_test + n_val):-n_test]
    train = subjects[:-(n_test + n_val)]
    return train, val, test

def plot_subjectwise_bvp(bvp_dict, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    for sid, bvp in bvp_dict.items():
        plt.figure(figsize=(12, 3))
        plt.plot(bvp)
        plt.title(f"Subject {sid} BVP")
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sid}.png"))
        plt.close()
