import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from tqdm import tqdm
import shutil

# from model.Transfuser2 import TransfuserSTMapModel
from model.STEM import TransfuserSTMapModel
from dataloader import Data_DG
from utils.loss.loss import NegPearson, PearsonMSELoss
from utils.metrics import calculate_hr_metrics, calculate_std, calculate_bvp_correlation

def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Attention visualization 디렉토리 초기화
attn_dir = "attn_vis"
if os.path.exists(attn_dir):
    shutil.rmtree(attn_dir)
os.makedirs(attn_dir, exist_ok=True)

config = {
    "epochs": 50,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "frames_num": 300,
    "step": 128,
    "frames_overlap": 150,
    "step_overlap": 0,
    "channels": 6,
    "root_dir": "/media/neuroai/T7/rPPG/STMap_raw/UBFC",
    # "root_dir": "/media/neuroai/T7/rPPG/STMap_raw/UBFC",
    # "root_dir": "/home/neuroai/Projects/DSTMap_v2/DST/PURE",
    "dataName": "UBFC",
    "STMap1": "vid_processed_stmap_rgb.png",
    "STMap2": "vid_processed_stmap_yuv.png",
    "seed": 42,
    "save_dir": "./checkpoints"
}

set_seed(config["seed"])

# Load dataset
print("[INFO] Loading dataset...")
dataset = Data_DG(
    version="v1",
    channels=config["channels"],
    root_dir=config["root_dir"],
    dataName=config["dataName"],
    STMap1=config["STMap1"],
    STMap2=config["STMap2"],
    frames_num=config["frames_num"],
    step=config["step"],
    frames_overlap=config["frames_overlap"],
    step_overlap=config["step_overlap"]
)

for i in range(3):
    _, _, bvp, _ = dataset[i]
    print(f"[DEBUG] Sample {i} BVP Mean/Std:", bvp.mean().item(), bvp.std().item())

# Train / Val / Test split (60/20/20)
total_len = len(dataset)
num_train = int(total_len * 0.6)
num_val = int(total_len * 0.2)
num_test = total_len - num_train - num_val

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset,
    [num_train, num_val, num_test],
    generator=torch.Generator().manual_seed(config["seed"])
)

print(f"[INFO] Dataset split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(config["save_dir"], exist_ok=True)

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

# 모델 선택
#model = SharedRowAttentionUNet(in_channels=3).to(device)
model = TransfuserSTMapModel(ch_in=3, base_dim=64).to(device)

# 손실 함수 및 최적화
criterion = NegPearson().to(device)
optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

# ✅ Cosine Annealing Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"],
    eta_min=1e-4
)

all_epoch_metrics = []

for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0.0

    for stmap1, stmap2, bvp, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
        optimizer.zero_grad()
        # stmap = torch.cat([stmap1, stmap2], dim=1)
        # output = model(stmap1, stmap2)
        output = model(stmap1, stmap2)
        if bvp.ndim == 3:
            bvp = bvp.squeeze(1)

        loss = criterion(output, bvp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds, all_gt = [], []
    with torch.no_grad():
        for stmap1, stmap2, bvp, _ in val_loader:
            stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
            # stmap = torch.cat([stmap1, stmap2], dim=1)
            # output = model(stmap1, stmap2)
            output = model(stmap1, stmap2)
            if bvp.ndim == 3:
                bvp = bvp.squeeze(1)

            loss = criterion(output, bvp)
            val_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_gt.append(bvp.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Val Loss: {avg_val_loss:.4f}")

    scheduler.step()

    all_preds = np.concatenate(all_preds, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    mae, mse, rmse, _, snr = calculate_hr_metrics(all_preds, all_gt)
    corr_bvp = calculate_bvp_correlation(all_preds, all_gt)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R: {corr_bvp:.2f}, SNR: {snr:.2f}")
    all_epoch_metrics.append((mae, mse, rmse, corr_bvp, snr))

    ckpt_path = os.path.join(config["save_dir"], f"epoch{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)

# Summary
metrics_array = np.array(all_epoch_metrics)
best_epoch = np.argmin(metrics_array[:, 1]) + 1  # MSE 기준
std_metrics = calculate_std(metrics_array)
best_metrics = metrics_array[best_epoch - 1]

print(f"\n📌 Best Epoch: {best_epoch}")
metric_names = ["MAE", "MSE", "RMSE", "R", "SNR"]
for name, mean_val, std_val in zip(metric_names, best_metrics, std_metrics):
    print(f"{name}: {mean_val:.2f} ± {std_val:.2f}")



# ✅ Test Set Evaluation
import matplotlib.pyplot as plt
from collections import defaultdict

print("\n[INFO] Evaluating on Test Set...")
model.eval()
test_preds, test_gt = [], []

# 개별 plot 저장을 위한 디렉토리 생성
plot_dir = os.path.join(config["save_dir"], "sample_plots")
os.makedirs(plot_dir, exist_ok=True)

sample_idx = 0

with torch.no_grad():
    for stmap1, stmap2, bvp, subject_ids in tqdm(test_loader, desc="Testing"):
        stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
        # stmap = torch.cat([stmap1, stmap2], dim=1)
        # output = model(stmap1, stmap2)
        output = model(stmap1, stmap2)
        if bvp.ndim == 3:
            bvp = bvp.squeeze(1)

        preds = output.cpu().numpy()
        gts = bvp.cpu().numpy()

        test_preds.append(preds)
        test_gt.append(gts)

        # ▶ 각 샘플에 대해 plot 저장
        for pred, gt, sid in zip(preds, gts, subject_ids):
            pred_norm = zscore(pred)
            gt_norm = zscore(gt)

            plt.figure(figsize=(10, 3))
            plt.plot(gt_norm, label="GT (z-score)", linewidth=1.5)
            plt.plot(pred_norm, label="Pred (z-score)", linewidth=1.5)
            plt.title(f"Sample {sample_idx} (Subject {sid})")
            plt.xlabel("Frame")
            plt.ylabel("Z-Score")
            plt.legend()
            plt.grid(True)
            plt.ylim([-3, 3])  # 선택적 고정 범위
            plt.tight_layout()

            save_path = os.path.join(plot_dir, f"{sample_idx:04d}_subj_{sid}_zscore.png")
            plt.savefig(save_path)
            plt.close()
            sample_idx += 1

print(f"[INFO] Saved {sample_idx} sample plots to: {plot_dir}")

# 📊 전체 성능 평가
test_preds = np.concatenate(test_preds, axis=0)
test_gt = np.concatenate(test_gt, axis=0)

mae, mse, rmse, _, snr = calculate_hr_metrics(test_preds, test_gt)
corr_bvp = calculate_bvp_correlation(test_preds, test_gt)

print("\n📊 Test Set Performance:")
print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R: {corr_bvp:.2f}, SNR: {snr:.2f}")
