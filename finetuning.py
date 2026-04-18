import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torch.utils.data import random_split

from model.tm2_6ch import UNetReconAndFusion  # 앞서 정의된 전체 통합 모델
from dataloader import Data_DG
from utils.metrics import calculate_hr_metrics
from utils.loss.loss import CombinedLoss


# -----------------------------
# 설정
# -----------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

config = {
    "epochs": 100,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "recon_ckpt": "pretrained.pth",
    "save_dir": "checkpoints_finetune",
    "recon_vis_dir": "recon_vis",
    "recon_type": "both",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "root_dir": "/media/neuroai/T7/rPPG/STMap_raw/UBFC",
    "dataName": "UBFC",
    "STMap1": "vid_processed_stmap_rgb.png",
    "STMap2": "vid_processed_stmap_yuv.png",
    "frames_num": 160,
    "step": 16,
    "frames_overlap": 80,
    "step_overlap": 8,
    "channels": 6,
}

set_seed(42)

os.makedirs(config["save_dir"], exist_ok=True)
os.makedirs(config["recon_vis_dir"], exist_ok=True)

# -----------------------------
# 모델 & 데이터 로딩
# -----------------------------
print("[INFO] Loading data...")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

model = UNetReconAndFusion(mode='finetune', reconstructor_ckpt="pretrained.pth").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
criterion = CombinedLoss(psd_weight=0.1, pearson_weight=0.05)

# -----------------------------
# 학습 루프
# -----------------------------
print("[INFO] Starting fine-tuning...")

state = torch.load("pretrained.pth")
print(state.keys())


for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0.0
    for stmap1, stmap2, bvp in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
        stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device).squeeze(1)

        pred = model(stmap1, stmap2)
        loss = criterion(pred, bvp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0.0
    all_preds, all_gts = [], []
    with torch.no_grad():
        for stmap1, stmap2, bvp in val_loader:
            stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device).squeeze(1)
            pred = model(stmap1, stmap2)
            loss = criterion(pred, bvp)
            val_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_gts.append(bvp.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    mae, mse, rmse, r, snr = calculate_hr_metrics(all_preds, all_gts)
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | MAE: {mae:.2f}, RMSE: {rmse:.2f}, R: {r:.2f}, SNR: {snr:.2f}")

    torch.save(model.state_dict(), os.path.join(config["save_dir"], f"epoch{epoch+1}.pth"))