# train_subjectwise.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.Transfuser import TransfuserSTMapModel
from dataloader2 import Data_DG_SubjectWise, subject_split, plot_subjectwise_bvp
from utils.loss.loss import NegPearson
from utils.metrics import calculate_hr_metrics, calculate_bvp_correlation

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------- Config -----------------------
config = {
    "epochs": 30,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "frames_num": 300,
    "step": 128,
    "root": "/media/neuroai/T7/rPPG/STMap_raw/PURE",
    "STMap1": "vid_stmap_rgb.png",
    "STMap2": "vid_stmap_yuv.png",
    "seed": 42,
    "save_dir": "./checkpoints_subjectwise"
}

set_seed(config["seed"])
os.makedirs(config["save_dir"], exist_ok=True)

# ----------------------- Subject Split & Loaders -----------------------
train_ids, val_ids, test_ids = subject_split(config["root"])

train_set = Data_DG_SubjectWise(config["root"], config["STMap1"], config["STMap2"],
                                 config["frames_num"], config["step"], train_ids, 'train', overlap=True)
val_set = Data_DG_SubjectWise(config["root"], config["STMap1"], config["STMap2"],
                               config["frames_num"], config["step"], val_ids, 'val', overlap=True)
test_set = Data_DG_SubjectWise(config["root"], config["STMap1"], config["STMap2"],
                                config["frames_num"], config["step"], test_ids, 'test', overlap=False)

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# ----------------------- Model -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransfuserSTMapModel(ch_in=3, base_dim=64).to(device)
criterion = NegPearson().to(device)
optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-4)

# ----------------------- Training -----------------------
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0.0
    for stmap1, stmap2, bvp, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
        optimizer.zero_grad()
        output = model(stmap1, stmap2)
        loss = criterion(output, bvp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0.0
    all_preds, all_gt = [], []
    with torch.no_grad():
        for stmap1, stmap2, bvp, _ in val_loader:
            stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
            output = model(stmap1, stmap2)
            loss = criterion(output, bvp)
            val_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_gt.append(bvp.cpu().numpy())
    print(f"Val Loss: {val_loss / len(val_loader):.4f}")

    scheduler.step()
    mae, mse, rmse, _, snr = calculate_hr_metrics(np.concatenate(all_preds), np.concatenate(all_gt))
    r = calculate_bvp_correlation(np.concatenate(all_preds), np.concatenate(all_gt))
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R: {r:.2f}, SNR: {snr:.2f}")
    torch.save(model.state_dict(), os.path.join(config["save_dir"], f"epoch{epoch+1}.pth"))

# ----------------------- Test Evaluation -----------------------
print("\n[INFO] Testing on held-out subjects...")
model.eval()
test_preds, test_gt = [], []
with torch.no_grad():
    for stmap1, stmap2, bvp, _ in tqdm(test_loader, desc="Testing"):
        stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
        output = model(stmap1, stmap2)
        test_preds.append(output.cpu().numpy())
        test_gt.append(bvp.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)
test_gt = np.concatenate(test_gt, axis=0)
mae, mse, rmse, _, snr = calculate_hr_metrics(test_preds, test_gt)
r = calculate_bvp_correlation(test_preds, test_gt)

print("\n📊 Final Test Performance:")
print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R: {r:.2f}, SNR: {snr:.2f}")

# ----------------------- Plot Test Subject BVP -----------------------
plot_subjectwise_bvp(test_set.get_subjectwise_bvp())
