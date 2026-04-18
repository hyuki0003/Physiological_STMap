import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from model.tm2_6ch import UNetReconAndFusion
from dataloader import Data_DG

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
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "save_path": "pretrained.pth",
    "recon_vis_dir": "recon_vis_pretrain",
    "root_dir": "/home/neuroai/Projects/DSTMap_v2/DST/UBFC",
    "dataName": "UBFC",
    "STMap1": "vid_stmap_rgb.png",
    "STMap2": "vid_stmap_yuv.png",
    "frames_num": 160,
    "step": 16,
    "frames_overlap": 80,
    "step_overlap": 8,
    "channels": 6,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

set_seed(42)

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

loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
model = UNetReconAndFusion(mode='pretrain').to(device)
optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

# -----------------------------
# 학습 루프
# -----------------------------
print("[INFO] Starting pretraining...")
for epoch in range(config["epochs"]):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (stmap1, stmap2, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        stmap1 = stmap1.to(config["device"])
        stmap2 = stmap2.to(config["device"])
        target = torch.cat([stmap1, stmap2], dim=1)  # [B, 6, H, T]

        optimizer.zero_grad()
        recon = model(stmap1, stmap2)  # [B, 6, H, T]
        loss = F.mse_loss(recon, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # ✅ 시각화 저장
        if batch_idx == 0 and epoch % 5 == 0:
            recon_rgb = recon[0, :3].detach().cpu().numpy()
            recon_yuv = recon[0, 3:].detach().cpu().numpy()
            orig_rgb = stmap1[0].detach().cpu().numpy()
            orig_yuv = stmap2[0].detach().cpu().numpy()

            def plot_stmap(stmap, title, fname):
                plt.figure(figsize=(10, 3))
                plt.imshow(np.mean(stmap, axis=0), cmap='viridis', aspect='auto')
                plt.colorbar()
                plt.title(title)
                plt.savefig(fname)
                plt.close()

            plot_stmap(orig_rgb, "RGB Input", f"{config['recon_vis_dir']}/epoch{epoch+1}_rgb_input.png")
            plot_stmap(recon_rgb, "RGB Recon", f"{config['recon_vis_dir']}/epoch{epoch+1}_rgb_recon.png")
            plot_stmap(orig_yuv, "YUV Input", f"{config['recon_vis_dir']}/epoch{epoch+1}_yuv_input.png")
            plot_stmap(recon_yuv, "YUV Recon", f"{config['recon_vis_dir']}/epoch{epoch+1}_yuv_recon.png")

    avg_loss = epoch_loss / len(loader)
    print(f"[Epoch {epoch+1}] Recon Loss: {avg_loss:.4f}")

# ✅ 저장
torch.save(model.state_dict(), config["save_path"])
print(f"[INFO] Pretrained model saved to {config['save_path']}")
