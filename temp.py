import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from tqdm import tqdm
from torchvision import models

from dataloader import Data_DG
from utils.loss.loss import CombinedLoss
from utils.metrics import calculate_hr_metrics, calculate_std


# Reproducibility setup
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Configuration
config = {
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "frames_num": 128,
    "step": 128,
    "frames_overlap": 64,
    "step_overlap": 0,
    "channels": 6,
    "root_dir": "/media/neuroai/T7/rPPG/STMap_raw/PURE",
    "dataName": "PURE",
    "STMap1": "vid_stmap_rgb.png",
    "STMap2": "vid_stmap_yuv.png",
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Train/Validation split
split_ratio = 0.8
num_train = int(len(dataset) * split_ratio)
num_val = len(dataset) - num_train
train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val],
                                                   generator=torch.Generator().manual_seed(config["seed"]))

print(f"[INFO] Training samples: {num_train}, Validation samples: {num_val}")

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)


# Define the ResNet Model
class SimpleResNet(nn.Module):
    def __init__(self, input_channels=6, output_length=128):
        super(SimpleResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_length)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # (B, 6, H, W)
        return self.resnet(x)


# Initialize model
model = SimpleResNet(input_channels=config["channels"], output_length=config["frames_num"]).to(device)

# Initialize criterion and optimizer
criterion = NegPearsonCorrLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model
print("[INFO] Starting training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(config["save_dir"], exist_ok=True)

all_epoch_metrics = []

for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0.0

    for stmap1, stmap2, bvp in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
        optimizer.zero_grad()
        outputs = model(stmap1, stmap2)

        # Normalize outputs and ground truth before loss calculation
        outputs_norm = outputs - outputs.mean(dim=1, keepdim=True)
        outputs_norm = outputs_norm / (outputs_norm.std(dim=1, keepdim=True) + 1e-8)
        bvp_norm = bvp - bvp.mean(dim=1, keepdim=True)
        bvp_norm = bvp_norm / (bvp_norm.std(dim=1, keepdim=True) + 1e-8)

        loss = criterion(outputs_norm, bvp_norm)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds, all_gt = [], []
    with torch.no_grad():
        for stmap1, stmap2, bvp in val_loader:
            stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
            outputs = model(stmap1, stmap2)

            # Normalize outputs and ground truth before loss calculation
            outputs_norm = outputs - outputs.mean(dim=1, keepdim=True)
            outputs_norm = outputs_norm / (outputs_norm.std(dim=1, keepdim=True) + 1e-8)
            bvp_norm = bvp - bvp.mean(dim=1, keepdim=True)
            bvp_norm = bvp_norm / (bvp_norm.std(dim=1, keepdim=True) + 1e-8)

            loss = criterion(outputs_norm, bvp_norm)
            val_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_gt.append(bvp.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Val Loss: {avg_val_loss:.4f}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    # Normalize GT before metrics
    all_gt = (all_gt - np.mean(all_gt, axis=1, keepdims=True)) / (np.std(all_gt, axis=1, keepdims=True) + 1e-8)

    print("[DEBUG] Pred Mean/Std:", np.mean(all_preds), np.std(all_preds))
    print("[DEBUG] GT Mean/Std:", np.mean(all_gt), np.std(all_gt))

    mae, mse, rmse, correlation, snr = calculate_hr_metrics(all_preds, all_gt)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R: {correlation:.2f}, SNR: {snr:.2f}")

    all_epoch_metrics.append((mae, mse, rmse, correlation, snr))

    ckpt_path = os.path.join(config["save_dir"], f"epoch{epoch + 1}.pth")
    torch.save(model.state_dict(), ckpt_path)

    scheduler.step()

# Summary
metrics_array = np.array(all_epoch_metrics)
best_epoch = np.argmin(metrics_array[:, 1]) + 1
std_metrics = calculate_std(metrics_array)

print(f"\nBest Epoch: {best_epoch}")
print(f"MAE: {metrics_array[best_epoch - 1][0]:.2f} ± {std_metrics[best_epoch - 1][0]:.2f}")
print(f"MSE: {metrics_array[best_epoch - 1][1]:.2f} ± {std_metrics[best_epoch - 1][1]:.2f}")
print(f"RMSE: {metrics_array[best_epoch - 1][2]:.2f} ± {std_metrics[best_epoch - 1][2]:.2f}")
print(f"R: {metrics_array[best_epoch - 1][3]:.2f} ± {std_metrics[best_epoch - 1][3]:.2f}")
print(f"SNR: {metrics_array[best_epoch - 1][4]:.2f} ± {std_metrics[best_epoch - 1][4]:.2f}")
