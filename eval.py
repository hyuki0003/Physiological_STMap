import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import argparse

from model.tm2 import SharedRowAttentionUNet  # 모델 구조
from dataloader import Data_DG  # 또는 새로운 Dataset 클래스
from utils.metrics import calculate_hr_metrics, calculate_bvp_correlation


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_gt = [], []

    with torch.no_grad():
        for stmap1, stmap2, bvp in tqdm(dataloader, desc="Evaluating"):
            stmap1, stmap2, bvp = stmap1.to(device), stmap2.to(device), bvp.to(device)
            if bvp.ndim == 3:
                bvp = bvp.squeeze(1)

            pred = model(stmap1, stmap2)
            all_preds.append(pred.cpu().numpy())
            all_gt.append(bvp.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    mae, mse, rmse, _, snr = calculate_hr_metrics(all_preds, all_gt)
    corr = calculate_bvp_correlation(all_preds, all_gt)

    print(f"✅ Evaluation Results:")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R: {corr:.2f}, SNR: {snr:.2f}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔄 Loading model...")
    model = SharedRowAttentionUNet(in_channels=3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"✅ Loaded checkpoint: {args.ckpt}")

    print("📦 Loading dataset...")
    dataset = Data_DG(
        version="v_eval",
        channels=6,
        root_dir=args.root_dir,
        dataName=args.data_name,
        STMap1=args.stmap1,
        STMap2=args.stmap2,
        frames_num=args.frames_num,
        step=args.step,
        frames_overlap=args.frames_overlap,
        step_overlap=args.step_overlap
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    evaluate(model, loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--data_name', type=str, default="PURE")
    parser.add_argument('--stmap1', type=str, default="vid_processed_stmap_rgb.png")
    parser.add_argument('--stmap2', type=str, default="vid_processed_stmap_yuv.png")
    parser.add_argument('--frames_num', type=int, default=160)
    parser.add_argument('--step', type=int, default=128)
    parser.add_argument('--frames_overlap', type=int, default=0)
    parser.add_argument('--step_overlap', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args)
