import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class SharedRowAttentionUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # RGB encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(128, 256)

        # YUV encoder
        self.yuv_enc1 = UNetBlock(in_channels, 64)
        self.yuv_pool1 = nn.MaxPool2d(2)
        self.yuv_enc2 = UNetBlock(64, 128)
        self.yuv_pool2 = nn.MaxPool2d(2)
        self.yuv_enc3 = UNetBlock(128, 256)

        # YUV decoder (for final attention-like weighting)
        self.yuv_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.yuv_dec2 = UNetBlock(256, 128)
        self.yuv_up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.yuv_dec1 = UNetBlock(128, 64)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.final = nn.Conv2d(64, 32, kernel_size=1)

        self.temporal_head = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1)
        )

    def forward(self, x_rgb, x_yuv, epoch=None):
        # RGB encoder
        f1 = self.enc1(x_rgb)
        p1 = self.pool1(f1)
        f2 = self.enc2(p1)
        p2 = self.pool2(f2)
        f3 = self.bottleneck(p2)

        # YUV encoder-decoder
        y1 = self.yuv_enc1(x_yuv)
        y2 = self.yuv_enc2(self.yuv_pool1(y1))
        y3 = self.yuv_enc3(self.yuv_pool2(y2))
        y_d2 = self.yuv_up2(y3)
        y_d2 = torch.cat([y_d2, y2], dim=1)
        y_d2 = self.yuv_dec2(y_d2)
        y_d1 = self.yuv_up1(y_d2)
        y_d1 = torch.cat([y_d1, y1], dim=1)
        y_d1 = self.yuv_dec1(y_d1)

        # Decoder (no attention)
        d2 = self.up2(f3)
        d2 = torch.cat([d2, f2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, f1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)

        # Weighting using YUV feature (row-wise mean)
        row_weight = y_d1.mean(dim=-1, keepdim=True)  # [B, C, H, 1]
        row_weight = row_weight.mean(dim=1, keepdim=True)  # [B, 1, H, 1]
        norm_weight = row_weight / (row_weight.sum(dim=2, keepdim=True) + 1e-6)  # [B, 1, H, 1]

        out_weighted = (out * norm_weight).sum(dim=2)  # [B, C, T]
        out_temporal = self.temporal_head(out_weighted)  # [B, 1, T]
        out = out_temporal + out_weighted.mean(dim=1, keepdim=True)
        return out.squeeze(1)  # [B, T]
