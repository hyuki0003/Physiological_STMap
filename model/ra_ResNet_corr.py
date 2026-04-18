import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from .rowAttn import RowCorrelationAttention

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, attn=None):
        x = self.conv(x)
        if attn is not None:
            attn = F.interpolate(attn, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + x * attn
        return x

class RowAttentionUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # RGB Encoder: shallow: kernel=7, mid: kernel=5, deep: kernel=3
        self.enc1 = UNetBlock(in_channels, 64, kernel_size=7)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(128, 256, kernel_size=3)
        # YUV Encoder for Attention: 동일한 커널 크기 적용
        self.yuv_enc1 = UNetBlock(in_channels, 64, kernel_size=7)
        self.yuv_pool1 = nn.MaxPool2d(2)
        self.yuv_enc2 = UNetBlock(64, 128, kernel_size=5)
        self.yuv_pool2 = nn.MaxPool2d(2)
        self.yuv_enc3 = UNetBlock(128, 256, kernel_size=3)
        # Row Attention Modules (row-to-row correlation 기반)
        self.attn1 = RowCorrelationAttention(64)
        self.attn2 = RowCorrelationAttention(128)
        self.attn3 = RowCorrelationAttention(256)
        # Decoder (기본적으로 3x3 커널 사용)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128, kernel_size=3)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64, kernel_size=3)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x_rgb, x_yuv, epoch=None):
        # YUV Encoder Path for Attention
        y1 = self.yuv_enc1(x_yuv)                        # [B, 64, H, T]
        y2 = self.yuv_enc2(self.yuv_pool1(y1))             # [B, 128, H/2, T/2]
        y3 = self.yuv_enc3(self.yuv_pool2(y2))             # [B, 256, H/4, T/4]
        a1 = self.attn1(y1)  # [B, H, H]
        a2 = self.attn2(y2)  # [B, H/2, H/2]
        a3 = self.attn3(y3)  # [B, H/4, H/4]
        if self.training and epoch is not None and epoch % 10 == 0:
            os.makedirs("attn_vis", exist_ok=True)
            def save_attn(attn, name):
                attn_map = attn[0].detach().cpu().numpy()  # [H, H] (row-to-row correlation map)
                plt.imshow(attn_map, cmap="viridis", aspect='auto')
                plt.colorbar()
                plt.title(f"{name} Attention Map (epoch {epoch})")
                plt.savefig(f"attn_vis/{name}_epoch{epoch}.png")
                plt.close()
            save_attn(a1, "row_attn1")
            save_attn(a2, "row_attn2")
            save_attn(a3, "row_attn3")

        a1_diag = torch.diagonal(a1, dim1=1, dim2=2).unsqueeze(1).unsqueeze(-1)  # [B, 1, H, 1]
        a2_diag = torch.diagonal(a2, dim1=1, dim2=2).unsqueeze(1).unsqueeze(-1)  # [B, 1, H/2, 1]
        a3_diag = torch.diagonal(a3, dim1=1, dim2=2).unsqueeze(1).unsqueeze(-1)  # [B, 1, H/4, 1]
        e1 = self.enc1(x_rgb, a1_diag)                   # [B, 64, H, T]
        p1 = self.pool1(e1)                              # [B, 64, H/2, T/2]
        e2 = self.enc2(p1, a2_diag)                        # [B, 128, H/2, T/2]
        p2 = self.pool2(e2)                              # [B, 128, H/4, T/4]
        b_feature = self.bottleneck(p2, a3_diag)           # [B, 256, H/4, T/4]
        d2 = self.up2(b_feature)                         # [B, 128, H/2, T/2]
        d2 = torch.cat([d2, e2], dim=1)                   # [B, 256, H/2, T/2]
        d2 = self.dec2(d2)                               # [B, 128, H/2, T/2]
        d1 = self.up1(d2)                              # [B, 64, H, T]
        d1 = torch.cat([d1, e1], dim=1)                   # [B, 128, H, T]
        d1 = self.dec1(d1)                               # [B, 64, H, T]
        out = self.final(d1)                             # [B, 1, H, T]
        out = out.squeeze(1)                             # [B, H, T]
        out_weighted = (out * a1_diag.squeeze(1)).sum(dim=1) / (a1_diag.squeeze(1).sum(dim=1) + 1e-6)  # [B, T]
        return out_weighted
