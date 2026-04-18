import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from .rowAttn import CrossRowCorrelationAttention


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

        # YUV decoder (for attention only)
        self.yuv_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.yuv_dec2 = UNetBlock(256, 128)
        self.yuv_up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.yuv_dec1 = UNetBlock(128, 64)

        # Attention modules
        self.attn1 = CrossRowCorrelationAttention()
        self.attn2 = CrossRowCorrelationAttention()
        self.attn3 = CrossRowCorrelationAttention()
        self.attn_d4 = CrossRowCorrelationAttention()

        # Decoder (attention-weighted skip connection)
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

        # YUV encoder
        y1 = self.yuv_enc1(x_yuv)
        y2 = self.yuv_enc2(self.yuv_pool1(y1))
        y3 = self.yuv_enc3(self.yuv_pool2(y2))

        # YUV decoder for final attention
        y_d2 = self.yuv_up2(y3)
        y_d2 = torch.cat([y_d2, y2], dim=1)
        y_d2 = self.yuv_dec2(y_d2)
        y_d1 = self.yuv_up1(y_d2)
        y_d1 = torch.cat([y_d1, y1], dim=1)
        y_d1 = self.yuv_dec1(y_d1)

        # Cross attention
        a1 = self.attn1(f1, y1)
        a2 = self.attn2(f2, y2)
        a3 = self.attn3(f3, y3)

        # attention-weighted skip
        a2_weight = a2.mean(dim=-1, keepdim=True).unsqueeze(1)  # [B, 1, H, 1]
        a1_weight = a1.mean(dim=-1, keepdim=True).unsqueeze(1)

        # a2_diag = a2.mean(dim=-1)  # [B, H]
        # attn_weight = a2_diag / (a2_diag.sum(dim=1, keepdim=True) + 1e-6)
        # a2_weight = attn_weight.unsqueeze(1).unsqueeze(-1)  # [B, 1, H, 1]
        #
        # a1_diag = a1.mean(dim=-1)  # [B, H]
        # attn_weight = a1_diag / (a1_diag.sum(dim=1, keepdim=True) + 1e-6)
        # a1_weight = attn_weight.unsqueeze(1).unsqueeze(-1)  # [B, 1, H, 1]

        d2 = self.up2(f3)
        d2 = torch.cat([d2, f2 * a2_weight], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, f1 * a1_weight], dim=1)
        d1 = self.dec1(d1)

        a4 = self.attn_d4(d1, y_d1)

        out = self.final(d1)

        # Attention visualization
        if self.training and epoch is not None and epoch % 10 == 0:
            os.makedirs("attn_vis", exist_ok=True)
            def save_attn(attn, name):
                attn_map = attn[0].detach().cpu().numpy()
                plt.imshow(attn_map, cmap="viridis", aspect='auto')
                plt.colorbar()
                plt.title(f"{name} Attention Map (epoch {epoch})")
                plt.savefig(f"attn_vis/{name}_epoch{epoch}.png")
                plt.close()
            save_attn(a1, "row_attn1")
            save_attn(a2, "row_attn2")
            save_attn(a3, "row_attn3")
            save_attn(a4, "row_attn4")

        # Temporal weighting
        a4_diag = a4.mean(dim=-1)  # [B, H]
        attn_weight = a4_diag / (a4_diag.sum(dim=1, keepdim=True) + 1e-6)
        a4_weight = attn_weight.unsqueeze(1).unsqueeze(-1)  # [B, 1, H, 1]

        # a4_weight = a4.mean(dim=-1, keepdim=True).unsqueeze(1)

        out_weighted = (out * a4_weight).sum(dim=2)  # [B, C, T]
        out_temporal = self.temporal_head(out_weighted)  # [B, 1, T]

        out = out_temporal + out_weighted.mean(dim=1, keepdim=True)
        return out.squeeze(1)  # [B, T]
