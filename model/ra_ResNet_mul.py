import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from .rowAttn import RowAttentionExtractor


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, attn=None):
        x = self.conv(x)
        if attn is not None:
            attn = F.interpolate(attn, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + x * attn  # Residual Attention
        return x


class RowAttentionUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # RGB Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(128, 256)

        # YUV Encoder for attention
        self.yuv_enc1 = UNetBlock(in_channels, 64)
        self.yuv_pool1 = nn.MaxPool2d(2)
        self.yuv_enc2 = UNetBlock(64, 128)
        self.yuv_pool2 = nn.MaxPool2d(2)
        self.yuv_enc3 = UNetBlock(128, 256)

        # Row Attention Modules
        self.attn1 = RowAttentionExtractor(64)
        self.attn2 = RowAttentionExtractor(128)
        self.attn3 = RowAttentionExtractor(256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x_rgb, x_yuv, epoch=None):
        # --- Attention path (YUV) ---
        y1 = self.yuv_enc1(x_yuv)
        y2 = self.yuv_enc2(self.yuv_pool1(y1))
        y3 = self.yuv_enc3(self.yuv_pool2(y2))

        a1 = self.attn1(y1)
        a2 = self.attn2(y2)
        a3 = self.attn3(y3)

        if self.training and epoch is not None and epoch % 10 == 0:
            os.makedirs("attn_vis", exist_ok=True)

            def save_attn(attn, name):
                attn_map = attn[0, 0].detach().cpu().numpy()
                plt.imshow(attn_map, cmap="viridis", aspect='auto')
                plt.colorbar()
                plt.title(f"{name} Attention Map (epoch {epoch})")
                plt.savefig(f"attn_vis/{name}_epoch{epoch}.png")
                plt.close()

            save_attn(a1, "row_attn1")
            save_attn(a2, "row_attn2")
            save_attn(a3, "row_attn3")

        # --- RGB encoding path with attention ---
        e1 = self.enc1(x_rgb, a1)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1, a2)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2, a3)

        # --- Decoder ---
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)  # [B, 1, H, T]
        out = out.squeeze(1)  # [B, H, T]
        return out.mean(dim=1)  # Row-wise 평균 → [B, T]
