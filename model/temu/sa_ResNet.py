### ResNetWithSelfAttention.py

import torch
import torch.nn as nn
import os
import shutil
import matplotlib.pyplot as plt
from .selfAttn import YUVSpatialAttention

class RGBTemporalUNet(nn.Module):
    def __init__(self, input_channels=3, final_out_channels=160):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(1, 9), padding=(0, 4)),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, final_out_channels, kernel_size=(1, 5), padding=(0, 2)),  # 🔥 여기서 T개 채널 출력
            nn.ReLU()
        )

    def forward(self, x, attn1=None, attn2=None, attn3=None, attn_fin=None):
        x = self.encoder1(x)
        if attn1 is not None:
            # x = x * attn1
            x = x
        x = self.encoder2(x)
        if attn2 is not None:
            # x = x * attn2
            x = x
        x = self.encoder3(x)
        if attn3 is not None:
            # x = x * attn3
            x = x
        if attn_fin is not None:
            x = x* attn_fin
        return self.decoder(x)



class STMapModulationModel(nn.Module):
    def __init__(self, output_length=160):
        super().__init__()
        self.output_length = output_length
        self.rgb_branch = RGBTemporalUNet(input_channels=6, final_out_channels=output_length)
        self.yuv_attn = YUVSpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP

    def forward(self, x_rgb, x_yuv, epoch=None):
        if self.training:
            if epoch is not None and epoch % 20 == 19:
                os.makedirs("attn_vis", exist_ok=True)

        attn1, attn2, attn3, attn_fin = self.yuv_attn(x_yuv)
        x_concat = torch.cat([x_rgb, x_yuv], dim=1)
        rgb_feat = self.rgb_branch(x_concat, attn1=attn1, attn2=attn2, attn3=attn3, attn_fin=attn_fin)

        # 🔥 attention 저장
        if self.training and epoch is not None and epoch % 20 == 19:
            for i, attn in enumerate([attn1, attn2, attn3, attn_fin], start=1):
                mask = attn[0, 0].detach().cpu().numpy()  # 첫 배치 첫 채널
                plt.imshow(mask, cmap='viridis')
                plt.colorbar()
                plt.title(f"YUV Attention Map (layer {i}) - Epoch {epoch + 1}")
                plt.savefig(f"attn_vis/attn_epoch{epoch + 1}_layer{i}.png")
                plt.close()

        pooled_feat = self.pool(rgb_feat).squeeze(-1).squeeze(-1)
        return pooled_feat



def STMapModel(output_length=160):
    return STMapModulationModel(output_length=output_length)
