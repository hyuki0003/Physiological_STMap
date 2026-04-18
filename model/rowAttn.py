import torch
import torch.nn as nn
import torch.nn.functional as F

# YUV로부터 Row-wise Attention 추출
class RowAttentionExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2  # ⬅️ 정수형으로 변환
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, T] → attention: [B, 1, H, T]
        return self.attn(x)


class RowAttentionSoftmax(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2  # ⬅️ 정수형으로 변환
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        raw_attn = self.attn(x).mean(dim=3, keepdim=True)  # [B, 1, H, 1]
        attn = torch.softmax(raw_attn, dim=2)
        return attn


class RowCorrelationAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, x):
        # x: [B, C, H, T]
        B, C, H, T = x.shape
        x_mean = x.mean(dim=1)  # [B, H, T]
        x_norm = x_mean - x_mean.mean(dim=2, keepdim=True)
        x_norm = x_norm / (x_norm.std(dim=2, keepdim=True) + 1e-6)
        attn = torch.bmm(x_norm, x_norm.transpose(1, 2)) / T  # [B, H, H]
        return attn

class CrossRowCorrelationAttention(nn.Module):
    def forward(self, x1, x2):  # x1: RGB, x2: YUV
        # x1, x2: [B, C, H, T]
        B, C, H, T = x1.shape

        x1_row = x1.mean(dim=1)  # [B, H, T]
        x2_row = x2.mean(dim=1)  # [B, H, T]

        x1_norm = (x1_row - x1_row.mean(dim=2, keepdim=True)) / (x1_row.std(dim=2, keepdim=True) + 1e-6)
        x2_norm = (x2_row - x2_row.mean(dim=2, keepdim=True)) / (x2_row.std(dim=2, keepdim=True) + 1e-6)

        sim = torch.bmm(x1_norm, x2_norm.transpose(1, 2)) / T  # [B, H, H]

        return sim  # Row-wise correlation attention map