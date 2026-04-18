import torch
import torch.nn as nn

# ---------------- Residual Encoder ----------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class DeepEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.layer1 = ResidualBlock(in_ch, base_ch)
        self.layer2 = ResidualBlock(base_ch, base_ch * 2, stride=2)
        self.layer3 = ResidualBlock(base_ch * 2, base_ch * 2)
        self.out_ch = base_ch * 2

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))  # (B, C, H↓, T↓)

# ---------------- Temporal Upsample Decoder ----------------

class TemporalUpsampleDecoder(nn.Module):
    def __init__(self, in_dim, scale_factor=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_dim, in_dim // 2, kernel_size=4, stride=scale_factor, padding=1),  # T x2
            nn.ReLU(),
            nn.Conv1d(in_dim // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, D, T↓)
        x = self.decoder(x)     # (B, 1, T)
        return x.squeeze(1)     # (B, T)

# ---------------- Full Model (No Attention) ----------------

class TransfuserSTMapModel(nn.Module):
    def __init__(self, ch_in=3, base_dim=64):
        super().__init__()
        self.rgb_encoder = DeepEncoder(ch_in, base_ch=base_dim)
        self.yuv_encoder = DeepEncoder(ch_in, base_ch=base_dim)
        self.temporal_decoder = None  # lazy init

    def forward(self, stmap_rgb, stmap_yuv):
        f_rgb = self.rgb_encoder(stmap_rgb)  # (B, F, H↓, T↓)
        f_yuv = self.yuv_encoder(stmap_yuv)  # (B, F, H↓, T↓)

        f_cat = torch.cat([f_rgb, f_yuv], dim=1)  # (B, 2F, H↓, T↓)
        B, C, H, T = f_cat.shape
        fused = f_cat.permute(0, 3, 2, 1).reshape(B, T, H * C)  # (B, T↓, D)

        if self.temporal_decoder is None:
            D = fused.shape[-1]
            self.temporal_decoder = TemporalUpsampleDecoder(in_dim=D).to(fused.device)

        out = self.temporal_decoder(fused)  # (B, T)
        return out