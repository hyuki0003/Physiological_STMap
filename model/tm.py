import torch
import torch.nn as nn

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

class SixChannelUNet(nn.Module):
    def __init__(self, in_channels=6):  # RGB + YUV
        super().__init__()

        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(128, 256)

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

    def forward(self, x):  # x: [B, 6, H, T]
        f1 = self.enc1(x)
        p1 = self.pool1(f1)
        f2 = self.enc2(p1)
        p2 = self.pool2(f2)
        f3 = self.bottleneck(p2)

        d2 = self.up2(f3)
        d2 = torch.cat([d2, f2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, f1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)             # [B, 32, H, T]
        out_avg = out.mean(dim=2)        # [B, 32, T]
        out_temp = self.temporal_head(out_avg)  # [B, 1, T]
        out = out_temp + out_avg.mean(dim=1, keepdim=True)  # [B, 1, T]
        return out.squeeze(1)  # [B, T]
