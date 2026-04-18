import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from .crossAttn import CrossAttention

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNetStage(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, stride=1):
        super().__init__()
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(in_channels, out_channels, stride, downsample)]
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)

class ResNetWithCrossAttention(nn.Module):
    def __init__(self, block, layers, shared_backbone=True, output_length=128):
        super().__init__()
        self.shared_backbone = shared_backbone
        self.output_length = output_length

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        def make_stages():
            in_c = 64
            s1 = ResNetStage(block, in_c, 64, layers[0], stride=1)
            in_c = 64 * block.expansion
            s2 = ResNetStage(block, in_c, 128, layers[1], stride=2)
            in_c = 128 * block.expansion
            s3 = ResNetStage(block, in_c, 256, layers[2], stride=2)
            in_c = 256 * block.expansion
            s4 = ResNetStage(block, in_c, 512, layers[3], stride=2)
            return s1, s2, s3, s4

        if shared_backbone:
            self.layer1, self.layer2, self.layer3, self.layer4 = make_stages()
            self.yuv_layer1 = self.layer1
            self.yuv_layer2 = self.layer2
            self.yuv_layer3 = self.layer3
            self.yuv_layer4 = self.layer4
        else:
            self.layer1, self.layer2, self.layer3, self.layer4 = make_stages()
            self.yuv_layer1, self.yuv_layer2, self.yuv_layer3, self.yuv_layer4 = make_stages()

        self.attn1 = CrossAttention(dim=64 * block.expansion)
        self.attn2 = CrossAttention(dim=128 * block.expansion)
        self.attn3 = CrossAttention(dim=256 * block.expansion)
        self.attn4 = CrossAttention(dim=512 * block.expansion)

        # Multi-level temporal predictors
        self.predictor1 = self._make_temporal_head(64 * block.expansion)
        self.predictor2 = self._make_temporal_head(128 * block.expansion)
        self.predictor3 = self._make_temporal_head(256 * block.expansion)
        self.predictor4 = self._make_temporal_head(512 * block.expansion)

    def _make_temporal_head(self, in_channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, self.output_length)),  # (B, C, 1, T)
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)  # → (B, 1, 1, T)
        )

    def forward(self, x_rgb, x_yuv, epoch=None):
        x_rgb = self.stem(x_rgb)
        x_yuv = self.stem(x_yuv)

        x1 = self.layer1(x_rgb)
        x_yuv = self.yuv_layer1(x_yuv)
        x1 = self._apply_attention(x1, x_yuv, self.attn1, epoch=epoch, layer_name='attn1')

        x2 = self.layer2(x1)
        x_yuv = self.yuv_layer2(x_yuv)
        x2 = self._apply_attention(x2, x_yuv, self.attn2, epoch=epoch, layer_name='attn2')

        x3 = self.layer3(x2)
        x_yuv = self.yuv_layer3(x_yuv)
        x3 = self._apply_attention(x3, x_yuv, self.attn3, epoch=epoch, layer_name='attn3')

        x4 = self.layer4(x3)
        x_yuv = self.yuv_layer4(x_yuv)
        x4 = self._apply_attention(x4, x_yuv, self.attn4, epoch=epoch, layer_name='attn4')

        p1 = self.predictor1(x1).squeeze(2).squeeze(1)  # [B, 1, 1, T] → [B, T]
        p2 = self.predictor2(x2).squeeze(2).squeeze(1)
        p3 = self.predictor3(x3).squeeze(2).squeeze(1)
        p4 = self.predictor4(x4).squeeze(2).squeeze(1)

        return p1, p2, p3, p4

    def _apply_attention(self, rgb_feat, yuv_feat, attn_layer, epoch=None, layer_name="attn"):
        B, C, H, W = rgb_feat.shape
        rgb_flat = rgb_feat.view(B, C, -1).permute(0, 2, 1)
        yuv_flat = yuv_feat.view(B, C, -1).permute(0, 2, 1)

        attn_mask = attn_layer(yuv_flat, rgb_flat, rgb_flat).sigmoid()
        attn_mask = attn_mask.permute(0, 2, 1).view(B, C, H, W)

        if self.training and epoch is not None and epoch % 20 == 0:
            os.makedirs("attn_vis", exist_ok=True)
            mask = attn_mask[0, 0].detach().cpu().numpy()
            plt.imshow(mask, cmap="viridis")
            plt.colorbar()
            plt.title(f"Attention Mask {layer_name} (Channel 0)")
            plt.savefig(f"attn_vis/{layer_name}_epoch{epoch}.png")
            plt.close()

        return rgb_feat + rgb_feat * attn_mask


def ResNet50WithAttention(shared_backbone=True, output_length=128):
    return ResNetWithCrossAttention(Bottleneck, [3, 4, 6, 3], shared_backbone=shared_backbone, output_length=output_length)
