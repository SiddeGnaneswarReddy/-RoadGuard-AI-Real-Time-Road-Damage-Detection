"""
Backbone architectures for GuardRailNet.
Includes custom convolutional and residual feature extractors with Spatial & Channel Attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + Activation (SiLU / Swish)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention for guardrail feature recalibration."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        att_avg = self.fc2(F.silu(self.fc1(avg_pool)))
        att_max = self.fc2(F.silu(self.fc1(max_pool)))
        scale = torch.sigmoid(att_avg + att_max)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on linear edge features of road barriers."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        scale = torch.sigmoid(self.bn(self.conv(feat)))
        return x * scale


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with CBAM attention for deep guardrail feature representation."""

    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.attention = CBAMBlock(channels) if use_attention else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attention(out)
        out = self.act(out + residual)
        return out


class GuardBackbone(nn.Module):
    """
    Multi-Scale Road Feature Backbone.
    Extracts features at 3 distinct resolutions:
    - P3 (Stride 8): Fine-grained details (guardrail posts, small cracks, delineators)
    - P4 (Stride 16): Medium details (standard barrier segments)
    - P5 (Stride 32): Global contextual semantic features (highway horizon, road bounds)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Stem: [B, 3, H, W] -> [B, c, H/2, W/2]
        self.stem = nn.Sequential(
            ConvBlock(in_channels, c // 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(c // 2, c, kernel_size=3, stride=1, padding=1),
        )

        # Stage 1: [B, c, H/2, W/2] -> [B, c*2, H/4, W/4]
        self.stage1_down = ConvBlock(c, c * 2, kernel_size=3, stride=2, padding=1)
        self.stage1_blocks = nn.Sequential(
            ResidualBlock(c * 2),
            ResidualBlock(c * 2)
        )

        # Stage 2 (P3 output, stride 8): [B, c*2, H/4, W/4] -> [B, c*4, H/8, W/8]
        self.stage2_down = ConvBlock(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.stage2_blocks = nn.Sequential(
            ResidualBlock(c * 4),
            ResidualBlock(c * 4)
        )

        # Stage 3 (P4 output, stride 16): [B, c*4, H/8, W/8] -> [B, c*8, H/16, W/16]
        self.stage3_down = ConvBlock(c * 4, c * 8, kernel_size=3, stride=2, padding=1)
        self.stage3_blocks = nn.Sequential(
            ResidualBlock(c * 8),
            ResidualBlock(c * 8)
        )

        # Stage 4 (P5 output, stride 32): [B, c*8, H/16, W/16] -> [B, c*16, H/32, W/32]
        self.stage4_down = ConvBlock(c * 8, c * 16, kernel_size=3, stride=2, padding=1)
        self.stage4_blocks = nn.Sequential(
            ResidualBlock(c * 16),
            ResidualBlock(c * 16)
        )

        self.out_channels = [c * 4, c * 8, c * 16]  # e.g., [128, 256, 512]

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1_down(x)
        x = self.stage1_blocks(x)

        x = self.stage2_down(x)
        p3 = self.stage2_blocks(x)  # Stride 8

        x = self.stage3_down(p3)
        p4 = self.stage3_blocks(x)  # Stride 16

        x = self.stage4_down(p4)
        p5 = self.stage4_blocks(x)  # Stride 32

        return p3, p4, p5
