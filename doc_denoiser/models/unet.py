# models/unet.py
# U-Net architecture for document denoising.
# U-Net is stronger than a simple autoencoder because it uses skip connections
# that preserve fine details (edges, text strokes) during reconstruction.

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive Conv -> BatchNorm -> ReLU blocks (a U-Net building block)."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net for image denoising.

    Input:  1-channel grayscale image (batch, 1, H, W)
    Output: 1-channel denoised image (batch, 1, H, W)

    Architecture:
      - 4-level encoder (downsampling path)
      - Bottleneck
      - 4-level decoder (upsampling path) with skip connections
      - Final 1x1 conv + Sigmoid

    Skip connections concatenate encoder features with decoder features,
    helping the network preserve fine details like text and edges.
    """

    def __init__(self):
        super(UNet, self).__init__()

        # ---- Encoder (downsampling) ----
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(512, 1024)

        # ---- Decoder (upsampling) ----
        # Each up-conv halves channels; after concatenation with skip, channels double again
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 512 (up) + 512 (skip) = 1024 input

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final 1x1 convolution to map 64 channels -> 1 channel
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass with skip connections.
        x: (batch, 1, H, W) — H and W should be divisible by 16 for clean up/downsampling.
        """
        # Encoder
        e1 = self.enc1(x)           # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 1024, H/16, W/16)

        # Decoder with skip connections (concatenation)
        d4 = self.up4(b)                          # (B, 512, H/8, W/8)
        d4 = self._crop_and_concat(e4, d4)        # concat with encoder feature
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._crop_and_concat(e3, d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._crop_and_concat(e2, d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._crop_and_concat(e1, d1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return self.sigmoid(out)

    @staticmethod
    def _crop_and_concat(encoder_feat, decoder_feat):
        """
        Crop encoder feature map to match decoder feature map size, then concatenate.
        This handles cases where input dimensions are not perfectly divisible.
        """
        _, _, h, w = decoder_feat.size()
        encoder_feat = encoder_feat[:, :, :h, :w]
        return torch.cat([encoder_feat, decoder_feat], dim=1)
