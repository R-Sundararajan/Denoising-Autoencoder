# models/autoencoder.py
# A simple convolutional autoencoder for document denoising.
# Architecture: Encoder compresses the image, Decoder reconstructs it.
# This is a good baseline model — fast to train, easy to understand.

import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    """
    Simple Convolutional Autoencoder for image denoising.

    Input:  1-channel grayscale image (batch, 1, H, W)
    Output: 1-channel denoised image (batch, 1, H, W)

    The encoder reduces spatial dimensions while increasing feature depth.
    The decoder mirrors the encoder to reconstruct the clean image.
    """

    def __init__(self):
        super(SimpleAutoencoder, self).__init__()

        # ---- Encoder ----
        # Each block: Conv -> BatchNorm -> ReLU -> MaxPool
        self.encoder = nn.Sequential(
            # Block 1: 1 -> 32 channels, halve spatial dims
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            # Block 3: 64 -> 128 channels (bottleneck)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/8, W/8
        )

        # ---- Decoder ----
        # Each block: ConvTranspose (upsample) -> BatchNorm -> ReLU
        self.decoder = nn.Sequential(
            # Block 1: 128 -> 64 channels, double spatial dims
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2: 64 -> 32 channels
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 32 -> 1 channel (output), use Sigmoid for [0, 1] range
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass: encode the noisy input, then decode to reconstruct.
        x: (batch, 1, H, W) — grayscale noisy image, values in [0, 1]
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
