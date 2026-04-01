# training/dataset.py
# PyTorch Dataset for loading document images and applying synthetic noise.
# During training, each clean image gets a randomly noised version as input.

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from training.augment import apply_random_noise


class DenoisingDataset(Dataset):
    """
    Dataset that loads clean grayscale images and creates noisy/clean pairs on the fly.

    - Scans a folder for image files (.png, .jpg, .jpeg, .bmp, .tiff)
    - Converts each image to grayscale and resizes to a fixed size
    - Applies random synthetic noise to create the input (noisy) image
    - The target (label) is the original clean image

    Args:
        data_dir: Path to folder containing clean document images.
        image_size: Resize all images to (image_size, image_size). Default 256.
        noise_strength: Maximum noise strength for augmentation. Default 0.1.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, data_dir: str, image_size: int = 256, noise_strength: float = 0.1):
        self.data_dir = data_dir
        self.image_size = image_size
        self.noise_strength = noise_strength

        # Collect all supported image file paths
        self.image_paths = []
        for fname in os.listdir(data_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                self.image_paths.append(os.path.join(data_dir, fname))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}. "
                             f"Supported formats: {self.SUPPORTED_EXTENSIONS}")

        print(f"[Dataset] Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            noisy_tensor: (1, H, W) float tensor — the noisy input
            clean_tensor: (1, H, W) float tensor — the clean target
        """
        # Load image as grayscale
        img = Image.open(self.image_paths[idx]).convert("L")

        # Resize to fixed dimensions
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to float32 numpy array in [0, 1]
        clean = np.array(img, dtype=np.float32) / 255.0

        # Apply random noise (strength varies each time for diversity)
        actual_strength = np.random.uniform(0.01, self.noise_strength)
        noisy = apply_random_noise(clean, strength=actual_strength)

        # Convert to PyTorch tensors: add channel dimension (1, H, W)
        clean_tensor = torch.from_numpy(clean).unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)

        return noisy_tensor, clean_tensor
