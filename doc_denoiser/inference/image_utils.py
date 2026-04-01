# inference/image_utils.py
# Image preprocessing and postprocessing utilities for inference.
# Handles conversion between PIL images, numpy arrays, and PyTorch tensors.

import numpy as np
from PIL import Image
import torch


def preprocess_image(pil_image: Image.Image, target_size: int = 256) -> tuple:
    """
    Prepare a PIL image for model inference.

    Steps:
    1. Convert to grayscale
    2. Record original size (for restoring later)
    3. Resize to model input size
    4. Normalize to [0, 1] float tensor
    5. Add batch and channel dimensions

    Args:
        pil_image: Input PIL Image (can be RGB or grayscale).
        target_size: Resize to (target_size, target_size) for the model.

    Returns:
        tensor: (1, 1, H, W) float tensor ready for model input.
        original_size: (width, height) of the original image for restoring.
    """
    # Convert to grayscale
    gray = pil_image.convert("L")
    original_size = gray.size  # (width, height)

    # Resize to model input dimensions
    resized = gray.resize((target_size, target_size), Image.BILINEAR)

    # Convert to float32 numpy array [0, 1]
    arr = np.array(resized, dtype=np.float32) / 255.0

    # Convert to tensor: (1, 1, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    return tensor, original_size


def postprocess_output(tensor: torch.Tensor, original_size: tuple) -> Image.Image:
    """
    Convert model output tensor back to a PIL image at the original resolution.

    Args:
        tensor: Model output, shape (1, 1, H, W) or (1, H, W), values in [0, 1].
        original_size: (width, height) to resize back to.

    Returns:
        PIL Image in grayscale ("L" mode).
    """
    # Remove batch and channel dims
    if tensor.dim() == 4:
        arr = tensor.squeeze(0).squeeze(0)
    elif tensor.dim() == 3:
        arr = tensor.squeeze(0)
    else:
        arr = tensor

    # Move to CPU if needed, convert to numpy
    arr = arr.detach().cpu().numpy()

    # Clip to [0, 1] and convert to uint8
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)

    # Create PIL image and resize to original dimensions
    img = Image.fromarray(arr, mode="L")
    img = img.resize(original_size, Image.BILINEAR)

    return img


def add_noise_to_pil(pil_image: Image.Image, noise_strength: float = 0.1) -> Image.Image:
    """
    Add Gaussian noise to a PIL image (used in the Streamlit UI for preview).

    Args:
        pil_image: Input PIL image (grayscale).
        noise_strength: Noise standard deviation.

    Returns:
        Noisy PIL image.
    """
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, noise_strength, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    noisy_uint8 = (noisy * 255).astype(np.uint8)
    return Image.fromarray(noisy_uint8, mode="L")
