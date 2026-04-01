# inference/predict.py
# Inference engine: loads model weights and runs denoising on images.
# Used by both the Streamlit app and standalone inference.

import os
import torch
from PIL import Image

from models import MODEL_REGISTRY
from inference.image_utils import preprocess_image, postprocess_output


# Default paths for model weights (relative to doc_denoiser/ root)
WEIGHT_PATHS = {
    "Simple Autoencoder": os.path.join("models", "weights", "simple_autoencoder.pth"),
    "U-Net": os.path.join("models", "weights", "u-net.pth"),
}


def load_model(model_name: str, weights_path: str = None, device: str = None):
    """
    Instantiate a model and load its trained weights.

    Args:
        model_name: "Simple Autoencoder" or "U-Net"
        weights_path: Path to .pth file. If None, uses default path.
        device: "cpu" or "cuda". If None, auto-detects.

    Returns:
        model: Loaded PyTorch model in eval mode.
        device: torch.device used.

    Raises:
        FileNotFoundError: If weights file doesn't exist.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Get model class from registry
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model = MODEL_REGISTRY[model_name]()

    # Resolve weights path
    if weights_path is None:
        weights_path = WEIGHT_PATHS.get(model_name, "")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            f"Please train the model first using:\n"
            f"  python -m training.train --model \"{model_name}\""
        )

    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[Predict] Loaded {model_name} from {weights_path} on {device}")
    return model, device


def denoise_image(model, device, pil_image: Image.Image, target_size: int = 256) -> Image.Image:
    """
    Run denoising inference on a single PIL image.

    Args:
        model: Loaded PyTorch model (in eval mode).
        device: torch.device.
        pil_image: Input noisy image (any mode — will be converted to grayscale).
        target_size: Model input size.

    Returns:
        Denoised PIL Image at the original resolution.
    """
    # Preprocess: PIL -> tensor
    input_tensor, original_size = preprocess_image(pil_image, target_size=target_size)
    input_tensor = input_tensor.to(device)

    # Run inference (no gradient computation needed)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess: tensor -> PIL at original size
    result = postprocess_output(output_tensor, original_size)
    return result


def check_weights_exist(model_name: str) -> bool:
    """Check if trained weights exist for a given model."""
    weights_path = WEIGHT_PATHS.get(model_name, "")
    return os.path.isfile(weights_path)
