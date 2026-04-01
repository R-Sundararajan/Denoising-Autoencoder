# models/__init__.py
# Central registry for all denoising models.

from .autoencoder import SimpleAutoencoder
from .unet import UNet

# Map model names (used in UI) to their classes
MODEL_REGISTRY = {
    "Simple Autoencoder": SimpleAutoencoder,
    "U-Net": UNet,
}


def get_model(name: str):
    """
    Get a model class by its display name.
    Returns an instantiated model (unloaded weights).
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
