# training/augment.py
# Synthetic noise generation for training document denoising models.
# Supports: Gaussian noise, salt-and-pepper noise, and Gaussian blur.
# These simulate real-world document degradation (scanner artifacts, fax noise, etc.)

import numpy as np
import cv2


def add_gaussian_noise(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """
    Add Gaussian (random) noise to an image.

    Args:
        image: Grayscale image as float32 array in [0, 1] range.
        strength: Standard deviation of noise (0.0 = no noise, 0.5 = heavy noise).

    Returns:
        Noisy image clipped to [0, 1].
    """
    noise = np.random.normal(0, strength, image.shape).astype(np.float32)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


def add_salt_and_pepper_noise(image: np.ndarray, strength: float = 0.05) -> np.ndarray:
    """
    Add salt-and-pepper (impulse) noise.
    Randomly sets pixels to 0 (pepper) or 1 (salt).

    Args:
        image: Grayscale image as float32 array in [0, 1].
        strength: Fraction of pixels affected (0.0 to 1.0).

    Returns:
        Noisy image.
    """
    noisy = image.copy()
    # Total number of affected pixels
    num_pixels = int(strength * image.size)

    # Salt (white pixels)
    salt_coords = tuple(np.random.randint(0, dim, num_pixels) for dim in image.shape)
    noisy[salt_coords] = 1.0

    # Pepper (black pixels)
    pepper_coords = tuple(np.random.randint(0, dim, num_pixels) for dim in image.shape)
    noisy[pepper_coords] = 0.0

    return noisy


def add_gaussian_blur(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to simulate out-of-focus or low-quality scans.

    Args:
        image: Grayscale image as float32 array in [0, 1].
        strength: Controls blur kernel size. Higher = blurrier.
                  Kernel size = 2 * int(strength * 5) + 1 (always odd, minimum 1).

    Returns:
        Blurred image.
    """
    kernel_size = max(1, int(strength * 5)) * 2 + 1  # Ensure odd number >= 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def apply_random_noise(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """
    Apply a random combination of noise types for training augmentation.
    This creates more diverse training data so the model generalizes better.

    Args:
        image: Grayscale image as float32 array in [0, 1].
        strength: Overall noise intensity.

    Returns:
        Noisy image with one or more noise types applied.
    """
    noisy = image.copy()

    # Randomly decide which noise types to apply
    if np.random.random() > 0.3:
        noisy = add_gaussian_noise(noisy, strength)
    if np.random.random() > 0.5:
        noisy = add_salt_and_pepper_noise(noisy, strength * 0.5)
    if np.random.random() > 0.6:
        noisy = add_gaussian_blur(noisy, strength * 0.5)

    return noisy
