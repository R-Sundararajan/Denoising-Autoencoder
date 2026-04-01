# training/train.py
# Training script for document denoising models.
# Supports training either the Simple Autoencoder or U-Net.
#
# Usage:
#   python -m training.train --model autoencoder --data_dir data/raw --epochs 50
#   python -m training.train --model unet --data_dir data/raw --epochs 50
#
# Run from the doc_denoiser/ root directory.

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MODEL_REGISTRY
from training.dataset import DenoisingDataset


def train(
    model_name: str,
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    image_size: int = 256,
    noise_strength: float = 0.15,
    save_dir: str = "models/weights",
):
    """
    Train a denoising model on clean document images.

    Args:
        model_name: "Simple Autoencoder" or "U-Net"
        data_dir: Path to folder with clean training images
        epochs: Number of training epochs
        batch_size: Images per batch
        learning_rate: Optimizer learning rate
        image_size: Resize images to this square size
        noise_strength: Max noise strength for augmentation
        save_dir: Where to save trained weights
    """
    # --- Setup device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # --- Load model ---
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_name]().to(device)
    print(f"[Train] Model: {model_name}")

    # --- Prepare dataset and dataloader ---
    dataset = DenoisingDataset(
        data_dir=data_dir,
        image_size=image_size,
        noise_strength=noise_strength,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # --- Loss function and optimizer ---
    # MSE loss measures pixel-wise difference between denoised output and clean target
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training loop ---
    print(f"[Train] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Forward pass
            output = model(noisy)
            loss = criterion(output, clean)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch [{epoch}/{epochs}]  Loss: {avg_loss:.6f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            os.makedirs(save_dir, exist_ok=True)
            # Create filename from model name (e.g., "simple_autoencoder.pth")
            weight_name = model_name.lower().replace(" ", "_") + ".pth"
            save_path = os.path.join(save_dir, weight_name)
            torch.save(model.state_dict(), save_path)
            print(f"  [Checkpoint] Saved to {save_path}")

    # --- Save final weights ---
    os.makedirs(save_dir, exist_ok=True)
    weight_name = model_name.lower().replace(" ", "_") + ".pth"
    save_path = os.path.join(save_dir, weight_name)
    torch.save(model.state_dict(), save_path)
    print(f"[Train] Training complete! Final weights saved to: {save_path}")

    return save_path


# --- Command-line interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a document denoising model")
    parser.add_argument(
        "--model",
        type=str,
        default="Simple Autoencoder",
        choices=["Simple Autoencoder", "U-Net"],
        help="Which model to train",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to folder containing clean training images",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=256, help="Image resize dimension")
    parser.add_argument("--noise_strength", type=float, default=0.15, help="Max noise strength")

    args = parser.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        noise_strength=args.noise_strength,
    )
