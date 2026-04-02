import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

from models import MODEL_REGISTRY
from training.augment import apply_random_noise

# Configuration
IMAGE_SIZE = 256
NOISE_STRENGTH = 0.15
DATA_DIR = "data/raw"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse)).item()

def calculate_mse(img1, img2):
    return torch.mean((img1 - img2) ** 2).item()

def load_trained_model(model_name, weight_path):
    model = MODEL_REGISTRY[model_name]()
    if not os.path.exists(weight_path):
        print(f"Warning: Weights not found at {weight_path}")
        return None
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate():
    print(f"Evaluating models on device: {device}")
    
    autoencoder = load_trained_model("Simple Autoencoder", "models/weights/simple_autoencoder.pth")
    unet = load_trained_model("U-Net", "models/weights/u-net.pth")
    
    if autoencoder is None or unet is None:
        print("One or both models could not be loaded. Please ensure training has generated weights.")
        return

    # Select 10 sample images for evaluation
    image_files = sorted(os.listdir(DATA_DIR))[:10]  # Just taking the first 10 for sample
    
    results = []

    for img_name in image_files:
        img_path = os.path.join(DATA_DIR, img_name)
        
        # Load and preprocess
        img = Image.open(img_path).convert("L")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        clean_np = np.array(img, dtype=np.float32) / 255.0
        
        # Apply noise
        noisy_np = apply_random_noise(clean_np, strength=NOISE_STRENGTH)
        
        # Convert to tensors
        clean_tensor = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).to(device)
        noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_ae = autoencoder(noisy_tensor)
            output_unet = unet(noisy_tensor)
            
        # Metrics for Autoencoder
        mse_ae = calculate_mse(output_ae, clean_tensor)
        psnr_ae = calculate_psnr(output_ae, clean_tensor)
        
        # Metrics for U-Net
        mse_unet = calculate_mse(output_unet, clean_tensor)
        psnr_unet = calculate_psnr(output_unet, clean_tensor)
        
        results.append({
            "Image": img_name,
            "AE_MSE": mse_ae,
            "AE_PSNR": psnr_ae,
            "UNet_MSE": mse_unet,
            "UNet_PSNR": psnr_unet
        })
        
        # Visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(clean_np, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(noisy_np, cmap="gray")
        axes[1].set_title(f"Noisy (Strength {NOISE_STRENGTH})")
        axes[1].axis("off")
        
        axes[2].imshow(output_ae.cpu().squeeze().numpy(), cmap="gray")
        axes[2].set_title(f"Autoencoder (PSNR: {psnr_ae:.2f})")
        axes[2].axis("off")
        
        axes[3].imshow(output_unet.cpu().squeeze().numpy(), cmap="gray")
        axes[3].set_title(f"U-Net (PSNR: {psnr_unet:.2f})")
        axes[3].axis("off")
        
        plt.tight_layout()
        viz_path = os.path.join(OUTPUT_DIR, f"compare_{img_name}")
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Processed {img_name}")

    # Write summary CSV
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "AE_MSE", "AE_PSNR", "UNet_MSE", "UNet_PSNR"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"Evaluation complete. Results saved to {OUTPUT_DIR}/")
    print(f"Metrics CSV saved to {csv_path}")

if __name__ == "__main__":
    evaluate()
