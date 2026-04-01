# 📄 Document Denoiser

A beginner-friendly deep learning project for **image and PDF denoising** using PyTorch and Streamlit.

Upload a noisy document image or PDF → select a model → get a clean, restored output.

---

## 📁 Project Structure

```
doc_denoiser/
├── app.py                  # Streamlit UI (main entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   ├── raw/                # Place clean training images here
│   └── processed/          # (optional) preprocessed data
├── models/
│   ├── __init__.py         # Model registry
│   ├── autoencoder.py      # Simple Autoencoder
│   ├── unet.py             # U-Net
│   └── weights/            # Trained .pth files go here
├── training/
│   ├── __init__.py
│   ├── train.py            # Training script (CLI)
│   ├── dataset.py          # PyTorch Dataset
│   └── augment.py          # Synthetic noise generation
├── inference/
│   ├── __init__.py
│   ├── predict.py          # Model loading & inference
│   ├── image_utils.py      # Image pre/post-processing
│   └── pdf_utils.py        # PDF ↔ image conversion
├── utils/
│   ├── __init__.py
│   ├── io_utils.py         # File I/O helpers
│   └── visualization.py    # Streamlit display helpers
└── outputs/
    ├── images/             # Saved denoised images
    └── pdfs/               # Saved denoised PDFs
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd doc_denoiser
pip install -r requirements.txt
```

> **Note:** For PyTorch with GPU support, install the CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Prepare training data

Place clean document images (PNG, JPG, BMP, TIFF) in:

```
data/raw/
```

These should be clear, noise-free document scans. The training pipeline will add synthetic noise automatically.

### 3. Train a model

```bash
cd doc_denoiser

# Train the Simple Autoencoder (fast, good starting point)
python -m training.train --model "Simple Autoencoder" --data_dir data/raw --epochs 50

# Train U-Net (stronger, slower)
python -m training.train --model "U-Net" --data_dir data/raw --epochs 50
```

**Optional arguments:**
- `--batch_size 8` — images per batch (reduce if you run out of memory)
- `--lr 0.001` — learning rate
- `--image_size 256` — resize dimension
- `--noise_strength 0.15` — max noise intensity during training

Trained weights are saved automatically to `models/weights/`.

### 4. Launch the app

```bash
cd doc_denoiser
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🎮 How to Use the App

1. **Upload** an image (PNG/JPG/BMP/TIFF) or a PDF.
2. **Select** a model from the sidebar dropdown.
3. **Toggle** "Add synthetic noise" and adjust the strength slider.
4. **View** the original → noisy → restored comparison.
5. **Download** the restored result (image or PDF).

---

## 🧠 Models

| Model | Speed | Quality | Best For |
|---|---|---|---|
| **Simple Autoencoder** | ⚡ Fast | Good | Light noise, quick testing |
| **U-Net** | 🐢 Slower | Better | Heavy noise, fine detail preservation |

**Recommendation:** Start with the **Simple Autoencoder** to verify your pipeline works, then switch to **U-Net** for production quality.

---

## 📦 Where to Place Trained Weights

Model weights (`.pth` files) should be placed in:

```
models/weights/
├── simple_autoencoder.pth    # Weights for Simple Autoencoder
└── u-net.pth                 # Weights for U-Net
```

The training script saves weights here automatically. If you train on a different machine, just copy the `.pth` files to this folder.

---

## 🔧 How It Works

### Training
1. Clean images are loaded from `data/raw/`.
2. Random synthetic noise (Gaussian, salt-and-pepper, blur) is applied on-the-fly.
3. The model learns to map noisy → clean.
4. Loss = MSE between model output and the original clean image.

### Inference (Images)
1. Input image is converted to grayscale and resized to 256×256.
2. The model produces a denoised output.
3. The output is resized back to the original dimensions.

### Inference (PDFs)
1. Each page is extracted as an image (using PyMuPDF).
2. Each page image is denoised individually.
3. Denoised pages are combined back into a PDF.

---

## 🔊 Synthetic Noise Types

| Noise Type | What It Simulates |
|---|---|
| **Gaussian** | Random sensor noise, scanner artifacts |
| **Salt & Pepper** | Dead pixels, transmission errors |
| **Gaussian Blur** | Out-of-focus scans, low-quality captures |

During training, these are applied randomly in combination for diversity.

---

## 📝 Notes

- All processing is done in **grayscale** (single channel). Color support can be added by changing `in_channels` from 1 to 3.
- Image dimensions should be divisible by 16 for U-Net (the code handles this via resize).
- Training on **GPU** is strongly recommended for U-Net.
- The app gracefully handles missing weights — it shows a clear error message instead of crashing.

---

## 🛠️ Extending the Project

- **Add new models:** Create a new file in `models/`, define the class, add it to `MODEL_REGISTRY` in `models/__init__.py`.
- **Add new noise types:** Add functions to `training/augment.py` and include them in `apply_random_noise()`.
- **Color image support:** Change `1` to `3` for input/output channels in models.
- **Larger images:** Increase `image_size` (may need more GPU memory).
