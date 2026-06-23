# Document Denoising Models

This project evaluates document denoising systems for document signal restoration, comparing the final **Fully Trained U-Net (50 Epochs)** against the baseline **Simple Autoencoder** using quantitative testing, qualitative visual evidence, and Signal Processing analysis.

## Features

- Document signal restoration
- Comparison of **Fully Trained U-Net (50 Epochs)** and **Simple Autoencoder**
- Synthetic signal degradation using:
  - Gaussian thermal noise
  - Impulse Salt & Pepper noise
  - Spatial Gaussian blurring
- Quantitative evaluation using PSNR and MSE
- Qualitative visual comparison of denoising outputs
- U-Net skip connections for preserving high-frequency structural details

## Problem Statement

Document images contain both broad low-frequency background regions and ultra-high-frequency structural details such as sharp black ink text strokes. Noise removal must suppress randomized signal degradation while preserving text edges, lines, and thin strokes.

The **Simple Autoencoder** struggles with preserving structural high frequencies because its low-dimensional central bottleneck behaves as a spatial low-pass filter. This removes noise but also destroys high-frequency gradients, resulting in blurred and illegible text.

## Solution Overview

The final system uses a **U-Net** architecture for document restoration. The U-Net adds **Skip Connections** that bypass the compression bottleneck, preserving sharp high-frequency text-edge geometry while allowing the downsampling path to smooth broad low-frequency background signals.

The decoder combines the filtered background state with preserved high-frequency geometry, subtracting predicted noise while anchoring structural elements in place.

## System Architecture

### Document Signal Topology

A raw document image contains two radically different bands of spatial frequencies:

1. **Low-Frequency Data**: The broad, flat white space of the paper background.
2. **Ultra-High-Frequency Data**: The sudden, extreme spatial gradients forming the sharp edges of black ink text strokes.

### Simple Autoencoder

The Simple Autoencoder operates by squeezing the image signal through a low-dimensional central bottleneck.

- **The Flaw**: This compression inherently acts as a brutal **spatial low-pass filter**.
- **The Result**: The decoder rebuilds from a smoothed latent state, producing a signal that is clean of noise but permanently blurred.

### U-Net

The U-Net redesigns the filtering paradigm by adding **Skip Connections**.

- **The Downsampling Path (Low-Pass Filter)**: Condenses the image and smooths broad, low-frequency background signals.
- **The Skip Connections (High-Pass Passthrough)**: Bypass the compression bottleneck and preserve sharp spatial gradients of input text edges.
- **The Decoder (Non-Linear Combiner)**: Receives both the cleanly filtered background state and the preserved high-frequency geometry.

## Technologies Used

### Models

- Fully Trained U-Net (50 Epochs)
- Simple Autoencoder (30 Epochs)

### Architecture Concepts

- Skip Connections
- Downsampling Path
- Decoder
- Spatial low-pass filter
- High-pass passthrough
- Non-linear adaptive filter

### Evaluation Metrics

- Peak Signal-to-Noise Ratio (PSNR)
- Mean Squared Error (MSE)

### Processing and Deployment Files

- `augment.py`
- `u-net.pth`
- `app.py`
- Streamlit environment

### Synthetic Degradation

- Gaussian thermal noise
- Impulse Salt & Pepper noise
- Spatial Gaussian blurring

## Methodology

A representative selection of diverse documents was tested using a rigid protocol with a **15%** injection of aggressive, mixed synthetic signal degradation:

- Gaussian thermal noise
- Impulse Salt & Pepper noise
- Spatial Gaussian blurring

The evaluation compares the **Simple Autoencoder** baseline against the final **U-Net (50 Epochs)** model.

The synthetic corruption engine `augment.py` trained the model against Gaussian statics, Impulse bursts, and Gaussian blurs at a **15%** strength factor.

## Results

### Final Core Metrics

| Metric | Simple Autoencoder (Baseline) | Final U-Net (50 Epochs) | Improvement Delta |
| :--- | :---: | :---: | :---: |
| **Peak PSNR (Printed Text)** | 26.02 dB | **40.92 dB** | `+14.90 dB` |
| **Peak PSNR (Diagram/Line)** | 28.89 dB | **32.98 dB** | `+4.09 dB` |
| **Average PSNR (All Samples)** | ~21.50 dB | **~28.50 dB** | `+7.00 dB` |
| **Average MSE (Error Rate)** | ~0.0070 | **~0.0028** | `-60% Error` |

The U-Net reaches mathematically exceptional levels of fidelity on structured text (**40.9 dB**). In image processing, any PSNR above **35 dB** denotes that human observers can barely discern a difference between the reconstruction and the absolute, pristine original.

## Screenshots / Examples

### Printed Dense Text

Evaluation file: `img_0.jpg`  
PSNR: **40.9 dB**

This sample demonstrates background static removal while preserving typeface sharpness.

![comparative output](/images/compare_img_0.jpg)

### General Forms & Lines

Evaluation file: `img_10.jpg`  
PSNR: **32.9 dB**

This sample demonstrates continuous structural boundaries and table lines without edge-bleeding.

![comparative output](/images/compare_img_10.jpg)

### Cursive / Light Outlines

Evaluation file: `img_10003.jpg`  
PSNR: **31.4 dB**

This sample demonstrates preservation of multi-pixel thin lines of faint ink against aggressive impulse noise.

![comparative output](/images/compare_img_10003.jpg)

## Usage

The `u-net.pth` model should be utilized as the default production model inside the `app.py` Streamlit environment.

## Future Improvements

- Improve model by training on more epochs with rgb images.

