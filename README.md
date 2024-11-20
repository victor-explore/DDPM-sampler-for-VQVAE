# Denoising Diffusion Probabilistic Model (DDPM) with VQ-VAE

This repository implements a Denoising Diffusion Probabilistic Model (DDPM) combined with a Vector Quantized Variational Autoencoder (VQ-VAE) for high-quality image generation. The model is trained on a butterfly image dataset and evaluated using Fréchet Inception Distance (FID).

## Architecture Overview

The implementation consists of three main components:

1. **VQ-VAE**: Compresses images into discrete latent representations
   - Encoder: Converts images to continuous latent space
   - Vector Quantizer: Discretizes latent space using a codebook
   - Decoder: Reconstructs images from quantized representations

2. **U-Net**: Core of the diffusion model
   - Predicts noise at each timestep
   - Features skip connections and time embeddings
   - Optimized for 8x8 latent space dimensions

3. **Diffusion Process**: 
   - Forward process: Gradually adds noise to latents
   - Reverse process: Generates new samples by denoising

## Requirements

```python
torch
torchvision
numpy
PIL
matplotlib
scipy
