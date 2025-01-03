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
```

## Usage

1. Train VQ-VAE:
```python
# Initialize and train VQ-VAE
vqvae_model = VQVAE(embedding_dim=128, num_embeddings=1024)
# Train VQ-VAE on image dataset
```

2. Generate Latent Representations:
```python
# Get VQ representations of training images
vq_representations = get_vq_representations(dataloader, VQVAE_model, device)
```

3. Train Diffusion Model:
```python
# Initialize and train U-Net
u_net_model = UNet()
# Train on VQ representations
```

4. Generate Samples:
```python
# Generate new samples
generated_sample = generate_sample(loaded_unet_model, num_timesteps)
decoded_image = vqvae_model.decoder(generated_sample)
```

## Model Architecture Details

### VQ-VAE
- Encoder: 4 convolutional layers with batch normalization
- Vector Quantizer: 1024 embedding vectors
- Decoder: 4 transposed convolutional layers

### U-Net
- Input: 128-channel 8x8 latent representations
- Time embedding dimension: 256
- Skip connections between encoder and decoder
- Residual connections in each block

## Evaluation

The model's performance is evaluated using Fréchet Inception Distance (FID):
1. Extract features using Inception-v3
2. Calculate statistics of real and generated samples
3. Compute FID score between distributions

## License

MIT License

## Acknowledgments

- Implementation inspired by OpenAI's DALL-E
- VQ-VAE architecture based on van den Oord et al.
- Diffusion model based on Ho et al.
Add to Conversation
