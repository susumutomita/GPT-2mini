#!/usr/bin/env python3
"""
Minimal DDPM (Denoising Diffusion Probabilistic Model) implementation.

Single-file implementation for learning diffusion models.
Train on MNIST to generate handwritten digits.

Usage:
    python train_diffusion_mini.py --steps 5000
    python train_diffusion_mini.py --steps 10000 --sample
    python train_diffusion_mini.py --noise_schedule cosine --steps 5000
"""

import argparse
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DiffusionConfig:
    image_size: int = 28
    in_channels: int = 1
    model_channels: int = 64
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    noise_schedule: str = "linear"  # "linear" or "cosine"

    # Experiment switches
    disable_residual: bool = False
    disable_time_emb: bool = False


# =============================================================================
# Noise Scheduler
# =============================================================================

class NoiseScheduler:
    """
    Manages the noise schedule for diffusion process.

    Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    """

    def __init__(self, config: DiffusionConfig, device: torch.device):
        self.timesteps = config.timesteps
        self.device = device

        if config.noise_schedule == "linear":
            self.betas = torch.linspace(
                config.beta_start, config.beta_end, config.timesteps, device=device
            )
        elif config.noise_schedule == "cosine":
            # Cosine schedule from "Improved DDPM" paper
            steps = config.timesteps + 1
            s = 0.008  # small offset to prevent beta from being too small
            t = torch.linspace(0, config.timesteps, steps, device=device)
            alphas_bar = torch.cos((t / config.timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown noise schedule: {config.noise_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)

        # For sampling
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward process: add noise to x_0 to get x_t.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def step(self, model_output: torch.Tensor, t: int, x_t: torch.Tensor):
        """
        Reverse process: denoise x_t to get x_{t-1}.

        p(x_{t-1} | x_t) = N(mu, sigma^2)
        where mu = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * predicted_noise)
        """
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]

        # Predict x_0 direction
        pred_x0_coef = beta_t / sqrt_one_minus_alpha_bar_t
        mean = sqrt_recip_alpha_t * (x_t - pred_x0_coef * model_output)

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            return mean + variance * noise


# =============================================================================
# U-Net Components
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal time embedding, similar to Transformer positional encoding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """
    Residual block with time conditioning.

    Architecture:
        x -> Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> + time_emb -> SiLU -> out
        |                                                                           |
        +---------------------------- (skip connection) ----------------------------+
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        disable_residual: bool = False,
        disable_time_emb: bool = False,
    ):
        super().__init__()
        self.disable_residual = disable_residual
        self.disable_time_emb = disable_time_emb

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if not disable_time_emb:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        if not self.disable_time_emb:
            # Add time embedding
            t_emb_proj = self.time_mlp(t_emb)[:, :, None, None]
            h = h + t_emb_proj

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        if self.disable_residual:
            return h
        else:
            return h + self.skip(x)


class Downsample(nn.Module):
    """Downsample by factor of 2 using strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by factor of 2 using nearest neighbor + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# =============================================================================
# U-Net Model
# =============================================================================

class UNet(nn.Module):
    """
    U-Net architecture for noise prediction.

    Architecture:
        Input (1, 28, 28)
        -> Down blocks (with skip connections)
        -> Middle block
        -> Up blocks (concatenate skip connections)
        -> Output (1, 28, 28)
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        ch = config.model_channels
        time_emb_dim = ch * 4

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(ch),
            nn.Linear(ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(config.in_channels, ch, 3, padding=1)

        # Downsampling path
        self.down1 = ResBlock(ch, ch, time_emb_dim, config.disable_residual, config.disable_time_emb)
        self.down2 = ResBlock(ch, ch, time_emb_dim, config.disable_residual, config.disable_time_emb)
        self.downsample1 = Downsample(ch)  # 28 -> 14

        self.down3 = ResBlock(ch, ch * 2, time_emb_dim, config.disable_residual, config.disable_time_emb)
        self.down4 = ResBlock(ch * 2, ch * 2, time_emb_dim, config.disable_residual, config.disable_time_emb)
        self.downsample2 = Downsample(ch * 2)  # 14 -> 7

        # Middle
        self.mid1 = ResBlock(ch * 2, ch * 2, time_emb_dim, config.disable_residual, config.disable_time_emb)
        self.mid2 = ResBlock(ch * 2, ch * 2, time_emb_dim, config.disable_residual, config.disable_time_emb)

        # Upsampling path (with skip connections, so input channels are doubled)
        self.upsample1 = Upsample(ch * 2)  # 7 -> 14
        self.up1 = ResBlock(ch * 4, ch * 2, time_emb_dim, config.disable_residual, config.disable_time_emb)
        self.up2 = ResBlock(ch * 2, ch * 2, time_emb_dim, config.disable_residual, config.disable_time_emb)

        self.upsample2 = Upsample(ch * 2)  # 14 -> 28
        self.up3 = ResBlock(ch * 3, ch, time_emb_dim, config.disable_residual, config.disable_time_emb)
        self.up4 = ResBlock(ch, ch, time_emb_dim, config.disable_residual, config.disable_time_emb)

        # Output
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, config.in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_emb(t)

        # Initial conv
        x = self.init_conv(x)

        # Downsampling
        h1 = self.down1(x, t_emb)
        h2 = self.down2(h1, t_emb)
        h2_down = self.downsample1(h2)

        h3 = self.down3(h2_down, t_emb)
        h4 = self.down4(h3, t_emb)
        h4_down = self.downsample2(h4)

        # Middle
        mid = self.mid1(h4_down, t_emb)
        mid = self.mid2(mid, t_emb)

        # Upsampling with skip connections
        up = self.upsample1(mid)
        up = torch.cat([up, h4], dim=1)  # Skip connection
        up = self.up1(up, t_emb)
        up = self.up2(up, t_emb)

        up = self.upsample2(up)
        up = torch.cat([up, h2], dim=1)  # Skip connection
        up = self.up3(up, t_emb)
        up = self.up4(up, t_emb)

        # Output
        out = self.out_norm(up)
        out = F.silu(out)
        out = self.out_conv(out)

        return out


# =============================================================================
# Training
# =============================================================================

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_mnist_dataloader(batch_size: int = 64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Scale to [-1, 1]
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def run_training(
    config: DiffusionConfig,
    steps: int = 5000,
    batch_size: int = 64,
    lr: float = 2e-4,
    sample_every: int = 1000,
    save_path: str = None,
):
    """Train the diffusion model."""
    device = get_device()
    print(f"Device: {device}")

    # Model and scheduler
    model = UNet(config).to(device)
    scheduler = NoiseScheduler(config, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Data
    dataloader = get_mnist_dataloader(batch_size)
    data_iter = iter(dataloader)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for step in range(steps):
        # Get batch (cycle through dataset)
        try:
            batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch, _ = next(data_iter)

        batch = batch.to(device)

        # Sample timesteps
        t = scheduler.sample_timesteps(batch.size(0))

        # Add noise
        noise = torch.randn_like(batch)
        x_t = scheduler.add_noise(batch, t, noise)

        # Predict noise
        predicted_noise = model(x_t, t)

        # Loss (simple MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % 100 == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.4f}")

        # Sample
        if sample_every > 0 and (step + 1) % sample_every == 0:
            generate_samples(model, scheduler, config, device, f"samples_step{step+1}.png")

    # Final sample
    generate_samples(model, scheduler, config, device, "samples_final.png")

    # Save model
    if save_path:
        torch.save({
            "model": model.state_dict(),
            "config": config,
        }, save_path)
        print(f"Model saved to {save_path}")

    return model


# =============================================================================
# Sampling
# =============================================================================

@torch.no_grad()
def generate_samples(
    model: UNet,
    scheduler: NoiseScheduler,
    config: DiffusionConfig,
    device: torch.device,
    save_path: str,
    n_samples: int = 16,
):
    """Generate samples using reverse diffusion process."""
    was_training = model.training
    model.train(False)

    # Start from pure noise
    x = torch.randn(n_samples, config.in_channels, config.image_size, config.image_size, device=device)

    # Reverse diffusion
    for t in reversed(range(config.timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_tensor)
        x = scheduler.step(predicted_noise, t, x)

    # Denormalize and save
    x = (x + 1) / 2  # [-1, 1] -> [0, 1]
    x = x.clamp(0, 1)

    # Create grid
    grid = make_grid(x, nrow=4)
    save_image(grid, save_path)
    print(f"Samples saved to {save_path}")

    model.train(was_training)


def make_grid(images: torch.Tensor, nrow: int = 4, padding: int = 2) -> torch.Tensor:
    """Arrange images in a grid."""
    n, c, h, w = images.shape
    ncol = (n + nrow - 1) // nrow

    grid_h = ncol * h + (ncol + 1) * padding
    grid_w = nrow * w + (nrow + 1) * padding
    grid = torch.ones(c, grid_h, grid_w, device=images.device)

    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        y = padding + row * (h + padding)
        x = padding + col * (w + padding)
        grid[:, y:y+h, x:x+w] = img

    return grid


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image."""
    from PIL import Image
    import numpy as np

    # Convert to numpy
    arr = tensor.cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # Remove channel dim for grayscale
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="L")
    img.save(path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train minimal diffusion model on MNIST")

    # Training
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--sample_every", type=int, default=1000, help="Sample every N steps (0 to disable)")

    # Model
    parser.add_argument("--channels", type=int, default=64, help="Model channels")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "cosine"])

    # Experiment switches
    parser.add_argument("--disable_residual", type=int, default=0, help="Disable residual connections")
    parser.add_argument("--disable_time_emb", type=int, default=0, help="Disable time embedding")

    # Save/Load
    parser.add_argument("--save", type=str, default=None, help="Save model path")
    parser.add_argument("--sample", action="store_true", help="Generate samples after training")

    args = parser.parse_args()

    # Config
    config = DiffusionConfig(
        model_channels=args.channels,
        timesteps=args.timesteps,
        noise_schedule=args.noise_schedule,
        disable_residual=bool(args.disable_residual),
        disable_time_emb=bool(args.disable_time_emb),
    )

    print("=" * 60)
    print("Diffusion Model Training")
    print("=" * 60)
    print(f"Noise schedule: {config.noise_schedule}")
    print(f"Timesteps: {config.timesteps}")
    print(f"Channels: {config.model_channels}")
    print(f"Disable residual: {config.disable_residual}")
    print(f"Disable time embedding: {config.disable_time_emb}")
    print("=" * 60)

    # Train
    model = run_training(
        config=config,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        sample_every=args.sample_every,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
