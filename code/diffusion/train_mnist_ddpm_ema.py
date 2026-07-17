"""
train_mnist_ddpm.py — DDPM on MNIST (Module 1)

Core math reference:
  L_simple = E_{t,x0,eps} [ || eps - eps_theta(x_t, t) ||^2 ]

Usage:
  python train_mnist_ddpm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import csv
import os
import random
import time

import numpy as np

# ---------- config ----------
SEED = 42
IMG_SIZE = 28
IN_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 40
LR = 2e-4
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.9999
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
SAMPLE_VARIANCE_TYPE = "posterior"  # "beta" or "posterior"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "samples"
LOG_CSV = os.path.join(OUT_DIR, "training_log.csv")
LOSS_PLOT = os.path.join(OUT_DIR, "training_loss.png")
os.makedirs(OUT_DIR, exist_ok=True)


def save_loss_plot(history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    mean_losses = [row["loss_mean"] for row in history]
    last_losses = [row["loss_last"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_losses, marker="o", linewidth=2, label="epoch mean loss")
    plt.plot(epochs, last_losses, alpha=0.35, linewidth=1, label="last batch loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("DDPM MNIST training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=150)
    plt.close()


def write_training_log(history):
    fieldnames = ["epoch", "global_step", "loss_mean", "loss_last", "learning_rate", "epoch_seconds"]
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


# ---------- noise schedule ----------
def linear_beta_schedule(timesteps: int):
    return torch.linspace(BETA_START, BETA_END, timesteps)


class NoiseSchedule:
    def __init__(self, timesteps: int = NUM_TIMESTEPS):
        self.T = timesteps
        beta = linear_beta_schedule(timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.posterior_variance = posterior_variance

    def to(self, device): # load tensor to GPU
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


# ---------- forward diffusion: q(x_t | x_0) ----------
def forward_diffusion(x0: torch.Tensor, t: torch.Tensor, noise_schedule: NoiseSchedule):
    alpha_bar_t = noise_schedule.alpha_bar[t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
    return xt, eps


# ---------- Exponential Moving Average (EMA) ----------
class EMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])


# ---------- simple UNet ----------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=t.device) * (-torch.log(torch.tensor(10000.0)) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def _group_count(channels: int, max_groups: int = 8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)

        self.enc1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.dec3 = ResBlock(base_channels * 4 * 2, base_channels * 2, time_emb_dim)
        self.dec2 = ResBlock(base_channels * 2 * 2, base_channels, time_emb_dim)
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)
        self.out = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        s1 = self.enc1(x, t_emb)
        s2 = self.enc2(self.pool(s1), t_emb)
        s3 = self.enc3(self.pool(s2), t_emb)

        b = self.bottleneck(self.pool(s3), t_emb)

        b_up = F.interpolate(b, size=s3.shape[-2:], mode="nearest")
        d3 = self.dec3(torch.cat([b_up, s3], dim=1), t_emb)

        d3_up = F.interpolate(d3, size=s2.shape[-2:], mode="nearest")
        d2 = self.dec2(torch.cat([d3_up, s2], dim=1), t_emb)

        d2_up = F.interpolate(d2, size=s1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat([d2_up, s1], dim=1), t_emb)

        return self.out(d1)


# ---------- sampling (DDPM ancestral) ----------
@torch.no_grad()
def sample(model, noise_schedule, num_images=64, variance_type=SAMPLE_VARIANCE_TYPE, device=DEVICE):
    if variance_type not in {"beta", "posterior"}:
        raise ValueError("variance_type must be 'beta' or 'posterior'")

    model.eval()
    x = torch.randn(num_images, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)

    for t in reversed(range(noise_schedule.T)):
        t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)
        beta_t = noise_schedule.beta[t]
        alpha_t = noise_schedule.alpha[t]
        alpha_bar_t = noise_schedule.alpha_bar[t]
        variance_t = beta_t if variance_type == "beta" else noise_schedule.posterior_variance[t]

        pred_eps = model(x, t_tensor)

        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        x = coeff1 * (x - coeff2 * pred_eps)

        if t > 0:
            x = x + torch.sqrt(variance_t) * torch.randn_like(x)

    model.train()
    return x


# ---------- training ----------
def train():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tf = transforms.Compose([
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize([0.5], [0.5]), # normalize to [-1, 1]
    ]) # transforms for MNIST dataset
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        generator=loader_generator,
    )

    model = UNet().to(DEVICE) # initialize the model
    noise_schedule = NoiseSchedule(NUM_TIMESTEPS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ema = EMA(model, decay=EMA_DECAY)
    global_step = 0
    history = []

    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for x0, _ in loader:
            x0 = x0.to(DEVICE)
            t = torch.randint(0, NUM_TIMESTEPS, (x0.size(0),), device=DEVICE)
            xt, eps = forward_diffusion(x0, t, noise_schedule)
            pred_eps = model(xt, t)
            loss = F.mse_loss(pred_eps, eps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            ema.update(model)
            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_batches += 1

        epoch_row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "loss_mean": epoch_loss_sum / epoch_batches,
            "loss_last": loss.item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(epoch_row)
        write_training_log(history)
        save_loss_plot(history)

        print(
            f"epoch {epoch+1}/{EPOCHS}  "
            f"loss_mean={epoch_row['loss_mean']:.6f}  "
            f"loss_last={epoch_row['loss_last']:.6f}"
        )

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            sample_model = UNet().to(DEVICE)
            sample_model.load_state_dict(model.state_dict())
            ema.copy_to(sample_model)
            samples = sample(sample_model, noise_schedule, num_images=64)
            save_image(samples, os.path.join(OUT_DIR, f"epoch_{epoch+1}.png"), nrow=8, normalize=True, value_range=(-1, 1))

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "ddpm_mnist.pt"))
    ema_model = UNet().to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    ema.copy_to(ema_model)
    torch.save(ema_model.state_dict(), os.path.join(OUT_DIR, "ddpm_mnist_ema.pt"))
    print(f"done — samples, log, and loss plot saved to {OUT_DIR}/")


if __name__ == "__main__":
    train()
