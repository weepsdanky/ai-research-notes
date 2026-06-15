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
import os

# ---------- config ----------
IMG_SIZE = 28
IN_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "samples"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- noise schedule ----------
def linear_beta_schedule(timesteps: int):
    return torch.linspace(BETA_START, BETA_END, timesteps)


class NoiseSchedule:
    def __init__(self, timesteps: int = NUM_TIMESTEPS):
        self.T = timesteps
        beta = linear_beta_schedule(timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self


# ---------- forward diffusion: q(x_t | x_0) ----------
def forward_diffusion(x0: torch.Tensor, t: torch.Tensor, noise_schedule: NoiseSchedule):
    alpha_bar_t = noise_schedule.alpha_bar[t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
    return xt, eps


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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim > 0 else None

    def forward(self, x, t_emb):
        h = F.silu(self.conv1(x))
        h = self.conv2(h)
        if self.time_mlp is not None:
            t_out = self.time_mlp(F.silu(t_emb))[:, :, None, None]
            h = h + t_out
        return F.silu(h)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)

        self.enc1 = ConvBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.dec3 = ConvBlock(base_channels * 4 * 2, base_channels * 2, time_emb_dim)
        self.dec2 = ConvBlock(base_channels * 2 * 2, base_channels, time_emb_dim)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, time_emb_dim)
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
def sample(model, noise_schedule, num_images=64):
    model.eval()
    x = torch.randn(num_images, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)

    for t in reversed(range(noise_schedule.T)):
        t_tensor = torch.full((num_images,), t, device=DEVICE, dtype=torch.long)
        beta_t = noise_schedule.beta[t]
        alpha_t = noise_schedule.alpha[t]
        alpha_bar_t = noise_schedule.alpha_bar[t]

        pred_eps = model(x, t_tensor)

        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        x = coeff1 * (x - coeff2 * pred_eps)

        if t > 0:
            noise = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            x = x + torch.sqrt(beta_t) * noise

    model.train()
    return x


# ---------- training ----------
def train():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = UNet().to(DEVICE)
    noise_schedule = NoiseSchedule(NUM_TIMESTEPS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    global_step = 0

    for epoch in range(EPOCHS):
        for x0, _ in loader:
            x0 = x0.to(DEVICE)
            t = torch.randint(0, NUM_TIMESTEPS, (x0.size(0),), device=DEVICE)
            xt, eps = forward_diffusion(x0, t, noise_schedule)
            pred_eps = model(xt, t)
            loss = F.mse_loss(pred_eps, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print(f"epoch {epoch+1}/{EPOCHS}  loss={loss.item():.6f}")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            samples = sample(model, noise_schedule, num_images=64)
            save_image(samples, os.path.join(OUT_DIR, f"epoch_{epoch+1}.png"), nrow=8, normalize=True, value_range=(-1, 1))

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "ddpm_mnist.pt"))
    print(f"done — samples saved to {OUT_DIR}/")


if __name__ == "__main__":
    train()
