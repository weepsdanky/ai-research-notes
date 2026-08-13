"""Plot DiT optimizer-step metrics written by train_cifar10_dit.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training(log_path: Path, output_path: Path) -> None:
    with log_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("step log contains no rows")
    steps = [int(row["step"]) for row in rows]
    loss = [float(row["loss"]) for row in rows]
    mse = [float(row["mse"]) for row in rows]
    vb = [float(row["vb"]) for row in rows]
    learning_rate = [float(row["learning_rate"]) for row in rows]
    throughput = [float(row["samples_per_second"]) for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, loss, alpha=0.5, label="hybrid loss")
    axes[0, 0].plot(steps, mse, alpha=0.7, label="epsilon MSE")
    axes[0, 0].set(title="Denoising objective", xlabel="optimizer step", ylabel="loss")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, vb, color="#e76f51", alpha=0.65)
    axes[0, 1].set(title="Learned-variance VB term", xlabel="optimizer step", ylabel="bits/dim")
    axes[1, 0].plot(steps, learning_rate, color="#2a9d8f")
    axes[1, 0].set(title="Learning rate", xlabel="optimizer step", ylabel="lr")
    axes[1, 1].plot(steps, throughput, color="#6c63ff", alpha=0.7)
    axes[1, 1].set(title="Training throughput", xlabel="optimizer step", ylabel="images / second")
    figure.suptitle("DiT CIFAR-10 training diagnostics", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=Path("training_curves.png"))
    args = parser.parse_args()
    plot_training(args.log, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
