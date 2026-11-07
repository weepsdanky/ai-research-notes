"""Generate bytes from a locally trained checkpoint; recompute the prefix for clarity."""
import argparse

import torch

from model import Config, Model
from train import device_for


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = p.parse_args()
    if args.tokens < 1 or args.temperature <= 0 or not args.prompt:
        p.error("Need nonempty prompt, tokens > 0, temperature > 0")
    device = device_for(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Model(Config(**checkpoint["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    stage = "sparse" if checkpoint["args"]["stage"] == "sparse" else "pretrain"
    context = checkpoint["args"]["seq"]
    ids = list(args.prompt.encode("utf-8"))
    torch.manual_seed(42)
    for _ in range(args.tokens):
        x = torch.tensor([ids[-context:]], device=device)
        logits, _, _ = model(x, stage)
        probs = (logits[0, -1] / args.temperature).softmax(-1).cpu()
        token = torch.multinomial(probs, 1).item()
        if token == 256:
            break
        ids.append(token)
    print(bytes(ids).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
