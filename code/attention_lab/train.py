"""Train a tiny character LM on a deliberately repeated learning sample."""
import argparse
import json
from pathlib import Path
import time

import torch

from attention import TinyLM

# A controlled overfit experiment, not a general-language benchmark.
TEXT = ("the cat sat on the mat.\n"
        "the dog sat on the rug.\n"
        "the cat and the dog play.\n") * 32


@torch.no_grad()
def generate(model, prompt, chars, count=120):
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = [stoi[ch] for ch in prompt]
    for _ in range(count):
        x = torch.tensor([ids[-model.context:]], device=next(model.parameters()).device)
        logits, _ = model(x)
        ids.append(logits[0, -1].argmax().item())
    return "".join(chars[i] for i in ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/overfit"))
    args = parser.parse_args()
    if min(args.steps, args.batch, args.context) < 1 or args.context >= len(TEXT):
        parser.error("steps/batch/context must be positive, and context shorter than TEXT")
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use --device cpu")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)
    model = TinyLM(len(chars), context=args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    info = {"device": device, "parameters": sum(p.numel() for p in model.parameters()),
            "vocab": len(chars), "steps": args.steps, "context": args.context, "batch": args.batch}
    print(json.dumps(info), flush=True)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        starts = torch.randint(len(data) - args.context, (args.batch,)).tolist()
        x = torch.stack([data[i:i + args.context] for i in starts]).to(device)
        y = torch.stack([data[i + 1:i + args.context + 1] for i in starts]).to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step={step} train_loss={loss.item():.4f} elapsed={time.monotonic()-started:.1f}s", flush=True)
    print("generated:\n" + generate(model, "the ", chars), flush=True)
    torch.save({"model": model.state_dict(), "chars": chars, "context": args.context}, args.out / "model.pt")
    (args.out / "config.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
