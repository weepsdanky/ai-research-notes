"""Three explicit phases, one training loop. Run on the Mac mini, not the authoring host."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import Config, Model, Indexer
from optim import make_optimizers


def device_for(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable: inspect Apple Silicon / macOS / native arm64 Python first")
    return name


class Corpus:
    def __init__(self, directory, seq):
        path = Path(directory)
        self.meta = json.loads((path / "meta.json").read_text())
        if self.meta["tokenizer"] != "Qwen/Qwen3.8-Flash-Next":
            raise ValueError("This profile requires the locked official tokenizer")
        self.arrays = {}
        for split in ("train", "val"):
            file = path / f"{split}.bin"
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest != self.meta["files"][split]["sha256"]:
                raise ValueError(f"Data hash mismatch: {file}")
            self.arrays[split] = np.memmap(file, dtype="<u4", mode="r")
            if len(self.arrays[split]) <= seq:
                raise ValueError(f"Too few tokens in {file}")

    def batch(self, split, batch, seq, rng, device):
        data = self.arrays[split]
        starts = torch.randint(len(data) - seq, (batch,), generator=rng).tolist()
        ids = np.stack([data[i:i + seq + 1] for i in starts]).astype(np.int64)
        ids = torch.from_numpy(ids).to(device)
        return ids[:, :-1], ids[:, 1:]


@torch.no_grad()
def evaluate(model, corpora, args, device):
    model.eval()
    report = {}
    captured = []
    handles = [m.register_forward_hook(lambda module, inputs, output: captured.append(tuple(t.detach() for t in output)))
               for m in model.modules() if isinstance(m, Indexer)]
    for index, corpus in enumerate(corpora):
        rng = torch.Generator().manual_seed(1234)
        losses, kls, dense = [], [], []
        diagnostics = []
        for _ in range(args.eval_batches):
            captured.clear()
            x, y = corpus.batch("val", args.batch, args.seq, rng, device)
            logits, kl, _ = model(x, args.stage)
            losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
            kls.append(kl.item())
            for scores, valid, selected, tokens in captured:
                values=scores[valid[None].expand_as(scores)]
                diagnostics.append(((values!=0).float().mean().item(), values.std().item(),
                                    (tokens>=0).sum(-1).float().mean().item()))
            if args.stage == "sparse":
                full, _, _ = model(x, "pretrain")
                dense.append(F.cross_entropy(full.flatten(0, 1), y.flatten()).item())
        prefix = f"val{index}"
        report[prefix + "/lm"] = sum(losses) / len(losses)
        report[prefix + "/kl"] = sum(kls) / len(kls)
        if diagnostics:
            report[prefix+"/indexer_nonzero"] = sum(d[0] for d in diagnostics)/len(diagnostics)
            report[prefix+"/indexer_std"] = sum(d[1] for d in diagnostics)/len(diagnostics)
            report[prefix+"/selected_tokens"] = sum(d[2] for d in diagnostics)/len(diagnostics)
        if dense:
            report[prefix + "/dense_lm_same_weights"] = sum(dense) / len(dense)
    for handle in handles: handle.remove()
    model.train()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["pretrain", "distill", "sparse"], default="pretrain")
    p.add_argument("--size", choices=["smoke", "mac"], default="mac")
    p.add_argument("--data", required=True)
    p.add_argument("--mix-data", help="Optional second corpus, sampled with probability 0.1 per microbatch")
    p.add_argument("--out", required=True)
    checkpoints = p.add_mutually_exclusive_group()
    checkpoints.add_argument("--init", help="Start a new phase from weights; reset optimizer and schedule")
    checkpoints.add_argument("--resume", help="Continue SAME phase, data and schedule from last.pt")
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--ns-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seconds", type=int, default=1200, help="Stop safely after this wall-time budget per invocation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-mode", choices=["paper", "hf"], help="Explicit source profile; inherited on init/resume")
    p.add_argument("--gr-mode", choices=["gated", "ungated"], help="Explicit Mac gate ablation; inherited on init/resume")
    p.add_argument("--schedule", choices=["auto", "cosine", "constant"], default="auto")
    p.add_argument("--clip", type=float, default=0.5)
    args = p.parse_args()
    if min(args.steps, args.seq, args.batch, args.accum, args.eval_every, args.eval_batches, args.seconds) < 1:
        p.error("Counts and time budget must be positive")
    if args.lr <= 0 or args.seq < 4 or args.seq > 512:
        p.error("Use lr > 0 and 4 <= seq <= 512 for this reference implementation")
    if args.stage != "pretrain" and not (args.init or args.resume):
        p.error("Distillation and sparse training require a previous phase checkpoint")
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    corpora = [Corpus(args.data, args.seq)]
    if args.mix_data:
        corpora.append(Corpus(args.mix_data, args.seq))
    out = Path(args.out)
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        p.error("Resume requires the original output directory")
    saved = None
    if args.init or args.resume:
        saved = torch.load(args.init or args.resume, map_location="cpu", weights_only=True)
        config = Config(**saved["config"])
        if args.init:
            allowed = {"pretrain": {"pretrain"}, "distill": {"pretrain"}, "sparse": {"distill", "sparse"}}
            if saved["args"]["stage"] not in allowed[args.stage]:
                p.error("Wrong phase order: pretrain -> distill -> sparse")
            if saved["step"] < saved["args"]["steps"]:
                p.error("Previous phase has not finished. Resume it before starting the next phase")
    else:
        width = {"smoke": 32, "mac": 64}[args.size]
        config = Config(width=width, hidden=width//4, rank=width//8, heads=2 if args.size=="smoke" else 12, experts=4 if args.size=="smoke" else 8, score_mode=args.score_mode or "paper", gr_mode=args.gr_mode or "gated")
    if args.score_mode is not None and args.score_mode != config.score_mode:
        if not args.init or args.stage != "distill":
            p.error("Changing source score profile requires a NEW distillation run from a completed backbone")
        config.score_mode = args.score_mode
    args.score_mode = config.score_mode
    if args.gr_mode is not None and args.gr_mode != config.gr_mode:
        if not args.init or args.stage != "pretrain":
            p.error("GR ablation changes require a NEW pretraining run")
        config.gr_mode = args.gr_mode
    args.gr_mode = config.gr_mode
    model = Model(config).to(device)
    if saved:
        model.load_state_dict(saved["model"])
    optimizers = make_optimizers(model, args.stage, args.optimizer, args.lr,ns_dtype=args.ns_dtype,weight_decay=args.weight_decay)
    start = 0
    if args.resume:
        for key in ("stage", "optimizer", "ns_dtype", "weight_decay", "steps", "seq", "batch", "accum", "lr", "seed", "score_mode", "gr_mode", "schedule", "clip"):
            if vars(args)[key] != saved["args"][key]:
                p.error(f"Resume mismatch for {key}; repeat the original command options")
        if [c.meta for c in corpora] != saved["corpora"]:
            p.error("Resume data mismatch")
        if Path(args.resume).resolve() != (out / "last.pt").resolve():
            p.error("Resume the original run's last.pt, with its original --out")
        for optimizer, state in zip(optimizers, saved["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        start = saved["step"]
        rng.set_state(saved["rng"])
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                     for name in ("model.py", "optim.py", "polar_coefficients.json", "train.py")}
    if args.resume and saved.get("source_hashes") != source_hashes:
        p.error("Resume source mismatch; retain the original implementation or explicitly migrate a NEW run")
    del saved
    metadata = dict(config=asdict(config), args=vars(args), device=device,
                    torch=str(torch.__version__), platform=platform.platform(),
                    source_hashes=source_hashes, parameters=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    corpora=[c.meta for c in corpora])
    if not args.resume:
        (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    def log(row):
        print(json.dumps(row), flush=True)
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def save(step):
        state = dict(model=model.state_dict(), config=asdict(config), args=vars(args), step=step,
                     optimizers=[o.state_dict() for o in optimizers], rng=rng.get_state(),
                     corpora=[c.meta for c in corpora], source_hashes=source_hashes)
        torch.save(state, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")

    began = time.monotonic()
    if not args.resume:
        log(dict(step=0, **evaluate(model, corpora, args, device)))
        save(0)
    warmup = min(50, max(1, args.steps // 10))
    model.train()
    for step in range(start + 1, args.steps + 1):
        tick = time.monotonic()
        progress = max(0, step - warmup) / max(1, args.steps - warmup)
        factor = min(1, step / warmup) * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        if args.schedule == "constant" or (args.schedule == "auto" and args.stage != "pretrain"):
            factor = 1.0
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
        totals = torch.zeros(3, device=device)
        for _ in range(args.accum):
            index = int(len(corpora) == 2 and torch.rand((), generator=rng).item() < 0.1)
            x, y = corpora[index].batch("train", args.batch, args.seq, rng, device)
            logits, kl, balance = model(x, args.stage)
            lm = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss = kl if args.stage == "distill" else lm + 0.001 * balance + kl
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite loss; last.pt is the last saved safe point")
            (loss / args.accum).backward()
            totals += torch.stack((lm.detach(), kl.detach(), balance.detach())) / args.accum
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             args.clip, error_if_nonfinite=True, foreach=False)
        for optimizer in optimizers:
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        seconds = time.monotonic() - tick
        stop = step == args.steps or time.monotonic() - began >= args.seconds
        row = dict(step=step, lm=totals[0].item(), kl=totals[1].item(), balance=totals[2].item(),
                   grad_norm=norm.item(), lr=args.lr * factor, step_seconds=seconds,
                   tokens_seen=step * args.batch * args.seq * args.accum,
                   tokens_per_second=args.batch * args.seq * args.accum / seconds,
                   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                   (1 if platform.system() == "Darwin" else 1024))
        if device == "mps":
            row["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        if step % args.eval_every == 0 or stop:
            row.update(evaluate(model, corpora, args, device))
            save(step)
        log(row)
        if stop:
            log(dict(stopped="phase_complete" if step == args.steps else "time_budget",
                     step=step, checkpoint=str(out / "last.pt")))
            break


if __name__ == "__main__":
    main()
