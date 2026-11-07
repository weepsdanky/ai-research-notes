"""Small FP32 Muon variant for studying the update on MPS; no fused matrices."""
import torch
from torch import nn

from model import GatedResidual, Indexer, MoE, RMSNorm


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0.01, momentum=0.95, ns_steps=8):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                                     momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                momentum = state["momentum"]
                momentum.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp(momentum, group["momentum"]).float()
                transposed = update.shape[0] > update.shape[1]
                if transposed:
                    update = update.T
                update = update / (update.norm() + 1e-7)
                for _ in range(group["ns_steps"]):
                    gram = update @ update.T
                    update = 3.4445 * update + (-4.775 * gram + 2.0315 * gram @ gram) @ update
                if transposed:
                    update = update.T
                # Match the update RMS convention used by scalable Muon.
                scale = 0.2 * max(p.shape) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)


def make_optimizers(model, stage, name, lr):
    index_ids = {id(p) for m in model.modules() if isinstance(m, Indexer) for p in m.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in index_ids if stage == "distill" else
                         (id(p) not in index_ids if stage == "pretrain" else True))
    # These stay on AdamW even when they happen to be two-dimensional.
    excluded = {id(p) for m in model.modules()
                if isinstance(m, (nn.Embedding, GatedResidual, RMSNorm, Indexer))
                for p in m.parameters()}
    excluded.update(id(p) for p in model.head.parameters())
    excluded.update(id(p) for m in model.modules() if isinstance(m, MoE) for p in m.router.parameters())
    muon, decay, no_decay = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if name == "muon" and p.ndim == 2 and id(p) not in excluded:
            muon.append(p)
        elif p.ndim >= 2 and id(p) not in excluded:
            decay.append(p)
        else:
            no_decay.append(p)
    optimizers = []
    if muon:
        optimizers.append(Muon(muon, lr))
    if decay or no_decay:
        optimizers.append(torch.optim.AdamW([
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0}], lr=lr, foreach=False))
    return optimizers
