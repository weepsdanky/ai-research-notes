"""Exact full-vocabulary CE with bounded projection chunks for Mac controls.

Checkpoint each projection during training so its token-by-vocabulary logits are
recomputed in backward. This does not truncate the vocabulary or the context.
It is a separate control-training component, not a measured long-context run.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def chunked_cross_entropy(hidden, weight, targets, chunk=16):
    """Mean next-token CE for valid token IDs; padding/ignored targets unsupported."""
    if chunk < 1 or hidden.shape[:-1] != targets.shape:
        raise ValueError('Positive chunk and aligned hidden/target shapes required')
    if targets.numel() == 0 or bool((targets < 0).any()):
        raise ValueError('Nonempty valid token targets required; no ignore-index labels')
    features = hidden.reshape(-1, hidden.shape[-1])
    labels = targets.reshape(-1)
    total = hidden.new_zeros((), dtype=torch.float32)

    def project_loss(x, head, y):
        return F.cross_entropy(F.linear(x, head).float(), y, reduction='sum')

    for start in range(0, len(labels), chunk):
        args = (features[start:start + chunk], weight, labels[start:start + chunk])
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            loss = checkpoint(project_loss, *args, use_reentrant=False)
        else:
            loss = project_loss(*args)
        total = total + loss
    return total / labels.numel()
