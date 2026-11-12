"""Exact full-vocabulary CE with immediate per-token-chunk backward.

Detach the hidden boundary, accumulate its adjoint and the head gradient, then
let the caller backpropagate that adjoint through the backbone once. No token or
vocabulary classes are discarded. Validation still uses the original CE.
"""
import torch
from torch.nn import functional as F


def streamed_lm_backward(hidden,weight,targets,chunk=16):
    if not torch.is_grad_enabled() or not weight.requires_grad:raise ValueError('Training gradients required')
    if chunk<1 or hidden.shape[:-1]!=targets.shape or targets.numel()==0 or bool((targets<0).any()):raise ValueError('Aligned valid targets and positive chunk required')
    boundary=hidden.detach().requires_grad_(True);flat=boundary.reshape(-1,boundary.shape[-1]);labels=targets.reshape(-1);parts=[]
    # Reverse backward order matches the usual reverse traversal of the original
    # chained chunk-loss sum; retain only detached scalar values for forward sum.
    for start in reversed(range(0,len(labels),chunk)):
        loss=F.cross_entropy(F.linear(flat[start:start+chunk],weight).float(),labels[start:start+chunk],reduction='sum')
        if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite full-vocabulary CE')
        (loss/labels.numel()).backward();parts.append(loss.detach())
    total=hidden.new_zeros((),dtype=torch.float32)
    for loss in reversed(parts):total=total+loss
    return total/labels.numel(),boundary.grad
