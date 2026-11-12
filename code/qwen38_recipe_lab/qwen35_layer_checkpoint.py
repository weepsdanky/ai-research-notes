"""Explicit long-training memory variant with unchanged parameters and operators.

Reentrant whole-layer rematerialization avoids retaining all recurrent autograd
nodes across layers. The original control sources and failed32K run stay frozen.
This is CPU activation recomputation, not a sparse/fused attention kernel.
"""
import torch
from torch.utils.checkpoint import checkpoint
from qwen35_controls import Qwen35Control


class LayerCheckpointControl(Qwen35Control):
    def hidden(self,ids):
        x=self.embedding(ids);routers=[]
        for layer in self.layers:
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                x,router=checkpoint(layer,x,use_reentrant=True)
            else:x,router=layer(x)
            routers.append(router)
        return self.norm(x),routers
