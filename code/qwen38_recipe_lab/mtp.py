"""Public vLLM residual_linear_shared MTP architecture, without GPU wrappers.

This exposes architecture/fusion/index reuse for testing. The production MTP
training loss weights/data/schedule are not specified by the architecture report.
It is NOT silently added to the already-started text training recipe.
"""
import math
import torch
from torch import nn
from model import RMSNorm,GatedResidual,Layer,rope,Model


def selected_attention(attention,x,tokens):
    """Same gather core, externally supplied selection; indexer is never called."""
    c=attention.c; b,t,_=x.shape; pos=torch.arange(t,device=x.device)
    q,gate=attention.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
    q=rope(attention.qnorm(q),pos,c.rotary_dim,c.rope_theta).transpose(1,2)
    k=rope(attention.knorm(attention.k(x).reshape(b,t,c.kv_heads,c.head_dim)),pos,c.rotary_dim,c.rope_theta)
    k=k.repeat_interleave(c.heads//c.kv_heads,2)
    v=attention.v(x).reshape(b,t,c.kv_heads,c.head_dim).repeat_interleave(c.heads//c.kv_heads,2)
    idx=tokens.clamp_min(0); batch=torch.arange(b,device=x.device)[:,None,None]
    keys=k[batch,idx].permute(0,3,1,2,4); values=v[batch,idx].permute(0,3,1,2,4)
    weights=((q[...,None,:]*keys).sum(-1)/math.sqrt(c.head_dim)).masked_fill(tokens[:,None]<0,-1e9).softmax(-1)
    y=(weights[...,None]*values).sum(-2).transpose(1,2)*gate.sigmoid()
    return attention.out(y.flatten(-2))

class MultiTokenPredictor(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.pre_fc_norm_embedding=RMSNorm(c.width)
        # vLLM applies this norm to the FLATTENED residual, not per branch.
        self.pre_fc_norm_hidden=RMSNorm(c.width*c.branches)
        self.fc_embedding=nn.Linear(c.width,c.width,bias=False)
        self.fc_hidden=nn.Linear(c.width,c.width,bias=False)  # shared by every branch
        self.layer=Layer(c,3)  # public one-layer QSA/MoE draft, no PLE
        assert self.layer.ple is None
        self.readout=GatedResidual(c,write=False)
        self.apply(Model._init)
    def fusion(self,embedding,hidden):
        c=self.c
        e=self.fc_embedding(self.pre_fc_norm_embedding(embedding))
        h=self.pre_fc_norm_hidden(hidden).unflatten(-1,(c.branches,c.width))
        return (self.fc_hidden(h)+e.unsqueeze(-2)).flatten(-2)
    def forward(self,embedding,hidden,stage='sparse',reuse_tokens=None):
        residual=self.fusion(embedding,hidden)
        x,write=self.layer.mix_residual(residual)
        if reuse_tokens is None:
            capture=[]
            handle=self.layer.mix.indexer.register_forward_hook(lambda m,i,o:capture.append(o[-1]))
            try: y,kl=self.layer.mix(x,stage)
            finally: handle.remove()
            tokens=capture[0] if capture else None
        else:
            if (reuse_tokens>=x.shape[1]).any(): raise ValueError('Reused index out of range')
            if (reuse_tokens>torch.arange(x.shape[1],device=x.device)[None,:,None]).any(): raise ValueError('Reused index is not causal')
            y=selected_attention(self.layer.mix,x,reuse_tokens); kl=x.new_zeros(())
            tokens=reuse_tokens
        residual=GatedResidual.add(residual,y,write)
        x,write=self.layer.moe_residual(residual); y,router_logits=self.layer.moe(x)
        residual=GatedResidual.add(residual,y,write)
        return self.readout(residual),residual,tokens,kl,router_logits
