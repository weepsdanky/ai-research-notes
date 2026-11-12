"""Untrained Mac text controls for public Qwen3.5 family structure.

Independent from the frozen Qwen3.8 recipe. The private28-layer Table1 config is
not known. FP32 reference, ordinary residuals, no GR/Ngram/QSA; no fused kernel.
Long-context training feasibility and numerical acceptance remain separate gates.
"""
from dataclasses import dataclass
import hashlib,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from model import GDN,MoE,RMSNorm,delta_scan,rope
from optim import head_pieces


@dataclass
class ControlConfig:
    vocab_size:int=248320
    width:int=64
    layers:int=4
    heads:int=8
    kv_heads:int=1
    head_dim:int=256
    rotary_dim:int=64
    gdn_k_heads:int=1
    gdn_v_heads:int=2
    gdn_k_dim:int=128
    gdn_v_dim:int=128
    hidden:int=16
    experts:int=64
    top_k:int=2
    rope_theta:float=10000000.0
    window:int=128
    query_chunk:int=128
    scan_chunk:int=128
    activation_checkpoint:bool=True
    def __post_init__(self):
        assert self.layers>0 and self.layers%4==0 and self.heads%self.kv_heads==0
        assert self.rotary_dim==self.head_dim//4 and self.rotary_dim%2==0
        assert self.gdn_v_heads==2*self.gdn_k_heads and self.gdn_k_dim==self.gdn_v_dim
        assert min(self.window,self.query_chunk,self.scan_chunk,self.width,self.hidden)>0
        assert 1<=self.top_k<=self.experts


def causal_window_mask(queries,keys,window=None):
    delta=queries[:,None]-keys[None,:]
    return (delta>=0) if window is None else (delta>=0)&(delta<window)


def gated_norm35(hidden,gate,weight):
    # Public Qwen3.5 RMSNormGated has a SiLU gate, not Qwen3.8's configured sigmoid.
    value=hidden.float()*torch.rsqrt(hidden.float().square().mean(-1,keepdim=True)+1e-6)
    return (value*weight.float()*F.silu(gate.float())).to(hidden.dtype)


class GDN35(GDN):
    def forward(self,x,cache=None):
        c=self.c;b,t,_=x.shape;kd=c.gdn_k_heads*c.gdn_k_dim
        projected=self.in_proj_qkv(x).transpose(1,2)
        previous=cache.get('conv') if cache is not None else None
        history=torch.cat((previous,projected),-1) if previous is not None else F.pad(projected,(3,0))
        if cache is not None:cache['conv']=history[...,-3:]
        streams=F.silu(self.conv(history)).transpose(1,2)
        q,k,v=streams.split((kd,kd,c.gdn_v_heads*c.gdn_v_dim),-1)
        q,k=[s.reshape(b,t,c.gdn_k_heads,c.gdn_k_dim).repeat_interleave(2,2) for s in (q,k)]
        q,k=[s.float()*torch.rsqrt(s.float().square().sum(-1,keepdim=True)+1e-6) for s in (q,k)]
        v=v.reshape(b,t,c.gdn_v_heads,c.gdn_v_dim)
        g=-self.A_log.float().exp()*F.softplus(self.in_proj_a(x).float()+self.dt_bias);beta=self.in_proj_b(x).sigmoid()
        state=cache.get('state') if cache is not None else None
        if state is None:state=q.new_zeros(b,c.gdn_v_heads,c.gdn_k_dim,c.gdn_v_dim)
        outputs=[]
        for start in range(0,t,c.scan_chunk):
            args=[value[:,start:start+c.scan_chunk] for value in (q,k,v,g,beta)]+[state]
            if c.activation_checkpoint and self.training and torch.is_grad_enabled():out,state=checkpoint(delta_scan,*args,use_reentrant=False)
            else:out,state=delta_scan(*args)
            outputs.append(out)
        if cache is not None:cache['state']=state
        out=torch.cat(outputs,1);gate=self.in_proj_z(x).reshape_as(out)
        return self.out_proj(gated_norm35(out,gate,self.norm_weight).flatten(-2))


class PlainAttention(nn.Module):
    def __init__(self,c,window=None):
        super().__init__();self.c=c;self.window=window
        self.q_gate=nn.Linear(c.width,2*c.heads*c.head_dim,bias=False)
        self.q_gate.weight._recipe_pieces=head_pieces(c.heads,c.head_dim,gated=True)
        self.k,self.v=[nn.Linear(c.width,c.kv_heads*c.head_dim,bias=False) for _ in range(2)]
        for p in (self.k.weight,self.v.weight):p._recipe_pieces=head_pieces(c.kv_heads,c.head_dim)
        self.qnorm,self.knorm=RMSNorm(c.head_dim),RMSNorm(c.head_dim)
        self.out=nn.Linear(c.heads*c.head_dim,c.width,bias=False)
    def forward(self,x):
        c=self.c;b,t,_=x.shape;positions=torch.arange(t,device=x.device)
        q,gate=self.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
        q=rope(self.qnorm(q),positions,c.rotary_dim,c.rope_theta).permute(0,2,1,3).reshape(b,c.kv_heads,c.heads//c.kv_heads,t,c.head_dim)
        k=rope(self.knorm(self.k(x).reshape(b,t,c.kv_heads,c.head_dim)),positions,c.rotary_dim,c.rope_theta).permute(0,2,3,1)
        v=self.v(x).reshape(b,t,c.kv_heads,c.head_dim).permute(0,2,1,3);pieces=[]
        def attend(qpart,keys,values,qpositions,kpositions):
            # Recompute the mask inside checkpoint: saving one bool mask per
            # query chunk would retain quadratic memory across a long sequence.
            mask=causal_window_mask(qpositions,kpositions,self.window).repeat(c.heads//c.kv_heads,1)
            return ((qpart@keys)/math.sqrt(c.head_dim)).masked_fill(~mask,-1e9).softmax(-1)@values
        for start in range(0,t,c.query_chunk):
            end=min(t,start+c.query_chunk);lo=0 if self.window is None else max(0,start-self.window+1)
            queries=q[:,:,:,start:end].reshape(b,c.kv_heads,-1,c.head_dim)
            args=(queries,k[:,:,:,lo:end],v[:,:,lo:end],positions[start:end],positions[lo:end])
            if c.activation_checkpoint and self.training and torch.is_grad_enabled():part=checkpoint(attend,*args,use_reentrant=False)
            else:part=attend(*args)
            pieces.append(part.reshape(b,c.kv_heads,c.heads//c.kv_heads,end-start,c.head_dim))
        y=torch.cat(pieces,3).permute(0,3,1,2,4).reshape(b,t,c.heads,c.head_dim)*gate.sigmoid()
        return self.out(y.flatten(-2))


class PlainLayer(nn.Module):
    def __init__(self,c,kind):
        super().__init__();assert kind in ('full','swa','gdn');self.kind=kind
        self.input_norm=RMSNorm(c.width);self.post_norm=RMSNorm(c.width)
        self.mixer=GDN35(c) if kind=='gdn' else PlainAttention(c,c.window if kind=='swa' else None)
        self.moe=MoE(c)
    def forward(self,x):
        x=x+self.mixer(self.input_norm(x));value,router=self.moe(self.post_norm(x))
        return x+value,router


class Qwen35Control(nn.Module):
    def __init__(self,c,arm,seed=42):
        super().__init__();assert arm in ('full','swa','gdn');self.config=c;self.arm=arm
        self.embedding=nn.Embedding(c.vocab_size,c.width)
        self.layers=nn.ModuleList([PlainLayer(c,'full' if arm=='full' or (i+1)%4==0 else arm) for i in range(c.layers)])
        self.norm=RMSNorm(c.width);self.head=nn.Linear(c.width,c.vocab_size,bias=False)
        # Mac control choice: name-local seeds preserve shared initial tensors across arms.
        with torch.no_grad():
            for name,p in self.named_parameters():
                generator=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8],'little')%(2**63))
                value=torch.empty_like(p,device='cpu')
                if name.endswith('A_log'):value.uniform_(.01,16,generator=generator).log_()
                elif name.endswith(('dt_bias','norm_weight')):value.fill_(1)
                elif p.ndim==1:value.zero_()
                else:value.normal_(std=.02,generator=generator)
                p.copy_(value.to(p.device))
    def hidden(self,ids):
        x=self.embedding(ids);routers=[]
        for layer in self.layers:x,router=layer(x);routers.append(router)
        return self.norm(x),routers
    def forward(self,ids):
        hidden,routers=self.hidden(ids)
        return self.head(hidden),routers
