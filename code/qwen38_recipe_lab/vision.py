"""Scaled public qwen4_exp visual encoder (Apache-2.0 upstream semantics).

Input patches are already preprocessed/grouped by the processor. Image decoding,
normalization and multimodal training are separate audit items, not invented here.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


def text_mrope(positions,rotary_dim=64,sections=(11,11,10),theta=1e7,dtype=torch.float32):
    """Public text-side 3-axis interleaved mRoPE; no multimodal training claim.

    positions is [3,B,T]. Text uses identical positions on all axes; image/video
    positions are supplied by a processor/packing layer, which is separate.
    """
    if positions.ndim!=3 or positions.shape[0]!=3 or sum(sections)!=rotary_dim//2:
        raise ValueError('Expected [3,B,T] positions and sections summing to rotary_dim/2')
    inverse=theta**(-torch.arange(0,rotary_dim,2,device=positions.device).float()/rotary_dim)
    frequencies=positions.float()[...,None]*inverse
    mixed=frequencies[0].clone()
    for axis in (1,2):
        slot=slice(axis,sections[axis]*3,3)
        mixed[...,slot]=frequencies[axis,...,slot]
    angles=torch.cat((mixed,mixed),-1)
    return angles.cos().to(dtype),angles.sin().to(dtype)

@dataclass
class VisionConfig:
    hidden_size:int=32
    intermediate_size:int=64
    num_heads:int=4
    depth:int=2
    in_channels:int=3
    patch_size:int=16
    temporal_patch_size:int=2
    spatial_merge_size:int=2
    num_position_embeddings:int=16
    out_hidden_size:int=64
    hidden_act:str='gelu_pytorch_tanh'
    _attn_implementation:str='eager'
    rope_parameters:dict=None
    initializer_range:float=0.02
    def __post_init__(self):
        if self.rope_parameters is None: self.rope_parameters={'rope_type':'axial','rope_theta':10000.0}
        assert self.hidden_size%self.num_heads==0 and (self.hidden_size//self.num_heads)%4==0
    @property
    def num_attention_heads(self): return self.num_heads

class PatchEmbed(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.proj=nn.Conv3d(c.in_channels,c.hidden_size,(c.temporal_patch_size,c.patch_size,c.patch_size),stride=(c.temporal_patch_size,c.patch_size,c.patch_size),bias=True)
    def forward(self,x):
        c=self.c
        return self.proj(x.reshape(-1,c.in_channels,c.temporal_patch_size,c.patch_size,c.patch_size)).flatten(1)

class VisionMLP(nn.Module):
    def __init__(self,c):
        super().__init__(); self.linear_fc1=nn.Linear(c.hidden_size,c.intermediate_size); self.linear_fc2=nn.Linear(c.intermediate_size,c.hidden_size)
    def forward(self,x): return self.linear_fc2(F.gelu(self.linear_fc1(x),approximate='tanh'))


def axial_cos_sin(positions,head_dim,theta):
    freq=theta**(-torch.arange(0,head_dim//2,2,device=positions.device).float()/(head_dim//2))
    angles=(positions[...,None]*freq).flatten(-2)
    return angles.repeat(1,2).cos(),angles.repeat(1,2).sin()

class VisionAttention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c; self.qkv=nn.Linear(c.hidden_size,3*c.hidden_size); self.proj=nn.Linear(c.hidden_size,c.hidden_size)
    def forward(self,x,cu_seqlens,position_embeddings):
        c=self.c; n=len(x); d=c.hidden_size//c.num_heads
        q,k,v=self.qkv(x).reshape(n,3,c.num_heads,d).unbind(1)
        cos,sin=position_embeddings
        def rotate(z):
            a,b=z.chunk(2,-1)
            return z*cos[:,None]+torch.cat((-b,a),-1)*sin[:,None]
        q,k=rotate(q),rotate(k)
        result=[]
        for begin,end in zip(cu_seqlens[:-1].tolist(),cu_seqlens[1:].tolist()):
            a,b,w=[z[begin:end].transpose(0,1) for z in (q,k,v)]
            result.append((((a@b.transpose(-1,-2))/(d**.5)).softmax(-1)@w).transpose(0,1))
        return self.proj(torch.cat(result).reshape(n,c.hidden_size))

class VisionBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); self.norm1=nn.LayerNorm(c.hidden_size,eps=1e-6); self.norm2=nn.LayerNorm(c.hidden_size,eps=1e-6)
        self.attn=VisionAttention(c); self.mlp=VisionMLP(c)
    def forward(self,x,cu_seqlens,position_embeddings):
        x=x+self.attn(self.norm1(x),cu_seqlens,position_embeddings)
        return x+self.mlp(self.norm2(x))

class Merger(nn.Module):
    def __init__(self,c):
        super().__init__(); self.hidden_size=c.hidden_size*c.spatial_merge_size**2
        self.norm=nn.LayerNorm(c.hidden_size,eps=1e-6)
        self.linear_fc1=nn.Linear(self.hidden_size,self.hidden_size); self.linear_fc2=nn.Linear(self.hidden_size,c.out_hidden_size)
    def forward(self,x):
        x=self.norm(x).reshape(-1,self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))

class VisionModel(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        self.patch_embed=PatchEmbed(c); self.pos_embed=nn.Embedding(c.num_position_embeddings,c.hidden_size)
        self.blocks=nn.ModuleList([VisionBlock(c) for _ in range(c.depth)]); self.merger=Merger(c)
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Embedding,nn.Conv3d)):
                nn.init.normal_(module.weight,std=c.initializer_range)
                if getattr(module,'bias',None) is not None: nn.init.zeros_(module.bias)
    def geometry(self,grid):
        c=self.config; m=c.spatial_merge_size; side=int(c.num_position_embeddings**.5)
        positions=[]; embeddings=[]; lengths=[]
        for t,h,w in grid.tolist():
            if h%m or w%m: raise ValueError('Vision grids must be divisible by spatial merge size')
            row,col=torch.meshgrid(torch.arange(h,device=grid.device),torch.arange(w,device=grid.device),indexing='ij')
            def grouped(z): return z.reshape(h//m,m,w//m,m,-1).permute(0,2,1,3,4).reshape(h*w,-1)
            positions.append(grouped(torch.stack((row,col),-1)).repeat(t,1))
            table=self.pos_embed.weight.reshape(side,side,c.hidden_size).permute(2,0,1)[None]
            resized=F.interpolate(table,size=(h,w),mode='bilinear',align_corners=True)[0].permute(1,2,0)
            embeddings.append(grouped(resized).repeat(t,1)); lengths.extend([h*w]*t)
        cu=torch.tensor([0,*lengths],device=grid.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
        return torch.cat(positions),torch.cat(embeddings),cu
    def forward(self,patches,grid):
        positions,embeddings,cu=self.geometry(grid)
        x=self.patch_embed(patches)+embeddings
        cos_sin=axial_cos_sin(positions,self.config.hidden_size//self.config.num_heads,self.config.rope_parameters['rope_theta'])
        for block in self.blocks: x=block(x,cu,cos_sin)
        return x,self.merger(x)
