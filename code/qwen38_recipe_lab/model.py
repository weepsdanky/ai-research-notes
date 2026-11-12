"""Small Qwen3.8-Flash-Next mathematical model, not an official checkpoint loader.

Shapes, hashing and GR semantics follow pinned Transformers qwen4_exp (Apache-2.0).
Paper QSA scores are the default; `hf` explicitly selects the upstream sqrt scaling.
See sources.lock.json, RECIPE_MATRIX.md and THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from optim import head_pieces,piece

@dataclass
class Config:
    vocab_size: int = 248320
    eos: int = 248044
    width: int = 64
    layers: int = 4
    heads: int = 12
    kv_heads: int = 1
    head_dim: int = 256
    rotary_dim: int = 64
    gdn_k_heads: int = 1
    gdn_v_heads: int = 3
    gdn_k_dim: int = 128
    gdn_v_dim: int = 128
    hidden: int = 16
    experts: int = 8
    top_k: int = 2
    branches: int = 4
    rank: int = 8
    index_heads: int = 4
    index_dim: int = 128
    block: int = 4
    budget: int = 32
    score_mode: str = 'paper'
    gr_mode: str = 'gated'
    ngram_base: int = 1024
    ngram_heads: int = 2
    ngram_layer: int = 2  # ONE-indexed, confirmed in pinned upstream decoder
    hash_seed: int = 1234
    rope_theta: float = 10000000.0
    def __post_init__(self):
        assert self.layers%4==0 and self.heads%self.kv_heads==0
        assert self.gdn_v_heads%self.gdn_k_heads==0
        assert self.rotary_dim%2==0 and self.rotary_dim<=min(self.head_dim,self.index_dim)
        assert self.width%(2*self.ngram_heads)==0 and self.branches==4
        assert self.budget>=2*self.block and self.budget%self.block==0
        assert self.score_mode in ('paper','hf') and 1<=self.top_k<=self.experts
        assert self.gr_mode in ('gated','ungated')

class RMSNorm(nn.Module):
    def __init__(self,dim,group=None):
        super().__init__(); self.weight=nn.Parameter(torch.zeros(dim)); self.group=group
    def forward(self,x):
        shape=x.shape
        z=x.float().reshape(*shape[:-1],-1,self.group) if self.group else x.float()
        z=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)
        return (z.reshape(shape)*(1+self.weight.float())).to(x.dtype)


def rope(x,positions,dim,theta):
    """[B,T,H,D], split-half rotation; identical text positions on all mRoPE axes."""
    freq=theta**(-torch.arange(0,dim,2,device=x.device).float()/dim)
    angle=positions.float()[:,None]*freq
    cos=torch.cat((angle.cos(),angle.cos()),-1)[None,:,None]
    sin=torch.cat((angle.sin(),angle.sin()),-1)[None,:,None]
    a,b=x[...,:dim//2],x[...,dim//2:dim]
    rotated=x[...,:dim]*cos+torch.cat((-b,a),-1)*sin
    return torch.cat((rotated,x[...,dim:]),-1)


def delta_scan(q,k,v,g,beta,state=None):
    """Inputs [B,T,H,D]; q/k are unit-normalized BEFORE q/sqrt(D)."""
    q,k,v,g,beta=[x.float() for x in (q,k,v,g,beta)]
    q=q/math.sqrt(q.shape[-1])
    if state is None: state=q.new_zeros(q.shape[0],q.shape[2],q.shape[3],v.shape[3])
    outputs=[]
    for t in range(q.shape[1]):
        state=state*g[:,t,:,None,None].exp()
        error=(v[:,t]-(state*k[:,t,:,:,None]).sum(-2))*beta[:,t,:,None]
        state=state+k[:,t,:,:,None]*error[...,None,:]
        outputs.append((state*q[:,t,:,:,None]).sum(-2))
    return torch.stack(outputs,1),state

class GDN(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        kd,vd=c.gdn_k_heads*c.gdn_k_dim,c.gdn_v_heads*c.gdn_v_dim
        self.in_proj_qkv=nn.Linear(c.width,2*kd+vd,bias=False)
        self.in_proj_qkv.weight._recipe_pieces=head_pieces(2*c.gdn_k_heads,c.gdn_k_dim)+[
            piece('muon',2*kd+i*c.gdn_v_dim,2*kd+(i+1)*c.gdn_v_dim,semantic=f'v_head{i}') for i in range(c.gdn_v_heads)]
        self.in_proj_z=nn.Linear(c.width,vd,bias=False)
        self.in_proj_z.weight._recipe_pieces=[piece('adamw',semantic='output_gate')]
        self.in_proj_a=nn.Linear(c.width,c.gdn_v_heads,bias=False)
        self.in_proj_b=nn.Linear(c.width,c.gdn_v_heads,bias=False)
        for p in (self.in_proj_a.weight,self.in_proj_b.weight): p._recipe_pieces=[piece('adamw',semantic='scalar_gate')]
        self.dt_bias=nn.Parameter(torch.ones(c.gdn_v_heads))
        self.A_log=nn.Parameter(torch.empty(c.gdn_v_heads).uniform_(0.01,16).log())
        self.conv=nn.Conv1d(2*kd+vd,2*kd+vd,4,groups=2*kd+vd,bias=False)
        self.norm_weight=nn.Parameter(torch.ones(c.gdn_v_dim))
        self.out_proj=nn.Linear(vd,c.width,bias=False)
    def forward(self,x,cache=None):
        c=self.c; b,t,_=x.shape; kd=c.gdn_k_heads*c.gdn_k_dim
        projected=self.in_proj_qkv(x).transpose(1,2)
        previous=cache.get('conv') if cache is not None else None
        history=torch.cat((previous,projected),-1) if previous is not None else F.pad(projected,(3,0))
        if cache is not None: cache['conv']=history[...,-3:]
        streams=F.silu(self.conv(history)).transpose(1,2)
        q,k,v=streams.split((kd,kd,c.gdn_v_heads*c.gdn_v_dim),-1)
        q,k=[s.reshape(b,t,c.gdn_k_heads,c.gdn_k_dim).repeat_interleave(c.gdn_v_heads//c.gdn_k_heads,2) for s in (q,k)]
        # Upstream l2norm is rsqrt(sum(x²)+eps), NOT F.normalize's clamp(norm,eps).
        q,k=[s.float()*torch.rsqrt(s.float().square().sum(-1,keepdim=True)+1e-6) for s in (q,k)]
        v=v.reshape(b,t,c.gdn_v_heads,c.gdn_v_dim)
        g=-self.A_log.float().exp()*F.softplus(self.in_proj_a(x).float()+self.dt_bias)
        out,state=delta_scan(q,k,v,g,self.in_proj_b(x).sigmoid(),None if cache is None else cache.get('state'))
        if cache is not None: cache['state']=state
        out=out*torch.rsqrt(out.square().mean(-1,keepdim=True)+1e-6)*self.norm_weight
        out=out*self.in_proj_z(x).reshape_as(out).sigmoid()
        return self.out_proj(out.flatten(-2))

class GatedResidual(nn.Module):
    def __init__(self,c,write=True):
        super().__init__(); self.c=c
        self.norm=RMSNorm(c.branches*c.width,c.width)
        self.down=nn.Linear(c.branches*c.width,c.rank,bias=False)
        self.up=nn.Linear(c.rank,c.branches*c.width,bias=False)
        self.inject=nn.Linear(c.branches*c.width,c.branches,bias=False) if write else None
    def forward(self,residual):
        c=self.c; normalized=self.norm(residual)
        if c.gr_mode=='ungated':
            x=normalized.unflatten(-1,(c.branches,c.width)).mean(-2)
            return x if self.inject is None else (x,torch.ones(*x.shape[:-1],c.branches,device=x.device,dtype=x.dtype))
        gates=self.up(F.silu(self.down(normalized)/c.branches)).sigmoid()
        x=(gates*normalized).unflatten(-1,(c.branches,c.width)).mean(-2)
        return x if self.inject is None else (x,2*(self.inject(normalized)/c.branches).sigmoid())
    @staticmethod
    def add(residual,y,write): return residual+(y.unsqueeze(-2)*write.unsqueeze(-1)).flatten(-2)

class Indexer(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.qk=nn.Linear(c.width,(c.index_heads+1)*c.index_dim,bias=False)
        self.qnorm,self.knorm=RMSNorm(c.index_dim),RMSNorm(c.index_dim)
    def forward(self,x):
        c=self.c; b,t,_=x.shape; r=c.block; blocks=t//r
        q,k=self.qk(x).split((c.index_heads*c.index_dim,c.index_dim),-1)
        q=rope(self.qnorm(q.reshape(b,t,c.index_heads,c.index_dim)),torch.arange(t,device=x.device),c.rotary_dim,c.rope_theta)
        k=k[:,:blocks*r].reshape(b,blocks,r,c.index_dim).float().mean(2)
        k=rope(self.knorm(k)[:,:,None],torch.arange(blocks,device=x.device)*r,c.rotary_dim,c.rope_theta)[:,:,0]
        scores=torch.einsum('bthd,bsd->bths',q,k).relu().sum(2)
        if c.score_mode=='hf': scores=scores/math.sqrt(c.index_dim)
        valid=(torch.arange(blocks,device=x.device)*r+r-1)[None,:]<=torch.arange(t,device=x.device)[:,None]
        scores=scores.masked_fill(~valid,-1e9)
        count=min(c.budget//r,blocks)
        # Stable ties prevent a future padded column from changing earlier Top-K.
        selected=scores.argsort(dim=-1,descending=True,stable=True)[...,:count]
        selected_valid=valid[None].expand(b,-1,-1).gather(-1,selected)
        selected=selected.masked_fill(~selected_valid,-1)
        tokens=(selected[...,None]*r+torch.arange(r,device=x.device)).flatten(-2)
        tokens=tokens.masked_fill(~selected_valid.repeat_interleave(r,-1),-1)
        tokens=F.pad(tokens,(0,c.budget-tokens.shape[-1]),value=-1)
        pos=torch.arange(t,device=x.device)
        tail=((pos+1)//r*r)[:,None]+torch.arange(r-1,device=x.device)
        tail=tail.masked_fill(tail>pos[:,None],-1)[None].expand(b,-1,-1)
        return scores,valid,selected,torch.cat((tokens,tail),-1)


def kl_loss(teacher,scores,eligible):
    teacher=teacher.detach()*eligible
    teacher=teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-12)
    logprob=scores.masked_fill(~eligible,-1e9).log_softmax(-1)
    # N = all query tokens, with zero contribution before the first complete block.
    return (teacher*(teacher.clamp_min(1e-12).log()-logprob)).sum(-1).mean()

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.q_gate=nn.Linear(c.width,2*c.heads*c.head_dim,bias=False)
        self.q_gate.weight._recipe_pieces=head_pieces(c.heads,c.head_dim,gated=True)
        self.k,self.v=[nn.Linear(c.width,c.kv_heads*c.head_dim,bias=False) for _ in range(2)]
        for p in (self.k.weight,self.v.weight): p._recipe_pieces=head_pieces(c.kv_heads,c.head_dim)
        self.qnorm,self.knorm=RMSNorm(c.head_dim),RMSNorm(c.head_dim)
        self.out=nn.Linear(c.heads*c.head_dim,c.width,bias=False)
        self.indexer=Indexer(c)
    def forward(self,x,stage='pretrain',cache=None,dense_oracle=False):
        c=self.c; new_t=x.shape[1]
        if cache is not None:
            if 'x' in cache: x=torch.cat((cache['x'],x),1)
            cache['x']=x  # Reference cache recomputes attention; not a fast KV kernel.
        b,t,_=x.shape; pos=torch.arange(t,device=x.device)
        q,gate=self.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
        q=rope(self.qnorm(q),pos,c.rotary_dim,c.rope_theta).transpose(1,2)
        k=rope(self.knorm(self.k(x).reshape(b,t,c.kv_heads,c.head_dim)),pos,c.rotary_dim,c.rope_theta)
        k=k.repeat_interleave(c.heads//c.kv_heads,2).transpose(1,2)
        v=self.v(x).reshape(b,t,c.kv_heads,c.head_dim).repeat_interleave(c.heads//c.kv_heads,2).transpose(1,2)
        kl=x.new_zeros(())
        if stage!='pretrain': scores,valid,selected,tokens=self.indexer(x.detach())
        if stage=='sparse' and not dense_oracle:
            index=tokens.clamp_min(0)
            batch=torch.arange(b,device=x.device)[:,None,None]
            keys=k.transpose(1,2)[batch,index].permute(0,3,1,2,4)
            values=v.transpose(1,2)[batch,index].permute(0,3,1,2,4)
            weights=((q.unsqueeze(-2)*keys).sum(-1)/math.sqrt(c.head_dim)).masked_fill(tokens[:,None]<0,-1e9).softmax(-1)
            y=(weights[...,None]*values).sum(-2)
            count=selected.shape[-1]
            teacher=weights.detach().mean(1)[...,:count*c.block].reshape(b,t,count,c.block).amax(-1)
            chosen_scores=scores.gather(-1,selected.clamp_min(0))
            kl=kl_loss(teacher,chosen_scores,selected>=0) if count else kl
        else:
            mask=pos[None,:]<=pos[:,None]
            if stage=='sparse':
                mask=torch.zeros(b,t,t+1,device=x.device,dtype=torch.bool).scatter(-1,torch.where(tokens>=0,tokens,t),True)[...,:t]
                mask=mask[:,None]
            weights=((q@k.transpose(-1,-2))/math.sqrt(c.head_dim)).masked_fill(~mask,-1e9).softmax(-1)
            y=weights@v
            if stage!='pretrain' and scores.shape[-1]:
                teacher=weights.detach().mean(1)[...,:scores.shape[-1]*c.block].reshape(b,t,-1,c.block).amax(-1)
                eligible=valid.expand(b,-1,-1)
                if stage=='sparse':
                    eligible=(selected[...,None]==torch.arange(scores.shape[-1],device=x.device)).any(-2)
                kl=kl_loss(teacher,scores,eligible)
        output=self.out((y.transpose(1,2)*gate.sigmoid()).flatten(-2))
        return output[:,-new_t:],kl

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gate,self.up=nn.Linear(c.width,c.hidden,bias=False),nn.Linear(c.width,c.hidden,bias=False)
        self.down=nn.Linear(c.hidden,c.width,bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))

class MoE(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.router=nn.Linear(c.width,c.experts,bias=False)
        self.experts=nn.ModuleList([SwiGLU(c) for _ in range(c.experts)])
        self.shared=SwiGLU(c); self.shared_expert_gate=nn.Linear(c.width,1,bias=False)
    def forward(self,x):
        shape=x.shape; flat=x.reshape(-1,shape[-1]); logits=self.router(flat)
        probs=logits.float().softmax(-1)
        w,ids=probs.topk(self.c.top_k,-1); w=w/w.sum(-1,keepdim=True)
        out=self.shared(flat)*self.shared_expert_gate(flat).sigmoid()
        for i,expert in enumerate(self.experts):
            rows,slots=torch.where(ids==i)
            if rows.numel(): out=out.index_add(0,rows,expert(flat[rows])*w[rows,slots,None])
        return out.reshape(shape),logits


def splitmix64(value):
    mask=(1<<64)-1
    value=(value+0x9E3779B97F4A7C15)&mask
    value=((value^(value>>30))*0xBF58476D1CE4E5B9)&mask
    value=((value^(value>>27))*0x94D049BB133111EB)&mask
    return (value^(value>>31))&mask


def primes_from(start,count):
    result=[]
    while len(result)<count:
        if start>=2 and all(start%d for d in range(2,math.isqrt(start)+1)): result.append(start)
        start+=1
    return result

class NGramEmbedding(nn.Module):
    """Pinned HF int64 addressing computed on CPU, not an MPS fallback setting."""
    def __init__(self,c):
        super().__init__(); self.c=c
        self.sizes=primes_from(c.ngram_base,2*c.ngram_heads)
        self.offsets=[sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.ngram_embedding=nn.Embedding(math.ceil(sum(self.sizes)/128)*128,c.width//len(self.sizes))
        bound=max(1,((1<<63)-1)//c.vocab_size//2)
        self.multipliers=torch.tensor([2*(splitmix64((c.hash_seed+0x9E3779B97F4A7C15*(i+1))&((1<<64)-1))%bound)+1 for i in range(3)])
    def indices(self,ids):
        c=self.c; ids=ids.to('cpu',dtype=torch.long); b,t=ids.shape
        pos=torch.arange(t)[None].expand(b,-1)
        eospos=torch.where(ids==c.eos,pos,-1)
        previous=torch.cat((torch.full((b,1),-1),eospos.cummax(1).values[:,:-1]),1)
        shifts=[ids]
        for offset in (1,2):
            shifted=ids.gather(1,(pos-offset).clamp_min(0))
            shifts.append(torch.where((pos-previous-1>=offset)&(pos>=offset),shifted,c.eos))
        result=[]
        mixed=shifts[0]*self.multipliers[0]
        for order in (2,3):
            mixed=mixed^(shifts[order-1]*self.multipliers[order-1])
            for h in range(c.ngram_heads):
                j=(order-2)*c.ngram_heads+h
                result.append(mixed.remainder(self.sizes[j])+self.offsets[j])
        return torch.stack(result,-1)
    def forward(self,ids,cache=None):
        n=ids.shape[1]
        if cache is not None and 'ids' in cache: ids=torch.cat((cache['ids'],ids),1)
        if cache is not None: cache['ids']=ids[:,-2:]
        indices=self.indices(ids)[:,-n:].to(self.ngram_embedding.weight.device)
        return self.ngram_embedding(indices).flatten(-2)

class PLE(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c; self.embedding=NGramEmbedding(c)
        d=c.width*c.branches
        self.key,self.value=nn.Linear(c.width,d,bias=False),nn.Linear(c.width,c.width,bias=False)
        self.qnorm,self.knorm,self.cnorm=[RMSNorm(d,c.width) for _ in range(3)]
        self.conv=nn.Conv1d(d,d,4,groups=d,dilation=3,bias=False)
    def forward(self,x,ids,cache=None):
        c=self.c; e=self.embedding(ids,cache)
        q,k=[n(z).unflatten(-1,(c.branches,c.width)) for n,z in ((self.qnorm,x),(self.knorm,self.key(e)))]
        gate=(q*k).sum(-1,keepdim=True)/math.sqrt(c.width)
        gate=gate.abs().clamp_min(1e-6).sqrt()*gate.sign()
        value=(gate.sigmoid()*self.value(e).unsqueeze(-2)).flatten(-2)
        normalized=self.cnorm(value).transpose(1,2)
        previous=cache.get('conv') if cache is not None else None
        history=torch.cat((previous,normalized),-1) if previous is not None else F.pad(normalized,(9,0))
        if cache is not None: cache['conv']=history[...,-9:]
        return value+F.silu(self.conv(history)).transpose(1,2)

class Layer(nn.Module):
    def __init__(self,c,index):
        super().__init__(); self.ple=PLE(c) if index+1==c.ngram_layer else None
        self.mix=Attention(c) if index%4==3 else GDN(c)
        self.moe=MoE(c)
        self.mix_residual,self.moe_residual=GatedResidual(c),GatedResidual(c)
    def forward(self,x,ids,stage,cache=None):
        def slot(key): return None if cache is None else cache.setdefault(key,{})
        if self.ple is not None: x=x+self.ple(x,ids,slot('ple'))
        y,write=self.mix_residual(x)
        if isinstance(self.mix,Attention): y,kl=self.mix(y,stage,slot('mix'))
        else: y,kl=self.mix(y,slot('mix')),x.new_zeros(())
        x=GatedResidual.add(x,y,write)
        y,write=self.moe_residual(x); y,logits=self.moe(y)
        return GatedResidual.add(x,y,write),kl,logits

class Model(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        self.embedding=nn.Embedding(c.vocab_size,c.width)
        self.layers=nn.ModuleList([Layer(c,i) for i in range(c.layers)])
        self.readout=GatedResidual(c,write=False)
        self.head=nn.Linear(c.width,c.vocab_size,bias=False)
        self.apply(self._init)
        for module in self.modules():
            if isinstance(module,PLE): nn.init.zeros_(module.conv.weight)
    @staticmethod
    def _init(module):
        if isinstance(module,(nn.Linear,nn.Embedding,nn.Conv1d)): nn.init.normal_(module.weight,std=0.02)
    def stage(self,stage):
        assert stage in ('pretrain','distill','sparse')
        ids={id(p) for m in self.modules() if isinstance(m,Indexer) for p in m.parameters()}
        for p in self.parameters(): p.requires_grad_((id(p) in ids) if stage=='distill' else (id(p) not in ids if stage=='pretrain' else True))
    def forward(self,ids,stage='pretrain',cache=None):
        residual=self.embedding(ids).repeat(1,1,self.config.branches)
        kl=residual.new_zeros(()); routers=[]
        for i,layer in enumerate(self.layers):
            residual,local,logits=layer(residual,ids,stage,None if cache is None else cache.setdefault(str(i),{}))
            kl=kl+local; routers.append(logits)
        probs=torch.cat(routers).softmax(-1)
        selected=probs.topk(self.config.top_k,-1).indices
        frequency=F.one_hot(selected,self.config.experts).float().sum(1).mean(0).detach()
        balance=self.config.experts*(frequency*probs.mean(0)).sum()
        return self.head(self.readout(residual)),kl/(self.config.layers//4),balance
