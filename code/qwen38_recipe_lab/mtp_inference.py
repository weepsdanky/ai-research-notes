"""Incremental eager MTP component: frozen QSA rows, live caches, explicit rollback.

Matches the reviewed public cache contract, not a complete speculative proposer
or a trained accepted-length experiment. Dense/Sparse FP32 references only.
"""
import math
import torch
from torch.nn import functional as F
from model import GatedResidual,rope


def pack_selection(tokens,capacity):
    if tokens.shape[-1]>capacity:raise ValueError('Selection exceeds capacity')
    valid=tokens>=0;order=(~valid).int().argsort(dim=-1,stable=True)
    compact=tokens.gather(-1,order)
    compact=F.pad(compact,(0,capacity-compact.shape[-1]),value=-1)
    return torch.cat((compact,valid.sum(-1,keepdim=True)),-1).to(torch.int32)


def unpack_selection(packed,capacity,positions):
    if packed.shape[-1]!=capacity+1:raise ValueError('Packed row width')
    count=packed[...,-1].long();indices=packed[...,:-1].long()
    if ((count<0)|(count>capacity)).any():raise ValueError('Invalid valid count')
    valid=torch.arange(capacity,device=packed.device)<count[...,None]
    if (valid&(indices<0)).any() or ((~valid)&(indices!=-1)).any():raise ValueError('Noncontiguous selection row')
    if (valid&(indices>positions[None,:,None])).any():raise ValueError('Noncausal selection')
    return indices


class IncrementalMTP:
    def __init__(self,module,stage='sparse'):
        if stage not in ('sparse','pretrain'):raise ValueError('Unsupported stage')
        assert module.layer.ple is None
        self.module,self.stage=module,stage

    def rollback(self,cache,length):
        if not 0<=length<=cache['k'].shape[1]:raise ValueError('Invalid rollback length')
        for name in ('k','v','raw'):
            if name in cache:cache[name]=cache[name][:,:length].clone()
        if 'pooled' in cache:cache['pooled']=cache['pooled'][:,:length//self.module.c.block].clone()
        cache.pop('selection',None)  # A new draft round must select again.

    @staticmethod
    def compact_requests(cache,indices):
        """Equal-context local requests only, not vLLM's ragged paged allocator."""
        for name,value in list(cache.items()):
            if isinstance(value,torch.Tensor):cache[name]=value.index_select(0,indices).clone()

    @torch.no_grad()
    def __call__(self,embedding,hidden,cache,reuse=False):
        m=self.module;c=m.c;b,t,_=embedding.shape
        if t<1 or hidden.shape!=(b,t,c.branches*c.width):raise ValueError('Input shape')
        if reuse and (self.stage!='sparse' or 'selection' not in cache):raise ValueError('Reuse requires a selected draft step0')
        if cache.get('stage',self.stage)!=self.stage:raise ValueError('Cache stage mismatch')
        cache['stage']=self.stage;pos=cache['k'].shape[1] if 'k' in cache else 0
        position=torch.arange(pos,pos+t,device=embedding.device)
        residual=m.fusion(embedding,hidden);x,write=m.layer.mix_residual(residual);a=m.layer.mix
        q,gate=a.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
        q=rope(a.qnorm(q),position,c.rotary_dim,c.rope_theta)
        k=rope(a.knorm(a.k(x).reshape(b,t,c.kv_heads,c.head_dim)),position,c.rotary_dim,c.rope_theta)
        v=a.v(x).reshape(b,t,c.kv_heads,c.head_dim)
        cache['k']=torch.cat((cache['k'],k),1) if pos else k
        cache['v']=torch.cat((cache['v'],v),1) if pos else v
        packed=None
        if self.stage=='sparse':
            index=a.indexer;iq,raw=index.qk(x).split((c.index_heads*c.index_dim,c.index_dim),-1)
            cache['raw']=torch.cat((cache['raw'],raw),1) if pos else raw
            old_groups=pos//c.block;groups=(pos+t)//c.block
            if 'pooled' not in cache:cache['pooled']=raw.new_empty((b,0,c.index_dim),dtype=torch.float32)
            if groups>old_groups:
                pooled=cache['raw'][:,old_groups*c.block:groups*c.block].reshape(b,groups-old_groups,c.block,c.index_dim).float().mean(2)
                pooled=index.knorm(pooled)[:,:,None]
                starts=torch.arange(old_groups,groups,device=x.device)*c.block
                pooled=rope(pooled,starts,c.rotary_dim,c.rope_theta)[:,:,0]
                cache['pooled']=torch.cat((cache['pooled'],pooled),1)
            capacity=c.budget+c.block-1
            if reuse:
                # Keep the entire step0 row, including its tail and valid count.
                packed=cache['selection'][:,None].expand(b,t,-1)
            else:
                iq=rope(index.qnorm(iq.reshape(b,t,c.index_heads,c.index_dim)),position,c.rotary_dim,c.rope_theta)
                scores=torch.einsum('bthd,bsd->bths',iq,cache['pooled']).relu().sum(2)
                if c.score_mode=='hf':scores=scores/math.sqrt(c.index_dim)
                valid=(torch.arange(groups,device=x.device)*c.block+c.block-1)[None]<=position[:,None]
                scores=scores.masked_fill(~valid,-1e9)
                selected=scores.argsort(dim=-1,descending=True,stable=True)[...,:min(c.budget//c.block,groups)]
                selected=selected.masked_fill(~valid[None].expand(b,-1,-1).gather(-1,selected),-1)
                tokens=(selected[...,None]*c.block+torch.arange(c.block,device=x.device)).flatten(-2)
                tokens=tokens.masked_fill(~(selected>=0).repeat_interleave(c.block,-1),-1)
                tail=((position+1)//c.block*c.block)[:,None]+torch.arange(c.block-1,device=x.device)
                tail=tail.masked_fill(tail>position[:,None],-1)[None].expand(b,-1,-1)
                packed=pack_selection(torch.cat((tokens,tail),-1),capacity)
                cache['selection']=packed[:,-1].clone()
            tokens=unpack_selection(packed,capacity,position)
            batch=torch.arange(b,device=x.device)[:,None,None]
            keys=cache['k'][batch,tokens.clamp_min(0)].permute(0,1,3,4,2)
            values=cache['v'][batch,tokens.clamp_min(0)].permute(0,1,3,2,4)
            queries=q.reshape(b,t,c.kv_heads,c.heads//c.kv_heads,c.head_dim)
            scores=(queries@keys)/math.sqrt(c.head_dim)
            invalid=tokens[:,:,None,None]<0
            weights=scores.masked_fill(invalid,-1e9).softmax(-1).masked_fill(invalid,0)
            y=(weights@values).reshape(b,t,c.heads,c.head_dim)
        else:
            queries=q.permute(0,2,1,3).reshape(b,c.kv_heads,c.heads//c.kv_heads,t,c.head_dim)
            keys=cache['k'].permute(0,2,3,1).unsqueeze(2);values=cache['v'].permute(0,2,1,3).unsqueeze(2)
            scores=(queries@keys)/math.sqrt(c.head_dim)
            valid=torch.arange(pos+t,device=x.device)[None]<=position[:,None]
            weights=scores.masked_fill(~valid,-1e9).softmax(-1)
            y=(weights@values).permute(0,3,1,2,4).reshape(b,t,c.heads,c.head_dim)
        y=a.out((y*gate.sigmoid()).flatten(-2));residual=GatedResidual.add(residual,y,write)
        x,write=m.layer.moe_residual(residual);residual=GatedResidual.add(residual,m.layer.moe(x)[0],write)
        return m.readout(residual),residual,packed
